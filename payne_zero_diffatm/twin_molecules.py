"""Batched, differentiable twin of the reference molecular-equilibrium stage.

Reproduces ``solve_molecular_equilibrium``
(``payne_zero_atmosphere/molecular_equilibrium.py:244-353``) — the molecule-
coupled replacement for the atomic electron-density fixed point that runs when
the atmosphere solver is called with ``molecules_enabled=True`` — as a
torch-native module batched over stars *and* layers jointly and differentiable
in temperature and gas pressure end to end. The reference chain per atmosphere
is

* the runtime seeds (``build_runtime_state`` / ``update_charge_square_density``
  — ``runtime_state.py:192-250``): total-particle density, total-nuclei
  density, charge-square density, mean nuclear mass;
* per-layer equilibrium constants
  (``_equilibrium_constants_kernel`` ``molecular_equilibrium.py:693-757``)
  with the H2 partition-function branch (:653-691), the dissociation
  polynomial branch, and the Saha-ratio branch driven by
  ``_compute_equilibrium_constants_for_layer_compiled`` (:1101-1147);
* a damped Newton solve in linear density space per layer
  (``_newton_matrix_kernel`` :759-859, ``_newton_update_kernel``
  :861-885, ``np.linalg.solve``) with the reference seeds
  (:258-277) and pressure-ratio carry-over (:279-295);
* the molecular population product (:315-330), the runtime-state updates
  (:304-313), and the partition-normalized refill
  (``_fill_partition_normalized_molecular_densities`` :356-485).

Design choices
--------------

Catalog. The twin reads the atmosphere solver's own catalog,
``source_catalogs/lines/molecular_equilibrium_atmosphere.npz`` via
``molecular_equilibrium_catalog_path()`` — the file ``prepare_population_state``
loads (``runner.py:470-478``) — plus the H2 partition table from
``atmosphere_tables/molecular_equilibrium_tables.npz``. The synthesis catalog
(``molecular_equilibrium_synthesis.npz``) is a different table and is not used.

Batching. Stars are batched, while layers retain the reference's serial carry:
each layer's Newton solve starts from the previous layer's converged densities
scaled by the pressure ratio (:281-288). This seed selects a materially
different root in some hot/low-gravity atmospheres, so it cannot be removed as
an optimization detail. The 80-layer loop remains autograd-safe, and every
Newton system inside it is solved jointly over the star batch. Both solvers
stop on the same relative-update criterion (<=1e-4); the checker measures
agreement to float64 round-off on all three trace atmospheres.

Newton. Residual and Jacobian assembly follow ``_newton_matrix_kernel``
exactly, including the negative-ion correction block (:831-857, species-100
components such as C-), with scatter-adds over static index lists built once
from the catalog. The damped update follows ``_newton_update_kernel``: the
0.69 sign-flip relaxation, the ``abs(d - delta)`` update with the d/100 floor,
and the sequential ``scale = sqrt(scale)`` rule — the latter vectorized as
``scale_i = 100 ** (0.5 ** k_i)`` where ``k_i`` is the number of earlier
floor-and-flip events along the equation axis (a cumsum, not a data-dependent
Python scalar). Convergence is a sticky per-point mask frozen with
``torch.where``; there is no ``.item()`` and no numpy in the compute path, so
the whole loop is autograd-safe. The reference's ``np.linalg.lstsq`` fallback
for singular Jacobians (:1195-1196) has no batched equivalent and is not
transcribed; it never triggers on the trace atmospheres.

Partitions. The Saha-ratio constants and the mode-3 partitions of the
normalized refill come from the ``twin_eos`` partition machinery
(``_assemble_partitions`` and helpers) evaluated at the molecular seed
charge-square density — the same kernel tables the reference's
``saha_partition_depth`` calls read. Coverage check: the deepest stage the
catalog needs is stage 5 (C/N/O/Na..S with six-component ion rows) and the
twin_eos mode-11 decode reaches stage ``work11 - 1 >= 5`` for every element in
the catalog (Cl stops at stage 4, which is all its five-component rows need).
The mode-12 fraction ratio needs only stages ``0..count-1`` and is
normalization-independent, so the reference's ``work_ion_count =
min(count+2, available)`` ladder depth does not enter. The reference guard
``fractions[0] <= 0 -> constant 0`` (:1127-1129) is transcribed in log space
against the float64 underflow point; it never fires on the trace atmospheres.

Known gaps. ``specific_internal_energy_mode`` (the convection
finite-difference path, :596-604 and :1212-1366) is out of scope. Population
mode 1 outputs are produced (the mode used by ``prepare_population_state``);
the mode 2/12 early return (:342-343) only skips the normalized refill and the
specific-energy update, both of which the twin always computes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from payne_zero_atmosphere.constants import (
    ATOMIC_MASS_GRAM_REFERENCE,
    BOLTZMANN_ERG_PER_K_REFERENCE,
    BOLTZMANN_EV_PER_K_REFERENCE,
    LIGHT_SPEED_CM_PER_S_EXACT,
    LIGHT_SPEED_CM_PER_S_REFERENCE,
    PLANCK_ERG_SECOND_REFERENCE,
)
from payne_zero_atmosphere.data_files import atmosphere_table_path
from payne_zero_atmosphere.molecular_data import read_molecular_equilibrium_catalog
from payne_zero_atmosphere.runtime_state import REFERENCE_ATOMIC_MASS_AMU
from payne_zero_atmosphere.source_catalogs import molecular_equilibrium_catalog_path

from .twin_eos import (
    MAX_ION_STAGES,
    N_ELEMENTS,
    SAHA_COEFFICIENT_REFERENCE,
    TwinEosTables,
    _assemble_partitions,
    _debye_lowering,
    _iron_temperature_bins,
    _ordinary_partition_base,
    _prepare_occupation_statics,
    _special_partition_base,
)


# Reference solver constants, molecular_equilibrium.py:48-49.
MAX_NEWTON_ITERATIONS = 200
NEWTON_TOLERANCE = 1.0e-4

# Rounded literals the molecular kernels use (molecular_equilibrium.py:373,
# :405-409, :1115). The Saha machinery keeps the twin_eos constants; these are
# the molecular block's own rounded values.
_KELVIN_PER_EV = 11604.5
_PARTITION_DENSITY_COEFFICIENT = 1.8786e20
_ELECTRON_PARTITION_COEFFICIENT = 2.0 * 2.4148e15

# exp() overflow guard for the constant branches. The reference H2 branch
# maps non-finite values to 0 (:155-156); the polynomial branch has no guard
# but stays far below overflow for any atmosphere temperature. Clamping at
# 700 keeps masked branches finite for autograd (0 * inf gradients) without
# changing any reachable value.
_LOG_OVERFLOW_GUARD = 700.0
# float64 underflow point for the fractions[0] <= 0 guard (log of the
# smallest positive normal, matching where the reference's linear fraction
# computation would flush to exactly 0).
_LOG_UNDERFLOW_GUARD = -700.0

_TWO_PI_TIMES_H2_MASS = 2.0 * np.pi * 1.008 * ATOMIC_MASS_GRAM_REFERENCE
_H2_DISSOCIATION_WAVENUMBER = 36118.11


@dataclass(frozen=True)
class TwinMolecularState:
    """One batch of converged molecular-equilibrium states.

    Scalar fields are ``(star, layer)``; molecular fields are
    ``(star, layer, molecule_count)`` and equation fields
    ``(star, layer, equation_count)``, following the reference's per-layer
    arrays (``MolecularEquilibriumState`` ``molecular_equilibrium.py:57-70``)
    and the trace layout (``molecular_populations`` (80, 170),
    ``molecular_equation_densities`` (80, 23)).

    Layout for downstream stages:

    * ``electron_density``, ``total_nuclei_number_density``, ``mass_density``
      and ``specific_internal_energy`` are the runtime-state updates the
      reference writes (:304-313, :346-353); the EOS/continuum chain consumes
      them in place of the atomic fixed point's values.
    * ``molecular_populations`` is the molecular number-density product
      (:315-330); ``partition_normalized_molecular_populations`` is the
      mode-1 refill (:356-485). ``populate_molecular_species`` (:488-593)
      selects between them by population mode when filling the packed tubes.
    * ``molecular_equation_densities`` is the *normalized* equation-density
      block — the reference refills it in place during the normalization pass
      (:372-415), and that is the layout the iter_1 traces store.
      ``raw_molecular_equation_densities`` keeps the pre-normalization Newton
      output (equation 0 is the heavy-nucleus density, the last equation the
      electron density when ``has_electron_equation``).
    * ``converged`` / ``iterations_used`` are diagnostics: the 0-based Newton
      sweep index at which each point first met the tolerance, or -1 where it
      never did (the reference returns the last iterate without raising).
    """

    electron_density: torch.Tensor
    total_nuclei_number_density: torch.Tensor
    mass_density: torch.Tensor
    mean_nuclear_mass_amu: torch.Tensor
    specific_internal_energy: torch.Tensor
    molecular_populations: torch.Tensor
    partition_normalized_molecular_populations: torch.Tensor
    molecular_equation_densities: torch.Tensor
    raw_molecular_equation_densities: torch.Tensor
    converged: torch.Tensor
    iterations_used: torch.Tensor


class TwinMoleculeTables:
    """Molecular catalog plus the static Newton-assembly index lists.

    Everything the reference kernels read from the catalog is loaded once and
    held as torch tensors; everything derived from *static* catalog structure
    (component scatter geometry, Jacobian sparsity, negative-ion flags, the
    Saha/partition row lists) is resolved on the host at load time. The EOS
    partition tables come from ``twin_eos.TwinEosTables`` (shared instance if
    passed, matching what a combined EOS+molecule driver would hold).
    """

    def __init__(
        self,
        *,
        catalog_path: Path | None = None,
        eos_tables: TwinEosTables | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.eos = (
            TwinEosTables(device=device, dtype=dtype)
            if eos_tables is None
            else eos_tables
        )

        catalog = read_molecular_equilibrium_catalog(
            Path(catalog_path)
            if catalog_path is not None
            else molecular_equilibrium_catalog_path()
        )
        molecule_count = int(catalog.molecule_count)
        equation_count = int(catalog.equation_count)
        self.molecule_count = molecule_count
        self.equation_count = equation_count

        with np.load(
            atmosphere_table_path("molecular_equilibrium_tables.npz"),
            allow_pickle=False,
        ) as table_data:
            h2_partition_table = np.asarray(
                table_data["h2_partition_function"], dtype=np.float64
            )

        def tensor(values, dtype=self.dtype):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

        molecule_codes = np.asarray(catalog.molecule_codes[:molecule_count])
        coefficients = np.asarray(
            catalog.equilibrium_coefficients[:6, :molecule_count], dtype=np.float64
        )
        starts = np.asarray(
            catalog.component_start_indices[: molecule_count + 1], dtype=np.int64
        )
        component_equations = np.asarray(
            catalog.component_equation_indices, dtype=np.int64
        )
        species_codes = np.asarray(
            catalog.equation_species_codes[:equation_count], dtype=np.int64
        )

        self.molecule_codes = tensor(molecule_codes)
        self.equilibrium_coefficients = tensor(coefficients)
        self.species_codes = tensor(species_codes, torch.int64)
        self.h2_partition_table = tensor(h2_partition_table)
        self.reference_atomic_mass_amu = tensor(REFERENCE_ATOMIC_MASS_AMU[:N_ELEMENTS])

        # --- static catalog structure ----------------------------------------
        component_count = starts[1:] - starts[:molecule_count]
        self.has_electron_equation = bool(species_codes[equation_count - 1] == 100)
        max_components = int(component_count.max()) if molecule_count else 0
        self.max_components = max_components

        # Padded component arrays, (molecule, slot): the mapped equation column
        # (raw index equation_count, the inverse-electron sentinel, maps to the
        # electron equation slot equation_count - 1, molecular_equilibrium.py
        # :800-805), the inverse flag, and the presence mask.
        comp_col = np.zeros((molecule_count, max_components), dtype=np.int64)
        comp_inverse = np.zeros((molecule_count, max_components), dtype=bool)
        comp_present = np.zeros((molecule_count, max_components), dtype=bool)
        for molecule_index in range(molecule_count):
            start, stop = starts[molecule_index], starts[molecule_index + 1]
            for slot, component_index in enumerate(range(start, stop)):
                raw = int(component_equations[component_index])
                comp_col[molecule_index, slot] = (
                    equation_count - 1 if raw == equation_count else raw
                )
                comp_inverse[molecule_index, slot] = raw == equation_count
                comp_present[molecule_index, slot] = True
        self.comp_col = tensor(comp_col, torch.int64)
        self.comp_inverse = tensor(comp_inverse, torch.bool)
        self.comp_present = tensor(comp_present, torch.bool)

        # Equation -> abundance column (species 1..99), -1 where none.
        abundance_index = np.full(equation_count, -1, dtype=np.int64)
        for equation_index in range(1, equation_count):
            code = int(species_codes[equation_index])
            if 0 < code < 100:
                abundance_index[equation_index] = code - 1
        self.abundance_index = tensor(abundance_index, torch.int64)

        # --- Newton scatter geometry over active molecules (count > 1) -------
        # Flat component list A: (molecule, column, sign). Jacobian pairs are
        # every (other, component) slot pair per molecule, :808-829.
        active = component_count > 1
        self.active_molecule = tensor(active, torch.bool)
        a_mol: list[int] = []
        a_col: list[int] = []
        a_sign: list[float] = []
        a_index: dict[tuple[int, int], int] = {}
        p_row: list[int] = []
        p_col: list[int] = []
        p_src: list[int] = []
        for molecule_index in range(molecule_count):
            if not active[molecule_index]:
                continue
            slots = range(int(component_count[molecule_index]))
            for slot in slots:
                a_index[(molecule_index, slot)] = len(a_mol)
                a_mol.append(molecule_index)
                a_col.append(int(comp_col[molecule_index, slot]))
                a_sign.append(-1.0 if comp_inverse[molecule_index, slot] else 1.0)
            for other_slot in slots:
                for slot in slots:
                    p_row.append(int(comp_col[molecule_index, other_slot]))
                    p_col.append(int(comp_col[molecule_index, slot]))
                    p_src.append(a_index[(molecule_index, slot)])
        self.a_mol = tensor(a_mol, torch.int64)
        self.a_col = tensor(a_col, torch.int64)
        self.a_sign = tensor(a_sign)
        self.p_flat_index = tensor(
            [row * equation_count + col for row, col in zip(p_row, p_col)],
            torch.int64,
        )
        self.p_src = tensor(p_src, torch.int64)

        # Negative-ion correction block (:831-857): molecules whose last
        # component sits in the species-100 equation. o_count is the number of
        # components mapping to the electron slot (normally one); the block
        # subtracts 2*term from the electron residual row per such component
        # and 2*derivative from every electron-row Jacobian entry of the
        # molecule. Note the correction derivative is always +term/d, even for
        # inverse-electron components.
        negative_mol: list[int] = []
        negative_ocount: list[float] = []
        nj_mol: list[int] = []
        nj_col: list[int] = []
        nj_scale: list[float] = []
        for molecule_index in range(molecule_count):
            if not active[molecule_index]:
                continue
            last_raw = int(component_equations[starts[molecule_index + 1] - 1])
            if not (
                last_raw < equation_count
                and int(species_codes[last_raw]) == 100
            ):
                continue
            count = int(component_count[molecule_index])
            o_count = int(
                sum(
                    comp_col[molecule_index, slot] == equation_count - 1
                    for slot in range(count)
                )
            )
            negative_mol.append(molecule_index)
            negative_ocount.append(float(o_count))
            for slot in range(count):
                nj_mol.append(molecule_index)
                nj_col.append(int(comp_col[molecule_index, slot]))
                nj_scale.append(-2.0 * o_count)
        self.negative_mol = tensor(negative_mol, torch.int64)
        self.negative_ocount = tensor(negative_ocount)
        self.nj_mol = tensor(nj_mol, torch.int64)
        self.nj_col = tensor(nj_col, torch.int64)
        self.nj_scale = tensor(nj_scale)

        # --- equilibrium-constant branches (:693-757) -------------------------
        first_coefficient = coefficients[0]
        ion_count = np.trunc(
            (molecule_codes - np.trunc(molecule_codes)) * 100.0 + 0.5
        ).astype(np.int64)
        self.ion_count = tensor(ion_count, torch.int64)
        self.component_count = tensor(component_count, torch.int64)
        self.h2_mask = tensor(np.abs(molecule_codes - 101.0) < 0.005, torch.bool)
        self.first_coefficient_zero = tensor(first_coefficient == 0.0, torch.bool)

        # Saha-ratio rows (:1118-1135): first coefficient zero, >1 component.
        # (molecule, 0-based element, component count).
        self.saha_rows = [
            (molecule_index, int(molecule_codes[molecule_index]) - 1, int(count))
            for molecule_index, count in enumerate(component_count)
            if first_coefficient[molecule_index] == 0.0 and count > 1
        ]
        # Mode-3 partition rows for the normalized refill (:465-485): first
        # coefficient zero, 1 <= Z <= 99; the kernel's single-value mode-3
        # output is the partition of stage min(count, work) - 1
        # (equation_of_state.py:861-874), with work = min(count+2, available);
        # available >= count for every catalog row (checked against the EOS
        # table decode), so selected = count - 1.
        self.branchb_rows = [
            (
                molecule_index,
                int(molecule_codes[molecule_index]) - 1,
                max(int(component_count[molecule_index]), 1) - 1,
            )
            for molecule_index in range(molecule_count)
            if first_coefficient[molecule_index] == 0.0
            and 1 <= int(molecule_codes[molecule_index]) <= N_ELEMENTS
        ]
        # Equation partition rows for the normalized refill (:372-415):
        # (equation, 0-based element); mode 3 with one stage returns U_0.
        self.equation_partition_rows = [
            (equation_index, int(species_codes[equation_index]) - 1)
            for equation_index in range(1, equation_count)
            if 0 < int(species_codes[equation_index]) < 100
        ]

        # Molecule masses for the normalized refill (:423-432): sum of the
        # component species masses, species 1..99 only.
        molecule_mass = np.zeros(molecule_count, dtype=np.float64)
        for molecule_index in range(molecule_count):
            start, stop = starts[molecule_index], starts[molecule_index + 1]
            for component_index in range(start, stop):
                raw = int(component_equations[component_index])
                if raw >= equation_count:
                    continue
                code = int(species_codes[raw])
                if 1 <= code <= REFERENCE_ATOMIC_MASS_AMU.size:
                    molecule_mass[molecule_index] += REFERENCE_ATOMIC_MASS_AMU[code - 1]
        self.molecule_mass_amu = tensor(molecule_mass)

    @classmethod
    def default(cls, **kwargs) -> "TwinMoleculeTables":
        return cls(**kwargs)


# --- equilibrium constants ----------------------------------------------------


def _h2_equilibrium_constant(temperature, h2_partition_table):
    """Vectorized H2 branch (molecular_equilibrium.py:129-156, :653-691).

    ``temperature`` is ``(points,)`` raw; the result is ``(points,)``. The
    overflow-to-zero guard of :155-156 becomes an exponent clamp plus a mask,
    which agrees wherever the reference value is finite.
    """

    finite_positive = torch.isfinite(temperature) & (temperature > 0.0)
    temperature = torch.where(
        finite_positive, temperature, torch.ones_like(temperature)
    )
    # Partition table interpolation (:117-126): clamp to [100, 19900], then
    # linear between the bracketing 100 K rows.
    interp_temperature = torch.clamp(temperature, min=100.0, max=19900.0)
    index = torch.clamp(torch.trunc(interp_temperature / 100.0).to(torch.int64), 1, 199)
    lower = h2_partition_table[index - 1]
    upper = h2_partition_table[index]
    partition = lower + (upper - lower) * (interp_temperature - index.to(lower.dtype) * 100.0) / 100.0

    denominator_argument = (
        _TWO_PI_TIMES_H2_MASS
        * BOLTZMANN_ERG_PER_K_REFERENCE
        / PLANCK_ERG_SECOND_REFERENCE**2
        * temperature
    )
    denominator_argument = torch.where(
        torch.isfinite(denominator_argument) & (denominator_argument > 0.0),
        denominator_argument,
        torch.full_like(denominator_argument, 1.0e-300),
    )
    denominator = denominator_argument**1.5
    exponent = (
        _H2_DISSOCIATION_WAVENUMBER
        * PLANCK_ERG_SECOND_REFERENCE
        * LIGHT_SPEED_CM_PER_S_REFERENCE
        / BOLTZMANN_ERG_PER_K_REFERENCE
        / torch.clamp(temperature, min=1.0e-30)
    )
    value = (
        partition
        * 2.0**1.5
        / 4.0
        / torch.clamp(denominator, min=1.0e-300)
        * torch.exp(torch.clamp(exponent, max=_LOG_OVERFLOW_GUARD))
    )
    return torch.where(exponent <= _LOG_OVERFLOW_GUARD, value, torch.zeros_like(value))


def _saha_ratio_constants(
    table: TwinMoleculeTables,
    temperature_k,
    thermal_energy_ev,
    lowering_ev,
    log_partition,
):
    """Saha-ratio constants for the first-coefficient-zero rows.

    Vectorized transcription of the mode-12 branch of
    ``_compute_equilibrium_constants_for_layer_compiled``
    (molecular_equilibrium.py:1118-1135). The constant is
    ``fractions[count-1]/fractions[0] * ne**ion_count``; the electron density
    cancels analytically against the Saha factor inside the fraction ratio
    (module docstring), so what remains is the ratio-chain product

        prod_s [ SAHA_COEFFICIENT * T^1.5 * U_s/U_{s-1}
                 * exp(-(chi_{s-1} - s*lowering)/kT_ev) ]

    evaluated from the twin_eos partition cube (``log_partition`` is
    ``(element, stage, points)`` with the reference's >=1 floor applied). The
    reference's ``fractions[0] <= 0 -> 0`` guard (:1127-1129) is applied at
    the float64 underflow point with the fraction-0 normalization taken over
    the requested stages only (the reference's two extra ladder stages move
    the guard by underflow-level amounts).
    """

    points = temperature_k.numel()
    dtype, device = temperature_k.dtype, temperature_k.device
    constants = torch.zeros(points, table.molecule_count, dtype=dtype, device=device)
    log_saha = math.log(SAHA_COEFFICIENT_REFERENCE) + 1.5 * torch.log(temperature_k)
    chi = table.eos.ionization_potential_ev
    for molecule_index, z0, count in table.saha_rows:
        log_weight = torch.zeros(points, dtype=dtype, device=device)
        log_chain = [log_weight]
        for stage in range(1, count):
            log_weight = log_weight + (
                log_saha
                + log_partition[z0, stage]
                - log_partition[z0, stage - 1]
                - (chi[z0, stage - 1] - stage * lowering_ev) / thermal_energy_ev
            )
            log_chain.append(log_weight)
        log_weight = torch.stack(log_chain, dim=1)  # (points, count)
        log_ratio_sum = log_weight[:, -1]
        # Underflow guard: the reference's linear fractions flush to exactly
        # 0 when the normalized weight falls below ~1e-308; logsumexp is
        # bracketed by the chain peak.
        log_peak = log_weight.amax(dim=1)
        guard = (log_peak <= -_LOG_UNDERFLOW_GUARD) & (
            log_peak - log_ratio_sum <= -_LOG_UNDERFLOW_GUARD
        )
        value = torch.exp(torch.clamp(log_ratio_sum, max=_LOG_OVERFLOW_GUARD))
        constants[:, molecule_index] = torch.where(
            guard & (log_ratio_sum <= _LOG_OVERFLOW_GUARD),
            value,
            torch.zeros_like(value),
        )
    return constants


def _equilibrium_constants(
    table: TwinMoleculeTables,
    temperature_raw,
    temperature_k,
    thermal_energy_ev,
    lowering_ev,
    log_partition,
):
    """Full (points, molecule) constant block (:693-757 and :1101-1147)."""

    dtype, device = temperature_k.dtype, temperature_k.device
    points = temperature_k.numel()
    coefficients = table.equilibrium_coefficients  # (6, molecule)
    c1 = coefficients[0][None, :]
    t_raw = temperature_raw[:, None]
    ln_temperature = torch.log(torch.clamp(t_raw, min=1.0e-300))

    polynomial = (
        coefficients[2][None, :]
        + (
            -coefficients[3][None, :]
            + (coefficients[4][None, :] - coefficients[5][None, :] * t_raw) * t_raw
        )
        * t_raw
    )
    exponent = (
        c1 / torch.clamp(t_raw / _KELVIN_PER_EV, min=1.0e-30)
        - coefficients[1][None, :]
        + polynomial * t_raw
        - 1.5
        * (
            table.component_count[None, :].to(dtype)
            - 2.0 * table.ion_count[None, :].to(dtype)
            - 1.0
        )
        * ln_temperature
    )
    polynomial_value = torch.exp(torch.clamp(exponent, max=_LOG_OVERFLOW_GUARD))
    polynomial_value = torch.where(
        (t_raw <= 10000.0) & (exponent <= _LOG_OVERFLOW_GUARD),
        polynomial_value,
        torch.zeros_like(polynomial_value),
    )

    h2_value = torch.where(
        t_raw <= 20000.0,
        _h2_equilibrium_constant(temperature_raw, table.h2_partition_table)[:, None],
        torch.zeros(points, 1, dtype=dtype, device=device),
    ).broadcast_to(points, table.molecule_count)

    saha_value = _saha_ratio_constants(
        table, temperature_k, thermal_energy_ev, lowering_ev, log_partition
    )
    saha_value = torch.where(
        table.component_count[None, :] > 1,
        saha_value,
        torch.ones_like(saha_value),
    )

    return torch.where(
        table.first_coefficient_zero[None, :],
        saha_value,
        torch.where(table.h2_mask[None, :], h2_value, polynomial_value),
    )


# --- molecular density products -------------------------------------------------


def _molecular_terms(constants, densities, table: TwinMoleculeTables):
    """The ``constants * prod(components)`` product, all molecules.

    ``constants``/return are ``(points, molecule)``, ``densities`` is
    ``(points, equation)``. Transcribes the component product of
    :318-330 / :798-805: normal components multiply, the inverse-electron
    sentinel divides by ``max(d_e, 1e-300)``.
    """

    term = constants
    gathered = densities[:, table.comp_col.reshape(-1)].reshape(
        densities.shape[0], table.molecule_count, table.max_components
    )
    for slot in range(table.max_components):
        present = table.comp_present[None, :, slot]
        value = gathered[:, :, slot]
        factor = torch.where(
            table.comp_inverse[None, :, slot],
            1.0 / torch.clamp(value, min=1.0e-300),
            value,
        )
        term = torch.where(present, term * factor, term)
    return term


# --- Newton assembly and update -----------------------------------------------


def _newton_matrix(term, densities, abundance, residual_seed, table: TwinMoleculeTables):
    """Residual and Jacobian for one sweep (:759-859), batched over points.

    ``term`` is the active-molecule product ``(points, molecule)`` (zero for
    inactive/zero-constant molecules, matching the ``term == 0: continue``
    skip); ``densities`` ``(points, equation)``; ``abundance``
    ``(points, equation)`` with the 1e-20 floor applied; ``residual_seed``
    ``(points,)`` is ``-P/(kT)``.
    """

    points, equation_count = densities.shape
    dtype, device = densities.dtype, densities.device

    residual = torch.zeros(points, equation_count, dtype=dtype, device=device)
    residual[:, 0] = residual_seed + densities[:, 1:].sum(dim=1)
    residual[:, 1:] = densities[:, 1:] - abundance[:, 1:] * densities[:, 0:1]
    if table.has_electron_equation:
        residual[:, equation_count - 1] = -densities[:, equation_count - 1]

    # Each active molecule contributes once to the total-particle equation,
    # independently of its component equations (:798-820).
    residual[:, 0] = residual[:, 0] + term[:, table.active_molecule].sum(dim=1)

    # Molecular terms scatter into every component row (:820).
    term_per_component = term[:, table.a_mol]
    residual.scatter_add_(
        1, table.a_col[None, :].expand(points, -1), term_per_component
    )
    # Negative-ion correction: electron row loses 2*term per electron-slot
    # component (:843-844).
    if table.negative_mol.numel():
        correction = (
            term[:, table.negative_mol] * table.negative_ocount[None, :]
        ).sum(dim=1)
        residual[:, equation_count - 1] = residual[:, equation_count - 1] - 2.0 * correction

    jacobian = torch.zeros(
        points, equation_count, equation_count, dtype=dtype, device=device
    )
    jacobian[:, 0, 1:] = 1.0
    jacobian[:, 1:, 0] = -abundance[:, 1:]
    diagonal = torch.arange(1, equation_count, device=device)
    jacobian[:, diagonal, diagonal] = 1.0
    if table.has_electron_equation:
        jacobian[:, equation_count - 1, equation_count - 1] = -1.0

    # Molecular derivative scatter (:807-829): for each component c of each
    # active molecule, derivative = +/- term / max(d[col_c], 1e-300) lands in
    # row 0 and in every component row of the molecule at column col_c.
    derivative = term_per_component * table.a_sign[None, :] / torch.clamp(
        densities[:, table.a_col], min=1.0e-300
    )
    jacobian[:, 0, :].scatter_add_(
        1, table.a_col[None, :].expand(points, -1), derivative
    )
    jacobian.reshape(points, -1).scatter_add_(
        1,
        table.p_flat_index[None, :].expand(points, -1),
        derivative[:, table.p_src],
    )
    # Negative-ion Jacobian correction (:845-857): electron-row entries of the
    # molecule lose 2 * (+term / d[col]).
    if table.nj_mol.numel():
        neg_derivative = (
            table.nj_scale[None, :]
            * term[:, table.nj_mol]
            / torch.clamp(densities[:, table.nj_col], min=1.0e-300)
        )
        jacobian[:, equation_count - 1, :].scatter_add_(
            1, table.nj_col[None, :].expand(points, -1), neg_derivative
        )
    return jacobian, residual


def _newton_update(densities, previous_delta, delta, tolerance):
    """One damped update sweep (:861-885), batched over points.

    Returns ``(new_densities, new_previous_delta, still_iterating)`` where
    ``still_iterating`` is the pre-damping convergence test ``(points,)``.
    The reference's sequential ``scale = sqrt(scale)`` mutation over the
    equation loop is vectorized as ``100 ** (0.5 ** k)`` with ``k`` the
    exclusive cumulative count of floor-and-flip events along the axis.
    """

    ratio = delta.abs() / torch.clamp(densities.abs(), min=1.0e-300)
    still_iterating = ratio.amax(dim=1) > tolerance
    flip = previous_delta * delta < 0.0
    delta = torch.where(flip, delta * 0.69, delta)
    updated = densities - delta
    take = updated.abs() >= densities / 100.0
    floor_flip = (~take) & flip
    floor_flip_count = floor_flip.to(torch.int64)
    count_before = torch.cumsum(floor_flip_count, dim=1) - floor_flip_count
    scale = 100.0 ** (0.5 ** count_before.to(densities.dtype))
    new_densities = torch.where(take, updated.abs(), densities / scale)
    return new_densities, delta, still_iterating


# --- the solve ------------------------------------------------------------------


def solve_molecular_equilibrium(
    temperature: torch.Tensor,
    gas_pressure: torch.Tensor,
    electron_density: torch.Tensor,
    abundances: torch.Tensor,
    *,
    table: TwinMoleculeTables,
    max_iterations: int = MAX_NEWTON_ITERATIONS,
    tolerance: float = NEWTON_TOLERANCE,
) -> TwinMolecularState:
    """Converged molecular-equilibrium state for a batch of atmospheres.

    ``temperature``, ``gas_pressure`` and ``electron_density`` are
    ``(star, layer)`` float64 tensors (the electron density is the incoming
    seed, used for the runtime-state charge-square seed exactly as
    ``build_runtime_state``/``update_charge_square_density`` do);
    ``abundances`` is ``(star, 99)`` or ``(99,)`` linear elemental number
    fractions, as in ``twin_eos.solve_populations``.

    The Newton solve runs a fixed ``max_iterations`` count (default 200, the
    reference cap) over all ``stars * layers`` points jointly, with a sticky
    convergence mask reproducing the reference's per-layer early break.
    Differentiable in temperature and gas pressure end to end.
    """

    dtype = table.dtype
    device = table.device
    temperature = torch.as_tensor(temperature, dtype=dtype, device=device)
    gas_pressure = torch.as_tensor(gas_pressure, dtype=dtype, device=device)
    electron_density_seed = torch.as_tensor(
        electron_density, dtype=dtype, device=device
    )
    abundances = torch.as_tensor(abundances, dtype=dtype, device=device)
    if abundances.dim() == 1:
        abundances = abundances[None, :]
    if temperature.shape != gas_pressure.shape or temperature.dim() != 2:
        raise ValueError("temperature and gas_pressure must be (star, layer)")
    if abundances.shape != (temperature.shape[0], N_ELEMENTS):
        raise ValueError("abundances must be (99,) or (star, 99)")

    stars, layers = temperature.shape
    points = stars * layers
    equation_count = table.equation_count

    def flat(x):
        return x.reshape(points)

    temperature_raw = flat(temperature)
    gas_pressure_flat = flat(gas_pressure)
    electron_flat = flat(electron_density_seed)

    # --- runtime seeds (runtime_state.py:192-250) ------------------------------
    thermal_energy_raw = temperature_raw * BOLTZMANN_ERG_PER_K_REFERENCE
    total_particle_density = gas_pressure_flat / torch.clamp(
        thermal_energy_raw, min=1.0e-300
    )
    excess = 2.0 * electron_flat - total_particle_density
    charge_square_seed = 2.0 * electron_flat + torch.where(
        excess > 0.0, 2.0 * excess, torch.zeros_like(excess)
    )
    abundance_points = (
        abundances.T[:, :, None]
        .expand(N_ELEMENTS, stars, layers)
        .reshape(N_ELEMENTS, points)
    )
    mean_nuclear_mass = (
        abundance_points * table.reference_atomic_mass_amu[:, None]
    ).sum(0)

    # Per-equation abundance vector with the 1e-20 floor
    # (molecular_equilibrium.py:222-241).
    abundance = torch.zeros(points, equation_count, dtype=dtype, device=device)
    valid = table.abundance_index >= 0
    abundance[:, valid] = torch.clamp(
        abundance_points[table.abundance_index[valid]].T, min=1.0e-20
    )

    # --- Saha partition cube at the molecular seed state -----------------------
    # Same temperature-only setup as twin_eos.solve_populations
    # (twin_eos.py:888-932), at the reference kernel's 1 K temperature floor
    # and the molecular seed charge-square density.
    eos = table.eos
    temperature_k = torch.clamp(temperature_raw, min=1.0)
    thermal_energy_erg = temperature_k * BOLTZMANN_ERG_PER_K_REFERENCE
    thermal_energy_ev = torch.clamp(
        temperature_k * BOLTZMANN_EV_PER_K_REFERENCE, min=1.0e-30
    )
    hc_over_kt = (
        PLANCK_ERG_SECOND_REFERENCE * LIGHT_SPEED_CM_PER_S_EXACT
    ) / torch.clamp(thermal_energy_erg, min=1.0e-300)
    log10_temperature = torch.log10(temperature_k)

    base = torch.zeros(N_ELEMENTS, MAX_ION_STAGES, points, dtype=dtype, device=device)
    ordinary = eos.packed_column >= 0
    if ordinary.any():
        base[ordinary] = _ordinary_partition_base(
            eos,
            eos.packed_column[ordinary],
            eos.ionization_potential_ev[ordinary],
            temperature_k,
        )
    for key_index, table_index in enumerate(eos.special_keys):
        entries = eos.special_key == key_index
        base[entries] = _special_partition_base(eos, table_index, hc_over_kt)
    occupation = _prepare_occupation_statics(
        eos, temperature_k, thermal_energy_ev, hc_over_kt
    )
    temp_bins = _iron_temperature_bins(log10_temperature)
    lowering = _debye_lowering(thermal_energy_erg, charge_square_seed)
    partition = _assemble_partitions(eos, base, occupation, temp_bins, lowering)
    log_partition = torch.log(partition)

    # --- equilibrium constants (frozen across the Newton sweep, as the
    # reference freezes them per layer, :1167-1171) ---------------------------
    constants = _equilibrium_constants(
        table,
        temperature_raw,
        temperature_k,
        thermal_energy_ev,
        lowering,
        log_partition,
    )

    residual_seed = -gas_pressure_flat / torch.clamp(thermal_energy_raw, min=1.0e-300)

    # --- layer-serial seeds + star-batched Newton (:258-295, :1180-1209) ------
    # The converged density vector from layer i seeds layer i+1 after scaling by
    # the gas-pressure ratio.  This selects a materially different Newton root
    # in some hot/low-gravity atmospheres, so it is physics, not merely an
    # optimization detail.  Stars remain batched; only the 80-layer carry is
    # sequential, exactly as in the reference.
    constants_grid = constants.reshape(stars, layers, table.molecule_count)
    abundance_grid = abundance.reshape(stars, layers, equation_count)
    residual_grid = residual_seed.reshape(stars, layers)
    pressure_grid = gas_pressure_flat.reshape(stars, layers)
    particle_grid = total_particle_density.reshape(stars, layers)
    temperature_grid = temperature_raw.reshape(stars, layers)
    density_layers = []
    convergence_layers = []
    iteration_layers = []
    previous_density = None
    equation_index = torch.arange(equation_count, device=device).unsqueeze(0)
    for layer in range(layers):
        if layer == 0:
            total_seed = torch.where(
                temperature_grid[:, 0] < 4000.0,
                particle_grid[:, 0],
                particle_grid[:, 0] / 2.0,
            )
            electron_seed = total_seed / 10.0
            densities_layer = electron_seed[:, None] * abundance_grid[:, 0]
            densities_layer = torch.where(
                equation_index == 0, total_seed[:, None], densities_layer
            )
            if table.has_electron_equation:
                densities_layer = torch.where(
                    equation_index == equation_count - 1,
                    electron_seed[:, None],
                    densities_layer,
                )
        else:
            ratio = pressure_grid[:, layer] / torch.clamp(
                pressure_grid[:, layer - 1], min=1.0e-300
            )
            densities_layer = previous_density * ratio[:, None]

        previous_delta = torch.zeros_like(densities_layer)
        converged_layer = torch.zeros(stars, dtype=torch.bool, device=device)
        iterations_layer = torch.full(
            (stars,), -1, dtype=torch.int64, device=device
        )
        for sweep in range(int(max_iterations)):
            term = _molecular_terms(
                constants_grid[:, layer], densities_layer, table
            )
            jacobian, residual = _newton_matrix(
                term,
                densities_layer,
                abundance_grid[:, layer],
                residual_grid[:, layer],
                table,
            )
            delta = torch.linalg.solve(jacobian, residual.unsqueeze(-1)).squeeze(-1)
            updated, updated_delta, still_iterating = _newton_update(
                densities_layer, previous_delta, delta, tolerance
            )
            active = ~converged_layer
            densities_layer = torch.where(active[:, None], updated, densities_layer)
            previous_delta = torch.where(
                active[:, None], updated_delta, previous_delta
            )
            newly = active & ~still_iterating
            iterations_layer = torch.where(
                newly, torch.full_like(iterations_layer, sweep), iterations_layer
            )
            converged_layer = converged_layer | newly
        density_layers.append(densities_layer)
        convergence_layers.append(converged_layer)
        iteration_layers.append(iterations_layer)
        previous_density = densities_layer

    densities = torch.stack(density_layers, dim=1).reshape(points, equation_count)
    converged = torch.stack(convergence_layers, dim=1).reshape(points)
    iterations_used = torch.stack(iteration_layers, dim=1).reshape(points)

    # --- molecular populations (:315-330) and runtime-state updates (:304-313) -
    molecular_populations = _molecular_terms(constants, densities, table)
    total_nuclei = densities[:, 0]
    mass_density = total_nuclei * mean_nuclear_mass * ATOMIC_MASS_GRAM_REFERENCE
    if table.has_electron_equation:
        electron_density = densities[:, equation_count - 1]
    else:
        electron_density = electron_flat

    # --- partition-normalized refill (:356-485) ---------------------------------
    normalized = densities.clone()
    for equation_index, z0 in table.equation_partition_rows:
        atomic_mass = table.reference_atomic_mass_amu[z0]
        denominator = (
            partition[z0, 0]
            * _PARTITION_DENSITY_COEFFICIENT
            * torch.sqrt(
                torch.clamp((atomic_mass * temperature_raw) ** 3, min=1.0e-300)
            )
        )
        normalized[:, equation_index] = densities[:, equation_index] / torch.clamp(
            denominator, min=1.0e-300
        )
    if table.has_electron_equation:
        normalized[:, equation_count - 1] = densities[:, equation_count - 1] / (
            _ELECTRON_PARTITION_COEFFICIENT
            * temperature_raw
            * torch.sqrt(torch.clamp(temperature_raw, min=1.0e-300))
        )

    normalized_populations = torch.zeros_like(molecular_populations)
    thermal_energy_ev_raw = temperature_raw / _KELVIN_PER_EV
    dissociation = torch.exp(
        torch.clamp(
            table.equilibrium_coefficients[0][None, :]
            / torch.clamp(thermal_energy_ev_raw[:, None], min=1.0e-300),
            max=_LOG_OVERFLOW_GUARD,
        )
    )
    first_nonzero = ~table.first_coefficient_zero
    if first_nonzero.any():
        value = _molecular_terms(dissociation, normalized, table)
        mass_factor = _PARTITION_DENSITY_COEFFICIENT * torch.sqrt(
            torch.clamp(
                (table.molecule_mass_amu[None, :] * temperature_raw[:, None]) ** 3,
                min=1.0e-300,
            )
        )
        normalized_populations = torch.where(
            first_nonzero[None, :],
            value * mass_factor,
            normalized_populations,
        )
    for molecule_index, z0, selected in table.branchb_rows:
        normalized_populations[:, molecule_index] = molecular_populations[
            :, molecule_index
        ] / torch.clamp(partition[z0, selected], min=1.0e-300)

    # --- population-mode-1 tail (:346-353) ---------------------------------------
    specific_internal_energy = (
        1.5 * gas_pressure_flat / torch.clamp(mass_density, min=1.0e-300)
    )

    def unflat(x):
        return x.reshape(stars, layers, *x.shape[1:])

    return TwinMolecularState(
        electron_density=unflat(electron_density),
        total_nuclei_number_density=unflat(total_nuclei),
        mass_density=unflat(mass_density),
        mean_nuclear_mass_amu=unflat(mean_nuclear_mass),
        specific_internal_energy=unflat(specific_internal_energy),
        molecular_populations=unflat(molecular_populations),
        partition_normalized_molecular_populations=unflat(normalized_populations),
        molecular_equation_densities=unflat(normalized),
        raw_molecular_equation_densities=unflat(densities),
        converged=unflat(converged),
        iterations_used=unflat(iterations_used),
    )
