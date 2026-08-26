"""Batched, differentiable twin of the reference EOS / population stage.

Reproduces the atomic (molecule-free) population phase of the classical
solver — ``prepare_population_state`` (``payne_zero_atmosphere/runner.py:452-537``)
with ``molecules_enabled=False`` — as a torch-native module that is batched over
stars and fully differentiable in its inputs. The reference chain is

* ``build_runtime_state`` / ``update_charge_square_density``
  (``runtime_state.py:192-250``) for the seeds,
* ``iterate_electron_density`` (``equation_of_state.py:1393-1500``, per-layer
  damped fixed point, max 200 iterations, tol 1e-4, damping 0.5) around the
  Saha kernel ``_saha_partition_depth_kernel`` (``equation_of_state.py:606-891``),
* ``populate_all_species`` (``equation_of_state.py:1629-1676``) refilling both
  packed population tables at the converged state, and
* ``update_doppler_line_strength_factors`` (``doppler.py:12-66``).

Design choices
--------------

Table set. The twin loads the ``atmosphere_tables/`` reference bundle
(``packed_level_metadata.npz``, ``ionization_potential_tables.npz``,
``iron_group_partition_tables.npz``, ``special_partition_tables.npz``,
``isotope_tables.npz``), not ``synthesis_tables/partition_saha_inputs.npz``.
The two are *not* the same data: the synthesis packed table has 374 columns
against the reference's 365, its ionization potentials differ, and it adds
special Ca/O partition branches the atmosphere kernel does not have (the
reference reaches special branches by packed table index — 1, 3, 4, 14, 45,
51, 52, 57, 63, 64, 91, 354, 355 — and treats Ca through the ordinary packed
table). Matching the reference kernel semantics therefore requires the
reference tables; there is no precision consequence, the data is identical to
what the numba path reads.

Batching. All physics is vectorized over ``N = stars * layers`` flattened
points; the per-element Saha stack is vectorized over a padded ``(99 elements,
10 stages, N)`` layout, so a whole batch is one tensor expression per fixed-
point sweep. No vmap is needed anywhere.

Fixed point. The electron-density iteration runs a fixed ``max_iterations``
count (default 200, the reference cap) with a sticky convergence mask: once a
point's relative update drops below ``tolerance`` its state is frozen with
``torch.where``, exactly reproducing the reference's per-layer early break.
There are no data-dependent Python scalars and no host synchronization in the
compute path, so the whole loop is autograd-safe. Convergence is reported per
point (``converged``, ``iterations_used``); unconverged points keep their last
iterate instead of raising.

Packed slot layout. Outputs follow the reference packed layout
(``population_layout.py``): slot 0-based, element ``Z <= 30`` starts at
``((Z-1)*(Z+2))//2`` with its ion stages consecutive, element ``Z >= 31``
starts at ``495 + (Z-31)*5`` using 3 of each 5-slot block. Slots past 837 are
the molecular region in the reference and stay zero here. The two population
tubes differ in stage coverage exactly like the reference schedule: the
ion-stage tube (mode 12) uses the charge-balance stage counts, the
partition-normalized tube (mode 11) uses the deeper mode-11 counts (10 stages
for the iron group Z=20..28), which changes the Saha normalization for those
elements — both ladders are evaluated.

Known gaps. Molecular coupling is out of scope: for cool atmospheres the
reference solves electron density through the molecular equilibrium, and this
module reproduces the atomic path only. ``check_twin_eos.py`` quantifies the
resulting electron-density discrepancy against the molecule-enabled reference.
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
    PLANCK_ERG_SECOND_REFERENCE,
    WAVENUMBER_PER_EV_REFERENCE,
)
from payne_zero_atmosphere.data_files import atmosphere_table_path
from payne_zero_atmosphere.population_layout import (
    decode_population_code,
    ion_stage_count_for_atomic_number,
    population_job_schedule,
)
from payne_zero_atmosphere.runtime_state import REFERENCE_ATOMIC_MASS_AMU


# Rounded EOS literals from equation_of_state.py:58-63. Restated here (rather
# than imported) so this module does not pull the numba kernel module in.
ELECTRON_CHARGE_ESU_REFERENCE = 4.801e-10
_ELECTRON_CHARGE_ESU_SQUARED = ELECTRON_CHARGE_ESU_REFERENCE**2
SAHA_COEFFICIENT_REFERENCE = 2.0 * 2.4148e15
ION_FRACTION_SCALE = (0.001, 0.01, 0.1, 1.0)

# Iron-group Debye-lowering grids, equation_of_state.py:1833-1842.
_DEBYE_LOWERING_GRID_CM = (500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0)
_DEBYE_LOWERING_LOG10_GRID = (
    2.69897,
    3.0,
    3.30103,
    3.60206,
    3.90309,
    4.20412,
    4.50515,
)

N_ELEMENTS = 99
MAX_ION_STAGES = 10
ION_STAGE_SLOTS = 1006
# Z 20..28 (0-based rows 19..27) form one contiguous iron-group block.
_IRON_ROW_START = 19
_IRON_ROW_STOP = 28

# Special partition branches reachable through the packed table index
# (equation_of_state.py:217-452). Table index 367 (O I) exists in the
# reference kernel but is unreachable: oxygen's block starts at column 27.
# Each entry: table_index -> (level-row field pair, level count, ground/base
# expression tag, Debye density-parameter coefficient in cm^-1).
_SPECIAL_DENSITY_COEFF_CM = {
    1: 109677.576 / (6.5 * 6.5),  # H I
    3: 109677.576 / (5.5 * 5.5),  # He I
    4: 4.0 * 109722.267 / (6.5 * 6.5),  # He II
    14: 109734.83 / (4.5 * 4.5),  # B I
    45: 109734.83 / (4.5 * 4.5),  # Na I
    51: 109734.83 / (4.5 * 4.5),  # Mg I
    52: 4.0 * 109734.83 / (5.5 * 5.5),  # Mg II
    57: 109735.08 / (5.5 * 5.5),  # Al I
    63: 0.0,  # Si I
    64: 4.0 * 109734.83 / (4.5 * 4.5),  # Si II
    91: 109734.83 / (5.5 * 5.5),  # K I
    354: 0.0,  # C I
    355: 0.0,  # C II
}

_SPECIAL_LEVEL_FIELDS = {
    1: ("hydrogen_neutral", 6),
    3: ("helium_neutral", 29),
    4: ("helium_singly_ionized", 6),
    14: ("boron_neutral", 7),
    45: ("sodium_neutral", 8),
    51: ("magnesium_neutral", 11),
    52: ("magnesium_singly_ionized", 6),
    57: ("aluminum_neutral", 9),
    63: ("silicon_neutral", 11),
    64: ("silicon_singly_ionized", 6),
    91: ("potassium_neutral", 8),
    354: ("carbon_neutral", 14),
    355: ("carbon_singly_ionized", 6),
}


def _start_and_available_ion_count(atomic_number: int, offsets: np.ndarray):
    """Host transcription of equation_of_state.py:73-91."""

    z = int(atomic_number)
    if z <= 28:
        start = int(offsets[z - 1])
        available = int(offsets[z] - start)
    else:
        start = 3 * z + 54
        available = 3
    if z == 6:
        start, available = 354, 6
    if z == 7:
        start, available = 360, 6
    if 20 <= z < 29:
        available = 10
    return start, available


def _mode11_stage_counts() -> dict[int, int]:
    """Mode-11 stage counts from the reference fill schedule."""

    counts: dict[int, int] = {}
    for job in population_job_schedule(include_molecules=False):
        if job.mode == 11:
            atomic_number, stage_count = decode_population_code(job.code)
            counts[atomic_number] = stage_count
    return counts


class TwinEosTables:
    """Reference EOS tables plus the static per-(element, stage) decode.

    Everything the reference kernel reads from data files is loaded once and
    held as torch tensors; everything the kernel derives from *static* per-
    (element, stage) information (ionization potentials with their fallback
    chain, packed statistical weights, block offsets, special-branch dispatch,
    slot geometry) is resolved once on the host at load time. Only quantities
    depending on temperature, electron density, or charge-square density are
    evaluated inside ``solve_populations``.
    """

    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        table_dir: Path | None = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        def load(name: str, key: str) -> np.ndarray:
            path = (
                Path(table_dir) / name
                if table_dir is not None
                else atmosphere_table_path(name)
            )
            with np.load(path, allow_pickle=False) as data:
                return np.asarray(data[key])

        packed_level_metadata = load("packed_level_metadata.npz", "packed_level_metadata")
        ionization_potential_cm = load(
            "ionization_potential_tables.npz", "ionization_potential_cm"
        )
        iron_group_grid = load(
            "iron_group_partition_tables.npz", "iron_group_partition_grid"
        )
        special_path = (
            Path(table_dir) / "special_partition_tables.npz"
            if table_dir is not None
            else atmosphere_table_path("special_partition_tables.npz")
        )
        with np.load(special_path, allow_pickle=False) as special_data:
            special = {key: special_data[key] for key in special_data.files}
        element_block_offsets = np.asarray(
            special["element_block_offsets"], dtype=np.int64
        )
        major_isotope_mass_amu = load("isotope_tables.npz", "major_isotope_mass_amu")

        def tensor(values, dtype=self.dtype):
            return torch.as_tensor(np.asarray(values), dtype=dtype, device=self.device)

        self.packed_level_metadata = tensor(packed_level_metadata, torch.int64)
        self.ion_fraction_scale = tensor(ION_FRACTION_SCALE)
        self.iron_group_partition_grid = tensor(iron_group_grid).reshape(7, -1)
        self.debye_lowering_grid_cm = tensor(_DEBYE_LOWERING_GRID_CM)
        self.debye_lowering_log10_grid = tensor(_DEBYE_LOWERING_LOG10_GRID)
        self.major_isotope_mass_amu = tensor(major_isotope_mass_amu)
        self.reference_atomic_mass_amu = tensor(REFERENCE_ATOMIC_MASS_AMU[:N_ELEMENTS])
        self.special_level_tables = {}
        for name, _count in _SPECIAL_LEVEL_FIELDS.values():
            self.special_level_tables[name + "_energy"] = tensor(
                special[f"{name}_level_energy_cm"]
            )
            self.special_level_tables[name + "_weight"] = tensor(
                special[f"{name}_level_statistical_weight"]
            )

        # --- static per-(element, stage) decode -----------------------------
        mode11_counts = _mode11_stage_counts()
        ionization_potential_cm = np.asarray(ionization_potential_cm, dtype=np.float64)
        packed_host = np.asarray(packed_level_metadata, dtype=np.int64)
        n_table_columns = packed_host.shape[1]

        chi = np.zeros((N_ELEMENTS, MAX_ION_STAGES), dtype=np.float64)
        stat_weight = np.zeros((N_ELEMENTS, MAX_ION_STAGES), dtype=np.float64)
        special_density_coeff = np.zeros((N_ELEMENTS, MAX_ION_STAGES), dtype=np.float64)
        special_key = np.full((N_ELEMENTS, MAX_ION_STAGES), -1, dtype=np.int64)
        packed_column = np.full((N_ELEMENTS, MAX_ION_STAGES), -1, dtype=np.int64)
        work12 = np.zeros(N_ELEMENTS, dtype=np.int64)
        work11 = np.zeros(N_ELEMENTS, dtype=np.int64)
        out12 = np.zeros(N_ELEMENTS, dtype=np.int64)
        out11 = np.zeros(N_ELEMENTS, dtype=np.int64)
        slot_start = np.zeros(N_ELEMENTS, dtype=np.int64)
        special_keys = sorted(_SPECIAL_DENSITY_COEFF_CM)

        for atomic_number in range(1, N_ELEMENTS + 1):
            z0 = atomic_number - 1
            start, available = _start_and_available_ion_count(
                atomic_number, element_block_offsets
            )
            n12 = ion_stage_count_for_atomic_number(atomic_number)
            n11 = mode11_counts[atomic_number]
            work = min(n12 + 2, available)
            work12[z0] = work
            work11[z0] = min(n11 + 2, available)
            out12[z0] = min(n12, work)
            out11[z0] = min(n11, work11[z0])
            slot_start[z0] = (
                ((atomic_number - 1) * (atomic_number + 2)) // 2
                if atomic_number <= 30
                else 495 + (atomic_number - 31) * 5
            )
            for stage in range(1, work11[z0] + 1):
                s0 = stage - 1
                table_index = start + s0  # 1-based, equation_of_state.py:655-664
                column = table_index - 1
                packed_value = int(packed_host[5, column]) if column < n_table_columns else 0
                packed_ionization = packed_value // 100
                stat_weight[z0, s0] = float(packed_value - packed_ionization * 100)
                potential_ev = float(packed_ionization) / 1000.0
                # equation_of_state.py:673-695: table lookup with fallback to
                # the previous slot, then to the previous stage.
                if atomic_number <= 30:
                    potential_index = (
                        atomic_number * (atomic_number + 1) // 2 + stage - 2
                    )
                else:
                    potential_index = atomic_number * 5 + 341 + stage - 2
                if 0 <= potential_index < ionization_potential_cm.size:
                    value = ionization_potential_cm[potential_index]
                    if value > 0.0:
                        potential_ev = value / WAVENUMBER_PER_EV_REFERENCE
                    elif (
                        potential_index - 1 >= 0
                        and ionization_potential_cm[potential_index - 1] > 0.0
                    ):
                        potential_ev = (
                            ionization_potential_cm[potential_index - 1]
                            / WAVENUMBER_PER_EV_REFERENCE
                        )
                if potential_ev <= 0.0 and stage > 1:
                    potential_ev = chi[z0, s0 - 1]
                chi[z0, s0] = potential_ev
                if table_index in _SPECIAL_DENSITY_COEFF_CM:
                    special_key[z0, s0] = special_keys.index(table_index)
                    special_density_coeff[z0, s0] = _SPECIAL_DENSITY_COEFF_CM[table_index]
                elif 20 <= atomic_number < 29:
                    pass  # iron-group grid branch
                else:
                    packed_column[z0, s0] = column

        self.ionization_potential_ev = tensor(chi)
        self.reference_temperature = tensor(
            np.maximum(chi * 2000.0 / 11.0, 1.0e-12)
        )
        self.statistical_weight = tensor(stat_weight)
        self.special_density_coeff_cm = tensor(special_density_coeff)
        self.special_key = tensor(special_key, torch.int64)
        self.packed_column = tensor(packed_column, torch.int64)
        self.work_stage_count12 = tensor(work12, torch.int64)
        self.work_stage_count11 = tensor(work11, torch.int64)
        self.slot_start = tensor(slot_start, torch.int64)
        self.special_keys = special_keys

        stage_axis = torch.arange(MAX_ION_STAGES, device=self.device)
        self.stage_charge = (stage_axis + 1).to(self.dtype)  # (stage,)
        work12_t = self.work_stage_count12[:, None]
        work11_t = self.work_stage_count11[:, None]
        self.stage_valid12 = stage_axis[None, :] < work12_t
        self.stage_valid11 = stage_axis[None, :] < work11_t
        self.output_mask12 = stage_axis[None, :] < tensor(out12, torch.int64)[:, None]
        self.output_mask11 = stage_axis[None, :] < tensor(out11, torch.int64)[:, None]

        # Slot scatter geometry: flat (element, stage) positions and their
        # packed slot targets, row-major so boolean-mask extraction lines up.
        flat_position = torch.arange(N_ELEMENTS * MAX_ION_STAGES, device=self.device)
        slot_of_position = (
            self.slot_start[:, None].expand(-1, MAX_ION_STAGES).reshape(-1)
            + stage_axis.repeat(N_ELEMENTS)
        )
        self._slot_index12 = slot_of_position[self.output_mask12.reshape(-1)]
        self._slot_index11 = slot_of_position[self.output_mask11.reshape(-1)]
        self._flat12 = flat_position[self.output_mask12.reshape(-1)]
        self._flat11 = flat_position[self.output_mask11.reshape(-1)]

        # Iron-group flat (stage, element) indices into the (56, 10, 9) slice
        # of the partition grid, in row-major (element, stage) order of rows
        # 19..27: position = stage * 9 + element_index.
        iron_stage, iron_element = torch.meshgrid(
            torch.arange(MAX_ION_STAGES, device=self.device),
            torch.arange(9, device=self.device),
            indexing="ij",
        )
        # meshgrid with indexing="ij" gives (stage, element); the (99, 10)
        # layout is (element, stage) — transpose to (9 elements, 10 stages).
        self._iron_grid_offset = (
            iron_stage.T * 9 + iron_element.T
        ).reshape(-1)  # (90,) flat within a temperature plane
        self._iron_charge = (iron_stage.T + 1).to(self.dtype).reshape(-1)

    @classmethod
    def default(cls, **kwargs) -> "TwinEosTables":
        return cls(**kwargs)


@dataclass(frozen=True)
class TwinPopulationState:
    """One batch of converged population states, matching the trace layout.

    All per-layer fields are ``(star, layer)``; the packed tubes are
    ``(star, layer, 1006)`` following the reference packed slot layout
    (module docstring). ``converged``/``iterations_used`` are diagnostics:
    ``iterations_used`` is the 0-based fixed-point sweep index at which each
    point first met the tolerance, or -1 where it never did.
    """

    electron_density: torch.Tensor
    total_nuclei_number_density: torch.Tensor
    mass_density: torch.Tensor
    charge_square_density: torch.Tensor
    mean_nuclear_mass_amu: torch.Tensor
    ion_stage_populations_by_packed_slot: torch.Tensor
    partition_normalized_populations_by_packed_slot: torch.Tensor
    fractional_doppler_widths: torch.Tensor
    partition_normalized_population_over_mass_density_and_fractional_doppler_width: (
        torch.Tensor
    )
    converged: torch.Tensor
    iterations_used: torch.Tensor


# --- partition functions ----------------------------------------------------


def _level_sum(tables: TwinEosTables, name: str, count: int, hc_over_kt):
    """Excited-level Boltzmann sum, levels 1..count-1 (vectorized over points)."""

    energy = tables.special_level_tables[name + "_energy"][1:count]
    weight = tables.special_level_tables[name + "_weight"][1:count]
    return (weight[:, None] * torch.exp(-energy[:, None] * hc_over_kt[None, :])).sum(0)


def _special_partition_base(tables: TwinEosTables, table_index: int, hc_over_kt):
    """Special light-element partition base (equation_of_state.py:217-452).

    ``hc_over_kt`` is ``(points,)``; the result is ``(points,)``. Each branch
    is the reference formula with its hard-coded high-lying terms; the level
    sums are vectorized but otherwise verbatim.
    """

    x = hc_over_kt
    if table_index == 1:  # H I
        return 2.0 + _level_sum(tables, "hydrogen_neutral", 6, x)
    if table_index == 3:  # He I
        return 1.0 + _level_sum(tables, "helium_neutral", 29, x)
    if table_index == 4:  # He II
        return 2.0 + _level_sum(tables, "helium_singly_ionized", 6, x)
    if table_index == 14:  # B I
        base = 2.0 + 4.0 * torch.exp(-15.25 * x)
        base = base + _level_sum(tables, "boron_neutral", 7, x)
        return base + (
            6.0 * torch.exp(-57786.80 * x)
            + 10.0 * torch.exp(-59989.0 * x)
            + 14.0 * torch.exp(-60031.03 * x)
            + 2.0 * torch.exp(-63561.0 * x)
        )
    if table_index == 45:  # Na I
        base = 2.0 + _level_sum(tables, "sodium_neutral", 8, x)
        return base + 10.0 * torch.exp(-34548.745 * x) + 14.0 * torch.exp(-34586.96 * x)
    if table_index == 51:  # Mg I
        base = 1.0 + _level_sum(tables, "magnesium_neutral", 11, x)
        return base + (
            5.0 * torch.exp(-53134.0 * x)
            + 15.0 * torch.exp(-54192.0 * x)
            + 28.0 * torch.exp(-54676.0 * x)
            + 9.0 * torch.exp(-57853.0 * x)
        )
    if table_index == 52:  # Mg II
        base = 2.0 + _level_sum(tables, "magnesium_singly_ionized", 6, x)
        return base + (
            10.0 * torch.exp(-93310.80 * x)
            + 14.0 * torch.exp(-93799.70 * x)
            + 6.0 * torch.exp(-97464.32 * x)
            + 10.0 * torch.exp(-103419.82 * x)
            + 14.0 * torch.exp(-103689.89 * x)
            + 18.0 * torch.exp(-103705.66 * x)
        )
    if table_index == 57:  # Al I
        base = 2.0 + 4.0 * torch.exp(-112.061 * x)
        base = base + _level_sum(tables, "aluminum_neutral", 9, x)
        return base + 10.0 * torch.exp(-42235.0 * x) + 14.0 * torch.exp(-43831.0 * x)
    if table_index == 63:  # Si I
        base = 1.0 + 3.0 * torch.exp(-77.115 * x) + 5.0 * torch.exp(-223.157 * x)
        base = base + _level_sum(tables, "silicon_neutral", 11, x)
        return base + (
            76.0 * torch.exp(-53000.0 * x)
            + 71.0 * torch.exp(-57000.0 * x)
            + 191.0 * torch.exp(-60000.0 * x)
            + 240.0 * torch.exp(-62000.0 * x)
            + 251.0 * torch.exp(-63000.0 * x)
            + 300.0 * torch.exp(-65000.0 * x)
        )
    if table_index == 64:  # Si II
        base = 2.0 + 4.0 * torch.exp(-287.32 * x)
        base = base + _level_sum(tables, "silicon_singly_ionized", 6, x)
        return base + (
            6.0 * torch.exp(-81231.59 * x)
            + 6.0 * torch.exp(-83937.08 * x)
            + 10.0 * torch.exp(-101024.09 * x)
            + 14.0 * torch.exp(-103556.35 * x)
            + 10.0 * torch.exp(-108800.0 * x)
            + 42.0 * torch.exp(-115000.0 * x)
            + 6.0 * torch.exp(-121000.0 * x)
            + 38.0 * torch.exp(-125000.0 * x)
            + 34.0 * torch.exp(-132000.0 * x)
        )
    if table_index == 91:  # K I
        base = 2.0 + _level_sum(tables, "potassium_neutral", 8, x)
        return base + 10.0 * torch.exp(-27397.077 * x) + 14.0 * torch.exp(-28127.85 * x)
    if table_index == 354:  # C I
        base = 1.0 + 3.0 * torch.exp(-16.42 * x) + 5.0 * torch.exp(-43.42 * x)
        base = base + _level_sum(tables, "carbon_neutral", 14, x)
        return base + (
            108.0 * torch.exp(-80000.0 * x)
            + 189.0 * torch.exp(-84000.0 * x)
            + 247.0 * torch.exp(-87000.0 * x)
            + 231.0 * torch.exp(-88000.0 * x)
            + 190.0 * torch.exp(-89000.0 * x)
            + 300.0 * torch.exp(-90000.0 * x)
        )
    if table_index == 355:  # C II
        base = 2.0 + 4.0 * torch.exp(-63.42 * x)
        base = base + _level_sum(tables, "carbon_singly_ionized", 6, x)
        return base + (
            6.0 * torch.exp(-131731.80 * x)
            + 4.0 * torch.exp(-142027.1 * x)
            + 10.0 * torch.exp(-145550.13 * x)
            + 10.0 * torch.exp(-150463.62 * x)
            + 2.0 * torch.exp(-157234.07 * x)
            + 6.0 * torch.exp(-162500.0 * x)
            + 42.0 * torch.exp(-168000.0 * x)
            + 56.0 * torch.exp(-178000.0 * x)
            + 102.0 * torch.exp(-183000.0 * x)
            + 400.0 * torch.exp(-188000.0 * x)
        )
    raise KeyError(f"no special partition branch for table index {table_index}")


def _ordinary_partition_base(
    tables: TwinEosTables, columns: torch.Tensor, chi_ev: torch.Tensor, temperature
):
    """Packed-table partition interpolation (equation_of_state.py:737-784).

    ``columns``/``chi_ev`` are ``(entries,)`` static per ordinary (element,
    stage); ``temperature`` is ``(points,)``. Returns ``(entries, points)``.
    Table-cell and digit decode are integer ops on the int64 metadata, so the
    decode is piecewise constant in temperature (zero gradient through the
    cell choice), exactly like the reference.
    """

    reference_temperature = torch.clamp(chi_ev * 2000.0 / 11.0, min=1.0e-12)
    ratio = temperature[None, :] / reference_temperature[:, None]
    temperature_bin = torch.clamp(torch.trunc(ratio - 0.5).to(torch.int64), 1, 9)
    delta = ratio - temperature_bin.to(temperature.dtype) - 0.5
    row = (temperature_bin + 1) // 2 - 1

    metadata = tables.packed_level_metadata
    packed = metadata[row, columns[:, None]]
    first = packed // 100000
    second = packed - first * 100000
    second_value = second // 10
    scale_index = torch.clamp(second - second_value * 10, 1, 4)
    scale_value = tables.ion_fraction_scale[scale_index - 1]

    next_packed = metadata[row + 1, columns[:, None]]
    next_first = next_packed // 100000
    next_scale_index = torch.clamp(next_packed % 10, 1, 4)
    next_scale_value = tables.ion_fraction_scale[next_scale_index - 1]

    odd = (temperature_bin % 2) == 1
    left_odd = first.to(temperature.dtype) * scale_value
    right_odd = second_value.to(temperature.dtype) * scale_value
    left_even = right_odd
    right_even = next_first.to(temperature.dtype) * next_scale_value
    left = torch.where(odd, left_odd, left_even)
    right = torch.where(odd, right_odd, right_even)

    floor_condition = (
        odd
        & (delta < 0.0)
        & (scale_index <= 1)
        & (torch.trunc(left_odd) == torch.trunc(right_odd + 0.5))
    )
    minimum_partition = torch.where(
        floor_condition, torch.trunc(left_odd), torch.ones_like(left_odd)
    )
    return torch.maximum(minimum_partition, left + (right - left) * delta)


def _iron_temperature_bins(log10_temperature):
    """PFIRON temperature brackets (equation_of_state.py:454-470).

    Returns lower/upper plane indices and the interpolation weight, each
    ``(points,)``. The reference uses C-style truncation toward zero, which
    ``torch.trunc`` reproduces.
    """

    log_t = log10_temperature
    hot = log_t > 4.0
    cool = log_t < 3.7
    bin_hot = torch.clamp(torch.trunc((log_t - 4.0) / 0.05).to(torch.int64) + 31, max=56)
    bin_cool = torch.clamp(torch.trunc((log_t - 3.32) / 0.02).to(torch.int64) + 2, min=2)
    bin_mid = torch.trunc((log_t - 3.7) / 0.03).to(torch.int64) + 21
    upper_bin = torch.where(hot, bin_hot, torch.where(cool, bin_cool, bin_mid))
    weight_hot = (log_t - (bin_hot - 31).to(log_t.dtype) * 0.05 - 4.0) / 0.05
    weight_cool = (log_t - (bin_cool - 2).to(log_t.dtype) * 0.02 - 3.32) / 0.02
    weight_mid = (log_t - (bin_mid - 21).to(log_t.dtype) * 0.03 - 3.7) / 0.03
    weight = torch.where(hot, weight_hot, torch.where(cool, weight_cool, weight_mid))
    upper_index = upper_bin - 1
    lower_index = upper_index - 1
    return lower_index, upper_index, weight


def _iron_group_partitions(tables: TwinEosTables, temp_bins, lowering_ev):
    """Iron-group bilinear interpolation (equation_of_state.py:493-555).

    ``temp_bins`` comes from ``_iron_temperature_bins`` (fixed across the
    fixed-point sweep); ``lowering_ev`` is the per-point Debye lowering
    ``(points,)``. Returns ``(9 elements, 10 stages, points)``.
    """

    lower_t, upper_t, weight_t = temp_bins
    grid = tables.iron_group_partition_grid  # (7 lowering, 56*90)
    offset = tables._iron_grid_offset[:, None]  # (90, 1), stage-major per element
    line_upper = upper_t[None, :] * 90 + offset  # (90, points)
    line_lower = lower_t[None, :] * 90 + offset
    weight = weight_t[None, :]

    lowering_cm = (
        tables._iron_charge[:, None] * lowering_ev[None, :] * WAVENUMBER_PER_EV_REFERENCE
    )
    # Reference bracket: first grid level strictly above the lowering; level 0
    # handles the weak-lowering branch without log interpolation.
    bracket = torch.searchsorted(
        tables.debye_lowering_grid_cm, lowering_cm.contiguous(), right=True
    )
    weak = bracket == 0
    bracket = torch.clamp(bracket, 1, 6)
    lowering_weight = (
        torch.log10(torch.clamp(lowering_cm, min=1.0e-30))
        - tables.debye_lowering_log10_grid[bracket - 1]
    ) / 0.30103

    def temperature_interp(level_index):
        upper_value = grid[level_index, line_upper]
        lower_value = grid[level_index, line_lower]
        return weight * upper_value + (1.0 - weight) * lower_value

    weak_value = temperature_interp(torch.zeros_like(bracket))
    upper_value = temperature_interp(bracket)
    lower_value = temperature_interp(bracket - 1)
    interpolated = torch.where(
        weak,
        weak_value,
        lowering_weight * upper_value + (1.0 - lowering_weight) * lower_value,
    )
    return interpolated.reshape(9, MAX_ION_STAGES, -1)


def _occupation_term(density_parameter, charge, thermal_energy_ev):
    """equation_of_state.py:175-184 (``_occupation_term_compiled``).

    The sqrt argument is capped at 1e100: that regime is only reachable from
    masked-out branches (every gate the reference applies keeps the argument
    below ~1e4), and the cap keeps masked intermediates finite so autograd
    does not produce 0 * inf gradients.
    """

    x = torch.sqrt(
        torch.clamp(
            13.595
            * charge
            * charge
            / torch.clamp(thermal_energy_ev * density_parameter, min=1.0e-300),
            max=1.0e100,
        )
    )
    polynomial = (1.0 / 3.0) + (
        1.0
        - (0.5 + (1.0 / 18.0 + density_parameter / 120.0) * density_parameter)
        * density_parameter
    ) * density_parameter
    return x * x * x * polynomial


# --- Saha ladder and the population solve -----------------------------------


def _saha_fractions(
    partition,
    ionization_potential_ev,
    lowering_ev,
    thermal_energy_ev,
    log_saha_factor,
    stage_valid,
):
    """Ion-stage fractions from the Saha chain (equation_of_state.py:818-849).

    The reference accumulates the ratio chain linearly and normalizes with a
    Horner pass; here the identical algebra is done in log space (cumsum +
    softmax), which agrees to float64 roundoff and cannot overflow.
    ``partition`` is ``(element, stage, point)`` with the reference's >=1
    floor already applied; ``stage_valid`` masks padding stages to zero
    weight.
    """

    charge_previous = torch.arange(
        1, MAX_ION_STAGES, dtype=partition.dtype, device=partition.device
    )[:, None]  # charge of the *lower* stage in each ratio
    log_partition = torch.log(partition)
    log_ratio = (
        log_saha_factor[None, None, :]
        + log_partition[:, 1:, :]
        - log_partition[:, :-1, :]
        - (
            ionization_potential_ev[:, :-1, None]
            - charge_previous[None, :, :] * lowering_ev[None, None, :]
        )
        / torch.clamp(thermal_energy_ev, min=1.0e-30)[None, None, :]
    )
    log_population = torch.cumsum(
        torch.cat([torch.zeros_like(log_ratio[:, :1, :]), log_ratio], dim=1), dim=1
    )
    log_population = torch.where(
        stage_valid[:, :, None], log_population, torch.full_like(log_population, -torch.inf)
    )
    peak = log_population.amax(dim=1, keepdim=True)
    weight = torch.exp(log_population - peak)
    return weight / weight.sum(dim=1, keepdim=True)


def _debye_lowering(thermal_energy_erg, charge_square_density):
    """Per-unit-charge Debye lowering in eV (equation_of_state.py:634-648)."""

    charge_square = torch.clamp(charge_square_density, min=1.0e-30)
    debye_length = torch.sqrt(
        thermal_energy_erg / (12.5664 * _ELECTRON_CHARGE_ESU_SQUARED * charge_square)
    )
    return torch.clamp(
        1.44e-7 / torch.clamp(debye_length, min=1.0e-300), max=1.0
    )


def _assemble_partitions(
    tables: TwinEosTables,
    base,
    occupation,
    temp_bins,
    lowering_ev,
):
    """Full partition cube at one Debye lowering.

    ``base`` holds the temperature-only branches (ordinary packed
    interpolation and the special light-element sums, iron slots zero);
    ``occupation`` bundles the precomputed statics of the occupation
    correction. The iron-group block and the lowering-dependent occupation
    addition are the only pieces refreshed per fixed-point sweep, matching
    what varies with the iterated charge state in the reference kernel.
    """

    static_gate, kT_eff, lower_density, sw_eff, lowering_floor = occupation
    partition = base
    iron = _iron_group_partitions(tables, temp_bins, lowering_ev)
    partition = torch.cat(
        [
            partition[:_IRON_ROW_START],
            iron,
            partition[_IRON_ROW_STOP:],
        ],
        dim=0,
    )

    charge = tables.stage_charge[None, :, None]
    stage_lowering = charge * lowering_ev[None, None, :]
    active = static_gate & (stage_lowering >= lowering_floor) & (stage_lowering > 0.0)
    upper_density = stage_lowering / kT_eff
    addition = (
        sw_eff
        * torch.exp(-tables.ionization_potential_ev[:, :, None] / kT_eff)
        * (
            _occupation_term(upper_density, charge, kT_eff)
            - _occupation_term(lower_density, charge, kT_eff)
        )
    )
    partition = partition + torch.where(active, addition, torch.zeros_like(addition))
    return torch.clamp(partition, min=1.0)


def _prepare_occupation_statics(tables: TwinEosTables, temperature, thermal_energy_ev, hc_over_kt):
    """Per-(element, stage, point) occupation-correction statics.

    Folds the reference's two occupation branches
    (``_occupation_correction_compiled`` equation_of_state.py:186-214 plus the
    ordinary-path gate at :792-813 and the special-path density parameter) into
    one masked tensor expression. Everything here depends on temperature only,
    so it is computed once per solve.
    """

    chi = tables.ionization_potential_ev
    reference_temperature = tables.reference_temperature
    sw_meta = tables.statistical_weight
    density_coeff = tables.special_density_coeff_cm
    special = (tables.special_key >= 0)[:, :, None]
    iron = torch.zeros_like(special)
    iron[_IRON_ROW_START:_IRON_ROW_STOP] = True

    points = temperature.numel()
    kT = thermal_energy_ev[None, None, :].broadcast_to(
        N_ELEMENTS, MAX_ION_STAGES, points
    )
    # Ordinary branch: capped thermal energy and 0.1/kT lower cutoff; the gate
    # needs nonzero metadata weight and T >= 4 * reference temperature.
    kT_capped = torch.where(
        temperature[None, None, :] > 11.0 * reference_temperature[:, :, None],
        11.0 * reference_temperature[:, :, None] * BOLTZMANN_EV_PER_K_REFERENCE,
        kT,
    )
    ordinary_gate = (sw_meta[:, :, None] != 0.0) & (
        temperature[None, None, :] >= 4.0 * reference_temperature[:, :, None]
    )
    # Special branch: density parameter is coeff * hc/kT, no temperature gate.
    special_gate = (density_coeff[:, :, None] > 0.0).broadcast_to(
        N_ELEMENTS, MAX_ION_STAGES, points
    )

    kT_eff = torch.where(special, kT, kT_capped)
    lower_density = torch.where(
        special,
        density_coeff[:, :, None] * hc_over_kt[None, None, :],
        0.1 / torch.clamp(kT_capped, min=1.0e-30),
    )
    sw_eff = torch.where(special, torch.clamp(sw_meta[:, :, None], min=2.0), sw_meta[:, :, None])
    static_gate = torch.where(special, special_gate, ordinary_gate)
    # Stage-lowering threshold: 0.1 eV for the ordinary branch, strict
    # positivity for the special branch (folded into ``active`` via > 0).
    lowering_floor = torch.where(
        special, torch.zeros_like(kT_eff), torch.full_like(kT_eff, 0.1)
    )

    # Entries with no active occupation branch (including the iron group,
    # whose kernel branch skips occupation entirely, and special branches
    # with a zero density coefficient) get inert statics so the masked
    # expression stays finite in both forward and backward.
    any_branch = (
        (special & (density_coeff[:, :, None] > 0.0))
        | (~special & (sw_meta[:, :, None] != 0.0))
    ) & ~iron
    kT_eff = torch.where(any_branch, kT_eff, torch.ones_like(kT_eff))
    lower_density = torch.where(any_branch, lower_density, torch.ones_like(lower_density))
    static_gate = static_gate & any_branch
    kT_eff = torch.clamp(kT_eff, min=1.0e-30)
    lower_density = torch.clamp(lower_density, min=1.0e-300)
    return (static_gate, kT_eff, lower_density, sw_eff, lowering_floor)


def solve_populations(
    temperature: torch.Tensor,
    gas_pressure: torch.Tensor,
    electron_density_seed: torch.Tensor,
    abundances: torch.Tensor,
    microturbulence: torch.Tensor,
    *,
    tables: TwinEosTables,
    max_iterations: int = 200,
    tolerance: float = 1.0e-4,
) -> TwinPopulationState:
    """Converged EOS population state for a batch of atmospheres.

    Parameters are ``(star, layer)`` float64 tensors except ``abundances``,
    which is ``(star, 99)`` or ``(99,)`` linear elemental number fractions
    (``runtime_state.py:165-180``: 0.92/0.08 H/He plus deck values).
    ``electron_density_seed`` is the atmosphere's incoming electron density
    and ``microturbulence`` the per-layer microturbulence in cm/s.

    The fixed point follows ``_iterate_electron_density_parallel``
    (``equation_of_state.py:983-1052``): charge-square density carries between
    sweeps, the update is damped by 0.5 with a 0.5*ne floor, and a sticky
    convergence mask freezes points once their relative change drops below
    ``tolerance`` — the reference's per-layer early break, expressed without
    data-dependent host control flow.
    """

    dtype = tables.dtype
    device = tables.device
    temperature = torch.as_tensor(temperature, dtype=dtype, device=device)
    gas_pressure = torch.as_tensor(gas_pressure, dtype=dtype, device=device)
    electron_density = torch.as_tensor(electron_density_seed, dtype=dtype, device=device)
    microturbulence = torch.as_tensor(microturbulence, dtype=dtype, device=device)
    abundances = torch.as_tensor(abundances, dtype=dtype, device=device)
    if abundances.dim() == 1:
        abundances = abundances[None, :]
    if temperature.shape != gas_pressure.shape or temperature.dim() != 2:
        raise ValueError("temperature and gas_pressure must be (star, layer)")
    if abundances.shape != (temperature.shape[0], N_ELEMENTS):
        raise ValueError("abundances must be (99,) or (star, 99)")

    stars, layers = temperature.shape
    points = stars * layers

    def flat(x):
        return x.reshape(points)

    # The kernel floors temperature at 1 K (equation_of_state.py:622); the
    # total-particle density and the Doppler widths use the raw column.
    temperature_k = torch.clamp(flat(temperature), min=1.0)
    thermal_energy_erg = temperature_k * BOLTZMANN_ERG_PER_K_REFERENCE
    thermal_energy_erg_raw = flat(temperature) * BOLTZMANN_ERG_PER_K_REFERENCE
    thermal_energy_ev = temperature_k * BOLTZMANN_EV_PER_K_REFERENCE
    hc_over_kt = (PLANCK_ERG_SECOND_REFERENCE * LIGHT_SPEED_CM_PER_S_EXACT) / torch.clamp(
        thermal_energy_erg, min=1.0e-300
    )
    log10_temperature = torch.log10(temperature_k)
    gas_pressure_flat = flat(gas_pressure)
    total_particle_density = gas_pressure_flat / torch.clamp(
        thermal_energy_erg_raw, min=1.0e-300
    )

    # Per-point abundance and mean nuclear mass (runtime_state.py:183-189).
    abundance_points = (
        abundances.T[:, :, None].expand(N_ELEMENTS, stars, layers).reshape(N_ELEMENTS, points)
    )
    mean_nuclear_mass_points = (
        abundance_points * tables.reference_atomic_mass_amu[:, None]
    ).sum(0)

    # --- temperature-only partition bases -----------------------------------
    base = torch.zeros(
        N_ELEMENTS, MAX_ION_STAGES, points, dtype=dtype, device=device
    )
    ordinary = tables.packed_column >= 0
    if ordinary.any():
        ordinary_values = _ordinary_partition_base(
            tables,
            tables.packed_column[ordinary],
            tables.ionization_potential_ev[ordinary],
            temperature_k,
        )
        base[ordinary] = ordinary_values
    for key_index, table_index in enumerate(tables.special_keys):
        entries = tables.special_key == key_index
        # Every special table index maps to exactly one (element, stage), so
        # the (points,) branch value broadcasts onto the single selected row.
        base[entries] = _special_partition_base(tables, table_index, hc_over_kt)
    occupation = _prepare_occupation_statics(
        tables, temperature_k, thermal_energy_ev, hc_over_kt
    )
    temp_bins = _iron_temperature_bins(log10_temperature)

    # --- seeds (runtime_state.py:192-250) ------------------------------------
    electron_density = flat(electron_density)
    excess = 2.0 * electron_density - total_particle_density
    charge_square_density = 2.0 * electron_density + torch.where(
        excess > 0.0, 2.0 * excess, torch.zeros_like(excess)
    )
    total_nuclei = total_particle_density - electron_density

    # Net charge per stage is the 0-based stage index (neutral contributes 0):
    # equation_of_state.py:1017-1020. The 1-based stage charge is only for the
    # Debye lowering inside the Saha kernel.
    charge_index = (tables.stage_charge - 1.0)[None, :, None]
    output_mask12 = tables.output_mask12[:, :, None]
    converged = torch.zeros(points, dtype=torch.bool, device=device)
    iterations_used = torch.full((points,), -1, dtype=torch.int64, device=device)

    for sweep in range(int(max_iterations)):
        lowering = _debye_lowering(thermal_energy_erg, charge_square_density)
        partition = _assemble_partitions(tables, base, occupation, temp_bins, lowering)
        log_saha_factor = (
            math.log(SAHA_COEFFICIENT_REFERENCE)
            + 1.5 * torch.log(temperature_k)
            - torch.log(torch.clamp(electron_density, min=1.0e-40))
        )
        fractions = _saha_fractions(
            partition,
            tables.ionization_potential_ev,
            lowering,
            thermal_energy_ev,
            log_saha_factor,
            tables.stage_valid12,
        )
        populations = (
            torch.where(output_mask12, fractions, torch.zeros_like(fractions))
            * (total_nuclei[None, None, :] * abundance_points[:, None, :])
        )
        updated_electron_density = (populations * charge_index).sum(dim=(0, 1))
        charge_square_partial = (populations * charge_index * charge_index).sum(
            dim=(0, 1)
        )

        new_electron_density = torch.maximum(
            updated_electron_density, electron_density * 0.5
        )
        new_electron_density = 0.5 * (new_electron_density + electron_density)
        relative_error = torch.abs(
            (electron_density - new_electron_density)
            / torch.clamp(new_electron_density, min=1.0e-300)
        )

        active = ~converged
        electron_density = torch.where(active, new_electron_density, electron_density)
        total_nuclei = torch.where(
            active, total_particle_density - new_electron_density, total_nuclei
        )
        charge_square_density = torch.where(
            active,
            charge_square_partial + new_electron_density,
            charge_square_density,
        )
        newly = active & (relative_error < tolerance)
        iterations_used = torch.where(
            newly, torch.full_like(iterations_used, sweep), iterations_used
        )
        converged = converged | newly

    # --- packed tubes at the converged state ---------------------------------
    # populate_all_species (equation_of_state.py:1503-1547) re-evaluates the
    # Saha chains at the converged (ne, charge-square) point: mode 12 for the
    # ion-stage tube, mode 11 for the partition-normalized tube. The mode-11
    # stage counts are deeper for the iron group, so its fractions differ.
    lowering = _debye_lowering(thermal_energy_erg, charge_square_density)
    partition = _assemble_partitions(tables, base, occupation, temp_bins, lowering)
    log_saha_factor = (
        math.log(SAHA_COEFFICIENT_REFERENCE)
        + 1.5 * torch.log(temperature_k)
        - torch.log(torch.clamp(electron_density, min=1.0e-40))
    )
    fractions12 = _saha_fractions(
        partition,
        tables.ionization_potential_ev,
        lowering,
        thermal_energy_ev,
        log_saha_factor,
        tables.stage_valid12,
    )
    fractions11 = _saha_fractions(
        partition,
        tables.ionization_potential_ev,
        lowering,
        thermal_energy_ev,
        log_saha_factor,
        tables.stage_valid11,
    )
    scale = total_nuclei[None, None, :] * abundance_points[:, None, :]
    ion_stage_values = torch.where(output_mask12, fractions12, torch.zeros_like(fractions12)) * scale
    normalized_values = (
        torch.where(
            tables.output_mask11[:, :, None], fractions11, torch.zeros_like(fractions11)
        )
        / partition
        * scale
    )

    def scatter_slots(values, flat_index, slot_index):
        slots = torch.zeros(points, ION_STAGE_SLOTS, dtype=dtype, device=device)
        extracted = values.reshape(N_ELEMENTS * MAX_ION_STAGES, points)[flat_index]
        return slots.scatter(
            1, slot_index[None, :].expand(points, -1), extracted.T
        ).reshape(stars, layers, ION_STAGE_SLOTS)

    ion_stage_populations = scatter_slots(
        ion_stage_values, tables._flat12, tables._slot_index12
    )
    partition_normalized_populations = scatter_slots(
        normalized_values, tables._flat11, tables._slot_index11
    )

    mass_density = (
        total_nuclei * mean_nuclear_mass_points * ATOMIC_MASS_GRAM_REFERENCE
    )

    # --- Doppler widths (doppler.py:36-62) -----------------------------------
    isotope_mass = tables.major_isotope_mass_amu[: ION_STAGE_SLOTS - 1]
    thermal_velocity_squared = torch.where(
        isotope_mass[None, :] > 0.0,
        2.0 * thermal_energy_erg_raw[:, None]
        / (isotope_mass[None, :] * ATOMIC_MASS_GRAM_REFERENCE),
        torch.full(
            (points, ION_STAGE_SLOTS - 1), torch.inf, dtype=dtype, device=device
        ),
    )
    doppler = (
        torch.sqrt(thermal_velocity_squared + flat(microturbulence)[:, None] ** 2)
        / LIGHT_SPEED_CM_PER_S_EXACT
    )
    fractional_doppler_widths = torch.zeros(
        points, ION_STAGE_SLOTS, dtype=dtype, device=device
    )
    fractional_doppler_widths[:, : ION_STAGE_SLOTS - 1] = doppler

    density_safe = torch.clamp(mass_density[:, None], min=1.0e-300)
    population_over_density_and_width = torch.zeros(
        points, ION_STAGE_SLOTS, dtype=dtype, device=device
    )
    normalized_slots = partition_normalized_populations.reshape(points, ION_STAGE_SLOTS)
    population_over_density_and_width[:, : ION_STAGE_SLOTS - 1] = torch.where(
        doppler > 0.0,
        normalized_slots[:, : ION_STAGE_SLOTS - 1] / (doppler * density_safe),
        torch.zeros_like(doppler),
    )

    def unflat(x):
        return x.reshape(stars, layers)

    return TwinPopulationState(
        electron_density=unflat(electron_density),
        total_nuclei_number_density=unflat(total_nuclei),
        mass_density=unflat(mass_density),
        charge_square_density=unflat(charge_square_density),
        mean_nuclear_mass_amu=unflat(mean_nuclear_mass_points),
        ion_stage_populations_by_packed_slot=ion_stage_populations,
        partition_normalized_populations_by_packed_slot=partition_normalized_populations,
        fractional_doppler_widths=fractional_doppler_widths.reshape(
            stars, layers, ION_STAGE_SLOTS
        ),
        partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
            population_over_density_and_width.reshape(stars, layers, ION_STAGE_SLOTS)
        ),
        converged=unflat(converged),
        iterations_used=unflat(iterations_used),
    )
