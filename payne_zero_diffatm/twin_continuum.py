"""Batched, differentiable twin of the reference continuum-opacity stage.

Reproduces ``compute_continuum_opacity_columns``
(``payne_zero_atmosphere/continuum_opacity.py:5834-5971``) on the solver's
opacity-sampling frequency grid (``build_opacity_sampling_grid``,
``continuum_opacity.py:6157-6197``) as a torch-native module that is batched
over stars and differentiable in its inputs. Every array the reference holds
as ``(layer, frequency)`` is ``(star, layer, frequency)`` here.

Table set. The twin loads the ``atmosphere_tables/`` reference bundle
(``continuum_opacity_tables.npz``, ``karzas_latter_tables.npz``,
``continuum_level_tables.npz``, ``molecular_equilibrium_tables.npz``; loaders
at ``continuum_opacity.py:1005-1084``), not
``synthesis_tables/continuum_tables.npz``. The synthesis torch continuum
(``payne_zero_synthesis/continuum.py``) is a single-star module built on a
*different* table bundle with its own frequency invariants; the reference
solver's ``atmosphere_tables/`` data is the fidelity target for this twin, so
the branch formulas below are transcriptions of the atmosphere module against
its own tables rather than ports of the synthesis module.

Flag coverage. The 20 IFOP flags follow the reference assembly. Verified:
0 (H bf/ff), 1 (H2+), 2 (H- bf/ff), 3 (Rayleigh H), 4 (He I),
5 (He II), 6 (He-), 7 (Rayleigh He), every flag-8 absorber
(CH/OH/H2 collision and C/Mg/Al/Si/Fe I), 9 (lukewarm metals),
10 (hot metals), 11 (electron scattering), and 12 (Rayleigh H2).
Flags 13, 15, 17, 19 are unused by the reference continuum
assembly; 14/16 are line-opacity flags handled by the line stage. Flag 18
(Rosseland-table continuum) is not implemented: it is off in
``DEFAULT_OPACITY_FLAGS`` (``payne_zero_atmosphere/config.py:54``) and the
first-iteration table is empty (``create_rosseland_opacity_table``,
``continuum_opacity.py:5709-5726``), so its contribution on the trace stars
is identically zero; requesting it raises ``NotImplementedError``.

Source function. Exactly like the reference: hydrogen (flag 0) and H- (flag
2) carry non-LTE branch sources, every other absorber is thermal (Planck),
and the assembled source is ``sum(alpha_i S_i) / sum(alpha_i)`` over
absorbers, falling back to the exact Planck function where total absorption
vanishes (``continuum_opacity.py:5962-5964``).

Batching and autograd. All physics is vectorized over ``(star, layer,
frequency)`` in float64 and the module is safe for ``autograd`` in temperature
and the populations. Grid construction and
per-(shell, angular momentum) table-column selection are host-side statics,
exactly as in the reference (piecewise-constant in the inputs). Memory is
~19 MB per star per ``(star, 80, 30000)`` float64 array; ``frequency_chunk``
splits the frequency axis into sequential chunks when a batch would
otherwise hold too many such arrays at once.

Known gaps. Flag 18 above. The hydrogen Gavrila Rayleigh tables in
``continuum_opacity_tables.npz`` are loaded (mirroring the reference loader)
but are dead data: no solver code path reads them. The Karzas-Latter
high-shell branch (shell > 15, ``continuum_opacity.py:1936-1973``) reads one
uninitialized table slot in the reference (``np.empty``); no implemented
branch reaches it with the packaged data, and the twin raises instead of
guessing if it ever would.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from payne_zero_atmosphere.constants import (
    ATOMIC_MASS_GRAM_REFERENCE,
    BOLTZMANN_ERG_PER_K_EXACT,
    BOLTZMANN_ERG_PER_K_REFERENCE,
    BOLTZMANN_EV_PER_K_REFERENCE,
    LIGHT_SPEED_ANGSTROM_PER_S,
    LIGHT_SPEED_CM_PER_S_EXACT,
    LIGHT_SPEED_CM_PER_S_REFERENCE,
    LIGHT_SPEED_NM_PER_S,
    PLANCK_ERG_SECOND_EXACT,
    PLANCK_ERG_SECOND_REFERENCE,
    REFERENCE_NATURAL_LOG_10,
    WAVENUMBER_PER_EV_REFERENCE,
)
from payne_zero_atmosphere.data_files import atmosphere_table_path
from payne_zero_atmosphere import continuum_opacity as _reference_continuum

from .grid_math import remap_to_grid


GRID_SIZE = 30000
# Teff thresholds -> start index, continuum_opacity.py:6164-6178.
_GRID_START_THRESHOLDS = (
    (30000.0, 3577),   # helium_ionized_edge_start
    (13000.0, 7027),   # helium_neutral_edge_start
    (7250.0, 9599),    # lyman_edge_start
    (4500.0, 11601),   # carbon_edge_start
)

_CONTINUUM_OPACITY_TABLE_FILE = "continuum_opacity_tables.npz"
_KARZAS_TABLE_FILE = "karzas_latter_tables.npz"
_LEVEL_TABLE_FILE = "continuum_level_tables.npz"
_MOLECULAR_TABLE_FILE = "molecular_equilibrium_tables.npz"

# Keys mirrored from continuum_opacity.py:885-988.
_CONTINUUM_TABLE_KEYS = (
    "coulomb_freefree_charge_log_offset",
    "hminus_boundfree_wavelength_nm",
    "hminus_boundfree_cross_section_cm2",
    "hminus_freefree_inverse_wavelength_grid",
    "hminus_freefree_theta_grid",
    "hminus_freefree_short_wavelength_table",
    "hminus_freefree_long_wavelength_table",
    "hydrogen_rayleigh_gavrila_main_table",
    "hydrogen_rayleigh_gavrila_ab_table",
    "hydrogen_rayleigh_gavrila_bc_table",
    "hydrogen_rayleigh_gavrila_cd_table",
    "hydrogen_rayleigh_gavrila_lyman_continuum_table",
    "hydrogen_rayleigh_gavrila_lyman_frequency_ratio_grid",
    "coulomb_freefree_gaunt_table",
    "hot_metal_boundfree_transition_table",
    "silicon_singly_ionized_peach_cross_section_table",
    "silicon_singly_ionized_peach_threshold_frequencies_hz",
    "silicon_singly_ionized_peach_natural_log_frequency_grid",
    "silicon_singly_ionized_peach_natural_log_temperature_grid",
    "ch_partition_table",
    "oh_partition_table",
    "ch_cross_section_table",
    "oh_cross_section_table",
    "hydrogen_molecule_h2_collision_table",
    "hydrogen_molecule_he_collision_table",
    "hydrogen_neutral_level_energy_cm",
    "hydrogen_neutral_level_statistical_weight",
)

_KARZAS_TABLE_KEYS = (
    "karzas_latter_log10_frequency_hz",
    "karzas_latter_total_log10_cross_section_cm2",
    "karzas_latter_angular_log10_cross_section_cm2",
    "karzas_latter_high_level_energy_offset_rydberg",
)

_CONTINUUM_LEVEL_TABLE_KEYS = (
    "hydrogen_neutral_level_energy_cm",
    "hydrogen_neutral_level_statistical_weight",
    "helium_neutral_level_energy_cm",
    "helium_neutral_level_statistical_weight",
    "helium_singly_ionized_level_energy_cm",
    "helium_singly_ionized_level_statistical_weight",
    "carbon_neutral_level_energy_cm",
    "carbon_neutral_level_statistical_weight",
    "magnesium_neutral_level_energy_cm",
    "magnesium_neutral_level_statistical_weight",
    "magnesium_singly_ionized_level_energy_cm",
    "magnesium_singly_ionized_level_statistical_weight",
    "aluminum_neutral_level_energy_cm",
    "aluminum_neutral_level_statistical_weight",
    "silicon_neutral_level_energy_cm",
    "silicon_neutral_level_statistical_weight",
    "silicon_singly_ionized_level_energy_cm",
    "silicon_singly_ionized_level_statistical_weight",
    "potassium_neutral_level_energy_cm",
    "potassium_neutral_level_statistical_weight",
    "calcium_neutral_level_energy_cm",
    "calcium_neutral_level_statistical_weight",
    "calcium_singly_ionized_level_energy_cm",
    "calcium_singly_ionized_level_statistical_weight",
    "element_block_offsets",
    "partition_interpolation_scale",
)

_MOLECULAR_TABLE_KEYS = ("h2_partition_function",)


class TwinContinuumTables:
    """One-time loader for the reference continuum tables and sampling grids.

    Every array of the four ``atmosphere_tables/`` archives is held once as a
    torch tensor, grouped into ``opacity`` / ``karzas`` / ``levels`` /
    ``molecular`` namespaces that mirror the reference dataclasses
    (``continuum_opacity.py:761-840``). ``sampling_grid`` reproduces
    ``build_opacity_sampling_grid`` (``continuum_opacity.py:6157-6197``)
    exactly; grids are cached per effective temperature.
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

        def load(filename: str, keys: tuple[str, ...]) -> SimpleNamespace:
            path = (
                Path(table_dir) / filename
                if table_dir is not None
                else atmosphere_table_path(filename)
            )
            with np.load(path, allow_pickle=False) as data:
                return SimpleNamespace(
                    **{
                        key: torch.as_tensor(
                            np.asarray(data[key], dtype=np.float64),
                            dtype=dtype,
                            device=self.device,
                        )
                        for key in keys
                    }
                )

        self.opacity = load(_CONTINUUM_OPACITY_TABLE_FILE, _CONTINUUM_TABLE_KEYS)
        self.karzas = load(_KARZAS_TABLE_FILE, _KARZAS_TABLE_KEYS)
        self.levels = load(_LEVEL_TABLE_FILE, _CONTINUUM_LEVEL_TABLE_KEYS)
        self.molecular = load(_MOLECULAR_TABLE_FILE, _MOLECULAR_TABLE_KEYS)
        helium_names = (
            "_HELIUM_NEUTRAL_STATISTICAL_WEIGHTS",
            "_HELIUM_NEUTRAL_EXCITATION_EV",
            "_HELIUM_NEUTRAL_THRESHOLD_FREQUENCY_HZ",
            "_HELIUM_GROUND_CROSS_SECTION_50_505",
            "_HELIUM_GROUND_CROSS_SECTION_20_50",
            "_HELIUM_GROUND_CROSS_SECTION_10_20",
            "_HELIUM_GROUND_CROSS_SECTION_0_10",
            "_HELIUM_1S2S_SINGLET_LOG_FREQUENCY",
            "_HELIUM_1S2S_SINGLET_LOG_CROSS_SECTION",
            "_HELIUM_1S2S_TRIPLET_LOG_FREQUENCY",
            "_HELIUM_1S2S_TRIPLET_LOG_CROSS_SECTION",
            "_HELIUM_1S2P_SINGLET_LOG_FREQUENCY",
            "_HELIUM_1S2P_SINGLET_LOG_CROSS_SECTION",
            "_HELIUM_1S2P_TRIPLET_LOG_FREQUENCY",
            "_HELIUM_1S2P_TRIPLET_LOG_CROSS_SECTION",
        )
        self.helium = SimpleNamespace(
            **{
                "he_" + name.removeprefix("_HELIUM_").lower(): torch.as_tensor(
                    getattr(_reference_continuum, name),
                    dtype=dtype,
                    device=self.device,
                )
                for name in helium_names
            }
        )
        self.lukewarm_silicon_table = torch.as_tensor(
            _reference_continuum._build_silicon_singly_ionized_lukewarm_table(),
            dtype=dtype,
            device=self.device,
        )
        self.hot_metal_transitions = tuple(
            tuple(float(value) for value in row)
            for row in self.opacity.hot_metal_boundfree_transition_table.detach().cpu()
        )
        self._grid_cache: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def default(cls, **kwargs) -> "TwinContinuumTables":
        return cls(**kwargs)

    def sampling_grid(
        self, effective_temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Opacity-sampling wavelengths (nm) and frequency quadrature weights.

        Exact transcription of ``build_opacity_sampling_grid``
        (``continuum_opacity.py:6157-6197``): a 30000-point log-lambda grid
        ``lambda_nm = 10**(1 + 1e-4 * (i + start_index - 1))`` with the
        Teff-dependent start index, trapezoidal frequency weights with the
        1.5x / 0.25x endpoint rules. Grid construction is host-side (the
        grid is not a differentiated quantity).
        """

        key = float(effective_temperature)
        if key not in self._grid_cache:
            start_index = 1
            for threshold, start in _GRID_START_THRESHOLDS:
                if key < threshold:
                    start_index = start
            one_based_index = torch.arange(
                1, GRID_SIZE + 1, dtype=self.dtype, device=self.device
            )
            wavelength_nm = 10.0 ** (
                1.0 + 0.0001 * (one_based_index + start_index - 1.0)
            )
            frequency = LIGHT_SPEED_NM_PER_S / wavelength_nm
            frequency_weights = torch.zeros(
                GRID_SIZE, dtype=self.dtype, device=self.device
            )
            frequency_weights[0] = (frequency[0] - frequency[1]) * 1.5
            frequency_weights[1:-1] = (frequency[:-2] - frequency[2:]) * 0.5
            frequency_weights[-1] = (frequency[-2] + frequency[-1]) * 0.25
            self._grid_cache[key] = (wavelength_nm, frequency_weights)
        return self._grid_cache[key]

    def sampling_frequency_hz(self, effective_temperature: float) -> torch.Tensor:
        """Frequency grid the solver feeds to the continuum assembly.

        Mirrors ``prepare_opacity_state`` (``payne_zero_atmosphere/runner.py:936-942``):
        ``2.99792458e17 / max(lambda_nm, 1e-300)``.
        """

        wavelength_nm, _ = self.sampling_grid(effective_temperature)
        return LIGHT_SPEED_NM_PER_S / torch.clamp(wavelength_nm, min=1.0e-300)


class TwinContinuumState:
    """The continuum input bundle: one batch of atmosphere population states.

    This is the batched analog of ``ContinuumAtmosphereState``
    (``continuum_opacity.py:857-882``), built by the same packing rule as
    ``build_continuum_atmosphere_state`` (``continuum_opacity.py:1087-1182``).
    Per-layer fields are ``(star, layer)`` float64 tensors; the packed tubes
    are ``(star, layer, 1006)`` following the reference packed slot layout
    (``population_layout.py``; also ``twin_eos.py``). The bundle is explicit:
    the two packed tubes plus temperature, mass density, electron density,
    gas pressure, per-layer elemental abundances ``(star, layer, 99)``, and
    hydrogen departure coefficients ``(star, layer, 6)`` (ones under LTE).
    Every per-species population the branches read is a slot of the packed
    tubes (H/He stages at slots 0-3, CH at 845, OH at 847, metals through
    their element blocks), exactly like the reference state.
    """

    def __init__(
        self,
        *,
        temperature: torch.Tensor,
        mass_density: torch.Tensor,
        electron_density: torch.Tensor,
        gas_pressure: torch.Tensor,
        elemental_abundances_by_layer: torch.Tensor,
        ion_stage_populations_by_packed_slot: torch.Tensor,
        partition_normalized_populations_by_packed_slot: torch.Tensor,
        hydrogen_departure_coefficients: torch.Tensor | None = None,
        tables: TwinContinuumTables,
    ):
        dtype = tables.dtype
        device = tables.device

        def tensor(values):
            # torch.as_tensor (not np.asarray first) so autograd graphs on
            # tensor inputs survive the conversion.
            return torch.as_tensor(values, dtype=dtype, device=device)

        temperature = tensor(temperature)
        if temperature.dim() != 2:
            raise ValueError("temperature must be (star, layer)")
        stars, layers = temperature.shape
        ion_tube = tensor(ion_stage_populations_by_packed_slot)
        partition_tube = tensor(partition_normalized_populations_by_packed_slot)
        if ion_tube.shape[:2] != (stars, layers) or ion_tube.shape[2] < 2:
            raise ValueError(
                "ion_stage_populations_by_packed_slot must be (star, layer, >=2)"
            )
        if partition_tube.shape[:2] != (stars, layers) or partition_tube.shape[2] <= 847:
            raise ValueError(
                "partition_normalized_populations_by_packed_slot must be "
                "(star, layer, >847) to include the CH and OH continuum slots"
            )
        if hydrogen_departure_coefficients is None:
            # build_continuum_atmosphere_state:1119-1121.
            departure = torch.ones(
                stars, layers, 6, dtype=dtype, device=device
            )
        else:
            departure = tensor(hydrogen_departure_coefficients)
            if departure.shape[0] != stars or departure.shape[1] != layers:
                raise ValueError(
                    "hydrogen_departure_coefficients must match (star, layer)"
                )
            if departure.dim() == 2:
                departure = departure[:, :, None]
            if departure.shape[2] < 6:
                # compute_hydrogen_opacity_columns:2908-2912.
                padding = torch.ones(
                    stars, layers, 6 - departure.shape[2], dtype=dtype, device=device
                )
                departure = torch.cat([departure, padding], dim=2)

        self.temperature = temperature
        self.mass_density = tensor(mass_density)
        self.electron_density = tensor(electron_density)
        self.gas_pressure = tensor(gas_pressure)
        self.elemental_abundances_by_layer = tensor(elemental_abundances_by_layer)
        self.ion_stage_populations_by_packed_slot = ion_tube
        self.partition_normalized_populations_by_packed_slot = partition_tube
        self.hydrogen_departure_coefficients = departure
        self.stars = stars
        self.layers = layers

    @classmethod
    def from_packed(cls, *, tables: TwinContinuumTables, **fields):
        """Build from trace-style arrays (see class docstring)."""

        return cls(tables=tables, **fields)

    # --- packed-slot views (continuum_opacity.py:1131-1181) -----------------
    @property
    def hydrogen_partition_normalized_neutral(self) -> torch.Tensor:
        return self.partition_normalized_populations_by_packed_slot[..., 0]

    @property
    def hydrogen_neutral_population(self) -> torch.Tensor:
        return self.ion_stage_populations_by_packed_slot[..., 0]

    @property
    def hydrogen_ionized_population(self) -> torch.Tensor:
        return self.ion_stage_populations_by_packed_slot[..., 1]

    @property
    def helium_neutral_population(self) -> torch.Tensor:
        return self.ion_stage_populations_by_packed_slot[..., 2]

    @property
    def helium_singly_ionized_population(self) -> torch.Tensor:
        return self.ion_stage_populations_by_packed_slot[..., 3]

    @property
    def helium_neutral_partition_normalized_population(self) -> torch.Tensor:
        return self.partition_normalized_populations_by_packed_slot[..., 2]

    @property
    def helium_singly_ionized_partition_normalized_population(self) -> torch.Tensor:
        return self.partition_normalized_populations_by_packed_slot[..., 3]

    @property
    def ch_population(self) -> torch.Tensor:
        return self.partition_normalized_populations_by_packed_slot[..., 845]

    @property
    def oh_population(self) -> torch.Tensor:
        return self.partition_normalized_populations_by_packed_slot[..., 847]


# --- shared helpers ---------------------------------------------------------


def _planck_frequency_exact(
    temperature: torch.Tensor, frequency_hz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched ``_planck_frequency_exact`` (``continuum_opacity.py:1447-1488``).

    ``temperature`` is ``(star, layer)``, ``frequency_hz`` ``(freq,)``; all
    three outputs are ``(star, layer, freq)``: exact Planck B_nu,
    ``exp(-h nu / kT)``, and the stimulated-emission factor ``1 - exp``.
    The Rayleigh-Jeans branch below ``h nu / kT = 1e-6`` and the non-finite
    floor are reproduced with ``torch.where``.
    """

    temperature = torch.as_tensor(temperature)
    frequency_hz = torch.as_tensor(frequency_hz)
    hnu_over_kt = (
        PLANCK_ERG_SECOND_EXACT
        * frequency_hz[None, None, :]
        / torch.clamp(
            BOLTZMANN_ERG_PER_K_EXACT * temperature[:, :, None], min=1.0e-300
        )
    )
    exponential = torch.exp(-hnu_over_kt)
    stimulated_emission = 1.0 - exponential

    rayleigh_jeans = hnu_over_kt < 1.0e-6
    planck_rayleigh = (
        2.0
        * BOLTZMANN_ERG_PER_K_EXACT
        * temperature[:, :, None]
        * frequency_hz[None, None, :] ** 2
        / LIGHT_SPEED_CM_PER_S_EXACT**2
    )
    planck_full = (
        2.0
        * PLANCK_ERG_SECOND_EXACT
        / LIGHT_SPEED_CM_PER_S_EXACT**2
        * frequency_hz[None, None, :] ** 3
        / torch.expm1(hnu_over_kt)
    )
    planck = torch.where(rayleigh_jeans, planck_rayleigh, planck_full)
    planck = torch.where(torch.isfinite(planck), planck, torch.zeros_like(planck))
    return planck, exponential, stimulated_emission


def _linter(
    x_table: torch.Tensor, y_table: torch.Tensor, x_new: torch.Tensor
) -> torch.Tensor:
    """Batched ATLAS LINTER (``_linter_point``, ``continuum_opacity.py:1599-1611``).

    Linear interpolation with endpoint extrapolation on an ascending table.
    The reference bracket walk from index 1 capped at ``n-1`` is exactly
    ``clamp(searchsorted(x_table, value, right=True), 1, n-1)``; the
    near-degenerate-denominator guard returns the left table value.
    """

    count = x_table.numel()
    index = torch.clamp(
        torch.searchsorted(x_table, x_new.contiguous(), right=True), 1, count - 1
    )
    x_left = x_table[index - 1]
    x_right = x_table[index]
    y_left = y_table[index - 1]
    y_right = y_table[index]
    denominator = x_right - x_left
    weight = (x_new - x_left) / denominator
    interpolated = y_left + (y_right - y_left) * weight
    return torch.where(torch.abs(denominator) < 1.0e-40, y_left, interpolated)


def _coulomb_freefree_gaunt(
    ion_charge: int,
    natural_log_frequency: torch.Tensor,
    natural_log_temperature: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Batched COULFF bilinear grid (``continuum_opacity.py:1557-1597``).

    ``ion_charge`` is a host-side integer exactly as in the reference (charges
    outside 1..6 return ones). ``natural_log_frequency`` is ``(freq,)`` and
    ``natural_log_temperature`` ``(star, layer)``; the output is
    ``(star, layer, freq)``. Integer bin indices truncate toward zero like
    the numba ``int()`` casts.
    """

    charge = int(ion_charge)
    frequency_log = torch.as_tensor(natural_log_frequency)
    temperature_log = torch.as_tensor(natural_log_temperature)
    shape = temperature_log.shape + frequency_log.shape
    if charge < 1 or charge > 6:
        return torch.ones(
            shape, dtype=temperature_log.dtype, device=temperature_log.device
        )

    z4log = float(tables.opacity.coulomb_freefree_charge_log_offset[charge - 1])
    gaunt_table = tables.opacity.coulomb_freefree_gaunt_table
    gamlog = 10.39638 - temperature_log[:, :, None] / 1.15129 + z4log
    hvktlg = (frequency_log[None, None, :] - temperature_log[:, :, None]) / 1.15129 - 20.63764
    igam = torch.clamp((gamlog + 7.0).to(torch.int64), 1, 10)
    ihvkt = torch.clamp((hvktlg + 9.0).to(torch.int64), 1, 11)
    p_weight = gamlog - (igam - 7.0).to(gamlog.dtype)
    q_weight = hvktlg - (ihvkt - 9.0).to(hvktlg.dtype)

    ig = igam - 1
    ih = ihvkt - 1
    a00 = gaunt_table[ih, ig]
    a01 = gaunt_table[ih + 1, ig]
    a10 = gaunt_table[ih, ig + 1]
    a11 = gaunt_table[ih + 1, ig + 1]
    return (1.0 - p_weight) * ((1.0 - q_weight) * a00 + q_weight * a01) + p_weight * (
        (1.0 - q_weight) * a10 + q_weight * a11
    )


def _karzas_latter_cross_section_grid(
    frequency_hz: torch.Tensor,
    *,
    effective_charge_squared: float,
    principal_quantum_number: int,
    orbital_angular_momentum: int,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Batched Karzas-Latter bf grid (``_karzas_latter_point_compiled``, :1872-1973).

    ``(charge_squared, shell, angular momentum)`` are host-side statics per
    call, as in the reference; ``frequency_hz`` is ``(freq,)`` and the result
    is ``(freq,)``. The table columns are *descending* in log10 frequency, so
    the reference's first-index-below bracket is
    ``searchsorted(-column, -log10_frequency, right=True)``; the bracket-0
    case wraps to the endpoint pair exactly like the reference's ``[-1]``
    indexing, and the above-table case returns the last table value.

    The high-shell branch (shell > 15) is not reachable from any implemented
    continuum branch with the packaged tables and raises, rather than
    reproducing the reference's uninitialized ``high_shell_frequency[0]``
    read (module docstring).
    """

    charge_squared = float(effective_charge_squared)
    shell = int(principal_quantum_number)
    angular_momentum = max(0, int(orbital_angular_momentum))
    frequency = torch.as_tensor(frequency_hz)
    result = torch.zeros_like(frequency)
    if charge_squared <= 0.0 or shell <= 0:
        return result
    if shell > 15:
        raise NotImplementedError(
            "Karzas-Latter shell > 15 is unreachable with the packaged tables"
        )

    frequency_column = tables.karzas.karzas_latter_log10_frequency_hz[:, shell - 1]
    if angular_momentum >= shell or shell > 6:
        value_column = tables.karzas.karzas_latter_total_log10_cross_section_cm2[
            :, shell - 1
        ]
    else:
        value_column = tables.karzas.karzas_latter_angular_log10_cross_section_cm2[
            angular_momentum, shell - 1, :
        ]
        if bool(torch.isnan(value_column[0])):
            return result

    log10_frequency = torch.log10(frequency / charge_squared)
    below_table = log10_frequency < frequency_column[-1]
    count = frequency_column.numel()
    bracket = torch.searchsorted(
        -frequency_column, (-log10_frequency).contiguous(), right=True
    )
    index_lo = torch.remainder(bracket - 1, count)
    index_hi = torch.clamp(bracket, max=count - 1)
    x_lo = frequency_column[index_lo]
    x_hi = frequency_column[index_hi]
    y_lo = value_column[index_lo]
    y_hi = value_column[index_hi]
    denominator = x_lo - x_hi
    weight = (log10_frequency - x_hi) / denominator
    log10_cross_section = (y_lo - y_hi) * weight + y_hi
    cross_section = torch.where(
        torch.abs(denominator) < 1.0e-15,
        10.0**y_lo,
        10.0**log10_cross_section,
    )
    above_table = bracket >= count
    cross_section = torch.where(
        above_table, 10.0 ** value_column[-1], cross_section
    )
    cross_section = cross_section / charge_squared
    cross_section = torch.where(
        (frequency > 0.0) & ~below_table, cross_section, torch.zeros_like(cross_section)
    )
    return cross_section


def _h2_equilibrium_constant(
    temperature: torch.Tensor, tables: TwinContinuumTables
) -> torch.Tensor:
    """Batched ``_h2_equilibrium_constant`` (``continuum_opacity.py:1297-1345``)."""

    safe_temperature = torch.where(
        torch.isfinite(temperature) & (temperature > 100.0),
        temperature,
        torch.full_like(temperature, 100.0),
    )
    safe_temperature = torch.minimum(
        safe_temperature, torch.full_like(safe_temperature, 19900.0)
    )
    table_index = torch.clamp(torch.floor(safe_temperature / 100.0).to(torch.int64), 1, 199)
    partition_table = tables.molecular.h2_partition_function
    partition = partition_table[table_index - 1] + (
        partition_table[table_index] - partition_table[table_index - 1]
    ) * (safe_temperature - table_index.to(safe_temperature.dtype) * 100.0) / 100.0
    denominator = (
        2.0
        * 3.14159
        * 1.008
        * ATOMIC_MASS_GRAM_REFERENCE
        * BOLTZMANN_ERG_PER_K_REFERENCE
        / (PLANCK_ERG_SECOND_REFERENCE**2)
        * safe_temperature
    ) ** 1.5
    equilibrium = (
        partition
        * (2.0**1.5)
        / 4.0
        / torch.clamp(denominator, min=1.0e-300)
        * torch.exp(
            36118.11
            * PLANCK_ERG_SECOND_REFERENCE
            * LIGHT_SPEED_CM_PER_S_REFERENCE
            / BOLTZMANN_ERG_PER_K_REFERENCE
            / safe_temperature
        )
    )
    return torch.where(torch.isfinite(equilibrium), equilibrium, torch.zeros_like(equilibrium))


def _molecular_hydrogen_population(
    temperature: torch.Tensor,
    hydrogen_neutral_partition_normalized_population: torch.Tensor,
    hydrogen_departure_coefficient: torch.Tensor | None,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Batched ``compute_molecular_hydrogen_population`` (:1348-1368)."""

    ground_population = hydrogen_neutral_partition_normalized_population
    if hydrogen_departure_coefficient is None:
        departure = torch.ones_like(temperature)
    else:
        departure = hydrogen_departure_coefficient
    return (ground_population * 2.0 * departure) ** 2 * _h2_equilibrium_constant(
        temperature, tables
    )


def _hydrogen_neutral_partition_normalized_population_from_neutral(
    temperature: torch.Tensor,
    hydrogen_neutral_population: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Batched ``_hydrogen_neutral_partition_normalized_population_from_neutral`` (:1371-1399)."""

    thermal_energy_ev = BOLTZMANN_EV_PER_K_REFERENCE * temperature
    energy_ev = (
        tables.opacity.hydrogen_neutral_level_energy_cm / WAVENUMBER_PER_EV_REFERENCE
    )
    weight = tables.opacity.hydrogen_neutral_level_statistical_weight
    boltzmann_factor = torch.exp(
        -energy_ev[:, None, None] / thermal_energy_ev[None, :, :]
    )
    boltzmann_factor = torch.where(
        torch.isfinite(boltzmann_factor),
        boltzmann_factor,
        torch.zeros_like(boltzmann_factor),
    )
    partition = (weight[:, None, None] * boltzmann_factor).sum(0)
    return hydrogen_neutral_population / torch.clamp(partition, min=1.0e-300)


# --- absorption branches ------------------------------------------------------
#
# Every branch takes ``(state, frequency_hz, tables)`` with ``frequency_hz``
# ``(freq,)`` and returns absorption ``(star, layer, freq)``; the two non-LTE
# branches (hydrogen, H-) also return their source. Branch names and the
# flag wiring match ``compute_continuum_opacity_columns``
# (``continuum_opacity.py:5834-5971``).


def _hydrogen_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flag 0: hydrogen bf/ff absorption and non-LTE source (:2868-3003)."""

    temperature = state.temperature
    mass_density = torch.clamp(state.mass_density, min=1.0e-300)
    electron_density = state.electron_density
    hydrogen_neutral_partition_normalized = state.hydrogen_partition_normalized_neutral
    hydrogen_ionized_population = state.hydrogen_ionized_population
    hydrogen_departure = state.hydrogen_departure_coefficients

    planck_nu, exp_hnu_over_kt, stimulated_emission = _planck_frequency_exact(
        temperature, frequency_hz
    )
    thermal_energy_ev = torch.clamp(
        temperature * BOLTZMANN_EV_PER_K_REFERENCE, min=1.0e-300
    )
    shell_number = torch.arange(
        1, 9, dtype=temperature.dtype, device=temperature.device
    )  # (8,)
    boltzmann_population = (
        torch.exp(
            -(13.595 - 13.595 / (shell_number * shell_number))[None, None, :]
            / thermal_energy_ev[:, :, None]
        )
        * 2.0
        * (shell_number * shell_number)[None, None, :]
        * hydrogen_neutral_partition_normalized[:, :, None]
        / mass_density[:, :, None]
    )
    departure_factor = torch.cat(
        [hydrogen_departure, torch.ones_like(hydrogen_departure[:, :, :2])], dim=2
    )  # shells 7-8 carry no departure factor (:2928-2929)
    boltzmann_population = boltzmann_population * departure_factor

    freefree_density_factor = (
        electron_density
        * hydrogen_ionized_population
        / mass_density
        / torch.sqrt(torch.clamp(temperature, min=1.0e-300))
    )
    xr = (
        hydrogen_neutral_partition_normalized
        * (thermal_energy_ev / 13.595)
        / mass_density
    )
    boltzmann_extension = torch.exp(-13.427 / thermal_energy_ev) * xr
    series_limit_extension = torch.exp(-13.595 / thermal_energy_ev) * xr
    coulomb_freefree = _coulomb_freefree_gaunt(
        1,
        torch.log(torch.clamp(frequency_hz, min=1.0e-300)),
        torch.log(torch.clamp(temperature, min=1.0e-300)),
        tables,
    )
    karzas_cross_sections = torch.stack(
        [
            _karzas_latter_cross_section_grid(
                frequency_hz,
                effective_charge_squared=1.0,
                principal_quantum_number=shell,
                orbital_angular_momentum=shell,
                tables=tables,
            )
            for shell in range(1, 9)
        ]
    )  # (8, freq)

    frequency_cubed = torch.clamp(frequency_hz**3, min=1.0e-300)
    freefree_coefficient = 3.6919e8 / frequency_cubed
    extension_coefficient = 2.815e29 / frequency_cubed
    extension_population = torch.where(
        frequency_hz[None, None, :] < 4.05933e13,
        series_limit_extension[:, :, None]
        / torch.clamp(exp_hnu_over_kt, min=1.0e-300),
        boltzmann_extension[:, :, None].expand_as(exp_hnu_over_kt),
    )

    absorption = (
        karzas_cross_sections[6][None, None, :] * boltzmann_population[:, :, 6:7]
        + karzas_cross_sections[7][None, None, :] * boltzmann_population[:, :, 7:8]
        + (extension_population - series_limit_extension[:, :, None])
        * extension_coefficient[None, None, :]
        + coulomb_freefree
        * freefree_density_factor[:, :, None]
        * freefree_coefficient[None, None, :]
    ) * stimulated_emission
    source_numerator = absorption * planck_nu

    for shell_index in range(6):
        departure = torch.clamp(
            hydrogen_departure[:, :, shell_index], min=1.0e-300
        )
        boundfree_term = (
            karzas_cross_sections[shell_index][None, None, :]
            * boltzmann_population[:, :, shell_index : shell_index + 1]
        )
        absorption = absorption + boundfree_term * (
            1.0 - exp_hnu_over_kt / departure[:, :, None]
        )
        source_numerator = source_numerator + (
            boundfree_term
            * planck_nu
            * stimulated_emission
            / departure[:, :, None]
        )

    source = torch.where(
        absorption > 0.0,
        source_numerator / torch.clamp(absorption, min=1.0e-300),
        planck_nu,
    )
    return absorption, source


# BEGIN BRANCH: hminus (flag 2)
def _hminus_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flag 2: H- bf/ff absorption and non-LTE source (:3125-3291)."""

    temperature = state.temperature
    mass_density = torch.clamp(state.mass_density, min=1.0e-300)
    electron_density = state.electron_density
    hydrogen_partition = state.hydrogen_partition_normalized_neutral
    hydrogen_departure = state.hydrogen_departure_coefficients[:, :, 0]
    hminus_departure = torch.ones_like(temperature)
    planck, exponential, stimulated = _planck_frequency_exact(
        temperature, frequency_hz
    )
    thermal_energy_ev = temperature * BOLTZMANN_EV_PER_K_REFERENCE
    hminus_population = (
        torch.exp(0.754209 / torch.clamp(thermal_energy_ev, min=1.0e-300))
        / (
            2.0
            * 2.4148e15
            * temperature
            * torch.sqrt(torch.clamp(temperature, min=1.0e-300))
        )
        * hminus_departure
        * hydrogen_departure
        * hydrogen_partition
        * electron_density
    )

    opacity = tables.opacity
    theta = 5040.0 / temperature
    inverse_wavelength = opacity.hminus_freefree_inverse_wavelength_grid
    theta_grid = opacity.hminus_freefree_theta_grid
    wavelength_log_grid = torch.log(91.134 / inverse_wavelength)
    short = opacity.hminus_freefree_short_wavelength_table
    long = opacity.hminus_freefree_long_wavelength_table
    freefree_table = torch.cat([short, long], dim=0).T
    freefree_log = torch.log(
        freefree_table / theta_grid[:, None]
        * 5040.0
        * BOLTZMANN_ERG_PER_K_EXACT
    ).T
    wavelength_nm = LIGHT_SPEED_NM_PER_S / torch.clamp(
        frequency_hz, min=1.0e-30
    )
    wavelength_log = torch.log(wavelength_nm)
    freefree_by_theta = torch.stack(
        [
            torch.exp(_linter(wavelength_log_grid, row, wavelength_log))
            for row in freefree_log.T
        ],
        dim=0,
    )
    theta_index = torch.clamp(
        torch.searchsorted(theta_grid, theta.contiguous(), right=True),
        1,
        theta_grid.numel() - 1,
    )
    theta_left = theta_grid[theta_index - 1]
    theta_right = theta_grid[theta_index]
    weight = (theta - theta_left) / (theta_right - theta_left)
    left = freefree_by_theta[theta_index - 1]
    right = freefree_by_theta[theta_index]
    freefree_theta = left + (right - left) * weight[:, :, None]
    freefree_theta = torch.where(
        ((theta_right - theta_left).abs() < 1.0e-40)[:, :, None],
        left,
        freefree_theta,
    )

    boundfree_cross_section = torch.zeros_like(frequency_hz)
    active = frequency_hz > 1.82365e14
    remapped = remap_to_grid(
        opacity.hminus_boundfree_wavelength_nm,
        opacity.hminus_boundfree_cross_section_cm2,
        wavelength_nm,
    )
    boundfree_cross_section = torch.where(active, remapped, boundfree_cross_section)
    freefree_absorption = freefree_theta * (
        hydrogen_partition
        * 2.0
        * hydrogen_departure
        * electron_density
        / mass_density
        * 1.0e-26
    )[:, :, None]
    boundfree_absorption = (
        boundfree_cross_section[None, None, :]
        * 1.0e-18
        * (1.0 - exponential / hminus_departure[:, :, None].clamp(min=1.0e-40))
        * hminus_population[:, :, None]
        / mass_density[:, :, None]
    )
    absorption = boundfree_absorption + freefree_absorption
    source_denominator = hminus_departure[:, :, None] - exponential
    boundfree_source = (
        boundfree_absorption
        * planck
        * stimulated
        / source_denominator.clamp(min=1.0e-40)
    )
    source = torch.where(
        absorption > 0.0,
        (boundfree_source + freefree_absorption * planck)
        / absorption.clamp(min=1.0e-300),
        planck,
    )
    return absorption, source
# END BRANCH: hminus


# BEGIN BRANCH: molecular_hydrogen_ion (flag 1)
def _molecular_hydrogen_ion_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 1: H2+ absorption (:3291-3376)."""

    temperature = state.temperature
    mass_density = state.mass_density.clamp(min=1.0e-300)
    hydrogen_levels = state.partition_normalized_populations_by_packed_slot[..., :2]
    hydrogen_departure = state.hydrogen_departure_coefficients[:, :, 0]
    _, _, stimulated = _planck_frequency_exact(temperature, frequency_hz)
    absorption = torch.zeros(
        state.stars, state.layers, frequency_hz.numel(),
        dtype=temperature.dtype, device=temperature.device,
    )
    active = frequency_hz <= 3.28805e15
    frequency = frequency_hz[active]
    log_frequency = torch.log(frequency)
    frequency_1e15 = frequency / 1.0e15
    fr = -3.0233e3 + (
        3.7797e2 + (
            -1.82496e1 + (3.9207e-1 - 3.1672e-3 * log_frequency)
            * log_frequency
        ) * log_frequency
    ) * log_frequency
    excitation = -7.342e-3 + (
        -2.409 + (
            1.028 + (-4.230e-1 + (1.224e-1 - 1.351e-2 * frequency_1e15)
                     * frequency_1e15) * frequency_1e15
        ) * frequency_1e15
    ) * frequency_1e15
    thermal_energy_ev = (temperature * BOLTZMANN_EV_PER_K_REFERENCE).clamp(
        min=1.0e-300
    )
    ground = hydrogen_levels[:, :, 0].clamp(min=1.0e-40)
    excited = hydrogen_levels[:, :, 1]
    value = (
        torch.exp(
            -excitation[None, None, :] / thermal_energy_ev[:, :, None]
            + fr[None, None, :]
            + torch.log(ground)[:, :, None]
        )
        * 2.0
        * hydrogen_departure[:, :, None]
        * excited[:, :, None]
        / mass_density[:, :, None]
        * stimulated[:, :, active]
    )
    absorption[:, :, active] = value
    return absorption
# END BRANCH: molecular_hydrogen_ion


# BEGIN BRANCH: helium_neutral (flag 4)
def _helium_ground_cross_section_grid(
    frequency_hz: torch.Tensor, tables: TwinContinuumTables
) -> torch.Tensor:
    """Vectorized He I ground-state table interpolation (:2549-2600)."""

    frequency = torch.as_tensor(frequency_hz)
    wavelength = LIGHT_SPEED_ANGSTROM_PER_S / frequency
    helium = tables.helium
    result = torch.zeros_like(frequency)

    def interpolate(
        mask: torch.Tensor,
        raw_index: torch.Tensor,
        table: torch.Tensor,
        origin: float,
        step: float,
        top_index: int,
    ) -> None:
        index = raw_index.to(torch.int64).clamp(2, top_index)
        value = (
            (wavelength - (top_index - index).to(wavelength.dtype) * step - origin)
            / step
            * (table[index - 2] - table[index - 1])
            + table[index - 1]
        ) * 1.0e-18
        result[mask] = value[mask]

    mask_50 = (frequency >= 5.945209e15) & (wavelength > 50.0)
    interpolate(
        mask_50, 93.0 - (wavelength - 50.0) / 5.0,
        helium.he_ground_cross_section_50_505, 50.0, 5.0, 92,
    )
    mask_20 = (frequency >= 5.945209e15) & (wavelength <= 50.0) & (wavelength > 20.0)
    interpolate(
        mask_20, 17.0 - (wavelength - 20.0) / 2.0,
        helium.he_ground_cross_section_20_50, 20.0, 2.0, 16,
    )
    mask_10 = (frequency >= 5.945209e15) & (wavelength <= 20.0) & (wavelength > 10.0)
    interpolate(
        mask_10, 12.0 - (wavelength - 10.0),
        helium.he_ground_cross_section_10_20, 10.0, 1.0, 11,
    )
    mask_0 = (frequency >= 5.945209e15) & (wavelength <= 10.0)
    interpolate(
        mask_0, 22.0 - wavelength / 0.5,
        helium.he_ground_cross_section_0_10, 0.0, 0.5, 21,
    )
    return result


def _helium_tabulated_cross_section_grid(
    frequency_hz: torch.Tensor,
    *,
    threshold_wavenumber_cm: float,
    log_frequency_table: torch.Tensor,
    log_cross_section_table: torch.Tensor,
    high_frequency_kind: str | None,
) -> torch.Tensor:
    """Vectorized excited-state He I table interpolation (:2603-2666)."""

    frequency = torch.as_tensor(frequency_hz)
    log_frequency = torch.log10(frequency)
    index = torch.searchsorted(
        -log_frequency_table, (-log_frequency).contiguous(), right=True
    ).clamp(1, 15)
    interpolated = (
        (log_frequency - log_frequency_table[index])
        / (log_frequency_table[index - 1] - log_frequency_table[index])
        * (log_cross_section_table[index - 1] - log_cross_section_table[index])
        + log_cross_section_table[index]
    )
    result = 10.0**interpolated
    if high_frequency_kind is not None:
        wavenumber = frequency / LIGHT_SPEED_CM_PER_S_EXACT
        high = frequency > 2.4 * 109722.267 * LIGHT_SPEED_CM_PER_S_EXACT
        if high_frequency_kind == "1s2s_singlet":
            kinetic = (wavenumber - 32033.214) / 109722.267
            epsilon = 2.0 * (kinetic - 2.612316) / 0.00322
            formula = (
                0.008175 * (484940.0 / wavenumber) ** 2.71 * 8.067e-18
                * (epsilon + 76.21) ** 2 / (1.0 + epsilon**2)
            )
        elif high_frequency_kind == "1s2s_triplet":
            kinetic = (wavenumber - 38454.691) / 109722.267
            epsilon = 2.0 * (kinetic - 2.47898) / 0.000780
            formula = (
                0.01521 * (470310.0 / wavenumber) ** 3.12 * 8.067e-18
                * (epsilon - 122.4) ** 2 / (1.0 + epsilon**2)
            )
        else:
            kinetic = (wavenumber - 27175.76) / 109722.267
            epsilon_s = 2.0 * (kinetic - 2.446534) / 0.01037
            epsilon_d = 2.0 * (kinetic - 2.59427) / 0.00538
            formula = (
                0.0009487 * (466750.0 / wavenumber) ** 3.69 * 8.067e-18
                * ((epsilon_s - 29.30) ** 2 / (1.0 + epsilon_s**2)
                   + (epsilon_d + 172.4) ** 2 / (1.0 + epsilon_d**2))
            )
        result = torch.where(high, formula, result)
    threshold = threshold_wavenumber_cm * LIGHT_SPEED_CM_PER_S_EXACT
    return torch.where(frequency >= threshold, result, torch.zeros_like(result))


def _helium_neutral_transition_grid(
    frequency_hz: torch.Tensor, tables: TwinContinuumTables
) -> tuple[torch.Tensor, torch.Tensor]:
    """He I low-level and high-n cross-section grids (:2669-2820)."""

    frequency = torch.as_tensor(frequency_hz)
    helium = tables.helium
    low = torch.zeros(
        10, frequency.numel(), dtype=frequency.dtype, device=frequency.device
    )
    low[0] = _helium_ground_cross_section_grid(frequency, tables)
    specifications = (
        (1, 38454.691,
         helium.he_1s2s_triplet_log_frequency, helium.he_1s2s_triplet_log_cross_section,
         "1s2s_triplet"),
        (2, 32033.214,
         helium.he_1s2s_singlet_log_frequency, helium.he_1s2s_singlet_log_cross_section,
         "1s2s_singlet"),
        (3, 29223.753,
         helium.he_1s2p_triplet_log_frequency, helium.he_1s2p_triplet_log_cross_section,
         None),
        (4, 27175.76,
         helium.he_1s2p_singlet_log_frequency, helium.he_1s2p_singlet_log_cross_section,
         "1s2p_singlet"),
    )
    for level, threshold_cm, log_f, log_cross, kind in specifications:
        low[level] = _helium_tabulated_cross_section_grid(
            frequency,
            threshold_wavenumber_cm=threshold_cm,
            log_frequency_table=log_f,
            log_cross_section_table=log_cross,
            high_frequency_kind=kind,
        )

    fixed = (
        (5, 1.236439, 3, 0), (6, 1.102898, 3, 0),
        (7, 1.045499, 3, 1), (8, 1.001427, 3, 2), (9, 0.9926, 3, 1),
    )
    for level, charge_squared, shell, angular_momentum in fixed:
        active = frequency >= helium.he_neutral_threshold_frequency_hz[level]
        cross = _karzas_latter_cross_section_grid(
            frequency, effective_charge_squared=charge_squared,
            principal_quantum_number=shell,
            orbital_angular_momentum=angular_momentum, tables=tables,
        )
        low[level] = torch.where(active, cross, low[level])

    rydberg_frequency = 109722.273 * LIGHT_SPEED_CM_PER_S_EXACT
    for level_cm, target_level in (
        (171135.000, 4), (169087.0, 3), (166277.546, 2), (159856.069, 1)
    ):
        threshold = (527490.06 - level_cm) * LIGHT_SPEED_CM_PER_S_EXACT
        cross = _karzas_latter_cross_section_grid(
            frequency, effective_charge_squared=threshold / rydberg_frequency,
            principal_quantum_number=1, orbital_angular_momentum=0, tables=tables,
        )
        low[target_level] = low[target_level] + torch.where(
            frequency >= threshold, cross, torch.zeros_like(cross)
        )
    for level_cm, target_level in (
        (186209.471, 9), (186101.0, 8), (185564.0, 7),
        (184864.0, 6), (183236.0, 5),
    ):
        threshold = (588451.59 - level_cm) * LIGHT_SPEED_CM_PER_S_EXACT
        cross = _karzas_latter_cross_section_grid(
            frequency, effective_charge_squared=threshold / rydberg_frequency,
            principal_quantum_number=1, orbital_angular_momentum=0, tables=tables,
        )
        low[target_level] = low[target_level] + torch.where(
            frequency >= threshold, cross, torch.zeros_like(cross)
        )

    high = torch.zeros(
        28, frequency.numel(), dtype=frequency.dtype, device=frequency.device
    )
    for shell in range(4, 28):
        high[shell] = _karzas_latter_cross_section_grid(
            frequency, effective_charge_squared=4.0 - 3.0 / shell**2,
            principal_quantum_number=1, orbital_angular_momentum=0, tables=tables,
        )
    return low, high


def _helium_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 4: He I absorption (:3006-3122)."""

    temperature = state.temperature
    mass_density = state.mass_density.clamp(min=1.0e-300)
    partition = state.helium_neutral_partition_normalized_population
    thermal_energy_ev = (temperature * BOLTZMANN_EV_PER_K_REFERENCE).clamp(
        min=1.0e-300
    )
    planck, exponential, stimulated = _planck_frequency_exact(
        temperature, frequency_hz
    )
    helium = tables.helium
    low_population = (
        torch.exp(
            -helium.he_neutral_excitation_ev[None, None, :]
            / thermal_energy_ev[:, :, None]
        )
        * helium.he_neutral_statistical_weights[None, None, :]
        * partition[:, :, None]
        / mass_density[:, :, None]
    )
    shells = torch.arange(
        28, dtype=temperature.dtype, device=temperature.device
    )
    high_population = (
        torch.exp(
            -24.587 * (1.0 - 1.0 / shells.clamp(min=1.0) ** 2)[None, None, :]
            / thermal_energy_ev[:, :, None]
        )
        * 4.0 * shells[None, None, :] ** 2
        * partition[:, :, None] / mass_density[:, :, None]
    )
    low_cross, high_cross = _helium_neutral_transition_grid(frequency_hz, tables)
    bound = low_population @ low_cross
    high_bound = high_population @ high_cross
    bound = bound + torch.where(
        (frequency_hz >= 1.25408e16)[None, None, :],
        high_bound, torch.zeros_like(high_bound),
    )

    freefree_density = (
        state.electron_density * state.helium_singly_ionized_population
        / mass_density / torch.sqrt(temperature.clamp(min=1.0e-300))
    )
    xr = partition * (4.0 / 2.0 / 13.595) * thermal_energy_ev / mass_density
    boltzmann_extension = torch.exp(-23.730 / thermal_energy_ev) * xr
    series_extension = torch.exp(-24.587 / thermal_energy_ev) * xr
    gaunt = _coulomb_freefree_gaunt(
        1, torch.log(frequency_hz.clamp(min=1.0e-300)),
        torch.log(temperature.clamp(min=1.0e-300)), tables,
    )
    low_frequency = frequency_hz < 2.055e14
    extension_population = torch.where(
        low_frequency[None, None, :],
        series_extension[:, :, None] / exponential,
        boltzmann_extension[:, :, None],
    )
    frequency_cubed = frequency_hz**3
    absorption = (
        (extension_population - series_extension[:, :, None])
        * (2.815e29 / frequency_cubed)[None, None, :]
        + bound
        + gaunt * (3.6919e8 / frequency_cubed)[None, None, :]
        * freefree_density[:, :, None]
    ) * stimulated
    return absorption, planck
# END BRANCH: helium_neutral


# BEGIN BRANCH: helium_ionized (flag 5)
def _helium_ionized_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 5: He II absorption (:3416-3523)."""

    temperature = state.temperature
    mass_density = state.mass_density.clamp(min=1.0e-300)
    electron_density = state.electron_density
    he_partition = state.helium_singly_ionized_partition_normalized_population
    he_doubly_ionized = state.ion_stage_populations_by_packed_slot[..., 4]
    _, exponential, stimulated = _planck_frequency_exact(temperature, frequency_hz)
    thermal_energy_ev = (temperature * BOLTZMANN_EV_PER_K_REFERENCE).clamp(
        min=1.0e-300
    )
    shell = torch.arange(
        1, 10, dtype=temperature.dtype, device=temperature.device
    )
    boltzmann_population = (
        torch.exp(
            -(54.403 - 54.403 / shell**2)[None, None, :]
            / thermal_energy_ev[:, :, None]
        )
        * 2.0
        * shell[None, None, :] ** 2
        * he_partition[:, :, None]
        / mass_density[:, :, None]
    )
    freefree_density = (
        electron_density * he_doubly_ionized / mass_density
        / torch.sqrt(temperature.clamp(min=1.0e-300))
    )
    xr = he_partition * (1.0 / 13.595) * thermal_energy_ev / mass_density
    boltzmann_extension = torch.exp(-53.859 / thermal_energy_ev) * xr
    series_extension = torch.exp(-54.403 / thermal_energy_ev) * xr
    gaunt = _coulomb_freefree_gaunt(
        2, torch.log(frequency_hz.clamp(min=1.0e-300)),
        torch.log(temperature.clamp(min=1.0e-300)), tables
    )
    cross_sections = torch.stack(
        [
            _karzas_latter_cross_section_grid(
                frequency_hz, effective_charge_squared=4.0,
                principal_quantum_number=index,
                orbital_angular_momentum=index, tables=tables
            )
            for index in range(1, 10)
        ], dim=0
    )
    frequency_cubed = frequency_hz**3
    freefree_coefficient = 3.6919e8 / frequency_cubed * 4.0
    extension_coefficient = 2.815e29 * 4.0 / frequency_cubed
    extension_population = torch.where(
        (frequency_hz < 1.31522e14)[None, None, :],
        series_extension[:, :, None] / exponential.clamp(min=1.0e-300),
        boltzmann_extension[:, :, None],
    )
    return (
        (extension_population - series_extension[:, :, None])
        * extension_coefficient[None, None, :]
        + torch.einsum("bli,if->blf", boltzmann_population, cross_sections)
        + gaunt * freefree_coefficient[None, None, :]
        * freefree_density[:, :, None]
    ) * stimulated
# END BRANCH: helium_ionized


# BEGIN BRANCH: heminus (flag 6)
def _heminus_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 6: He- absorption (:3376-3416)."""

    temperature = state.temperature
    frequency = frequency_hz
    mass_density = state.mass_density.clamp(min=1.0e-300)
    a_coeff = 3.397e-1 + (-5.216e14 + 7.039e30 / frequency) / frequency
    b_coeff = -4.116e3 + (1.067e19 + 8.135e34 / frequency) / frequency
    c_coeff = 5.081e8 + (-8.724e22 - 5.659e37 / frequency) / frequency
    return (
        (
            a_coeff[None, None, :] * temperature[:, :, None]
            + b_coeff[None, None, :]
            + c_coeff[None, None, :] / temperature[:, :, None]
        )
        / 1.0e15
        * state.electron_density[:, :, None]
        / 1.0e15
        * state.helium_neutral_population[:, :, None]
        / 1.0e15
        / mass_density[:, :, None]
    )
# END BRANCH: heminus


# BEGIN BRANCH: carbon_neutral (flag 8)
def _carbon_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: C I absorption (:3523-3759)."""

    temperature = state.temperature
    mass_density = state.mass_density.clamp(min=1.0e-300)
    population = state.partition_normalized_populations_by_packed_slot[..., 20]
    _, _, stimulated = _planck_frequency_exact(temperature, frequency_hz)
    lyman = frequency_hz <= 3.28805e15
    rydberg = 109732.298
    energy = torch.tensor(
        [79314.86,78731.27,78529.62,78309.76,78226.35,77679.82,
         73975.91,72610.72,71374.90,70743.95,69722.00,68856.33,
         61981.82,60373.00,21648.01,10192.63,43.42,16.42,0.00,
         119878.0,105798.7,97878.0,75254.93,64088.85,33735.20],
        dtype=temperature.dtype, device=temperature.device,
    )
    weight = torch.tensor(
        [9,3,7,15,21,5,1,5,9,3,15,3,3,9,1,5,5,3,1,3,3,5,12,15,5],
        dtype=temperature.dtype, device=temperature.device,
    )
    hc_over_kt = (
        PLANCK_ERG_SECOND_EXACT * LIGHT_SPEED_CM_PER_S_EXACT
        / (BOLTZMANN_ERG_PER_K_EXACT * temperature).clamp(min=1.0e-300)
    )
    boltzmann = weight[None, None, :] * torch.exp(
        -energy[None, None, :] * hc_over_kt[:, :, None]
    )
    wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
    rows = [torch.zeros_like(frequency_hz) for _ in range(25)]
    limit1, limit2 = 90862.70, 90820.42
    limit2b, limit3 = limit2 + 63.42, limit2 + 43003.3
    for level in range(14):
        threshold = limit1 - float(energy[level])
        angular = 2 if level < 6 else (1 if level < 12 else 0)
        cross = _karzas_latter_cross_section_grid(
            frequency_hz, effective_charge_squared=9.0 / rydberg * threshold,
            principal_quantum_number=3, orbital_angular_momentum=angular,
            tables=tables,
        )
        rows[level] = torch.where(
            lyman & (wavenumber >= threshold), cross, torch.zeros_like(cross)
        )
    for limit, limit_weight in ((limit2, 1.0 / 3.0), (limit2b, 2.0 / 3.0)):
        active = lyman & (wavenumber >= limit - float(energy[14]))
        background = 10.0 ** (
            -16.80 - (wavenumber - limit + energy[14]) / 3.0 / rydberg
        )
        resonance = (wavenumber - 97700.0) * 2.0 / 2743.0
        resonant = (68.0e-18 * resonance + 118.0e-18) / (resonance**2 + 1.0)
        rows[14] = rows[14] + torch.where(
            active, (background + resonant) * limit_weight,
            torch.zeros_like(background),
        )
        active = lyman & (wavenumber >= limit - float(energy[15]))
        background = 10.0 ** (
            -16.80 - (wavenumber - limit + energy[15]) / 3.0 / rydberg
        )
        r1 = (wavenumber - 93917.0) * 2.0 / 9230.0
        r2 = (wavenumber - 111130.0) * 2.0 / 2743.0
        resonant1 = (22.0e-18 * r1 + 26.0e-18) / (r1**2 + 1.0)
        resonant2 = (-10.5e-18 * r2 + 46.0e-18) / (r2**2 + 1.0)
        rows[15] = rows[15] + torch.where(
            active, (background + resonant1 + resonant2) * limit_weight,
            torch.zeros_like(background),
        )
        for level in range(16, 19):
            active = lyman & (wavenumber >= limit - float(energy[level]))
            value = 10.0 ** (
                -16.80 - (wavenumber - limit + energy[level]) / 3.0 / rydberg
            ) * limit_weight
            rows[level] = rows[level] + torch.where(
                active, value, torch.zeros_like(value)
            )
    for level in range(19, 25):
        threshold = limit3 - float(energy[level])
        cross = 3.0 * _karzas_latter_cross_section_grid(
            frequency_hz, effective_charge_squared=4.0 / rydberg * threshold,
            principal_quantum_number=2, orbital_angular_momentum=1, tables=tables
        )
        rows[level] = torch.where(
            lyman & (wavenumber >= threshold), cross, torch.zeros_like(cross)
        )
    cross_sections = torch.stack(rows, dim=0)
    frequency_factor = 2.815e29 / frequency_hz**3
    kramers_lower = torch.maximum(
        torch.full_like(wavenumber, limit2 - rydberg / 16.0),
        limit2 - wavenumber,
    )
    freefree = (
        frequency_factor[None, None, :] * 6.0
        / (rydberg * hc_over_kt[:, :, None])
        * (torch.exp(-kramers_lower[None, None, :] * hc_over_kt[:, :, None])
           - torch.exp(-limit2 * hc_over_kt)[:, :, None])
    )
    profile = freefree + torch.einsum("bli,if->blf", boltzmann, cross_sections)
    return torch.where(
        lyman[None, None, :],
        profile * stimulated * population[:, :, None] / mass_density[:, :, None],
        torch.zeros_like(profile),
    )
# END BRANCH: carbon_neutral


# BEGIN BRANCH: magnesium_neutral (flag 8)
def _magnesium_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: Mg I absorption (:3759-3959)."""

    temperature = state.temperature
    mass_density = state.mass_density.clamp(min=1.0e-300)
    population = state.partition_normalized_populations_by_packed_slot[..., 77]
    _, _, stimulated = _planck_frequency_exact(temperature, frequency_hz)
    lyman = frequency_hz <= 3.28805e15
    rydberg, limit = 109732.298, 61671.02
    energy_values = [54676.710,54676.438,54192.284,53134.642,49346.729,
                     47957.034,47847.797,46403.065,43503.333,41197.043,
                     35051.264,21919.178,21870.464,21850.405,0.0]
    energy = torch.tensor(energy_values, dtype=temperature.dtype, device=temperature.device)
    weight = torch.tensor(
        [21,7,15,5,3,15,9,5,1,3,3,5,3,1,1],
        dtype=temperature.dtype, device=temperature.device,
    )
    hc = PLANCK_ERG_SECOND_EXACT * LIGHT_SPEED_CM_PER_S_EXACT / (
        BOLTZMANN_ERG_PER_K_EXACT * temperature
    ).clamp(min=1.0e-300)
    boltzmann = weight[None, None, :] * torch.exp(-energy[None, None, :] * hc[:, :, None])
    wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
    rows = [torch.zeros_like(frequency_hz) for _ in range(15)]
    for level in range(5):
        threshold = limit - energy_values[level]
        angular = 3 if level < 2 else (2 if level < 4 else 1)
        cross = _karzas_latter_cross_section_grid(
            frequency_hz, effective_charge_squared=16.0 / rydberg * threshold,
            principal_quantum_number=4, orbital_angular_momentum=angular,
            tables=tables,
        )
        rows[level] = torch.where(
            lyman & (wavenumber >= threshold), cross, torch.zeros_like(cross)
        )
    empirical = {
        5: (25.0e-18, 13713.986, 2.7), 6: (33.8e-18, 13823.223, 2.8),
        7: (45.0e-18, 15267.955, 2.7), 8: (0.43e-18, 18167.687, 2.6),
        9: (2.1e-18, 20473.617, 2.6),
    }
    for level, (scale, edge, power) in empirical.items():
        value = scale * (edge / wavenumber) ** power
        rows[level] = torch.where(
            lyman & (wavenumber >= limit - energy_values[level]), value,
            torch.zeros_like(value),
        )
    ratio = 26619.756 / wavenumber
    value = 16.0e-18 * ratio**2.1 - 7.8e-18 * ratio**9.5
    rows[10] = torch.where(
        lyman & (wavenumber >= limit - energy_values[10]), value,
        torch.zeros_like(value),
    )
    for level in range(11, 14):
        ratio = 39759.842 / wavenumber
        value = torch.maximum(20.0e-18 * ratio**2.7, 40.0e-18 * ratio**14)
        rows[level] = torch.where(
            lyman & (wavenumber >= limit - energy_values[level]), value,
            torch.zeros_like(value),
        )
    value = 1.1e-18 * ((limit - energy_values[14]) / wavenumber) ** 10
    rows[14] = torch.where(
        lyman & (wavenumber >= limit - energy_values[14]), value,
        torch.zeros_like(value),
    )
    cross_sections = torch.stack(rows)
    lower = torch.maximum(
        torch.full_like(wavenumber, limit - rydberg / 25.0), limit - wavenumber
    )
    freefree = (
        (2.815e29 / frequency_hz**3)[None, None, :] * 2.0
        / (rydberg * hc[:, :, None])
        * (torch.exp(-lower[None, None, :] * hc[:, :, None])
           - torch.exp(-limit * hc)[:, :, None])
    )
    profile = freefree + torch.einsum("bli,if->blf", boltzmann, cross_sections)
    return torch.where(
        lyman[None, None, :],
        profile * stimulated * population[:, :, None] / mass_density[:, :, None],
        torch.zeros_like(profile),
    )
# END BRANCH: magnesium_neutral


# BEGIN BRANCH: aluminum_neutral (flag 8)
def _aluminum_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: Al I absorption (:5424-5469)."""

    mass_density = state.mass_density.clamp(min=1.0e-300)
    population = state.partition_normalized_populations_by_packed_slot[..., 90]
    _, _, stimulated = _planck_frequency_exact(state.temperature, frequency_hz)
    wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
    cross_section = torch.zeros_like(frequency_hz)
    limit = 48278.37
    active = frequency_hz <= 3.28805e15
    upper = active & (wavenumber >= limit - 112.061)
    lower = active & (wavenumber >= limit)
    cross_section = torch.where(
        upper, 6.5e-17 * ((limit - 112.061) / wavenumber) ** 5 * 4.0,
        cross_section,
    )
    cross_section = cross_section + torch.where(
        lower, 6.5e-17 * (limit / wavenumber) ** 5 * 2.0,
        torch.zeros_like(cross_section),
    )
    return (
        population[:, :, None] * stimulated / mass_density[:, :, None]
        * cross_section[None, None, :]
    )
# END BRANCH: aluminum_neutral


# BEGIN BRANCH: silicon_neutral (flag 8)
def _silicon_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: Si I absorption (:3959-4240)."""

    temperature = state.temperature
    mass_density = state.mass_density.clamp(min=1.0e-300)
    population = state.partition_normalized_populations_by_packed_slot[..., 104]
    _, _, stimulated = _planck_frequency_exact(temperature, frequency_hz)
    lyman = frequency_hz <= 3.28805e15
    rydberg = 109732.298
    energy_values = [59962.284,59100.0,59077.112,58893.40,58801.529,58777.0,
        57488.974,56503.346,54225.621,53387.34,53362.24,51612.012,
        50533.424,50189.389,49965.894,49399.670,49128.131,48161.459,
        47351.554,47284.061,40991.884,39859.920,15394.370,6298.850,
        223.157,77.115,0.0,94000.0,79664.0,72000.0,56698.738,45303.310,33326.053]
    weights = [9,56,15,7,3,28,21,5,15,3,7,1,9,5,21,3,9,15,5,3,3,9,1,5,5,3,1,3,3,5,12,15,5]
    levels = [(4,2),(4,3),(4,2),(4,2),(4,2),(4,3),(4,2),(4,2),
              (3,2),(3,2),(3,2),(4,1),(3,2),(4,1),(3,2),(4,1),
              (4,1),(4,1),(3,2),(4,1),(4,0),(4,0)]
    factors = [16,16,16,16,16,16,16,16,9,9,9,16,9,16,9,16,16,16,9,16,16,16]
    energy = torch.tensor(energy_values, dtype=temperature.dtype, device=temperature.device)
    weight = torch.tensor(weights, dtype=temperature.dtype, device=temperature.device)
    hc = PLANCK_ERG_SECOND_EXACT * LIGHT_SPEED_CM_PER_S_EXACT / (
        BOLTZMANN_ERG_PER_K_EXACT * temperature
    ).clamp(min=1.0e-300)
    boltzmann = weight[None, None, :] * torch.exp(-energy[None, None, :] * hc[:, :, None])
    wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
    rows = [torch.zeros_like(frequency_hz) for _ in range(33)]
    limit1 = 65939.18
    for level, ((principal, angular), factor) in enumerate(zip(levels, factors)):
        threshold = limit1 - energy_values[level]
        cross = _karzas_latter_cross_section_grid(
            frequency_hz, effective_charge_squared=factor / rydberg * threshold,
            principal_quantum_number=principal, orbital_angular_momentum=angular,
            tables=tables,
        )
        rows[level] = torch.where(
            lyman & (wavenumber >= threshold), cross, torch.zeros_like(cross)
        )
    for limit, limit_weight in ((65747.55, 1.0/3.0), (65747.55+287.45, 2.0/3.0)):
        active = lyman & (wavenumber >= limit - energy_values[22])
        resonance = (wavenumber - 70000.0) * 2.0 / 6500.0
        value = (37.0e-18 * (50353.180 / wavenumber)**2.40
                 + (97.0e-18*resonance + 94.0e-18)/(resonance**2+1.0)) * limit_weight
        rows[22] = rows[22] + torch.where(active, value, torch.zeros_like(value))
        active = lyman & (wavenumber >= limit - energy_values[23])
        resonance = (wavenumber - 78600.0) * 2.0 / 13000.0
        value = (24.5e-18 * (59448.700 / wavenumber)**1.85
                 + (-10.0e-18*resonance + 77.0e-18)/(resonance**2+1.0)) * limit_weight
        rows[23] = rows[23] + torch.where(active, value, torch.zeros_like(value))
        for level in (24,25,26):
            active = lyman & (wavenumber >= limit - energy_values[level])
            ratio = 65524.393 / wavenumber
            effective_weight = 2.0/3.0 if level == 25 else limit_weight
            value = torch.where(wavenumber <= 74000.0, 72.0e-18*ratio**1.90,
                                93.0e-18*ratio**4.00) * effective_weight
            rows[level] = rows[level] + torch.where(active, value, torch.zeros_like(value))
    limit3 = 65747.5 + 42824.35
    for level in range(27,33):
        threshold = limit3 - energy_values[level]
        cross = 3.0 * _karzas_latter_cross_section_grid(
            frequency_hz, effective_charge_squared=9.0/rydberg*threshold,
            principal_quantum_number=3, orbital_angular_momentum=1, tables=tables
        )
        rows[level] = torch.where(
            lyman & (wavenumber >= threshold), cross, torch.zeros_like(cross)
        )
    cross_sections = torch.stack(rows)
    freefree_limit = 65747.55
    lower = torch.maximum(torch.full_like(wavenumber, freefree_limit-rydberg/25.0),
                          freefree_limit-wavenumber)
    freefree = ((2.815e29/frequency_hz**3)[None,None,:]*6.0
                /(rydberg*hc[:,:,None])
                *(torch.exp(-lower[None,None,:]*hc[:,:,None])
                  - torch.exp(-freefree_limit*hc)[:,:,None]))
    profile = freefree + torch.einsum("bli,if->blf", boltzmann, cross_sections)
    return torch.where(
        lyman[None,None,:], profile*stimulated*population[:,:,None]/mass_density[:,:,None],
        torch.zeros_like(profile)
    )
# END BRANCH: silicon_neutral


# BEGIN BRANCH: iron_neutral (flag 8)
def _iron_neutral_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: Fe I absorption (:5469-5802)."""

    temperature = state.temperature
    mass_density = state.mass_density.clamp(min=1.0e-300)
    population = state.partition_normalized_populations_by_packed_slot[..., 350]
    _, _, stimulated = _planck_frequency_exact(temperature, frequency_hz)
    wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
    weights = [25,35,21,15,9,35,33,21,27,49,9,21,27,9,9,25,33,15,35,3,5,11,15,13,
               15,9,21,15,21,25,35,9,5,45,27,21,15,21,15,25,21,35,5,15,45,35,55,25]
    energies = [500,7500,12500,17500,19000,19500,19500,21000,22000,23000,23000,
                24000,24000,24500,24500,26000,26500,26500,27000,27500,28500,29000,
                29500,29500,29500,30000,31500,31500,33500,33500,34000,34500,34500,
                35000,35500,37000,37000,37000,38500,40000,40000,41000,41000,43000,
                43000,43000,43000,44000]
    thresholds = [63500,58500,53500,59500,45000,44500,44500,43000,58000,41000,54000,
                  40000,40000,57500,55500,38000,57500,57500,37000,54500,53500,55000,
                  34500,34500,34500,34000,32500,32500,32500,32500,32000,29500,29500,
                  31000,30500,29000,27000,54000,27500,24000,47000,23000,44000,42000,
                  42000,21000,42000,42000]
    hc = PLANCK_ERG_SECOND_EXACT * LIGHT_SPEED_CM_PER_S_EXACT / (
        BOLTZMANN_ERG_PER_K_EXACT * temperature
    ).clamp(min=1.0e-300)
    profile = torch.zeros(
        state.stars, state.layers, frequency_hz.numel(),
        dtype=temperature.dtype, device=temperature.device,
    )
    for weight, energy, threshold in zip(weights, energies, thresholds):
        active = wavenumber >= threshold
        cross = 3.0e-18 / (
            1.0 + ((threshold + 3000.0 - wavenumber) / threshold / 0.1) ** 4
        )
        profile = profile + (
            weight * torch.exp(-energy * hc)[:, :, None]
            * torch.where(active, cross, torch.zeros_like(cross))[None, None, :]
        )
    active = wavenumber >= 21000.0
    return torch.where(
        active[None,None,:],
        profile * stimulated * population[:,:,None] / mass_density[:,:,None],
        torch.zeros_like(profile),
    )
# END BRANCH: iron_neutral


# BEGIN BRANCH: molecular_continuum (flag 8)
def _molecular_continuum_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 8: CH/OH/H2 molecular continuum absorption (:5188-5259)."""

    temperature = state.temperature
    mass_density = state.mass_density.clamp(min=1.0e-300)
    _, _, stimulated = _planck_frequency_exact(temperature, frequency_hz)
    absorption = torch.zeros(
        state.stars, state.layers, frequency_hz.numel(),
        dtype=temperature.dtype, device=temperature.device,
    )
    photon_energy = (
        frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT / WAVENUMBER_PER_EV_REFERENCE
    )

    def molecular_cross_section(kind: str) -> torch.Tensor:
        if kind == "ch":
            raw_index = (photon_energy * 10.0).to(torch.int64)
            active = (raw_index >= 20) & (raw_index < 105)
            index = raw_index[active]
            lower = index.to(temperature.dtype) * 0.1
            fraction = (photon_energy[active] - lower) / 0.1
            table = tables.opacity.ch_cross_section_table
            partition_table = tables.opacity.ch_partition_table
        else:
            raw_index = (photon_energy * 10.0).to(torch.int64) - 20
            active = (raw_index > 0) & (raw_index < 130)
            index = raw_index[active] - 1
            lower = raw_index[active].to(temperature.dtype) * 0.1 + 2.0
            fraction = (photon_energy[active] - lower) / 0.1
            table = tables.opacity.oh_cross_section_table
            partition_table = tables.opacity.oh_partition_table
        cross_log = table[index] + (table[index + 1] - table[index]) * fraction[:, None]
        partition_index = ((temperature - 1000.0) / 200.0).to(torch.int64).clamp(0, 39)
        partition_lower = partition_index.to(temperature.dtype) * 200.0 + 1000.0
        partition = partition_table[partition_index] + (
            partition_table[partition_index + 1] - partition_table[partition_index]
        ) * (temperature - partition_lower) / 200.0
        temperature_index = ((temperature - 2000.0) / 500.0).to(torch.int64).clamp(0, 13)
        temperature_lower = temperature_index.to(temperature.dtype) * 500.0 + 2000.0
        temperature_fraction = (temperature - temperature_lower) / 500.0
        selected0 = cross_log[:, temperature_index].permute(1, 2, 0)
        selected1 = cross_log[:, temperature_index + 1].permute(1, 2, 0)
        interpolated = selected0 + (selected1 - selected0) * temperature_fraction[:, :, None]
        values = 10.0**interpolated * partition[:, :, None]
        result = torch.zeros_like(absorption)
        result[:, :, active] = values
        return torch.where(
            (temperature < 9000.0)[:, :, None], result, torch.zeros_like(result)
        )

    if bool((temperature < 9000.0).any()):
        absorption = absorption + molecular_cross_section("ch") * (
            state.ch_population / mass_density
        )[:, :, None] * stimulated
        absorption = absorption + molecular_cross_section("oh") * (
            state.oh_population / mass_density
        )[:, :, None] * stimulated

        hydrogen_partition = _hydrogen_neutral_partition_normalized_population_from_neutral(
            temperature, state.hydrogen_neutral_population, tables
        )
        molecular_hydrogen = _molecular_hydrogen_population(
            temperature, hydrogen_partition,
            state.hydrogen_departure_coefficients[:, :, 0], tables
        )
        molecular_hydrogen = torch.where(
            temperature > 20000.0, torch.zeros_like(molecular_hydrogen),
            molecular_hydrogen,
        )
        wavenumber = frequency_hz / LIGHT_SPEED_CM_PER_S_EXACT
        active = wavenumber <= 20000.0
        active_wavenumber = wavenumber[active]
        w_index = (active_wavenumber / 250.0).to(torch.int64).clamp(max=79)
        w_fraction = (active_wavenumber - 250.0 * w_index) / 250.0
        index0 = w_index.clamp(max=80)
        index1 = (w_index + 1).clamp(max=80)
        h2h2 = (
            tables.opacity.hydrogen_molecule_h2_collision_table[index0]
            * (1.0 - w_fraction[:, None])
            + tables.opacity.hydrogen_molecule_h2_collision_table[index1]
            * w_fraction[:, None]
        )
        h2he = (
            tables.opacity.hydrogen_molecule_he_collision_table[index0]
            * (1.0 - w_fraction[:, None])
            + tables.opacity.hydrogen_molecule_he_collision_table[index1]
            * w_fraction[:, None]
        )
        t_index = (temperature / 1000.0).to(torch.int64).clamp(1, 6)
        t_fraction = ((temperature - 1000.0 * t_index) / 1000.0).clamp(0.0, 1.0)
        h2h2_log = (
            h2h2[:, t_index - 1].permute(1, 2, 0) * t_fraction[:, :, None]
            + h2h2[:, t_index].permute(1, 2, 0) * (1.0 - t_fraction[:, :, None])
        )
        h2he_log = (
            h2he[:, t_index - 1].permute(1, 2, 0) * t_fraction[:, :, None]
            + h2he[:, t_index].permute(1, 2, 0) * (1.0 - t_fraction[:, :, None])
        )
        collision = (
            (10.0**h2he_log * state.helium_neutral_population[:, :, None]
             + 10.0**h2h2_log * molecular_hydrogen[:, :, None])
            * molecular_hydrogen[:, :, None] / mass_density[:, :, None]
            * stimulated[:, :, active]
        )
        absorption[:, :, active] = absorption[:, :, active] + collision
    return absorption
# END BRANCH: molecular_continuum


# BEGIN BRANCH: lukewarm_metal (flag 9)
def _seaton_bound_free_cross_section(
    threshold_frequency_hz: float,
    threshold_cross_section: float,
    power: float,
    asymptotic_constant: float,
    frequency_hz: torch.Tensor,
) -> torch.Tensor:
    """Vectorized Seaton bound-free profile (:2010-2023)."""

    ratio = threshold_frequency_hz / frequency_hz
    exponent = int(2.0 * power + 0.01)
    cross = (
        threshold_cross_section
        * (asymptotic_constant + (1.0 - asymptotic_constant) * ratio)
        * torch.sqrt(ratio**float(exponent))
    )
    return torch.where(
        frequency_hz >= threshold_frequency_hz, cross, torch.zeros_like(cross)
    )


def _lukewarm_metal_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 9: lukewarm metal absorption (:4686-4955)."""

    temperature = state.temperature
    frequency = frequency_hz
    wavenumber = frequency / LIGHT_SPEED_CM_PER_S_EXACT
    mass_density = state.mass_density.clamp(min=1.0e-300)
    planck, exp_hnu_over_kt, stimulated = _planck_frequency_exact(
        temperature, frequency
    )
    del planck
    hc_over_kt = (
        PLANCK_ERG_SECOND_EXACT * LIGHT_SPEED_CM_PER_S_EXACT
        / (BOLTZMANN_ERG_PER_K_EXACT * temperature).clamp(min=1.0e-300)
    )
    thermal_energy_ev = BOLTZMANN_EV_PER_K_REFERENCE * temperature
    population = state.partition_normalized_populations_by_packed_slot
    nitrogen_population = population[..., 27]
    oxygen_population = population[..., 35]
    carbon_ionized_population = population[..., 21]
    magnesium_ionized_population = population[..., 78]
    silicon_ionized_population = population[..., 105]
    calcium_ionized_population = population[..., 210]

    nitrogen_x853 = _seaton_bound_free_cross_section(
        3.517915e15, 1.142e-17, 2.0, 4.29, frequency
    )
    nitrogen_x1020 = _seaton_bound_free_cross_section(
        2.941534e15, 4.41e-18, 1.5, 3.85, frequency
    )
    nitrogen_x1130 = _seaton_bound_free_cross_section(
        2.653317e15, 4.2e-18, 1.5, 4.34, frequency
    )
    nitrogen_profile = (
        4.0 * nitrogen_x853[None, None, :]
        + nitrogen_x1020[None, None, :] * (
            10.0 * torch.exp(-2.384 / thermal_energy_ev)
        )[:, :, None]
        + nitrogen_x1130[None, None, :] * (
            6.0 * torch.exp(-3.575 / thermal_energy_ev)
        )[:, :, None]
    )
    oxygen_profile = 9.0 * _seaton_bound_free_cross_section(
        3.28805e15, 2.94e-18, 1.0, 2.66, frequency
    )

    magnesium_energy = torch.tensor(
        [112197.0, 108900.0, 103705.66, 103689.89, 103419.82,
         97464.32, 92790.51, 93799.70, 93310.80, 80639.85,
         69804.95, 71490.54, 35730.36, 0.0],
        dtype=temperature.dtype, device=temperature.device,
    )
    magnesium_weight = torch.tensor(
        [98.0, 72.0, 18.0, 14.0, 10.0, 6.0, 2.0,
         14.0, 10.0, 6.0, 2.0, 10.0, 6.0, 2.0],
        dtype=temperature.dtype, device=temperature.device,
    )
    magnesium_effective_charge = (
        49.0, 36.0, 25.0, 25.0, 25.0, 25.0, 25.0,
        16.0, 16.0, 16.0, 16.0, 9.0, 9.0,
    )
    magnesium_principal = (7, 6, 5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3)
    magnesium_angular = (7, 6, 4, 3, 2, 1, 0, 3, 2, 1, 0, 2, 1)
    magnesium_limit = 121267.61
    magnesium_rydberg = 109732.298
    magnesium_threshold = magnesium_limit - magnesium_energy
    magnesium_cross = torch.zeros(
        14, frequency.numel(), dtype=temperature.dtype, device=temperature.device
    )
    running_threshold = -float("inf")
    for level in range(13):
        threshold = float(magnesium_threshold[level])
        running_threshold = max(running_threshold, threshold)
        cross = _karzas_latter_cross_section_grid(
            frequency,
            effective_charge_squared=(
                magnesium_effective_charge[level] / magnesium_rydberg * threshold
            ),
            principal_quantum_number=magnesium_principal[level],
            orbital_angular_momentum=magnesium_angular[level], tables=tables,
        )
        magnesium_cross[level] = torch.where(
            wavenumber >= running_threshold, cross, torch.zeros_like(cross)
        )
    ratio = magnesium_threshold[13] / wavenumber.clamp(min=1.0e-300)
    magnesium_cross[13] = torch.where(
        wavenumber >= magnesium_threshold[13],
        0.14e-18 * (6.700 * ratio**4 - 5.700 * ratio**5),
        torch.zeros_like(ratio),
    )
    magnesium_boltzmann = (
        magnesium_weight[None, None, :]
        * torch.exp(-magnesium_energy[None, None, :] * hc_over_kt[:, :, None])
    )
    magnesium_dot = magnesium_boltzmann @ magnesium_cross
    magnesium_limit_boltzmann = torch.exp(-magnesium_limit * hc_over_kt)
    magnesium_kramers = magnesium_limit - magnesium_rydberg * 4.0 / 64.0
    magnesium_exponent = torch.maximum(
        torch.full_like(wavenumber, magnesium_kramers),
        magnesium_limit - wavenumber,
    )
    magnesium_profile = (
        (2.815e29 / frequency**3 * 16.0)[None, None, :]
        / (magnesium_rydberg * 4.0 * hc_over_kt[:, :, None])
        * (torch.exp(-magnesium_exponent[None, None, :] * hc_over_kt[:, :, None])
           - magnesium_limit_boltzmann[:, :, None])
        + magnesium_dot
    )

    carbon_energy = torch.tensor(
        [179073.05, 178955.94, 178495.47, 175292.30, 173347.84,
         168978.34, 168124.17, 162522.34, 157234.07, 145550.1,
         131731.8, 116537.65, 42.28, 202188.07, 199965.31,
         198856.92, 198431.96, 196572.80, 195786.71, 190000.0,
         188601.54, 186452.13, 184690.98, 182036.89, 181741.65,
         177787.22, 167009.29, 110651.76, 96493.74, 74931.11,
         43035.8, 230407.2, 150464.6, 142027.1],
        dtype=temperature.dtype, device=temperature.device,
    )
    carbon_weight = torch.tensor(
        [18.0, 14.0, 10.0, 6.0, 2.0, 14.0, 10.0, 6.0, 1.0,
         10.0, 6.0, 1.0, 3.0, 6.0, 10.0, 12.0, 10.0, 20.0,
         28.0, 2.0, 10.0, 12.0, 4.0, 6.0, 20.0, 6.0, 12.0,
         6.0, 2.0, 10.0, 12.0, 6.0, 10.0, 4.0],
        dtype=temperature.dtype, device=temperature.device,
    )
    carbon_principal = (5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3)
    carbon_angular = (4, 3, 2, 1, 0, 3, 2, 1, 0, 2, 1, 0)
    carbon_rydberg = 109732.298
    carbon_limit_1 = 196664.7
    carbon_limit_2 = carbon_limit_1 + 52367.06
    carbon_limit_3 = carbon_limit_1 + 137425.70
    carbon_cross = torch.zeros(
        34, frequency.numel(), dtype=temperature.dtype, device=temperature.device
    )

    def carbon_levels(
        levels: range, limit: float, principal_values, angular_values,
        multiplier: float = 1.0,
    ) -> None:
        running = -float("inf")
        for offset, level in enumerate(levels):
            threshold = limit - float(carbon_energy[level])
            running = max(running, threshold)
            principal = (
                principal_values[offset]
                if isinstance(principal_values, tuple) else principal_values
            )
            angular = (
                angular_values[offset]
                if isinstance(angular_values, tuple) else angular_values
            )
            charge_factor = float(principal**2)
            cross = multiplier * _karzas_latter_cross_section_grid(
                frequency,
                effective_charge_squared=charge_factor / carbon_rydberg * threshold,
                principal_quantum_number=principal,
                orbital_angular_momentum=angular, tables=tables,
            )
            carbon_cross[level] = torch.where(
                wavenumber >= running, cross, torch.zeros_like(cross)
            )

    carbon_levels(range(0, 12), carbon_limit_1, carbon_principal, carbon_angular)
    carbon_levels(range(13, 19), carbon_limit_2, 3, 2)
    carbon_levels(range(19, 25), carbon_limit_2, 3, 1)
    carbon_levels(range(25, 27), carbon_limit_2, 3, 0)
    carbon_levels(range(31, 34), carbon_limit_3, 2, 1, 3.0)
    carbon_boltzmann = (
        carbon_weight[None, None, :]
        * torch.exp(-carbon_energy[None, None, :] * hc_over_kt[:, :, None])
    )
    carbon_dot = carbon_boltzmann @ carbon_cross
    carbon_boltzmann_1 = torch.exp(-carbon_limit_1 * hc_over_kt)
    carbon_boltzmann_2 = torch.exp(-carbon_limit_2 * hc_over_kt)
    carbon_frequency = (2.815e29 * 16.0 / frequency**3)[None, None, :]
    carbon_exponent_1 = torch.maximum(
        torch.full_like(wavenumber, carbon_limit_1 - carbon_rydberg * 4.0 / 36.0),
        carbon_limit_1 - wavenumber,
    )
    carbon_exponent_2 = torch.maximum(
        torch.full_like(wavenumber, carbon_limit_2 - carbon_rydberg * 4.0 / 16.0),
        carbon_limit_2 - wavenumber,
    )
    carbon_profile = (
        carbon_frequency / (carbon_rydberg * 4.0 * hc_over_kt[:, :, None])
        * (torch.exp(-carbon_exponent_1[None, None, :] * hc_over_kt[:, :, None])
           - carbon_boltzmann_1[:, :, None])
        + carbon_frequency * 9.0
        / (carbon_rydberg * 4.0 * hc_over_kt[:, :, None])
        * (torch.exp(-carbon_exponent_2[None, None, :] * hc_over_kt[:, :, None])
           - carbon_boltzmann_2[:, :, None])
        + carbon_dot
    )

    temperature_log10 = torch.log(temperature.clamp(min=1.0e-300)) / REFERENCE_NATURAL_LOG_10
    silicon_t_index = ((temperature_log10 - 3.48) / 0.02).to(torch.int64).clamp(1, 50)
    silicon_t_fraction = (
        temperature_log10 - 3.48 - silicon_t_index.to(temperature.dtype) * 0.02
    ) / 0.02
    silicon_helper = (
        torch.exp(-131838.4 * hc_over_kt)
        + 9.0 * torch.exp(-184563.09 * hc_over_kt)
    ) / (109732.298 * 4.0 * hc_over_kt)
    silicon_uses_table = wavenumber >= 12192.48
    silicon_w_bin = (wavenumber * 0.001).to(torch.int64).clamp(1, 199)
    silicon_w_fraction = (
        wavenumber - silicon_w_bin.to(wavenumber.dtype) * 1000.0
    ) / 1000.0
    table = tables.lukewarm_silicon_table
    w0 = (silicon_w_bin - 1)[None, None, :]
    w1 = silicon_w_bin[None, None, :]
    t0 = (silicon_t_index - 1)[:, :, None]
    t1 = silicon_t_index[:, :, None]
    h00, h01 = table[w0, t0], table[w0, t1]
    h10, h11 = table[w1, t0], table[w1, t1]
    h0 = h00 * (1.0 - silicon_t_fraction[:, :, None]) + h01 * silicon_t_fraction[:, :, None]
    h1 = h10 * (1.0 - silicon_t_fraction[:, :, None]) + h11 * silicon_t_fraction[:, :, None]
    silicon_table_profile = torch.exp(
        h0 * (1.0 - silicon_w_fraction[None, None, :])
        + h1 * silicon_w_fraction[None, None, :]
    )
    silicon_low_profile = (
        (2.815e29 * 16.0 / frequency**3)[None, None, :]
        * (1.0 / exp_hnu_over_kt.clamp(min=1.0e-300) - 1.0)
        * silicon_helper[:, :, None]
    )
    silicon_profile = torch.where(
        silicon_uses_table[None, None, :], silicon_table_profile, silicon_low_profile
    )

    calcium_x1044 = torch.where(
        frequency >= 2.870454e15,
        5.4e-20 * (2.870454e15 / frequency) ** 3,
        torch.zeros_like(frequency),
    )
    calcium_x1218 = torch.where(
        frequency >= 2.460127e15,
        1.64e-17 * torch.sqrt(2.460127e15 / frequency),
        torch.zeros_like(frequency),
    )
    calcium_x1420 = _seaton_bound_free_cross_section(
        2.110779e15, 4.13e-18, 3.0, 0.69, frequency
    )
    calcium_profile = (
        2.0 * calcium_x1044[None, None, :]
        + calcium_x1218[None, None, :] * (
            10.0 * torch.exp(-1.697 / thermal_energy_ev)
        )[:, :, None]
        + calcium_x1420[None, None, :] * (
            6.0 * torch.exp(-3.142 / thermal_energy_ev)
        )[:, :, None]
    )

    common = stimulated / mass_density[:, :, None]
    return common * (
        nitrogen_profile * nitrogen_population[:, :, None]
        + oxygen_profile[None, None, :] * oxygen_population[:, :, None]
        + calcium_profile * calcium_ionized_population[:, :, None]
        + carbon_profile * carbon_ionized_population[:, :, None]
        + magnesium_profile * magnesium_ionized_population[:, :, None]
        + silicon_profile * silicon_ionized_population[:, :, None]
    )
# END BRANCH: lukewarm_metal


# BEGIN BRANCH: hot_metal (flag 10)
def _hot_metal_absorption(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
) -> torch.Tensor:
    """Flag 10: hot metal absorption (:5279-5424)."""

    temperature = state.temperature
    electron_density = state.electron_density
    mass_density = state.mass_density.clamp(min=1.0e-300)
    partition = state.partition_normalized_populations_by_packed_slot
    ion = state.ion_stage_populations_by_packed_slot
    _, _, stimulated = _planck_frequency_exact(temperature, frequency_hz)
    temperature_log = torch.log(temperature.clamp(min=1.0e-10))
    thermal_energy_ev = BOLTZMANN_EV_PER_K_REFERENCE * temperature
    hot_population = torch.cat(
        [partition[..., 20:24], partition[..., 27:32],
         partition[..., 35:41], partition[..., 54:60]], dim=-1
    )
    charge_square = []
    for ion_charge in range(1, 6):
        total = torch.zeros_like(temperature)
        for start in (21, 28, 36, 55, 78, 105, 136, 351):
            source_index = start + ion_charge - 1
            if source_index < ion.shape[-1]:
                total = total + ion_charge**2 * ion[..., source_index]
        charge_square.append(total)

    log_frequency = torch.log(frequency_hz)
    freefree_sum = torch.zeros(
        state.stars, state.layers, frequency_hz.numel(),
        dtype=temperature.dtype, device=temperature.device,
    )
    for ion_charge in range(1, 6):
        freefree_sum = freefree_sum + _coulomb_freefree_gaunt(
            ion_charge, log_frequency, temperature_log, tables
        ) * charge_square[ion_charge - 1][:, :, None]
    opacity = (
        freefree_sum * (3.6919e8 / frequency_hz[None, None, :] ** 3)
        * electron_density[:, :, None]
        / torch.sqrt(temperature.clamp(min=1.0e-30))[:, :, None]
    )
    for row in tables.hot_metal_transitions:
        threshold_frequency, cross_section, alpha, power, multiplier, excitation, raw_index = row
        active = frequency_hz >= threshold_frequency
        ratio = threshold_frequency / frequency_hz
        transition_cross_section = (
            cross_section * (alpha + ratio - alpha * ratio)
            * torch.sqrt(ratio ** int(power))
        )
        population_index = max(0, min(int(raw_index) - 1, 20))
        weighted = (
            transition_cross_section[None, None, :]
            * hot_population[:, :, population_index, None] * multiplier
        )
        contribution = weighted * torch.exp(
            -excitation / thermal_energy_ev.clamp(min=1.0e-30)
        )[:, :, None]
        opacity = opacity + torch.where(
            active[None, None, :] & (weighted > opacity / 100.0),
            contribution, torch.zeros_like(contribution)
        )
    return opacity * stimulated / mass_density[:, :, None]
# END BRANCH: hot_metal


# --- scattering ---------------------------------------------------------------


def _continuum_scattering(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    tables: TwinContinuumTables,
    flags: list[int],
) -> torch.Tensor:
    """Batched ``compute_continuum_scattering_columns`` (:6056-6154).

    Electron scattering (flag 11), Rayleigh H (flag 3), Rayleigh He (flag 7),
    and Rayleigh H2 (flag 12, only evaluated when flag 3 is on, reproducing
    the reference's dependence on the recomputed neutral-hydrogen ground
    population). Output is ``(star, layer, freq)``.
    """

    temperature = state.temperature
    mass_density = torch.clamp(state.mass_density, min=1.0e-300)
    scattering = torch.zeros(
        state.stars,
        state.layers,
        frequency_hz.numel(),
        dtype=temperature.dtype,
        device=temperature.device,
    )

    if flags[11] == 1:
        scattering = scattering + (
            0.6653e-24 * state.electron_density / mass_density
        )[:, :, None]

    hydrogen_neutral_partition_normalized_population = None
    if flags[3] == 1:
        hydrogen_neutral_partition_normalized_population = (
            _hydrogen_neutral_partition_normalized_population_from_neutral(
                temperature, state.hydrogen_neutral_population, tables
            )
        )
        hydrogen_departure = state.hydrogen_departure_coefficients[:, :, 0]
        population_over_density = (
            hydrogen_neutral_partition_normalized_population
            * 2.0
            * hydrogen_departure
            / mass_density
        )
        wavelength_angstrom = LIGHT_SPEED_ANGSTROM_PER_S / torch.clamp(
            frequency_hz, max=2.463e15
        )
        wavelength_squared = wavelength_angstrom * wavelength_angstrom
        cross_section = (
            5.799e-13
            + 1.422e-6 / wavelength_squared
            + 2.784 / (wavelength_squared * wavelength_squared)
        ) / (wavelength_squared * wavelength_squared)
        scattering = scattering + (
            population_over_density[:, :, None] * cross_section[None, None, :]
        )

    if flags[7] == 1:
        helium_neutral = state.helium_neutral_population
        wave = LIGHT_SPEED_ANGSTROM_PER_S / torch.clamp(frequency_hz, max=5.15e15)
        wave_squared = wave * wave
        cross_section = (
            5.484e-14
            / (wave_squared * wave_squared)
            * (
                1.0
                + (2.44e5 + 5.94e10 / torch.clamp(wave_squared - 2.90e5, min=1.0e-10))
                / wave_squared
            )
            ** 2
        )
        scattering = scattering + (
            (helium_neutral / mass_density)[:, :, None] * cross_section[None, None, :]
        )

    if flags[12] == 1 and hydrogen_neutral_partition_normalized_population is not None:
        hydrogen_departure = state.hydrogen_departure_coefficients[:, :, 0]
        molecular_hydrogen = _molecular_hydrogen_population(
            temperature,
            hydrogen_neutral_partition_normalized_population,
            hydrogen_departure,
            tables,
        )
        molecular_hydrogen = torch.where(
            temperature > 20000.0, torch.zeros_like(molecular_hydrogen), molecular_hydrogen
        )
        wave = LIGHT_SPEED_ANGSTROM_PER_S / torch.clamp(frequency_hz, max=2.922e15)
        wave_squared = wave * wave
        cross_section = (
            8.14e-13 + 1.28e-6 / wave_squared + 1.61 / (wave_squared * wave_squared)
        ) / (wave_squared * wave_squared)
        scattering = scattering + (
            (molecular_hydrogen / mass_density)[:, :, None]
            * cross_section[None, None, :]
        )

    return scattering


# --- assembly -----------------------------------------------------------------


def continuum_opacity(
    state: TwinContinuumState,
    frequency_hz: torch.Tensor,
    *,
    tables: TwinContinuumTables,
    flags: list[int] | tuple[int, ...] | None = None,
    frequency_chunk: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Continuum absorption, scattering, and source on a shared frequency grid.

    Batched ``compute_continuum_opacity_columns``
    (``continuum_opacity.py:5834-5971``). ``frequency_hz`` is a one-dimensional
    grid shared by the whole star batch (the reference grid depends on Teff;
    batch stars with the same grid start class together, see
    ``TwinContinuumTables.sampling_grid``). ``flags`` are the 20 IFOP values
    (default all ones, padded with zeros like the reference); flag 18 raises
    ``NotImplementedError`` (module docstring). Returns ``(absorption,
    scattering, source)``, each ``(star, layer, freq)`` float64.
    ``frequency_chunk`` evaluates the grid in sequential chunks and
    concatenates, bounding peak memory.
    """

    frequency_hz = torch.as_tensor(
        frequency_hz, dtype=tables.dtype, device=tables.device
    )
    if frequency_hz.dim() != 1:
        raise ValueError("frequency_hz must be one-dimensional")
    resolved_flags = [1] * 20 if flags is None else [int(value) for value in flags]
    if len(resolved_flags) < 20:
        resolved_flags.extend([0] * (20 - len(resolved_flags)))
    if resolved_flags[18] == 1:
        raise NotImplementedError(
            "flag 18 (Rosseland-table continuum) is not implemented; it is off "
            "in DEFAULT_OPACITY_FLAGS and its first-iteration table is empty"
        )
    if frequency_chunk is not None and frequency_hz.numel() > frequency_chunk:
        absorption_parts, scattering_parts, source_parts = [], [], []
        for start in range(0, frequency_hz.numel(), int(frequency_chunk)):
            chunk = frequency_hz[start : start + int(frequency_chunk)]
            a, s, src = continuum_opacity(
                state,
                chunk,
                tables=tables,
                flags=resolved_flags,
                frequency_chunk=None,
            )
            absorption_parts.append(a)
            scattering_parts.append(s)
            source_parts.append(src)
        return (
            torch.cat(absorption_parts, dim=2),
            torch.cat(scattering_parts, dim=2),
            torch.cat(source_parts, dim=2),
        )

    planck_nu, _, _ = _planck_frequency_exact(state.temperature, frequency_hz)
    absorption = torch.zeros_like(planck_nu)
    source_numerator = torch.zeros_like(planck_nu)

    if resolved_flags[0] == 1:
        hydrogen_absorption, hydrogen_source = _hydrogen_absorption(
            state, frequency_hz, tables
        )
        absorption = absorption + hydrogen_absorption
        source_numerator = source_numerator + hydrogen_absorption * hydrogen_source

    if resolved_flags[2] == 1:
        hminus_absorption, hminus_source = _hminus_absorption(
            state, frequency_hz, tables
        )
        absorption = absorption + hminus_absorption
        source_numerator = source_numerator + hminus_absorption * hminus_source

    thermal_absorption = torch.zeros_like(planck_nu)
    if resolved_flags[1] == 1:
        thermal_absorption = thermal_absorption + _molecular_hydrogen_ion_absorption(
            state, frequency_hz, tables
        )
    if resolved_flags[4] == 1:
        helium_neutral_absorption, _ = _helium_neutral_absorption(
            state, frequency_hz, tables
        )
        thermal_absorption = thermal_absorption + helium_neutral_absorption
    if resolved_flags[5] == 1:
        thermal_absorption = thermal_absorption + _helium_ionized_absorption(
            state, frequency_hz, tables
        )
    if resolved_flags[6] == 1:
        thermal_absorption = thermal_absorption + _heminus_absorption(
            state, frequency_hz, tables
        )
    if resolved_flags[8] == 1:
        thermal_absorption = thermal_absorption + (
            _molecular_continuum_absorption(state, frequency_hz, tables)
            + _carbon_neutral_absorption(state, frequency_hz, tables)
            + _magnesium_neutral_absorption(state, frequency_hz, tables)
            + _aluminum_neutral_absorption(state, frequency_hz, tables)
            + _silicon_neutral_absorption(state, frequency_hz, tables)
            + _iron_neutral_absorption(state, frequency_hz, tables)
        )
    if resolved_flags[9] == 1:
        thermal_absorption = thermal_absorption + _lukewarm_metal_absorption(
            state, frequency_hz, tables
        )
    if resolved_flags[10] == 1:
        thermal_absorption = thermal_absorption + _hot_metal_absorption(
            state, frequency_hz, tables
        )

    absorption = absorption + thermal_absorption
    source_numerator = source_numerator + thermal_absorption * planck_nu

    active = absorption > 0.0
    source = torch.where(
        active,
        source_numerator / torch.where(active, absorption, torch.ones_like(absorption)),
        planck_nu,
    )

    scattering = _continuum_scattering(state, frequency_hz, tables, resolved_flags)
    return absorption, scattering, source
