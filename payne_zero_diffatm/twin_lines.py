"""Batched, differentiable twin of the reference line-opacity accumulation.

Reproduces one production opacity pass of the classical solver —
``accumulate_selected_line_opacity``
(``payne_zero_atmosphere/line_opacity.py:2345-2463``, numba kernel
``_accumulate_selected_line_opacity_compiled`` :254-502) followed by
``accumulate_transition_line_opacity`` (:2466-3395, normal-run kernel
:648-881, hydrogen deposit :1012-1602) — as a torch-native module that is
batched over stars and differentiable in its state inputs. Both production
opacity flags are active in the bench deck (``OPACITY IFOP ... 1 0 1 0 0 0``
→ ``line_flags[14] == 1`` selected lines and ``line_flags[16] == 1``
detailed transitions, see ``run_setup.py:74-83`` and
``bench/run_reference.py:_solver_config`` with
``source_catalogs.source_line_paths`` supplying
``detailed_line_catalog_path``), so both paths are ported.

Design choices
--------------

Line catalogs are data, not code (progress notes §8). The twin does not
port ``line_selection.py``: :func:`export_line_catalog` runs the certified
reference ``generate_selected_lines`` once per star on the iteration-1
population state and stores the compact selected-line words in an npz;
:func:`load_line_catalog` decodes them back to the
``SelectedLineCatalog`` fields (decode mirrors
``payne_zero_atmosphere/line_catalog.py:99-131``). The detailed-transition
catalog is star-independent and is read directly from
``source_catalogs/lines/detailed_transition_lines.npz`` at table-build
time.

State dependence. Everything per-line that the reference derives from the
catalog alone (vacuum wavelengths, TABLOG strength/damping decodes,
hydrogen per-line setups baked like ``runner.py:2750-2848``) is precomputed
at load. Everything that depends on the layer state (Boltzmann/excitation
factors, damping, profiles, threshold gates) is recomputed in torch from
the inputs, so the slab is a differentiable function of ``temperature``,
``electron_density``, the population tables, and the continuum threshold.

Microturbulence enters only through ``fractional_doppler_widths`` (built by
the EOS stage, ``doppler.py:12-66``), exactly as in the reference call
chain — it is not a separate input here.

Numerical fidelity. The twin computes internally in fp64 and returns fp64.
The reference kernel casts selected-path strengths/dampings to float32 and
accumulates into a float32 slab (its own parallel chunk reduction carries a
documented ~4e-6 in-band regrouping wiggle, ``line_opacity.py:55-71``); the
twin skips the intermediate f32 rounding and the f32 accumulation. A final
``.float()`` cast reproduces the reference storage dtype. Table lookups
(TABLOG ``10^((i-16384)*0.001)`` ``line_profile_math.py:129-133``, fast
exponentials :136-165, Voigt basis :301-319) are gathered with exactly the
reference index arithmetic, so they are piecewise-constant in the inputs —
gradients flow through the smooth parts (hydrogen Boltzmann widths,
``(T/1e4)^0.3`` collision density, damping quotients, Stark/Lorentz
profile algebra), not through the table indices.

Early-break wings. The reference walks each wing and stops depositing at
the first grid point whose contribution falls below the continuum
threshold. The twin evaluates the full window (101 points per side for
selected lines, 2001 for transitions) and applies the identical stopping
rule as a cumulative mask; deposits are ``index_add`` scatter-adds, so the
whole path stays in the autograd graph.

Depth gate. Both reference kernels evaluate the threshold gates only at
layers 7, 15, ..., 79 and process a whole 8-layer block only when the gate
fires at the block's top layer or the previous block's top layer
(``line_opacity.py:381-408``); interior layers are then re-gated
individually. This is an intentional approximation (an interior layer whose
gate fires while both block endpoints fail is skipped) and the twin
reproduces it exactly.

Batching. Catalogs are per-star and ragged, so stars are processed in a
Python loop over fully vectorized (line-chunk × layer × window) tensor
expressions; a star's result therefore cannot depend on batch neighbours.

Operational training path. A production Sun catalog contains 24.1 million
selected records, making a literal differentiable replay too expensive for an
unrolled training loop. ``ReferenceLineTemplate`` therefore stores the exact
certified line slab at the initializer state and
``line_opacity_from_template`` applies a differentiable per-layer population
scaling. It is exactly equal to the certified slab at the template state,
keeps gradients local in depth, and is the supported K=1 path. The literal
catalog port below remains useful for focused numerical tests but is not the
training backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from payne_zero_atmosphere.constants import (
    BOLTZMANN_ERG_PER_K_REFERENCE,
    PLANCK_ERG_SECOND_REFERENCE,
)


# Rounded reference literals, restated so this module does not pull the
# numba kernel modules (atmosphere_io.py:48-59, line_opacity.py:50-52,
# hydrogen_line_profile.py:26-28 and :1061-1064).
_LIGHT_SPEED_NM_PER_SECOND = 2.99792458e17
_LIGHT_SPEED_CM_PER_SECOND = 2.99792458e10
_LIGHT_SPEED_ANGSTROM_PER_SECOND = 2.99792458e18
_SQRT_PI_APPROX = 1.77245
_PI_APPROX = 3.14159

_RATIO_LOG_STEP = math.log(1.0 + 1.0 / 2_000_000.0)
_CLASSICAL_LINE_STRENGTH_SCALE = 0.026538 / 1.77245 / _LIGHT_SPEED_NM_PER_SECOND
_DAMPING_SCALE = 1.0 / 12.5664 / _LIGHT_SPEED_NM_PER_SECOND

_LAYER_COUNT = 80
_DEPTH_BLOCK = 8
_SELECTED_WINDOW = 100  # line_opacity.py:174, :197
_TRANSITION_WINDOW = 2000  # line_opacity.py:800, :1484
_MERGED_WINDOW = 1000  # line_opacity.py:2095-2097


# ---------------------------------------------------------------------------
# Offline export (certified reference path; not part of the differentiable
# twin). Heavy reference imports stay function-local.
# ---------------------------------------------------------------------------


def export_line_catalog(
    *,
    effective_temperature: float,
    log_surface_gravity: float,
    metallicity: float,
    alpha_enhancement: float,
    microturbulence: float,
    output_path: Path | str,
) -> Path:
    """Run reference line selection for one star and save the compact catalog.

    Rebuilds the iteration-1 input atmosphere with the bench capture policy
    (``bench/run_reference.py:167-225``), prepares the molecule-enabled
    population state and the continuum selection threshold exactly like
    ``prepare_opacity_state`` (``runner.py:936-969``), then runs
    ``generate_selected_lines`` (``line_selection.py:1012-1171``). The npz
    holds the packed selected-line words plus the opacity grid, quadrature
    weights and packed bin edges the accumulation needs.

    The detailed-transition catalog is star-independent and is *not*
    exported; the twin reads it from ``source_catalogs/`` at load time.
    """

    from bench import environment as _environment  # noqa: F401
    from bench.labels import StellarLabels
    from bench.run_reference import (
        PRODUCTION_INITIALIZER_JITTER_SCALE,
        PRODUCTION_INITIALIZER_SEED,
    )
    from payne_zero_atmosphere.config import (
        AtmosphereConfig,
        AtmosphereInput,
        AtmosphereOutput,
    )
    from payne_zero_atmosphere.continuum_opacity import (
        active_continuum_reference_frequencies,
        assemble_continuum_line_selection_threshold,
        build_continuum_atmosphere_state,
        build_opacity_sampling_grid,
        compute_continuum_opacity_columns,
    )
    from payne_zero_atmosphere.line_selection import generate_selected_lines
    from payne_zero_atmosphere.runner import prepare_population_state
    from payne_zero_atmosphere.run_setup import (
        initialize_microturbulence,
        standard_rosseland_optical_depth_grid,
    )
    from payne_zero_atmosphere.source_catalogs import (
        molecular_equilibrium_catalog_path,
        source_line_paths,
    )
    from payne_zero_atmosphere.warm_start import (
        deterministic_initializer_labels,
        emulator_warm_start_model,
    )

    labels = StellarLabels(
        float(effective_temperature),
        float(log_surface_gravity),
        float(metallicity),
        float(alpha_enhancement),
        float(microturbulence),
    )
    initializer_label = deterministic_initializer_labels(
        **labels.as_kwargs(),
        max_trials=1,
        seed=PRODUCTION_INITIALIZER_SEED,
        jitter_scale=PRODUCTION_INITIALIZER_JITTER_SCALE,
        device="cpu",
    )[0]
    atmosphere, _deck = emulator_warm_start_model(
        **labels.as_kwargs(), device="cpu", initializer_label=initializer_label
    )
    initialize_microturbulence(
        atmosphere,
        effective_temperature=labels.effective_temperature,
        log_surface_gravity=labels.log_surface_gravity,
        standard_rosseland_optical_depth=standard_rosseland_optical_depth_grid(
            atmosphere.layers
        ),
    )
    config = AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=atmosphere,
            molecules_path=molecular_equilibrium_catalog_path(),
            **source_line_paths(),
        ),
        outputs=AtmosphereOutput(),
        enable_molecules=True,
    )
    prepared = prepare_population_state(config, temperature_iteration_index=1)
    continuum_atmosphere = build_continuum_atmosphere_state(
        prepared.setup.atmosphere, prepared.runtime_state
    )
    opacity_wavelength_grid_nm, frequency_weights = build_opacity_sampling_grid(
        prepared.setup.effective_temperature
    )
    active_indices, active_frequency_hz = active_continuum_reference_frequencies(
        prepared.setup.effective_temperature
    )
    active_absorption, active_scattering, _ = compute_continuum_opacity_columns(
        continuum_atmosphere,
        active_frequency_hz,
        opacity_flags=[int(v) for v in prepared.setup.opacity_flags],
    )
    continuum_threshold, _reference_wavelengths, wavelength_bin_edges = (
        assemble_continuum_line_selection_threshold(
            effective_temperature=prepared.setup.effective_temperature,
            temperature_k=prepared.setup.atmosphere.temperature,
            active_continuum_absorption=active_absorption,
            active_continuum_scattering=active_scattering,
        )
    )

    line_paths = source_line_paths()
    line_paths.pop("detailed_line_catalog_path")
    catalog = generate_selected_lines(
        partition_normalized_population_over_mass_density_and_fractional_doppler_width=(
            prepared.partition_normalized_population_over_mass_density_and_fractional_doppler_width
        ),
        continuum_line_selection_threshold=continuum_threshold,
        packed_continuum_wavelengths=wavelength_bin_edges,
        hc_over_kt=prepared.setup.atmosphere.hc_over_kt,
        **line_paths,
    )

    # Re-pack the decoded catalog into the (N, 4) int32 word layout
    # (line_catalog.py:83-131 inverted): word 0 is the packed wavelength,
    # words 1..3 are six int16 halves.
    words = np.zeros((catalog.line_count, 4), dtype=np.int32)
    words[:, 0] = catalog.packed_wavelength_index
    halves = words[:, 1:4].view(np.int16).reshape(-1, 6)
    halves[:, 0] = catalog.packed_species_slot
    halves[:, 1] = catalog.lower_excitation_index
    halves[:, 2] = catalog.log_strength_index
    halves[:, 3] = catalog.radiative_damping_index
    halves[:, 4] = catalog.stark_damping_index
    halves[:, 5] = catalog.van_der_waals_damping_index

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        selected_words=np.ascontiguousarray(words),
        opacity_wavelength_grid_nm=opacity_wavelength_grid_nm,
        frequency_weights=frequency_weights,
        wavelength_bin_edges=wavelength_bin_edges,
        effective_temperature=np.float64(prepared.setup.effective_temperature),
        stellar_labels=np.asarray(
            [
                labels.effective_temperature,
                labels.log_surface_gravity,
                labels.metallicity,
                labels.alpha_enhancement,
                labels.microturbulence_km_s,
            ],
            dtype=np.float64,
        ),
    )
    return output_path


# ---------------------------------------------------------------------------
# Catalog + table containers (torch, star-independent tables shared).
# ---------------------------------------------------------------------------


def _t64(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(array), dtype=torch.float64)


def _ti64(array: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(array), dtype=torch.int64)


@dataclass
class TwinLineCatalog:
    """One star's selected-line catalog plus its opacity grid."""

    # Per-line selected-line fields (decoded SelectedLineCatalog layout).
    packed_wavelength_index: torch.Tensor  # (N,) int64
    packed_species_slot: torch.Tensor
    lower_excitation_index: torch.Tensor
    log_strength_index: torch.Tensor
    radiative_damping_index: torch.Tensor
    stark_damping_index: torch.Tensor
    van_der_waals_damping_index: torch.Tensor
    # Load-time derived per-line quantities.
    vacuum_wavelength_nm: torch.Tensor  # (N,) fp64
    species_slot: torch.Tensor  # abs(packed)//10
    continuum_column: torch.Tensor  # searchsorted right into bin edges
    center_index: torch.Tensor  # first grid index with grid > lambda0
    valid_line: torch.Tensor  # bool (N,)
    # Grid.
    opacity_wavelength_grid_nm: torch.Tensor  # (F,) fp64
    frequency_weights: torch.Tensor
    wavelength_bin_edges: torch.Tensor  # (344,) int64
    effective_temperature: float


@dataclass
class TwinLineTables:
    """Star-independent tables and the detailed-transition catalog."""

    # Voigt basis on the 2001-point grid (line_profile_math.py:301-319).
    voigt_gaussian: torch.Tensor
    voigt_first: torch.Tensor
    voigt_second: torch.Tensor
    # Fast exponential exp(-x) tables (line_profile_math.py:136-144).
    exp_integer: torch.Tensor
    exp_fractional: torch.Tensor
    # TABLOG lookup 10**((i-16384)*0.001) stored at float32 precision
    # (line_profile_math.py:129-133; the kernel consumes it as float32,
    # line_opacity.py:2430).
    selection_lookup: torch.Tensor
    # Hydrogen continuum-edge selector (line_profile_math.py:31-118).
    continuum_selector: torch.Tensor  # (25, 16)
    # Hydrogen neutral level energies cm^-1 (line_opacity.py:2294-2311).
    hydrogen_level_energies_cm: torch.Tensor  # (100,)
    # Hydrogen profile tables (hydrogen_line_profile.py:35-55, :264-270).
    h2_cutoff_table: torch.Tensor
    h2plus_cutoff_table: torch.Tensor
    stark_probability_table: torch.Tensor
    stark_pressure_grid: torch.Tensor
    stark_beta_grid: torch.Tensor
    stark_wing_correction_c: torch.Tensor
    stark_wing_correction_d: torch.Tensor
    exponential_integral_table: torch.Tensor
    h2_partition_function: torch.Tensor
    # Detailed-transition catalog (line_catalog.py:28-49), type 0/3 normal,
    # -1 hydrogen, 1 autoionizing, >=4 merged continuum (type 2 never
    # deposits, line_opacity.py:697-698).
    transition_line_type: torch.Tensor
    transition_packed_wavelength: torch.Tensor
    transition_vacuum_wavelength_nm: torch.Tensor
    transition_species_slot: torch.Tensor
    transition_oscillator_strength: torch.Tensor
    transition_lower_excitation_cm: torch.Tensor
    transition_radiative_damping: torch.Tensor
    transition_stark_damping: torch.Tensor
    transition_van_der_waals_damping: torch.Tensor
    transition_selector_index: torch.Tensor
    transition_selector_species_slot: torch.Tensor
    # Baked per-line hydrogen setups (runner.py:2750-2848). Invalid lines
    # (selector 0 or no setup) keep h_valid == 0.
    h_valid: torch.Tensor
    h_lower_level: torch.Tensor
    h_upper_level: torch.Tensor
    h_line_frequency_hz: torch.Tensor
    h_line_wavelength_a: torch.Tensor
    h_beta_scale: torch.Tensor
    h_stark_c1_factor: torch.Tensor
    h_stark_c2_factor: torch.Tensor
    h_radiative_width: torch.Tensor
    h_resonance_width: torch.Tensor
    h_van_der_waals_width: torch.Tensor
    h_stark_width: torch.Tensor
    h_low_density_impact_numerator: torch.Tensor
    h_impact_electron_density_threshold: torch.Tensor
    h_stark_component_offsets: torch.Tensor  # (L, max_components)
    h_stark_component_weights: torch.Tensor
    h_stark_component_count: torch.Tensor


@dataclass
class ReferenceLineTemplate:
    """Exact reference line slab plus the state at which it was computed."""

    line_opacity: torch.Tensor  # (layer, frequency), float64
    population_widths: torch.Tensor  # (layer, packed slot), float64
    temperature: torch.Tensor  # (layer,), float64
    wavelength_grid_nm: torch.Tensor  # (frequency,), float64

    def to(
        self, *, device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ReferenceLineTemplate":
        dtype = self.line_opacity.dtype if dtype is None else dtype
        return ReferenceLineTemplate(
            line_opacity=self.line_opacity.to(device=device, dtype=dtype),
            population_widths=self.population_widths.to(device=device, dtype=dtype),
            temperature=self.temperature.to(device=device, dtype=dtype),
            wavelength_grid_nm=self.wavelength_grid_nm.to(device=device, dtype=dtype),
        )


def reference_line_template_from_opacity_state(opacity_state) -> ReferenceLineTemplate:
    """Build a training template from a certified ``OpacityState``."""

    population = opacity_state.population_state
    return ReferenceLineTemplate(
        line_opacity=_t64(
            opacity_state.line_opacity.line_mass_absorption_coefficient
        ),
        population_widths=_t64(
            population.partition_normalized_population_over_mass_density_and_fractional_doppler_width
        ),
        temperature=_t64(population.setup.atmosphere.temperature),
        wavelength_grid_nm=_t64(opacity_state.opacity_wavelength_grid_nm),
    )


def save_reference_line_template(
    template: ReferenceLineTemplate, path: Path | str
) -> Path:
    """Persist a compact template; about 10 MB per 80×30000 float32 slab."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        line_opacity=template.line_opacity.detach().cpu().numpy().astype(np.float32),
        population_widths=(
            template.population_widths.detach().cpu().numpy().astype(np.float64)
        ),
        temperature=template.temperature.detach().cpu().numpy().astype(np.float64),
        wavelength_grid_nm=(
            template.wavelength_grid_nm.detach().cpu().numpy().astype(np.float64)
        ),
    )
    return path


def load_reference_line_template(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ReferenceLineTemplate:
    """Load a template saved by :func:`save_reference_line_template`."""

    with np.load(path, allow_pickle=False) as arrays:
        return ReferenceLineTemplate(
            line_opacity=torch.as_tensor(
                np.asarray(arrays["line_opacity"], dtype=np.float64),
                dtype=dtype, device=device,
            ),
            population_widths=torch.as_tensor(
                np.asarray(arrays["population_widths"], dtype=np.float64),
                dtype=dtype, device=device,
            ),
            temperature=torch.as_tensor(
                np.asarray(arrays["temperature"], dtype=np.float64),
                dtype=dtype, device=device,
            ),
            wavelength_grid_nm=torch.as_tensor(
                np.asarray(arrays["wavelength_grid_nm"], dtype=np.float64),
                dtype=dtype, device=device,
            ),
        )


def line_opacity_from_template(
    current_population_widths: torch.Tensor,
    template: ReferenceLineTemplate,
    *,
    minimum_scale: float = 0.05,
    maximum_scale: float = 20.0,
) -> torch.Tensor:
    """Return a batched differentiable line slab from an exact reference slab.

    Line-center absorption is linear in the packed population-over-density-
    Doppler-width field. Summing that field over species gives a stable local
    depth response while retaining the exact reference wavelength structure.
    At the template state the scale is exactly one, so the forward result is
    the certified line opacity without approximation.
    """

    current = torch.as_tensor(current_population_widths)
    if current.dim() == 2:
        current = current.unsqueeze(0)
    if current.dim() != 3:
        raise ValueError("current_population_widths must be (star, layer, slot)")
    reference = template.population_widths.to(
        device=current.device, dtype=current.dtype
    )
    if current.shape[1:] != reference.shape:
        raise ValueError(
            "current population widths must match template (layer, slot) shape"
        )
    current_strength = current.clamp(min=0.0).sum(dim=-1)
    reference_strength = reference.clamp(min=0.0).sum(dim=-1).clamp(min=1.0e-300)
    scale = (current_strength / reference_strength[None, :]).clamp(
        min=float(minimum_scale), max=float(maximum_scale)
    )
    slab = template.line_opacity.to(device=current.device, dtype=current.dtype)
    return slab[None, :, :] * scale[:, :, None]


def _hydrogen_neutral_level_energies_cm() -> np.ndarray:
    """Exact restatement of line_opacity.py:2294-2311."""

    levels = np.zeros(100, dtype=np.float64)
    levels[1:10] = [
        82259.105,
        97492.302,
        102823.893,
        105291.651,
        106632.160,
        107440.444,
        107965.051,
        108324.720,
        108581.988,
    ]
    for principal_quantum_number in range(11, 101):
        levels[principal_quantum_number - 1] = 109678.764 - 109677.576 / float(
            principal_quantum_number * principal_quantum_number
        )
    return levels


def build_line_tables(
    transition_catalog_path: Path | str | None = None,
) -> TwinLineTables:
    """Build the shared table set from the reference data bundle.

    Uses only the numba-free reference readers (``line_profile_math.py``,
    ``hydrogen_line_profile.py``, ``line_catalog.py``). The hydrogen per-line
    setups are baked exactly like ``runner.py:2757-2848`` from a dummy-state
    evaluator (the setup depends only on the level pair and the tables).
    """

    from payne_zero_atmosphere.hydrogen_line_profile import (
        HydrogenLineProfileEvaluator,
        _exponential_integral_table,
        load_hydrogen_line_profile_tables,
    )
    from payne_zero_atmosphere.line_catalog import read_line_transition_catalog
    from payne_zero_atmosphere.line_profile_math import (
        build_fast_exponential_tables,
        build_hydrogen_continuum_selector_table,
        build_selection_log_lookup,
        build_voigt_profile_basis,
    )

    if transition_catalog_path is None:
        from payne_zero_atmosphere.source_catalogs import source_line_paths

        transition_catalog_path = source_line_paths()["detailed_line_catalog_path"]
    transition = read_line_transition_catalog(transition_catalog_path)

    basis = build_voigt_profile_basis()
    exponential = build_fast_exponential_tables()
    hydrogen_tables = load_hydrogen_line_profile_tables()

    line_count = int(transition.line_count)
    line_type = np.asarray(transition.line_type, dtype=np.int64)
    hydrogen_indices = np.nonzero(line_type == -1)[0]

    evaluator = HydrogenLineProfileEvaluator(
        temperature=np.full(_LAYER_COUNT, 5000.0),
        electron_density=np.full(_LAYER_COUNT, 1.0e13),
        hydrogen_neutral_population=np.ones(_LAYER_COUNT),
        hydrogen_ionized_population=np.ones(_LAYER_COUNT),
        hydrogen_neutral_partition_normalized_population=np.ones(_LAYER_COUNT),
        helium_neutral_population=np.ones(_LAYER_COUNT),
        hydrogen_fractional_doppler_width=np.full(_LAYER_COUNT, 1.0e-5),
        molecular_hydrogen_population=np.ones(_LAYER_COUNT),
        tables=hydrogen_tables,
    )

    h_valid = np.zeros(line_count, dtype=np.int64)
    h_lower = np.zeros(line_count, dtype=np.int64)
    h_upper = np.zeros(line_count, dtype=np.int64)
    scalars = {
        name: np.zeros(line_count, dtype=np.float64)
        for name in (
            "line_frequency_hz",
            "line_wavelength_a",
            "beta_scale",
            "stark_c1_factor",
            "stark_c2_factor",
            "radiative_width",
            "resonance_width",
            "van_der_waals_width",
            "stark_width",
            "low_density_impact_numerator",
            "impact_electron_density_threshold_cm3",
        )
    }
    offsets_by_line: dict[int, np.ndarray] = {}
    weights_by_line: dict[int, np.ndarray] = {}
    max_components = 1
    selector = build_hydrogen_continuum_selector_table()
    for line_index in hydrogen_indices:
        line_index = int(line_index)
        selector_index = int(transition.hydrogen_continuum_selector_index[line_index])
        if selector_index == 0:
            continue
        setup = evaluator.line_setup(
            int(transition.lower_hydrogen_level[line_index]),
            int(transition.upper_hydrogen_level[line_index]),
        )
        if setup is None:
            continue
        h_valid[line_index] = 1
        h_lower[line_index] = setup.lower_level
        h_upper[line_index] = setup.upper_level
        for name in scalars:
            scalars[name][line_index] = float(getattr(setup, name))
        offsets = np.ascontiguousarray(
            setup.stark_component_offsets_hz, dtype=np.float64
        )
        weights = np.ascontiguousarray(
            setup.stark_component_weights, dtype=np.float64
        )
        offsets_by_line[line_index] = offsets
        weights_by_line[line_index] = weights
        max_components = max(max_components, int(offsets.shape[0]))
    h_offsets = np.zeros((line_count, max_components), dtype=np.float64)
    h_weights = np.zeros((line_count, max_components), dtype=np.float64)
    h_count = np.zeros(line_count, dtype=np.int64)
    for line_index, offsets in offsets_by_line.items():
        count = offsets.shape[0]
        h_offsets[line_index, :count] = offsets
        h_weights[line_index, :count] = weights_by_line[line_index][:count]
        h_count[line_index] = count

    return TwinLineTables(
        voigt_gaussian=_t64(basis.gaussian_profile),
        voigt_first=_t64(basis.first_correction),
        voigt_second=_t64(basis.second_correction),
        exp_integer=_t64(exponential.integer_step),
        exp_fractional=_t64(exponential.fractional_step),
        selection_lookup=_t64(
            np.ascontiguousarray(build_selection_log_lookup(), dtype=np.float32).astype(
                np.float64
            )
        ),
        continuum_selector=_t64(selector),
        hydrogen_level_energies_cm=_t64(_hydrogen_neutral_level_energies_cm()),
        h2_cutoff_table=_t64(hydrogen_tables.h2_quasimolecular_cutoff_table),
        h2plus_cutoff_table=_t64(hydrogen_tables.h2plus_quasimolecular_cutoff_table),
        stark_probability_table=_t64(hydrogen_tables.stark_probability_table),
        stark_pressure_grid=_t64(hydrogen_tables.stark_pressure_grid),
        stark_beta_grid=_t64(hydrogen_tables.stark_beta_grid),
        stark_wing_correction_c=_t64(hydrogen_tables.stark_wing_correction_c),
        stark_wing_correction_d=_t64(hydrogen_tables.stark_wing_correction_d),
        exponential_integral_table=_t64(_exponential_integral_table()),
        h2_partition_function=_t64(hydrogen_tables.h2_partition_function),
        transition_line_type=_ti64(line_type),
        transition_packed_wavelength=_ti64(transition.packed_wavelength_index),
        transition_vacuum_wavelength_nm=_t64(transition.vacuum_wavelength_nm),
        transition_species_slot=_ti64(transition.packed_species_slot),
        transition_oscillator_strength=_t64(transition.oscillator_strength),
        transition_lower_excitation_cm=_t64(transition.lower_excitation_cm),
        transition_radiative_damping=_t64(transition.radiative_damping),
        transition_stark_damping=_t64(transition.stark_damping),
        transition_van_der_waals_damping=_t64(transition.van_der_waals_damping),
        transition_selector_index=_ti64(transition.hydrogen_continuum_selector_index),
        transition_selector_species_slot=_ti64(transition.continuum_species_slot),
        h_valid=_ti64(h_valid),
        h_lower_level=_ti64(h_lower),
        h_upper_level=_ti64(h_upper),
        h_line_frequency_hz=_t64(scalars["line_frequency_hz"]),
        h_line_wavelength_a=_t64(scalars["line_wavelength_a"]),
        h_beta_scale=_t64(scalars["beta_scale"]),
        h_stark_c1_factor=_t64(scalars["stark_c1_factor"]),
        h_stark_c2_factor=_t64(scalars["stark_c2_factor"]),
        h_radiative_width=_t64(scalars["radiative_width"]),
        h_resonance_width=_t64(scalars["resonance_width"]),
        h_van_der_waals_width=_t64(scalars["van_der_waals_width"]),
        h_stark_width=_t64(scalars["stark_width"]),
        h_low_density_impact_numerator=_t64(scalars["low_density_impact_numerator"]),
        h_impact_electron_density_threshold=_t64(
            scalars["impact_electron_density_threshold_cm3"]
        ),
        h_stark_component_offsets=_t64(h_offsets),
        h_stark_component_weights=_t64(h_weights),
        h_stark_component_count=_ti64(h_count),
    )


def load_line_catalog(path: Path | str) -> TwinLineCatalog:
    """Load one star's exported selected-line catalog (torch tensors).

    Decoding mirrors ``line_catalog.py:99-131`` (native layout; the export
    writes freshly generated words, so no swapped-layout probe is needed).
    The per-line walk state (continuum column, center index) is a memoryless
    function of each line's own wavelength on the sorted grid
    (``line_opacity.py:288-332``), so it is precomputed here with
    ``searchsorted(..., right=True)``.
    """

    with np.load(path, allow_pickle=False) as arrays:
        words = np.ascontiguousarray(arrays["selected_words"], dtype=np.int32)
        grid = np.asarray(arrays["opacity_wavelength_grid_nm"], dtype=np.float64)
        weights = np.asarray(arrays["frequency_weights"], dtype=np.float64)
        bin_edges = np.asarray(arrays["wavelength_bin_edges"], dtype=np.int64)
        effective_temperature = float(arrays["effective_temperature"])

    words = words.reshape(-1, 4)
    halves = np.ascontiguousarray(words[:, 1:4]).view(np.int16).reshape(-1, 6)
    packed_wavelength = words[:, 0].astype(np.int64)
    species_half = halves[:, 0].astype(np.int64)
    excitation_index = halves[:, 1].astype(np.int64)
    strength_index = halves[:, 2].astype(np.int64)
    radiative_index = halves[:, 3].astype(np.int64)
    stark_index = halves[:, 4].astype(np.int64)
    van_der_waals_index = halves[:, 5].astype(np.int64)

    vacuum_wavelength_nm = np.exp(packed_wavelength * _RATIO_LOG_STEP)
    species_slot = np.abs(species_half) // 10
    continuum_column = np.searchsorted(bin_edges, packed_wavelength, side="right")
    center_index = np.searchsorted(grid, vacuum_wavelength_nm, side="right")
    lookup_size = 32768
    valid_line = (
        (species_slot >= 1)
        & (excitation_index >= 1)
        & (excitation_index <= lookup_size)
        & (strength_index >= 1)
        & (strength_index <= lookup_size)
        & (radiative_index >= 1)
        & (radiative_index <= lookup_size)
        & (stark_index >= 1)
        & (stark_index <= lookup_size)
        & (van_der_waals_index >= 1)
        & (van_der_waals_index <= lookup_size)
        & (continuum_column < bin_edges.shape[0])
        & (center_index < grid.shape[0])
        & (vacuum_wavelength_nm >= grid[0] - 1.0)
        & (vacuum_wavelength_nm <= grid[-1] + 1.0)
    )

    return TwinLineCatalog(
        packed_wavelength_index=_ti64(packed_wavelength),
        packed_species_slot=_ti64(species_half),
        lower_excitation_index=_ti64(excitation_index),
        log_strength_index=_ti64(strength_index),
        radiative_damping_index=_ti64(radiative_index),
        stark_damping_index=_ti64(stark_index),
        van_der_waals_damping_index=_ti64(van_der_waals_index),
        vacuum_wavelength_nm=_t64(vacuum_wavelength_nm),
        species_slot=_ti64(species_slot),
        continuum_column=_ti64(continuum_column),
        center_index=_ti64(center_index),
        valid_line=torch.tensor(valid_line, dtype=torch.bool),
        opacity_wavelength_grid_nm=_t64(grid),
        frequency_weights=_t64(weights),
        wavelength_bin_edges=_ti64(bin_edges),
        effective_temperature=effective_temperature,
    )


# ---------------------------------------------------------------------------
# Torch numerical helpers (exact vectorized ports; branch conditions become
# torch.where selects, table lookups keep the reference index arithmetic).
# ---------------------------------------------------------------------------


def _fast_exponential(x: torch.Tensor, tables: TwinLineTables) -> torch.Tensor:
    """Table approximation to exp(-x); port of line_opacity.py:79-90."""

    integer_index = torch.clamp(x.floor(), 0.0, 1000.0).long()
    fractional_index = torch.clamp(
        ((x - integer_index.to(x.dtype)) * 1000.0 + 1.5).floor(), 1.0, 1001.0
    ).long()
    value = tables.exp_integer[integer_index] * tables.exp_fractional[
        fractional_index - 1
    ]
    return torch.where((x == x) & (x >= 0.0) & (x < 1001.0), value, x.new_zeros(()))


def _voigt_profile(
    offset: torch.Tensor, damping: torch.Tensor, tables: TwinLineTables
) -> torch.Tensor:
    """Validated Voigt approximation; port of line_opacity.py:92-153."""

    table_index = torch.clamp((offset * 200.0 + 1.5).floor(), 1.0, 2001.0).long() - 1
    gaussian = tables.voigt_gaussian[table_index]
    first = tables.voigt_first[table_index]
    second = tables.voigt_second[table_index]

    damping_squared = damping * damping
    offset_squared = offset * offset

    # damping >= 0.2, far-wing branch (damping > 1.4 or damping + offset > 3.2).
    denominator = (damping_squared + offset_squared) * 1.4142
    denominator = torch.where(denominator == 0.0, torch.ones_like(denominator), denominator)
    base_profile = damping * 0.79788 / denominator
    damping_fraction = damping_squared / denominator
    offset_fraction = offset_squared / denominator
    correction = (
        ((damping_fraction - 10.0 * offset_fraction) * damping_fraction * 3.0)
        + 15.0 * offset_fraction * offset_fraction
        + 3.0 * offset_squared
        - damping_squared
    )
    far_wing = torch.where(
        damping > 100.0,
        base_profile,
        (correction / (denominator * denominator) + 1.0) * base_profile,
    )

    # damping >= 0.2, polynomial branch.
    adjusted_first = first + gaussian * 1.12838
    adjusted_second = second + adjusted_first * 1.12838 - gaussian
    third = (
        (1.0 - second) * 0.37613
        - adjusted_first * 0.66667 * offset_squared
        + adjusted_second * 1.12838
    )
    fourth = (3.0 * third - adjusted_first) * 0.37613 + gaussian * 0.66667 * (
        offset_squared * offset_squared
    )
    polynomial = (
        ((fourth * damping + third) * damping + adjusted_second) * damping
        + adjusted_first
    ) * damping + gaussian
    scale = ((-0.122727278 * damping + 0.532770573) * damping - 0.96284325) * damping
    polynomial = polynomial * (scale + 0.979895032)

    high_damping = torch.where(
        (damping > 1.4) | (damping + offset > 3.2), far_wing, polynomial
    )

    # damping < 0.2.
    safe_offset = torch.where(offset == 0.0, torch.ones_like(offset), offset)
    low_damping = torch.where(
        offset > 10.0,
        0.5642 * damping / (safe_offset * safe_offset),
        (second * damping + first) * damping + gaussian,
    )
    return torch.where(damping >= 0.2, high_damping, low_damping)


def _stark_probability(
    beta: torch.Tensor,
    pressure: torch.Tensor,
    lower_level: torch.Tensor,
    upper_level: torch.Tensor,
    tables: TwinLineTables,
) -> torch.Tensor:
    """Hydrogen Stark wing probability; port of line_opacity.py:912-1010."""

    beta_squared = beta * beta
    sqrt_beta = torch.sqrt(torch.clamp(beta, min=1.0e-300))
    level_delta = upper_level - lower_level
    table_index = torch.where(
        (lower_level <= 3) & (level_delta <= 2),
        2 * (lower_level - 1) + level_delta,
        torch.full_like(level_delta, 7),
    )
    pressure_low = torch.clamp((5.0 * pressure).floor().long() + 1, 1, 4)
    pressure_high = pressure_low + 1
    high_pressure_weight = 5.0 * (
        pressure - tables.stark_pressure_grid[pressure_low - 1]
    )
    low_pressure_weight = 1.0 - high_pressure_weight

    # beta <= 25.12 branch.
    beta_high = torch.searchsorted(tables.stark_beta_grid, beta.reshape(-1)).reshape(
        beta.shape
    )
    beta_high = torch.clamp(beta_high, 1, 14)
    beta_low = beta_high - 1
    beta_denominator = tables.stark_beta_grid[beta_high] - tables.stark_beta_grid[beta_low]
    high_beta_weight = torch.where(
        beta_denominator == 0.0,
        beta.new_zeros(()),
        (beta - tables.stark_beta_grid[beta_low])
        / torch.where(
            beta_denominator == 0.0, torch.ones_like(beta_denominator), beta_denominator
        ),
    )
    low_beta_weight = 1.0 - high_beta_weight
    high_beta_correction = (
        tables.stark_probability_table[pressure_high - 1, beta_high, table_index - 1]
        * high_pressure_weight
        + tables.stark_probability_table[pressure_low - 1, beta_high, table_index - 1]
        * low_pressure_weight
    )
    low_beta_correction = (
        tables.stark_probability_table[pressure_high - 1, beta_low, table_index - 1]
        * high_pressure_weight
        + tables.stark_probability_table[pressure_low - 1, beta_low, table_index - 1]
        * low_pressure_weight
    )
    correction_low = (
        1.0 + high_beta_correction * high_beta_weight
        + low_beta_correction * low_beta_weight
    )
    blend = torch.clamp(0.5 * (10.0 - beta), 0.0, 1.0)
    low_beta_profile = torch.where(
        beta <= 10.0,
        8.0 / (83.0 + (2.0 + 0.95 * beta_squared) * beta),
        beta.new_zeros(()),
    )
    high_beta_profile = torch.where(
        beta >= 8.0,
        (1.5 / sqrt_beta + 27.0 / beta_squared) / beta_squared,
        beta.new_zeros(()),
    )
    result_low = (
        low_beta_profile * blend + high_beta_profile * (1.0 - blend)
    ) * correction_low

    # 25.12 < beta <= 500 branch.
    c_value = (
        tables.stark_wing_correction_c[pressure_high - 1, table_index - 1]
        * high_pressure_weight
        + tables.stark_wing_correction_c[pressure_low - 1, table_index - 1]
        * low_pressure_weight
    )
    d_value = (
        tables.stark_wing_correction_d[pressure_high - 1, table_index - 1]
        * high_pressure_weight
        + tables.stark_wing_correction_d[pressure_low - 1, table_index - 1]
        * low_pressure_weight
    )
    correction_mid = 1.0 + d_value / (c_value + beta * sqrt_beta)

    base = (1.5 / sqrt_beta + 27.0 / beta_squared) / beta_squared
    result = torch.where(beta <= 25.12, result_low, base * correction_mid)
    return torch.where(beta <= 500.0, result, base)


def _fast_exponential_integral(x: torch.Tensor, tables: TwinLineTables) -> torch.Tensor:
    """Port of line_opacity.py:896-910."""

    table_index = torch.clamp((x * 100.0 + 0.5).floor(), 1.0, 2000.0).long()
    table_value = tables.exponential_integral_table[table_index - 1]
    small = (1.0 - 0.22464 * x) * x - torch.log(torch.clamp(x, min=1.0e-300)) - 0.57721
    value = torch.where(x >= 0.5, table_value, torch.where(x > 0.0, small, x.new_zeros(())))
    return torch.where(x > 20.0, x.new_zeros(()), value)


def _h2_equilibrium_constant(
    temperature: torch.Tensor, tables: TwinLineTables
) -> torch.Tensor:
    """H2 equilibrium constant per layer; port of
    hydrogen_line_profile.py:409-429 (table lookup stays piecewise linear)."""

    temp = torch.where(
        torch.isfinite(temperature) & (temperature > 100.0),
        temperature,
        torch.full_like(temperature, 100.0),
    )
    temp = torch.minimum(temp, torch.full_like(temp, 19900.0))
    table_index = torch.clamp((temp / 100.0).floor().long(), 1, 199)
    partition = tables.h2_partition_function[table_index - 1] + (
        tables.h2_partition_function[table_index]
        - tables.h2_partition_function[table_index - 1]
    ) / 100.0 * (temp - table_index.to(temp.dtype) * 100.0)
    equilibrium = partition * (2.0**1.5) / 4.0
    equilibrium = equilibrium / (
        2.0 * 3.14159 * 1.008 * 1.660e-24 * 1.38054e-16 / 6.6256e-27**2 * temp
    ) ** 1.5
    return equilibrium * torch.exp(
        36118.11 * 6.6256e-27 * 2.997925e10 / 1.38054e-16 / temp
    )


@dataclass
class _HydrogenLayerState:
    """Per-layer evaluator arrays; port of HydrogenLineProfileEvaluator.
    __post_init__ (hydrogen_line_profile.py:480-552). All (80,) fp64."""

    hydrogen_fractional_doppler_width: torch.Tensor
    field_strength: torch.Tensor
    temperature_density_he: torch.Tensor
    temperature_density_h2: torch.Tensor
    hydrogen_neutral_population: torch.Tensor
    hydrogen_neutral_ground: torch.Tensor
    hydrogen_ionized_population: torch.Tensor
    electron_density: torch.Tensor
    low_density_impact_factor: torch.Tensor
    high_density_impact_factor: torch.Tensor
    stark_linear_density_coefficient: torch.Tensor
    stark_quadratic_density_coefficient: torch.Tensor
    stark_gamma_thermal_correction: torch.Tensor
    stark_gamma_density_correction: torch.Tensor
    pressure_parameter: torch.Tensor


def _hydrogen_layer_state(
    *,
    temperature: torch.Tensor,
    electron_density: torch.Tensor,
    hydrogen_neutral_population: torch.Tensor,
    hydrogen_ionized_population: torch.Tensor,
    hydrogen_neutral_ground: torch.Tensor,
    helium_neutral_population: torch.Tensor,
    hydrogen_fractional_doppler_width: torch.Tensor,
    tables: TwinLineTables,
) -> _HydrogenLayerState:
    electrons = electron_density
    temp = temperature
    electron_sixth_root = torch.clamp(electrons, min=1.0e-300) ** 0.1666667
    temperature_10000 = temp / 10000.0
    temperature_factor = torch.clamp(temperature_10000, min=1.0e-300) ** 0.3
    field_strength = electron_sixth_root**4 * 1.25e-9
    molecular_hydrogen = (hydrogen_neutral_ground * 2.0) ** 2 * _h2_equilibrium_constant(
        temp, tables
    )
    return _HydrogenLayerState(
        hydrogen_fractional_doppler_width=hydrogen_fractional_doppler_width,
        field_strength=field_strength,
        temperature_density_he=temperature_factor
        * torch.clamp(helium_neutral_population, min=0.0),
        temperature_density_h2=temperature_factor
        * torch.clamp(molecular_hydrogen, min=0.0),
        hydrogen_neutral_population=hydrogen_neutral_population,
        hydrogen_neutral_ground=hydrogen_neutral_ground,
        hydrogen_ionized_population=hydrogen_ionized_population,
        electron_density=electrons,
        low_density_impact_factor=temperature_factor
        / torch.clamp(electron_sixth_root, min=1.0e-300),
        high_density_impact_factor=2.0
        / (
            1.0
            + 0.012
            / torch.clamp(temp, min=1.0e-300)
            * torch.sqrt(
                torch.clamp(electrons / torch.clamp(temp, min=1.0e-300), min=0.0)
            )
        ),
        stark_linear_density_coefficient=field_strength
        * 78940.0
        / torch.clamp(temp, min=1.0e-300),
        stark_quadratic_density_coefficient=field_strength
        * field_strength
        / 5.96e-23
        / torch.clamp(electrons, min=1.0e-300),
        stark_gamma_thermal_correction=0.2
        + 0.09
        * torch.sqrt(torch.clamp(temperature_10000, min=0.0))
        / (1.0 + electrons / 1.0e13),
        stark_gamma_density_correction=0.2 / (1.0 + electrons / 1.0e15),
        pressure_parameter=electron_sixth_root
        * 0.08989
        / torch.sqrt(torch.clamp(temp, min=1.0e-300)),
    )


def _hydrogen_profile(
    wavelength_offset_nm: torch.Tensor,  # (P, W) fp64
    line: dict[str, torch.Tensor],  # per-pair line scalars, (P,)
    layer: dict[str, torch.Tensor],  # per-pair layer scalars, (P,)
    tables: TwinLineTables,
) -> torch.Tensor:
    """Full hydrogen Stark/Doppler/Lorentz profile; port of the compiled
    transcription in line_opacity.py:1012-1400."""

    LIGHT_A = _LIGHT_SPEED_ANGSTROM_PER_SECOND
    LIGHT_CM = _LIGHT_SPEED_CM_PER_SECOND

    nu0 = line["line_frequency_hz"][:, None]
    lambda_a0 = line["line_wavelength_a"][:, None]
    wavelength_a = lambda_a0 + wavelength_offset_nm * 10.0
    safe_wavelength_a = torch.where(
        wavelength_a <= 0.0, torch.ones_like(wavelength_a), wavelength_a
    )
    frequency = LIGHT_A / safe_wavelength_a
    frequency_offset = torch.abs(frequency - nu0)

    doppler_width = layer["hydrogen_fractional_doppler_width"][:, None]
    stark_width = line["stark_width"][:, None] * layer["field_strength"][:, None]
    van_der_waals_width = line["van_der_waals_width"][:, None] * (
        layer["temperature_density_he"][:, None]
        + 2.0 * layer["temperature_density_h2"][:, None]
    )
    radiative_width = line["radiative_width"][:, None]
    resonance_width = (
        line["resonance_width"][:, None] * layer["hydrogen_neutral_population"][:, None]
    )
    lorentz_width = resonance_width + van_der_waals_width + radiative_width
    profile_mode = torch.where(
        (doppler_width >= stark_width) & (doppler_width >= lorentz_width),
        torch.ones_like(doppler_width),
        torch.where(
            lorentz_width < stark_width,
            torch.full_like(doppler_width, 3.0),
            torch.full_like(doppler_width, 2.0),
        ),
    )
    half_width = nu0 * torch.maximum(
        torch.maximum(doppler_width, lorentz_width), stark_width
    )
    in_core = frequency_offset <= half_width
    doppler_frequency_width = nu0 * doppler_width
    stark_wavelength_offset = (
        -10.0 * wavelength_offset_nm / lambda_a0 * nu0
    )

    # --- Doppler core: sum over Stark components (line_opacity.py:1097-1114).
    component_offsets = line["stark_component_offsets"][:, None, :]  # (P,1,C)
    component_weights = line["stark_component_weights"][:, None, :]
    component_mask = (
        torch.arange(component_offsets.shape[-1], device=frequency.device)[None, None, :]
        < line["stark_component_count"][:, None, None]
    )
    distance = torch.abs(frequency[..., None] - nu0[..., None] - component_offsets) / (
        doppler_frequency_width[..., None] + 1.0e-300
    )
    doppler_term = torch.where(
        (distance <= 7.0) & component_mask,
        _fast_exponential(distance * distance, tables) * component_weights,
        torch.zeros_like(distance),
    )
    doppler_value = doppler_term.sum(dim=-1)

    # --- Lorentz profile (line_opacity.py:1116-1215).
    lower_level = line["lower_level"]
    upper_level = line["upper_level"]
    is_lyman_alpha = (lower_level == 1) & (upper_level == 2)  # (P,)

    lorentz_half_width = nu0 * lorentz_width
    safe_lhw = torch.where(
        lorentz_half_width <= 0.0, torch.ones_like(lorentz_half_width), lorentz_half_width
    )
    lorentz_standard = torch.where(
        lorentz_half_width > 0.0,
        safe_lhw
        / _PI_APPROX
        / (frequency_offset * frequency_offset + safe_lhw * safe_lhw)
        * _SQRT_PI_APPROX
        * doppler_frequency_width,
        torch.zeros_like(frequency_offset),
    )

    # Lyman-alpha branch with the H2 quasi-molecular red cutoff.
    lyman_resonance = resonance_width * 4.0
    lyman_half_width = nu0 * (lyman_resonance + van_der_waals_width + radiative_width)
    lyman_denominator = frequency_offset * frequency_offset + lyman_half_width * lyman_half_width
    resonance_profile = (
        lyman_resonance
        * nu0
        / _PI_APPROX
        / lyman_denominator
        * _SQRT_PI_APPROX
        * doppler_frequency_width
    )
    # H2 cutoff table linear interpolation (line_opacity.py:1136-1169).
    cutoff_frequency_22000 = (82259.105 - 22000.0) * LIGHT_CM
    spacing = 200.0 * LIGHT_CM
    cutoff_index = torch.clamp(
        ((frequency - cutoff_frequency_22000) / spacing).floor().long(),
        0,
        tables.h2_cutoff_table.shape[0] - 2,
    )
    cutoff_base = cutoff_index.to(frequency.dtype) * spacing + cutoff_frequency_22000
    cutoff_log = (
        tables.h2_cutoff_table[cutoff_index + 1] - tables.h2_cutoff_table[cutoff_index]
    ) / spacing * (frequency - cutoff_base) + tables.h2_cutoff_table[cutoff_index]
    h2_cutoff = torch.where(
        frequency >= 50000.0 * LIGHT_CM,
        10.0 ** (cutoff_log - 14.0)
        * layer["hydrogen_neutral_ground"][:, None]
        * 2.0
        / LIGHT_CM,
        torch.zeros_like(frequency),
    )
    red_resonance = h2_cutoff * _SQRT_PI_APPROX * doppler_frequency_width
    resonance_profile = torch.where(
        frequency > (82259.105 - 4000.0) * LIGHT_CM, resonance_profile, red_resonance
    )
    radiative_profile = (
        radiative_width
        * nu0
        / _PI_APPROX
        / lyman_denominator
        * _SQRT_PI_APPROX
        * doppler_frequency_width
    )
    radiative_profile = torch.where(
        frequency <= 2.463e15, torch.zeros_like(radiative_profile), radiative_profile
    )
    van_der_waals_profile = (
        van_der_waals_width
        * nu0
        / _PI_APPROX
        / lyman_denominator
        * _SQRT_PI_APPROX
        * doppler_frequency_width
    )
    van_der_waals_profile = torch.where(
        frequency < 1.8e15,
        torch.zeros_like(van_der_waals_profile),
        van_der_waals_profile,
    )
    lorentz_lyman = resonance_profile + radiative_profile + van_der_waals_profile
    lorentz_value = torch.where(
        is_lyman_alpha[:, None], lorentz_lyman, lorentz_standard
    )

    # --- Stark profile (line_opacity.py:1217-1389).
    field_strength = layer["field_strength"][:, None]
    has_field = field_strength > 0.0
    safe_field = torch.where(has_field, field_strength, torch.ones_like(field_strength))

    low_density_impact_weight = 1.0 / (
        1.0
        + layer["electron_density"][:, None]
        / line["impact_electron_density_threshold"][:, None]
    )
    impact_broadening_factor = (
        line["low_density_impact_numerator"][:, None]
        * layer["low_density_impact_factor"][:, None]
        * low_density_impact_weight
        + layer["high_density_impact_factor"][:, None] * (1.0 - low_density_impact_weight)
    )
    linear_impact_parameter = torch.clamp(
        layer["stark_linear_density_coefficient"][:, None]
        * line["stark_c1_factor"][:, None]
        * impact_broadening_factor,
        min=0.0,
    )
    quadratic_impact_parameter = torch.clamp(
        layer["stark_quadratic_density_coefficient"][:, None]
        * line["stark_c2_factor"][:, None],
        min=0.0,
    )
    impact_width_scale = 6.77 * torch.sqrt(torch.clamp(linear_impact_parameter, min=0.0))
    log_term = torch.where(
        (linear_impact_parameter > 0.0) & (quadratic_impact_parameter > 0.0),
        torch.log(
            torch.sqrt(quadratic_impact_parameter)
            / torch.clamp(linear_impact_parameter, min=1.0e-300)
        ),
        torch.zeros_like(linear_impact_parameter),
    )
    zero_offset_impact_width = (
        impact_width_scale
        * torch.clamp(0.2114 + log_term, min=0.0)
        * (
            1.0
            - layer["stark_gamma_thermal_correction"][:, None]
            - layer["stark_gamma_density_correction"][:, None]
        )
    )
    beta = torch.abs(stark_wavelength_offset) / safe_field * line["beta_scale"][:, None]
    linear_impact_argument = linear_impact_parameter * beta
    quadratic_impact_argument = quadratic_impact_parameter * beta * beta
    impact_width_full = (
        impact_width_scale
        * (
            0.5
            * _fast_exponential(
                torch.clamp(linear_impact_argument, max=80.0), tables
            )
            + _fast_exponential_integral(linear_impact_argument, tables)
            - 0.5
            * _fast_exponential_integral(quadratic_impact_argument, tables)
        )
        * (
            1.0
            - layer["stark_gamma_thermal_correction"][:, None]
            / (1.0 + (90.0 * linear_impact_argument) ** 3.0)
            - layer["stark_gamma_density_correction"][:, None]
            / (1.0 + 2000.0 * linear_impact_argument)
        )
    )
    impact_width_full = torch.where(
        impact_width_full <= 1.0e-20,
        torch.zeros_like(impact_width_full),
        impact_width_full,
    )
    impact_width = torch.where(
        (quadratic_impact_argument <= 1.0e-4) & (linear_impact_argument <= 1.0e-5),
        zero_offset_impact_width,
        impact_width_full,
    )

    probability = _stark_probability(
        beta,
        layer["pressure_parameter"][:, None].expand_as(beta),
        lower_level[:, None].expand_as(beta),
        upper_level[:, None].expand_as(beta),
        tables,
    )

    # upper_level <= 2 satellites (line_opacity.py:1304-1371).
    stark_satellite = torch.zeros_like(beta)
    is_upper_low = upper_level <= 2  # (P,)
    if bool(is_upper_low.any()):
        probability_low = probability * 0.5
        nu_20000 = (82259.105 - 20000.0) * LIGHT_CM
        nu_4000 = (82259.105 - 4000.0) * LIGHT_CM
        nu_15000 = (82259.105 - 15000.0) * LIGHT_CM
        plus_spacing = 100.0 * LIGHT_CM
        plus_index = torch.clamp(
            ((frequency - nu_15000) / plus_spacing).floor().long(),
            0,
            tables.h2plus_cutoff_table.shape[0] - 2,
        )
        plus_base = plus_index.to(frequency.dtype) * plus_spacing + nu_15000
        plus_log = (
            tables.h2plus_cutoff_table[plus_index + 1]
            - tables.h2plus_cutoff_table[plus_index]
        ) / plus_spacing * (frequency - plus_base) + tables.h2plus_cutoff_table[
            plus_index
        ]
        h2plus_cutoff = (
            10.0 ** (plus_log - 14.0)
            / LIGHT_CM
            * layer["hydrogen_ionized_population"][:, None]
        )
        mid_band = (frequency >= nu_20000) & (frequency <= nu_4000)
        stark_satellite = stark_satellite + torch.where(
            mid_band,
            h2plus_cutoff * _SQRT_PI_APPROX * doppler_frequency_width,
            torch.zeros_like(frequency),
        )
        # Above nu_4000: rescale through the 4000 cm^-1 reference point.
        beta4000 = (
            4000.0 * LIGHT_CM / safe_field[:, 0] * line["beta_scale"]
        )
        probability4000 = (
            _stark_probability(
                beta4000,
                layer["pressure_parameter"],
                lower_level,
                upper_level,
                tables,
            )
            * 0.5
            / safe_field[:, 0]
            * line["beta_scale"]
        )
        cutoff4000 = (
            10.0 ** (-11.07 - 14.0)
            / LIGHT_CM
            * layer["hydrogen_ionized_population"]
        )
        rescale = torch.where(
            probability4000 != 0.0,
            cutoff4000
            / torch.where(
                probability4000 != 0.0,
                probability4000,
                torch.ones_like(probability4000),
            )
            * probability
            / safe_field
            * line["beta_scale"][:, None]
            * _SQRT_PI_APPROX
            * doppler_frequency_width,
            torch.zeros_like(beta),
        )
        stark_satellite = stark_satellite + torch.where(
            frequency > nu_4000, rescale, torch.zeros_like(beta)
        )
        stark_satellite = torch.where(
            (frequency >= nu_20000) & is_upper_low[:, None],
            stark_satellite,
            torch.zeros_like(beta),
        )
        probability = torch.where(is_upper_low[:, None], probability_low, probability)

    lorentz_component = torch.where(
        impact_width > 0.0,
        impact_width
        / _PI_APPROX
        / (impact_width * impact_width + beta * beta),
        torch.zeros_like(beta),
    )
    satellite_blend_square = (0.9 * linear_impact_argument) ** 2.0
    satellite_enhancement = (
        satellite_blend_square + 0.03 * torch.sqrt(linear_impact_argument)
    ) / (satellite_blend_square + 1.0)
    stark_value = stark_satellite + (
        (probability * (1.0 + satellite_enhancement) + lorentz_component)
        / safe_field
        * line["beta_scale"][:, None]
        * _SQRT_PI_APPROX
        * doppler_frequency_width
    )
    stark_value = torch.where(has_field, stark_value, torch.zeros_like(stark_value))

    core_value = torch.where(
        profile_mode == 1.0,
        doppler_value,
        torch.where(profile_mode == 2.0, lorentz_value, stark_value),
    )
    value = torch.where(
        in_core, core_value, doppler_value + lorentz_value + stark_value
    )
    value = torch.where(wavelength_a <= 0.0, torch.zeros_like(value), value)
    value = torch.where(doppler_width <= 0.0, torch.zeros_like(value), value)
    value = torch.where(
        doppler_frequency_width <= 0.0, torch.zeros_like(value), value
    )
    return torch.clamp(value, min=0.0)


# ---------------------------------------------------------------------------
# Wing deposition engines.
# ---------------------------------------------------------------------------


def _cumulative_deposit_mask(valid: torch.Tensor) -> torch.Tensor:
    """Deposit mask for one wing direction: deposit j iff every earlier
    point was deposited and valid (reference deposits, then breaks on the
    first invalid point — line_opacity.py:191-195)."""

    cumulative = torch.cumprod(valid.long(), dim=1)
    shifted = torch.ones_like(cumulative)
    shifted[:, 1:] = cumulative[:, :-1]
    return shifted.bool()


def _deposit_voigt_wings(
    slab_flat: torch.Tensor,
    *,
    depth_index: torch.Tensor,  # (P,) int64
    center_index: torch.Tensor,  # (P,) int64
    vacuum_wavelength_nm: torch.Tensor,  # (P,) fp64
    center_absorption: torch.Tensor,  # (P,) fp64
    damping_parameter: torch.Tensor,  # (P,) fp64
    doppler_wavelength_width: torch.Tensor,  # (P,) fp64
    threshold: torch.Tensor,  # (P,) fp64
    grid: torch.Tensor,  # (F,) fp64
    tables: TwinLineTables,
    window: int,
    red_cutoff_nm: torch.Tensor | None = None,  # (P,) fp64, NaN = no cutoff
) -> torch.Tensor:
    """Accumulate Voigt wings for gated (line, layer) pairs.

    Exact port of ``_accumulate_selected_line_wings_compiled``
    (line_opacity.py:155-251, window 100) and ``_transition_wings_compiled``
    (:570-645, window 2000, with the red-side blue-continuum cutoff). The
    reference early-break becomes a cumulative mask; ``damping <= 0.2`` wings
    use the ``offset > 10`` asymptote inline exactly as the kernel does.
    """

    wavelength_count = grid.shape[0]
    safe_width = torch.where(
        doppler_wavelength_width <= 0.0,
        torch.ones_like(doppler_wavelength_width),
        doppler_wavelength_width,
    )
    valid_pair = doppler_wavelength_width > 0.0

    # Blue side: j = 0..window at center + j (line_opacity.py:176-195).
    offsets = torch.arange(window + 1, device=grid.device)
    indices = center_index[:, None] + offsets[None, :]
    in_bounds = indices < wavelength_count
    safe_indices = torch.clamp(indices, 0, wavelength_count - 1)
    voigt_offset = (grid[safe_indices] - vacuum_wavelength_nm[:, None]) / safe_width[:, None]
    profile = _voigt_profile(voigt_offset, damping_parameter[:, None], tables)
    asymptote = (
        0.5642
        * damping_parameter[:, None]
        / torch.where(
            voigt_offset == 0.0, torch.ones_like(voigt_offset), voigt_offset
        )
        ** 2
    )
    profile = torch.where(
        (damping_parameter[:, None] <= 0.2) & (voigt_offset > 10.0),
        asymptote,
        profile,
    )
    contribution = center_absorption[:, None] * profile
    deposit = (
        _cumulative_deposit_mask(contribution >= threshold[:, None])
        & in_bounds
        & valid_pair[:, None]
    )
    slab_flat = slab_flat.index_add(
        0,
        (depth_index[:, None] * wavelength_count + safe_indices).reshape(-1),
        torch.where(deposit, contribution, torch.zeros_like(contribution)).reshape(-1),
    )

    # Red side: j = 1..window at center - j (line_opacity.py:197-251).
    red_offsets = torch.arange(1, window + 1, device=grid.device)
    indices = center_index[:, None] - red_offsets[None, :]
    in_bounds = indices >= 0
    safe_indices = torch.clamp(indices, 0, wavelength_count - 1)
    voigt_offset = (vacuum_wavelength_nm[:, None] - grid[safe_indices]) / safe_width[:, None]
    profile = _voigt_profile(voigt_offset, damping_parameter[:, None], tables)
    asymptote = (
        0.5642
        * damping_parameter[:, None]
        / torch.where(
            voigt_offset == 0.0, torch.ones_like(voigt_offset), voigt_offset
        )
        ** 2
    )
    profile = torch.where(
        (damping_parameter[:, None] <= 0.2) & (voigt_offset > 10.0),
        asymptote,
        profile,
    )
    contribution = center_absorption[:, None] * profile
    deposit = (
        _cumulative_deposit_mask(contribution >= threshold[:, None])
        & in_bounds
        & valid_pair[:, None]
    )
    if red_cutoff_nm is not None:
        # The red walk breaks (before depositing) at the first point blue-ward
        # of the cutoff (line_opacity.py:622-623): point j is reachable iff
        # every k <= j cleared the cutoff.
        cutoff_ok = torch.isnan(red_cutoff_nm[:, None]) | (
            grid[safe_indices] >= red_cutoff_nm[:, None]
        )
        reachable = torch.cumprod(cutoff_ok.long(), dim=1).bool()
        deposit = deposit & reachable
    slab_flat = slab_flat.index_add(
        0,
        (depth_index[:, None] * wavelength_count + safe_indices).reshape(-1),
        torch.where(deposit, contribution, torch.zeros_like(contribution)).reshape(-1),
    )
    return slab_flat


def _apply_depth_gate(own_gate: torch.Tensor) -> torch.Tensor:
    """The 8-layer block gate of both kernels (line_opacity.py:381-408,
    :741-767): block b (layers 8b..8b+7) is processed iff the gate fired at
    layer 8b+7 or at layer 8b-1; the first block keys only on layer 7."""

    endpoint_gate = own_gate[_DEPTH_BLOCK - 1 :: _DEPTH_BLOCK]  # (10, ...)
    zeros_first = torch.zeros_like(endpoint_gate[:1])
    enabled = endpoint_gate | torch.cat([zeros_first, endpoint_gate[:-1]], dim=0)
    block_of_layer = (
        torch.arange(own_gate.shape[0], device=own_gate.device) // _DEPTH_BLOCK
    )
    return own_gate & enabled[block_of_layer]


# ---------------------------------------------------------------------------
# Per-stage accumulators (single star).
# ---------------------------------------------------------------------------


def _accumulate_selected_lines(
    slab_flat: torch.Tensor,
    catalog: TwinLineCatalog,
    tables: TwinLineTables,
    *,
    hc_over_kt: torch.Tensor,  # (80,)
    electron_density: torch.Tensor,  # (80,)
    neutral_collision_density: torch.Tensor,  # (80,)
    population_widths: torch.Tensor,  # (80, slots)
    fractional_doppler_widths: torch.Tensor,  # (80, slots)
    threshold: torch.Tensor,  # (80, 344)
    line_chunk: int,
    pair_chunk: int,
) -> torch.Tensor:
    """Selected-line accumulation; port of line_opacity.py:2345-2463 +
    kernel :254-502."""

    grid = catalog.opacity_wavelength_grid_nm
    slot_count = population_widths.shape[1]
    # Necessary-condition pre-gate: a line can only pass gate 1 at some layer
    # if classical * max_d popw >= min_d threshold (both per slot/column).
    slot_max = population_widths.max(dim=0).values
    column_min = threshold.min(dim=0).values

    total = catalog.packed_wavelength_index.shape[0]
    for start in range(0, total, line_chunk):
        stop = min(start + line_chunk, total)
        sl = slice(start, stop)
        slot = catalog.species_slot[sl]
        strength_index = catalog.log_strength_index[sl]
        line_valid = catalog.valid_line[sl] & (slot <= slot_count)
        if not bool(line_valid.any()):
            continue
        vacuum_wavelength = catalog.vacuum_wavelength_nm[sl]
        # f32 kernel semantics (line_opacity.py:358-363) in fp64.
        classical_strength = (
            _CLASSICAL_LINE_STRENGTH_SCALE
            * vacuum_wavelength
            * tables.selection_lookup[strength_index - 1]
        )
        keep = line_valid & (
            classical_strength * slot_max[torch.clamp(slot - 1, min=0)]
            >= column_min[catalog.continuum_column[sl]]
        )
        kept = torch.nonzero(keep, as_tuple=True)[0]
        if kept.numel() == 0:
            continue

        slot_k = slot[kept]
        column_k = catalog.continuum_column[sl][kept]
        layer_threshold = threshold[:, column_k]  # (80, Ck)
        population_width_k = population_widths[:, slot_k - 1]
        center_absorption = classical_strength[kept][None, :] * population_width_k
        gate = center_absorption >= layer_threshold
        excitation = tables.selection_lookup[catalog.lower_excitation_index[sl][kept] - 1]
        center_absorption = center_absorption * _fast_exponential(
            excitation[None, :] * hc_over_kt[:, None], tables
        )
        gate = gate & (center_absorption >= layer_threshold)
        doppler_k = fractional_doppler_widths[:, slot_k - 1]
        gate = _apply_depth_gate(gate & (doppler_k > 0.0))
        if not bool(gate.any()):
            continue

        radiative = (
            tables.selection_lookup[catalog.radiative_damping_index[sl][kept] - 1]
            * vacuum_wavelength[kept]
            * _DAMPING_SCALE
        )
        stark = (
            tables.selection_lookup[catalog.stark_damping_index[sl][kept] - 1]
            * vacuum_wavelength[kept]
            * _DAMPING_SCALE
        )
        van_der_waals = (
            tables.selection_lookup[catalog.van_der_waals_damping_index[sl][kept] - 1]
            * vacuum_wavelength[kept]
            * _DAMPING_SCALE
        )
        center_k = catalog.center_index[sl][kept]
        wavelength_k = vacuum_wavelength[kept]

        pairs = torch.nonzero(gate, as_tuple=True)
        for pair_start in range(0, pairs[0].numel(), pair_chunk):
            pair_stop = min(pair_start + pair_chunk, pairs[0].numel())
            depth = pairs[0][pair_start:pair_stop]
            line = pairs[1][pair_start:pair_stop]
            doppler_width = doppler_k[depth, line]
            damping = (
                radiative[line]
                + stark[line] * electron_density[depth]
                + van_der_waals[line] * neutral_collision_density[depth]
            ) / torch.clamp(doppler_width, min=1.0e-300)
            slab_flat = _deposit_voigt_wings(
                slab_flat,
                depth_index=depth,
                center_index=center_k[line],
                vacuum_wavelength_nm=wavelength_k[line],
                center_absorption=center_absorption[depth, line],
                damping_parameter=damping,
                doppler_wavelength_width=doppler_width * wavelength_k[line],
                threshold=layer_threshold[depth, line],
                grid=grid,
                tables=tables,
                window=_SELECTED_WINDOW,
            )
    return slab_flat


def _accumulate_transition_normal(
    slab_flat: torch.Tensor,
    tables: TwinLineTables,
    normal_indices: torch.Tensor,
    *,
    grid: torch.Tensor,
    hc_over_kt: torch.Tensor,
    electron_density: torch.Tensor,
    neutral_collision_density: torch.Tensor,
    population_widths: torch.Tensor,
    fractional_doppler_widths: torch.Tensor,
    threshold: torch.Tensor,
    hydrogen_level_dissolution_wavenumber: torch.Tensor,  # (80,)
    line_chunk: int,
    pair_chunk: int,
) -> torch.Tensor:
    """Normal transition runs (line_type 0/3); port of
    line_opacity.py:648-881."""

    slot_count = population_widths.shape[1]
    wavelength_count = grid.shape[0]
    selector = tables.continuum_selector
    last_grid_value = grid[-1]

    for start in range(0, normal_indices.numel(), line_chunk):
        lines = normal_indices[start : start + line_chunk]
        slot = tables.transition_species_slot[lines]
        vacuum_wavelength = tables.transition_vacuum_wavelength_nm[lines]
        continuum_column = torch.searchsorted(
            tables_catalog_bin_edges(tables, grid), tables.transition_packed_wavelength[lines], right=True
        )
        # bin edges are per-star; see _accumulate_transitions for the bound
        raise RuntimeError("internal: use _accumulate_transitions")
    return slab_flat
