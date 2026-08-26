"""``ReducedAtmosphere(m, T, labels) -> FullAtmosphere`` via certified physics.

This is Ting's "one exact population/opacity/transfer synchronization"
(``solver-in-the-loop-prior-work.md`` Sec 2.1), assembled from the reference
solver's own already-refactored, already-certified building blocks
(``payne_zero_atmosphere/runner.py``). No new physics is written: EOS,
opacity, transfer, and hydrostatic pressure are the exact functions the
production solver calls.

Why this cannot be one call to ``run_single_iteration``: that function
also *applies* the temperature correction and remaps the grid
(``runner.py:1445`` ``remap_finalized_iteration_state``), which moves
``column_mass``/``temperature`` away from the input -- exactly the kind of
silent 6-field patch Part 2 forbids. Reconstruction here stops one layer
lower, at the pre-remap, pre-correction quantities that are still indexed on
the *original* (m,T)-pinned grid:

- ``electron_density``: ``population.runtime_state.electron_density`` (EOS
  output, a function of the input T and gas-pressure guess only).
- ``rosseland_opacity``: ``finalization.rosseland_opacity`` (opacity-stage
  output, pre-remap).
- ``radiative_acceleration``: ``finalization.radiative_pressure_state.radiative_acceleration``
  (transfer-stage output, pre-remap).
- ``gas_pressure``: NOT an EOS output -- ``build_runtime_state`` copies it
  straight from the input guess. It is instead updated by
  ``hydrostatic.integrate_hydrostatic_pressure`` using the *radiation*
  pressure the transfer pass just produced, with column mass held at the
  reduced state's exact truth throughout.

Because opacity/populations depend on pressure and pressure depends on
radiation pressure (a transfer-stage output), one pass is not exactly
self-consistent -- this mirrors why Ting's own one-synchronization
experiment still drifted. ``n_synchronizations`` repeats the pass, feeding
each pass's hydrostatic pressure into the next, while m and T never move.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere, parse_atmosphere_deck
from payne_zero_atmosphere.config import AtmosphereInput
from payne_zero_atmosphere.constants import BOLTZMANN_ERG_PER_K_REFERENCE
from payne_zero_atmosphere.hydrostatic import integrate_hydrostatic_pressure
from payne_zero_atmosphere.run_setup import resolve_run_setup
from payne_zero_atmosphere.runner import (
    accumulate_transfer_state,
    finalize_transfer_state,
    prepare_opacity_state,
    prepare_population_state,
)
from payne_zero_atmosphere.warm_start import (
    emulator_warm_start_model,
    format_warm_start_deck,
)

# Reuse bench's production AtmosphereConfig builder rather than duplicating
# the molecular/line-catalog wiring. iterations_per_trial=1 is irrelevant
# here -- we never call run_atmosphere_model / run_single_iteration through
# this config, only resolve_run_setup + the individual pipeline stages.
from bench.run_reference import _solver_config

DEFAULT_SYNCHRONIZATION_PASSES = 3
DEFAULT_MAX_SYNCHRONIZATIONS = 8
DEFAULT_PRESSURE_TOLERANCE_DEX = 1.0e-3


class ReconstructionConvergenceError(RuntimeError):
    """Raised when adaptive pressure synchronization does not settle."""

    def __init__(
        self,
        message: str,
        *,
        pressure_change_dex_by_pass: list[float] | tuple[float, ...] = (),
    ) -> None:
        super().__init__(message)
        self.pressure_change_dex_by_pass = tuple(
            float(value) for value in pressure_change_dex_by_pass
        )


@dataclass(frozen=True)
class ReducedAtmosphere:
    """The state Part 2 asks whether is sufficient: m(tau), T(tau), labels."""

    column_mass: np.ndarray
    temperature: np.ndarray
    labels: dict[str, float]


@dataclass(frozen=True)
class ReconstructionResult:
    """The reconstructed full atmosphere plus per-pass diagnostics."""

    atmosphere: ModelAtmosphere
    n_synchronizations: int
    n_evaluations: int
    n_pressure_updates: int
    pressure_change_dex_by_pass: list[float]
    synchronized: bool
    gas_pressure_by_pass: list[np.ndarray]
    electron_density_by_pass: list[np.ndarray]
    rosseland_opacity_by_pass: list[np.ndarray]
    radiative_acceleration_by_pass: list[np.ndarray]


def _emulator_seed_atmosphere(
    reduced: ReducedAtmosphere, *, allow_extrapolation: bool = False
) -> ModelAtmosphere:
    """Six-field warm start for labels, with m,T pinned to the reduced state.

    The warm start supplies everything a reduced state genuinely does not
    carry: an initial pressure guess (refined below by hydrostatic
    synchronization) and the metadata (Teff, logg, opacity flags, ...) that
    ``resolve_run_setup`` reads off the atmosphere object. Only column_mass
    and temperature are overwritten -- the warm start's own guesses for
    those two are discarded, never blended.
    """

    warm_start_atmosphere, _deck = emulator_warm_start_model(
        device="cpu",
        allow_extrapolation=allow_extrapolation,
        **reduced.labels,
    )
    column_mass = np.asarray(reduced.column_mass, dtype=np.float64)
    temperature = np.asarray(reduced.temperature, dtype=np.float64)
    if column_mass.shape != warm_start_atmosphere.column_mass.shape:
        raise ValueError(
            "ReducedAtmosphere.column_mass must match the production layer "
            f"count {warm_start_atmosphere.column_mass.shape}, got {column_mass.shape}"
        )
    if np.any(np.diff(column_mass) <= 0.0):
        raise ValueError("ReducedAtmosphere.column_mass must be strictly increasing")
    return dataclasses.replace(
        warm_start_atmosphere,
        column_mass=column_mass,
        temperature=temperature,
    )


def _physical_seed_atmosphere(
    reduced: ReducedAtmosphere,
    *,
    electron_fraction: float = 1.0e-4,
    pressure_scale: float = 1.0,
) -> ModelAtmosphere:
    """Physics-default seed: no neural network, no truth profile.

    The reduced state genuinely lacks only an initial guess for the fields
    the synchronization below re-derives anyway, plus the deck metadata. Both
    come from labels and first principles here:

    - ``gas_pressure = g * column_mass`` -- hydrostatic balance with the
      not-yet-known radiation and turbulent terms ignored; the first
      synchronization pass re-introduces them via
      ``integrate_hydrostatic_pressure``.
    - ``electron_density = electron_fraction * P / (k_B T)`` -- only a
      positive seed for validation; the molecular EOS re-solves charge
      conservation from its own seeds and overwrites this value.
    - ``rosseland_opacity = 0.34``, ``radiative_acceleration = 0`` -- positive /
      finite placeholders overwritten by the first opacity/transfer pass.
    - ``microturbulence = label * 1e5`` cm/s, matching the grey-start
      benchmark's label-faithful fill.

    ``electron_fraction`` and ``pressure_scale`` exist only so
    seed-independence checks can perturb the guesses; production use should
    leave both at their defaults.

    The deck round-trip (``format_warm_start_deck`` + ``parse_atmosphere_deck``)
    guarantees the metadata and abundances are byte-identical to what the
    production warm-start path produces. Its ``%.3E`` quantization would touch
    the pinned fields, so column_mass and temperature are restored to the
    reduced state's exact values afterwards -- the same treatment the emulator
    seed gives them.
    """

    labels = reduced.labels
    column_mass = np.asarray(reduced.column_mass, dtype=np.float64)
    temperature = np.asarray(reduced.temperature, dtype=np.float64)
    if np.any(np.diff(column_mass) <= 0.0):
        raise ValueError("ReducedAtmosphere.column_mass must be strictly increasing")
    if not 0.0 < electron_fraction < 1.0:
        raise ValueError("electron_fraction must lie in (0, 1)")
    if not np.isfinite(pressure_scale) or pressure_scale <= 0.0:
        raise ValueError("pressure_scale must be finite and positive")
    gravity = 10.0 ** float(labels["log_surface_gravity"])
    gas_pressure = np.maximum(gravity * column_mass * pressure_scale, 1.0e-20)
    electron_density = np.maximum(
        electron_fraction * gas_pressure / (BOLTZMANN_ERG_PER_K_REFERENCE * temperature),
        1.0e-20,
    )
    table = np.zeros((column_mass.size, 9), dtype=np.float64)
    table[:, 0] = column_mass
    table[:, 1] = temperature
    table[:, 2] = gas_pressure
    table[:, 3] = electron_density
    table[:, 4] = 0.34  # rosseland_opacity placeholder
    table[:, 5] = 0.0  # radiative_acceleration placeholder
    table[:, 6] = float(labels["microturbulence_km_s"]) * 1.0e5
    deck = format_warm_start_deck(
        effective_temperature=float(labels["effective_temperature"]),
        log_surface_gravity=float(labels["log_surface_gravity"]),
        metallicity=float(labels["metallicity"]),
        alpha_enhancement=float(labels["alpha_enhancement"]),
        layer_table=table,
        title="Payne Zero reduced-state physical seed",
    )
    seeded = parse_atmosphere_deck(deck, source="<reduced-state-physical-seed>")
    return dataclasses.replace(
        seeded,
        column_mass=column_mass,
        temperature=temperature,
    )


def _seed_atmosphere(
    reduced: ReducedAtmosphere,
    *,
    seed: str | ModelAtmosphere = "physical",
    allow_extrapolation: bool = False,
    seed_kwargs: dict | None = None,
) -> ModelAtmosphere:
    """Build the seed atmosphere for synchronization; see the seed modes.

    ``seed`` is ``"physical"``, ``"emulator"``, or a pre-built
    ``ModelAtmosphere`` (used by checks that must construct the seed in a
    separate process -- e.g. when the torch-based emulator warm start cannot
    share a process with the numba pipeline). A pre-built seed must already
    carry the reduced state's exact (m, T); this function only validates.
    """

    if isinstance(seed, ModelAtmosphere):
        if seed_kwargs:
            raise ValueError("seed_kwargs only apply to the 'physical' seed")
        column_mass = np.asarray(seed.column_mass, dtype=np.float64)
        if column_mass.shape != np.asarray(reduced.column_mass).shape:
            raise ValueError(
                "pre-built seed layer count does not match the reduced state"
            )
        if np.any(np.diff(column_mass) <= 0.0):
            raise ValueError("pre-built seed column_mass must be strictly increasing")
        return seed
    if seed == "physical":
        return _physical_seed_atmosphere(reduced, **(seed_kwargs or {}))
    if seed == "emulator":
        if seed_kwargs:
            raise ValueError("seed_kwargs only apply to the 'physical' seed")
        return _emulator_seed_atmosphere(
            reduced, allow_extrapolation=allow_extrapolation
        )
    raise ValueError(
        f"unknown seed mode {seed!r}; expected 'physical', 'emulator', "
        "or a pre-built ModelAtmosphere"
    )


def reconstruct_full_atmosphere(
    reduced: ReducedAtmosphere,
    *,
    n_synchronizations: int | None = DEFAULT_SYNCHRONIZATION_PASSES,
    max_synchronizations: int = DEFAULT_MAX_SYNCHRONIZATIONS,
    pressure_tolerance_dex: float = DEFAULT_PRESSURE_TOLERANCE_DEX,
    allow_extrapolation: bool = False,
    seed: str | ModelAtmosphere = "physical",
    seed_kwargs: dict | None = None,
) -> ReconstructionResult:
    """Materialize P, n_e, kappa_R, g_rad from (m, T, labels) alone.

    m and T are never modified. Zero new physics: every call below is a
    certified ``payne_zero_atmosphere`` function, at the same granularity
    ``run_single_iteration`` uses internally, stopping short of the
    temperature-correction/remap step that would move the grid.

    ``seed`` selects where the initial guesses for the derived fields come
    from. ``"physical"`` (default) uses only labels and hydrostatic balance
    (``P = g*m``, ``n_e = 1e-4*P/kT``); ``"emulator"`` reproduces the
    historical shortcut of seeding from the six-field warm-start network; a
    pre-built ``ModelAtmosphere`` is used as-is (with m, T validation) for
    checks that build the seed in a separate process. All converge to the
    same synchronized state -- the molecular EOS re-solves charge
    conservation from its own seeds, and the hydrostatic passes re-derive
    pressure -- so the seed is an implementation detail, not a learned input.

    ``seed_kwargs`` optionally forwards perturbation hooks
    (``electron_fraction``, ``pressure_scale``) to the physical seed builder;
    it exists for seed-independence checks, not for production use.

    If ``n_synchronizations`` is an integer, the historical fixed-pass mode is
    used. It performs that many hydrostatic pressure updates and then performs
    one final materialization at the resulting pressure, so all returned fields
    were computed from the same pressure. If it is ``None``, pressure updates
    continue until the maximum layer-wise change is below
    ``pressure_tolerance_dex`` or ``max_synchronizations`` is reached.
    """

    if n_synchronizations is not None and n_synchronizations < 1:
        raise ValueError("n_synchronizations must be >= 1 or None")
    if max_synchronizations < 1:
        raise ValueError("max_synchronizations must be >= 1")
    if not np.isfinite(pressure_tolerance_dex) or pressure_tolerance_dex <= 0.0:
        raise ValueError("pressure_tolerance_dex must be finite and positive")

    current_atmosphere = _seed_atmosphere(
        reduced,
        seed=seed,
        allow_extrapolation=allow_extrapolation,
        seed_kwargs=seed_kwargs,
    )
    config = _solver_config(
        current_atmosphere,
        iterations_per_trial=1,
        structured_atmosphere_path=None,
        debug_state_path=None,
    )

    gas_pressure_by_pass: list[np.ndarray] = []
    electron_density_by_pass: list[np.ndarray] = []
    rosseland_opacity_by_pass: list[np.ndarray] = []
    radiative_acceleration_by_pass: list[np.ndarray] = []
    pressure_change_dex_by_pass: list[float] = []

    def evaluate_current_atmosphere():
        """Evaluate all dependent fields from one exact pressure state."""

        pass_config = dataclasses.replace(
            config,
            inputs=dataclasses.replace(
                config.inputs, initial_atmosphere=current_atmosphere
            ),
        )
        setup = resolve_run_setup(pass_config)

        population = prepare_population_state(
            pass_config,
            temperature_iteration_index=1,
            setup=setup,
            molecular_thermal_energy_erg=setup.atmosphere.thermal_energy_erg,
        )
        opacity = prepare_opacity_state(
            pass_config,
            population_state=population,
            temperature_iteration_index=1,
        )
        transfer = accumulate_transfer_state(opacity)
        finalization = finalize_transfer_state(
            transfer,
            iteration_index=1,
            temperature_iteration_seed=10,
            convection_enabled=pass_config.enable_convection,
            molecular_convection_thermal_tracks_perturbation=(
                pass_config.molecular_convection_thermal_tracks_perturbation
            ),
        )

        fields = {
            "gas_pressure": np.asarray(current_atmosphere.gas_pressure, dtype=np.float64).copy(),
            "electron_density": np.asarray(
                population.runtime_state.electron_density, dtype=np.float64
            ).copy(),
            "rosseland_opacity": np.asarray(
                finalization.rosseland_opacity, dtype=np.float64
            ).copy(),
            "radiative_acceleration": np.asarray(
                finalization.radiative_pressure_state.radiative_acceleration,
                dtype=np.float64,
            ).copy(),
        }
        candidate_pressure = integrate_hydrostatic_pressure(
            current_atmosphere,
            surface_gravity_cgs=setup.surface_gravity_cgs,
            integrated_radiation_pressure=(
                finalization.radiative_pressure_state.integrated_radiation_pressure
            ),
            turbulent_pressure=np.zeros(current_atmosphere.layers, dtype=np.float64),
        )
        return fields, candidate_pressure

    def pressure_change_dex(old_pressure: np.ndarray, new_pressure: np.ndarray) -> float:
        return float(
            np.max(
                np.abs(
                    np.log10(np.maximum(new_pressure, 1.0e-300))
                    - np.log10(np.maximum(old_pressure, 1.0e-300))
                )
            )
        )

    def append_evaluation(fields: dict[str, np.ndarray], pressure_delta: float) -> None:
        gas_pressure_by_pass.append(fields["gas_pressure"])
        electron_density_by_pass.append(fields["electron_density"])
        rosseland_opacity_by_pass.append(fields["rosseland_opacity"])
        radiative_acceleration_by_pass.append(fields["radiative_acceleration"])
        pressure_change_dex_by_pass.append(float(pressure_delta))

    def make_result(*, requested_passes: int, synchronized: bool) -> ReconstructionResult:
        return ReconstructionResult(
            atmosphere=current_atmosphere,
            n_synchronizations=requested_passes,
            n_evaluations=len(pressure_change_dex_by_pass),
            n_pressure_updates=max(0, len(pressure_change_dex_by_pass) - 1),
            pressure_change_dex_by_pass=pressure_change_dex_by_pass,
            synchronized=bool(synchronized),
            gas_pressure_by_pass=gas_pressure_by_pass,
            electron_density_by_pass=electron_density_by_pass,
            rosseland_opacity_by_pass=rosseland_opacity_by_pass,
            radiative_acceleration_by_pass=radiative_acceleration_by_pass,
        )

    if n_synchronizations is None:
        for _ in range(max_synchronizations):
            fields, candidate_pressure = evaluate_current_atmosphere()
            delta = pressure_change_dex(fields["gas_pressure"], candidate_pressure)
            append_evaluation(fields, delta)
            if delta <= pressure_tolerance_dex:
                current_atmosphere = dataclasses.replace(
                    current_atmosphere,
                    gas_pressure=fields["gas_pressure"],
                    electron_density=fields["electron_density"],
                    rosseland_opacity=fields["rosseland_opacity"],
                    radiative_acceleration=fields["radiative_acceleration"],
                )
                return make_result(
                    requested_passes=len(pressure_change_dex_by_pass),
                    synchronized=True,
                )
            current_atmosphere = dataclasses.replace(
                current_atmosphere,
                gas_pressure=candidate_pressure,
                electron_density=fields["electron_density"],
                rosseland_opacity=fields["rosseland_opacity"],
                radiative_acceleration=fields["radiative_acceleration"],
            )
        raise ReconstructionConvergenceError(
            "reduced-state pressure synchronization did not converge within "
            f"{max_synchronizations} passes; last change was "
            f"{pressure_change_dex_by_pass[-1]:.3e} dex",
            pressure_change_dex_by_pass=pressure_change_dex_by_pass,
        )

    for _ in range(n_synchronizations):
        fields, candidate_pressure = evaluate_current_atmosphere()
        delta = pressure_change_dex(fields["gas_pressure"], candidate_pressure)
        append_evaluation(fields, delta)
        current_atmosphere = dataclasses.replace(
            current_atmosphere,
            gas_pressure=candidate_pressure,
            electron_density=fields["electron_density"],
            rosseland_opacity=fields["rosseland_opacity"],
            radiative_acceleration=fields["radiative_acceleration"],
        )

    # Legacy fixed-pass mode still returns a coherent state: after the last
    # pressure update, materialize all dependent fields from that final
    # pressure instead of pairing the new pressure with the previous pass's
    # EOS/opacity/transfer outputs.
    fields, candidate_pressure = evaluate_current_atmosphere()
    delta = pressure_change_dex(fields["gas_pressure"], candidate_pressure)
    append_evaluation(fields, delta)
    current_atmosphere = dataclasses.replace(
        current_atmosphere,
        electron_density=fields["electron_density"],
        rosseland_opacity=fields["rosseland_opacity"],
        radiative_acceleration=fields["radiative_acceleration"],
    )
    return make_result(
        requested_passes=n_synchronizations,
        synchronized=delta <= pressure_tolerance_dex,
    )
