"""A small physics-driven warm-start pilot.

This module deliberately stays below the production atmosphere solver. It
uses the production EOS/population and continuum-opacity routines on a coarse
32-layer grid, compresses the sampled continuum opacity into four frequency
groups, and solves a two-stream transfer problem. Two damped updates are made
with convection disabled and then two with the EOS-based convection diagnostic
enabled. The resulting ``(m, T)`` is interpolated to the standard 80-layer grid
and handed to the unchanged production solver.

The current line-opacity kernels require exactly 80 layers. Consequently this
first pilot does not pretend to include line opacity in the coarse pre-solver:
line opacity is opened by the downstream exact solver. That boundary is
intentional and is part of the pilot provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


COARSE_LAYER_COUNT = 32
OPACITY_GROUP_COUNT = 4
LOG_TAU_START = -6.875
LOG_TAU_STOP = 3.0
DEFAULT_FREQUENCY_GRID_STRIDE = 256
DEFAULT_STAGE_UPDATES = 2
DEFAULT_DAMPING = 0.5
_FOUR_PI_REFERENCE = 12.5664
_STEFAN_BOLTZMANN_REFERENCE = 5.6697e-5
_LIGHT_SPEED_CM_PER_S = 2.99792458e10


@dataclass(frozen=True)
class GroupedContinuumOpacity:
    """Four-group opacity and source columns on the coarse grid."""

    group_slices: tuple[tuple[int, int], ...]
    frequency_hz: np.ndarray
    frequency_weights: np.ndarray
    absorption: np.ndarray
    scattering: np.ndarray
    source: np.ndarray
    rosseland_opacity: np.ndarray


@dataclass(frozen=True)
class TwoStreamResult:
    """Frequency-group transfer diagnostics in Eddington-flux units."""

    group_flux: np.ndarray
    total_flux: np.ndarray
    radiative_acceleration: np.ndarray
    integrated_radiation_pressure: np.ndarray
    negative_flux_layers: int


@dataclass(frozen=True)
class HomotopyResult:
    """The coarse physical seed and its solver-independent diagnostics."""

    coarse_tau: np.ndarray
    column_mass: np.ndarray
    temperature: np.ndarray
    rosseland_opacity: np.ndarray
    diagnostics: dict[str, Any]


def coarse_rosseland_tau(layer_count: int = COARSE_LAYER_COUNT) -> np.ndarray:
    """Return the logarithmic optical-depth grid used by the pilot."""

    count = int(layer_count)
    if count < 4:
        raise ValueError("the coarse grid needs at least four layers")
    return 10.0 ** np.linspace(LOG_TAU_START, LOG_TAU_STOP, count)


def frequency_group_slices(
    frequency_weights: np.ndarray,
    *,
    group_count: int = OPACITY_GROUP_COUNT,
) -> tuple[tuple[int, int], ...]:
    """Split a monotone frequency grid into equal-measure contiguous groups."""

    weights = np.asarray(frequency_weights, dtype=np.float64).reshape(-1)
    groups = int(group_count)
    if weights.size < groups:
        raise ValueError("frequency grid must have at least one sample per group")
    if groups < 1 or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("frequency weights must be finite and positive")

    cumulative = np.cumsum(weights)
    total = float(cumulative[-1])
    edges = [0]
    for group_index in range(1, groups):
        target = total * group_index / groups
        edge = int(np.searchsorted(cumulative, target, side="left")) + 1
        edge = max(edge, edges[-1] + 1)
        edge = min(edge, weights.size - (groups - group_index))
        edges.append(edge)
    edges.append(weights.size)
    return tuple((edges[i], edges[i + 1]) for i in range(groups))


def _planck_source(
    frequency_hz: np.ndarray,
    temperature_k: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return production-convention Planck source and stimulated emission."""

    from payne_zero_atmosphere.constants import (
        BOLTZMANN_ERG_PER_K_REFERENCE,
        PLANCK_ERG_SECOND_REFERENCE,
    )

    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    temperature = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
    x = frequency[None, :] * PLANCK_ERG_SECOND_REFERENCE / (
        BOLTZMANN_ERG_PER_K_REFERENCE * np.maximum(temperature[:, None], 1.0)
    )
    exponential = np.exp(-x)
    stimulated = np.maximum(1.0 - exponential, 1.0e-300)
    source = (
        1.47439e-2
        * (frequency[None, :] / 1.0e15) ** 3
        * exponential
        / stimulated
    )
    return source, stimulated


def build_grouped_continuum_opacity(
    *,
    frequency_hz: np.ndarray,
    frequency_weights: np.ndarray,
    absorption: np.ndarray,
    scattering: np.ndarray,
    source: np.ndarray,
    temperature_k: np.ndarray,
    group_count: int = OPACITY_GROUP_COUNT,
) -> GroupedContinuumOpacity:
    """Collapse real sampled continuum columns into Planck/Rosseland groups."""

    frequency = np.asarray(frequency_hz, dtype=np.float64).reshape(-1)
    weights = np.asarray(frequency_weights, dtype=np.float64).reshape(-1)
    temperature = np.asarray(temperature_k, dtype=np.float64).reshape(-1)
    abs_column = np.asarray(absorption, dtype=np.float64)
    scat_column = np.asarray(scattering, dtype=np.float64)
    source_column = np.asarray(source, dtype=np.float64)
    expected_shape = (temperature.size, frequency.size)
    if not (
        abs_column.shape == scat_column.shape == source_column.shape == expected_shape
    ):
        raise ValueError("opacity columns must have shape (layers, frequencies)")
    if np.any(~np.isfinite(abs_column)) or np.any(~np.isfinite(scat_column)):
        raise ValueError("opacity columns must be finite")
    if np.any(abs_column < 0.0) or np.any(scat_column < 0.0):
        raise ValueError("opacity columns must be non-negative")

    planck, stimulated = _planck_source(frequency, temperature)
    x = frequency[None, :] * 6.6256e-27 / (
        1.38054e-16 * np.maximum(temperature[:, None], 1.0)
    )
    derivative = planck * x / (
        np.maximum(temperature[:, None], 1.0) * stimulated
    )
    groups = frequency_group_slices(weights, group_count=group_count)
    layer_count = temperature.size
    group_frequency = np.empty(len(groups), dtype=np.float64)
    group_weights = np.empty(len(groups), dtype=np.float64)
    group_absorption = np.empty((layer_count, len(groups)), dtype=np.float64)
    group_scattering = np.empty_like(group_absorption)
    group_source = np.empty_like(group_absorption)
    group_rosseland = np.empty_like(group_absorption)

    for group_index, (start, stop) in enumerate(groups):
        local_weights = weights[start:stop]
        weight = float(np.sum(local_weights))
        weighted_planck = local_weights[None, :] * planck[:, start:stop]
        planck_norm = np.maximum(np.sum(weighted_planck, axis=1), 1.0e-300)
        group_weights[group_index] = weight
        group_frequency[group_index] = float(
            np.sum(local_weights * frequency[start:stop]) / weight
        )
        group_absorption[:, group_index] = np.sum(
            weighted_planck * abs_column[:, start:stop], axis=1
        ) / planck_norm
        group_scattering[:, group_index] = np.sum(
            weighted_planck * scat_column[:, start:stop], axis=1
        ) / planck_norm

        absorption_weight = local_weights[None, :] * abs_column[:, start:stop]
        absorption_emission = np.sum(
            absorption_weight * source_column[:, start:stop], axis=1
        )
        group_source[:, group_index] = np.where(
            group_absorption[:, group_index] > 1.0e-300,
            absorption_emission
            / np.maximum(group_absorption[:, group_index], 1.0e-300),
            np.sum(weighted_planck, axis=1),
        )

        total = np.maximum(
            abs_column[:, start:stop] + scat_column[:, start:stop], 1.0e-300
        )
        derivative_weight = local_weights[None, :] * derivative[:, start:stop]
        group_rosseland[:, group_index] = np.sum(
            derivative_weight, axis=1
        ) / np.maximum(np.sum(derivative_weight / total, axis=1), 1.0e-300)

    rosseland = (
        np.sum(
            group_rosseland ** -1.0 * group_weights[None, :],
            axis=1,
        )
        / np.maximum(np.sum(group_weights), 1.0e-300)
    ) ** -1.0
    return GroupedContinuumOpacity(
        group_slices=groups,
        frequency_hz=group_frequency,
        frequency_weights=group_weights,
        absorption=group_absorption,
        scattering=group_scattering,
        source=group_source,
        rosseland_opacity=np.maximum(rosseland, 1.0e-30),
    )


def two_stream_transfer(
    *,
    column_mass: np.ndarray,
    opacity: GroupedContinuumOpacity,
    target_integrated_flux: float | None = None,
) -> TwoStreamResult:
    """Solve independent two-stream group transfers.

    The source is linearly integrated across each optical-depth cell.  A
    midpoint source is not adequate on the logarithmic coarse grid: the
    deepest cell can span hundreds of optical depths, so its midpoint creates
    a spurious deep flux.  When the target flux is supplied, the lower
    boundary is also set from the local diffusion slope and the group share of
    the target flux.
    """

    from payne_zero_atmosphere.radiative_transfer import integrate_on_depth_grid

    mass = np.asarray(column_mass, dtype=np.float64).reshape(-1)
    if mass.size != opacity.absorption.shape[0] or np.any(np.diff(mass) <= 0.0):
        raise ValueError("column_mass must match opacity layers and be increasing")
    layer_count, group_count = opacity.absorption.shape
    mu = 1.0 / np.sqrt(3.0)
    group_flux = np.zeros((layer_count, group_count), dtype=np.float64)

    for group_index in range(group_count):
        total_opacity = np.maximum(
            opacity.absorption[:, group_index] + opacity.scattering[:, group_index],
            1.0e-30,
        )
        source = np.maximum(opacity.source[:, group_index], 0.0)
        delta_tau = 0.5 * (total_opacity[1:] + total_opacity[:-1]) * np.diff(mass)
        transmission = np.exp(-np.clip(delta_tau / mu, 0.0, 700.0))

        outward = np.zeros(layer_count, dtype=np.float64)
        inward = np.zeros(layer_count, dtype=np.float64)
        if target_integrated_flux is None:
            outward[-1] = source[-1]
        else:
            source_total = float(np.sum(np.maximum(opacity.source[-1], 0.0)))
            group_share = (
                float(source[-1]) / source_total
                if source_total > 0.0
                else 1.0 / group_count
            )
            last_delta_tau = max(float(delta_tau[-1]), 1.0e-30)
            source_slope = float(source[-1] - source[-2]) / last_delta_tau
            target_group_flux = float(target_integrated_flux) * group_share
            # For a diffusion solution I_in ~= S - mu*dS/dtau.  Choose the
            # outgoing lower-boundary intensity so that the group flux is the
            # prescribed share of the stellar flux.
            outward[-1] = (
                source[-1]
                - mu * source_slope
                + 2.0 * target_group_flux / mu
            )
        for layer_index in range(layer_count - 2, -1, -1):
            cell_delta_tau = max(float(delta_tau[layer_index]), 1.0e-30)
            source_slope = (
                float(source[layer_index + 1] - source[layer_index])
                / cell_delta_tau
            )
            source_integral = (
                source[layer_index] * (1.0 - transmission[layer_index])
                + source_slope
                * (
                    mu * (1.0 - transmission[layer_index])
                    - cell_delta_tau * transmission[layer_index]
                )
            )
            outward[layer_index] = (
                outward[layer_index + 1] * transmission[layer_index]
                + source_integral
            )
        for layer_index in range(layer_count - 1):
            cell_delta_tau = max(float(delta_tau[layer_index]), 1.0e-30)
            source_slope = (
                float(source[layer_index + 1] - source[layer_index])
                / cell_delta_tau
            )
            source_integral = (
                source[layer_index + 1] * (1.0 - transmission[layer_index])
                - source_slope
                * (
                    mu * (1.0 - transmission[layer_index])
                    - cell_delta_tau * transmission[layer_index]
                )
            )
            inward[layer_index + 1] = (
                inward[layer_index] * transmission[layer_index]
                + source_integral
            )
        group_flux[:, group_index] = 0.5 * mu * (outward - inward)

    total_flux = np.sum(group_flux, axis=1)
    radiative_acceleration = (
        _FOUR_PI_REFERENCE
        / _LIGHT_SPEED_CM_PER_S
        * np.sum(
            group_flux * (opacity.absorption + opacity.scattering),
            axis=1,
        )
    )
    integrated_pressure = integrate_on_depth_grid(
        mass,
        radiative_acceleration,
        surface_value=float(radiative_acceleration[0] * mass[0]),
    )
    return TwoStreamResult(
        group_flux=group_flux,
        total_flux=total_flux,
        radiative_acceleration=radiative_acceleration,
        integrated_radiation_pressure=integrated_pressure,
        negative_flux_layers=int(np.count_nonzero(total_flux <= 0.0)),
    )


def _make_seed_atmosphere(
    labels: np.ndarray,
    tau: np.ndarray,
    column_mass: np.ndarray,
    temperature: np.ndarray,
    rosseland_opacity: np.ndarray,
):
    """Build a solver-compatible complete table without an emulator."""

    from experiments.analytic_initializer.no_emulator_bridge import analytic_seed_model

    return analytic_seed_model(
        np.asarray(labels, dtype=np.float64),
        np.asarray(column_mass, dtype=np.float64),
        np.asarray(temperature, dtype=np.float64),
        np.log10(np.maximum(np.asarray(rosseland_opacity, dtype=np.float64), 1.0e-30)),
        np.asarray(tau, dtype=np.float64),
    )


def _evaluate_real_continuum(
    atmosphere,
    *,
    frequency_grid_stride: int,
    group_count: int,
    target_integrated_flux: float | None = None,
):
    """Evaluate the production EOS and continuum opacity on one coarse state."""

    from bench.run_reference import _solver_config
    from payne_zero_atmosphere.continuum_opacity import (
        build_continuum_atmosphere_state,
        build_opacity_sampling_grid,
        compute_continuum_opacity_columns,
    )
    from payne_zero_atmosphere.runner import prepare_population_state

    config = _solver_config(
        atmosphere,
        iterations_per_trial=1,
        structured_atmosphere_path=None,
        debug_state_path=None,
    )
    config = replace(config, opacity_frequency_grid_stride=int(frequency_grid_stride))
    population = prepare_population_state(config, temperature_iteration_index=1)
    continuum = build_continuum_atmosphere_state(
        atmosphere,
        population.runtime_state,
    )
    wavelength_nm, frequency_weights = build_opacity_sampling_grid(
        population.setup.effective_temperature,
        frequency_grid_stride=int(frequency_grid_stride),
    )
    frequency_hz = 2.99792458e17 / wavelength_nm
    absorption, scattering, source = compute_continuum_opacity_columns(
        continuum,
        frequency_hz,
        opacity_flags=population.setup.opacity_flags,
    )
    grouped = build_grouped_continuum_opacity(
        frequency_hz=frequency_hz,
        frequency_weights=frequency_weights,
        absorption=absorption,
        scattering=scattering,
        source=source,
        temperature_k=atmosphere.temperature,
        group_count=group_count,
    )
    transfer = two_stream_transfer(
        column_mass=atmosphere.column_mass,
        opacity=grouped,
        target_integrated_flux=target_integrated_flux,
    )
    return config, population, grouped, transfer


def _eos_convection_flux(
    *,
    atmosphere,
    population,
    grouped: GroupedContinuumOpacity,
    transfer: TwoStreamResult,
    surface_gravity_cgs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the existing EOS finite differences for one MLT convection pass."""

    from payne_zero_atmosphere.continuum_opacity import (
        create_rosseland_opacity_table,
        ingest_rosseland_opacity_table,
    )
    from payne_zero_atmosphere.convection import (
        compute_convection,
    )
    from payne_zero_atmosphere.runner import compute_convection_finite_difference_samples
    from payne_zero_atmosphere.radiative_transfer import integrate_on_depth_grid

    rosseland_table = create_rosseland_opacity_table(atmosphere.layers)
    ingest_rosseland_opacity_table(
        rosseland_table,
        temperature_k=atmosphere.temperature,
        gas_pressure=atmosphere.gas_pressure,
        rosseland_opacity=grouped.rosseland_opacity,
    )
    rosseland_tau = integrate_on_depth_grid(
        atmosphere.column_mass,
        grouped.rosseland_opacity,
        surface_value=float(grouped.rosseland_opacity[0] * atmosphere.column_mass[0]),
    )
    samples = compute_convection_finite_difference_samples(
        atmosphere=atmosphere,
        runtime_state=population.runtime_state,
        absolute_radiation_pressure=transfer.integrated_radiation_pressure,
        rosseland_optical_depth=rosseland_tau,
        temperature_iteration_seed=10,
        temperature_iteration_cache=population.temperature_iteration_cache,
        molecules_enabled=population.setup.molecules_enabled,
        molecular_state=population.molecular_state,
    )
    result = compute_convection(
        rosseland_table=rosseland_table,
        column_mass=atmosphere.column_mass,
        rosseland_optical_depth=rosseland_tau,
        temperature_k=atmosphere.temperature,
        gas_pressure=atmosphere.gas_pressure,
        mass_density=population.runtime_state.mass_density,
        rosseland_opacity=grouped.rosseland_opacity,
        microturbulence=atmosphere.microturbulence,
        absolute_radiation_pressure=transfer.integrated_radiation_pressure,
        total_pressure=float(surface_gravity_cgs) * atmosphere.column_mass,
        surface_gravity_cgs=float(surface_gravity_cgs),
        target_integrated_eddington_flux=(
            _STEFAN_BOLTZMANN_REFERENCE
            / _FOUR_PI_REFERENCE
            * float(population.setup.effective_temperature) ** 4
        ),
        mixing_length=1.25,
        overshoot_weight=0.0,
        convection_enabled=True,
        zero_top_layer_count=max(3, atmosphere.layers // 8),
        specific_internal_energy_plus_temperature=(
            samples.specific_internal_energy_plus_temperature
        ),
        specific_internal_energy_minus_temperature=(
            samples.specific_internal_energy_minus_temperature
        ),
        specific_internal_energy_plus_pressure=samples.specific_internal_energy_plus_pressure,
        specific_internal_energy_minus_pressure=samples.specific_internal_energy_minus_pressure,
        density_plus_temperature=samples.density_plus_temperature,
        density_minus_temperature=samples.density_minus_temperature,
        density_plus_pressure=samples.density_plus_pressure,
        density_minus_pressure=samples.density_minus_pressure,
    )
    return (
        np.nan_to_num(result.convective_flux, nan=0.0, posinf=0.0, neginf=0.0),
        np.asarray(result.adiabatic_gradient, dtype=np.float64),
        np.asarray(result.logarithmic_temperature_pressure_gradient, dtype=np.float64),
    )


def physical_homotopy_seed(
    labels: np.ndarray,
    *,
    layer_count: int = COARSE_LAYER_COUNT,
    group_count: int = OPACITY_GROUP_COUNT,
    frequency_grid_stride: int = DEFAULT_FREQUENCY_GRID_STRIDE,
    stage_updates: int = DEFAULT_STAGE_UPDATES,
    damping: float = DEFAULT_DAMPING,
) -> HomotopyResult:
    """Build a no-emulator coarse physical seed for one five-label star."""

    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.shape != (5,) or np.any(~np.isfinite(values)):
        raise ValueError("labels must be one finite five-label vector")
    if int(stage_updates) < 1:
        raise ValueError("stage_updates must be positive")
    if not 0.0 < float(damping) <= 1.0:
        raise ValueError("damping must lie in (0, 1]")

    from payne_zero_atmosphere.radiative_transfer import integrate_on_depth_grid

    tau = coarse_rosseland_tau(layer_count)
    temperature = values[0] * (0.75 * (tau + 2.0 / 3.0)) ** 0.25
    column_mass = tau / 0.34
    rosseland_opacity = np.full(tau.size, 0.34, dtype=np.float64)
    atmosphere = _make_seed_atmosphere(
        values, tau, column_mass, temperature, rosseland_opacity
    )
    diagnostics: list[dict[str, Any]] = []
    physical_evaluations = 0
    target_flux = (
        _STEFAN_BOLTZMANN_REFERENCE
        / _FOUR_PI_REFERENCE
        * float(values[0]) ** 4
    )

    for stage in ("radiative", "convective"):
        for update_index in range(int(stage_updates)):
            _config, population, grouped, transfer = _evaluate_real_continuum(
                atmosphere,
                frequency_grid_stride=int(frequency_grid_stride),
                group_count=int(group_count),
                target_integrated_flux=target_flux,
            )
            physical_evaluations += 1
            if stage == "convective":
                convective_flux, adiabatic_gradient, logarithmic_gradient = (
                    _eos_convection_flux(
                        atmosphere=atmosphere,
                        population=population,
                        grouped=grouped,
                        transfer=transfer,
                        surface_gravity_cgs=population.setup.surface_gravity_cgs,
                    )
                )
            else:
                convective_flux = np.zeros_like(temperature)
                adiabatic_gradient = np.zeros_like(temperature)
                logarithmic_gradient = np.zeros_like(temperature)

            total_flux = transfer.total_flux + convective_flux
            safe_flux = np.where(
                np.isfinite(total_flux) & (total_flux > 0.0),
                total_flux,
                0.25 * target_flux,
            )
            if stage == "radiative":
                # First put the state onto a plausible physical optical
                # coordinate.  Changing T and m simultaneously while the
                # opacity is still far from its EOS fixed point makes the
                # density-opacity feedback chase a moving target.
                flux_log_step = np.zeros_like(temperature)
                temperature_new = temperature.copy()
            else:
                flux_log_step = 0.25 * np.log10(
                    target_flux / np.maximum(safe_flux, 1.0e-300)
                )
                flux_log_step = np.clip(flux_log_step, -0.25, 0.25)
                temperature_new = temperature * 10.0 ** (
                    float(damping) * flux_log_step
                )

            mass_target = integrate_on_depth_grid(
                tau,
                1.0 / np.maximum(grouped.rosseland_opacity, 1.0e-30),
                surface_value=float(
                    tau[0] / np.maximum(grouped.rosseland_opacity[0], 1.0e-30)
                ),
            )
            # The EOS opacity depends on density, so m(tau) is a coupled
            # fixed point rather than a one-shot integral.  Update it during
            # both stages with a bounded log relaxation; holding it fixed
            # through the radiative stage leaves the seed on the wrong optical
            # coordinate for low-gravity and hot stars.
            mass_log_step = np.clip(
                np.log10(np.maximum(mass_target, 1.0e-300))
                - np.log10(np.maximum(column_mass, 1.0e-300)),
                -1.0,
                1.0,
            )
            column_mass_new = column_mass * 10.0 ** (
                float(damping) * mass_log_step
            )
            gas_pressure_new = np.maximum(
                population.setup.surface_gravity_cgs * column_mass_new
                - transfer.integrated_radiation_pressure,
                1.0e-12,
            )
            atmosphere = replace(
                atmosphere,
                column_mass=column_mass_new,
                temperature=temperature_new,
                gas_pressure=gas_pressure_new,
                electron_density=np.maximum(
                    population.runtime_state.electron_density,
                    1.0e-20,
                ),
                rosseland_opacity=np.maximum(grouped.rosseland_opacity, 1.0e-30),
                radiative_acceleration=transfer.radiative_acceleration,
                convective_flux=convective_flux,
            )
            column_mass = column_mass_new
            temperature = temperature_new
            rosseland_opacity = np.maximum(grouped.rosseland_opacity, 1.0e-30)
            diagnostics.append(
                {
                    "stage": stage,
                    "update": int(update_index + 1),
                    "frequency_samples": int(
                        sum(stop - start for start, stop in grouped.group_slices)
                    ),
                    "opacity_groups": int(len(grouped.group_slices)),
                    "maximum_abs_log10_mass_step": float(np.max(np.abs(mass_log_step))),
                    "maximum_abs_log10_temperature_step": float(
                        np.max(np.abs(float(damping) * flux_log_step))
                    ),
                    "flux_ratio_median": float(
                        np.median(safe_flux / max(target_flux, 1.0e-300))
                    ),
                    "flux_ratio_p95": float(
                        np.percentile(safe_flux / max(target_flux, 1.0e-300), 95.0)
                    ),
                    "flux_ratio_min": float(
                        np.min(safe_flux / max(target_flux, 1.0e-300))
                    ),
                    "flux_ratio_max": float(
                        np.max(safe_flux / max(target_flux, 1.0e-300))
                    ),
                    "negative_flux_layers": int(transfer.negative_flux_layers),
                    "convective_layers": int(np.count_nonzero(convective_flux > 0.0)),
                    "adiabatic_gradient_median": float(
                        np.median(adiabatic_gradient)
                    ),
                    "logarithmic_gradient_median": float(
                        np.median(logarithmic_gradient)
                    ),
                    "temperature_range": [
                        float(np.min(temperature)),
                        float(np.max(temperature)),
                    ],
                    "mass_range": [
                        float(np.min(column_mass)),
                        float(np.max(column_mass)),
                    ],
                }
            )

    return HomotopyResult(
        coarse_tau=tau,
        column_mass=column_mass,
        temperature=temperature,
        rosseland_opacity=rosseland_opacity,
        diagnostics={
            "status": "pilot_not_production",
            "coarse_layer_count": int(layer_count),
            "opacity_group_count": int(group_count),
            "frequency_grid_stride": int(frequency_grid_stride),
            "stage_updates": int(stage_updates),
            "damping": float(damping),
            "physical_evaluations": int(physical_evaluations),
            "line_opacity": "deferred_to_unchanged_80_layer_solver",
            "mass_updates": "radiative_and_convective",
            "mass_step_clip_dex": 1.0,
            "update_order": "mass_first_radiative_then_temperature_convective",
            "updates": diagnostics,
        },
    )


def resample_to_production_grid(
    result: HomotopyResult,
    *,
    layer_count: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate the coarse ``(m,T,kappa)`` state onto the production grid."""

    count = int(layer_count)
    if count < 2:
        raise ValueError("production grid needs at least two layers")
    target_tau = 10.0 ** (LOG_TAU_START + 0.125 * np.arange(count, dtype=np.float64))
    source_log_tau = np.log10(np.asarray(result.coarse_tau, dtype=np.float64))
    target_log_tau = np.log10(target_tau)
    mass = np.exp(
        np.interp(target_log_tau, source_log_tau, np.log(result.column_mass))
    )
    temperature = np.exp(
        np.interp(target_log_tau, source_log_tau, np.log(result.temperature))
    )
    opacity = np.exp(
        np.interp(target_log_tau, source_log_tau, np.log(result.rosseland_opacity))
    )
    mass = np.maximum.accumulate(mass)
    for index in range(1, mass.size):
        if mass[index] <= mass[index - 1]:
            mass[index] = np.nextafter(mass[index - 1], np.inf)
    return target_tau, mass, temperature, opacity
