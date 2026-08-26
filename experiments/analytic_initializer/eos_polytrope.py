"""One-shot EOS-driven adiabatic projection for the parity warm start.

The projection is deliberately a solver-in-the-loop operation.  It consumes
the pressure and adiabatic gradient produced by the first exact solver
iteration; it never integrates against the pressure predicted by the analytic
initializer and it has no fitted constants.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imports only for static checking
    from payne_zero_atmosphere.runner import IterationRemap, SingleIterationResult
    from payne_zero_atmosphere.run_setup import RunSetup


DEFAULT_MIN_LAYER = 36


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive ``(start, stop)`` runs from a boolean mask."""

    values = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(values & np.r_[True, ~values[:-1]])
    stops = np.flatnonzero(values & np.r_[~values[1:], True])
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def project_log_temperature(
    temperature: np.ndarray,
    pressure: np.ndarray,
    logarithmic_gradient: np.ndarray,
    adiabatic_gradient: np.ndarray,
    *,
    min_layer: int = DEFAULT_MIN_LAYER,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project convective components onto the EOS adiabatic relation.

    ``pressure`` is the pressure returned by the exact solver.  A convective
    component is selected by the same local Schwarzschild comparison used by
    the solver, ``logarithmic_gradient > adiabatic_gradient``.  The original
    profile is continued by a constant log-temperature offset below each
    component so that both boundaries stay continuous.
    """

    thermal = np.asarray(temperature, dtype=np.float64)
    total_pressure = np.asarray(pressure, dtype=np.float64)
    current_gradient = np.asarray(logarithmic_gradient, dtype=np.float64)
    grad_ad = np.asarray(adiabatic_gradient, dtype=np.float64)
    arrays = (thermal, total_pressure, current_gradient, grad_ad)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("polytrope inputs must be one-dimensional")
    if len({array.size for array in arrays}) != 1 or thermal.size < 2:
        raise ValueError("polytrope inputs must share at least two layers")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("polytrope inputs must be finite")
    if np.any(thermal <= 0.0):
        raise ValueError("polytrope temperature must be strictly positive")
    if np.any(total_pressure <= 0.0) or np.any(np.diff(total_pressure) <= 0.0):
        raise ValueError("polytrope pressure must be positive and strictly increasing")

    layer_floor = max(0, int(min_layer))
    if layer_floor >= thermal.size:
        active = np.zeros(thermal.size, dtype=bool)
    else:
        active = np.arange(thermal.size) >= layer_floor
        active &= current_gradient > grad_ad

    if not np.any(active):
        return thermal.copy(), {
            "projected": False,
            "component_count": 0,
            "components": [],
            "max_abs_delta_lnT": 0.0,
            "adiabatic_residual_max": 0.0,
            "temperature_unchanged": True,
            "min_layer": layer_floor,
        }

    # A valid EOS adiabatic gradient is needed on every step that will be
    # integrated, including the stable layer immediately above a crossing.
    for start, stop in _contiguous_true_runs(active):
        anchor = max(start - 1, 0)
        required = grad_ad[anchor : stop + 1]
        if np.any(required <= 0.0) or np.any(required >= 1.0):
            raise ValueError("EOS adiabatic gradient must lie strictly between 0 and 1")

    log_pressure = np.log(total_pressure)
    original_log_temperature = np.log(thermal)
    projected_log_temperature = original_log_temperature.copy()
    residual_max = 0.0
    components: list[dict[str, int]] = []

    for start, stop in _contiguous_true_runs(active):
        anchor = max(start - 1, 0)
        baseline_at_stop = float(projected_log_temperature[stop])
        for index in range(anchor, stop):
            pressure_step = log_pressure[index + 1] - log_pressure[index]
            slope = 0.5 * (grad_ad[index] + grad_ad[index + 1])
            projected_log_temperature[index + 1] = (
                projected_log_temperature[index] + slope * pressure_step
            )
            residual_max = max(
                residual_max,
                abs(
                    (projected_log_temperature[index + 1]
                     - projected_log_temperature[index])
                    / pressure_step
                    - slope
                ),
            )

        offset = projected_log_temperature[stop] - baseline_at_stop
        if stop + 1 < projected_log_temperature.size:
            projected_log_temperature[stop + 1 :] += offset
        components.append({"start": int(start), "stop": int(stop)})

    projected_temperature = np.exp(projected_log_temperature)
    if not np.all(np.isfinite(projected_temperature)) or np.any(
        projected_temperature <= 0.0
    ):
        raise ValueError("EOS polytrope projection produced a non-positive or non-finite temperature")

    return projected_temperature, {
        "projected": True,
        "component_count": len(components),
        "components": components,
        "max_abs_delta_lnT": float(
            np.max(np.abs(projected_log_temperature - original_log_temperature))
        ),
        "adiabatic_residual_max": float(residual_max),
        "temperature_unchanged": bool(
            np.array_equal(projected_temperature, thermal)
        ),
        "min_layer": layer_floor,
        "adiabatic_gradient_min": float(np.min(grad_ad)),
        "adiabatic_gradient_max": float(np.max(grad_ad)),
    }


def apply_eos_polytrope_projection(
    remapped: "IterationRemap",
    *,
    surface_gravity_cgs: float,
    min_layer: int = DEFAULT_MIN_LAYER,
) -> dict[str, Any]:
    """Apply the projection to one exact solver iteration's remapped state."""

    # Keep the pure projection tests independent of the production Numba
    # stack.  The import is needed only inside the real solver hook.
    from payne_zero_atmosphere.radiative_transfer import remap_to_grid

    finalization = remapped.finalization
    convection = finalization.convection_result
    if convection is None:
        return {
            "projected": False,
            "component_count": 0,
            "components": [],
            "max_abs_delta_lnT": 0.0,
            "adiabatic_residual_max": 0.0,
            "temperature_unchanged": True,
            "reason": "solver did not return a convection result",
            "min_layer": int(min_layer),
        }

    source_tau = np.asarray(finalization.rosseland_optical_depth, dtype=np.float64)
    target_tau = np.asarray(remapped.standard_rosseland_optical_depth, dtype=np.float64)
    current_gradient, _ = remap_to_grid(
        source_tau,
        np.asarray(convection.logarithmic_temperature_pressure_gradient, dtype=np.float64),
        target_tau,
    )
    grad_ad, _ = remap_to_grid(
        source_tau,
        np.asarray(convection.adiabatic_gradient, dtype=np.float64),
        target_tau,
    )
    column_mass = np.asarray(remapped.atmosphere.column_mass, dtype=np.float64)
    turbulent_pressure = np.asarray(remapped.turbulent_pressure, dtype=np.float64)
    surface_radiation_pressure = float(
        finalization.radiative_pressure_state.surface_radiation_pressure_constant
    )
    total_pressure = (
        float(surface_gravity_cgs) * column_mass
        + surface_radiation_pressure
        + turbulent_pressure
    )
    projected, diagnostics = project_log_temperature(
        remapped.atmosphere.temperature,
        total_pressure,
        current_gradient,
        grad_ad,
        min_layer=min_layer,
    )
    if diagnostics["projected"]:
        remapped.atmosphere.temperature[:] = projected
    diagnostics.update(
        {
            "pressure_source": "post_iteration_hydrostatic_total_pressure",
            "surface_gravity_cgs": float(surface_gravity_cgs),
            "surface_radiation_pressure": surface_radiation_pressure,
        }
    )
    return diagnostics


def make_eos_polytrope_hook():
    """Return a one-shot after-iteration hook for the experimental funnel."""

    applied = False

    def hook(
        iteration_index: int,
        setup: "RunSetup",
        step: "SingleIterationResult",
    ) -> dict[str, Any] | None:
        nonlocal applied
        if applied or int(iteration_index) != 1:
            return None
        applied = True
        return apply_eos_polytrope_projection(
            step.remapped,
            surface_gravity_cgs=setup.surface_gravity_cgs,
        )

    return hook


def run_with_eos_polytrope(config):
    """Run the exact solver with the one-shot experimental projection."""

    from payne_zero_atmosphere.runner import _run_atmosphere_model

    return _run_atmosphere_model(
        config,
        after_iteration_hook=make_eos_polytrope_hook(),
    )


def make_eos_polytrope_diagnostic_hook():
    """Return a diagnostic-only hook recording the post-projection state.

    It intentionally never mutates the atmosphere.  The projected temperature
    is evaluated with the same after-iteration total pressure used by the
    experimental hook, then compared with the pre- and post-projection solver
    thermodynamic state.
    """

    recorded = False

    def hook(
        iteration_index: int,
        setup: "RunSetup",
        step: "SingleIterationResult",
    ) -> dict[str, Any] | None:
        nonlocal recorded
        if recorded or int(iteration_index) < 2:
            return None
        recorded = True
        if step.carry.remapped is None:
            raise RuntimeError("diagnostic hook needs an earlier remapped state")

        from payne_zero_atmosphere.radiative_transfer import remap_to_grid

        previous = step.carry.remapped
        previous_finalization = previous.finalization
        previous_convection = previous_finalization.convection_result
        if previous_convection is None:
            return {"projected_diagnostic": False, "reason": "no previous convection"}

        source_tau = np.asarray(
            previous_finalization.rosseland_optical_depth, dtype=np.float64
        )
        target_tau = np.asarray(
            previous.standard_rosseland_optical_depth, dtype=np.float64
        )
        current_gradient, _ = remap_to_grid(
            source_tau,
            np.asarray(
                previous_convection.logarithmic_temperature_pressure_gradient,
                dtype=np.float64,
            ),
            target_tau,
        )
        grad_ad, _ = remap_to_grid(
            source_tau,
            np.asarray(previous_convection.adiabatic_gradient, dtype=np.float64),
            target_tau,
        )
        column_mass = np.asarray(previous.atmosphere.column_mass, dtype=np.float64)
        total_pressure = (
            float(setup.surface_gravity_cgs) * column_mass
            + float(
                previous_finalization.radiative_pressure_state.surface_radiation_pressure_constant
            )
            + np.asarray(previous.turbulent_pressure, dtype=np.float64)
        )
        projected_temperature, projection = project_log_temperature(
            previous.atmosphere.temperature,
            total_pressure,
            current_gradient,
            grad_ad,
        )
        if not projection["projected"]:
            projection["projected_diagnostic"] = True
            return projection

        current_tau = np.asarray(
            step.transfer.rosseland_optical_depth, dtype=np.float64
        )
        current_convection = step.transfer.finalization.convection_result
        if current_convection is None:
            raise RuntimeError("diagnostic hook needs current convection results")
        post_gradient, _ = remap_to_grid(
            current_tau,
            np.asarray(
                current_convection.logarithmic_temperature_pressure_gradient,
                dtype=np.float64,
            ),
            target_tau,
        )
        post_ad, _ = remap_to_grid(
            current_tau,
            np.asarray(
                current_convection.adiabatic_gradient,
                dtype=np.float64,
            ),
            target_tau,
        )
        flux = np.asarray(
            step.transfer.temperature_correction_state.integrated_eddington_flux,
            dtype=np.float64,
        )
        remap_flux, _ = remap_to_grid(current_tau, flux, target_tau)
        target_flux = (
            5.6697e-5 / 12.5664 * float(setup.effective_temperature) ** 4
        )
        convective = np.asarray(
            step.transfer.opacity_state.population_state.setup.atmosphere.convective_flux,
            dtype=np.float64,
        )
        components = projection["components"]
        starts = [item["start"] for item in components]
        stops = [item["stop"] for item in components]
        first_start = min(starts)
        last_stop = max(stops)
        component_slice = slice(first_start, last_stop + 1)
        projection.update(
            {
                "projected_diagnostic": True,
                "pre_gradient_mean_in_components": float(
                    np.mean(current_gradient[component_slice])
                ),
                "post_gradient_mean_in_components": float(
                    np.mean(post_gradient[component_slice])
                ),
                "pre_grad_ad_mean_in_components": float(
                    np.mean(grad_ad[component_slice])
                ),
                "post_grad_ad_mean_in_components": float(
                    np.mean(post_ad[component_slice])
                ),
                "post_projected_temperature_range": [
                    float(np.min(projected_temperature[component_slice])),
                    float(np.max(projected_temperature[component_slice])),
                ],
                "flux_relative_error_mean_in_components": float(
                    np.mean((remap_flux - target_flux) / target_flux)
                ),
                "convective_flux_mean_in_components": float(
                    np.mean(convective[component_slice])
                ),
            }
        )
        return projection

    return hook


def run_with_eos_polytrope_diagnostics(config):
    """Run the exact solver with the diagnostic-only projection hook."""

    from payne_zero_atmosphere.runner import _run_atmosphere_model

    return _run_atmosphere_model(
        config,
        after_iteration_hook=make_eos_polytrope_diagnostic_hook(),
    )
