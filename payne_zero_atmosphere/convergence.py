"""Atmosphere convergence diagnostics used by the production runner."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def max_normalized_column_delta(
    before: np.ndarray,
    after: np.ndarray,
    *,
    floor: float = 1.0e-300,
    symmetric: bool = False,
) -> float:
    """Return the largest layer-wise normalized column change."""

    before_array = np.asarray(before, dtype=np.float64)
    after_array = np.asarray(after, dtype=np.float64)
    if before_array.shape != after_array.shape or before_array.size == 0:
        return float("nan")

    if symmetric:
        denominator = np.maximum.reduce(
            [
                np.abs(before_array),
                np.abs(after_array),
                np.full(before_array.shape, floor, dtype=np.float64),
            ]
        )
    else:
        denominator = np.maximum(np.abs(before_array), floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.abs(after_array - before_array) / denominator
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def deep_layer_relative_temperature_change(
    before: np.ndarray, after: np.ndarray
) -> float:
    """Return the maximum relative temperature change in the deep layers.

    The production threshold is evaluated over layers 39 through ``layers - 6``
    on the standard 80-layer grid. Smaller grids use every layer.
    """

    before_array = np.asarray(before, dtype=np.float64)
    after_array = np.asarray(after, dtype=np.float64)
    if (
        before_array.shape != after_array.shape
        or before_array.ndim != 1
        or before_array.size == 0
    ):
        return float("nan")

    layers = before_array.size
    start = 39
    stop = layers - 5
    if stop - start < 1:
        start, stop = 0, layers

    old_temperature = before_array[start:stop]
    new_temperature = after_array[start:stop]
    if not np.all(np.isfinite(old_temperature)) or not np.all(
        np.isfinite(new_temperature)
    ):
        return float("inf")
    with np.errstate(divide="ignore", invalid="ignore"):
        fractional_temperature_change = np.abs(
            new_temperature - old_temperature
        ) / np.abs(new_temperature)
    if not np.all(np.isfinite(fractional_temperature_change)):
        return float("inf")
    return float(np.max(fractional_temperature_change))


def temperature_changes_within_limits(
    *,
    deep_layer_change: float,
    all_layer_change: float,
    maximum_deep_layer_change: float,
    maximum_all_layer_change: float | None,
) -> bool:
    """Evaluate the declared structural fixed-point stopping limits.

    The optional all-layer test catches slowly relaxing upper layers seen by
    strong-line cores while retaining the historical deep-only behavior when
    no all-layer limit is requested.
    """

    deep_ok = np.isfinite(deep_layer_change) and deep_layer_change < float(
        maximum_deep_layer_change
    )
    if maximum_all_layer_change is None:
        return bool(deep_ok)
    all_ok = np.isfinite(all_layer_change) and all_layer_change < float(
        maximum_all_layer_change
    )
    return bool(deep_ok and all_ok)


@dataclass(frozen=True)
class ConvergenceStopDecision:
    """What one iteration's residual does to the solver's stop state."""

    consecutive_converged_iterations: int
    converged: bool
    force_exact_opacity: bool


def evaluate_convergence_stop(
    *,
    enable_convergence_stop: bool,
    iteration_index: int,
    minimum_iterations_before_convergence: int,
    required_consecutive_converged_iterations: int,
    temperature_change_within_limit: bool,
    opacity_recomputed: bool,
    consecutive_converged_iterations: int,
) -> ConvergenceStopDecision:
    """Apply the stop policy to one completed iteration.

    THE OPACITY-LAGGING INVARIANT, enforced here and nowhere else:

        ``converged`` is never ``True`` for an iteration with
        ``opacity_recomputed=False``.

    A lagged iteration measured its temperature change against an opacity
    operator built from an *earlier* atmosphere, so its residual is not
    evidence about the true fixed point. The policy therefore treats a lagged
    iteration as unable to create confidence but still able to destroy it:

    - it never increments ``consecutive_converged_iterations`` and never sets
      ``converged``;
    - it still resets the counter when the state is visibly still moving,
      because a large change is real information regardless of which operator
      produced it;
    - when it *looks* converged the counter is left untouched and
      ``force_exact_opacity`` is raised, which pushes the next iteration back
      onto exact opacity so the candidate fixed point is re-tested against the
      true operator instead of being accepted or discarded on stale evidence.

    With opacity lagging off, ``opacity_recomputed`` is always ``True`` and
    this reduces, branch for branch, to the historical stop policy.
    """

    if not opacity_recomputed:
        if not temperature_change_within_limit:
            return ConvergenceStopDecision(
                consecutive_converged_iterations=0,
                converged=False,
                force_exact_opacity=False,
            )
        return ConvergenceStopDecision(
            consecutive_converged_iterations=int(consecutive_converged_iterations),
            converged=False,
            force_exact_opacity=True,
        )

    if (
        enable_convergence_stop
        and int(iteration_index) >= int(minimum_iterations_before_convergence)
        and temperature_change_within_limit
    ):
        consecutive = int(consecutive_converged_iterations) + 1
    else:
        consecutive = 0
    converged = bool(enable_convergence_stop) and consecutive >= int(
        required_consecutive_converged_iterations
    )
    return ConvergenceStopDecision(
        consecutive_converged_iterations=consecutive,
        converged=converged,
        force_exact_opacity=False,
    )
