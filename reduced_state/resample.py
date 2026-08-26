"""Represent (m,T)(tau) at an arbitrary internal resolution, then remap back.

Part 3 asks whether the *discreteness* of the current 80-point representation
limits restart quality. The repo's own ``continuity/`` harness already
answered the closely related "is 80 too coarse a quadrature" question (no --
16x refinement changes the closure residual in the 4th decimal place,
``solver-in-the-loop-continuity.md``). What it did not test is what a future
continuous emulator's *internal* resolution does to restart behaviour once
its output is materialized on the grid the solver actually consumes -- the
solver has no depth-count parameter (``layer_count == 80`` is hard-asserted
in ``line_opacity.py``), so any emulator's prediction lands on the fixed
80-point ``tau_std`` grid before the solver ever sees it.

This module simulates that: resample the true ``(m,T)`` curve onto N
log-tau-spaced points (the emulator's hypothetical internal resolution) via
cubic-spline-in-log-log interpolation -- the same method
``continuity/closure.py`` already uses -- then interpolate back onto the
production ``tau_std`` grid. N < 80 loses information a coarser emulator
would also lose; N > 80 is a smoothed/oversampled reference in the limit of
a very fine internal grid, round-tripped through the same two interpolation
steps for a fair comparison. N == 80 is (near-)identity.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from payne_zero_atmosphere.run_setup import standard_rosseland_optical_depth_grid

PRODUCTION_LAYER_COUNT = 80


def _log_log_spline_resample(
    tau_source: np.ndarray, values_source: np.ndarray, tau_target: np.ndarray
) -> np.ndarray:
    """Cubic spline in log(tau) vs log(value), evaluated at ``tau_target``."""

    log_tau_source = np.log10(np.asarray(tau_source, dtype=np.float64))
    log_values_source = np.log10(np.asarray(values_source, dtype=np.float64))
    spline = CubicSpline(log_tau_source, log_values_source, bc_type="natural")
    log_tau_target = np.log10(np.asarray(tau_target, dtype=np.float64))
    return 10.0 ** spline(log_tau_target)


def resample_profile_via_intermediate_grid(
    tau_std: np.ndarray,
    values: np.ndarray,
    n_intermediate: int,
    *,
    tau_target: np.ndarray | None = None,
) -> np.ndarray:
    """Round-trip ``values(tau_std)`` through an ``n_intermediate``-point grid.

    Fits a cubic spline (log-log) through the true ``(tau_std, values)``
    points, evaluates it at ``n_intermediate`` log-spaced points spanning the
    same tau range, fits a second spline through *those* points, and
    evaluates back at ``tau_target`` (``tau_std`` by default -- the
    production grid). This is deliberately two interpolation passes, not
    one, because a real emulator only ever produces the intermediate
    representation; the production grid never sees the true curve directly.
    """

    tau_std = np.asarray(tau_std, dtype=np.float64)
    tau_target = tau_std if tau_target is None else np.asarray(tau_target, dtype=np.float64)
    if n_intermediate < 4:
        raise ValueError("n_intermediate must be >= 4 for a cubic spline")

    log_tau_intermediate = np.linspace(
        np.log10(tau_std[0]), np.log10(tau_std[-1]), int(n_intermediate)
    )
    tau_intermediate = 10.0**log_tau_intermediate
    values_intermediate = _log_log_spline_resample(tau_std, values, tau_intermediate)
    return _log_log_spline_resample(tau_intermediate, values_intermediate, tau_target)


def resample_reduced_state(
    tau_std: np.ndarray,
    column_mass: np.ndarray,
    temperature: np.ndarray,
    n_intermediate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Round-trip (m,T) through an N-point intermediate grid, back onto tau_std.

    Returns ``(column_mass, temperature)`` on the same ``tau_std`` grid as
    the input. At ``n_intermediate == len(tau_std)`` this is close to the
    identity (residual double-interpolation error only); it degrades as
    ``n_intermediate`` decreases and, in the other direction, saturates as
    ``n_intermediate`` grows, exactly mirroring the continuity harness's own
    refinement-scan finding for the closure residual.
    """

    tau_std = np.asarray(tau_std, dtype=np.float64)
    resampled_column_mass = resample_profile_via_intermediate_grid(
        tau_std, column_mass, n_intermediate
    )
    resampled_temperature = resample_profile_via_intermediate_grid(
        tau_std, temperature, n_intermediate
    )
    # column_mass must stay strictly increasing for the solver's own seed
    # validator (run_setup.validate_atmosphere_seed); spline overshoot near
    # sharp features is the only way this could fail, and enforcing it here
    # keeps failures visible in the *reconstruction*, not a mysterious
    # ValueError deep in the solver.
    if np.any(np.diff(resampled_column_mass) <= 0.0):
        raise ValueError(
            f"resampling at n_intermediate={n_intermediate} produced a "
            "non-monotonic column_mass profile"
        )
    return resampled_column_mass, resampled_temperature


def representation_error(
    tau_std: np.ndarray,
    column_mass: np.ndarray,
    temperature: np.ndarray,
    n_intermediate: int,
) -> dict[str, np.ndarray]:
    """Per-layer relative error introduced purely by the N-point round-trip."""

    resampled_m, resampled_t = resample_reduced_state(
        tau_std, column_mass, temperature, n_intermediate
    )
    column_mass = np.asarray(column_mass, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    return {
        "column_mass_relative_error": np.abs(resampled_m - column_mass) / column_mass,
        "temperature_relative_error": np.abs(resampled_t - temperature) / temperature,
    }


def production_tau_grid() -> np.ndarray:
    return standard_rosseland_optical_depth_grid(PRODUCTION_LAYER_COUNT)
