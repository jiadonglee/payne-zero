"""Low-dimensional, solver-free physical residual initializer prototype.

This is the next bounded experiment after the coarse homotopy pilot.  It does
not fit atmosphere targets and does not load an emulator.  A small set of
anchor/increment coefficients defines positive, strictly increasing ``T`` and
``m`` on the 32-point grid.  Each trial evaluates the real EOS/continuum
opacity and the grouped transfer, then minimizes flux and Rosseland-depth
residuals with bounded Gauss--Newton steps through ``scipy``.

The prototype intentionally stops at continuum opacity on the coarse grid;
the unchanged 80-layer production solver still opens the existing line
opacity path after the seed is materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .physical_homotopy import (
    COARSE_LAYER_COUNT,
    DEFAULT_FREQUENCY_GRID_STRIDE,
    HomotopyResult,
    _evaluate_real_continuum,
    _make_seed_atmosphere,
    coarse_rosseland_tau,
    resample_to_production_grid,
)


DEFAULT_KNOT_COUNT = 5
DEFAULT_MAX_NFEV = 8
DEFAULT_REGULARIZATION = 0.02
_REFERENCE_OPACITY = 0.34
_FOUR_PI_REFERENCE = 12.5664
_STEFAN_BOLTZMANN_REFERENCE = 5.6697e-5


@dataclass(frozen=True)
class PhysicalResidualResult:
    """Optimized coarse seed plus auditable residual diagnostics."""

    coarse_tau: np.ndarray
    column_mass: np.ndarray
    temperature: np.ndarray
    rosseland_opacity: np.ndarray
    coefficients: np.ndarray
    diagnostics: dict[str, Any]


def _increment_basis(tau: np.ndarray, knot_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return interpolation weights for log-increment corrections."""

    depth = np.asarray(tau, dtype=np.float64).reshape(-1)
    if depth.size < 4 or np.any(~np.isfinite(depth)) or np.any(np.diff(depth) <= 0.0):
        raise ValueError("tau must be finite and strictly increasing")
    count = int(knot_count)
    if count < 2:
        raise ValueError("knot_count must be at least two")
    log_depth = np.log(depth)
    knots = np.linspace(log_depth[0], log_depth[-1], count)
    midpoints = 0.5 * (log_depth[1:] + log_depth[:-1])
    basis = np.zeros((depth.size - 1, count), dtype=np.float64)
    for row, value in enumerate(midpoints):
        right = int(np.searchsorted(knots, value, side="right"))
        right = min(max(right, 1), count - 1)
        left = right - 1
        fraction = (value - knots[left]) / (knots[right] - knots[left])
        basis[row, left] = 1.0 - fraction
        basis[row, right] = fraction
    return knots, basis


def _positive_monotone_profile(
    base: np.ndarray,
    *,
    anchor_dex: float,
    increment_dex: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Apply smooth log-increment corrections while preserving monotonicity."""

    reference = np.asarray(base, dtype=np.float64).reshape(-1)
    if np.any(reference <= 0.0) or not np.all(np.isfinite(reference)):
        raise ValueError("base profile must be finite and positive")
    correction = np.asarray(increment_dex, dtype=np.float64).reshape(-1)
    if correction.shape != (basis.shape[1],):
        raise ValueError("increment coefficient count does not match basis")
    base_log_increment = np.diff(np.log(reference))
    if np.any(base_log_increment <= 0.0):
        raise ValueError("base profile must be strictly increasing")
    corrected_log_increment = base_log_increment * 10.0 ** np.clip(
        basis @ correction, -2.0, 2.0
    )
    log_profile = np.log(reference[0]) + np.log(10.0) * float(anchor_dex)
    log_profile = log_profile + np.concatenate(
        [np.zeros(1), np.cumsum(corrected_log_increment)]
    )
    return np.exp(np.clip(log_profile, -700.0, 700.0))


def _decode_coefficients(
    coefficients: np.ndarray,
    *,
    tau: np.ndarray,
    effective_temperature: float,
    basis: np.ndarray,
    knot_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the 2*(1+K) bounded coefficients into ``(m,T)``."""

    values = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    expected = 2 * (1 + int(knot_count))
    if values.shape != (expected,):
        raise ValueError(f"coefficients must have shape ({expected},)")
    grey_temperature = float(effective_temperature) * (
        0.75 * (tau + 2.0 / 3.0)
    ) ** 0.25
    grey_mass = tau / _REFERENCE_OPACITY
    temperature = _positive_monotone_profile(
        grey_temperature,
        anchor_dex=float(values[0]),
        increment_dex=values[1 : 1 + knot_count],
        basis=basis,
    )
    offset = 1 + knot_count
    mass = _positive_monotone_profile(
        grey_mass,
        anchor_dex=float(values[offset]),
        increment_dex=values[offset + 1 : offset + 1 + knot_count],
        basis=basis,
    )
    return mass, temperature


def _residual_components(
    labels: np.ndarray,
    coefficients: np.ndarray,
    *,
    tau: np.ndarray,
    basis: np.ndarray,
    knot_count: int,
    frequency_grid_stride: int,
    regularization: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate physical residual blocks for one coefficient vector."""

    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    target_flux = (
        _STEFAN_BOLTZMANN_REFERENCE
        / _FOUR_PI_REFERENCE
        * float(values[0]) ** 4
    )
    try:
        mass, temperature = _decode_coefficients(
            coefficients,
            tau=tau,
            effective_temperature=float(values[0]),
            basis=basis,
            knot_count=knot_count,
        )
        atmosphere = _make_seed_atmosphere(
            values,
            tau,
            mass,
            temperature,
            np.full(tau.size, _REFERENCE_OPACITY, dtype=np.float64),
        )
        _config, _population, grouped, transfer = _evaluate_real_continuum(
            atmosphere,
            frequency_grid_stride=int(frequency_grid_stride),
            group_count=4,
            target_integrated_flux=target_flux,
        )
        from payne_zero_atmosphere.radiative_transfer import integrate_on_depth_grid

        rosseland = np.maximum(grouped.rosseland_opacity, 1.0e-30)
        physical_tau = integrate_on_depth_grid(
            mass,
            rosseland,
            surface_value=float(mass[0] * rosseland[0]),
        )
        flux_ratio = np.asarray(transfer.total_flux, dtype=np.float64) / target_flux
        local_tau_ratio = (
            0.5 * (rosseland[1:] + rosseland[:-1]) * np.diff(mass) / np.diff(tau)
        )
        flux_residual = np.log10(np.clip(flux_ratio, 1.0e-4, 1.0e4))
        tau_residual = np.log10(np.clip(local_tau_ratio, 1.0e-4, 1.0e4))
        negative_flux_penalty = np.where(transfer.total_flux > 0.0, 0.0, 1.0)
        coefficient_scale = np.maximum(
            np.concatenate(
                [np.full(1 + knot_count, 0.25), np.full(1 + knot_count, 0.10)]
            ),
            1.0e-12,
        )
        residual = np.concatenate(
            [
                flux_residual,
                tau_residual,
                negative_flux_penalty,
                float(regularization) * np.asarray(coefficients) / coefficient_scale,
            ]
        )
        diagnostics = {
            "finite": bool(
                np.all(np.isfinite(flux_ratio))
                and np.all(np.isfinite(local_tau_ratio))
                and np.all(np.isfinite(mass))
                and np.all(np.isfinite(temperature))
            ),
            "flux_ratio_median": float(np.median(flux_ratio)),
            "flux_ratio_p95": float(np.percentile(np.abs(flux_ratio), 95.0)),
            "local_tau_ratio_median": float(np.median(local_tau_ratio)),
            "local_tau_ratio_p95": float(
                np.percentile(np.abs(local_tau_ratio), 95.0)
            ),
            "cumulative_tau_ratio_p95": float(
                np.percentile(np.abs(physical_tau / tau), 95.0)
            ),
            "negative_flux_layers": int(np.count_nonzero(transfer.total_flux <= 0.0)),
            "mass_range": [float(np.min(mass)), float(np.max(mass))],
            "temperature_range": [
                float(np.min(temperature)),
                float(np.max(temperature)),
            ],
            "rosseland_opacity_range": [
                float(np.min(grouped.rosseland_opacity)),
                float(np.max(grouped.rosseland_opacity)),
            ],
        }
        if not diagnostics["finite"]:
            raise FloatingPointError("non-finite physical residual state")
        return residual, diagnostics
    except Exception:
        penalty = np.full(3 * tau.size - 1 + coefficients.size, 10.0)
        return penalty, {
            "finite": False,
            "flux_ratio_median": None,
            "flux_ratio_p95": None,
            "local_tau_ratio_median": None,
            "local_tau_ratio_p95": None,
            "cumulative_tau_ratio_p95": None,
            "negative_flux_layers": None,
            "error_state": True,
        }


def physical_residual_seed(
    labels: np.ndarray,
    *,
    knot_count: int = DEFAULT_KNOT_COUNT,
    frequency_grid_stride: int = DEFAULT_FREQUENCY_GRID_STRIDE,
    max_nfev: int = DEFAULT_MAX_NFEV,
    regularization: float = DEFAULT_REGULARIZATION,
) -> PhysicalResidualResult:
    """Optimize a twelve-coefficient continuum physical residual seed."""

    values = np.asarray(labels, dtype=np.float64).reshape(-1)
    if values.shape != (5,) or np.any(~np.isfinite(values)):
        raise ValueError("labels must be one finite five-label vector")
    if int(knot_count) < 2:
        raise ValueError("knot_count must be at least two")
    if int(max_nfev) < 1:
        raise ValueError("max_nfev must be positive")
    tau = coarse_rosseland_tau(COARSE_LAYER_COUNT)
    _knots, basis = _increment_basis(tau, int(knot_count))
    parameter_count = 2 * (1 + int(knot_count))
    initial = np.zeros(parameter_count, dtype=np.float64)
    lower = np.concatenate(
        [
            np.full(1, -0.50),
            np.full(knot_count, -0.10),
            np.full(1, -1.50),
            np.full(knot_count, -2.50),
        ]
    )
    upper = np.concatenate(
        [
            np.full(1, 0.50),
            np.full(knot_count, 0.10),
            np.full(1, 1.50),
            np.full(knot_count, 2.50),
        ]
    )
    evaluation_count = 0
    last_diagnostics: dict[str, Any] = {}

    def objective(coefficients: np.ndarray) -> np.ndarray:
        nonlocal evaluation_count, last_diagnostics
        evaluation_count += 1
        residual, last_diagnostics = _residual_components(
            values,
            coefficients,
            tau=tau,
            basis=basis,
            knot_count=int(knot_count),
            frequency_grid_stride=int(frequency_grid_stride),
            regularization=float(regularization),
        )
        return residual

    initial_residual = objective(initial)
    optimization = least_squares(
        objective,
        initial,
        bounds=(lower, upper),
        method="trf",
        x_scale="jac",
        max_nfev=int(max_nfev),
        ftol=1.0e-4,
        xtol=1.0e-4,
        gtol=1.0e-4,
    )
    final_residual = objective(optimization.x)
    mass, temperature = _decode_coefficients(
        optimization.x,
        tau=tau,
        effective_temperature=float(values[0]),
        basis=basis,
        knot_count=int(knot_count),
    )
    atmosphere = _make_seed_atmosphere(
        values,
        tau,
        mass,
        temperature,
        np.full(tau.size, _REFERENCE_OPACITY, dtype=np.float64),
    )
    _config, _population, grouped, _transfer = _evaluate_real_continuum(
        atmosphere,
        frequency_grid_stride=int(frequency_grid_stride),
        group_count=4,
        target_integrated_flux=(
            _STEFAN_BOLTZMANN_REFERENCE
            / _FOUR_PI_REFERENCE
            * float(values[0]) ** 4
        ),
    )
    diagnostics = {
        "status": "prototype_not_production",
        "coarse_layer_count": int(tau.size),
        "opacity_group_count": 4,
        "frequency_grid_stride": int(frequency_grid_stride),
        "knot_count": int(knot_count),
        "parameter_count": int(parameter_count),
        "max_nfev": int(max_nfev),
        "evaluation_count": int(evaluation_count),
        "optimizer_status": int(optimization.status),
        "optimizer_message": str(optimization.message),
        "optimizer_success": bool(optimization.success),
        "initial_residual_l2": float(np.linalg.norm(initial_residual)),
        "final_residual_l2": float(np.linalg.norm(final_residual)),
        "initial_residual_rms": float(np.sqrt(np.mean(initial_residual**2))),
        "final_residual_rms": float(np.sqrt(np.mean(final_residual**2))),
        "final_physical": last_diagnostics,
        "line_opacity": "deferred_to_unchanged_80_layer_solver",
    }
    return PhysicalResidualResult(
        coarse_tau=tau,
        column_mass=mass,
        temperature=temperature,
        rosseland_opacity=np.maximum(grouped.rosseland_opacity, 1.0e-30),
        coefficients=np.asarray(optimization.x, dtype=np.float64),
        diagnostics=diagnostics,
    )


def resample_residual_seed(
    result: PhysicalResidualResult,
    *,
    layer_count: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adapt the residual prototype to the existing solver bridge."""

    homotopy_like = HomotopyResult(
        coarse_tau=result.coarse_tau,
        column_mass=result.column_mass,
        temperature=result.temperature,
        rosseland_opacity=result.rosseland_opacity,
        diagnostics=result.diagnostics,
    )
    return resample_to_production_grid(homotopy_like, layer_count=layer_count)
