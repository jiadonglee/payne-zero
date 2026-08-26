"""A positive local-opacity closure for the H3 discovery probe.

The atmosphere equation on a Rosseland grid is

    d m / d tau = 1 / kappa_R.

This module tests whether a small, label-conditioned approximation to the
local opacity is enough to integrate that equation.  The closure is not
presented as a new opacity law: it is a physics-constrained empirical
closure.  It uses local ``(T, P)`` together with the five labels and optical
depth, predicts ``log10(kappa_R)`` with a low-order polynomial, and exponentiates
the result so opacity stays positive.  Temperature-regime blending is smooth
at evaluation time; hard regimes are used only to keep the offline fit
well-conditioned.

The explicit separation between fitting kappa and integrating m is useful:
the first question is whether the local variables carry enough information;
the second is whether their errors accumulate acceptably in column mass.
Neither result is a production solver bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates import temperature_regime_weights, temperature_regimes  # noqa: F401
from .discovery import Corpus, Split, polynomial_exponents, polynomial_features
from .profile_closure import integrate_mass_from_opacity


LOCAL_FEATURE_NAMES = (
    "log10_temperature_K",
    "log10_gas_pressure_dyn_cm2",
    "temperature_coordinate_5040_over_Teff",
    "log10_surface_gravity_cgs",
    "metallicity",
    "alpha_enhancement",
    "log10_microturbulence_km_s",
    "log10_rosseland_tau",
)


@dataclass(frozen=True)
class LocalOpacityParameters:
    """Constants for the offline local-opacity closure."""

    degree: int
    exponents: np.ndarray
    feature_center: np.ndarray
    feature_scale: np.ndarray
    coefficients_by_regime: np.ndarray
    temperature_boundaries: tuple[float, float] = (5500.0, 7500.0)
    smoothing_width_K: float = 250.0
    opacity_floor_dex: float = -12.0
    opacity_ceiling_dex: float = 6.0

    @property
    def term_count(self) -> int:
        return int(self.exponents.shape[0])

    @property
    def coefficient_count(self) -> int:
        return int(np.count_nonzero(self.coefficients_by_regime))


def local_opacity_features(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Build dimensionless local features for one or many depth rows.

    ``labels`` can be ``(N, 5)`` or ``(rows, 5)``.  The other arguments must
    broadcast to the same one-dimensional row count.  The features are
    deliberately named in physical units before standardization; the fitted
    polynomial therefore remains inspectable and cheap to evaluate.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("labels must have shape (N, 5)")
    local_temperature = np.asarray(temperature, dtype=np.float64).reshape(-1)
    pressure = np.asarray(gas_pressure, dtype=np.float64).reshape(-1)
    depth = np.asarray(tau, dtype=np.float64).reshape(-1)
    if not (local_temperature.size == pressure.size == depth.size == values.shape[0]):
        raise ValueError("local features must have the same row count")
    if (
        np.any(~np.isfinite(local_temperature))
        or np.any(~np.isfinite(pressure))
        or np.any(~np.isfinite(depth))
        or np.any(local_temperature <= 0.0)
        or np.any(pressure <= 0.0)
        or np.any(depth <= 0.0)
    ):
        raise ValueError("local features must be finite and positive where logged")
    if np.any(~np.isfinite(values)) or np.any(values[:, 0] <= 0.0) or np.any(values[:, 4] <= 0.0):
        raise ValueError("labels must be finite with positive Teff and microturbulence")
    return np.column_stack(
        (
            np.log10(local_temperature),
            np.log10(pressure),
            5040.0 / values[:, 0],
            values[:, 1],
            values[:, 2],
            values[:, 3],
            np.log10(values[:, 4]),
            np.log10(depth),
        )
    )


def _sample_training_rows(
    corpus: Corpus,
    split: Split,
    *,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose whole stars, then all 80 layers, to preserve depth structure."""

    if max_rows < corpus.layers:
        raise ValueError("max_rows must be at least one complete profile")
    star_count = min(split.train.size, max(1, int(max_rows // corpus.layers)))
    generator = np.random.default_rng(int(seed))
    stars = np.sort(generator.choice(split.train, size=star_count, replace=False))
    return np.repeat(stars, corpus.layers), np.tile(np.arange(corpus.layers), star_count)


def _design_for_rows(
    corpus: Corpus,
    star_indices: np.ndarray,
    layer_indices: np.ndarray,
    *,
    degree: int,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    exponents: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = local_opacity_features(
        corpus.labels[star_indices],
        corpus.temperature[star_indices, layer_indices],
        corpus.gas_pressure[star_indices, layer_indices],
        corpus.tau[layer_indices],
    )
    if exponents is None:
        exponents = polynomial_exponents(raw.shape[1], degree)
    design, used_center, used_scale = polynomial_features(
        raw, exponents, center=center, scale=scale
    )
    target = np.log10(corpus.rosseland_opacity[star_indices, layer_indices])
    return design, target, used_center, used_scale


def fit_local_opacity_parameters(
    corpus: Corpus,
    split: Split,
    *,
    degree: int = 2,
    max_training_rows: int = 120_000,
    ridge: float = 1.0e-6,
    seed: int = 20260816,
) -> tuple[LocalOpacityParameters, dict[str, float]]:
    """Fit a regime-wise local opacity closure on the training split."""

    if degree < 0:
        raise ValueError("degree must be non-negative")
    stars, layers = _sample_training_rows(
        corpus, split, max_rows=max_training_rows, seed=seed
    )
    raw = local_opacity_features(
        corpus.labels[stars], corpus.temperature[stars, layers], corpus.gas_pressure[stars, layers], corpus.tau[layers]
    )
    exponents = polynomial_exponents(raw.shape[1], degree)
    design, feature_center, feature_scale = polynomial_features(raw, exponents)
    target = np.log10(corpus.rosseland_opacity[stars, layers])
    regimes = temperature_regimes(corpus.labels)
    coefficients = np.zeros((3, exponents.shape[0]), dtype=np.float64)
    star_regimes = regimes[stars]
    for regime_index in range(3):
        mask = star_regimes == regime_index
        if int(mask.sum()) < exponents.shape[0]:
            raise ValueError(f"not enough sampled rows in regime {regime_index}")
        gram = design[mask].T @ design[mask]
        penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients[regime_index] = np.linalg.solve(
            gram + penalty,
            design[mask].T @ target[mask],
        )

    parameters = LocalOpacityParameters(
        degree=int(degree),
        exponents=exponents,
        feature_center=feature_center,
        feature_scale=feature_scale,
        coefficients_by_regime=coefficients,
    )
    fit_prediction = _predict_rows(
        corpus, stars, layers, parameters, smooth=False
    )
    residual = fit_prediction - target
    metrics = {
        "training_row_count": float(target.size),
        "term_count": float(parameters.term_count),
        "coefficient_count": float(parameters.coefficient_count),
        "training_rmse_dex": float(np.sqrt(np.mean(residual**2))),
        "training_p95_dex": float(np.percentile(np.abs(residual), 95.0)),
    }
    return parameters, metrics


def _predict_rows(
    corpus: Corpus,
    star_indices: np.ndarray,
    layer_indices: np.ndarray,
    parameters: LocalOpacityParameters,
    *,
    smooth: bool,
) -> np.ndarray:
    raw = local_opacity_features(
        corpus.labels[star_indices],
        corpus.temperature[star_indices, layer_indices],
        corpus.gas_pressure[star_indices, layer_indices],
        corpus.tau[layer_indices],
    )
    design, _, _ = polynomial_features(
        raw,
        parameters.exponents,
        center=parameters.feature_center,
        scale=parameters.feature_scale,
    )
    values = design @ parameters.coefficients_by_regime.T
    if smooth:
        weights = temperature_regime_weights(
            corpus.labels[star_indices],
            boundaries=parameters.temperature_boundaries,
            width_K=parameters.smoothing_width_K,
        )
        values = np.sum(values * weights, axis=1)
    else:
        values = values[np.arange(values.shape[0]), temperature_regimes(corpus.labels[star_indices])]
    return np.clip(values, parameters.opacity_floor_dex, parameters.opacity_ceiling_dex)


def predict_local_log_opacity(
    labels: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray,
    tau: np.ndarray,
    parameters: LocalOpacityParameters,
    *,
    smooth: bool = True,
) -> np.ndarray:
    """Evaluate the closure, returning a positive-opacity logarithm."""

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    local_temperature = np.asarray(temperature, dtype=np.float64)
    pressure = np.asarray(gas_pressure, dtype=np.float64)
    depth = np.asarray(tau, dtype=np.float64)
    if local_temperature.ndim == 1:
        local_temperature = local_temperature[None, :]
    if pressure.ndim == 1:
        pressure = pressure[None, :]
    if local_temperature.shape != pressure.shape or local_temperature.shape[0] != values.shape[0]:
        raise ValueError("temperature and gas_pressure must have shape (N, layers)")
    if local_temperature.shape[1] != depth.size:
        raise ValueError("profile arrays must match tau length")
    stars = np.repeat(np.arange(values.shape[0]), depth.size)
    layers = np.tile(np.arange(depth.size), values.shape[0])
    raw = local_opacity_features(
        values[stars], local_temperature.reshape(-1), pressure.reshape(-1), depth[layers]
    )
    design, _, _ = polynomial_features(
        raw,
        parameters.exponents,
        center=parameters.feature_center,
        scale=parameters.feature_scale,
    )
    regime_values = design @ parameters.coefficients_by_regime.T
    if smooth:
        weights = temperature_regime_weights(
            values,
            boundaries=parameters.temperature_boundaries,
            width_K=parameters.smoothing_width_K,
        )
        predicted = np.sum(regime_values * weights[stars], axis=1)
    else:
        predicted = regime_values[
            np.arange(regime_values.shape[0]), temperature_regimes(values)[stars]
        ]
    return np.clip(
        predicted.reshape(values.shape[0], depth.size),
        parameters.opacity_floor_dex,
        parameters.opacity_ceiling_dex,
    )


def integrate_self_consistent_mass(
    labels: np.ndarray,
    temperature: np.ndarray,
    tau: np.ndarray,
    parameters: LocalOpacityParameters,
    *,
    iterations: int = 6,
    surface_mass: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the cheap hydrostatic-opacity fixed point on a fixed tau grid.

    The light closure uses ``P = g m`` as its pressure scale.  This is the
    intended H3 runtime skeleton: opacity evaluation, a positive trapezoidal
    integral, and a few log-space fixed-point updates.  Radiation pressure and
    the full EOS are intentionally absent here and remain later solver gates.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    local_temperature = np.asarray(temperature, dtype=np.float64)
    if local_temperature.ndim == 1:
        local_temperature = local_temperature[None, :]
    depth = np.asarray(tau, dtype=np.float64)
    if local_temperature.shape != (values.shape[0], depth.size):
        raise ValueError("temperature must have shape (N, len(tau))")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    gravity = 10.0 ** values[:, 1]
    # The first pass uses the optical-depth seed and a hydrostatic pressure
    # scale.  Subsequent passes are log-relaxed to avoid an opacity runaway.
    initial_pressure = np.maximum(gravity[:, None] * np.maximum(depth[None, :], 1.0e-30), 1.0e-12)
    log_opacity = predict_local_log_opacity(
        values, local_temperature, initial_pressure, depth, parameters
    )
    mass = integrate_mass_from_opacity(depth, log_opacity, surface_mass=surface_mass)
    for _ in range(int(iterations) - 1):
        pressure = np.maximum(gravity[:, None] * mass, 1.0e-12)
        log_opacity = predict_local_log_opacity(
            values, local_temperature, pressure, depth, parameters
        )
        candidate = integrate_mass_from_opacity(
            depth, log_opacity, surface_mass=surface_mass
        )
        mass = 10.0 ** (0.5 * (np.log10(np.maximum(mass, 1.0e-300)) + np.log10(np.maximum(candidate, 1.0e-300))))
    pressure = np.maximum(gravity[:, None] * mass, 1.0e-12)
    log_opacity = predict_local_log_opacity(
        values, local_temperature, pressure, depth, parameters
    )
    return mass, log_opacity


def profile_invariants(
    column_mass: np.ndarray,
    temperature: np.ndarray,
) -> dict[str, int]:
    """Count violations that must be zero before any solver smoke test."""

    mass = np.asarray(column_mass, dtype=np.float64)
    thermal = np.asarray(temperature, dtype=np.float64)
    return {
        "nonfinite_mass_profiles": int(np.sum(~np.all(np.isfinite(mass), axis=1))),
        "nonpositive_mass_profiles": int(np.sum(~np.all(mass > 0.0, axis=1))),
        "nonmonotone_mass_profiles": int(np.sum(~np.all(np.diff(mass, axis=1) > 0.0, axis=1))),
        "nonfinite_temperature_profiles": int(np.sum(~np.all(np.isfinite(thermal), axis=1))),
        "nonpositive_temperature_profiles": int(np.sum(~np.all(thermal > 0.0, axis=1))),
    }
