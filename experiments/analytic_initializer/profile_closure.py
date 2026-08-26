"""Diagnostic opacity-profile closure for the second hypothesis.

This module is an offline probe, not the final compact formula.  It uses a
small number of SVD depth modes inside three temperature regimes to answer a
specific physical question: if an approximate local Rosseland-opacity profile
were available, would integrating it recover column mass?  The answer tells
us whether the next search should focus on opacity physics or on the surface
boundary condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates import (
    REGIME_BOUNDARIES,
    temperature_regime_weights,
    temperature_regimes,
)
from .discovery import Corpus, Split, label_features, polynomial_exponents, polynomial_features


@dataclass(frozen=True)
class ProfileClosureParameters:
    """Low-rank parameters for one label-conditioned target profile."""

    degree: int
    components: int
    exponents: np.ndarray
    feature_center: np.ndarray
    feature_scale: np.ndarray
    target_center_by_regime: np.ndarray
    basis_by_regime: np.ndarray
    coefficients_by_regime: np.ndarray
    regime_boundaries: tuple[float, float] = REGIME_BOUNDARIES
    # Zero keeps the original hard regime assignment.  A positive width blends
    # the three regime predictions instead, which is what ``temperature_regimes``
    # asks for in its own docstring: the seams are an artefact of the diagnostic
    # split, and the default 7500 K seam sits inside the radiative/convective
    # transition where the deep error peaks.
    smoothing_width_K: float = 0.0

    @property
    def term_count(self) -> int:
        return int(self.coefficients_by_regime.size)


def integrate_mass_from_opacity(
    tau: np.ndarray,
    log_opacity: np.ndarray,
    *,
    surface_mass: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate ``dm/dtau = 1/kappa`` on the fixed production grid."""

    depth = np.asarray(tau, dtype=np.float64)
    log_kappa = np.asarray(log_opacity, dtype=np.float64)
    if log_kappa.ndim != 2 or log_kappa.shape[1] != depth.size:
        raise ValueError("log_opacity must have shape (N, len(tau))")
    if np.any(np.diff(depth) <= 0.0):
        raise ValueError("tau must be strictly increasing")
    opacity = 10.0**np.clip(log_kappa, -30.0, 30.0)
    mass = np.empty_like(opacity)
    if surface_mass is None:
        mass[:, 0] = depth[0] / opacity[:, 0]
    else:
        surface = np.asarray(surface_mass, dtype=np.float64)
        if surface.shape != (opacity.shape[0],):
            raise ValueError("surface_mass must have one value per profile")
        mass[:, 0] = surface
    increments = 0.5 * np.diff(depth)[None, :] * (
        1.0 / opacity[:, 1:] + 1.0 / opacity[:, :-1]
    )
    mass[:, 1:] = mass[:, :1] + np.cumsum(increments, axis=1)
    return mass


def fit_profile_closure(
    corpus: Corpus,
    split: Split,
    *,
    target: np.ndarray,
    degree: int = 3,
    components: int = 3,
    regime_boundaries: tuple[float, float] = REGIME_BOUNDARIES,
    smoothing_width_K: float = 0.0,
) -> ProfileClosureParameters:
    """Fit a regime-wise low-rank profile map for an offline closure probe.

    Training rows are always assigned to exactly one regime; ``smoothing_width_K``
    only affects prediction, where it blends the three regime maps instead of
    picking one.  Keeping the fit hard means the stored constants are identical
    either way, so a smooth-versus-hard comparison isolates the seam and not a
    change in what was fitted.
    """

    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != corpus.labels.shape[0]:
        raise ValueError("target must have shape (N, depth) with one row per star")
    if degree < 0 or components < 1:
        raise ValueError("degree must be non-negative and components positive")
    if smoothing_width_K < 0.0 or not np.isfinite(smoothing_width_K):
        raise ValueError("smoothing_width_K must be finite and non-negative")

    features = label_features(corpus.labels)
    center = features[split.train].mean(axis=0)
    scale = np.maximum(features[split.train].std(axis=0), 1.0e-12)
    normalized = (features - center) / scale
    exponents = polynomial_exponents(features.shape[1], degree)
    train_design, _, _ = polynomial_features(
        normalized[split.train], exponents, center=np.zeros(5), scale=np.ones(5)
    )
    regimes = temperature_regimes(corpus.labels, boundaries=regime_boundaries)
    target_center_by_regime = np.zeros((3, values.shape[1]), dtype=np.float64)
    basis_by_regime = np.zeros((3, components, values.shape[1]), dtype=np.float64)
    coefficients_by_regime = np.zeros((3, exponents.shape[0], components), dtype=np.float64)

    for regime_index in range(3):
        mask = regimes[split.train] == regime_index
        if int(mask.sum()) < components:
            raise ValueError(f"not enough training rows in regime {regime_index}")
        training_values = values[split.train][mask]
        local_center = training_values.mean(axis=0)
        target_center_by_regime[regime_index] = local_center
        _, _, right_vectors = np.linalg.svd(
            training_values - local_center,
            full_matrices=False,
        )
        basis = right_vectors[:components]
        coefficients = np.linalg.lstsq(
            train_design[mask],
            (training_values - local_center) @ basis.T,
            rcond=None,
        )[0]
        basis_by_regime[regime_index] = basis
        coefficients_by_regime[regime_index] = coefficients

    return ProfileClosureParameters(
        degree=int(degree),
        components=int(components),
        exponents=exponents,
        feature_center=center,
        feature_scale=scale,
        target_center_by_regime=target_center_by_regime,
        basis_by_regime=basis_by_regime,
        coefficients_by_regime=coefficients_by_regime,
        regime_boundaries=(float(regime_boundaries[0]), float(regime_boundaries[1])),
        smoothing_width_K=float(smoothing_width_K),
    )


def evaluate_profile_closure(
    labels: np.ndarray,
    parameters: ProfileClosureParameters,
) -> np.ndarray:
    """Predict on whatever depth axis the closure was fitted on.

    The stored basis is tabulated per depth index, so the length of the output
    is fixed by the fit.  ``predict_profile_closure`` is the wrapper for the
    common case where that axis is the tau grid and the caller has one to
    check against; targets that live on intervals or on a single scalar per
    star call this directly.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    features = label_features(values)
    normalized = (
        features - parameters.feature_center
    ) / np.maximum(parameters.feature_scale, 1.0e-12)
    design, _, _ = polynomial_features(
        normalized,
        parameters.exponents,
        center=np.zeros(5),
        scale=np.ones(5),
    )
    depth = int(parameters.basis_by_regime.shape[2])
    prediction = np.zeros((values.shape[0], depth), dtype=np.float64)

    if parameters.smoothing_width_K > 0.0:
        weights = temperature_regime_weights(
            values,
            boundaries=parameters.regime_boundaries,
            width_K=parameters.smoothing_width_K,
        )
        for regime_index in range(3):
            basis = parameters.basis_by_regime[regime_index]
            coefficients = parameters.coefficients_by_regime[regime_index]
            regime_prediction = parameters.target_center_by_regime[regime_index] + (
                design @ coefficients
            ) @ basis
            prediction += weights[:, regime_index, None] * regime_prediction
        return prediction

    regimes = temperature_regimes(values, boundaries=parameters.regime_boundaries)
    for regime_index in range(3):
        mask = regimes == regime_index
        if not np.any(mask):
            continue
        basis = parameters.basis_by_regime[regime_index]
        coefficients = parameters.coefficients_by_regime[regime_index]
        prediction[mask] = parameters.target_center_by_regime[regime_index] + (
            design[mask] @ coefficients
        ) @ basis
    return prediction


def predict_profile_closure(
    labels: np.ndarray,
    tau: np.ndarray,
    parameters: ProfileClosureParameters,
) -> np.ndarray:
    """Predict a tau-grid profile using only stored analytic-fit constants."""

    if len(tau) != parameters.basis_by_regime.shape[2]:
        raise ValueError("tau length does not match the fitted profile basis")
    return evaluate_profile_closure(labels, parameters)
