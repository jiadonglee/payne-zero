"""Physics-shaped analytic candidate families.

The first candidate is deliberately modest: it keeps the Eddington temperature
law and replaces the constant 0.34 opacity used by the grey benchmark with a
positive, label-dependent effective opacity.  It is a falsifiable H1 baseline,
not the final formula.  Its purpose is to measure whether the mass coordinate
alone can be repaired before adding local opacity and convection physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .discovery import (
    Corpus,
    Split,
    label_features,
    polynomial_exponents,
    polynomial_features,
)


REGIME_NAMES = ("cool", "warm", "hot")
REGIME_BOUNDARIES = (5500.0, 7500.0)


def temperature_regimes(
    labels: np.ndarray,
    *,
    boundaries: tuple[float, float] = REGIME_BOUNDARIES,
) -> np.ndarray:
    """Return the exploratory temperature regimes used by H1.

    H1 uses hard regions only as a diagnostic control.  The production
    candidate must replace these boundaries with smooth switches.

    ``boundaries`` is exposed so an ablation can move the seams away from the
    radiative/convective transition, which the default 7500 K seam sits inside.
    """

    values = np.asarray(labels, dtype=np.float64)
    first, second = float(boundaries[0]), float(boundaries[1])
    if not first < second:
        raise ValueError("temperature boundaries must be increasing")
    regime = np.empty(values.shape[0], dtype=np.int64)
    regime[values[:, 0] < first] = 0
    regime[(values[:, 0] >= first) & (values[:, 0] < second)] = 1
    regime[values[:, 0] >= second] = 2
    return regime


def temperature_regime_weights(
    labels: np.ndarray,
    *,
    boundaries: tuple[float, float] = REGIME_BOUNDARIES,
    width_K: float = 250.0,
) -> np.ndarray:
    """Return smooth cool/warm/hot weights that sum to one.

    The smooth counterpart to ``temperature_regimes``.  As ``width_K`` shrinks
    these weights approach the hard one-hot assignment, so a candidate can be
    scored either way against the same fitted constants.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("labels must have shape (N, 5)")
    if width_K <= 0.0 or not np.isfinite(width_K):
        raise ValueError("width_K must be finite and positive")
    first, second = (float(boundaries[0]), float(boundaries[1]))
    if not first < second:
        raise ValueError("temperature boundaries must be increasing")
    # Ordered logistic gates avoid negative middle weights when the two
    # transition tails overlap.  The exponent is clipped because a narrow width
    # otherwise overflows well before it changes the saturated answer.
    def _gate(argument: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(np.clip(argument, -700.0, 700.0)))

    hot = _gate(-(values[:, 0] - second) / width_K)
    cool = _gate((values[:, 0] - first) / width_K)
    warm = np.maximum(1.0 - cool - hot, 0.0)
    total = cool + warm + hot
    return np.column_stack((cool / total, warm / total, hot / total))


@dataclass(frozen=True)
class ScalarOpacityParameters:
    """Fitted constants for the H1 effective-opacity candidate."""

    degree: int
    exponents: np.ndarray
    feature_center: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    regime_names: tuple[str, ...] = REGIME_NAMES

    @property
    def coefficient_count(self) -> int:
        return int(np.count_nonzero(self.coefficients))


def _effective_opacity_target(corpus: Corpus) -> np.ndarray:
    """Estimate one positive opacity per atmosphere from tau/m.

    The first two layers are omitted because the production surface seed uses
    a separate closed-form convention.  H1 intentionally tests only the
    interior scale, leaving the top-boundary problem visible in later gates.
    """

    local = corpus.tau[None, 2:] / np.maximum(corpus.column_mass[:, 2:], 1.0e-300)
    return np.log10(np.median(local, axis=1))


def fit_scalar_opacity_parameters(
    corpus: Corpus,
    split: Split,
    *,
    degree: int = 2,
    ridge: float = 1.0e-8,
) -> tuple[ScalarOpacityParameters, dict[str, float]]:
    """Fit a piecewise polynomial effective-opacity law on the fit split."""

    if degree < 0:
        raise ValueError("degree must be non-negative")
    raw_features = label_features(corpus.labels)
    center = raw_features[split.train].mean(axis=0)
    scale = np.maximum(raw_features[split.train].std(axis=0), 1.0e-12)
    normalized = (raw_features - center) / scale
    exponents = polynomial_exponents(normalized.shape[1], degree)
    train_features, _, _ = polynomial_features(
        normalized[split.train], exponents, center=np.zeros(5), scale=np.ones(5)
    )
    validation_features, _, _ = polynomial_features(
        normalized[split.validation], exponents, center=np.zeros(5), scale=np.ones(5)
    )
    target = _effective_opacity_target(corpus)
    regime = temperature_regimes(corpus.labels)
    coefficients = np.zeros((len(REGIME_NAMES), exponents.shape[0]), dtype=np.float64)
    prediction = np.zeros(split.validation.size, dtype=np.float64)

    for regime_index in range(len(REGIME_NAMES)):
        train_mask = regime[split.train] == regime_index
        validation_mask = regime[split.validation] == regime_index
        if not np.any(train_mask) or not np.any(validation_mask):
            raise ValueError(f"regime {REGIME_NAMES[regime_index]} is empty")
        design = train_features[train_mask]
        gram = design.T @ design
        penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        coefficients[regime_index] = np.linalg.solve(
            gram + penalty,
            design.T @ target[split.train][train_mask],
        )
        prediction[validation_mask] = (
            validation_features[validation_mask] @ coefficients[regime_index]
        )

    truth = target[split.validation]
    residual = prediction - truth
    metrics = {
        "validation_rmse_dex": float(np.sqrt(np.mean(residual**2))),
        "validation_p95_dex": float(np.percentile(np.abs(residual), 95.0)),
        "validation_max_dex": float(np.max(np.abs(residual))),
        "coefficient_count": float(np.count_nonzero(coefficients)),
        "term_count": float(coefficients.size),
    }
    return (
        ScalarOpacityParameters(
            degree=int(degree),
            exponents=exponents,
            feature_center=center,
            feature_scale=scale,
            coefficients=coefficients,
        ),
        metrics,
    )


def predict_effective_opacity(
    labels: np.ndarray,
    parameters: ScalarOpacityParameters,
) -> np.ndarray:
    """Evaluate H1's positive effective opacity in cm^2 g^-1."""

    raw_features = label_features(np.asarray(labels, dtype=np.float64))
    normalized = (
        raw_features - parameters.feature_center
    ) / np.maximum(parameters.feature_scale, 1.0e-12)
    features, _, _ = polynomial_features(
        normalized,
        parameters.exponents,
        center=np.zeros(5),
        scale=np.ones(5),
    )
    regimes = temperature_regimes(np.asarray(labels, dtype=np.float64))
    log_opacity = np.zeros(raw_features.shape[0], dtype=np.float64)
    for regime_index in range(len(REGIME_NAMES)):
        mask = regimes == regime_index
        log_opacity[mask] = features[mask] @ parameters.coefficients[regime_index]
    return 10.0**log_opacity


def build_h1_reduced_state(
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    parameters: ScalarOpacityParameters | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build H1's analytic ``(m,T)`` state.

    Returns ``(column_mass, temperature, effective_opacity)`` with one row per
    input label.  The state is intentionally not converted to a full solver
    atmosphere here; the solver bridge is a later gate.
    """

    values = np.asarray(labels, dtype=np.float64)
    depth = np.asarray(tau, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("labels must have shape (N, 5)")
    if depth.ndim != 1 or depth.size < 2 or np.any(~np.isfinite(depth)):
        raise ValueError("tau must be a finite one-dimensional grid")
    if np.any(np.diff(depth) <= 0.0) or np.any(depth <= 0.0):
        raise ValueError("tau must be strictly increasing and positive")

    effective_opacity = (
        predict_effective_opacity(values, parameters)
        if parameters is not None
        else np.full(values.shape[0], 0.34, dtype=np.float64)
    )
    temperature = values[:, 0, None] * (
        0.75 * (depth[None, :] + 2.0 / 3.0)
    ) ** 0.25
    column_mass = depth[None, :] / effective_opacity[:, None]
    if np.any(~np.isfinite(column_mass)) or np.any(np.diff(column_mass, axis=1) <= 0.0):
        raise ValueError("H1 produced an invalid column-mass profile")
    return column_mass, temperature, effective_opacity
