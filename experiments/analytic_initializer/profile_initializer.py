"""A complete no-emulator profile formula assembled from H2 closures.

This is the first candidate that produces both fields the solver actually
needs from labels alone: a positive opacity profile is integrated to obtain
monotone column mass, while a Hopf-normalized temperature profile is predicted
with the same low-rank label coordinates.  It is intentionally kept under
``experiments`` until a real-solver smoke test is passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .candidates import REGIME_BOUNDARIES
from .discovery import Corpus, Split, grey_temperature
from .profile_closure import (
    ProfileClosureParameters,
    fit_profile_closure,
    integrate_mass_from_opacity,
    predict_profile_closure,
)


@dataclass(frozen=True)
class AnalyticProfileParameters:
    """Stored constants for the no-emulator H2 profile formula."""

    temperature: ProfileClosureParameters
    opacity: ProfileClosureParameters

    @property
    def coefficient_count(self) -> int:
        return int(
            self.temperature.coefficients_by_regime.size
            + self.opacity.coefficients_by_regime.size
        )

    @property
    def basis_value_count(self) -> int:
        return int(
            self.temperature.basis_by_regime.size
            + self.opacity.basis_by_regime.size
            + self.temperature.target_center_by_regime.size
            + self.opacity.target_center_by_regime.size
        )


def _closure_arrays(prefix: str, closure: ProfileClosureParameters) -> dict[str, np.ndarray]:
    """Serialize one profile closure without Python object pickling."""

    return {
        f"{prefix}_degree": np.asarray(closure.degree, dtype=np.int64),
        f"{prefix}_components": np.asarray(closure.components, dtype=np.int64),
        f"{prefix}_exponents": np.asarray(closure.exponents, dtype=np.int64),
        f"{prefix}_feature_center": np.asarray(closure.feature_center, dtype=np.float64),
        f"{prefix}_feature_scale": np.asarray(closure.feature_scale, dtype=np.float64),
        f"{prefix}_target_center_by_regime": np.asarray(
            closure.target_center_by_regime, dtype=np.float64
        ),
        f"{prefix}_basis_by_regime": np.asarray(
            closure.basis_by_regime, dtype=np.float64
        ),
        f"{prefix}_coefficients_by_regime": np.asarray(
            closure.coefficients_by_regime, dtype=np.float64
        ),
        f"{prefix}_regime_boundaries": np.asarray(
            closure.regime_boundaries, dtype=np.float64
        ),
        f"{prefix}_smoothing_width_K": np.asarray(
            closure.smoothing_width_K, dtype=np.float64
        ),
    }


def save_analytic_profile_parameters(
    path: Path | str, parameters: AnalyticProfileParameters
) -> Path:
    """Save fitted H2 constants as a portable compressed NumPy asset."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": np.asarray("payne_zero_analytic_profile_parameters_v1"),
        "temperature": np.asarray("temperature"),
        "opacity": np.asarray("opacity"),
    }
    payload.update(_closure_arrays("temperature", parameters.temperature))
    payload.update(_closure_arrays("opacity", parameters.opacity))
    np.savez_compressed(destination, **payload)
    return destination


def _load_closure_arrays(data, prefix: str) -> ProfileClosureParameters:
    """Reconstruct one profile closure from ``save_analytic_profile_parameters``."""

    return ProfileClosureParameters(
        degree=int(np.asarray(data[f"{prefix}_degree"]).item()),
        components=int(np.asarray(data[f"{prefix}_components"]).item()),
        exponents=np.asarray(data[f"{prefix}_exponents"], dtype=np.int64),
        feature_center=np.asarray(data[f"{prefix}_feature_center"], dtype=np.float64),
        feature_scale=np.asarray(data[f"{prefix}_feature_scale"], dtype=np.float64),
        target_center_by_regime=np.asarray(
            data[f"{prefix}_target_center_by_regime"], dtype=np.float64
        ),
        basis_by_regime=np.asarray(
            data[f"{prefix}_basis_by_regime"], dtype=np.float64
        ),
        coefficients_by_regime=np.asarray(
            data[f"{prefix}_coefficients_by_regime"], dtype=np.float64
        ),
        regime_boundaries=tuple(
            float(value)
            for value in np.asarray(data[f"{prefix}_regime_boundaries"], dtype=np.float64)
        ),
        smoothing_width_K=float(
            np.asarray(data[f"{prefix}_smoothing_width_K"]).item()
        ),
    )


def load_analytic_profile_parameters(
    path: Path | str,
) -> AnalyticProfileParameters:
    """Load a portable H2 parameter asset and validate its format marker."""

    with np.load(Path(path), allow_pickle=False) as data:
        marker = str(np.asarray(data["format"]).item())
        if marker != "payne_zero_analytic_profile_parameters_v1":
            raise ValueError(f"unsupported analytic profile parameter format: {marker}")
        return AnalyticProfileParameters(
            temperature=_load_closure_arrays(data, "temperature"),
            opacity=_load_closure_arrays(data, "opacity"),
        )


def fit_analytic_profile_parameters(
    corpus: Corpus,
    split: Split,
    *,
    degree: int = 3,
    components: int = 5,
    regime_boundaries: tuple[float, float] = REGIME_BOUNDARIES,
    smoothing_width_K: float = 0.0,
) -> AnalyticProfileParameters:
    """Fit the Hopf residual and opacity profile without a neural network."""

    grey = grey_temperature(corpus.labels[:, 0], corpus.tau)
    temperature_target = np.log10(corpus.temperature / grey)
    opacity_target = np.log10(corpus.rosseland_opacity)
    options = {
        "degree": degree,
        "components": components,
        "regime_boundaries": regime_boundaries,
        "smoothing_width_K": smoothing_width_K,
    }
    return AnalyticProfileParameters(
        temperature=fit_profile_closure(corpus, split, target=temperature_target, **options),
        opacity=fit_profile_closure(corpus, split, target=opacity_target, **options),
    )


def predict_analytic_reduced_state(
    labels: np.ndarray,
    tau: np.ndarray,
    parameters: AnalyticProfileParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(column_mass, temperature, log10_opacity)`` from labels."""

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    depth = np.asarray(tau, dtype=np.float64)
    temperature_residual = predict_profile_closure(
        values, depth, parameters.temperature
    )
    log_opacity = predict_profile_closure(values, depth, parameters.opacity)
    temperature = grey_temperature(values[:, 0], depth) * 10.0**temperature_residual
    column_mass = integrate_mass_from_opacity(depth, log_opacity)
    return column_mass, temperature, log_opacity


def score_analytic_reduced_state(
    truth_mass: np.ndarray,
    truth_temperature: np.ndarray,
    predicted_mass: np.ndarray,
    predicted_temperature: np.ndarray,
) -> dict[str, float]:
    """Summarize the two fields on a held-out set."""

    mass_error = np.abs(np.log10(predicted_mass) - np.log10(truth_mass))
    temperature_error = np.abs(
        predicted_temperature / truth_temperature - 1.0
    )
    return {
        "mass_dex_p50": float(np.percentile(mass_error, 50.0)),
        "mass_dex_p95": float(np.percentile(mass_error, 95.0)),
        "mass_dex_max": float(np.max(mass_error)),
        "temperature_relative_p50": float(np.percentile(temperature_error, 50.0)),
        "temperature_relative_p95": float(np.percentile(temperature_error, 95.0)),
        "temperature_relative_max": float(np.max(temperature_error)),
    }
