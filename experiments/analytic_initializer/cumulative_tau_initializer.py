"""Closed-form cumulative-``tau`` initializer with no neural network.

In ``x = ln(tau)``, four non-negative logistic windows partition depth.  The
logarithmic gradients of temperature and column mass are positive weighted
sums of those windows.  Their antiderivatives are softplus differences, so the
two profiles are continuous closed-form functions and monotone by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import nnls

from .discovery import (
    LABEL_FIELDS,
    grey_temperature,
    polynomial_exponents,
    polynomial_features,
)
from .physical_labels import (
    PHYSICAL_DEGREE_CAPS,
    capped_polynomial_exponents,
    feature_map,
)


FORMAT_MARKER = "payne_zero_cumulative_tau_parameters_v1"
DEFAULT_BOUNDARIES_TAU = (1.0e-2, 1.0, 100.0)
DEFAULT_ANCHOR_TAU = 0.013335
DEFAULT_SLOPE_FLOOR = 1.0e-6
DEFAULT_SLOPE_CEILING = 10.0


def _validate_width_and_boundaries(
    width: float, boundaries_tau: Sequence[float]
) -> tuple[float, np.ndarray]:
    width = float(width)
    boundaries = np.asarray(boundaries_tau, dtype=np.float64)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("width must be finite and positive")
    if (
        boundaries.shape != (3,)
        or np.any(~np.isfinite(boundaries))
        or np.any(boundaries <= 0.0)
        or np.any(np.diff(boundaries) <= 0.0)
    ):
        raise ValueError("boundaries_tau must contain three increasing positive values")
    return width, np.log(boundaries)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """Stable logistic function using NumPy's stable log-add-exp."""

    return np.exp(-np.logaddexp(0.0, -np.asarray(value, dtype=np.float64)))


def logistic_partition_windows(
    x: np.ndarray,
    *,
    width: float,
    boundaries_tau: Sequence[float] = DEFAULT_BOUNDARIES_TAU,
) -> np.ndarray:
    """Return the four non-negative logistic partition windows at ``x=ln tau``."""

    width, boundaries = _validate_width_and_boundaries(width, boundaries_tau)
    coordinate = np.asarray(x, dtype=np.float64)
    if np.any(~np.isfinite(coordinate)):
        raise ValueError("x must be finite")
    sigmoid = _sigmoid((coordinate[..., None] - boundaries) / width)
    windows = np.stack(
        (
            1.0 - sigmoid[..., 0],
            sigmoid[..., 0] - sigmoid[..., 1],
            sigmoid[..., 1] - sigmoid[..., 2],
            sigmoid[..., 2],
        ),
        axis=-1,
    )
    # Ordered boundaries make the differences non-negative analytically.
    # Clipping only removes possible one-ulp negative roundoff in the tails.
    return np.maximum(windows, 0.0)


def integrated_partition_windows(
    x: np.ndarray,
    anchor_x: float,
    *,
    width: float,
    boundaries_tau: Sequence[float] = DEFAULT_BOUNDARIES_TAU,
) -> np.ndarray:
    """Integrate each window exactly from ``anchor_x`` to ``x``.

    The primitive of ``sigmoid((x-b)/w)`` is
    ``w * softplus((x-b)/w)``.  Subtracting primitives at the anchor gives the
    requested closed-form integral, including negative values above the anchor.
    """

    width, boundaries = _validate_width_and_boundaries(width, boundaries_tau)
    coordinate = np.asarray(x, dtype=np.float64)
    anchor = float(anchor_x)
    if np.any(~np.isfinite(coordinate)) or not np.isfinite(anchor):
        raise ValueError("x and anchor_x must be finite")

    def primitive(values: np.ndarray) -> np.ndarray:
        softplus = width * np.logaddexp(
            0.0, (values[..., None] - boundaries) / width
        )
        return np.stack(
            (
                values - softplus[..., 0],
                softplus[..., 0] - softplus[..., 1],
                softplus[..., 1] - softplus[..., 2],
                softplus[..., 2],
            ),
            axis=-1,
        )

    return primitive(coordinate) - primitive(np.asarray(anchor))


def anchor_layer(tau: np.ndarray, anchor_tau: float = DEFAULT_ANCHOR_TAU) -> int:
    """Return the layer nearest the frozen optical-depth anchor."""

    depth = np.asarray(tau, dtype=np.float64)
    if (
        depth.ndim != 1
        or depth.size < 2
        or np.any(~np.isfinite(depth))
        or np.any(depth <= 0.0)
        or np.any(np.diff(depth) <= 0.0)
    ):
        raise ValueError("tau must be a finite, positive, strictly increasing grid")
    if not np.isfinite(anchor_tau) or anchor_tau <= 0.0:
        raise ValueError("anchor_tau must be finite and positive")
    return int(np.argmin(np.abs(depth - float(anchor_tau))))


def fit_oracle_targets(
    tau: np.ndarray,
    temperature: np.ndarray,
    column_mass: np.ndarray,
    effective_temperature: np.ndarray,
    *,
    width: float,
    boundaries_tau: Sequence[float] = DEFAULT_BOUNDARIES_TAU,
    anchor_tau: float = DEFAULT_ANCHOR_TAU,
    slope_floor: float = DEFAULT_SLOPE_FLOOR,
) -> np.ndarray:
    """Fit each star's two anchors and eight positive slopes with NNLS.

    Returned columns are ``T_anchor_correction``, ``ln(m_anchor)``, four
    ``ln(T_slope)`` values and four ``ln(m_slope)`` values.
    """

    depth = np.asarray(tau, dtype=np.float64)
    thermal = np.asarray(temperature, dtype=np.float64)
    mass = np.asarray(column_mass, dtype=np.float64)
    teff = np.asarray(effective_temperature, dtype=np.float64).reshape(-1)
    index = anchor_layer(depth, anchor_tau)
    if (
        thermal.ndim != 2
        or mass.shape != thermal.shape
        or thermal.shape[1] != depth.size
        or thermal.shape[0] != teff.size
    ):
        raise ValueError("profiles must have shape (N, len(tau)) and match Teff")
    if (
        np.any(~np.isfinite(thermal))
        or np.any(~np.isfinite(mass))
        or np.any(thermal <= 0.0)
        or np.any(mass <= 0.0)
    ):
        raise ValueError("temperature and column mass must be finite and positive")
    if not np.isfinite(slope_floor) or slope_floor <= 0.0:
        raise ValueError("slope_floor must be finite and positive")

    x = np.log(depth)
    design = integrated_partition_windows(
        x,
        x[index],
        width=width,
        boundaries_tau=boundaries_tau,
    )
    grey_anchor = grey_temperature(teff, depth)[:, index]
    targets = np.empty((thermal.shape[0], 10), dtype=np.float64)
    targets[:, 0] = np.log(thermal[:, index] / grey_anchor)
    targets[:, 1] = np.log(mass[:, index])
    log_thermal = np.log(thermal)
    log_mass = np.log(mass)
    for row in range(thermal.shape[0]):
        thermal_slopes, _ = nnls(
            design, log_thermal[row] - log_thermal[row, index]
        )
        mass_slopes, _ = nnls(design, log_mass[row] - log_mass[row, index])
        targets[row, 2:6] = np.log(np.maximum(thermal_slopes, slope_floor))
        targets[row, 6:10] = np.log(np.maximum(mass_slopes, slope_floor))
    return targets


@dataclass(frozen=True)
class CumulativeTauParameters:
    """Constants of one degree/width cumulative-``tau`` label map."""

    degree: int
    width: float
    boundaries_tau: np.ndarray
    anchor_tau: float
    tau_lower: float
    tau_upper: float
    label_features: str
    exponents: np.ndarray
    feature_center: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    support_lower: np.ndarray
    support_upper: np.ndarray
    slope_floor: float = DEFAULT_SLOPE_FLOOR
    slope_ceiling: float = DEFAULT_SLOPE_CEILING

    def __post_init__(self) -> None:
        if int(self.degree) not in (1, 2, 3):
            raise ValueError("degree must be 1, 2, or 3")
        width, _ = _validate_width_and_boundaries(self.width, self.boundaries_tau)
        feature_width = 5 if self.label_features == "standard" else 7
        if self.label_features not in {"standard", "physical"}:
            raise ValueError("label_features must be 'standard' or 'physical'")
        exponents = np.asarray(self.exponents, dtype=np.int64)
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        lower = np.asarray(self.support_lower, dtype=np.float64)
        upper = np.asarray(self.support_upper, dtype=np.float64)
        if exponents.ndim != 2 or exponents.shape[1] != feature_width:
            raise ValueError("exponents do not match the selected label feature map")
        if (
            center.shape != (feature_width,)
            or scale.shape != (feature_width,)
            or np.any(scale <= 0.0)
        ):
            raise ValueError("feature scaling does not match the selected feature map")
        if coefficients.shape != (exponents.shape[0], 10):
            raise ValueError("coefficients must have shape (terms, 10)")
        if lower.shape != (5,) or upper.shape != (5,) or np.any(upper < lower):
            raise ValueError("label support must contain five ordered bounds")
        if (
            not 0.0 < float(self.tau_lower) < float(self.tau_upper)
            or not float(self.tau_lower) <= float(self.anchor_tau) <= float(self.tau_upper)
        ):
            raise ValueError("tau support must be positive and contain the anchor")
        if (
            not 0.0 < float(self.slope_floor) <= float(self.slope_ceiling)
            or not np.isfinite(self.slope_ceiling)
        ):
            raise ValueError("slope bounds must be finite, positive, and ordered")
        object.__setattr__(self, "width", width)
        object.__setattr__(
            self, "boundaries_tau", np.asarray(self.boundaries_tau, dtype=np.float64)
        )
        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "support_lower", lower)
        object.__setattr__(self, "support_upper", upper)

    @property
    def term_count(self) -> int:
        return int(self.exponents.shape[0])

    @property
    def fitted_parameter_count(self) -> int:
        return int(self.coefficients.size)

    @property
    def stored_float_count(self) -> int:
        return int(
            self.coefficients.size
            + self.feature_center.size
            + self.feature_scale.size
            + self.support_lower.size
            + self.support_upper.size
            + self.boundaries_tau.size
            + 6
        )


@dataclass(frozen=True)
class CumulativeTauPrediction:
    column_mass: np.ndarray
    temperature: np.ndarray
    opacity: np.ndarray
    mass_log_slope: np.ndarray


def fit_cumulative_tau_parameters(
    labels: np.ndarray,
    tau: np.ndarray,
    oracle_targets: np.ndarray,
    train_indices: Sequence[int],
    *,
    degree: int,
    width: float,
    label_features_name: str = "standard",
    support_indices: Sequence[int] | None = None,
    boundaries_tau: Sequence[float] = DEFAULT_BOUNDARIES_TAU,
    anchor_tau: float = DEFAULT_ANCHOR_TAU,
    slope_floor: float = DEFAULT_SLOPE_FLOOR,
    slope_ceiling: float = DEFAULT_SLOPE_CEILING,
    ridge: float = 1.0e-8,
) -> CumulativeTauParameters:
    """Fit the continuous label-to-anchor/log-slope polynomial."""

    values = np.asarray(labels, dtype=np.float64)
    targets = np.asarray(oracle_targets, dtype=np.float64)
    train = np.asarray(train_indices, dtype=np.int64)
    depth = np.asarray(tau, dtype=np.float64)
    index = anchor_layer(depth, anchor_tau)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("labels must have shape (N, 5)")
    if targets.shape != (values.shape[0], 10) or np.any(~np.isfinite(targets)):
        raise ValueError("oracle_targets must have finite shape (N, 10)")
    if train.size == 0 or np.any(train < 0) or np.any(train >= values.shape[0]):
        raise ValueError("train_indices must select at least one valid row")
    if int(degree) not in (1, 2, 3):
        raise ValueError("degree must be 1, 2, or 3")
    features = feature_map(label_features_name)(values)
    exponents = (
        polynomial_exponents(features.shape[1], int(degree))
        if label_features_name == "standard"
        else capped_polynomial_exponents(
            int(degree), tuple(min(cap, int(degree)) for cap in PHYSICAL_DEGREE_CAPS)
        )
    )
    design, center, scale = polynomial_features(features[train], exponents)
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty, design.T @ targets[train]
    )
    support_rows = (
        np.arange(values.shape[0], dtype=np.int64)
        if support_indices is None
        else np.asarray(support_indices, dtype=np.int64)
    )
    if (
        support_rows.size == 0
        or np.any(support_rows < 0)
        or np.any(support_rows >= values.shape[0])
    ):
        raise ValueError("support_indices must select at least one valid row")
    return CumulativeTauParameters(
        degree=int(degree),
        width=float(width),
        boundaries_tau=np.asarray(boundaries_tau, dtype=np.float64),
        anchor_tau=float(depth[index]),
        tau_lower=float(depth[0]),
        tau_upper=float(depth[-1]),
        label_features=str(label_features_name),
        exponents=exponents,
        feature_center=center,
        feature_scale=scale,
        coefficients=coefficients,
        support_lower=values[support_rows].min(axis=0),
        support_upper=values[support_rows].max(axis=0),
        slope_floor=float(slope_floor),
        slope_ceiling=float(slope_ceiling),
    )


def _require_support(
    labels: np.ndarray, tau: np.ndarray, parameters: CumulativeTauParameters
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 5 or np.any(~np.isfinite(values)):
        raise ValueError("labels must be finite with shape (N, 5)")
    if np.any(values < parameters.support_lower) or np.any(
        values > parameters.support_upper
    ):
        raise ValueError("labels are outside the fitted cumulative-tau support")
    depth = np.asarray(tau, dtype=np.float64)
    if (
        depth.ndim != 1
        or depth.size < 2
        or np.any(~np.isfinite(depth))
        or np.any(depth <= 0.0)
        or np.any(np.diff(depth) <= 0.0)
    ):
        raise ValueError("tau must be finite, positive, and strictly increasing")
    tolerance = 32.0 * np.finfo(np.float64).eps
    if (
        depth[0] < parameters.tau_lower * (1.0 - tolerance)
        or depth[-1] > parameters.tau_upper * (1.0 + tolerance)
    ):
        raise ValueError("tau is outside the fitted cumulative-tau support")
    return values, depth


def predict_cumulative_tau_state(
    labels: np.ndarray,
    tau: np.ndarray,
    parameters: CumulativeTauParameters,
    *,
    check_support: bool = True,
) -> CumulativeTauPrediction:
    """Evaluate ``(m,T,kappa,s_m)`` on any supported increasing tau grid."""

    if check_support:
        values, depth = _require_support(labels, tau, parameters)
    else:
        values = np.asarray(labels, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        depth = np.asarray(tau, dtype=np.float64)
    design, _, _ = polynomial_features(
        feature_map(parameters.label_features)(values),
        parameters.exponents,
        center=parameters.feature_center,
        scale=parameters.feature_scale,
    )
    mapped = design @ parameters.coefficients
    lower = np.log(parameters.slope_floor)
    upper = np.log(parameters.slope_ceiling)
    thermal_slopes = np.exp(np.clip(mapped[:, 2:6], lower, upper))
    mass_slopes = np.exp(np.clip(mapped[:, 6:10], lower, upper))
    x = np.log(depth)
    anchor_x = np.log(parameters.anchor_tau)
    integrated = integrated_partition_windows(
        x,
        anchor_x,
        width=parameters.width,
        boundaries_tau=parameters.boundaries_tau,
    )
    windows = logistic_partition_windows(
        x, width=parameters.width, boundaries_tau=parameters.boundaries_tau
    )
    grey_anchor = grey_temperature(values[:, 0], np.asarray([parameters.anchor_tau]))[
        :, 0
    ]
    log_temperature_anchor = np.log(grey_anchor) + np.clip(mapped[:, 0], -20.0, 20.0)
    log_mass_anchor = np.clip(mapped[:, 1], -100.0, 100.0)
    log_temperature = (
        log_temperature_anchor[:, None] + thermal_slopes @ integrated.T
    )
    log_mass = log_mass_anchor[:, None] + mass_slopes @ integrated.T
    temperature = np.exp(log_temperature)
    column_mass = np.exp(log_mass)
    mass_log_slope = mass_slopes @ windows.T
    opacity = depth[None, :] / (column_mass * mass_log_slope)
    arrays = (temperature, column_mass, mass_log_slope, opacity)
    if any(np.any(~np.isfinite(array)) or np.any(array <= 0.0) for array in arrays):
        raise FloatingPointError("cumulative-tau prediction is not finite and positive")
    if np.any(np.diff(temperature, axis=1) <= 0.0) or np.any(
        np.diff(column_mass, axis=1) <= 0.0
    ):
        raise FloatingPointError("cumulative-tau prediction lost strict monotonicity")
    return CumulativeTauPrediction(
        column_mass=column_mass,
        temperature=temperature,
        opacity=opacity,
        mass_log_slope=mass_log_slope,
    )


def save_cumulative_tau_parameters(
    path: Path | str, parameters: CumulativeTauParameters
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        format=np.asarray(FORMAT_MARKER),
        degree=np.asarray(parameters.degree, dtype=np.int64),
        width=np.asarray(parameters.width),
        boundaries_tau=parameters.boundaries_tau,
        anchor_tau=np.asarray(parameters.anchor_tau),
        tau_lower=np.asarray(parameters.tau_lower),
        tau_upper=np.asarray(parameters.tau_upper),
        label_features=np.asarray(parameters.label_features),
        exponents=parameters.exponents,
        feature_center=parameters.feature_center,
        feature_scale=parameters.feature_scale,
        coefficients=parameters.coefficients,
        support_lower=parameters.support_lower,
        support_upper=parameters.support_upper,
        slope_floor=np.asarray(parameters.slope_floor),
        slope_ceiling=np.asarray(parameters.slope_ceiling),
    )
    return destination


def load_cumulative_tau_parameters(path: Path | str) -> CumulativeTauParameters:
    with np.load(Path(path), allow_pickle=False) as data:
        marker = str(np.asarray(data["format"]).item())
        if marker != FORMAT_MARKER:
            raise ValueError(f"unsupported cumulative-tau parameter format: {marker}")
        return CumulativeTauParameters(
            degree=int(np.asarray(data["degree"]).item()),
            width=float(np.asarray(data["width"]).item()),
            boundaries_tau=np.asarray(data["boundaries_tau"], dtype=np.float64),
            anchor_tau=float(np.asarray(data["anchor_tau"]).item()),
            tau_lower=float(np.asarray(data["tau_lower"]).item()),
            tau_upper=float(np.asarray(data["tau_upper"]).item()),
            label_features=str(np.asarray(data["label_features"]).item()),
            exponents=np.asarray(data["exponents"], dtype=np.int64),
            feature_center=np.asarray(data["feature_center"], dtype=np.float64),
            feature_scale=np.asarray(data["feature_scale"], dtype=np.float64),
            coefficients=np.asarray(data["coefficients"], dtype=np.float64),
            support_lower=np.asarray(data["support_lower"], dtype=np.float64),
            support_upper=np.asarray(data["support_upper"], dtype=np.float64),
            slope_floor=float(np.asarray(data["slope_floor"]).item()),
            slope_ceiling=float(np.asarray(data["slope_ceiling"]).item()),
        )
