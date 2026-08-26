"""A depth basis that is a function of tau rather than a table indexed by layer.

What still made the H2 formula a compressed grid rather than a formula was its
depth axis.  The stored constants included a mean profile and five SVD modes
per regime, each an eighty-vector tied to the production grid -- 2880 of the
4591 fitted floats -- and asking for any other grid raised.  Nothing about the
label half had that problem: a polynomial in five labels evaluates anywhere.

Here the eighty-vectors are replaced by Chebyshev series in ``ln tau``.  The
production grid is exactly uniform in ``ln tau``, which makes it a good
interval for a polynomial basis, and the series evaluates at any depth, so the
formula stops caring which grid it is asked about.  A degree-P series costs
P+1 numbers where a tabulated mode cost eighty.

Two measurements set the achievable resolution.  Projecting the corpus targets
onto the basis and reading the error back out gives, over 52199 rows:

    P    log10(T/T_grey) p95     column mass p95 after integrating kappa
    8    7.7e-3 dex              0.063 dex
    12   3.8e-3 dex              0.024 dex
    16   2.5e-3 dex              0.011 dex

The opacity column is the one that matters and it is far gentler than the raw
``log10(kappa)`` error suggests, because column mass is an integral of
``1/kappa``: at P=16 only ten rows (0.02 percent) exceed 0.3 dex in kappa, and
they do it at the bottom of the grid where ``1/kappa`` contributes almost
nothing to the integral.  H2 itself sits at 0.087 dex in column mass, so the
depth basis stops being the limiting term somewhere below P=12.

The fit is unchanged in structure: a regime-wise mean, a low-rank set of depth
shapes, and a polynomial in the labels for the amplitudes.  Only the space the
mean and the shapes live in has changed, and they are projected into it
*before* the singular value decomposition, so the decomposition never spends a
mode on structure the basis cannot carry.  Because the projection is done in
layer space against the same uniform measure the old fit used, the low-rank
geometry is the one that was already validated rather than a new one.

The mean profile and the modes get separate degrees.  The mean carries most of
the shape and is worth resolving; the modes are corrections and are the term
the budget multiplies by the mode count, so buying resolution there is several
times more expensive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates import REGIME_BOUNDARIES, temperature_regime_weights, temperature_regimes
from .discovery import Corpus, Split, polynomial_exponents, polynomial_features
from .physical_labels import capped_polynomial_exponents, feature_map


@dataclass(frozen=True)
class DepthNormalization:
    """Maps ``ln tau`` onto the interval a Chebyshev series is stable on.

    Two stored numbers.  They also define where the formula may be evaluated:
    a Chebyshev series diverges outside its interval as readily as the label
    polynomial diverges outside its box, so the same refusal applies.
    """

    center: float
    half_width: float

    def __post_init__(self) -> None:
        if not self.half_width > 0.0 or not np.isfinite(self.half_width):
            raise ValueError("half_width must be finite and positive")
        if not np.isfinite(self.center):
            raise ValueError("center must be finite")

    @classmethod
    def from_grid(cls, tau: np.ndarray) -> "DepthNormalization":
        values = np.log(np.asarray(tau, dtype=np.float64))
        if values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError("tau must hold at least two finite positive depths")
        low, high = float(values.min()), float(values.max())
        return cls(center=0.5 * (low + high), half_width=0.5 * (high - low))

    def coordinate(self, tau: np.ndarray) -> np.ndarray:
        values = np.asarray(tau, dtype=np.float64)
        if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
            raise ValueError("tau must be finite and positive")
        return (np.log(values) - self.center) / self.half_width

    def design(self, tau: np.ndarray, degree: int) -> np.ndarray:
        """Return the Chebyshev design matrix of shape ``(len(tau), degree+1)``."""

        if degree < 0:
            raise ValueError("degree must be non-negative")
        return np.polynomial.chebyshev.chebvander(self.coordinate(tau), int(degree))

    def require_support(self, tau: np.ndarray, *, tolerance: float = 1.0e-9) -> None:
        """Raise if any depth falls outside the interval the series is fitted on."""

        coordinate = self.coordinate(tau)
        if np.any(np.abs(coordinate) > 1.0 + tolerance):
            low = float(np.exp(self.center - self.half_width))
            high = float(np.exp(self.center + self.half_width))
            values = np.asarray(tau, dtype=np.float64)
            raise ValueError(
                f"tau in [{values.min():.6g}, {values.max():.6g}] is outside the "
                f"fitted depth interval [{low:.6g}, {high:.6g}]"
            )


@dataclass(frozen=True)
class AnalyticDepthClosure:
    """A label-conditioned profile whose depth dependence is a Chebyshev series."""

    normalization: DepthNormalization
    center_degree: int
    mode_degree: int
    exponents: np.ndarray
    feature_center: np.ndarray
    feature_scale: np.ndarray
    center_by_regime: np.ndarray
    modes_by_regime: np.ndarray
    coefficients_by_regime: np.ndarray
    regime_boundaries: tuple[float, float] = REGIME_BOUNDARIES
    smoothing_width_K: float = 0.0
    #: Which label coordinates the amplitudes were fitted against.  Stored
    #: because prediction has to rebuild the same ones; see ``physical_labels``.
    label_features: str = "standard"

    @property
    def components(self) -> int:
        return int(self.modes_by_regime.shape[1])

    @property
    def regimes(self) -> int:
        return int(self.center_by_regime.shape[0])

    @property
    def stored_float_count(self) -> int:
        """Fitted floats only; the integer exponent table is structure."""

        return int(
            2
            + self.feature_center.size
            + self.feature_scale.size
            + self.center_by_regime.size
            + self.modes_by_regime.size
            + self.coefficients_by_regime.size
            + len(self.regime_boundaries)
            + 1
        )


def _project(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Orthogonally project rows of ``values`` onto the span of ``design``.

    The projection is taken in layer space so that the least-squares geometry
    the original fit used is preserved exactly; the fitted objects therefore
    lie in the Chebyshev span and converting them to coefficients afterwards
    is lossless.
    """

    coefficients, *_ = np.linalg.lstsq(design, np.asarray(values, dtype=np.float64).T, rcond=None)
    return (design @ coefficients).T


def _coefficients(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return the Chebyshev coefficients of rows already inside the span."""

    coefficients, *_ = np.linalg.lstsq(design, np.asarray(values, dtype=np.float64).T, rcond=None)
    return coefficients.T


def fit_analytic_depth_closure(
    corpus: Corpus,
    split: Split,
    *,
    target: np.ndarray,
    degree: int = 3,
    components: int = 5,
    center_degree: int = 16,
    mode_degree: int = 12,
    regime_boundaries: tuple[float, float] | None = REGIME_BOUNDARIES,
    smoothing_width_K: float = 0.0,
    label_features: str = "standard",
    degree_caps: tuple[int, ...] | None = None,
) -> AnalyticDepthClosure:
    """Fit a regime-wise low-rank profile whose depth axis is analytic.

    ``regime_boundaries`` may be ``None``, which fits one regime over the whole
    label range.  Segmentation triples every stored depth object, so whether it
    earns its cost is a budget question and not a physical one.
    """

    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != corpus.labels.shape[0]:
        raise ValueError("target must have shape (N, depth) with one row per star")
    if values.shape[1] != corpus.tau.size:
        raise ValueError("target must be sampled on the corpus tau grid")
    if degree < 0 or components < 1:
        raise ValueError("degree must be non-negative and components positive")
    if center_degree < 0 or mode_degree < 0:
        raise ValueError("depth degrees must be non-negative")
    if smoothing_width_K < 0.0 or not np.isfinite(smoothing_width_K):
        raise ValueError("smoothing_width_K must be finite and non-negative")

    normalization = DepthNormalization.from_grid(corpus.tau)
    center_design = normalization.design(corpus.tau, center_degree)
    mode_design = normalization.design(corpus.tau, mode_degree)

    features = feature_map(label_features)(corpus.labels)
    width = features.shape[1]
    center = features[split.train].mean(axis=0)
    scale = np.maximum(features[split.train].std(axis=0), 1.0e-12)
    normalized = (features - center) / scale
    # A per-feature cap keeps the already-nonlinear ionization coordinates from
    # multiplying everything else three deep, which costs terms and buys
    # nothing measurable.
    exponents = (
        polynomial_exponents(width, degree)
        if degree_caps is None
        else capped_polynomial_exponents(degree, degree_caps)
    )
    if exponents.shape[1] != width:
        raise ValueError("degree_caps must name exactly one cap per feature")
    train_design, _, _ = polynomial_features(
        normalized[split.train], exponents, center=np.zeros(width), scale=np.ones(width)
    )

    if regime_boundaries is None:
        regimes = np.zeros(corpus.labels.shape[0], dtype=np.int64)
        regime_count = 1
        boundaries = (float("-inf"), float("inf"))
    else:
        regimes = temperature_regimes(corpus.labels, boundaries=regime_boundaries)
        regime_count = 3
        boundaries = (float(regime_boundaries[0]), float(regime_boundaries[1]))

    center_by_regime = np.zeros((regime_count, center_degree + 1), dtype=np.float64)
    modes_by_regime = np.zeros((regime_count, components, mode_degree + 1), dtype=np.float64)
    coefficients_by_regime = np.zeros(
        (regime_count, exponents.shape[0], components), dtype=np.float64
    )

    for regime_index in range(regime_count):
        mask = regimes[split.train] == regime_index
        if int(mask.sum()) < components:
            raise ValueError(f"not enough training rows in regime {regime_index}")
        training_values = values[split.train][mask]
        # The mean is projected first so the residual the decomposition sees is
        # already free of anything the basis cannot carry.
        local_center = _project(center_design, training_values.mean(axis=0)[None, :])[0]
        residual = _project(mode_design, training_values - local_center)
        _, _, right_vectors = np.linalg.svd(residual, full_matrices=False)
        basis = right_vectors[:components]
        amplitudes = np.linalg.lstsq(train_design[mask], residual @ basis.T, rcond=None)[0]
        center_by_regime[regime_index] = _coefficients(center_design, local_center[None, :])[0]
        modes_by_regime[regime_index] = _coefficients(mode_design, basis)
        coefficients_by_regime[regime_index] = amplitudes

    return AnalyticDepthClosure(
        normalization=normalization,
        center_degree=int(center_degree),
        mode_degree=int(mode_degree),
        exponents=exponents,
        feature_center=center,
        feature_scale=scale,
        center_by_regime=center_by_regime,
        modes_by_regime=modes_by_regime,
        coefficients_by_regime=coefficients_by_regime,
        regime_boundaries=boundaries,
        smoothing_width_K=float(smoothing_width_K),
        label_features=str(label_features),
    )


def evaluate_analytic_depth_closure(
    labels: np.ndarray,
    tau: np.ndarray,
    closure: AnalyticDepthClosure,
    *,
    check_support: bool = True,
) -> np.ndarray:
    """Evaluate the closure at any depths inside the fitted interval."""

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    depth = np.asarray(tau, dtype=np.float64)
    if check_support:
        closure.normalization.require_support(depth)

    center_design = closure.normalization.design(depth, closure.center_degree)
    mode_design = closure.normalization.design(depth, closure.mode_degree)
    features = feature_map(closure.label_features)(values)
    width = features.shape[1]
    normalized = (features - closure.feature_center) / np.maximum(
        closure.feature_scale, 1.0e-12
    )
    design, _, _ = polynomial_features(
        normalized, closure.exponents, center=np.zeros(width), scale=np.ones(width)
    )

    prediction = np.zeros((values.shape[0], depth.size), dtype=np.float64)
    if closure.regimes == 1:
        centered = center_design @ closure.center_by_regime[0]
        shapes = mode_design @ closure.modes_by_regime[0].T
        return centered[None, :] + (design @ closure.coefficients_by_regime[0]) @ shapes.T

    if closure.smoothing_width_K > 0.0:
        weights = temperature_regime_weights(
            values,
            boundaries=closure.regime_boundaries,
            width_K=closure.smoothing_width_K,
        )
        for regime_index in range(closure.regimes):
            centered = center_design @ closure.center_by_regime[regime_index]
            shapes = mode_design @ closure.modes_by_regime[regime_index].T
            regime_prediction = centered[None, :] + (
                design @ closure.coefficients_by_regime[regime_index]
            ) @ shapes.T
            prediction += weights[:, regime_index, None] * regime_prediction
        return prediction

    regimes = temperature_regimes(values, boundaries=closure.regime_boundaries)
    for regime_index in range(closure.regimes):
        mask = regimes == regime_index
        if not np.any(mask):
            continue
        centered = center_design @ closure.center_by_regime[regime_index]
        shapes = mode_design @ closure.modes_by_regime[regime_index].T
        prediction[mask] = centered[None, :] + (
            design[mask] @ closure.coefficients_by_regime[regime_index]
        ) @ shapes.T
    return prediction
