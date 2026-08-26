"""The emulator-free warm start as a formula rather than a compressed grid.

``monotone_initializer`` made the H2 start physically well formed but left it
tabulated: its mean profiles and depth modes were eighty-vectors indexed by
layer, it raised on any other grid, and 2880 of its 4591 fitted floats were
that table.  Here the depth axis is a Chebyshev series in ``ln tau``
(``analytic_depth``), so the same object evaluates at any depth inside the
fitted interval and costs P+1 numbers per shape instead of eighty.

Everything else is carried over unchanged: the same four invariants, the same
label support box, and the same monotone anchor-and-increment representation
for temperature.  One thing had to change to make the guarantee survive the
move, and it is recorded in ``monotone_temperature``: the floor that keeps
temperature increasing used to be stated per interval, which is a statement
about a grid, and drifted 1.9 percent between an eighty-layer grid and a
791-layer one over the same interval.  It is now a floor on the gradient.

What the change bought, measured on the held-out split against H2 as the
reference point (4591 floats, temperature relative p95 0.0201, deep 0.0199,
column mass p95 0.0869):

    configuration   stored floats   temperature   deep     column mass
    COMPACT_        589             0.0389        0.0392   0.1698
    PARITY_         2407            0.0201        0.0199   0.0870
    PHYSICAL_       3851            0.0146        0.0152   0.0597

Grid independence is not a trade: at 2407 floats the formula reproduces H2 to
the digit while being 1.9 times smaller, because depth resolution and label
resolution can now be traded against each other instead of being locked to
eighty layers and five modes.  Meeting the 600-float budget costs roughly a
factor of two in both fields.

``PHYSICAL_CONFIGURATION`` is the same depth resolution with two Saha ionized
fractions added to the label coordinates (``physical_labels``).  It is still
smaller than the tabulated H2 asset it replaces and about thirty percent more
accurate in both fields, which is a much better return than anything available
by spending the same floats on depth.  Whether the solver cares about any of
these differences is not answerable offline and is the next measurement.

The depth interval is guarded like the label box.  A Chebyshev series diverges
outside its interval exactly as the label polynomial diverges outside its box,
so both refusals are the same refusal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analytic_depth import (
    AnalyticDepthClosure,
    DepthNormalization,
    evaluate_analytic_depth_closure,
    fit_analytic_depth_closure,
)
from .candidates import REGIME_BOUNDARIES
from .discovery import Corpus, Split, grey_temperature
from .monotone_temperature import (
    GRADIENT_FLOOR,
    LabelSupport,
    fit_label_support,
    project_to_monotone,
    require_label_support,
)
from .physical_labels import PHYSICAL_DEGREE_CAPS
from .profile_closure import integrate_mass_from_opacity

FORMAT_MARKER = "payne_zero_compact_profile_parameters_v1"

#: Reproduces H2 to the digit at 1.9 times fewer stored floats.
PARITY_CONFIGURATION = {
    "temperature": {"degree": 3, "components": 5, "center_degree": 22, "mode_degree": 18},
    "opacity": {"degree": 3, "components": 5, "center_degree": 18, "mode_degree": 18},
}
#: The same depth resolution with Saha ionization added to the label
#: coordinates.  Costs more stored floats than ``PARITY_CONFIGURATION`` and is
#: substantially more accurate in both fields; see ``physical_labels``.
PHYSICAL_CONFIGURATION = {
    "temperature": {
        "degree": 3,
        "components": 5,
        "center_degree": 22,
        "mode_degree": 18,
        "label_features": "physical",
        "degree_caps": PHYSICAL_DEGREE_CAPS,
    },
    "opacity": {
        "degree": 3,
        "components": 5,
        "center_degree": 18,
        "mode_degree": 18,
        "label_features": "physical",
        "degree_caps": PHYSICAL_DEGREE_CAPS,
    },
}
#: The most accurate pair that fits the 600-float budget, balanced so neither
#: field is much further from H2 than the other.
COMPACT_CONFIGURATION = {
    "temperature": {"degree": 2, "components": 3, "center_degree": 14, "mode_degree": 10},
    "opacity": {"degree": 2, "components": 2, "center_degree": 10, "mode_degree": 10},
}

# The two closures are fitted on the same split with the same label features,
# so their centering and scaling are the same numbers stored twice.  Two per
# feature, and the feature count depends on which map was used.
def _shared_label_scaling(parameters) -> int:
    return 2 * int(parameters.temperature.feature_center.size)


# The support box plus the gradient floor.
_GUARD_FLOATS = 11


@dataclass(frozen=True)
class CompactProfileParameters:
    """Stored constants for the grid-free emulator-free profile formula."""

    temperature: AnalyticDepthClosure
    opacity: AnalyticDepthClosure
    support: LabelSupport
    gradient_floor: float = GRADIENT_FLOOR

    @property
    def stored_float_count(self) -> int:
        """Distinct fitted floats, counting the shared label scaling once.

        Integer exponent tables are excluded: they are the structure of the
        polynomial, the same for every fit of a given degree, and would be
        emitted as a loop rather than as stored constants.
        """

        return int(
            self.temperature.stored_float_count
            + self.opacity.stored_float_count
            - _shared_label_scaling(self)
            + _GUARD_FLOATS
        )


def fit_compact_profile_parameters(
    corpus: Corpus,
    split: Split,
    *,
    configuration: dict | None = None,
    regime_boundaries: tuple[float, float] | None = REGIME_BOUNDARIES,
    smoothing_width_K: float = 0.0,
    gradient_floor: float = GRADIENT_FLOOR,
) -> CompactProfileParameters:
    """Fit both closures, each with its own depth and label resolution.

    The two fields are given separate configurations because they do not want
    the same one.  Temperature is well served by a high label degree and few
    depth modes; column mass collapses without modes -- a single-mode opacity
    closure reaches 0.53 dex where a two-mode one reaches 0.17 -- and cares
    much less about label degree.  Spending a joint budget as though the two
    were interchangeable wastes it.
    """

    chosen = PARITY_CONFIGURATION if configuration is None else configuration
    shared = {
        "regime_boundaries": regime_boundaries,
        "smoothing_width_K": smoothing_width_K,
    }
    grey = grey_temperature(corpus.labels[:, 0], corpus.tau)
    temperature = fit_analytic_depth_closure(
        corpus,
        split,
        target=np.log10(corpus.temperature / grey),
        **chosen["temperature"],
        **shared,
    )
    opacity = fit_analytic_depth_closure(
        corpus,
        split,
        target=np.log10(corpus.rosseland_opacity),
        **chosen["opacity"],
        **shared,
    )
    if not np.array_equal(temperature.feature_center, opacity.feature_center):
        raise ValueError("the two closures disagree on the label scaling")
    return CompactProfileParameters(
        temperature=temperature,
        opacity=opacity,
        # As in the tabulated version, the box describes the domain of the
        # model grid rather than the accident of the training draw, and only
        # the bounding box is read.
        support=fit_label_support(corpus.labels),
        gradient_floor=float(gradient_floor),
    )


def predict_compact_reduced_state(
    labels: np.ndarray,
    tau: np.ndarray,
    parameters: CompactProfileParameters,
    *,
    check_support: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(column_mass, temperature, log10_opacity)`` at any depths.

    Unlike the tabulated version this imposes no relationship between ``tau``
    and the grid the constants were fitted on, only that the depths lie inside
    the fitted interval.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    depth = np.asarray(tau, dtype=np.float64)
    if depth.ndim != 1 or depth.size < 2:
        raise ValueError("tau must be a one-dimensional grid of at least two depths")
    if np.any(np.diff(depth) <= 0.0):
        raise ValueError("tau must be strictly increasing")
    if check_support:
        require_label_support(values, parameters.support)
        parameters.temperature.normalization.require_support(depth)
        parameters.opacity.normalization.require_support(depth)

    log_opacity = evaluate_analytic_depth_closure(
        values, depth, parameters.opacity, check_support=False
    )
    residual = evaluate_analytic_depth_closure(
        values, depth, parameters.temperature, check_support=False
    )
    predicted = grey_temperature(values[:, 0], depth) * 10.0 ** np.clip(
        residual, -30.0, 30.0
    )
    temperature = project_to_monotone(
        depth, values[:, 0], predicted, floor=parameters.gradient_floor
    )
    column_mass = integrate_mass_from_opacity(depth, log_opacity)
    return column_mass, temperature, log_opacity


def _closure_arrays(prefix: str, closure: AnalyticDepthClosure) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_depth_center": np.asarray(closure.normalization.center, dtype=np.float64),
        f"{prefix}_depth_half_width": np.asarray(
            closure.normalization.half_width, dtype=np.float64
        ),
        f"{prefix}_center_degree": np.asarray(closure.center_degree, dtype=np.int64),
        f"{prefix}_mode_degree": np.asarray(closure.mode_degree, dtype=np.int64),
        f"{prefix}_exponents": np.asarray(closure.exponents, dtype=np.int64),
        f"{prefix}_feature_center": np.asarray(closure.feature_center, dtype=np.float64),
        f"{prefix}_feature_scale": np.asarray(closure.feature_scale, dtype=np.float64),
        f"{prefix}_center_by_regime": np.asarray(closure.center_by_regime, dtype=np.float64),
        f"{prefix}_modes_by_regime": np.asarray(closure.modes_by_regime, dtype=np.float64),
        f"{prefix}_coefficients_by_regime": np.asarray(
            closure.coefficients_by_regime, dtype=np.float64
        ),
        f"{prefix}_regime_boundaries": np.asarray(
            closure.regime_boundaries, dtype=np.float64
        ),
        f"{prefix}_smoothing_width_K": np.asarray(
            closure.smoothing_width_K, dtype=np.float64
        ),
        f"{prefix}_label_features": np.asarray(closure.label_features),
    }


def _load_closure(data, prefix: str) -> AnalyticDepthClosure:
    return AnalyticDepthClosure(
        normalization=DepthNormalization(
            center=float(np.asarray(data[f"{prefix}_depth_center"]).item()),
            half_width=float(np.asarray(data[f"{prefix}_depth_half_width"]).item()),
        ),
        center_degree=int(np.asarray(data[f"{prefix}_center_degree"]).item()),
        mode_degree=int(np.asarray(data[f"{prefix}_mode_degree"]).item()),
        exponents=np.asarray(data[f"{prefix}_exponents"], dtype=np.int64),
        feature_center=np.asarray(data[f"{prefix}_feature_center"], dtype=np.float64),
        feature_scale=np.asarray(data[f"{prefix}_feature_scale"], dtype=np.float64),
        center_by_regime=np.asarray(data[f"{prefix}_center_by_regime"], dtype=np.float64),
        modes_by_regime=np.asarray(data[f"{prefix}_modes_by_regime"], dtype=np.float64),
        coefficients_by_regime=np.asarray(
            data[f"{prefix}_coefficients_by_regime"], dtype=np.float64
        ),
        regime_boundaries=tuple(
            float(value)
            for value in np.asarray(data[f"{prefix}_regime_boundaries"], dtype=np.float64)
        ),
        smoothing_width_K=float(np.asarray(data[f"{prefix}_smoothing_width_K"]).item()),
        label_features=str(np.asarray(data[f"{prefix}_label_features"]).item()),
    )


def save_compact_profile_parameters(
    path: Path | str, parameters: CompactProfileParameters
) -> Path:
    """Save fitted constants as a portable compressed NumPy asset."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "format": np.asarray(FORMAT_MARKER),
        "gradient_floor": np.asarray(parameters.gradient_floor, dtype=np.float64),
        "support_lower": np.asarray(parameters.support.lower, dtype=np.float64),
        "support_upper": np.asarray(parameters.support.upper, dtype=np.float64),
    }
    payload.update(_closure_arrays("temperature", parameters.temperature))
    payload.update(_closure_arrays("opacity", parameters.opacity))
    np.savez_compressed(destination, **payload)
    return destination


def load_compact_profile_parameters(path: Path | str) -> CompactProfileParameters:
    """Load a portable parameter asset and validate its format marker."""

    with np.load(Path(path), allow_pickle=False) as data:
        marker = str(np.asarray(data["format"]).item())
        if marker != FORMAT_MARKER:
            raise ValueError(f"unsupported compact profile parameter format: {marker}")
        return CompactProfileParameters(
            temperature=_load_closure(data, "temperature"),
            opacity=_load_closure(data, "opacity"),
            support=LabelSupport(
                lower=np.asarray(data["support_lower"], dtype=np.float64),
                upper=np.asarray(data["support_upper"], dtype=np.float64),
            ),
            gradient_floor=float(np.asarray(data["gradient_floor"]).item()),
        )
