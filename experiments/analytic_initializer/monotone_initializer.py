"""The emulator-free warm start with all four physical invariants enforced.

This is H2 with two guards added and nothing else changed.  The opacity half is
untouched and still carries column mass: a positive ``kappa`` integrated
through ``dm/dtau = 1/kappa`` gives a strictly increasing ``m``.  The
temperature half is fitted exactly as H2 fits it, then emitted through the
monotone anchor-and-increment representation, so a strictly increasing ``T`` is
a property of the construction rather than of the fit.  A support box is stored
with the constants and checked on every call, because a degree-three polynomial
extrapolates without complaint and without bound.

The four invariants the assembled formula guarantees for any labels inside the
support box:

* ``kappa > 0``             -- the closure predicts ``log10(kappa)``
* ``m`` strictly increasing -- positive integrand, cumulative sum
* ``T > 0``                 -- the profile is an exponential
* ``T`` strictly increasing -- positive increments, cumulative sum

Because the fitted constants are the same quantities H2 already fitted, an
existing H2 asset can be adopted through ``from_analytic`` instead of refitted.
That matters: the recorded 60-star funnel ran on ``h2_profile_parameters_v1``,
and adopting those constants keeps the convergence result attached to the same
numbers rather than to a new fit nobody has run the solver on.

``AnalyticProfileParameters`` in ``profile_initializer`` is left alone for the
same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .candidates import REGIME_BOUNDARIES
from .discovery import Corpus, Split, grey_temperature
from .monotone_temperature import (
    GRADIENT_FLOOR,
    LabelSupport,
    fit_label_support,
    project_to_monotone,
    require_label_support,
)
from .profile_closure import (
    ProfileClosureParameters,
    evaluate_profile_closure,
    fit_profile_closure,
    integrate_mass_from_opacity,
)
from .profile_initializer import (
    AnalyticProfileParameters,
    _closure_arrays,
    _load_closure_arrays,
)

FORMAT_MARKER = "payne_zero_monotone_profile_parameters_v2"


@dataclass(frozen=True)
class MonotoneProfileParameters:
    """Stored constants for the monotone emulator-free profile formula."""

    temperature: ProfileClosureParameters
    opacity: ProfileClosureParameters
    support: LabelSupport
    gradient_floor: float = GRADIENT_FLOOR

    @classmethod
    def from_analytic(
        cls,
        parameters: AnalyticProfileParameters,
        support: LabelSupport,
        *,
        gradient_floor: float = GRADIENT_FLOOR,
    ) -> "MonotoneProfileParameters":
        """Adopt an existing H2 asset, adding only the two guards."""

        return cls(
            temperature=parameters.temperature,
            opacity=parameters.opacity,
            support=support,
            gradient_floor=float(gradient_floor),
        )

    @property
    def stored_float_count(self) -> int:
        """Count every fitted float, which is the budget the plan tracks.

        Integer exponent tables are excluded: they are the structure of the
        polynomial, identical for every fit of the same degree, and would be
        written out as a loop rather than as stored constants.
        """

        total = 2 * int(self.support.lower.size) + 1
        for closure in (self.temperature, self.opacity):
            total += int(
                closure.feature_center.size
                + closure.feature_scale.size
                + closure.target_center_by_regime.size
                + closure.basis_by_regime.size
                + closure.coefficients_by_regime.size
            )
        return total


def fit_monotone_profile_parameters(
    corpus: Corpus,
    split: Split,
    *,
    degree: int = 3,
    components: int = 5,
    regime_boundaries: tuple[float, float] = REGIME_BOUNDARIES,
    smoothing_width_K: float = 0.0,
    gradient_floor: float = GRADIENT_FLOOR,
) -> MonotoneProfileParameters:
    """Fit the temperature and opacity closures and take the support box."""

    options = {
        "degree": degree,
        "components": components,
        "regime_boundaries": regime_boundaries,
        "smoothing_width_K": smoothing_width_K,
    }
    grey = grey_temperature(corpus.labels[:, 0], corpus.tau)
    return MonotoneProfileParameters(
        temperature=fit_profile_closure(
            corpus, split, target=np.log10(corpus.temperature / grey), **options
        ),
        opacity=fit_profile_closure(
            corpus, split, target=np.log10(corpus.rosseland_opacity), **options
        ),
        # The box comes from the whole corpus, not from the training split.
        # It is a statement about the domain the model grid was computed over,
        # and the corpus is that grid; the split is a random draw inside it, so
        # its own extremes are an accident that would reject validation rows
        # sitting 1e-4 dex outside them.  Only the bounding box is read, never
        # a profile, so no fitted quantity sees a held-out row.
        support=fit_label_support(corpus.labels),
        gradient_floor=float(gradient_floor),
    )


def predict_monotone_reduced_state(
    labels: np.ndarray,
    tau: np.ndarray,
    parameters: MonotoneProfileParameters,
    *,
    check_support: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(column_mass, temperature, log10_opacity)`` from labels."""

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if check_support:
        require_label_support(values, parameters.support)
    depth = np.asarray(tau, dtype=np.float64)
    if depth.size != parameters.temperature.basis_by_regime.shape[2]:
        raise ValueError("tau length does not match the fitted profile basis")
    if depth.size != parameters.opacity.basis_by_regime.shape[2]:
        raise ValueError("tau length does not match the fitted opacity basis")

    log_opacity = evaluate_profile_closure(values, parameters.opacity)
    residual = evaluate_profile_closure(values, parameters.temperature)
    predicted = grey_temperature(values[:, 0], depth) * 10.0 ** np.clip(
        residual, -30.0, 30.0
    )
    temperature = project_to_monotone(
        depth, values[:, 0], predicted, floor=parameters.gradient_floor
    )
    column_mass = integrate_mass_from_opacity(depth, log_opacity)
    return column_mass, temperature, log_opacity


def save_monotone_profile_parameters(
    path: Path | str, parameters: MonotoneProfileParameters
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


def load_monotone_profile_parameters(path: Path | str) -> MonotoneProfileParameters:
    """Load a portable parameter asset and validate its format marker."""

    with np.load(Path(path), allow_pickle=False) as data:
        marker = str(np.asarray(data["format"]).item())
        if marker != FORMAT_MARKER:
            raise ValueError(f"unsupported monotone profile parameter format: {marker}")
        return MonotoneProfileParameters(
            temperature=_load_closure_arrays(data, "temperature"),
            opacity=_load_closure_arrays(data, "opacity"),
            support=LabelSupport(
                lower=np.asarray(data["support_lower"], dtype=np.float64),
                upper=np.asarray(data["support_upper"], dtype=np.float64),
            ),
            gradient_floor=float(np.asarray(data["gradient_floor"]).item()),
        )
