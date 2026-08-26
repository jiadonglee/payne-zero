"""Monotone-by-construction temperature, and a support box for the labels.

H2 predicts ``log10(T / T_grey)`` layer by layer, so nothing stops the fitted
residual from bending the profile back on itself: 858 of the 52199 corpus rows
(1.64 percent) come back with a temperature inversion the truth does not have,
concentrated at the hot end and reaching -122 K per layer.  The solver's own
initializer contract already refuses a non-monotone column mass
(``payne_zero_atmosphere/warm_start.py``); temperature has no equivalent guard,
so an unphysical start is accepted in silence.

The fix here is the one H2 already uses for column mass.  Mass is safe because
the formula predicts ``log10(kappa)`` and integrates ``dm/dtau = 1/kappa``: the
integrand cannot change sign, so the integral cannot turn around.  Temperature
gets the same treatment.  The profile is carried not as eighty layer values but
as a top-layer anchor plus the logarithm of the per-interval increment in
``ln T``, summed along the grid.  Because ``10**v > 0`` for every finite ``v``,
anything expressed this way is strictly increasing, so monotonicity becomes a
property of the representation rather than something a caller has to test for.

The increments are defined on intervals rather than on layers, which makes the
discrete sum the exact inverse of the discrete difference on any grid: a round
trip through the transform returns the input to machine precision, and the
reconstruction adds no discretization error of its own on top of the fit
error.

Increments are floored before the logarithm is taken, and the floor is placed
on the *gradient* ``d ln T / d ln tau`` rather than on the increment itself.
That distinction is not cosmetic.  A per-interval floor is a statement about
the grid: halve the spacing and the same floor clamps twice as many intervals,
which made the same formula drift 1.9 percent between an eighty-layer grid and
a 791-layer one covering the identical interval.  A gradient floor scales with
the spacing and so means the same thing on every grid.

At 1e-4 the floor touches 0.0375 percent of the corpus intervals, and even a
profile clamped at every single layer could drift only 2.3e-3 in ``ln T`` over
the whole grid, because the bound is the floor times the total span of
``ln tau``.  It is invisible at the precision anything here is measured to.

Fitting and representation are deliberately separated, and the separation was
measured rather than assumed.  Fitting the increments directly -- the obvious
reading of the paragraph above -- makes the profile three times worse: held-out
temperature p95 goes from 0.0197 to 0.0560 and the deep band from 0.0196 to
0.0633, because a least-squares fit balances the error of each increment on its
own while the profile is their cumulative sum, and 79 independent errors random
walk.  So the fit stays where H2 already had it, on the cumulative quantity
``log10(T / T_grey)``, and only the final representation is the anchor and
increments: predict the profile, take its increments, floor them, integrate
back.  That composition costs 0.0004 in held-out p95 and returns a strictly
increasing profile for every row.

The 65 corpus rows whose truth really is non-monotone cannot be represented,
by design.  Their inversions sit in intervals 0-4 and 78 -- the first and last
intervals of the grid -- and amount to 69 intervals out of 4.1 million.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .discovery import LABEL_FIELDS, grey_temperature

#: Smallest ``d ln T / d ln tau`` the transform will represent.  A gradient
#: rather than an increment, so that it means the same thing on any grid; see
#: the module docstring for why this specific value is free.
GRADIENT_FLOOR = 1.0e-4


def log_increment_target(
    tau: np.ndarray,
    temperature: np.ndarray,
    *,
    floor: float = GRADIENT_FLOOR,
) -> np.ndarray:
    """Return ``log10`` of the floored per-interval increment of ``ln T``.

    The result has one fewer column than ``temperature``: increments live on
    the intervals between layers, which is what makes the reconstruction an
    exact inverse.  ``floor`` bounds ``d ln T / d ln tau``, so the increment it
    enforces on each interval is proportional to that interval's width.
    """

    depth = np.asarray(tau, dtype=np.float64)
    values = np.asarray(temperature, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != depth.size:
        raise ValueError("temperature must have shape (N, len(tau))")
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("temperature must be finite and positive")
    if not floor > 0.0 or not np.isfinite(floor):
        raise ValueError("floor must be finite and positive")
    span = np.diff(np.log(depth))
    if np.any(span <= 0.0):
        raise ValueError("tau must be strictly increasing")
    increments = np.diff(np.log(values), axis=1)
    return np.log10(np.maximum(increments, float(floor) * span[None, :]))


def anchor_target(
    tau: np.ndarray,
    temperature: np.ndarray,
    effective_temperature: np.ndarray,
) -> np.ndarray:
    """Return the top-layer anchor as ``log10(T_top / T_grey_top)``.

    Normalizing by the grey temperature keeps the anchor a small dimensionless
    number -- it spans roughly -0.21 to -0.07 across the corpus -- so the
    quantity being fitted is a correction rather than a temperature.  The shape
    is ``(N, 1)`` so the anchor can be fitted by the same closure machinery as
    a profile of depth one.
    """

    depth = np.asarray(tau, dtype=np.float64)
    values = np.asarray(temperature, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != depth.size:
        raise ValueError("temperature must have shape (N, len(tau))")
    if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("temperature must be finite and positive")
    grey = grey_temperature(effective_temperature, depth)
    return np.log10(values[:, :1] / grey[:, :1])


def rebuild_temperature(
    tau: np.ndarray,
    effective_temperature: np.ndarray,
    anchor: np.ndarray,
    log_increments: np.ndarray,
) -> np.ndarray:
    """Rebuild a strictly increasing temperature from anchor and increments.

    This is the exact inverse of ``anchor_target`` and ``log_increment_target``
    up to the floor those apply.  Strict monotonicity holds for any finite
    input because every increment is an exponential.
    """

    depth = np.asarray(tau, dtype=np.float64)
    start = np.asarray(anchor, dtype=np.float64).reshape(-1, 1)
    steps = np.asarray(log_increments, dtype=np.float64)
    if steps.ndim != 2 or steps.shape[1] != depth.size - 1:
        raise ValueError("log_increments must have shape (N, len(tau) - 1)")
    if start.shape[0] != steps.shape[0]:
        raise ValueError("anchor and log_increments must agree on the row count")
    grey = grey_temperature(effective_temperature, depth)
    # Clipping only guards the exponential against a pathological fit; the
    # bound is far outside the corpus range of the fitted quantity.
    log_top = np.log(grey[:, :1]) + np.log(10.0) * np.clip(start, -30.0, 30.0)
    increments = 10.0 ** np.clip(steps, -30.0, 30.0)
    log_temperature = np.concatenate(
        [log_top, log_top + np.cumsum(increments, axis=1)], axis=1
    )
    return np.exp(log_temperature)


def project_to_monotone(
    tau: np.ndarray,
    effective_temperature: np.ndarray,
    temperature: np.ndarray,
    *,
    floor: float = GRADIENT_FLOOR,
) -> np.ndarray:
    """Return the predicted profile as an element of the monotone family.

    The round trip is deliberate rather than wasteful: it is what makes the
    output a member of the anchor-and-increments family by construction, so
    strict monotonicity is a property of the representation and not something
    a caller has to test for.  Rows that were already increasing come back
    unchanged to within 9e-16.
    """

    return rebuild_temperature(
        tau,
        effective_temperature,
        anchor_target(tau, temperature, effective_temperature),
        log_increment_target(tau, temperature, floor=floor),
    )


@dataclass(frozen=True)
class LabelSupport:
    """The label box a set of fitted constants is allowed to be used in.

    A degree-three polynomial in five labels is unbounded outside the box it
    was fitted in, and it fails loudly nowhere: evaluating the H2 constants at
    Teff = 12000 K, one and a half thousand kelvin above the corpus, returns a
    profile peaking at 62543 K.  Refusing the call is the only honest answer,
    and it matches what the production initializer already does through
    ``_require_initializer_bounds``.
    """

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.shape != (len(LABEL_FIELDS),) or upper.shape != lower.shape:
            raise ValueError(f"support bounds must have shape ({len(LABEL_FIELDS)},)")
        if np.any(upper < lower):
            raise ValueError("support upper bounds must not fall below lower bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


def fit_label_support(labels: np.ndarray) -> LabelSupport:
    """Take the support box from the rows the constants were fitted on."""

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABEL_FIELDS):
        raise ValueError(f"labels must have shape (N, {len(LABEL_FIELDS)})")
    if values.shape[0] == 0:
        raise ValueError("cannot take a support box from an empty label set")
    return LabelSupport(lower=values.min(axis=0), upper=values.max(axis=0))


def require_label_support(labels: np.ndarray, support: LabelSupport) -> None:
    """Raise if any row falls outside the fitted support box."""

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != len(LABEL_FIELDS):
        raise ValueError(f"labels must have shape (N, {len(LABEL_FIELDS)})")
    if not np.all(np.isfinite(values)):
        raise ValueError("labels must be finite")
    below = values < support.lower
    above = values > support.upper
    if not (below.any() or above.any()):
        return
    violations = []
    for index, name in enumerate(LABEL_FIELDS):
        if not (below[:, index].any() or above[:, index].any()):
            continue
        column = values[:, index]
        violations.append(
            f"{name} in [{column.min():.6g}, {column.max():.6g}] outside "
            f"[{support.lower[index]:.6g}, {support.upper[index]:.6g}]"
        )
    raise ValueError(
        "labels are outside the fitted analytic support: " + ", ".join(violations)
    )
