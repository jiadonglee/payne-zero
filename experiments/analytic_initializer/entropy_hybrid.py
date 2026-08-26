"""The polytrope route, retried on convergence instead of on dex.

The v1 and v2 entropy closures were vetoed by a pre-registered gate of 0.015 to
0.020 dex on the deep band.  That gate has since been shown not to predict
anything: five arms spanning a factor of 2.7 in offline accuracy all converge
within 53 to 55 of 60 stars, the most accurate one is no faster, and the least
accurate one is the only one that reaches a star nothing else does.  Measured
on the vetoed families' own statistic in their own hard bin, the arms that
actually work sit at 0.18 to 0.28 while v2 was refused at 0.099.  The gate was
an order of magnitude stricter than anything the project has ever run.

So the family is re-opened, under a decision rule written down first, in
``notes/entropy_closure_convergence_retest.md``.

What this module builds is deliberately the smallest thing that answers the
question.  Only the convective part is substituted; the two halves that are
already validated are reused unchanged:

* column mass comes from the compact formula's opacity closure -- a positive
  kappa integrated through ``dm/dtau = 1/kappa``, so mass is monotone by
  construction and is literally the integrated opacity;
* the radiative temperature branch comes from the compact formula too;
* pressure is ``P = g m`` from that mass, never truth;
* below the Schwarzschild crossing the gradient is replaced by the dual-crossing
  closure and integrated downward.

Integration is anchored *at the crossing*, not at the surface.  Anchoring at
the surface would rebuild the whole profile out of an integrated gradient and
put the well-behaved radiative region at the mercy of accumulated quadrature
error, which is not the thing under test.  Above the crossing the compact
profile is kept exactly.

The closure itself carries **three global constants** -- an adiabatic gradient
and two boundary-bump amplitudes -- with no label dependence at all.  That is
the most polytrope-shaped, most "good enough guess" form the family has, and it
is the version worth testing first: if three numbers are enough the result is
interesting, and if they are not, a label-dependent version is the obvious next
step rather than a rescue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .compact_initializer import CompactProfileParameters, predict_compact_reduced_state
from .entropy_closure_v2 import (
    AE_MAX,
    AX_MAX,
    AX_MIN,
    GAMMA_AD_MAX,
    GAMMA_AD_MIN,
    SWITCH_WIDTH_DEX,
    dual_crossing_gradient,
    radiative_gradient,
    schwarzschild_crossings,
)
from .monotone_temperature import GRADIENT_FLOOR, project_to_monotone

#: The deep window the whole line of work reports on.
DEEP_START, DEEP_TRIM = 39, 5


@dataclass(frozen=True)
class EntropyHybridClosure:
    """Three global constants: an adiabat and two boundary bumps."""

    gamma_ad: float
    a_enter: float
    a_exit: float

    def __post_init__(self) -> None:
        if not GAMMA_AD_MIN <= self.gamma_ad <= GAMMA_AD_MAX:
            raise ValueError(f"gamma_ad must lie in [{GAMMA_AD_MIN}, {GAMMA_AD_MAX}]")
        if not 0.0 <= self.a_enter <= AE_MAX:
            raise ValueError(f"a_enter must lie in [0, {AE_MAX}]")
        if not AX_MIN <= self.a_exit <= AX_MAX:
            raise ValueError(f"a_exit must lie in [{AX_MIN}, {AX_MAX}]")

    @property
    def stored_float_count(self) -> int:
        return 3


def crossing_layers(
    log_pressure: np.ndarray,
    grad_radiative: np.ndarray,
    gamma_ad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Schwarzschild enter and exit layers for every row.

    Split out from the gradient because the scan depends only on ``gamma_ad``:
    a sweep over the two bump amplitudes can reuse one scan instead of
    repeating it, which is what makes fitting three constants cheap.
    """

    enter = np.empty(log_pressure.shape[0], dtype=np.int64)
    leave = np.empty(log_pressure.shape[0], dtype=np.int64)
    floor = np.full(log_pressure.shape[1], float(gamma_ad))
    for row in range(log_pressure.shape[0]):
        enter[row], leave[row] = schwarzschild_crossings(
            log_pressure[row], grad_radiative[row], floor
        )
    return enter, leave


def deep_temperature(
    labels: np.ndarray,
    mass: np.ndarray,
    temperature: np.ndarray,
    log_opacity: np.ndarray,
    closure: EntropyHybridClosure,
    *,
    crossings: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Replace the temperature below the crossing with the closure's.

    Rows that never go convective are returned untouched, which is the correct
    answer rather than a fallback: no crossing means the radiative branch is
    the whole story.
    """

    values = np.asarray(labels, dtype=np.float64)
    gravity = 10.0 ** values[:, 1]
    pressure = np.maximum(gravity[:, None] * mass, 1.0e-300)
    log_pressure = np.log10(pressure)
    kappa = 10.0 ** np.clip(log_opacity, -30.0, 30.0)

    grad_radiative = np.empty_like(temperature)
    for row in range(values.shape[0]):
        grad_radiative[row] = radiative_gradient(
            kappa[row], pressure[row], values[row, 0], gravity[row], temperature[row]
        )

    enter, leave = (
        crossings
        if crossings is not None
        else crossing_layers(log_pressure, grad_radiative, closure.gamma_ad)
    )

    updated = temperature.copy()
    layers = temperature.shape[1]
    for row in range(values.shape[0]):
        start = int(enter[row])
        if start < 0 or start >= layers - 1:
            continue
        stop = int(leave[row])
        gradient, _, _ = dual_crossing_gradient(
            log_pressure[row],
            float(log_pressure[row, start]),
            float(log_pressure[row, stop]) if stop < layers else float(log_pressure[row, -1]) + 3.0,
            grad_radiative[row],
            np.full(layers, closure.gamma_ad),
            closure.a_enter,
            closure.a_exit,
            width_dex=SWITCH_WIDTH_DEX,
        )
        # Anchored at the crossing: everything above it is the compact profile,
        # untouched, so only the convective region is under test.  Trapezoid
        # rather than a left Riemann sum -- measured, not assumed: integrating
        # the *truth* gradient over this window costs 0.062 in mean deep
        # |d ln T| with a left sum and 0.012 with a trapezoid, and 0.062 is as
        # large as the entire error of the profile being replaced.
        step = np.diff(log_pressure[row]) * np.log(10.0)
        midpoint = 0.5 * (gradient[start:-1] + gradient[start + 1 :])
        log_temperature = np.log(temperature[row, start]) + np.cumsum(
            midpoint * step[start:]
        )
        updated[row, start + 1 :] = np.exp(log_temperature)
    return updated


def predict_entropy_hybrid_state(
    labels: np.ndarray,
    tau: np.ndarray,
    compact: CompactProfileParameters,
    closure: EntropyHybridClosure,
    *,
    check_support: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(column_mass, temperature, log10_opacity)``.

    The monotone projection is applied at the end exactly as it is for every
    other arm, so the four physical invariants hold here too and the comparison
    is not confounded by one arm being guarded and another not.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    mass, temperature, log_opacity = predict_compact_reduced_state(
        values, tau, compact, check_support=check_support
    )
    replaced = deep_temperature(values, mass, temperature, log_opacity, closure)
    guarded = project_to_monotone(tau, values[:, 0], replaced, floor=GRADIENT_FLOOR)
    return mass, guarded, log_opacity


def deep_error(temperature: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-star maximum ``|d ln T|`` over the deep window.

    The statistic the vetoed oracles were scored on, so the retest can be read
    against the numbers that produced the veto.
    """

    window = slice(DEEP_START, temperature.shape[1] - DEEP_TRIM)
    return np.abs(np.log(temperature) - np.log(truth))[:, window].max(axis=1)


def fit_entropy_hybrid_closure(
    labels: np.ndarray,
    tau: np.ndarray,
    truth: np.ndarray,
    compact: CompactProfileParameters,
    *,
    gamma_grid: np.ndarray | None = None,
    amplitude_steps: int = 11,
) -> tuple[EntropyHybridClosure, dict]:
    """Fit the three constants by grid search on the deep profile error.

    A deterministic grid rather than a stochastic optimizer: three parameters
    on bounded intervals do not need differential evolution, and a grid cannot
    hand back a different answer on a different seed -- which matters for a
    family that has already been vetoed twice.

    The objective is offline, because convergence cannot be optimized against
    at this cost.  The *decision* is on convergence; this only chooses where in
    the family to stand.
    """

    values = np.asarray(labels, dtype=np.float64)
    mass, temperature, log_opacity = predict_compact_reduced_state(
        values, tau, compact, check_support=False
    )
    gravity = 10.0 ** values[:, 1]
    pressure = np.maximum(gravity[:, None] * mass, 1.0e-300)
    log_pressure = np.log10(pressure)
    kappa = 10.0 ** np.clip(log_opacity, -30.0, 30.0)
    grad_radiative = np.empty_like(temperature)
    for row in range(values.shape[0]):
        grad_radiative[row] = radiative_gradient(
            kappa[row], pressure[row], values[row, 0], gravity[row], temperature[row]
        )

    if gamma_grid is None:
        gamma_grid = np.linspace(GAMMA_AD_MIN, GAMMA_AD_MAX, 8)
    enters = np.linspace(0.0, AE_MAX, amplitude_steps)
    exits = np.linspace(AX_MIN, AX_MAX, amplitude_steps)

    best: tuple[float, EntropyHybridClosure] | None = None
    trace = []
    for gamma in gamma_grid:
        # One scan per gamma, reused across the amplitude sweep.
        crossings = crossing_layers(log_pressure, grad_radiative, float(gamma))
        for a_enter in enters:
            for a_exit in exits:
                candidate = EntropyHybridClosure(float(gamma), float(a_enter), float(a_exit))
                replaced = deep_temperature(
                    values, mass, temperature, log_opacity, candidate, crossings=crossings
                )
                guarded = project_to_monotone(tau, values[:, 0], replaced)
                score = float(np.mean(deep_error(guarded, truth)))
                trace.append({"gamma_ad": float(gamma), "a_enter": float(a_enter),
                              "a_exit": float(a_exit), "mean_deep_error": score})
                if best is None or score < best[0]:
                    best = (score, candidate)

    assert best is not None
    return best[1], {
        "objective": "mean over stars of the per-star max |d ln T| in the deep window",
        "stars": int(values.shape[0]),
        "grid": {
            "gamma_ad": [float(x) for x in gamma_grid],
            "amplitude_steps": amplitude_steps,
        },
        "best_mean_deep_error": best[0],
        "evaluations": len(trace),
    }
