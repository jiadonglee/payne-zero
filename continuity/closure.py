"""Optical-depth closure measurements on converged truth atmospheres.

The solver ties column mass to optical depth by ``dtau = kappa_R dm``, evaluated
as a parabolic quadrature over the column-mass coordinate and seeded at the top
of the grid with

    tau[0] = kappa_R[0] * m[0]

(``rosseland_mean.py:72-78``). That seed approximates ``int_0^{m0} kappa dm`` by
assuming kappa is constant everywhere above the first grid point.

Ting's continuity hypothesis is that the 13% upper-layer column-mass drift he
measured when rematerialising the six fields from ``(m, T)`` is an artifact of
the fixed 80-point depth grid ``tau_std = 10**(-6.875 + 0.125*i)``, and that a
denser grid — or a predicted rather than fixed tau — would remove it.

This module supplies the primitives that test it. Nothing here recomputes
physics: every measurement runs on the stored truth fields, which is what makes
the scan cheap enough to run over the whole 52,199-star corpus.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

from payne_zero_atmosphere.radiative_transfer import integrate_on_depth_grid

#: Field order of ``atmosphere_profiles`` in the training corpus.
TARGET_FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)

COLUMN_MASS = TARGET_FIELDS.index("column_mass")
ROSSELAND_OPACITY = TARGET_FIELDS.index("rosseland_opacity")


def seed_residual(column_mass: np.ndarray, opacity: np.ndarray, tau: np.ndarray):
    """Return ``log10(kappa0*m0 / tau0)`` for a batch of atmospheres.

    This is the top-of-grid seed alone, in closed form — no quadrature is
    involved, which is exactly why grid refinement cannot move it. Shapes are
    ``(star, layer)``; only layer 0 is read.
    """

    m0 = np.asarray(column_mass, dtype=np.float64)[..., 0]
    k0 = np.asarray(opacity, dtype=np.float64)[..., 0]
    t0 = np.asarray(tau, dtype=np.float64)[..., 0]
    return np.log10(k0 * m0) - np.log10(t0)


def refine_log_grid(log_tau: np.ndarray, factor: int) -> np.ndarray:
    """Subdivide every interval of ``log_tau`` into ``factor`` pieces.

    The endpoints are preserved, so the top boundary ``tau_min`` — and with it
    the seed — is unchanged. That is deliberate: densifying the grid and moving
    its top boundary are the two separate refinements Ting's hypothesis
    conflates, and only the first is tested here.
    """

    if factor == 1:
        return np.asarray(log_tau, dtype=np.float64)
    pieces = [
        np.linspace(log_tau[i], log_tau[i + 1], factor + 1)[:-1]
        for i in range(len(log_tau) - 1)
    ]
    return np.concatenate(pieces + [np.asarray(log_tau[-1:], dtype=np.float64)])


def closure_residual_at_refinement(
    column_mass: np.ndarray,
    opacity: np.ndarray,
    tau: np.ndarray,
    factor: int,
) -> np.ndarray:
    """Re-run the solver's own closure on a ``factor``-times finer grid.

    ``column_mass`` and ``opacity`` are interpolated onto the finer grid with a
    cubic spline in log-log — both are smooth in these coordinates — and the
    unmodified ``integrate_on_depth_grid`` is applied. The result is read back
    at the original nodes and returned as ``log10(tau_closure / tau_std)``.

    A residual that shrinks with ``factor`` means the 80-point quadrature is
    under-resolved. A residual that does not move means it is already converged.
    """

    log_tau = np.log10(np.asarray(tau, dtype=np.float64))
    log_m = np.log10(np.asarray(column_mass, dtype=np.float64))
    log_k = np.log10(np.asarray(opacity, dtype=np.float64))

    if factor == 1:
        fine_m, fine_k = 10.0**log_m, 10.0**log_k
        take = np.arange(len(log_tau))
    else:
        fine_log_tau = refine_log_grid(log_tau, factor)
        fine_m = 10.0 ** CubicSpline(log_tau, log_m)(fine_log_tau)
        fine_k = 10.0 ** CubicSpline(log_tau, log_k)(fine_log_tau)
        take = np.arange(0, len(fine_log_tau), factor)

    closure = integrate_on_depth_grid(
        fine_m, fine_k, surface_value=float(fine_k[0] * fine_m[0])
    )
    return np.log10(closure[take][: len(log_tau)]) - log_tau


def residual_to_column_mass_drift(residual: np.ndarray) -> np.ndarray:
    """Convert a closure residual in dex to the equivalent fractional drift in m.

    Near the top ``tau`` is very nearly linear in ``m``, so restoring the grid
    value of tau at fixed opacity means scaling m by ``10**(-residual)``.
    """

    return 10.0 ** (-np.asarray(residual, dtype=np.float64)) - 1.0


def load_corpus(path):
    """Return ``(profiles, tau, iterations, labels)`` from the truth corpus."""

    import json

    data = np.load(path, allow_pickle=True)
    labels = [json.loads(text) for text in data["labels_json"]]
    return (
        data["atmosphere_profiles"],
        data["standard_rosseland_optical_depth"],
        data["iterations_to_convergence"].astype(np.float64),
        labels,
    )
