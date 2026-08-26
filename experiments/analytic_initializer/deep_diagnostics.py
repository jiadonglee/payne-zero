"""Depth-resolved diagnostics shared by the analytic-initializer probes.

The solver funnel only records whether a star converged, which cannot say
*where* an initial state is wrong.  These helpers answer that offline: they
split a predicted-versus-truth error into the surface band and the deep band
the production convergence stop actually watches, and they mark which layers
the truth profiles carry convectively.

Every probe that compares initializer variants must use the same band split and
the same convection criterion, or its numbers cannot be set beside another
probe's.  That is why they live here rather than inside one runner.
"""

from __future__ import annotations

import numpy as np

# Mirrors ``payne_zero_atmosphere/convergence.py:40``: the production stop takes
# the largest relative temperature change over layers 39..layers-5 and compares
# it against ``maximum_deep_layer_relative_temperature_change`` (5.0e-4).  An
# initializer is judged on that band whether or not it was fitted for it.
DEEP_WINDOW_START = 39
DEEP_WINDOW_TRIM = 5

# Bins chosen to straddle the radiative/convective transition rather than to
# divide the label range evenly: the interesting behaviour is between 6500 and
# 9000 K, where a deep convection zone stops being universal.
TEFF_BINS = (
    (4000, 5000),
    (5000, 5750),
    (5750, 6500),
    (6500, 7000),
    (7000, 7500),
    (7500, 8000),
    (8000, 9000),
    (9000, 10500),
)

# Fraction of the radiative gradient below which a layer counts as convective.
# A strict ``grad_true < grad_rad`` would flag numerical noise in the deepest
# radiative layers, so the test asks for a clear flux deficit instead.
CONVECTIVE_FLUX_DEFICIT = 0.9


def deep_window(layers: int) -> tuple[int, int]:
    """Return the ``(start, stop)`` layer band the convergence stop watches."""

    start = DEEP_WINDOW_START
    stop = int(layers) - DEEP_WINDOW_TRIM
    if stop - start < 1:
        return 0, int(layers)
    return start, stop


def error_bands(
    corpus,
    indices: np.ndarray,
    *,
    mass: np.ndarray,
    temperature: np.ndarray,
    log_opacity: np.ndarray,
) -> dict[str, np.ndarray]:
    """Split predicted-versus-truth error into the surface and deep bands.

    ``mass``, ``temperature`` and ``log_opacity`` are the initializer's
    prediction for ``indices``, so a caller can pass any candidate's output.
    """

    rows = np.asarray(indices, dtype=np.int64)
    start, stop = deep_window(corpus.layers)
    temperature_error = np.abs(
        np.log10(temperature) - np.log10(corpus.temperature[rows])
    )
    mass_error = np.abs(np.log10(mass) - np.log10(corpus.column_mass[rows]))
    opacity_error = np.abs(log_opacity - np.log10(corpus.rosseland_opacity[rows]))
    return {
        "temperature_surface": temperature_error[:, :start].max(axis=1),
        "temperature_deep": temperature_error[:, start:stop].max(axis=1),
        "temperature_deep_argmax_layer": start
        + temperature_error[:, start:stop].argmax(axis=1),
        "mass_surface": mass_error[:, :start].max(axis=1),
        "mass_deep": mass_error[:, start:stop].max(axis=1),
        "opacity_deep": opacity_error[:, start:stop].max(axis=1),
    }


def convective_diagnostics(corpus, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(subadiabatic_mask, convective_onset_layer)`` from truth fields.

    ``grad_rad`` is the plane-parallel grey diffusion gradient
    ``(3/16) kappa P Teff^4 / (g T^4)``.  A layer counts as convective when the
    true ``dlnT/dlnP`` falls clearly below it, meaning radiation alone is no
    longer carrying the flux.  Stars with no such layer get an onset of ``-1``.
    """

    rows = np.asarray(indices, dtype=np.int64)
    gravity = 10.0 ** corpus.labels[rows, 1]
    effective_temperature = corpus.labels[rows, 0]
    pressure = corpus.gas_pressure[rows]
    temperature = corpus.temperature[rows]
    opacity = corpus.rosseland_opacity[rows]

    grad_rad = (3.0 / 16.0) * opacity * pressure * (
        effective_temperature[:, None] ** 4
    ) / (gravity[:, None] * temperature**4)
    grad_true = np.gradient(np.log(temperature), axis=1) / np.gradient(
        np.log(pressure), axis=1
    )
    subadiabatic = grad_true < CONVECTIVE_FLUX_DEFICIT * grad_rad
    onset = np.where(subadiabatic.any(axis=1), subadiabatic.argmax(axis=1), -1)
    return subadiabatic, onset


def bin_by_effective_temperature(
    effective_temperature: np.ndarray,
    deep_error: np.ndarray,
    onset: np.ndarray,
    *,
    minimum_count: int = 30,
) -> list[dict[str, object]]:
    """Summarize deep error and convection incidence per effective-temperature bin."""

    rows: list[dict[str, object]] = []
    for low, high in TEFF_BINS:
        mask = (effective_temperature >= low) & (effective_temperature < high)
        if int(mask.sum()) < minimum_count:
            continue
        with_convection = onset[mask] >= 0
        rows.append(
            {
                "effective_temperature_low": low,
                "effective_temperature_high": high,
                "star_count": int(mask.sum()),
                "temperature_deep_dex_p50": float(np.median(deep_error[mask])),
                "temperature_deep_dex_p95": float(np.quantile(deep_error[mask], 0.95)),
                "no_convection_zone_fraction": float((~with_convection).mean()),
                "convective_onset_layer_p10": (
                    float(np.percentile(onset[mask][with_convection], 10))
                    if with_convection.any()
                    else None
                ),
                "convective_onset_layer_p90": (
                    float(np.percentile(onset[mask][with_convection], 90))
                    if with_convection.any()
                    else None
                ),
            }
        )
    return rows
