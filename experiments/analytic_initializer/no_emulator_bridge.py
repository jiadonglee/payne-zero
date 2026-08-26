"""Build a solver-compatible seed from an analytic reduced state.

The bridge is metadata-only: it never loads a checkpoint.  The solver will
replace the provisional pressure, electron density, and acceleration during
its physical passes; they are present only because the DECK6 interface
requires a complete positive table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type-checking-only imports
    from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere


def _electron_fraction(temperature: np.ndarray) -> np.ndarray:
    """A bounded ionization-scale guess, not a replacement for the EOS."""

    values = np.asarray(temperature, dtype=np.float64)
    hot = 1.0 / (1.0 + np.exp(-(values - 7000.0) / 1200.0))
    cool = 1.0e-5 + 0.15 * hot
    return np.clip(cool, 1.0e-5, 0.2)


def analytic_seed_model(
    labels: np.ndarray,
    column_mass: np.ndarray,
    temperature: np.ndarray,
    log_opacity: np.ndarray,
    tau: np.ndarray,
) -> ModelAtmosphere:
    """Materialize one or many analytic reduced states as DECK6 models."""

    # Keep solver imports inside the bridge.  Offline discovery and its unit
    # tests must remain usable in a lightweight environment whose Numba/NumPy
    # pair cannot import the production package.
    from payne_zero_atmosphere.atmosphere_io import parse_atmosphere_deck
    from payne_zero_atmosphere.constants import BOLTZMANN_ERG_PER_K_REFERENCE
    from payne_zero_atmosphere.warm_start import format_warm_start_deck

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 1 or values.shape != (5,):
        raise ValueError("labels must have shape (5,)")
    depth = np.asarray(tau, dtype=np.float64)
    mass = np.asarray(column_mass, dtype=np.float64)
    thermal = np.asarray(temperature, dtype=np.float64)
    opacity = 10.0 ** np.clip(np.asarray(log_opacity, dtype=np.float64), -12.0, 6.0)
    if not (mass.shape == thermal.shape == opacity.shape == depth.shape):
        raise ValueError("analytic seed fields must share the tau grid")
    if np.any(~np.isfinite(mass)) or np.any(~np.isfinite(thermal)) or np.any(~np.isfinite(opacity)):
        raise ValueError("analytic seed fields must be finite")
    if np.any(mass <= 0.0) or np.any(thermal <= 0.0) or np.any(opacity <= 0.0):
        raise ValueError("analytic seed fields must be positive")
    if np.any(np.diff(mass) <= 0.0):
        raise ValueError("analytic column mass must be strictly increasing")

    gravity = 10.0 ** float(values[1])
    gas_pressure = np.maximum(gravity * mass, 1.0e-12)
    electron_density = np.maximum(
        _electron_fraction(thermal) * gas_pressure
        / (BOLTZMANN_ERG_PER_K_REFERENCE * thermal),
        1.0e-6,
    )
    layer_table = np.zeros((depth.size, 9), dtype=np.float64)
    layer_table[:, 0] = mass
    layer_table[:, 1] = thermal
    layer_table[:, 2] = gas_pressure
    layer_table[:, 3] = electron_density
    layer_table[:, 4] = opacity
    layer_table[:, 5] = 0.0
    layer_table[:, 6] = float(values[4]) * 1.0e5
    deck = format_warm_start_deck(
        effective_temperature=float(values[0]),
        log_surface_gravity=float(values[1]),
        metallicity=float(values[2]),
        alpha_enhancement=float(values[3]),
        layer_table=layer_table,
        title="Payne Zero analytic physics closure",
    )
    return parse_atmosphere_deck(deck, source="analytic physics closure")
