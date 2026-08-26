"""Tests for the first physics-shaped analytic candidate."""

from __future__ import annotations

import numpy as np

from experiments.analytic_initializer.candidates import (
    build_h1_reduced_state,
    temperature_regimes,
)


def test_temperature_regimes_cover_the_full_input() -> None:
    labels = np.array(
        [
            [5000.0, 2.0, -1.0, 0.1, 1.0],
            [6500.0, 3.0, -0.5, 0.2, 2.0],
            [9000.0, 4.0, 0.0, 0.3, 3.0],
        ]
    )
    np.testing.assert_array_equal(temperature_regimes(labels), [0, 1, 2])


def test_h1_grey_state_is_positive_and_strictly_monotone() -> None:
    labels = np.array(
        [
            [4500.0, 1.0, -2.0, 0.0, 0.8],
            [5777.0, 4.44, 0.0, 0.0, 2.0],
            [10000.0, 5.0, 0.3, 0.4, 3.5],
        ]
    )
    tau = 10.0 ** (-6.875 + 0.125 * np.arange(80))
    mass, temperature, opacity = build_h1_reduced_state(labels, tau)
    assert mass.shape == temperature.shape == (3, 80)
    assert opacity.shape == (3,)
    assert np.all(np.isfinite(mass))
    assert np.all(np.isfinite(temperature))
    assert np.all(mass > 0.0)
    assert np.all(temperature > 0.0)
    assert np.all(np.diff(mass, axis=1) > 0.0)
