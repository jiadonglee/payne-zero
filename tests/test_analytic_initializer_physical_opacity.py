"""Tests for the positive local-opacity closure probe."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.analytic_initializer.physical_opacity import (
    integrate_self_consistent_mass,
    local_opacity_features,
    profile_invariants,
    temperature_regime_weights,
)


def _labels() -> np.ndarray:
    return np.array(
        [
            [4500.0, 1.0, -2.0, 0.0, 0.8],
            [6500.0, 3.0, -0.5, 0.2, 2.0],
            [9500.0, 5.0, 0.2, 0.4, 3.5],
        ]
    )


def test_local_features_and_smooth_regime_weights_are_finite() -> None:
    labels = _labels()
    tau = 10.0 ** (-6.875 + 0.125 * np.arange(8))
    temperature = labels[:, 0, None] * (0.75 * (tau[None, :] + 2.0 / 3.0)) ** 0.25
    pressure = 10.0 ** labels[:, 1, None] * tau[None, :]
    rows = local_opacity_features(
        np.repeat(labels, tau.size, axis=0),
        temperature.reshape(-1),
        pressure.reshape(-1),
        np.tile(tau, labels.shape[0]),
    )
    assert rows.shape == (labels.shape[0] * tau.size, 8)
    assert np.all(np.isfinite(rows))
    weights = temperature_regime_weights(labels)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights >= 0.0)


def test_fixed_point_mass_is_positive_and_monotone() -> None:
    labels = _labels()
    tau = 10.0 ** (-6.875 + 0.125 * np.arange(8))
    temperature = labels[:, 0, None] * (0.75 * (tau[None, :] + 2.0 / 3.0)) ** 0.25
    # A constant positive closure is enough to test the integration contract.
    from experiments.analytic_initializer.physical_opacity import LocalOpacityParameters
    from experiments.analytic_initializer.discovery import polynomial_exponents

    exponents = polynomial_exponents(8, 0)
    parameters = LocalOpacityParameters(
        degree=0,
        exponents=exponents,
        feature_center=np.zeros(8),
        feature_scale=np.ones(8),
        coefficients_by_regime=np.zeros((3, 1)),
    )
    mass, log_opacity = integrate_self_consistent_mass(
        labels, temperature, tau, parameters, iterations=3
    )
    assert mass.shape == temperature.shape
    assert log_opacity.shape == temperature.shape
    assert profile_invariants(mass, temperature) == {
        "nonfinite_mass_profiles": 0,
        "nonpositive_mass_profiles": 0,
        "nonmonotone_mass_profiles": 0,
        "nonfinite_temperature_profiles": 0,
        "nonpositive_temperature_profiles": 0,
    }


def test_no_emulator_bridge_materializes_a_valid_deck() -> None:
    try:
        from experiments.analytic_initializer.no_emulator_bridge import analytic_seed_model
        import bench.environment  # noqa: F401 - configure Numba before solver import
    except ImportError as exc:
        pytest.skip(f"production solver environment unavailable: {exc}")
    labels = _labels()[1]
    tau = 10.0 ** (-6.875 + 0.125 * np.arange(8))
    mass = tau / 0.34
    temperature = labels[0] * (0.75 * (tau + 2.0 / 3.0)) ** 0.25
    try:
        model = analytic_seed_model(
            labels,
            mass,
            temperature,
            np.log10(np.full(tau.size, 0.34)),
            tau,
        )
    except ImportError as exc:
        pytest.skip(f"production solver environment unavailable: {exc}")
    assert model.layers == tau.size
    assert np.all(model.gas_pressure > 0.0)
    assert np.all(model.electron_density > 0.0)
    assert np.all(np.diff(model.column_mass) > 0.0)
