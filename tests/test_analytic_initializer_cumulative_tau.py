"""Tests for the closed-form cumulative-tau initializer."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.analytic_initializer.cumulative_tau_initializer import (
    fit_cumulative_tau_parameters,
    fit_oracle_targets,
    integrated_partition_windows,
    load_cumulative_tau_parameters,
    logistic_partition_windows,
    predict_cumulative_tau_state,
    save_cumulative_tau_parameters,
)
from experiments.analytic_initializer.discovery import grey_temperature, label_features


def _synthetic():
    generator = np.random.default_rng(17)
    rows = 60
    labels = np.column_stack(
        (
            generator.uniform(4000.0, 10000.0, rows),
            generator.uniform(0.5, 5.0, rows),
            generator.uniform(-2.5, 0.5, rows),
            generator.uniform(-0.1, 0.5, rows),
            generator.uniform(0.5, 4.0, rows),
        )
    )
    tau = np.logspace(-6.0, 3.0, 80)
    width = 0.7
    anchor_tau = tau[np.argmin(np.abs(tau - 0.013335))]
    integrated = integrated_partition_windows(
        np.log(tau), np.log(anchor_tau), width=width
    )
    features = label_features(labels)
    normalized = (features - features.mean(axis=0)) / features.std(axis=0)
    t_anchor = -0.10 + 0.01 * normalized[:, 0] - 0.005 * normalized[:, 2]
    m_anchor = -2.0 + 0.08 * normalized[:, 1] + 0.04 * normalized[:, 2]
    t_slopes = np.exp(
        np.column_stack(
            (
                -2.0 + 0.05 * normalized[:, 0],
                -1.5 + 0.03 * normalized[:, 1],
                -1.8 + 0.02 * normalized[:, 2],
                -2.2 + 0.02 * normalized[:, 3],
            )
        )
    )
    m_slopes = np.exp(
        np.column_stack(
            (
                -0.1 + 0.02 * normalized[:, 0],
                0.1 + 0.02 * normalized[:, 1],
                -0.2 + 0.02 * normalized[:, 2],
                -0.3 + 0.02 * normalized[:, 4],
            )
        )
    )
    grey_anchor = grey_temperature(labels[:, 0], np.asarray([anchor_tau]))[:, 0]
    temperature = np.exp(
        (np.log(grey_anchor) + t_anchor)[:, None] + t_slopes @ integrated.T
    )
    mass = np.exp(m_anchor[:, None] + m_slopes @ integrated.T)
    targets = fit_oracle_targets(
        tau, temperature, mass, labels[:, 0], width=width
    )
    parameters = fit_cumulative_tau_parameters(
        labels,
        tau,
        targets,
        np.arange(45),
        degree=1,
        width=width,
        support_indices=np.arange(rows),
    )
    return labels, tau, parameters


def test_logistic_windows_are_nonnegative_partition_of_unity() -> None:
    x = np.linspace(-30.0, 20.0, 1001)
    windows = logistic_partition_windows(x, width=0.35)
    assert windows.shape == (x.size, 4)
    assert np.all(windows >= 0.0)
    np.testing.assert_allclose(windows.sum(axis=1), 1.0, atol=2.0e-15)


def test_analytic_integral_derivative_matches_windows() -> None:
    x = np.linspace(-12.0, 8.0, 101)
    epsilon = 1.0e-6
    plus = integrated_partition_windows(x + epsilon, -4.3, width=0.7)
    minus = integrated_partition_windows(x - epsilon, -4.3, width=0.7)
    derivative = (plus - minus) / (2.0 * epsilon)
    expected = logistic_partition_windows(x, width=0.7)
    np.testing.assert_allclose(derivative, expected, rtol=2.0e-8, atol=2.0e-9)


def test_prediction_is_positive_monotone_and_obeys_opacity_identity() -> None:
    labels, tau, parameters = _synthetic()
    prediction = predict_cumulative_tau_state(labels[45:], tau, parameters)
    assert np.all(np.isfinite(prediction.temperature))
    assert np.all(np.isfinite(prediction.column_mass))
    assert np.all(prediction.temperature > 0.0)
    assert np.all(prediction.column_mass > 0.0)
    assert np.all(np.diff(prediction.temperature, axis=1) > 0.0)
    assert np.all(np.diff(prediction.column_mass, axis=1) > 0.0)
    assert np.all(prediction.opacity > 0.0)
    identity = (
        prediction.opacity
        * prediction.column_mass
        * prediction.mass_log_slope
        / tau[None, :]
    )
    np.testing.assert_allclose(identity, 1.0, rtol=2.0e-15, atol=2.0e-15)


def test_label_and_tau_support_are_guarded() -> None:
    labels, tau, parameters = _synthetic()
    outside = labels[0].copy()
    outside[0] = parameters.support_upper[0] + 1.0
    with pytest.raises(ValueError, match="labels are outside"):
        predict_cumulative_tau_state(outside, tau, parameters)
    with pytest.raises(ValueError, match="tau is outside"):
        predict_cumulative_tau_state(
            labels[0], np.logspace(-7.0, 3.0, 80), parameters
        )


def test_parameter_asset_round_trip_preserves_prediction(tmp_path) -> None:
    labels, tau, parameters = _synthetic()
    path = save_cumulative_tau_parameters(tmp_path / "cumulative.npz", parameters)
    reloaded = load_cumulative_tau_parameters(path)
    before = predict_cumulative_tau_state(labels[45:], tau, parameters)
    after = predict_cumulative_tau_state(labels[45:], tau, reloaded)
    for name in ("temperature", "column_mass", "opacity", "mass_log_slope"):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
