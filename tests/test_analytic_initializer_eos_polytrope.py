"""Unit tests for the solver-driven EOS polytrope projection."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.analytic_initializer.eos_polytrope import (
    project_log_temperature,
)


def _pressure(layer_count: int = 10) -> np.ndarray:
    return np.exp(np.linspace(0.0, 4.0, layer_count))


def test_constant_gamma_recovers_the_analytic_polytrope() -> None:
    pressure = _pressure()
    temperature = np.full(pressure.size, 1000.0)
    grad_ad = np.full(pressure.size, 0.4)
    current = grad_ad.copy()
    current[2:] += 0.1

    projected, diagnostics = project_log_temperature(
        temperature, pressure, current, grad_ad, min_layer=2
    )

    expected = temperature.copy()
    expected[2:] = 1000.0 * (pressure[2:] / pressure[1]) ** 0.4
    np.testing.assert_allclose(projected, expected, rtol=0.0, atol=1.0e-11)
    assert diagnostics["projected"] is True
    assert diagnostics["component_count"] == 1
    assert diagnostics["adiabatic_residual_max"] < 1.0e-12


def test_variable_eos_gradient_is_integrated_layer_by_layer() -> None:
    pressure = _pressure()
    temperature = np.full(pressure.size, 1200.0)
    grad_ad = np.linspace(0.20, 0.40, pressure.size)
    current = grad_ad.copy()
    current[2:7] += 0.05

    projected, diagnostics = project_log_temperature(
        temperature, pressure, current, grad_ad, min_layer=2
    )

    expected_log_temperature = np.log(temperature)
    for index in range(1, 6):
        expected_log_temperature[index + 1] = expected_log_temperature[index] + (
            0.5 * (grad_ad[index] + grad_ad[index + 1])
            * (np.log(pressure[index + 1]) - np.log(pressure[index]))
        )
    expected_log_temperature[7:] += (
        expected_log_temperature[6] - np.log(temperature[6])
    )
    np.testing.assert_allclose(
        projected, np.exp(expected_log_temperature), rtol=0.0, atol=1.0e-11
    )
    assert diagnostics["component_count"] == 1


def test_no_convection_is_an_exact_no_op() -> None:
    pressure = _pressure()
    temperature = np.linspace(1000.0, 2500.0, pressure.size)
    grad_ad = np.full(pressure.size, 0.4)
    current = grad_ad - 0.05

    projected, diagnostics = project_log_temperature(
        temperature, pressure, current, grad_ad
    )

    np.testing.assert_array_equal(projected, temperature)
    assert diagnostics["projected"] is False
    assert diagnostics["temperature_unchanged"] is True


def test_multiple_components_preserve_stable_region_gradients() -> None:
    pressure = _pressure()
    temperature = np.full(pressure.size, 1000.0)
    grad_ad = np.full(pressure.size, 0.3)
    current = grad_ad.copy()
    current[[2, 3, 6, 7]] += 0.1

    projected, diagnostics = project_log_temperature(
        temperature, pressure, current, grad_ad, min_layer=1
    )
    log_temperature = np.log(projected)
    log_pressure = np.log(pressure)

    assert diagnostics["component_count"] == 2
    # The stable layers after each component retain the original local slope.
    assert log_temperature[4] - log_temperature[3] == pytest.approx(0.0)
    assert log_temperature[8] - log_temperature[7] == pytest.approx(0.0)
    # The projected convective slopes are the EOS slopes.
    slopes = np.diff(log_temperature) / np.diff(log_pressure)
    np.testing.assert_allclose(slopes[[1, 2, 5, 6]], 0.3, atol=1.0e-12)


@pytest.mark.parametrize(
    "pressure",
    [np.array([1.0, 0.0, 2.0]), np.array([1.0, 2.0, 1.5])],
)
def test_invalid_pressure_is_rejected(pressure: np.ndarray) -> None:
    temperature = np.full(pressure.size, 1000.0)
    gradient = np.full(pressure.size, 0.5)
    grad_ad = np.full(pressure.size, 0.4)
    with pytest.raises(ValueError, match="pressure"):
        project_log_temperature(temperature, pressure, gradient, grad_ad, min_layer=1)


def test_invalid_active_eos_gradient_is_rejected() -> None:
    pressure = _pressure()
    temperature = np.full(pressure.size, 1000.0)
    gradient = np.full(pressure.size, 0.5)
    grad_ad = np.full(pressure.size, 0.4)
    grad_ad[3] = 0.0

    with pytest.raises(ValueError, match="adiabatic gradient"):
        project_log_temperature(temperature, pressure, gradient, grad_ad, min_layer=1)


def test_projection_module_has_no_emulator_dependency() -> None:
    source = __import__(
        "experiments.analytic_initializer.eos_polytrope",
        fromlist=["__file__"],
    ).__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    assert "torch" not in text.lower()
    assert "checkpoint" not in text.lower()
