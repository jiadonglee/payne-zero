"""Unit tests for the v2 dual-crossing entropy closure.

The family was vetoed by its pre-registered representation oracle, but the
module is kept as a documented negative reference, so its mechanics (counts,
serialization, crossings, monotonicity, torch-free runtime) must still hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.analytic_initializer.discovery import polynomial_exponents
from experiments.analytic_initializer.entropy_closure_v2 import (
    AE_MAX,
    AX_MAX,
    AX_MIN,
    CLOSURE_PARAM_DEGREE,
    OPACITY_DEGREE,
    OPACITY_MODES,
    TEMPERATURE_DEGREE,
    TEMPERATURE_MODES,
    EntropyClosureV2Parameters,
    FORMAT_MARKER,
    chebyshev_basis,
    dual_crossing_gradient,
    integrate_gradient,
    load_entropy_closure_v2,
    model_spec_sha256,
    predict_compact_reduced_state,
    sample_constant_parameters,
    save_entropy_closure_v2,
    schwarzschild_crossings,
)

TAU = np.logspace(-2.0, 2.0, 80)


def _minimal_parameters() -> EntropyClosureV2Parameters:
    """Healthy constants in the physical range (not random junk)."""

    from experiments.analytic_initializer.discovery import polynomial_exponents

    opacity_exponents = polynomial_exponents(5, OPACITY_DEGREE)
    temperature_exponents = polynomial_exponents(5, TEMPERATURE_DEGREE)
    surface_exponents = polynomial_exponents(5, 2)
    closure_exponents = polynomial_exponents(5, CLOSURE_PARAM_DEGREE)
    center = np.array([7000.0, 4.0, 0.0, 0.0, 2.0])
    scale = np.array([1000.0, 0.5, 0.5, 0.1, 0.5])
    opacity_coefficients = np.zeros((OPACITY_MODES, opacity_exponents.shape[0]))
    opacity_coefficients[0, 0] = -1.0
    temperature_coefficients = np.zeros((TEMPERATURE_MODES, temperature_exponents.shape[0]))
    surface_mass_coefficients = np.zeros(surface_exponents.shape[0])
    surface_mass_coefficients[0] = np.log10(0.01 / 0.1)
    gamma_ad_coefficients = np.zeros(closure_exponents.shape[0])
    a_enter_coefficients = np.zeros(closure_exponents.shape[0])
    a_exit_coefficients = np.zeros(closure_exponents.shape[0])
    return EntropyClosureV2Parameters(
        feature_center=center,
        feature_scale=scale,
        opacity_coefficients=opacity_coefficients,
        temperature_coefficients=temperature_coefficients,
        surface_mass_coefficients=surface_mass_coefficients,
        gamma_ad_coefficients=gamma_ad_coefficients,
        a_enter_coefficients=a_enter_coefficients,
        a_exit_coefficients=a_exit_coefficients,
        corpus_sha256="test",
        model_spec_sha256=model_spec_sha256(),
    )


def test_format_marker() -> None:
    assert FORMAT_MARKER == "payne_zero_entropy_closure_v2"


def test_constant_budget() -> None:
    parameters = sample_constant_parameters()
    assert parameters.base_float_count == 581
    assert parameters.fitted_float_count <= 600
    with_exit = EntropyClosureV2Parameters(
        feature_center=parameters.feature_center,
        feature_scale=parameters.feature_scale,
        opacity_coefficients=parameters.opacity_coefficients,
        temperature_coefficients=parameters.temperature_coefficients,
        surface_mass_coefficients=parameters.surface_mass_coefficients,
        gamma_ad_coefficients=parameters.gamma_ad_coefficients,
        a_enter_coefficients=parameters.a_enter_coefficients,
        a_exit_coefficients=parameters.a_exit_coefficients,
        exit_logp_coefficients=np.zeros(_closure_len()),
        corpus_sha256="c",
        model_spec_sha256="m",
    )
    assert with_exit.fitted_float_count == 587


def _closure_len() -> int:
    return int(polynomial_exponents(5, CLOSURE_PARAM_DEGREE).shape[0])


def test_serialization_roundtrip(tmp_path) -> None:
    parameters = sample_constant_parameters(seed=7)
    path = save_entropy_closure_v2(tmp_path / "params.npz", parameters)
    loaded = load_entropy_closure_v2(path)
    assert loaded.fitted_float_count == parameters.fitted_float_count
    np.testing.assert_allclose(loaded.feature_center, parameters.feature_center)
    np.testing.assert_allclose(loaded.opacity_coefficients, parameters.opacity_coefficients)
    assert loaded.corpus_sha256 == parameters.corpus_sha256
    assert loaded.model_spec_sha256 == parameters.model_spec_sha256


def test_wrong_format_rejected(tmp_path) -> None:
    path = tmp_path / "wrong.npz"
    np.savez_compressed(path, format=np.asarray("something_else"))
    with pytest.raises(ValueError):
        load_entropy_closure_v2(path)


def test_schwarzschild_no_crossing() -> None:
    log_p = np.linspace(0.0, 5.0, 80)
    grad_rad = np.full(80, 0.05)
    gamma_ad = np.full(80, 0.3)
    enter, exit_layer = schwarzschild_crossings(log_p, grad_rad, gamma_ad)
    assert enter == -1
    assert exit_layer == 80


def test_schwarzschild_dual_crossing() -> None:
    log_p = np.linspace(0.0, 5.0, 80)
    grad_rad = np.full(80, 0.05)
    grad_rad[40:50] = 0.5  # convective block
    gamma_ad = np.full(80, 0.3)
    enter, exit_layer = schwarzschild_crossings(log_p, grad_rad, gamma_ad)
    assert enter == 40
    assert exit_layer >= 50


def test_dual_crossing_gradient_shapes() -> None:
    log_p = np.linspace(0.0, 5.0, 80)
    grad_rad = np.full(80, 0.2)
    gamma_ad = np.full(80, 0.3)
    gradient, w_enter, w_exit = dual_crossing_gradient(
        log_p, 2.0, 4.0, grad_rad, gamma_ad, 0.1, -0.2
    )
    assert gradient.shape == (80,)
    assert w_enter.shape == (80,)
    assert w_exit.shape == (80,)
    assert np.all(np.isfinite(gradient))


def test_integrate_gradient_recovers_constant() -> None:
    log_p = np.linspace(0.0, 5.0, 80)
    gradient = np.full(80, 0.0)
    temperature = integrate_gradient(log_p, gradient, np.log(6000.0))
    np.testing.assert_allclose(temperature, 6000.0, rtol=1.0e-9)


def test_xr_minmax_ranges() -> None:
    assert AX_MIN == -0.50
    assert AX_MAX == 0.50
    assert AE_MAX == 0.50


def test_predict_finite_monotone() -> None:
    parameters = _minimal_parameters()
    labels = np.array([[7000.0, 4.0, 0.0, 0.0, 2.0]])
    mass, temperature, log_kappa, diagnostics = predict_compact_reduced_state(
        labels, TAU, parameters
    )
    assert np.isfinite(mass).all()
    assert np.isfinite(temperature).all()
    assert np.isfinite(log_kappa).all()
    assert np.all(np.diff(mass.flatten()) > 0.0)
    assert "enter_layer" in diagnostics
    assert "exit_layer" in diagnostics


def test_predict_single_equals_batch() -> None:
    parameters = _minimal_parameters()
    labels = np.array([[7000.0, 4.0, 0.0, 0.0, 2.0], [7500.0, 4.5, 0.2, 0.0, 2.0]])
    mass, temperature, log_kappa, diagnostics = predict_compact_reduced_state(
        labels, TAU, parameters
    )
    single_mass, single_temperature, single_log_kappa, single_diag = (
        predict_compact_reduced_state(labels[0], TAU, parameters)
    )
    np.testing.assert_allclose(single_mass, mass[0], rtol=1.0e-9)
    np.testing.assert_allclose(single_temperature, temperature[0], rtol=1.0e-9)
    np.testing.assert_allclose(single_log_kappa, log_kappa[0], rtol=1.0e-9)
    assert single_diag["enter_layer"].shape == ()


def test_label_normalization_requires_five_features() -> None:
    parameters = _minimal_parameters()
    with pytest.raises(ValueError):
        predict_compact_reduced_state(np.zeros((1, 4)), TAU, parameters)


def test_chebyshev_constant_mode() -> None:
    basis = chebyshev_basis(np.linspace(-2.0, 2.0, 10), 4)
    np.testing.assert_allclose(basis[:, 0], 1.0)
    assert basis.shape == (10, 4)


def test_model_spec_hash_stable() -> None:
    assert len(model_spec_sha256()) == 64
    assert model_spec_sha256() == model_spec_sha256()


def test_no_torch_import_in_prediction_path() -> None:
    import sys

    assert "torch" not in sys.modules
    _minimal_parameters()
    predict_compact_reduced_state(
        np.array([[7000.0, 4.0, 0.0, 0.0, 2.0]]), TAU, sample_constant_parameters()
    )
    assert "torch" not in sys.modules
