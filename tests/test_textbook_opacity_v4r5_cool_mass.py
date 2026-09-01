"""Unit tests for the v4r5 cool-mass decomposition helpers."""

from __future__ import annotations

import numpy as np

from experiments.analytic_initializer.profile_closure import integrate_mass_from_opacity
from experiments.analytic_initializer.run_textbook_opacity_v4r5_cool_mass_decomposition import (
    blend_log_opacity_by_temperature,
    cool_gate_mask,
    crossing_increment_mask,
    decide_cool_mass_decomposition,
    decide_v4r6_license,
    first_layer_at_or_above,
    integrate_mass_from_start_layer,
    local_increment_residual,
    log_mass_residual,
    oracle_boundary_column_mass,
    wholly_in_domain_increment_mask,
)


def _surface_error_profiles() -> dict[str, np.ndarray]:
    n_stars, n_layers = 6, 40
    tau = np.logspace(-6, 2, n_layers)
    teff = np.full(n_stars, 4800.0)
    temperature = np.empty((n_stars, n_layers))
    temperature[:, :8] = 3500.0
    temperature[:, 8:] = 4500.0
    stored_kappa = np.full((n_stars, n_layers), 1.0)
    predicted_kappa = stored_kappa.copy()
    predicted_kappa[:, :8] = 0.2
    stored_mass = integrate_mass_from_opacity(tau, np.log10(stored_kappa))
    predicted_mass = integrate_mass_from_opacity(tau, np.log10(predicted_kappa))
    return {
        "tau": tau,
        "teff": teff,
        "temperature": temperature,
        "stored_kappa": stored_kappa,
        "predicted_kappa": predicted_kappa,
        "stored_mass": stored_mass,
        "predicted_mass": predicted_mass,
    }


def test_first_layer_at_or_above_returns_n_layers_when_missing():
    temperature = np.array([[3000.0, 3500.0], [4100.0, 5000.0]])
    start = first_layer_at_or_above(temperature, 4000.0)
    np.testing.assert_array_equal(start, [2, 0])


def test_start_layer_integral_matches_surface_integral_when_start_is_zero():
    tau = np.logspace(-6, 1, 24)
    log_kappa = np.log10(np.linspace(0.2, 2.0, 24)[None, :] * np.array([[1.0], [1.3]]))
    start = np.zeros(2, dtype=np.int64)
    restarted = integrate_mass_from_start_layer(tau, log_kappa, start)
    surface = integrate_mass_from_opacity(tau, log_kappa)
    np.testing.assert_allclose(restarted, surface, rtol=1.0e-12, atol=1.0e-15)


def test_hybrid_log_opacity_keeps_inner_prediction_and_outer_stored():
    temperature = np.array([[3000.0, 4500.0]])
    inner = np.array([[1.0, 2.0]])
    outer = np.array([[8.0, 9.0]])
    blended = blend_log_opacity_by_temperature(inner, outer, temperature, 4000.0)
    np.testing.assert_allclose(blended, [[8.0, 2.0]])


def test_oracle_boundary_cancels_a_constant_outer_mass_offset():
    profiles = _surface_error_profiles()
    start = first_layer_at_or_above(profiles["temperature"], 4000.0)
    np.testing.assert_array_equal(start, np.full(6, 8))
    oracle = oracle_boundary_column_mass(
        profiles["predicted_mass"], profiles["stored_mass"], start
    )
    gate = cool_gate_mask(profiles["teff"], profiles["temperature"], 4000.0)
    residual = log_mass_residual(oracle, profiles["stored_mass"])
    np.testing.assert_allclose(residual[gate], 0.0, atol=1.0e-12)


def test_in_domain_increments_ignore_a_pure_outer_opacity_error():
    profiles = _surface_error_profiles()
    increment = local_increment_residual(
        profiles["predicted_mass"], profiles["stored_mass"]
    )
    in_domain = wholly_in_domain_increment_mask(profiles["temperature"], 4000.0)
    crossing = crossing_increment_mask(profiles["temperature"], 4000.0)
    np.testing.assert_allclose(increment[in_domain], 0.0, atol=1.0e-12)
    assert np.all(increment[crossing] > 0.0)


def test_surface_error_profiles_yield_surface_integral_dominated_verdict():
    profiles = _surface_error_profiles()
    tau = profiles["tau"]
    temperature = profiles["temperature"]
    teff = profiles["teff"]
    log_pred = np.log10(profiles["predicted_kappa"])
    log_stored = np.log10(profiles["stored_kappa"])
    hybrid = integrate_mass_from_opacity(
        tau,
        blend_log_opacity_by_temperature(log_pred, log_stored, temperature, 4000.0),
    )
    truth = integrate_mass_from_opacity(tau, log_stored)
    start = first_layer_at_or_above(temperature, 4000.0)
    oracle = oracle_boundary_column_mass(
        profiles["predicted_mass"], profiles["stored_mass"], start
    )
    gate = cool_gate_mask(teff, temperature, 4000.0)
    in_domain = wholly_in_domain_increment_mask(temperature, 4000.0)
    increment = local_increment_residual(
        profiles["predicted_mass"], profiles["stored_mass"]
    )
    surface_p95 = float(
        np.percentile(
            np.abs(log_mass_residual(profiles["predicted_mass"], profiles["stored_mass"])[gate]),
            95.0,
        )
    )
    hybrid_p95 = float(
        np.percentile(np.abs(log_mass_residual(hybrid, profiles["stored_mass"])[gate]), 95.0)
    )
    oracle_p95 = float(
        np.percentile(np.abs(log_mass_residual(oracle, profiles["stored_mass"])[gate]), 95.0)
    )
    increment_p95 = float(np.percentile(np.abs(increment[in_domain]), 95.0))
    truth_p95 = float(
        np.percentile(np.abs(log_mass_residual(truth, profiles["stored_mass"])[gate]), 95.0)
    )
    decision = decide_cool_mass_decomposition(
        surface_p95_dex=surface_p95,
        hybrid_p95_dex=hybrid_p95,
        oracle_p95_dex=oracle_p95,
        in_domain_increment_p95_dex=increment_p95,
        truth_kappa_p95_dex=truth_p95,
        expected_surface_p95_dex=None,
    )
    assert surface_p95 > 0.20
    assert decision["verdict"] == "SURFACE_INTEGRAL_DOMINATED"
    assert decision["hybrid_pass"]
    assert decision["oracle_pass"]
    assert decision["increment_pass"]


def test_all_three_bits_failing_is_in_domain_dominated():
    decision = decide_cool_mass_decomposition(
        surface_p95_dex=0.2375,
        hybrid_p95_dex=0.22,
        oracle_p95_dex=0.21,
        in_domain_increment_p95_dex=0.25,
        truth_kappa_p95_dex=0.006,
    )
    assert decision["verdict"] == "IN_DOMAIN_COOL_OPACITY_DOMINATED"


def test_disagreeing_bits_are_mixed():
    decision = decide_cool_mass_decomposition(
        surface_p95_dex=0.2375,
        hybrid_p95_dex=0.18,
        oracle_p95_dex=0.21,
        in_domain_increment_p95_dex=0.05,
        truth_kappa_p95_dex=0.006,
    )
    assert decision["verdict"] == "MIXED"
    assert abs(decision["explained_fraction_of_p95_excess"] - (0.2375 - 0.18) / 0.0375) < 1.0e-12


def test_failed_reproduce_and_truth_sanity_are_inconclusive():
    missed = decide_cool_mass_decomposition(
        surface_p95_dex=0.30,
        hybrid_p95_dex=0.10,
        oracle_p95_dex=0.10,
        in_domain_increment_p95_dex=0.10,
        truth_kappa_p95_dex=0.006,
    )
    assert missed["verdict"] == "INCONCLUSIVE"
    assert missed["inconclusive_reason"] == "did_not_reproduce_v4r5_cool_mass_p95"

    broken_integral = decide_cool_mass_decomposition(
        surface_p95_dex=0.2375,
        hybrid_p95_dex=0.10,
        oracle_p95_dex=0.10,
        in_domain_increment_p95_dex=0.10,
        truth_kappa_p95_dex=0.08,
    )
    assert broken_integral["verdict"] == "INCONCLUSIVE"
    assert (
        broken_integral["inconclusive_reason"]
        == "stored_kappa_integral_does_not_recover_mass"
    )


def test_v4r6_license_requires_surface_fraction_and_production_miss():
    licensed = decide_v4r6_license(
        verdict="SURFACE_INTEGRAL_DOMINATED",
        explained_fraction=1.1,
        v4r5_minus_production_signed_median_dex=-0.12,
        v4r5_minus_stored_signed_median_dex=-0.18,
        dominant_production_flag_name="H_bf_ff",
    )
    assert licensed["licensed"] is True
    assert "Bell & Berrington" in licensed["named_construction"]

    lines = decide_v4r6_license(
        verdict="SURFACE_INTEGRAL_DOMINATED",
        explained_fraction=1.0,
        v4r5_minus_production_signed_median_dex=-0.01,
        v4r5_minus_stored_signed_median_dex=-0.20,
        dominant_production_flag_name="H_bf_ff",
    )
    assert lines["licensed"] is False
    assert lines["reason"] == "LINES_DOMINATE_OUTER"

    in_domain = decide_v4r6_license(
        verdict="IN_DOMAIN_COOL_OPACITY_DOMINATED",
        explained_fraction=0.2,
        v4r5_minus_production_signed_median_dex=-0.20,
        v4r5_minus_stored_signed_median_dex=-0.20,
        dominant_production_flag_name="H_bf_ff",
    )
    assert in_domain["licensed"] is False
    assert in_domain["reason"] == "verdict_is_not_surface_or_mixed"
