"""Tests for the named-constant textbook opacity/ODE diagnostic."""

import numpy as np

from experiments.analytic_initializer.textbook_opacity import (
    DEFAULT_TEXTBOOK_CONSTANTS,
    V4R1_FORMAL_TEMPERATURE_FLOOR_K,
    V4R2_FORMAL_TEMPERATURE_FLOOR_K,
    V4R3_FORMAL_TEMPERATURE_FLOOR_K,
    V4R4_FORMAL_TEMPERATURE_FLOOR_K,
    V4R5_FORMAL_TEMPERATURE_FLOOR_K,
    V4R6_FORMAL_TEMPERATURE_FLOOR_K,
    V4R6_PER_N_TEMPERATURE_CEILING_K,
    WINDOW_NAMES,
    build_textbook_reduced_state,
    build_textbook_reduced_state_v3,
    build_textbook_reduced_state_v4,
    frequency_window_edges_hz,
    hminus_freefree_rosseland_opacity,
    integrate_hydrostatic_opacity_ode,
    _textbook_opacity_node_components_from_state,
    rosseland_frequency_nodes,
    rosseland_window_weights,
    saha_aware_adiabatic_gradient,
    saha_electron_diagnostics,
    saha_electron_diagnostics_v4r1,
    saha_electron_diagnostics_v4r3,
    textbook_opacity_components,
    textbook_opacity_node_components,
    textbook_opacity_node_components_v4r1,
    textbook_opacity_node_components_v4r3,
    textbook_opacity_window_components,
    textbook_rosseland_opacity_v4,
    textbook_rosseland_opacity_v4r1,
    textbook_rosseland_opacity_v4r3,
    textbook_opacity_node_components_v4r4,
    textbook_rosseland_opacity_v4r4,
    textbook_opacity_node_components_v4r5,
    textbook_rosseland_opacity_v4r5,
    textbook_opacity_node_components_v4r6,
    textbook_rosseland_opacity_v4r6,
    build_textbook_reduced_state_v4r3,
    predict_textbook_reduced_state_v4r3,
    build_textbook_reduced_state_v4r6,
    predict_textbook_reduced_state_v4r6,
)


def _inputs():
    labels = np.asarray(
        [
            [5000.0, 4.0, -1.0, 0.2, 1.0],
            [8000.0, 3.0, 0.0, 0.0, 1.5],
        ]
    )
    tau = 10.0 ** np.linspace(-6.875, 3.0, 12)
    temperature = labels[:, 0, None] * (0.75 * (tau[None, :] + 2.0 / 3.0)) ** 0.25
    pressure = 10.0 ** labels[:, 1, None] * tau[None, :] / 0.34
    return labels, tau, temperature, pressure


def test_textbook_opacity_components_are_positive_and_finite():
    labels, _, temperature, pressure = _inputs()
    components = textbook_opacity_components(labels, temperature, pressure)
    assert set(components) == {
        "hminus_boundfree",
        "hminus_freefree",
        "hydrogen_balmer_paschen_boundfree",
        "kramers_freefree",
        "electron_scattering",
        "total",
    }
    for value in components.values():
        assert np.all(np.isfinite(value))
        assert np.all(value >= 0.0)
    assert np.all(components["total"] > 0.0)
    np.testing.assert_allclose(
        components["total"],
        sum(components[name] for name in components if name != "total"),
    )


def test_saha_diagnostics_and_ad_gradient_stay_bounded():
    labels, _, temperature, pressure = _inputs()
    state = saha_electron_diagnostics(labels, temperature, pressure)
    assert np.all(state["electron_density_cm3"] > 0.0)
    assert np.all((state["hydrogen_ionized_fraction"] >= 0.0))
    assert np.all(state["hydrogen_ionized_fraction"] <= 1.0)
    gradient = saha_aware_adiabatic_gradient(labels, temperature, pressure)
    assert np.all((gradient >= 0.1) & (gradient <= 0.4))


def test_opacity_ode_is_positive_and_monotone():
    labels, tau, temperature, _ = _inputs()
    mass = integrate_hydrostatic_opacity_ode(
        labels, tau, temperature, substeps_per_layer=2
    )
    assert mass.shape == temperature.shape
    assert np.all(np.isfinite(mass))
    assert np.all(mass > 0.0)
    assert np.all(np.diff(mass, axis=1) > 0.0)


def test_convection_build_returns_complete_diagnostic_state():
    labels, tau, _, _ = _inputs()
    mass, temperature, diagnostics = build_textbook_reduced_state(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    assert mass.shape == temperature.shape == (labels.shape[0], tau.size)
    assert np.all(np.isfinite(mass)) and np.all(mass > 0.0)
    assert np.all(np.isfinite(temperature)) and np.all(temperature > 0.0)
    assert diagnostics["rosseland_opacity"].shape == mass.shape
    assert diagnostics["convective_mask"].dtype == bool


def test_gray_hminus_rosseland_reference_is_reproduced():
    labels = np.asarray([[5777.0, 4.44, 0.0, 0.0, 1.0]])
    temperature = np.asarray([[5000.0]])
    pressure = np.asarray([[1.0e5]])
    opacity = hminus_freefree_rosseland_opacity(labels, temperature, pressure)[0, 0]
    rho = (
        DEFAULT_TEXTBOOK_CONSTANTS.neutral_mean_molecular_weight
        * DEFAULT_TEXTBOOK_CONSTANTS.hydrogen_mass_g
        * pressure[0, 0]
        / (DEFAULT_TEXTBOOK_CONSTANTS.boltzmann_erg_per_K * temperature[0, 0])
    )
    reference = 2.5e-31 * (0.0134 / 0.02) * np.sqrt(rho) * 5000.0**9
    assert np.isclose(opacity, reference, rtol=1.0e-12)


def test_solar_photosphere_total_opacity_is_in_the_literature_window():
    labels = np.asarray([[5777.0, 4.44, 0.0, 0.0, 1.0]])
    temperature = np.asarray([[5777.0]])
    pressure = np.asarray([[1.0e5]])
    total = textbook_opacity_components(labels, temperature, pressure)["total"][0, 0]
    # A solar photosphere Rosseland opacity of roughly 0.75 cm2 g-1 is the
    # single-point physical sanity check; the acceptance band is +/-0.2 dex.
    assert 0.75 / 10.0**0.2 <= total <= 0.75 * 10.0**0.2


def test_solar_component_values_stay_within_the_preregistered_dex_tolerance():
    labels = np.asarray([[5777.0, 4.44, 0.0, 0.0, 1.0]])
    temperature = np.asarray([[5777.0]])
    pressure = np.asarray([[1.0e5]])
    components = textbook_opacity_components(labels, temperature, pressure)
    # Fixed solar-point reference values for the named branches.  These are
    # intentionally broad (+/-0.2 dex): the test catches unit/coefficient
    # mistakes without turning the compact seed into a fitted opacity table.
    references = {
        "hminus_boundfree": 8.5e-2,
        "hminus_freefree": 6.3e-1,
        "hydrogen_balmer_paschen_boundfree": 3.1e-2,
        "kramers_freefree": 9.6e-2,
        "electron_scattering": 4.4e-5,
    }
    for name, reference in references.items():
        error_dex = abs(np.log10(components[name][0, 0] / reference))
        assert error_dex <= 0.2, (name, components[name][0, 0], reference)


def test_v3_frequency_edges_and_rosseland_weights_are_fixed_and_normalized():
    edges = frequency_window_edges_hz()
    assert len(edges) == len(WINDOW_NAMES) + 1
    assert edges[0] == 0.0 and np.isinf(edges[-1])
    assert np.all(np.diff(edges[1:-1]) > 0.0)
    weights = rosseland_window_weights(np.asarray([[3200.0, 5777.0, 12000.0]]))
    assert weights.shape == (1, 3, len(WINDOW_NAMES))
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, rtol=0.0, atol=2.0e-14)


def test_v3_window_components_are_monotone_and_harmonically_combined():
    labels, _, temperature, pressure = _inputs()
    components = textbook_opacity_window_components(labels, temperature, pressure)
    expected = {
        "hminus_boundfree",
        "hminus_freefree",
        "hydrogen_paschen_boundfree",
        "hydrogen_balmer_boundfree",
        "kramers_freefree",
        "electron_scattering",
        "hydrogen_rayleigh_scattering",
        "window_opacity",
        "window_weights",
        "total",
    }
    assert set(components) == expected
    windows = components["window_opacity"]
    weights = components["window_weights"]
    assert windows.shape == temperature.shape + (len(WINDOW_NAMES),)
    assert np.all(np.diff(windows, axis=-1) >= 0.0)
    arithmetic = np.sum(weights * windows, axis=-1)
    assert np.all(components["total"] <= arithmetic * (1.0 + 1.0e-13))
    assert np.all(np.isfinite(components["total"]))
    assert np.all(components["total"] > 0.0)


def test_v3_solar_opacity_remains_in_the_reference_window():
    labels = np.asarray([[5777.0, 4.44, 0.0, 0.0, 1.0]])
    temperature = np.asarray([[5777.0]])
    pressure = np.asarray([[1.0e5]])
    total = textbook_opacity_window_components(labels, temperature, pressure)["total"][0, 0]
    assert 0.75 / 10.0**0.2 <= total <= 0.75 * 10.0**0.2


def test_v3_opacity_ode_and_convection_seed_are_positive_and_monotone():
    labels, tau, _, _ = _inputs()
    mass, temperature, diagnostics = build_textbook_reduced_state_v3(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    assert mass.shape == temperature.shape == (labels.shape[0], tau.size)
    assert np.all(np.isfinite(mass)) and np.all(mass > 0.0)
    assert np.all(np.diff(mass, axis=1) > 0.0)
    assert np.all(np.isfinite(temperature)) and np.all(temperature > 0.0)
    assert diagnostics["rosseland_opacity"].shape == mass.shape


def test_v4_nodes_are_normalized_and_have_real_frequency_dependence():
    labels, _, temperature, pressure = _inputs()
    frequencies, weights, u = rosseland_frequency_nodes(temperature)
    expected_shape = temperature.shape + (len(WINDOW_NAMES), 32)
    assert frequencies.shape == weights.shape == u.shape == expected_shape
    assert np.all(np.isfinite(frequencies)) and np.all(frequencies > 0.0)
    assert np.all(np.isfinite(weights)) and np.all(weights > 0.0)
    np.testing.assert_allclose(
        weights.sum(axis=(-2, -1)),
        np.ones(temperature.shape),
        rtol=0.0,
        atol=3.0e-14,
    )
    components = textbook_opacity_node_components(labels, temperature, pressure)
    assert np.ptp(components["hminus_boundfree"], axis=-1).max() > 0.0
    assert np.ptp(components["hydrogen_freefree"], axis=-1).max() > 0.0
    np.testing.assert_allclose(
        textbook_rosseland_opacity_v4(labels, temperature, pressure),
        1.0
        / np.sum(
            components["node_weights"] / components["total"], axis=(-2, -1)
        ),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_v4_opacity_ode_and_convection_seed_are_positive_and_monotone():
    labels, tau, _, _ = _inputs()
    mass, temperature, diagnostics = build_textbook_reduced_state_v4(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    assert mass.shape == temperature.shape == (labels.shape[0], tau.size)
    assert np.all(np.isfinite(mass)) and np.all(mass > 0.0)
    assert np.all(np.diff(mass, axis=1) > 0.0)
    assert np.all(np.isfinite(temperature)) and np.all(temperature > 0.0)
    assert diagnostics["rosseland_opacity"].shape == mass.shape


def test_v4_historical_regression_values_are_unchanged():
    labels, _, temperature, pressure = _inputs()
    expected = np.asarray(
        [
            [
                6.8007433459175569e-04,
                2.5326459455216955e-04,
                1.0205758377127645e-04,
                5.4231884901421422e-05,
                5.5409550547615463e-05,
                1.2137056628675234e-04,
                5.2147439322051384e-04,
                3.6698064618388465e-03,
                1.9866967944619704e-01,
                4.8449862823560707e01,
                2.0008522400284392e03,
                4.8116776480041572e03,
            ],
            [
                2.9313967069902408e-01,
                2.9110812760614413e-01,
                2.7515260565372435e-01,
                2.2002070156631673e-01,
                1.5870861793871721e-01,
                1.3035111703686605e-01,
                1.6247788163296442e-01,
                6.5410215809389238e-01,
                3.1807695755333125e01,
                2.8520546399173696e02,
                1.2487043297136263e02,
                4.5945863382274396e01,
            ],
        ]
    )
    np.testing.assert_allclose(
        textbook_rosseland_opacity_v4(labels, temperature, pressure),
        expected,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_v4r1_saha_closure_includes_all_donors_and_balances_charge():
    labels, _, temperature, pressure = _inputs()
    state = saha_electron_diagnostics_v4r1(labels, temperature, pressure)
    for donor in ("Na", "K", "Ca", "Mg", "Fe", "Al", "Si"):
        fraction = state[f"donor_ionized_fraction_{donor}"]
        assert np.all((fraction >= 0.0) & (fraction <= 1.0))
    assert np.max(np.abs(state["charge_balance_relative_residual"])) < 1.0e-10


def test_v4r1_donor_closure_raises_cool_solar_electron_density():
    labels = np.asarray([[4500.0, 4.5, 0.0, 0.0, 1.0]])
    temperature = np.asarray([[3500.0, 4000.0]])
    pressure = np.asarray([[1.0e4, 3.0e4]])
    historical = saha_electron_diagnostics(
        labels, temperature, pressure
    )["electron_density_cm3"]
    repaired = saha_electron_diagnostics_v4r1(
        labels, temperature, pressure
    )["electron_density_cm3"]
    assert np.all(repaired > historical)


def test_v4r1_john_freefree_has_no_second_stimulated_emission_factor():
    labels = np.asarray([[5000.0, 4.0, 0.0, 0.0, 1.0]])
    temperature = np.asarray([[5000.0]])
    pressure = np.asarray([[1.0e5]])
    state = saha_electron_diagnostics_v4r1(labels, temperature, pressure)
    historical_convention = _textbook_opacity_node_components_from_state(
        labels,
        temperature,
        pressure,
        state,
        apply_stimulated_emission_to_hminus_freefree=True,
    )
    repaired_convention = _textbook_opacity_node_components_from_state(
        labels,
        temperature,
        pressure,
        state,
        apply_stimulated_emission_to_hminus_freefree=False,
    )
    stimulated = -np.expm1(-historical_convention["frequency_nodes_u"])
    active = repaired_convention["hminus_freefree"] > 0.0
    np.testing.assert_allclose(
        historical_convention["hminus_freefree"][active],
        repaired_convention["hminus_freefree"][active] * stimulated[active],
        rtol=2.0e-15,
        atol=0.0,
    )


def test_v4r1_nodes_and_rosseland_mean_are_positive_and_finite():
    labels, _, temperature, pressure = _inputs()
    components = textbook_opacity_node_components_v4r1(
        labels, temperature, pressure
    )
    opacity = textbook_rosseland_opacity_v4r1(
        labels, temperature, pressure
    )
    assert np.all(np.isfinite(components["total"]))
    assert np.all(components["total"] > 0.0)
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity > 0.0)


def test_v4r1_molecule_ablation_verdict_is_mutually_exclusive():
    from experiments.analytic_initializer.textbook_opacity_v4r1_molecule_verdict import (
        decide_molecule_ablation,
    )

    molecular = decide_molecule_ablation(
        {
            "molecular_effect": {
                "signed_median_dex": 0.12,
                "p50_abs_dex": 0.12,
                "p95_abs_dex": 0.20,
            },
            "v4r1_minus_atomic": {
                "signed_median_dex": -0.02,
                "p50_abs_dex": 0.04,
                "p95_abs_dex": 0.09,
            },
            "v4r1_minus_molecular": {
                "signed_median_dex": -0.14,
                "p50_abs_dex": 0.14,
                "p95_abs_dex": 0.28,
            },
        }
    )
    assert molecular["verdict"] == "MOLECULAR_CONTINUUM_DOMINATES"
    assert molecular["atomic_aligned"] is True
    assert molecular["atomic_ir_remains"] is False

    atomic = decide_molecule_ablation(
        {
            "molecular_effect": {
                "signed_median_dex": 0.01,
                "p50_abs_dex": 0.02,
                "p95_abs_dex": 0.04,
            },
            "v4r1_minus_atomic": {
                "signed_median_dex": -0.08,
                "p50_abs_dex": 0.12,
                "p95_abs_dex": 0.36,
            },
            "v4r1_minus_molecular": {
                "signed_median_dex": -0.09,
                "p50_abs_dex": 0.13,
                "p95_abs_dex": 0.37,
            },
        }
    )
    assert atomic["verdict"] == "ATOMIC_IR_REMAINS"
    assert atomic["atomic_aligned"] is False
    assert atomic["atomic_ir_remains"] is True

    mixed = decide_molecule_ablation(
        {
            "molecular_effect": {
                "signed_median_dex": 0.08,
                "p50_abs_dex": 0.08,
                "p95_abs_dex": 0.12,
            },
            "v4r1_minus_atomic": {
                "signed_median_dex": -0.04,
                "p50_abs_dex": 0.06,
                "p95_abs_dex": 0.25,
            },
            "v4r1_minus_molecular": {
                "signed_median_dex": -0.12,
                "p50_abs_dex": 0.12,
                "p95_abs_dex": 0.32,
            },
        }
    )
    assert mixed["verdict"] == "MIXED_MOLECULAR_PLUS_ATOMIC_IR"


def test_v4r2_domain_floor_is_a_declared_raise_not_a_fit():
    assert V4R1_FORMAL_TEMPERATURE_FLOOR_K == 3200.0
    assert V4R2_FORMAL_TEMPERATURE_FLOOR_K == 4000.0
    assert V4R3_FORMAL_TEMPERATURE_FLOOR_K == V4R2_FORMAL_TEMPERATURE_FLOOR_K
    assert V4R2_FORMAL_TEMPERATURE_FLOOR_K > V4R1_FORMAL_TEMPERATURE_FLOOR_K


def test_v4r3_density_drops_when_hydrogen_ionizes():
    cool_labels = np.asarray([[4500.0, 4.5, 0.0, 0.0, 1.0]])
    cool_temperature = np.asarray([[4000.0]])
    cool_pressure = np.asarray([[1.0e5]])
    hot_labels = np.asarray([[20000.0, 4.0, 0.0, 0.0, 1.0]])
    hot_temperature = np.asarray([[20000.0]])
    hot_pressure = np.asarray([[1.0e5]])
    cool = saha_electron_diagnostics_v4r3(cool_labels, cool_temperature, cool_pressure)
    hot = saha_electron_diagnostics_v4r3(hot_labels, hot_temperature, hot_pressure)
    historical_hot = saha_electron_diagnostics_v4r1(
        hot_labels, hot_temperature, hot_pressure
    )
    assert cool["mean_molecular_weight"][0, 0] > 1.2
    assert hot["mean_molecular_weight"][0, 0] < 0.8
    assert hot["rho_g_cm3"][0, 0] < 0.7 * historical_hot["rho_g_cm3"][0, 0]
    assert np.max(np.abs(cool["charge_balance_relative_residual"])) < 1.0e-10
    assert np.max(np.abs(hot["charge_balance_relative_residual"])) < 1.0e-10


def test_v4r3_adds_nonnegative_h2plus_and_heminus_and_stays_finite():
    labels, _, temperature, pressure = _inputs()
    components = textbook_opacity_node_components_v4r3(labels, temperature, pressure)
    assert np.all(components["h2plus"] >= 0.0)
    assert np.all(components["heminus"] >= 0.0)
    opacity = textbook_rosseland_opacity_v4r3(labels, temperature, pressure)
    historical = textbook_rosseland_opacity_v4r1(labels, temperature, pressure)
    assert np.all(np.isfinite(opacity) & (opacity > 0.0))
    assert np.all(opacity >= historical * 0.1)


def test_v4r3_solver_seed_is_finite_monotone_and_unfitted():
    labels, tau, _, _ = _inputs()
    mass, temperature, diagnostics = build_textbook_reduced_state_v4r3(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    predicted_mass, predicted_temperature, log_opacity = (
        predict_textbook_reduced_state_v4r3(
            labels, tau, include_convection=True, substeps_per_layer=2
        )
    )
    assert mass.shape == temperature.shape == (labels.shape[0], tau.size)
    assert np.all(np.isfinite(mass) & (mass > 0.0))
    assert np.all(np.diff(mass, axis=1) > 0.0)
    assert np.all(np.isfinite(temperature) & (temperature > 0.0))
    np.testing.assert_allclose(predicted_mass, mass)
    np.testing.assert_allclose(predicted_temperature, temperature)
    np.testing.assert_allclose(
        10.0 ** log_opacity, diagnostics["rosseland_opacity"], rtol=0.0, atol=1.0e-12
    )


def test_v4r6_solver_seed_is_finite_monotone_and_unfitted():
    labels, tau, _, _ = _inputs()
    mass, temperature, diagnostics = build_textbook_reduced_state_v4r6(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    predicted_mass, predicted_temperature, log_opacity = (
        predict_textbook_reduced_state_v4r6(
            labels, tau, include_convection=True, substeps_per_layer=2
        )
    )
    assert mass.shape == temperature.shape == (labels.shape[0], tau.size)
    assert np.all(np.isfinite(mass) & (mass > 0.0))
    assert np.all(np.diff(mass, axis=1) > 0.0)
    assert np.all(np.isfinite(temperature) & (temperature > 0.0))
    np.testing.assert_allclose(predicted_mass, mass)
    np.testing.assert_allclose(predicted_temperature, temperature)
    np.testing.assert_allclose(
        10.0 ** log_opacity, diagnostics["rosseland_opacity"], rtol=0.0, atol=1.0e-12
    )


def test_v4r6_seed_differs_from_v4r3_only_through_opacity():
    labels, tau, _, _ = _inputs()
    mass_v4r3, temperature_v4r3, _ = predict_textbook_reduced_state_v4r3(
        labels, tau, include_convection=False, substeps_per_layer=2
    )
    mass_v4r6, temperature_v4r6, _ = predict_textbook_reduced_state_v4r6(
        labels, tau, include_convection=False, substeps_per_layer=2
    )
    np.testing.assert_allclose(temperature_v4r6, temperature_v4r3, rtol=0.0, atol=0.0)
    assert not np.allclose(mass_v4r6, mass_v4r3, rtol=1.0e-12, atol=0.0)


def test_batched_opacity_ode_matches_one_star_at_a_time():
    labels, tau, temperature, _ = _inputs()
    batched = integrate_hydrostatic_opacity_ode(
        labels,
        tau,
        temperature,
        substeps_per_layer=2,
        opacity_function=textbook_rosseland_opacity_v4r3,
    )
    sequential = np.vstack(
        [
            integrate_hydrostatic_opacity_ode(
                labels[index : index + 1],
                tau,
                temperature[index : index + 1],
                substeps_per_layer=2,
                opacity_function=textbook_rosseland_opacity_v4r3,
            )
            for index in range(labels.shape[0])
        ]
    )
    np.testing.assert_allclose(batched, sequential, rtol=1.0e-12, atol=0.0)


def test_v4r4_helium_stages_sum_to_total_helium_and_stay_out_of_charge_balance():
    cool_labels = np.asarray([[4500.0, 4.5, 0.0, 0.0, 1.0]])
    cool_temperature = np.asarray([[4500.0]])
    cool_pressure = np.asarray([[1.0e5]])
    hot_labels = np.asarray([[20000.0, 4.0, 0.0, 0.0, 1.0]])
    hot_temperature = np.asarray([[20000.0]])
    hot_pressure = np.asarray([[1.0e5]])
    helium_per_hydrogen = (
        DEFAULT_TEXTBOOK_CONSTANTS.helium_mass_fraction
        / (4.0 * DEFAULT_TEXTBOOK_CONSTANTS.hydrogen_mass_fraction)
    )
    for labels, temperature, pressure, cool_photosphere in (
        (cool_labels, cool_temperature, cool_pressure, True),
        (hot_labels, hot_temperature, hot_pressure, False),
    ):
        state = saha_electron_diagnostics_v4r3(labels, temperature, pressure)
        components = textbook_opacity_node_components_v4r4(
            labels, temperature, pressure
        )
        n_i = components["helium_i_density_cm3"]
        n_ii = components["helium_ii_density_cm3"]
        n_iii = components["helium_iii_density_cm3"]
        n_he = state["hydrogen_number_density_cm3"] * helium_per_hydrogen
        np.testing.assert_allclose(n_i + n_ii + n_iii, n_he, rtol=1.0e-10, atol=0.0)
        n_i_fraction = n_i / np.maximum(n_he, 1.0e-300)
        if cool_photosphere:
            assert np.all(n_i_fraction > 0.99)
        else:
            assert np.all(n_i_fraction < 0.5)
        np.testing.assert_allclose(
            components["electron_density_cm3"],
            state["electron_density_cm3"],
            rtol=0.0,
            atol=0.0,
        )


def test_v4r4_heii_is_negligible_in_cool_photosphere_and_raises_hot_opacity():
    cool_labels = np.asarray([[4500.0, 4.5, 0.0, 0.0, 1.0]])
    cool_temperature = np.asarray([[4500.0]])
    cool_pressure = np.asarray([[1.0e5]])
    hot_labels = np.asarray([[20000.0, 4.0, 0.0, 0.0, 1.0]])
    hot_temperature = np.asarray([[20000.0]])
    hot_pressure = np.asarray([[1.0e5]])
    # Hydrogenic He II n=1 sits at 54.418 eV.  At 20000 K the Rosseland
    # harmonic mean is still set by balmer_to_lyman, so kappa_R does not
    # move by 1.2x there.  40000 K is where the above_lyman window carries
    # enough weight for the frozen node law to raise kappa_R.
    rosseland_hot_labels = np.asarray([[40000.0, 4.0, 0.0, 0.0, 1.0]])
    rosseland_hot_temperature = np.asarray([[40000.0]])
    cool = textbook_opacity_node_components_v4r4(
        cool_labels, cool_temperature, cool_pressure
    )
    heii_cool = (
        cool["helium_ionized_boundfree"] + cool["helium_ionized_freefree"]
    )
    assert np.all(heii_cool / np.maximum(cool["total"], 1.0e-30) < 1.0e-4)
    cool_v4r4 = textbook_rosseland_opacity_v4r4(
        cool_labels, cool_temperature, cool_pressure
    )
    cool_v4r3 = textbook_rosseland_opacity_v4r3(
        cool_labels, cool_temperature, cool_pressure
    )
    ratio_cool = cool_v4r4 / cool_v4r3
    assert np.all((ratio_cool >= 0.99) & (ratio_cool <= 1.01))
    hot = textbook_opacity_node_components_v4r4(
        hot_labels, hot_temperature, hot_pressure
    )
    heii_hot = (
        hot["helium_ionized_boundfree"] + hot["helium_ionized_freefree"]
    )
    assert np.max(heii_hot / np.maximum(hot["total"], 1.0e-30)) > 0.5
    hot_v4r4 = textbook_rosseland_opacity_v4r4(
        rosseland_hot_labels, rosseland_hot_temperature, hot_pressure
    )
    hot_v4r3 = textbook_rosseland_opacity_v4r3(
        rosseland_hot_labels, rosseland_hot_temperature, hot_pressure
    )
    assert np.all(np.isfinite(hot_v4r4) & (hot_v4r4 > 0.0))
    assert np.all(np.isfinite(hot_v4r3) & (hot_v4r3 > 0.0))
    assert np.all(hot_v4r4 > 1.2 * hot_v4r3)


def test_v4r4_components_are_nonnegative_finite_and_do_not_mutate_v4r3():
    labels, _, temperature, pressure = _inputs()
    historical_before = textbook_rosseland_opacity_v4r3(
        labels, temperature, pressure
    )
    components = textbook_opacity_node_components_v4r4(
        labels, temperature, pressure
    )
    assert np.all(components["helium_ionized_boundfree"] >= 0.0)
    assert np.all(components["helium_ionized_freefree"] >= 0.0)
    assert np.all(np.isfinite(components["total"]))
    opacity = textbook_rosseland_opacity_v4r4(labels, temperature, pressure)
    assert np.all(np.isfinite(opacity) & (opacity > 0.0))
    historical_after = textbook_rosseland_opacity_v4r3(
        labels, temperature, pressure
    )
    np.testing.assert_allclose(
        historical_before, historical_after, rtol=0.0, atol=0.0
    )


def test_v4r4_domain_floor_is_unchanged():
    assert V4R4_FORMAL_TEMPERATURE_FLOOR_K == V4R3_FORMAL_TEMPERATURE_FLOOR_K
    assert V4R3_FORMAL_TEMPERATURE_FLOOR_K == 4000.0
    assert V4R4_FORMAL_TEMPERATURE_FLOOR_K == 4000.0


def test_v4r5_ground_anchor_matches_v4r3_in_cool_photosphere():
    labels = np.asarray([[8000.0, 4.0, 0.0, 0.0, 1.0]])
    temperature = np.asarray([[8000.0]])
    pressure = np.asarray([[1.0e5]])
    v4r3 = textbook_rosseland_opacity_v4r3(labels, temperature, pressure)
    v4r5 = textbook_rosseland_opacity_v4r5(labels, temperature, pressure)
    np.testing.assert_allclose(v4r5, v4r3, rtol=1.0e-3, atol=0.0)
    cool_labels = np.asarray([[4500.0, 4.5, 0.0, 0.0, 1.0]])
    cool_temperature = np.asarray([[4500.0]])
    cool_pressure = np.asarray([[1.0e5]])
    cool_v4r3 = textbook_rosseland_opacity_v4r3(
        cool_labels, cool_temperature, cool_pressure
    )
    cool_v4r5 = textbook_rosseland_opacity_v4r5(
        cool_labels, cool_temperature, cool_pressure
    )
    np.testing.assert_allclose(cool_v4r5, cool_v4r3, rtol=1.0e-6, atol=0.0)


def test_v4r5_ground_holds_all_neutrals_and_recovers_hot_lyman():
    labels = np.asarray([[40000.0, 4.0, 0.0, 0.0, 1.0]])
    temperature = np.asarray([[40000.0]])
    pressure = np.asarray([[1.0e5]])
    state = saha_electron_diagnostics_v4r3(labels, temperature, pressure)
    components = textbook_opacity_node_components_v4r5(
        labels, temperature, pressure
    )
    n_hi = state["hydrogen_neutral_density_cm3"]
    np.testing.assert_allclose(
        components["hydrogen_ground_density_cm3"],
        n_hi,
        rtol=1.0e-12,
        atol=0.0,
    )
    assert np.all(components["hydrogen_bound_level_population_sum_cm3"] > 5.0 * n_hi)
    v4r3 = textbook_rosseland_opacity_v4r3(labels, temperature, pressure)
    v4r5 = textbook_rosseland_opacity_v4r5(labels, temperature, pressure)
    assert np.all(np.isfinite(v4r5) & (v4r5 > 0.0))
    assert np.all(v4r5 > 2.0 * v4r3)


def test_v4r5_does_not_change_edge_power_or_mutate_history():
    assert DEFAULT_TEXTBOOK_CONSTANTS.hydrogen_boundfree_edge_cross_section_power == 2.0
    labels, _, temperature, pressure = _inputs()
    v4r3_before = textbook_rosseland_opacity_v4r3(labels, temperature, pressure)
    v4r4_before = textbook_rosseland_opacity_v4r4(labels, temperature, pressure)
    components = textbook_opacity_node_components_v4r5(
        labels, temperature, pressure
    )
    for name in (
        "hminus_boundfree",
        "hminus_freefree",
        "hydrogen_boundfree",
        "hydrogen_freefree",
        "electron_scattering",
        "hydrogen_rayleigh_scattering",
        "h2plus",
        "heminus",
    ):
        assert np.all(components[name] >= 0.0)
    assert np.all(np.isfinite(components["total"]))
    opacity = textbook_rosseland_opacity_v4r5(labels, temperature, pressure)
    assert np.all(np.isfinite(opacity) & (opacity > 0.0))
    np.testing.assert_allclose(
        textbook_rosseland_opacity_v4r3(labels, temperature, pressure),
        v4r3_before,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        textbook_rosseland_opacity_v4r4(labels, temperature, pressure),
        v4r4_before,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        components["electron_density_cm3"],
        saha_electron_diagnostics_v4r3(labels, temperature, pressure)[
            "electron_density_cm3"
        ],
        rtol=0.0,
        atol=0.0,
    )


def test_v4r5_domain_floor_is_unchanged():
    assert V4R5_FORMAL_TEMPERATURE_FLOOR_K == V4R3_FORMAL_TEMPERATURE_FLOOR_K
    assert V4R5_FORMAL_TEMPERATURE_FLOOR_K == 4000.0


def test_v4r6_matches_v4r5_above_per_n_ceiling():
    labels = np.asarray(
        [[20000.0, 4.0, 0.0, 0.0, 1.0], [40000.0, 4.0, 0.0, 0.0, 1.0]]
    )
    temperature = np.asarray([[20000.0], [40000.0]])
    pressure = np.asarray([[1.0e5], [1.0e5]])
    v4r5 = textbook_rosseland_opacity_v4r5(labels, temperature, pressure)
    v4r6 = textbook_rosseland_opacity_v4r6(labels, temperature, pressure)
    np.testing.assert_allclose(v4r6, v4r5, rtol=0.0, atol=0.0)


def test_v4r6_lowers_balmer_photosphere_and_keeps_n1_edge():
    labels = np.asarray([[8000.0, 4.0, 0.0, 0.0, 1.0]])
    temperature = np.asarray([[8000.0]])
    pressure = np.asarray([[1.0e5]])
    v4r5 = textbook_rosseland_opacity_v4r5(labels, temperature, pressure)
    v4r6 = textbook_rosseland_opacity_v4r6(labels, temperature, pressure)
    assert np.all(np.isfinite(v4r6) & (v4r6 > 0.0))
    assert np.all(v4r6 < 0.95 * v4r5)
    edges = DEFAULT_TEXTBOOK_CONSTANTS.hydrogen_published_threshold_cross_section_cm2
    assert edges[0] == DEFAULT_TEXTBOOK_CONSTANTS.hydrogen_ground_edge_cross_section_cm2
    assert edges[1] == 1.40e-17
    assert edges[2] == 2.16e-17
    assert DEFAULT_TEXTBOOK_CONSTANTS.hydrogen_boundfree_edge_cross_section_power == 2.0


def test_v4r6_does_not_mutate_v4r5_or_change_domain_floor():
    labels, _, temperature, pressure = _inputs()
    before = textbook_rosseland_opacity_v4r5(labels, temperature, pressure)
    components = textbook_opacity_node_components_v4r6(labels, temperature, pressure)
    for name in (
        "hminus_boundfree",
        "hminus_freefree",
        "hydrogen_boundfree",
        "hydrogen_freefree",
        "h2plus",
        "heminus",
    ):
        assert np.all(components[name] >= 0.0)
    opacity = textbook_rosseland_opacity_v4r6(labels, temperature, pressure)
    assert np.all(np.isfinite(opacity) & (opacity > 0.0))
    np.testing.assert_allclose(
        textbook_rosseland_opacity_v4r5(labels, temperature, pressure),
        before,
        rtol=0.0,
        atol=0.0,
    )
    assert V4R6_FORMAL_TEMPERATURE_FLOOR_K == V4R5_FORMAL_TEMPERATURE_FLOOR_K
    assert V4R6_PER_N_TEMPERATURE_CEILING_K == 15000.0


def test_v4r6_grey_keeps_eddington_temperature_and_leaves_registered_seed():
    labels = np.asarray([[4000.0, 4.5, 0.0, 0.0, 1.0]])
    tau = 10.0 ** np.linspace(-6.875, 3.0, 12)
    grey = labels[:, 0, None] * (0.75 * (tau[None, :] + 2.0 / 3.0)) ** 0.25
    mass_grey, temperature_grey, _ = predict_textbook_reduced_state_v4r6(
        labels, tau, include_convection=False, substeps_per_layer=2
    )
    mass_convective, temperature_convective, _ = predict_textbook_reduced_state_v4r6(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    mass_default, temperature_default, _ = predict_textbook_reduced_state_v4r6(
        labels, tau, substeps_per_layer=2
    )
    np.testing.assert_allclose(temperature_grey, grey, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        temperature_default, temperature_convective, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(mass_default, mass_convective, rtol=0.0, atol=0.0)
    assert not np.allclose(temperature_grey, temperature_convective)
    assert not np.allclose(mass_grey, mass_convective, rtol=1.0e-12, atol=0.0)


