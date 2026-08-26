"""Tests for the named-constant textbook opacity/ODE diagnostic."""

import numpy as np

from experiments.analytic_initializer.textbook_opacity import (
    DEFAULT_TEXTBOOK_CONSTANTS,
    build_textbook_reduced_state,
    hminus_freefree_rosseland_opacity,
    integrate_hydrostatic_opacity_ode,
    saha_aware_adiabatic_gradient,
    saha_electron_diagnostics,
    textbook_opacity_components,
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
