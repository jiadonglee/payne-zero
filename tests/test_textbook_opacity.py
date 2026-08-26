"""Tests for the named-constant textbook opacity/ODE diagnostic."""

import numpy as np

from experiments.analytic_initializer.textbook_opacity import (
    build_textbook_reduced_state,
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
        "kramers_freefree_boundfree",
        "electron_scattering",
        "total",
    }
    for value in components.values():
        assert np.all(np.isfinite(value))
        assert np.all(value > 0.0)
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
