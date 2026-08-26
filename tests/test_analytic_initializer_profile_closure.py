"""Tests for the opacity-to-column-mass closure probe."""

from __future__ import annotations

import numpy as np

from experiments.analytic_initializer.profile_closure import (
    integrate_mass_from_opacity,
)


def test_constant_opacity_integral_matches_tau_over_kappa() -> None:
    tau = np.logspace(-6, 2, 80)
    opacity = np.full((2, tau.size), 0.5)
    mass = integrate_mass_from_opacity(tau, np.log10(opacity))
    expected = np.broadcast_to(tau[None, :] / 0.5, mass.shape)
    # Trapezoidal integration is exact for constant opacity up to the first
    # finite-grid interval, where the surface seed is explicitly defined.
    np.testing.assert_allclose(mass, expected, rtol=2.0e-3, atol=1.0e-12)


def test_surface_mass_override_is_respected() -> None:
    tau = np.logspace(-6, 2, 80)
    opacity = np.full((3, tau.size), 0.25)
    surface = np.array([1.0e-5, 2.0e-5, 3.0e-5])
    mass = integrate_mass_from_opacity(tau, np.log10(opacity), surface_mass=surface)
    np.testing.assert_allclose(mass[:, 0], surface)
    assert np.all(np.diff(mass, axis=1) > 0.0)
