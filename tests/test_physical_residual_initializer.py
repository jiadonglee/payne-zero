"""Pure-contract tests for the low-dimensional residual prototype."""

from __future__ import annotations

import numpy as np

from experiments.analytic_initializer.physical_homotopy import coarse_rosseland_tau
from experiments.analytic_initializer.physical_residual_initializer import (
    _decode_coefficients,
    _increment_basis,
    _positive_monotone_profile,
)


def test_zero_coefficients_reproduce_the_reference_profiles() -> None:
    tau = coarse_rosseland_tau()
    _, basis = _increment_basis(tau, 5)
    base = tau / 0.34
    reconstructed = _positive_monotone_profile(
        base,
        anchor_dex=0.0,
        increment_dex=np.zeros(5),
        basis=basis,
    )
    assert np.allclose(reconstructed, base)


def test_increment_parameterization_stays_strictly_monotone() -> None:
    tau = coarse_rosseland_tau()
    _, basis = _increment_basis(tau, 5)
    profile = _positive_monotone_profile(
        tau / 0.34,
        anchor_dex=0.4,
        increment_dex=np.array([1.0, -1.0, 0.5, -0.5, 0.2]),
        basis=basis,
    )
    assert np.all(np.isfinite(profile))
    assert np.all(profile > 0.0)
    assert np.all(np.diff(profile) > 0.0)


def test_decoder_returns_positive_monotone_mass_and_temperature() -> None:
    tau = coarse_rosseland_tau()
    _, basis = _increment_basis(tau, 5)
    mass, temperature = _decode_coefficients(
        np.zeros(12),
        tau=tau,
        effective_temperature=4500.0,
        basis=basis,
        knot_count=5,
    )
    assert np.all(np.isfinite(mass)) and np.all(np.isfinite(temperature))
    assert np.all(np.diff(mass) > 0.0)
    assert np.all(np.diff(temperature) > 0.0)
