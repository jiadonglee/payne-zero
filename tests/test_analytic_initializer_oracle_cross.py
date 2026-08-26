"""Unit checks for the oracle cross-arm construction."""

import numpy as np
import pytest

from experiments.analytic_initializer.run_oracle_cross import (
    GREY_OPACITY_CM2_G,
    build_oracle_reduced_state,
)


def _inputs():
    labels = np.asarray(
        [[5000.0, 4.0, -1.0, 0.2, 1.0], [8000.0, 3.0, 0.0, 0.0, 1.5]]
    )
    tau = np.asarray([1.0e-3, 1.0e-2, 1.0e-1, 1.0])
    truth_mass = np.asarray(
        [[0.01, 0.08, 0.9, 8.0], [0.02, 0.12, 1.1, 7.0]]
    )
    truth_temperature = np.asarray(
        [[4200.0, 4500.0, 5200.0, 6800.0], [7000.0, 7600.0, 9000.0, 11000.0]]
    )
    return labels, tau, truth_mass, truth_temperature


def test_cross_arms_change_only_the_declared_field():
    labels, tau, truth_mass, truth_temperature = _inputs()
    left = build_oracle_reduced_state(
        labels,
        tau,
        truth_mass,
        truth_temperature,
        mass_source="truth",
        temperature_source="grey",
    )
    right = build_oracle_reduced_state(
        labels,
        tau,
        truth_mass,
        truth_temperature,
        mass_source="grey",
        temperature_source="truth",
    )
    grey_temperature = labels[:, 0, None] * (0.75 * (tau[None, :] + 2.0 / 3.0)) ** 0.25
    grey_mass = np.broadcast_to(
        tau[None, :] / GREY_OPACITY_CM2_G, truth_mass.shape
    )
    np.testing.assert_allclose(left[0], truth_mass)
    np.testing.assert_allclose(left[1], grey_temperature)
    np.testing.assert_allclose(right[0], grey_mass)
    np.testing.assert_allclose(right[1], truth_temperature)
    np.testing.assert_allclose(left[2], right[2], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(left[2], np.log10(GREY_OPACITY_CM2_G))


def test_cross_arm_rejects_non_monotone_truth_mass():
    labels, tau, truth_mass, truth_temperature = _inputs()
    truth_mass[0, 2] = truth_mass[0, 1]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_oracle_reduced_state(
            labels,
            tau,
            truth_mass,
            truth_temperature,
            mass_source="truth",
            temperature_source="grey",
        )
