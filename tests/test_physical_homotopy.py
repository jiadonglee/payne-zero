"""Pure-contract tests for the coarse physical homotopy pilot."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.analytic_initializer.physical_homotopy import (
    GroupedContinuumOpacity,
    HomotopyResult,
    build_grouped_continuum_opacity,
    coarse_rosseland_tau,
    frequency_group_slices,
    resample_to_production_grid,
    two_stream_transfer,
)


def test_coarse_grid_and_groups_are_fixed_and_monotone() -> None:
    tau = coarse_rosseland_tau()
    assert tau.size == 32
    assert np.all(np.diff(tau) > 0.0)
    groups = frequency_group_slices(np.ones(17), group_count=4)
    assert groups == ((0, 5), (5, 9), (9, 13), (13, 17))


def test_grouped_opacity_is_positive_and_finite() -> None:
    layers, frequencies = 8, 20
    frequency = np.geomspace(3.0e12, 2.0e15, frequencies)
    weights = np.linspace(1.0, 2.0, frequencies)
    temperature = np.linspace(3500.0, 12000.0, layers)
    absorption = np.full((layers, frequencies), 0.2)
    scattering = np.full((layers, frequencies), 0.03)
    source = np.full((layers, frequencies), 1.0e-4)
    grouped = build_grouped_continuum_opacity(
        frequency_hz=frequency,
        frequency_weights=weights,
        absorption=absorption,
        scattering=scattering,
        source=source,
        temperature_k=temperature,
    )
    assert grouped.absorption.shape == (layers, 4)
    assert np.all(np.isfinite(grouped.rosseland_opacity))
    assert np.all(grouped.rosseland_opacity > 0.0)
    assert np.all(np.isfinite(grouped.source))
    for group_index, (start, stop) in enumerate(grouped.group_slices):
        assert np.allclose(
            grouped.source[:, group_index],
            1.0e-4 * np.sum(weights[start:stop]),
        )


def test_two_stream_preserves_finite_monotone_contract() -> None:
    layers = 12
    tau = 10.0 ** np.linspace(-6.0, 2.0, layers)
    mass = tau / 0.3
    frequency = np.geomspace(3.0e12, 2.0e15, 20)
    weights = np.ones(20)
    temperature = np.linspace(4000.0, 14000.0, layers)
    grouped = build_grouped_continuum_opacity(
        frequency_hz=frequency,
        frequency_weights=weights,
        absorption=np.full((layers, 20), 0.3),
        scattering=np.full((layers, 20), 0.01),
        source=np.full((layers, 20), 1.0e-4),
        temperature_k=temperature,
    )
    transfer = two_stream_transfer(column_mass=mass, opacity=grouped)
    assert transfer.group_flux.shape == (layers, 4)
    assert np.all(np.isfinite(transfer.total_flux))
    assert np.all(np.isfinite(transfer.integrated_radiation_pressure))


def test_two_stream_grey_source_has_no_deep_cell_flux_spike() -> None:
    """A linear grey source must remain a nearly constant flux solution."""

    layers = 32
    tau = coarse_rosseland_tau(layers)
    target_flux = 1.0
    source = 3.0 * target_flux * (tau + 2.0 / 3.0)
    grouped = GroupedContinuumOpacity(
        group_slices=((0, 1),),
        frequency_hz=np.array([1.0]),
        frequency_weights=np.array([1.0]),
        absorption=np.ones((layers, 1)),
        scattering=np.zeros((layers, 1)),
        source=source[:, None],
        rosseland_opacity=np.ones(layers),
    )
    transfer = two_stream_transfer(
        column_mass=tau,
        opacity=grouped,
        target_integrated_flux=target_flux,
    )
    ratio = transfer.total_flux / target_flux
    assert np.all(np.isfinite(ratio))
    assert float(np.max(ratio)) < 1.15
    assert float(np.min(ratio)) > 0.90


def test_resample_requires_positive_strict_mass() -> None:
    coarse_tau = coarse_rosseland_tau()
    result = HomotopyResult(
        coarse_tau=coarse_tau,
        column_mass=coarse_tau / 0.3,
        temperature=np.linspace(3500.0, 10000.0, 32),
        rosseland_opacity=np.full(32, 0.3),
        diagnostics={},
    )
    target_tau, mass, temperature, opacity = resample_to_production_grid(result)
    assert target_tau.size == mass.size == temperature.size == opacity.size == 80
    assert np.all(np.diff(mass) > 0.0)
    assert np.all(temperature > 0.0)
    assert np.all(opacity > 0.0)


def test_frequency_group_rejects_nonpositive_weights() -> None:
    with pytest.raises(ValueError):
        frequency_group_slices(np.array([1.0, 0.0, 1.0]), group_count=2)
