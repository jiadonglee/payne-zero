"""Tests for the reduced-state resample and reconstruction modules.

The resample round-trip test is fast (no solver, pure numpy/scipy) and runs
by default. The reconstruction test calls the real solver's population,
opacity, transfer, and hydrostatic stages and is opt-in, matching
``tests/test_integration.py``'s convention:

    PAYNE_ZERO_RUN_SOLVER=1 python -m pytest tests/test_reduced_state.py -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from experiments.reduced_state_emulator.compose_physical_predictions import compose
from experiments.reduced_state_emulator.predict_physical import (
    combine_physical_predictions,
)
from reduced_state.emulator import (
    PhysicalReducedStateEmulator,
    decode_physical_state,
    fit_physical_standardization,
    grey_temperature,
    label_features,
    production_tau_grid,
)
from reduced_state.resample import (
    production_tau_grid,
    representation_error,
    resample_reduced_state,
)


SOLAR_LABELS = dict(
    effective_temperature=5777.0,
    log_surface_gravity=4.44,
    metallicity=0.0,
    alpha_enhancement=0.0,
    microturbulence_km_s=2.0,
)


def _gray_atmosphere_profile(tau_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A smooth, strictly-increasing synthetic (m,T)(tau) profile for fast tests.

    Not a converged solver output -- just a physically plausible smooth
    curve (Eddington-gray T(tau), m roughly proportional to tau) so the
    resample round-trip has something realistic to bite on without paying
    for a real solve.
    """

    temperature = SOLAR_LABELS["effective_temperature"] * (
        0.75 * (tau_std + 2.0 / 3.0)
    ) ** 0.25
    column_mass = 10.0 * tau_std ** 0.95
    return column_mass, temperature


def test_resample_round_trip_improves_with_resolution():
    tau_std = production_tau_grid()
    column_mass, temperature = _gray_atmosphere_profile(tau_std)

    errors_by_n = {}
    for n in (20, 40, 80, 160, 320):
        err = representation_error(tau_std, column_mass, temperature, n)
        errors_by_n[n] = float(np.max(err["column_mass_relative_error"]))

    # A coarser intermediate grid should never reconstruct better than a
    # finer one on a smooth profile -- monotone non-increasing error.
    ordered = [errors_by_n[n] for n in sorted(errors_by_n)]
    for worse, better in zip(ordered, ordered[1:]):
        assert better <= worse + 1e-9

    # And by 320 points the round-trip should be very close to identity.
    assert errors_by_n[320] < 1e-4


def test_resample_at_native_resolution_is_near_identity():
    tau_std = production_tau_grid()
    column_mass, temperature = _gray_atmosphere_profile(tau_std)
    resampled_m, resampled_t = resample_reduced_state(
        tau_std, column_mass, temperature, len(tau_std)
    )
    np.testing.assert_allclose(resampled_m, column_mass, rtol=1e-6)
    np.testing.assert_allclose(resampled_t, temperature, rtol=1e-6)


def test_resample_rejects_grid_too_coarse_for_a_cubic_spline():
    tau_std = production_tau_grid()
    column_mass, temperature = _gray_atmosphere_profile(tau_std)
    with pytest.raises(ValueError):
        resample_reduced_state(tau_std, column_mass, temperature, 3)


def test_physical_coordinates_round_trip_and_preserve_monotonicity():
    tau = production_tau_grid()
    labels = np.array(
        [
            [5777.0, 4.44, 0.0, 0.0, 2.0],
            [6200.0, 4.10, -0.5, 0.1, 1.5],
        ],
        dtype=np.float64,
    )
    column_mass = np.stack((10.0 * tau**0.95, 7.0 * tau**0.90))
    temperature = np.stack(
        (
            grey_temperature(labels[0, 0], tau),
            grey_temperature(labels[1, 0], tau) * 1.02,
        )
    )
    standardization = fit_physical_standardization(
        label_features(labels), column_mass, temperature, labels[:, 0], tau
    )
    features = label_features(labels)
    standardized = (
        features - standardization.feature_mean
    ) / standardization.feature_std
    increments = np.empty_like(column_mass)
    increments[:, 0] = column_mass[:, 0]
    increments[:, 1:] = np.diff(column_mass, axis=1)
    raw_temperature = torch.as_tensor(
        (
            np.log10(temperature / grey_temperature(labels[:, 0], tau))
            - standardization.log_temperature_ratio_mean
        )
        / standardization.log_temperature_ratio_std,
        dtype=torch.float64,
    )
    raw_mass = torch.as_tensor(
        (
            np.log10(increments) - standardization.log_mass_increment_mean
        )
        / standardization.log_mass_increment_std,
        dtype=torch.float64,
    )
    decoded_mass, decoded_temperature = decode_physical_state(
        raw_temperature,
        raw_mass,
        standardization,
        torch.as_tensor(labels[:, 0], dtype=torch.float64),
        tau,
    )
    np.testing.assert_allclose(decoded_mass.numpy(), column_mass, rtol=1e-12)
    np.testing.assert_allclose(decoded_temperature.numpy(), temperature, rtol=1e-12)
    assert np.all(np.diff(decoded_mass.numpy(), axis=1) > 0.0)


def test_physical_model_always_decodes_positive_monotone_mass():
    model = PhysicalReducedStateEmulator(width=8, depth=2)
    features = torch.zeros((4, 5), dtype=torch.float64)
    raw_temperature, raw_mass = model(features)
    tau = production_tau_grid()
    labels = torch.full((4,), 5777.0, dtype=torch.float64)
    standardization = fit_physical_standardization(
        np.zeros((2, 5)),
        np.stack((10.0 * tau**0.95, 11.0 * tau**0.90)),
        np.stack((grey_temperature(5777.0, tau), grey_temperature(5777.0, tau))),
        np.array([5777.0, 5777.0]),
        tau,
    )
    mass, temperature = decode_physical_state(
        raw_temperature, raw_mass, standardization, labels, tau
    )
    assert torch.isfinite(mass).all()
    assert torch.isfinite(temperature).all()
    assert torch.all(mass > 0.0)
    assert torch.all(torch.diff(mass, dim=1) > 0.0)


def test_physical_ensemble_regional_override_is_explicit_and_monotone():
    labels = np.array(
        [
            [7000.0, 2.0, -2.0, 0.2, 1.0],
            [5000.0, 2.0, -2.0, 0.2, 1.0],
        ]
    )
    masses = [
        np.array([[1.0, 2.0], [1.0, 2.0]]),
        np.array([[1.2, 2.2], [1.2, 2.2]]),
        np.array([[1.4, 2.4], [1.4, 2.4]]),
    ]
    temperatures = [
        np.array([[700.0, 800.0], [500.0, 600.0]]),
        np.array([[710.0, 810.0], [510.0, 610.0]]),
        np.array([[720.0, 820.0], [520.0, 620.0]]),
    ]
    mass, temperature, selected = combine_physical_predictions(
        masses,
        temperatures,
        labels,
        (7, 8, 9),
        regional_seed=8,
    )
    np.testing.assert_array_equal(selected, [True, False])
    np.testing.assert_allclose(mass[0], masses[1][0])
    np.testing.assert_allclose(mass[1], np.median([arm[1] for arm in masses], axis=0))
    np.testing.assert_allclose(temperature[0], temperatures[1][0])
    assert np.all(np.diff(mass, axis=1) > 0.0)


def _composition_payload(labels, offset=0.0):
    count = len(labels)
    mass = np.tile([1.0 + offset, 2.0 + offset], (count, 1))
    return {
        "star_indices": np.arange(count),
        "labels": np.asarray(labels),
        "column_mass": mass,
        "temperature": np.tile([5000.0 + offset, 5100.0 + offset], (count, 1)),
        "truth_column_mass": np.tile([1.0, 2.0], (count, 1)),
        "truth_temperature": np.tile([5000.0, 5100.0], (count, 1)),
    }


def test_regional_alpha_vmic_box_selects_only_matching_star_and_stays_monotone():
    labels = [[5500, 4.0, 0.0, 0.5, 3.5], [5500, 4.0, 0.0, 0.2, 2.0]]
    base, override = _composition_payload(labels), _composition_payload(labels, 0.2)
    mass, _, selected = compose(base, override, alpha_min=0.4, vmic_min=3.0)
    np.testing.assert_array_equal(selected, [True, False])
    np.testing.assert_allclose(mass[0], override["column_mass"][0])
    np.testing.assert_allclose(mass[1], base["column_mass"][1])
    assert np.all(np.diff(mass, axis=1) > 0.0)


def test_solver_tail_region_matches_training_definition():
    labels = [[5500, 3.1, 0, 0.0, 1.0], [5500, 4.0, 0, 0.0, 3.1],
              [5500, 4.0, 0, 0.5, 1.0], [5500, 4.0, 0, 0.0, 1.0]]
    base, override = _composition_payload(labels), _composition_payload(labels, 0.2)
    _, _, selected = compose(base, override, solver_tail_region=True)
    expected = ((np.asarray(labels)[:, 1] < 3.2) | (np.asarray(labels)[:, 4] > 3.0)
                | (np.asarray(labels)[:, 3] > 0.4))
    np.testing.assert_array_equal(selected, expected)


def test_solver_tail_region_rejects_box_flags():
    base = _composition_payload([[5500, 4.0, 0, 0.0, 1.0]])
    with pytest.raises(SystemExit):
        compose(base, base, solver_tail_region=True, alpha_min=0.4)


pytestmark_solver = pytest.mark.skipif(
    os.environ.get("PAYNE_ZERO_RUN_SOLVER") != "1",
    reason="set PAYNE_ZERO_RUN_SOLVER=1 to run the real solver (~1-2 min)",
)


@pytestmark_solver
def test_reconstruction_from_truth_is_physically_ordered_and_finite():
    """Reconstructing from a converged truth (m,T) should give a sane atmosphere.

    Not a parity test (see experiments/reduced_state_parity for the
    quantitative error-vs-tau comparison against corpus truth) -- this only
    checks the reconstruction interface's basic physical contract: finite,
    positive, monotonic column mass, and m/T pinned exactly to the input.
    """

    from continuity.closure import TARGET_FIELDS, load_corpus
    from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere
    from payne_zero_atmosphere.data_files import atmosphere_emulator_dir

    corpus_path = (
        atmosphere_emulator_dir() / "five_label" / "strict_truth_52199.npz"
    )
    profiles, tau_std, _iters, labels = load_corpus(corpus_path)
    field_index = {name: i for i, name in enumerate(TARGET_FIELDS)}
    star = 0
    column_mass = profiles[star, :, field_index["column_mass"]]
    temperature = profiles[star, :, field_index["temperature"]]
    label_fields = (
        "effective_temperature",
        "log_surface_gravity",
        "metallicity",
        "alpha_enhancement",
        "microturbulence_km_s",
    )
    star_labels = {key: labels[star][key] for key in label_fields}

    reduced = ReducedAtmosphere(
        column_mass=column_mass, temperature=temperature, labels=star_labels
    )
    result = reconstruct_full_atmosphere(reduced, n_synchronizations=2)
    atmosphere = result.atmosphere

    assert result.n_synchronizations == 2
    assert result.n_evaluations == 3
    assert result.n_pressure_updates == 2
    assert len(result.pressure_change_dex_by_pass) == 3
    np.testing.assert_array_equal(
        atmosphere.gas_pressure, result.gas_pressure_by_pass[-1]
    )
    np.testing.assert_array_equal(
        atmosphere.electron_density, result.electron_density_by_pass[-1]
    )
    np.testing.assert_array_equal(
        atmosphere.rosseland_opacity, result.rosseland_opacity_by_pass[-1]
    )
    np.testing.assert_array_equal(
        atmosphere.radiative_acceleration,
        result.radiative_acceleration_by_pass[-1],
    )
    np.testing.assert_array_equal(atmosphere.column_mass, column_mass)
    np.testing.assert_array_equal(atmosphere.temperature, temperature)
    for field in (
        atmosphere.gas_pressure,
        atmosphere.electron_density,
        atmosphere.rosseland_opacity,
        atmosphere.radiative_acceleration,
    ):
        assert np.all(np.isfinite(field))
    assert np.all(atmosphere.gas_pressure > 0.0)
    assert np.all(atmosphere.electron_density > 0.0)
    assert np.all(atmosphere.rosseland_opacity > 0.0)
    assert np.all(np.diff(atmosphere.column_mass) > 0.0)
