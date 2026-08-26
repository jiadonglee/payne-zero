"""Fast tests for the analytic grey-start benchmark."""

from __future__ import annotations

import numpy as np
import json

from experiments.reduced_state_emulator.grey_start_benchmark import (
    _worker,
    analytic_grey_atmosphere,
    decode_interpolated_full_state,
    encode_interpolated_full_state,
    full_state_to_atmosphere,
    has_finite_final_iteration,
    inverse_temperature_interpolation_coordinates,
    interpolated_full_state,
    interpolated_reduced_state,
    interpolation_coordinates,
    iteration_runtime_summary,
    load_full_donor_pool,
    load_donor_pool,
    representative_positions,
)
from payne_zero_atmosphere.run_setup import standard_rosseland_optical_depth_grid
from experiments.reduced_state_emulator.run_initializer_improvement_long import (
    _prepare_spectral_subset,
)


LABELS = {
    "effective_temperature": 5777.0,
    "log_surface_gravity": 4.44,
    "metallicity": 0.0,
    "alpha_enhancement": 0.0,
    "microturbulence_km_s": 2.0,
}


def test_analytic_grey_seed_matches_declared_equations():
    atmosphere = analytic_grey_atmosphere(LABELS)
    tau = standard_rosseland_optical_depth_grid(80)
    expected_temperature = LABELS["effective_temperature"] * (
        0.75 * (tau + 2.0 / 3.0)
    ) ** 0.25

    np.testing.assert_allclose(atmosphere.column_mass, tau / 0.34, rtol=6.0e-6)
    # The benchmark deliberately passes through the same fixed-width deck
    # quantization as every other solver start.
    np.testing.assert_allclose(atmosphere.temperature, expected_temperature, rtol=1.1e-5)
    np.testing.assert_allclose(
        atmosphere.gas_pressure,
        10.0 ** LABELS["log_surface_gravity"] * atmosphere.column_mass,
        rtol=4.0e-4,
    )
    assert np.all(np.diff(atmosphere.column_mass) > 0.0)
    assert np.all(np.isfinite(atmosphere.temperature))
    assert atmosphere.metadata["effective_temperature"] == "5777.000000"


def test_perturbed_grey_seed_is_deterministic_and_monotonic():
    first = analytic_grey_atmosphere(LABELS, perturbation_seed=20260812)
    second = analytic_grey_atmosphere(LABELS, perturbation_seed=20260812)
    other = analytic_grey_atmosphere(LABELS, perturbation_seed=20260813)

    np.testing.assert_array_equal(first.column_mass, second.column_mass)
    np.testing.assert_array_equal(first.temperature, second.temperature)
    assert not np.array_equal(first.temperature, other.temperature)
    assert np.all(np.diff(first.column_mass) > 0.0)
    assert np.all(first.temperature > 0.0)


def test_representative_positions_are_fixed_unique_and_spanning():
    labels = np.column_stack(
        (
            np.linspace(3500.0, 10000.0, 60),
            np.linspace(0.5, 5.0, 60),
            np.sin(np.arange(60)),
            np.cos(np.arange(60)),
            np.linspace(0.5, 4.0, 60) ** 2,
        )
    )
    first = representative_positions(labels, 12)
    second = representative_positions(labels, 12)
    assert first == second
    assert len(first) == len(set(first)) == 12
    assert min(first) >= 0 and max(first) < len(labels)


def test_iteration_runtime_separates_first_load_from_later_iterations():
    records = [
        {
            "trials": [
                {
                    "diagnostics": {
                        "iteration_timings": [
                            {"iteration": 1, "total_seconds": 100.0},
                            {"iteration": 2, "total_seconds": 8.0},
                            {"iteration": 3, "total_seconds": 10.0},
                        ]
                    }
                }
            ]
        }
    ]
    result = iteration_runtime_summary(records)
    assert result["first_iteration_mean_seconds"] == 100.0
    assert result["later_iteration_mean_seconds"] == 9.0


def test_grey60_only_accepts_numeric_grey30_final_states():
    finite = {
        "trials": [{"diagnostics": {"iteration_timings": [
            {"deep_layer_relative_temperature_change": 0.02}
        ]}}]
    }
    nonfinite = {
        "trials": [{"diagnostics": {"iteration_timings": [
            {"deep_layer_relative_temperature_change": None}
        ]}}]
    }
    assert has_finite_final_iteration(finite)
    assert not has_finite_final_iteration(nonfinite)
    assert not has_finite_final_iteration({"trials": []})


def test_spectral_subset_uses_solver_product_slug(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"spectral_selection": {"star_indices": [14767]}})
    )
    products = tmp_path / "products"
    subset = tmp_path / "subset"
    expected = "t06230.2_g+4.69_m-1.79_a+0.39_x3.14"
    for arm in ("production_six_field", "learned_reduced_state"):
        source = products / arm / f"{expected}.npz"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"product")

    result = _prepare_spectral_subset(manifest, products, subset)
    assert result["requested_slugs"] == [expected]
    assert (subset / "production_six_field" / f"{expected}.npz").is_symlink()


def _synthetic_donors(count: int = 40):
    """A donor pool whose (m, T) profiles are a smooth function of the labels."""

    tau = standard_rosseland_optical_depth_grid(80)
    table = np.column_stack(
        (
            np.linspace(4000.0, 9000.0, count),
            np.linspace(1.0, 5.0, count),
            np.linspace(-2.0, 0.3, count),
            np.linspace(0.0, 0.4, count),
        )
    )
    reduced = np.empty((count, tau.size, 2))
    for row in range(count):
        reduced[row, :, 0] = tau / 0.34 * 10.0 ** (-table[row, 1])
        reduced[row, :, 1] = table[row, 0] * (0.75 * (tau + 2.0 / 3.0)) ** 0.25
    coordinates = interpolation_coordinates(table)
    scale = np.maximum(coordinates.std(axis=0), 1.0e-12)
    return table, coordinates, scale, reduced


def test_interpolation_reproduces_a_donor_it_is_handed_exactly():
    table, coordinates, scale, reduced = _synthetic_donors()
    target = 17
    label_dict = {
        "effective_temperature": table[target, 0],
        "log_surface_gravity": table[target, 1],
        "metallicity": table[target, 2],
        "alpha_enhancement": table[target, 3],
        "microturbulence_km_s": 2.0,
    }
    column_mass, temperature, diagnostics = interpolated_reduced_state(
        label_dict, coordinates, scale, reduced
    )
    # A zero-distance donor must take all the weight rather than divide by zero.
    assert diagnostics["nearest_distance"] == 0.0
    assert diagnostics["top_weight"] == 1.0
    np.testing.assert_allclose(column_mass, reduced[target, :, 0], rtol=1.0e-12)
    np.testing.assert_allclose(temperature, reduced[target, :, 1], rtol=1.0e-12)


def test_interpolated_state_is_monotone_positive_and_bounded_by_donors():
    table, coordinates, scale, reduced = _synthetic_donors()
    label_dict = {
        "effective_temperature": 0.5 * (table[10, 0] + table[11, 0]),
        "log_surface_gravity": 0.5 * (table[10, 1] + table[11, 1]),
        "metallicity": 0.5 * (table[10, 2] + table[11, 2]),
        "alpha_enhancement": 0.5 * (table[10, 3] + table[11, 3]),
        "microturbulence_km_s": 2.0,
    }
    column_mass, temperature, _ = interpolated_reduced_state(
        label_dict, coordinates, scale, reduced, neighbours=8
    )
    assert np.all(np.diff(column_mass) > 0.0)
    assert np.all(temperature > 0.0)
    # Log-space convex weights cannot leave the donor envelope.
    assert column_mass.min() >= reduced[:, :, 0].min() * (1.0 - 1.0e-12)
    assert temperature.max() <= reduced[:, :, 1].max() * (1.0 + 1.0e-12)


def _synthetic_complete_donors(count: int = 40):
    table, coordinates, scale, reduced = _synthetic_donors(count)
    coordinates = inverse_temperature_interpolation_coordinates(table)
    scale = np.maximum(coordinates.std(axis=0), 1.0e-12)
    tau = standard_rosseland_optical_depth_grid(80)
    profiles = np.zeros((count, tau.size, 6), dtype=np.float64)
    for row in range(count):
        profiles[row, :, 0] = reduced[row, :, 0]
        profiles[row, :, 1] = reduced[row, :, 1]
        profiles[row, :, 2] = 10.0 ** table[row, 1] * profiles[row, :, 0]
        profiles[row, :, 3] = 1.0e12 * profiles[row, :, 1] / profiles[row, :, 2]
        profiles[row, :, 4] = 0.1 + 0.01 * row + 0.001 * tau
        profiles[row, :, 5] = (row - 10.0) * np.sqrt(tau + 0.01)
    encoded = encode_interpolated_full_state(profiles, table[:, 0], tau)
    return table, coordinates, scale, profiles, encoded, tau


def test_full_state_interpolation_round_trips_an_exact_donor_and_deck():
    table, coordinates, scale, profiles, encoded, tau = _synthetic_complete_donors()
    target = 17
    label_dict = {
        "effective_temperature": table[target, 0],
        "log_surface_gravity": table[target, 1],
        "metallicity": table[target, 2],
        "alpha_enhancement": table[target, 3],
        "microturbulence_km_s": 2.0,
    }
    profile, diagnostics = interpolated_full_state(
        label_dict,
        coordinates,
        scale,
        encoded,
        tau,
        donor_indices=np.arange(len(table)),
    )
    assert diagnostics["nearest_distance"] == 0.0
    assert diagnostics["top_weight"] == 1.0
    np.testing.assert_allclose(profile, profiles[target], rtol=2.0e-12, atol=1.0e-12)
    atmosphere = full_state_to_atmosphere(profile, label_dict)
    assert atmosphere.column_mass.shape == (80,)
    assert np.all(np.isfinite(atmosphere.temperature))
    assert np.all(np.diff(atmosphere.column_mass) > 0.0)


def test_full_state_interpolation_is_finite_monotone_and_uses_zero_safe_gradients():
    table, coordinates, scale, profiles, encoded, tau = _synthetic_complete_donors()
    label_dict = {
        "effective_temperature": 0.5 * (table[10, 0] + table[11, 0]),
        "log_surface_gravity": 0.5 * (table[10, 1] + table[11, 1]),
        "metallicity": 0.5 * (table[10, 2] + table[11, 2]),
        "alpha_enhancement": 0.5 * (table[10, 3] + table[11, 3]),
        "microturbulence_km_s": 2.0,
    }
    profile, _ = interpolated_full_state(
        label_dict, coordinates, scale, encoded, tau, neighbours=8
    )
    assert np.all(np.isfinite(profile))
    assert np.all(profile[:, :5] > 0.0)
    assert np.all(np.diff(profile[:, 0]) > 0.0)
    zero_gradient = profiles.copy()
    zero_gradient[:, :, 5] = 0.0
    zero_encoded = encode_interpolated_full_state(zero_gradient, table[:, 0], tau)
    zero_profile = decode_interpolated_full_state(
        np.mean(zero_encoded[:8], axis=0), label_dict["effective_temperature"], tau
    )
    np.testing.assert_allclose(zero_profile[:, 5], 0.0, atol=1.0e-12)


def test_donor_pool_excludes_every_scored_manifest_star(tmp_path):
    corpus = tmp_path / "corpus.npz"
    count = 12
    labels_json = np.asarray(
        [
            json.dumps(
                {
                    "effective_temperature": 5000.0 + 100.0 * i,
                    "log_surface_gravity": 2.0 + 0.1 * i,
                    "metallicity": -1.0,
                    "alpha_enhancement": 0.2,
                    "microturbulence_km_s": 2.0,
                }
            )
            for i in range(count)
        ]
    )
    np.savez(
        corpus,
        labels_json=labels_json,
        atmosphere_profiles=np.ones((count, 80, 6)),
    )
    manifest = tmp_path / "scored.json"
    manifest.write_text(json.dumps({"star_indices": [1, 4, 9]}))

    _coords, _scale, reduced, provenance, excluded = load_donor_pool(
        corpus, (manifest, tmp_path / "absent.json")
    )
    assert excluded == {1, 4, 9}
    assert provenance["donor_star_count"] == count - 3
    assert reduced.shape == (count - 3, 80, 2)
    # A manifest that is not on disk is skipped, not silently counted as empty.
    assert [entry["star_count"] for entry in provenance["excluded_manifests"]] == [3]


def test_full_donor_pool_excludes_target_and_preserves_complete_state(tmp_path):
    table, _coordinates, _scale, profiles, _encoded, tau = _synthetic_complete_donors(12)
    corpus = tmp_path / "complete_corpus.npz"
    labels_json = np.asarray(
        [
            json.dumps(
                {
                    "effective_temperature": row[0],
                    "log_surface_gravity": row[1],
                    "metallicity": row[2],
                    "alpha_enhancement": row[3],
                    "microturbulence_km_s": 2.0,
                }
            )
            for row in table
        ]
    )
    np.savez(
        corpus,
        labels_json=labels_json,
        atmosphere_profiles=profiles,
        standard_rosseland_optical_depth=np.repeat(tau[None, :], 12, axis=0),
    )
    manifest = tmp_path / "scored_complete.json"
    manifest.write_text(json.dumps({"star_indices": [1, 4, 9]}))
    coords, scale, encoded, loaded_tau, provenance, excluded, donor_indices = load_full_donor_pool(
        corpus, (manifest,)
    )
    assert excluded == {1, 4, 9}
    assert provenance["donor_star_count"] == 9
    assert encoded.shape == (9, 80, 6)
    assert loaded_tau.shape == (80,)
    assert not set(donor_indices) & {1, 4, 9}
    assert coords.shape == (9, 4)
    assert scale.shape == (4,)


def test_initializer_failure_is_recorded_without_aborting_batch():
    record = _worker(
        (
            "learned_reduced_state",
            np.ones((80, 6)),
            LABELS,
            (np.zeros(80), np.ones(80) * 5000.0),
            None,
            {"source": "learned_reduced_state"},
        )
    )
    assert not record["converged"]
    assert record["total_iterations"] == 0
    assert "strictly increasing" in record["trials"][0]["error"]
