"""Unit tests for the solver-path geometry and deterministic candidate gates."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.reduced_state_emulator.convergence_geometry import (
    ARM_NAMES,
    direction_cosine,
    make_candidate_row,
    pca_2d,
    progress_fraction,
    select_candidate,
    state_distance,
    state_vector,
)


def _records(two: int = 3, six: int = 8, interp: int = 6):
    return {
        arm: {
            "star": {
                "converged": True,
                "converging_trial_iterations": count,
            }
        }
        for arm, count in (
            ("learned_reduced_state", two),
            ("production_six_field", six),
            ("interpolated_full_state", interp),
        )
    }


def _state(values: np.ndarray) -> np.ndarray:
    return np.concatenate((values, values))


def test_state_vector_and_distance_use_log_dex_coordinates():
    temperature = np.array([100.0, 200.0])
    pressure = np.array([10.0, 20.0])
    vector = state_vector(temperature, pressure)
    assert vector.shape == (4,)
    reference = state_vector(temperature * 10.0, pressure * 10.0)
    metrics = state_distance(vector, reference)
    np.testing.assert_allclose(metrics["temperature_rms_dex"], 1.0)
    np.testing.assert_allclose(metrics["pressure_rms_dex"], 1.0)
    np.testing.assert_allclose(metrics["combined_rms_dex"], 1.0)


def test_direction_and_progress_are_positive_for_a_direct_step():
    reference = np.zeros(4)
    previous = np.ones(4)
    following = np.full(4, 0.5)
    assert direction_cosine(previous, following, reference) == pytest.approx(1.0)
    assert progress_fraction(previous, following, reference) == pytest.approx(0.5)


def test_candidate_requires_six_field_closer_but_two_field_more_aligned():
    reference = np.zeros(4)
    initial = {
        arm: {"star": vector}
        for arm, vector in (
            ("learned_reduced_state", np.full(4, 2.0)),
            ("production_six_field", np.full(4, 0.5)),
            ("interpolated_full_state", np.full(4, 1.0)),
        )
    }
    first = {
        "learned_reduced_state": {"star": np.full(4, 1.0)},
        "production_six_field": {"star": np.full(4, 0.75)},
    }
    row = make_candidate_row(
        "star",
        records=_records(),
        initial_states=initial,
        first_states=first,
        reference=reference,
        category="ordinary",
    )
    assert row["eligible"]
    assert row["gates"]["six_field_closer_in_temperature"]
    assert row["gates"]["six_field_closer_in_pressure"]
    assert row["gates"]["two_field_first_step_more_aligned"]
    assert row["iterations"]["learned_reduced_state"] == 3


def test_candidate_selector_uses_gap_then_alignment_then_distance():
    common = {
        "eligible": True,
        "alignment_advantage_two_minus_six": 0.1,
        "distance_advantage_two_over_six": 2.0,
    }
    rows = [
        {
            **common,
            "slug": "larger-gap",
            "iteration_gap_six_minus_two": 5,
        },
        {
            **common,
            "slug": "larger-alignment",
            "iteration_gap_six_minus_two": 4,
            "alignment_advantage_two_minus_six": 0.9,
        },
    ]
    assert select_candidate(rows)["slug"] == "larger-gap"


def test_pca_is_finite_and_returns_two_components():
    states = np.vstack(
        [
            np.zeros(4),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        ]
    )
    coordinates, explained, components = pca_2d(states)
    assert coordinates.shape == (3, 2)
    assert explained.shape == (2,)
    assert components.shape == (2, 4)
    assert np.all(np.isfinite(coordinates))
    assert np.all(explained >= 0.0)
