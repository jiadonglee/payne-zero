import json
from pathlib import Path

import numpy as np

from experiments.reduced_state_emulator.m_star_bootstrap_v1r2_marcs100 import (
    CAMPAIGN,
    TARGET_PER_CLASS,
    TEMPERATURE_PRIORITY,
    build_candidates,
    build_tracks,
    class_status,
    select_training_records,
)


def test_fixed_pool_has_balanced_train_candidates_and_reserved_tracks() -> None:
    tracks = build_tracks()
    candidates = build_candidates(tracks)
    assert len(tracks) == 24
    assert len(candidates) == 216
    assert len({row["candidate_id"] for row in candidates}) == 216
    assert sum(row["class"] == "giant" for row in candidates) == 108
    assert sum(row["class"] == "dwarf" for row in candidates) == 108
    assert {row["role"] for row in candidates} == {"train"}
    assert sum(row["role"] == "validation" for row in tracks) == 4
    assert sum(row["role"] == "sealed" for row in tracks) == 2
    assert sum(row["role"] == "train" for row in tracks) == 18


def test_priority_spans_temperature_before_repeating_the_pool() -> None:
    candidates = build_candidates()
    giants = [row for row in candidates if row["class"] == "giant"]
    assert giants[0]["temperature_K"] == TEMPERATURE_PRIORITY[0]
    assert giants[9]["temperature_K"] == TEMPERATURE_PRIORITY[1]
    assert [row["priority"] for row in giants] == list(range(108))


def _record(stellar_class: str, priority: int, eligible: bool) -> dict:
    logg = 1.5 if stellar_class == "giant" else 5.0
    return {
        "campaign": CAMPAIGN,
        "candidate_id": f"{stellar_class}-{priority}",
        "priority": priority,
        "class": stellar_class,
        "role": "train",
        "track": {
            "track_id": f"{stellar_class}-track",
            "log_surface_gravity": logg,
        },
        "training_eligible": eligible,
    }


def test_selection_takes_exactly_first_50_eligible_per_class() -> None:
    records = []
    for stellar_class in ("giant", "dwarf"):
        records.extend(
            _record(stellar_class, priority, eligible=priority % 7 != 0)
            for priority in range(80)
        )
    selected = select_training_records(records)
    assert len(selected) == 2 * TARGET_PER_CLASS == 100
    for stellar_class in ("giant", "dwarf"):
        priorities = [
            row["priority"]
            for row in selected
            if row["class"] == stellar_class
        ]
        expected = [
            value for value in range(80) if value % 7 != 0
        ][:TARGET_PER_CLASS]
        assert priorities == expected


def test_class_status_counts_attempted_eligible_and_pending(tmp_path: Path) -> None:
    candidates = build_candidates()
    protocol = {
        "campaign": CAMPAIGN,
        "split": {"new_train_candidates": candidates},
    }
    giant = next(row for row in candidates if row["class"] == "giant")
    case_path = (
        tmp_path
        / "cases"
        / "giant"
        / giant["track"]["track_id"]
        / f"t{int(giant['temperature_K']):04d}"
        / "case.json"
    )
    case_path.parent.mkdir(parents=True)
    case_path.write_text(
        json.dumps({**giant, "training_eligible": True})
    )
    status = class_status(
        tmp_path,
        protocol=protocol,
        stellar_class="giant",
    )
    assert status["attempted_count"] == 1
    assert status["eligible_count"] == 1
    assert status["pending_count"] == 107
    assert status["quota_reached"] is False


def test_selected_profiles_can_form_100_by_80_arrays() -> None:
    mass = np.geomspace(1.0e-6, 100.0, 80)
    temperature = np.linspace(2500.0, 6500.0, 80)
    masses = np.stack([mass] * (2 * TARGET_PER_CLASS))
    temperatures = np.stack([temperature] * (2 * TARGET_PER_CLASS))
    assert masses.shape == (100, 80)
    assert temperatures.shape == (100, 80)
