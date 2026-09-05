import pytest

from experiments.reduced_state_emulator.cool_star_adaptive import (
    backtrack_step,
    proposed_temperature,
)
from experiments.reduced_state_emulator.m_star_atlas_continuation_opened_tracks_v1 import (
    GRID_TEMPERATURES,
    INITIAL_STEP_K,
    MINIMUM_STEP_K,
    PROBES,
    SEEDS,
    WALK_TARGETS,
    nominal_steps,
    probe_candidate_id,
    probe_walk_decision,
    validate_walk_decision,
)


def test_initial_step_is_50_and_backtrack_floors_at_25() -> None:
    assert INITIAL_STEP_K == 50.0
    assert MINIMUM_STEP_K == 25.0
    assert backtrack_step(INITIAL_STEP_K, MINIMUM_STEP_K) == 25.0
    assert backtrack_step(MINIMUM_STEP_K, MINIMUM_STEP_K) == 25.0


def test_steps_never_overshoot_the_target() -> None:
    assert nominal_steps(3500.0, 3400.0, 50.0) == [3450.0, 3400.0]
    assert nominal_steps(3700.0, 3600.0, 50.0) == [3650.0, 3600.0]
    assert nominal_steps(3400.0, 3200.0, 50.0) == [
        3350.0,
        3300.0,
        3250.0,
        3200.0,
    ]
    assert nominal_steps(3700.0, 3000.0, 50.0)[-1] == 3000.0
    waypoints = nominal_steps(3500.0, 3000.0, 50.0)
    assert all(waypoint > 3000.0 for waypoint in waypoints[:-1])


def test_backtracked_retry_approaches_in_25k_from_last_accept() -> None:
    current, target = 3700.0, 3600.0
    step = INITIAL_STEP_K
    first = proposed_temperature(current, target, step)
    assert first == 3650.0
    step = backtrack_step(step, MINIMUM_STEP_K)
    retries = []
    while abs(current - target) > 1.0e-9:
        current = proposed_temperature(current, target, step)
        retries.append(current)
    assert retries == [3675.0, 3650.0, 3625.0, 3600.0]
    assert min(retries) >= target


def test_probe_targets_are_the_frozen_v1r2_nodes() -> None:
    assert probe_candidate_id("A") == (
        "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3400"
    )
    assert probe_candidate_id("B") == (
        "g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3600"
    )
    assert PROBES["A"]["seed"] == "probe_a"
    assert SEEDS["probe_a"]["temperature_K"] == 3500.0
    assert SEEDS["probe_b"]["temperature_K"] == 3700.0
    for targets in WALK_TARGETS.values():
        assert all(value in GRID_TEMPERATURES for value in targets)


def test_walk_targets_and_chain_seeds_stay_on_their_track() -> None:
    from experiments.reduced_state_emulator.m_star_atlas_continuation_opened_tracks_v1 import (
        candidate_id,
    )

    assert PROBES["A"]["track"]["metallicity"] == 0.0
    assert PROBES["B"]["track"]["metallicity"] == -0.5
    track_a_ids = {
        candidate_id(PROBES["A"]["track"], value) for value in WALK_TARGETS["A"]
    }
    track_b_ids = {
        candidate_id(PROBES["B"]["track"], value) for value in WALK_TARGETS["B"]
    }
    assert track_a_ids & track_b_ids == set()
    # the chain seed of every walk cell is the next warmer cell of the same
    # track, so the chain cools monotonically from the probe target
    for name in ("A", "B"):
        chain = [
            probe_candidate_id(name),
            *(candidate_id(PROBES[name]["track"], value) for value in WALK_TARGETS[name]),
        ]
        temperatures = [float(item.split("_t")[-1]) for item in chain]
        assert temperatures == sorted(temperatures, reverse=True)


def test_probe_decision_walks_both_single_or_stops() -> None:
    both = probe_walk_decision({"A": True, "B": True})
    assert both["decision"] == "walk_both"
    assert both["tracks"]["A"]["targets_K"] == [3200.0, 3100.0, 3000.0]
    assert both["tracks"]["B"]["targets_K"] == [
        3500.0,
        3400.0,
        3300.0,
        3200.0,
        3100.0,
        3000.0,
    ]
    single = probe_walk_decision({"A": False, "B": True})
    assert single["decision"] == "walk_single"
    assert single["tracks"]["A"]["opened"] is False
    assert single["tracks"]["A"]["targets_K"] == []
    assert single["tracks"]["B"]["opened"] is True
    stop = probe_walk_decision({"A": False, "B": False})
    assert stop["decision"] == "stop"
    assert all(not track["opened"] for track in stop["tracks"].values())


def _decision() -> dict:
    decision = probe_walk_decision({"A": True, "B": False})
    decision["protocol_hash"] = "abc123"
    return decision


def _protocol() -> dict:
    return {"protocol_hash": "abc123"}


def test_validate_walk_decision_accepts_the_frozen_plan() -> None:
    validate_walk_decision(_decision(), protocol=_protocol())


def test_validate_walk_decision_rejects_drift() -> None:
    decision = _decision()
    decision["campaign"] = "other"
    with pytest.raises(ValueError):
        validate_walk_decision(decision, protocol=_protocol())

    decision = _decision()
    decision["protocol_hash"] = "other"
    with pytest.raises(ValueError):
        validate_walk_decision(decision, protocol=_protocol())

    decision = _decision()
    decision["decision"] = "walk_both"
    with pytest.raises(ValueError):
        validate_walk_decision(decision, protocol=_protocol())

    decision = _decision()
    decision["tracks"]["A"]["targets_K"] = [3200.0, 3100.0, 3000.0, 2900.0]
    with pytest.raises(ValueError):
        validate_walk_decision(decision, protocol=_protocol())

    decision = _decision()
    decision["tracks"]["B"]["opened"] = True
    with pytest.raises(ValueError):
        validate_walk_decision(decision, protocol=_protocol())

    decision = _decision()
    decision["tracks"]["A"]["probe_eligible"] = False
    with pytest.raises(ValueError):
        validate_walk_decision(decision, protocol=_protocol())
