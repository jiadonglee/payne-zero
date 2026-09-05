from experiments.reduced_state_emulator.m_star_bootstrap_v1r3_dwarf_iter120 import (
    ITERATION_CAP,
    TARGET_NEW_DWARFS,
    _select_rescue_candidates,
)


def _candidate(candidate_id: str, priority: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "priority": priority,
        "class": "dwarf",
    }


def _record(
    candidate_id: str,
    *,
    iterations: int,
    valid: bool,
    eligible: bool = False,
    failure_reason: str = "primary_solver,self_restart",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "class": "dwarf",
        "training_eligible": eligible,
        "failure_reason": failure_reason,
        "primary": {
            "iterations": iterations,
            "state_quality": {"valid": valid},
        },
    }


def test_rescue_changes_only_the_iteration_budget() -> None:
    assert ITERATION_CAP == 120
    assert TARGET_NEW_DWARFS == 24


def test_rescue_selects_only_cap_limited_valid_failures_in_priority_order() -> None:
    protocol = {
        "split": {
            "new_train_candidates": [
                _candidate("late", 8),
                _candidate("first", 2),
                _candidate("early_fail", 3),
                _candidate("bad_state", 4),
                _candidate("flux_only", 5),
                _candidate("already_good", 6),
            ]
        }
    }
    records = [
        _record("late", iterations=60, valid=True),
        _record("first", iterations=60, valid=True),
        _record("early_fail", iterations=40, valid=True),
        _record("bad_state", iterations=60, valid=False),
        _record(
            "flux_only",
            iterations=20,
            valid=True,
            failure_reason="primary_flux_gate",
        ),
        _record(
            "already_good",
            iterations=20,
            valid=True,
            eligible=True,
            failure_reason="",
        ),
    ]
    selected = _select_rescue_candidates(protocol, records)
    assert [row["candidate_id"] for row in selected] == ["first", "late"]
