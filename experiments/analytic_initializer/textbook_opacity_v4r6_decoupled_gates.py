"""Frozen gates for the v4r6 decoupled grey-mass / convective-T candidate.

These thresholds are copied from
``notes/textbook_opacity_v4r6_decoupled_mgrey_tconv_workplan_20260828.md``.
They are not to be edited after a solver result is seen.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

CANDIDATE = "v4r6_decoupled_mgrey_tconv_v1"
TEFF_SPLIT_K = 7500.0
LOGG_SPLIT = 3.5

DEVELOPMENT_COMPLETE_RECORDS = 60
DEVELOPMENT_SOLVER_ERRORS = 0
DEVELOPMENT_FINITE_SEEDS = 60
DEVELOPMENT_COOL_CONVERGED_MIN = 11
DEVELOPMENT_COOL_STAR_COUNT = 27
DEVELOPMENT_HOT_CONVERGED_MIN = 30
DEVELOPMENT_HOT_STAR_COUNT = 33
DEVELOPMENT_TOTAL_CONVERGED_MIN = 41
DEVELOPMENT_LOSSES_AMONG_GREY_MAX = 2
DEVELOPMENT_NET_GAIN_MIN = 4
DEVELOPMENT_TIMEOUTS_MAX = 3

FAIL_STOP_STRUCTURAL = "FAIL_STOP_STRUCTURAL"
FAIL_STOP_DEVELOPMENT = "FAIL_STOP_DEVELOPMENT"
FAIL_STOP_FRESH_OPEN = "FAIL_STOP_FRESH_OPEN"
INCONCLUSIVE_RUNTIME = "INCONCLUSIVE_RUNTIME"
PASS_TO_FRESH_OPEN = "PASS_TO_FRESH_OPEN"
PASS_TO_COUPLED_ODE_PREREGISTRATION = "PASS_TO_COUPLED_ODE_PREREGISTRATION"


def records_by_index(records: Sequence[Mapping[str, object]]) -> dict[int, Mapping[str, object]]:
    by_index: dict[int, Mapping[str, object]] = {}
    for record in records:
        index = int(record["corpus_index"])
        if index in by_index:
            raise ValueError(f"duplicate corpus index {index}")
        by_index[index] = record
    return by_index


def split_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    cool = [item for item in records if float(item["effective_temperature"]) < TEFF_SPLIT_K]
    hot = [item for item in records if float(item["effective_temperature"]) >= TEFF_SPLIT_K]
    dwarf = [item for item in records if float(item["log_surface_gravity"]) >= LOGG_SPLIT]
    giant = [item for item in records if float(item["log_surface_gravity"]) < LOGG_SPLIT]
    return {
        "all": list(records),
        "cool": cool,
        "hot": hot,
        "dwarf": dwarf,
        "giant": giant,
    }


def outcome_counts(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    outcomes = [str(item.get("solver_outcome", "")) for item in records]
    return {
        "star_count": len(records),
        "converged_count": int(sum(bool(item.get("converged")) for item in records)),
        "timeout_count": outcomes.count("timeout"),
        "error_count": outcomes.count("error"),
        "not_converged_count": outcomes.count("not_converged"),
    }


def paired_solver_counts(
    candidate_records: Sequence[Mapping[str, object]],
    control_records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    candidate = records_by_index(candidate_records)
    control = records_by_index(control_records)
    shared = sorted(set(candidate) & set(control))
    if len(shared) != len(candidate) or len(shared) != len(control):
        raise ValueError("paired arms do not share the same corpus indices")
    both = candidate_only = control_only = neither = 0
    losses_among_control_successes = 0
    control_success_count = 0
    for index in shared:
        won = bool(candidate[index].get("converged"))
        control_won = bool(control[index].get("converged"))
        both += won and control_won
        candidate_only += won and not control_won
        control_only += control_won and not won
        neither += (not won) and (not control_won)
        if control_won:
            control_success_count += 1
            if not won:
                losses_among_control_successes += 1
    return {
        "shared_count": len(shared),
        "both_success": both,
        "candidate_only": candidate_only,
        "control_only": control_only,
        "neither": neither,
        "net_gain": candidate_only - control_only,
        "losses_among_control_successes": losses_among_control_successes,
        "control_success_count": control_success_count,
    }


def exact_mcnemar_one_sided(wins: int, losses: int) -> float:
    """One-sided exact McNemar: P(X >= wins) under Binomial(wins+losses, 1/2)."""

    total = int(wins) + int(losses)
    if total == 0:
        return 1.0
    return float(
        sum(math.comb(total, k) for k in range(int(wins), total + 1)) / (2.0**total)
    )


def development_gate(
    candidate: Mapping[str, object],
    grey_control: Mapping[str, object],
) -> dict[str, object]:
    """Decide whether the exposed development-60 may continue to fresh-open."""

    records = list(candidate["records"])  # type: ignore[arg-type]
    grey_records = list(grey_control["records"])  # type: ignore[arg-type]
    splits = split_records(records)
    counts = outcome_counts(records)
    cool = outcome_counts(splits["cool"])
    hot = outcome_counts(splits["hot"])
    provenance = dict(candidate.get("initializer_provenance") or {})
    finite_seeds = int(provenance.get("finite_seed_count", -1))
    paired = paired_solver_counts(records, grey_records)

    runtime_ok = (
        counts["star_count"] == DEVELOPMENT_COMPLETE_RECORDS
        and counts["error_count"] == DEVELOPMENT_SOLVER_ERRORS
        and len(grey_records) == DEVELOPMENT_COMPLETE_RECORDS
    )
    checks = {
        "complete_records": counts["star_count"] == DEVELOPMENT_COMPLETE_RECORDS,
        "solver_errors": counts["error_count"] == DEVELOPMENT_SOLVER_ERRORS,
        "finite_seeds": finite_seeds == DEVELOPMENT_FINITE_SEEDS,
        "cool_convergence": cool["converged_count"] >= DEVELOPMENT_COOL_CONVERGED_MIN,
        "hot_convergence": hot["converged_count"] >= DEVELOPMENT_HOT_CONVERGED_MIN,
        "total_convergence": counts["converged_count"] >= DEVELOPMENT_TOTAL_CONVERGED_MIN,
        "losses_among_grey_converged": (
            int(paired["losses_among_control_successes"]) <= DEVELOPMENT_LOSSES_AMONG_GREY_MAX
        ),
        "net_paired_gain": int(paired["net_gain"]) >= DEVELOPMENT_NET_GAIN_MIN,
        "timeouts": counts["timeout_count"] <= DEVELOPMENT_TIMEOUTS_MAX,
    }
    if not runtime_ok:
        decision = INCONCLUSIVE_RUNTIME
    elif all(checks.values()):
        decision = PASS_TO_FRESH_OPEN
    else:
        decision = FAIL_STOP_DEVELOPMENT
    return {
        "decision": decision,
        "checks": checks,
        "counts": counts,
        "cool": cool,
        "hot": hot,
        "paired_vs_grey": paired,
        "finite_seed_count": finite_seeds,
        "thresholds": {
            "cool_converged_min": DEVELOPMENT_COOL_CONVERGED_MIN,
            "hot_converged_min": DEVELOPMENT_HOT_CONVERGED_MIN,
            "total_converged_min": DEVELOPMENT_TOTAL_CONVERGED_MIN,
            "losses_among_grey_max": DEVELOPMENT_LOSSES_AMONG_GREY_MAX,
            "net_gain_min": DEVELOPMENT_NET_GAIN_MIN,
            "timeouts_max": DEVELOPMENT_TIMEOUTS_MAX,
        },
    }
