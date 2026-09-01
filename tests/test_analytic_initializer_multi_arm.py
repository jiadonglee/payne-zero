"""Tests for the multi-arm convergence comparison.

This is the code that will produce the headline convergence numbers, so the
statistics are pinned rather than trusted: the exact test, the interval, the
first-trial rule that keeps production's second trial out of the paired
comparison, and the power guard that stops a null result being read as
evidence of equivalence.
"""

from __future__ import annotations

import pytest

from experiments.analytic_initializer.run_arm_comparison import _exact_mcnemar
from experiments.analytic_initializer.run_multi_arm_comparison import (
    MIN_DISCORDANT_FOR_SIGNIFICANCE,
    _first_trial,
    _paired,
    _wilson,
)


# --- the exact test ---------------------------------------------------------


def test_no_disagreement_is_not_evidence() -> None:
    assert _exact_mcnemar(0, 0) == 1.0


def test_a_symmetric_split_is_not_evidence() -> None:
    assert _exact_mcnemar(3, 3) == 1.0


def test_the_observed_60_star_split_is_null() -> None:
    """3 versus 2 was once written up as the analytic arm winning. It is not."""

    assert _exact_mcnemar(3, 2) == pytest.approx(1.0)


def test_six_same_direction_pairs_is_the_threshold() -> None:
    """The constant the power guard is built on, checked rather than asserted."""

    assert _exact_mcnemar(5, 0) > 0.05
    assert _exact_mcnemar(6, 0) < 0.05
    assert _exact_mcnemar(MIN_DISCORDANT_FOR_SIGNIFICANCE, 0) < 0.05


def test_the_test_is_two_sided() -> None:
    assert _exact_mcnemar(6, 0) == _exact_mcnemar(0, 6)


# --- the interval -----------------------------------------------------------


def test_wilson_stays_inside_the_unit_interval_at_a_clean_sweep() -> None:
    """Why Wilson and not the normal approximation: 60/60 must not exceed 1."""

    low, high = _wilson(60, 60)
    assert high == pytest.approx(1.0)
    assert high <= 1.0 + 1.0e-12
    # The paper reports 60/60 as ">=94% (Wilson, 95%)"; this is that number.
    assert low == pytest.approx(0.94, abs=0.01)


def test_wilson_brackets_the_point_estimate() -> None:
    low, high = _wilson(55, 60)
    assert low < 55 / 60 < high


def test_wilson_on_no_data_is_uninformative() -> None:
    assert _wilson(0, 0) == (0.0, 1.0)


# --- the first-trial rule ---------------------------------------------------


def test_production_is_scored_on_its_first_trial_only() -> None:
    """Production gets two trials; the formula arms get one. Only trial one
    is comparable, and a rescue on trial two must not count here."""

    rescued = {"converged": True, "first_trial_converged": False, "trials_used": 2}
    assert _first_trial(rescued) is False


def test_a_formula_arm_without_the_field_falls_back_to_convergence() -> None:
    assert _first_trial({"converged": True}) is True
    assert _first_trial({"converged": False}) is False


# --- the paired table -------------------------------------------------------


def _arm(**outcomes) -> dict[int, dict]:
    return {index: {"converged": value} for index, value in outcomes.items()}


def test_the_paired_table_counts_only_disagreements_as_evidence() -> None:
    left = _arm(**{"1": True, "2": True, "3": False, "4": True})
    right = _arm(**{"1": True, "2": False, "3": False, "4": False})
    left = {int(k): v for k, v in left.items()}
    right = {int(k): v for k, v in right.items()}
    result = _paired(left, right, [1, 2, 3, 4])
    assert result["both"] == 1
    assert result["left_only"] == 2
    assert result["right_only"] == 0
    assert result["neither"] == 1
    assert result["discordant"] == 2
    assert result["can_reach_significance"] is False


def test_the_power_guard_flips_only_with_enough_disagreement() -> None:
    shared = list(range(MIN_DISCORDANT_FOR_SIGNIFICANCE))
    left = {index: {"converged": True} for index in shared}
    right = {index: {"converged": False} for index in shared}
    result = _paired(left, right, shared)
    assert result["discordant"] == MIN_DISCORDANT_FOR_SIGNIFICANCE
    assert result["can_reach_significance"] is True
    assert result["exact_mcnemar_p"] < 0.05

    fewer = shared[:-1]
    result = _paired(left, right, fewer)
    assert result["can_reach_significance"] is False
    assert result["exact_mcnemar_p"] > 0.05


# --- the solve dispatch -----------------------------------------------------
#
# The bug this section exists for: ``_solve_payload`` dispatched on
# ``payload["arm"] == "analytic"``, so three formula arms added later fell
# through to the emulator.  Nothing looked wrong -- the emulator converges --
# and the only visible symptom was ``trials_used`` coming back as 2 from arms
# allocated exactly one trial.


def test_a_supplied_reduced_state_routes_to_the_formula_path(monkeypatch) -> None:
    import numpy as np

    from experiments.analytic_initializer import run_h2_solver_funnel as funnel

    called = {}
    monkeypatch.setattr(
        funnel, "_solve_analytic",
        lambda *a, **k: called.setdefault("path", "analytic") or {"converged": True},
    )
    monkeypatch.setattr(
        funnel, "_solve_production",
        lambda *a, **k: called.setdefault("path", "production") or {"converged": True},
    )
    for arm in (
        "analytic",
        "compact600",
        "parity",
        "physical",
        "textbook_v4r3",
        "textbook_v4r6",
        "textbook_v4r6_grey",
        "textbook_v4r6_decoupled",
    ):
        called.clear()
        funnel._solve_payload(
            {
                "arm": arm,
                "labels": np.zeros(5),
                "mass": np.ones(3),
                "temperature": np.ones(3),
                "log_opacity": np.zeros(3),
                "tau": np.ones(3),
            }
        )
        assert called["path"] == "analytic", f"{arm} went down the emulator path"


def test_payload_iterations_reach_the_formula_solver(monkeypatch) -> None:
    import numpy as np

    from experiments.analytic_initializer import run_h2_solver_funnel as funnel

    seen: dict[str, int] = {}

    def fake_analytic(*_args, **kwargs):
        seen["iterations"] = int(kwargs["iterations_per_trial"])
        return {"converged": True}

    monkeypatch.setattr(funnel, "_solve_analytic", fake_analytic)
    monkeypatch.setattr(
        funnel,
        "_solve_production",
        lambda *_a, **_k: {"converged": True},
    )
    payload = {
        "arm": "textbook_v4r6_decoupled",
        "labels": np.zeros(5),
        "mass": np.ones(3),
        "temperature": np.ones(3),
        "log_opacity": np.zeros(3),
        "tau": np.ones(3),
        "iterations_per_trial": 60,
    }
    funnel._solve_payload(payload)
    assert seen["iterations"] == 60
    seen.clear()
    del payload["iterations_per_trial"]
    funnel._solve_payload(payload)
    assert seen["iterations"] == 15


def test_no_reduced_state_routes_to_the_emulator() -> None:
    import numpy as np

    from experiments.analytic_initializer import run_h2_solver_funnel as funnel

    import unittest.mock as mock

    with mock.patch.object(funnel, "_solve_production", return_value={"converged": True}) as prod, \
         mock.patch.object(funnel, "_solve_analytic") as analytic:
        funnel._solve_payload(
            {"arm": "production", "labels": np.zeros(5), "mass": None,
             "temperature": None, "log_opacity": None, "tau": np.ones(3)}
        )
    assert prod.called and not analytic.called
