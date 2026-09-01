"""Tests for the matched v4r6 60-iteration policy."""

from __future__ import annotations

from experiments.analytic_initializer.textbook_opacity_v4r6_policy60 import (
    ABSOLUTE_COOL_MIN,
    ABSOLUTE_HOT_MIN,
    ABSOLUTE_TIMEOUTS_MAX,
    ABSOLUTE_TOTAL_MIN,
    ARM_OUTPUTS,
    COOL_NET_GAIN_VS_GREY_MIN,
    HOT_NET_LOSS_VS_GREY_MAX,
    ITERATIONS,
    PER_STAR_TIMEOUT_SECONDS,
    POLICY,
    TOTAL_NET_GAIN_VS_GREY_MIN,
    continuation_gate,
)


def _records(
    cool_successes: set[int],
    hot_successes: set[int],
    *,
    timeout_indices: set[int] | None = None,
) -> list[dict[str, object]]:
    timeout_indices = timeout_indices or set()
    records: list[dict[str, object]] = []
    for index in range(60):
        cool = index < 27
        converged = index in (cool_successes if cool else hot_successes)
        if converged:
            outcome = "converged"
        elif index in timeout_indices:
            outcome = "timeout"
        else:
            outcome = "not_converged"
        records.append(
            {
                "corpus_index": index,
                "effective_temperature": 5000.0 if cool else 9000.0,
                "log_surface_gravity": 4.0,
                "converged": converged,
                "solver_outcome": outcome,
            }
        )
    return records


def _payload(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "records": records,
        "initializer_provenance": {"finite_seed_count": 60},
    }


def test_policy60_identity_and_outputs_are_versioned() -> None:
    assert POLICY == "v4r6_analytic_warm_start_policy60_v1"
    assert ITERATIONS == 60
    assert PER_STAR_TIMEOUT_SECONDS == 900
    assert set(ARM_OUTPUTS) == {
        "textbook_v4r6_decoupled",
        "textbook_v4r6_grey",
        "textbook_v4r6",
    }
    assert len(set(ARM_OUTPUTS.values())) == 3
    assert all("policy60" in path.name for path in ARM_OUTPUTS.values())


def test_policy60_sequence_is_matched_and_sequential() -> None:
    from experiments.analytic_initializer.run_textbook_opacity_v4r6_policy60_sequence import (
        ARM_ORDER,
    )

    assert ARM_ORDER == (
        "textbook_v4r6_decoupled",
        "textbook_v4r6_grey",
        "textbook_v4r6",
    )


def test_policy60_continuation_thresholds_are_frozen() -> None:
    assert ABSOLUTE_TOTAL_MIN == 54
    assert ABSOLUTE_COOL_MIN == 23
    assert ABSOLUTE_HOT_MIN == 29
    assert ABSOLUTE_TIMEOUTS_MAX == 6
    assert TOTAL_NET_GAIN_VS_GREY_MIN == 0
    assert COOL_NET_GAIN_VS_GREY_MIN == 4
    assert HOT_NET_LOSS_VS_GREY_MAX == 2


def test_policy60_gate_passes_only_for_absolute_and_matched_success() -> None:
    # Grey: 19 cool + 31 hot = 50. Candidate: 23 cool + 31 hot = 54.
    grey = _payload(_records(set(range(19)), set(range(27, 58))))
    candidate = _payload(_records(set(range(23)), set(range(27, 58))))
    convective = _payload(_records(set(range(10)), set(range(27, 47))))
    result = continuation_gate(candidate, grey, convective)
    assert all(result["checks"].values())
    assert (
        result["decision"]
        == "CONTINUE_TO_POLICY60_FRESH_OPEN_PREREGISTRATION"
    )
    assert result["authorizes_fresh_open_execution"] is False


def test_policy60_gate_stops_when_grey_is_better() -> None:
    grey = _payload(_records(set(range(27)), set(range(27, 60))))
    candidate = _payload(_records(set(range(23)), set(range(27, 58))))
    convective = _payload(_records(set(range(10)), set(range(27, 47))))
    result = continuation_gate(candidate, grey, convective)
    assert result["checks"]["total_net_gain_vs_grey"] is False
    assert result["checks"]["cool_net_gain_vs_grey"] is False
    assert result["decision"] == "STOP_POLICY60_MATCHED_DEVELOPMENT"
