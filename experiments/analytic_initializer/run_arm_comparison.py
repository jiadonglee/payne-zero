"""Pair the analytic and production funnel arms and settle the attribution.

A one-armed funnel cannot tell an initializer defect from a solver hard region.
``runs_baseline.log`` records the production path itself failing on warm
mid-gravity and hot low-gravity stars, and the four-initializer benchmark puts
production at 192/200 -- so a 37/40 analytic result is not on its own evidence
of anything.  This probe compares the two arms star by star on the same draw
and the same 15-iteration single-trial policy, and reports the paired test
rather than two independent rates.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Stars the first funnel attempt singled out: three finite non-convergences and
# the row that hung.  Gate A asks whether production also struggles on them.
KNOWN_PROBLEM_INDICES = (11206, 13265, 33356, 34042)


def _load_records(paths: list[Path]) -> dict[int, dict]:
    """Merge one arm's records, which may be split across several runs.

    A funnel stopped partway and resumed on the remaining stars leaves more
    than one file per arm; a star solved twice keeps the later record.
    """

    by_index: dict[int, dict] = {}
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            # A run that was interrupted never wrote its summary, so the
            # streamed rows are the only record of it.  Reading them directly is
            # the point of streaming them.
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            records = json.loads(text)["records"]
        seen = {int(item["corpus_index"]) for item in records}
        if len(seen) != len(records):
            raise ValueError(f"{path} contains duplicate corpus indices")
        for item in records:
            by_index[int(item["corpus_index"])] = item
    return by_index


def _exact_mcnemar(first_only: int, second_only: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs.

    The arms solve the same stars, so the concordant pairs carry no information
    about which arm is better; only the disagreements do.
    """

    total = first_only + second_only
    if total == 0:
        return 1.0
    smaller = min(first_only, second_only)
    tail = sum(math.comb(total, k) for k in range(smaller + 1)) / (2.0**total)
    return float(min(1.0, 2.0 * tail))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analytic",
        type=Path,
        nargs="+",
        default=[Path("results/analytic_initializer/h2_solver_funnel60.json")],
    )
    parser.add_argument(
        "--production",
        type=Path,
        nargs="+",
        default=[Path("results/analytic_initializer/production_control_arm60.json")],
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help=(
            "compare exactly these stars; both arms must cover every one of "
            "them. Defaults to requiring the two arms to cover the same set."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/analytic_initializer/arm_comparison.json"),
    )
    args = parser.parse_args(argv)

    analytic = _load_records(args.analytic)
    production = _load_records(args.production)
    if args.indices is not None:
        wanted = set(args.indices)
        for name, arm in (("analytic", analytic), ("production", production)):
            uncovered = sorted(wanted - set(arm))
            if uncovered:
                raise ValueError(
                    f"the {name} arm is missing requested stars: {uncovered}"
                )
    else:
        wanted = set(analytic)
        if set(analytic) != set(production):
            difference = sorted(set(analytic) ^ set(production))
            raise ValueError(
                "the two arms did not solve the same stars; symmetric "
                f"difference: {difference}. Pass --indices to compare a subset "
                "on purpose."
            )

    rows = []
    for index in sorted(wanted):
        left = analytic[index]
        right = production[index]
        rows.append(
            {
                "corpus_index": index,
                "effective_temperature": left["effective_temperature"],
                "log_surface_gravity": left["log_surface_gravity"],
                "metallicity": left["metallicity"],
                "analytic_outcome": left["solver_outcome"],
                "analytic_converged": bool(left["converged"]),
                "analytic_iterations": left["iterations_completed"],
                "production_outcome": right["solver_outcome"],
                "production_first_trial_converged": bool(right["first_trial_converged"]),
                "production_first_trial_iterations": right.get("first_trial_iterations"),
                "production_converged_with_retry": bool(right["converged"]),
                "production_iterations": right["iterations_completed"],
            }
        )

    analytic_ok = [row["analytic_converged"] for row in rows]
    production_ok = [row["production_first_trial_converged"] for row in rows]
    analytic_only = sum(1 for a, p in zip(analytic_ok, production_ok) if a and not p)
    production_only = sum(1 for a, p in zip(analytic_ok, production_ok) if p and not a)

    both = [
        row
        for row in rows
        if row["analytic_converged"] and row["production_first_trial_converged"]
    ]
    iteration_delta = [
        int(row["analytic_iterations"]) - int(row["production_first_trial_iterations"])
        for row in both
        if row["analytic_iterations"] is not None
        and row["production_first_trial_iterations"] is not None
    ]

    problem_rows = [row for row in rows if row["corpus_index"] in KNOWN_PROBLEM_INDICES]
    production_also_fails = [
        row["corpus_index"]
        for row in problem_rows
        if not row["production_first_trial_converged"]
    ]

    gate = {
        "known_problem_indices": list(KNOWN_PROBLEM_INDICES),
        "production_first_trial_also_fails": sorted(production_also_fails),
        "production_first_trial_also_fails_count": len(production_also_fails),
        "threshold_count": 2,
        # Two or more shared failures means the hard tail is a property of the
        # solver in that region, not of the analytic seed, and the reliability
        # argument for a closure rewrite would rest on nothing.
        "hard_tail_is_solver_side": bool(len(production_also_fails) >= 2),
    }

    result = {
        "gate_a": gate,
        "format": "payne_zero_analytic_initializer_arm_comparison_v1",
        "status": "paired_real_solver_comparison",
        "policy": {
            "iterations_per_trial": 15,
            "note": (
                "The paired comparison uses the production arm's first trial, "
                "which matches the analytic arm's single-trial budget. The "
                "two-trial rate is reported alongside because the "
                "four-initializer benchmark is defined against it."
            ),
        },
        "sources": {
            "analytic": [str(path) for path in args.analytic],
            "production": [str(path) for path in args.production],
        },
        "comparison_scope": (
            "requested_subset" if args.indices is not None else "full_paired_funnel"
        ),
        "star_count": len(rows),
        "analytic_converged_count": sum(analytic_ok),
        "production_first_trial_converged_count": sum(production_ok),
        "production_with_retry_converged_count": sum(
            bool(row["production_converged_with_retry"]) for row in rows
        ),
        "paired_test": {
            "analytic_only_converged": analytic_only,
            "production_only_converged": production_only,
            "exact_mcnemar_two_sided_p": _exact_mcnemar(analytic_only, production_only),
            "note": (
                "Meaningful only over an unbiased draw. On a subset chosen "
                "because the analytic arm failed there, this test is "
                "conditioned on the outcome and must not be read as a rate "
                "comparison."
            ),
        },
        "iterations_where_both_converged": {
            "star_count": len(iteration_delta),
            "analytic_minus_production_mean": (
                float(sum(iteration_delta) / len(iteration_delta))
                if iteration_delta
                else None
            ),
            "analytic_slower_count": sum(1 for value in iteration_delta if value > 0),
            "analytic_faster_count": sum(1 for value in iteration_delta if value < 0),
            "tied_count": sum(1 for value in iteration_delta if value == 0),
        },
        "known_problem_stars": problem_rows,
        "records": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"analytic {result['analytic_converged_count']}/{result['star_count']}  "
        f"production 1-trial {result['production_first_trial_converged_count']}"
        f"/{result['star_count']}  "
        f"production 2-trial {result['production_with_retry_converged_count']}"
        f"/{result['star_count']}"
    )
    print(json.dumps(result["paired_test"], indent=2, sort_keys=True))
    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
