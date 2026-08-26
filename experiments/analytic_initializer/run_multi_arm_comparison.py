"""Does the solver care about the offline accuracy differences between arms?

Three structural rewrites of the emulator-free warm start now exist, and offline
they span a factor of 2.7 in held-out temperature error at stored-constant
counts from 589 to 3851.  Every one of them is physically well formed and grid
independent; none of that is worth anything unless the solver behaves.  This
probe answers the only question offline work cannot: whether the accuracy
spread shows up as a convergence spread.

The comparison is paired.  Every formula arm is seeded from the same drawn
stars, runs the same single 15-iteration trial, and differs only in which
constants produced ``(m, T)``.  Production is included as the incumbent and is
allowed its two trials, so it is reported both ways: its first-trial result is
the like-for-like number, and its full result is the deployed baseline.

**On what this can and cannot conclude.**  Sixty stars cannot reach
significance even in principle.  The exact McNemar test needs at least six
same-direction discordant pairs to clear p < 0.05, and at the roughly 4.5
percent discordance these arms show, sixty paired stars expect 2.7.  Eighty
percent power would need about 290.  So a difference here is reportable as
"not worse", never as "better", and the numbers below are printed with their
discordant-pair counts so that is checkable rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.analytic_initializer.run_arm_comparison import (
    _exact_mcnemar,
    _load_records,
)

try:  # scipy is present in .venv; the probe degrades rather than dies without it
    from scipy.stats import wilcoxon as _wilcoxon
except ImportError:  # pragma: no cover
    _wilcoxon = None

DEFAULT_OUTPUT = Path("results/analytic_initializer/multi_arm_comparison.json")

#: Offline held-out numbers for each arm, from the probes that produced them.
#: Carried here so the convergence result can be read against what it was
#: supposed to be sensitive to.
OFFLINE = {
    "analytic": {"stored_floats": 4580, "temperature_p95": 0.0197, "mass_p95": 0.0869,
                 "source": "tabulated H2, results/analytic_initializer/monotone_invariants.json"},
    "compact600": {"stored_floats": 589, "temperature_p95": 0.0389, "mass_p95": 0.1698,
                   "source": "results/analytic_initializer/compact_frontier.json"},
    "parity": {"stored_floats": 2407, "temperature_p95": 0.0201, "mass_p95": 0.0870,
               "source": "results/analytic_initializer/compact_frontier.json"},
    "physical": {"stored_floats": 3851, "temperature_p95": 0.0146, "mass_p95": 0.0597,
                 "source": "results/analytic_initializer/label_map_probe.json"},
    "production": {"stored_floats": None, "temperature_p95": None, "mass_p95": None,
                   "source": "the deployed emulator"},
}

DEFAULT_ARMS = {
    "analytic": [
        Path("results/analytic_initializer/h2_solver_funnel60.jsonl"),
        Path("results/analytic_initializer/h2_solver_funnel60_analytic_rest.json"),
    ],
    "production": [Path("results/analytic_initializer/h2_solver_funnel60_production.json")],
    "compact600": [Path("results/analytic_initializer/funnel60_compact600.json")],
    "parity": [Path("results/analytic_initializer/funnel60_parity.json")],
    "physical": [Path("results/analytic_initializer/funnel60_physical.json")],
}

#: Minimum same-direction discordant pairs for the exact test to be able to
#: reach p < 0.05 at all.  Below this, a null result carries no information.
MIN_DISCORDANT_FOR_SIGNIFICANCE = 6


def _wilson(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval, which behaves at the top of the range."""

    if total == 0:
        return (0.0, 1.0)
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2.0 * total)) / denominator
    spread = (
        z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total * total))
    ) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _first_trial(record: dict) -> bool:
    """Converged on the first solver trial.

    Formula arms are allocated exactly one trial, so for them this is just
    convergence.  Production keeps every trial, which is what makes the
    like-for-like comparison possible at all.
    """

    if "first_trial_converged" in record:
        return bool(record["first_trial_converged"])
    return bool(record.get("converged", False))


def _iterations(record: dict):
    """First-trial iteration count, which is the comparable one."""

    return record.get("first_trial_iterations") or record.get("iterations_completed")


def _paired_iterations(left: dict[int, dict], right: dict[int, dict], shared: list[int]) -> dict:
    """Compare iteration counts on the stars both arms converged.

    The binary outcome cannot reach significance at sixty stars, but the
    iteration count can: it uses every concordant star rather than only the
    handful of disagreements, so the same run supports a real test here where
    it supports none above.  Wilcoxon signed-rank rather than a t-test, because
    these are small skewed integers.
    """

    both = [
        index for index in shared if _first_trial(left[index]) and _first_trial(right[index])
    ]
    a = [_iterations(left[index]) for index in both]
    b = [_iterations(right[index]) for index in both]
    if not both or any(value is None for value in a + b):
        return {"stars": len(both), "comparable": False}
    difference = [x - y for x, y in zip(a, b)]
    ordered_a, ordered_b = sorted(a), sorted(b)
    result = {
        "stars": len(both),
        "comparable": True,
        "median_left": ordered_a[len(a) // 2],
        "median_right": ordered_b[len(b) // 2],
        "mean_difference": sum(difference) / len(difference),
        "left_faster_stars": sum(1 for d in difference if d < 0),
        "right_faster_stars": sum(1 for d in difference if d > 0),
        "tied_stars": sum(1 for d in difference if d == 0),
    }
    if _wilcoxon is not None and any(d != 0 for d in difference):
        result["wilcoxon_signed_rank_p"] = float(_wilcoxon(a, b).pvalue)
    return result


def _paired(left: dict[int, dict], right: dict[int, dict], shared: list[int]) -> dict:
    both = left_only = right_only = neither = 0
    for index in shared:
        a, b = _first_trial(left[index]), _first_trial(right[index])
        both += a and b
        left_only += a and not b
        right_only += b and not a
        neither += not a and not b
    discordant = left_only + right_only
    return {
        "both": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": neither,
        "discordant": discordant,
        "exact_mcnemar_p": _exact_mcnemar(left_only, right_only),
        "can_reach_significance": discordant >= MIN_DISCORDANT_FOR_SIGNIFICANCE,
    }


def _projection_touched(indices) -> set[int]:
    """Stars where the tabulated H2 profile needs monotone repair.

    These confound any comparison against the ``analytic`` arm, which is the
    original H2 with no guards: every other formula arm applies the monotone
    projection, so on these stars the arms differ by the projection as well as
    by whatever is under test.  Three of the sixty funnel stars qualify.
    """

    import numpy as np

    from experiments.analytic_initializer.discovery import DEFAULT_CORPUS, load_strict_truth
    from experiments.analytic_initializer.profile_initializer import (
        load_analytic_profile_parameters,
        predict_analytic_reduced_state,
    )

    asset = Path("results/analytic_initializer/h2_profile_parameters_v1.npz")
    if not asset.is_file():
        return set()
    corpus = load_strict_truth(DEFAULT_CORPUS)
    rows = sorted(indices)
    _, temperature, _ = predict_analytic_reduced_state(
        corpus.labels[rows], corpus.tau, load_analytic_profile_parameters(asset)
    )
    broken = ~np.all(np.diff(temperature, axis=1) > 0.0, axis=1)
    return {rows[i] for i in range(len(rows)) if broken[i]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    arms: dict[str, dict[int, dict]] = {}
    missing: list[str] = []
    for name, paths in DEFAULT_ARMS.items():
        present = [path for path in paths if path.is_file()]
        if not present:
            missing.append(name)
            continue
        arms[name] = _load_records(present)
    if missing:
        print(f"missing arms (skipped): {', '.join(missing)}")

    shared = sorted(set.intersection(*(set(records) for records in arms.values())))
    if not shared:
        raise SystemExit("the arms share no stars")

    summary = {}
    for name, records in arms.items():
        first = [index for index in shared if _first_trial(records[index])]
        full = [
            index for index in shared if bool(records[index].get("converged", False))
        ]
        iterations = [
            int(records[index]["iterations_completed"])
            for index in first
            if records[index].get("iterations_completed") is not None
        ]
        low, high = _wilson(len(first), len(shared))
        summary[name] = {
            **OFFLINE.get(name, {}),
            "stars": len(shared),
            "first_trial_converged": len(first),
            "converged_any_trial": len(full),
            "first_trial_rate": len(first) / len(shared),
            "first_trial_wilson95": [low, high],
            "median_iterations_on_first_trial": (
                sorted(iterations)[len(iterations) // 2] if iterations else None
            ),
            "mean_iterations_on_first_trial": (
                sum(iterations) / len(iterations) if iterations else None
            ),
            "failed_indices": [index for index in shared if index not in first],
        }

    comparisons = {}
    for name in arms:
        if name == "production":
            continue
        for reference in ("production", "analytic", "parity"):
            if reference not in arms or reference == name:
                continue
            comparisons[f"{name}_vs_{reference}"] = {
                "convergence": _paired(arms[name], arms[reference], shared),
                "iterations": _paired_iterations(arms[name], arms[reference], shared),
            }

    touched = _projection_touched(shared)
    sensitivity = {}
    if touched and "analytic" in arms:
        clean = [index for index in shared if index not in touched]
        for name in arms:
            if name in ("analytic", "production"):
                continue
            sensitivity[f"{name}_vs_analytic"] = {
                "all_stars": _paired_iterations(arms[name], arms["analytic"], shared),
                "projection_touched_removed": _paired_iterations(
                    arms[name], arms["analytic"], clean
                ),
            }

    shared_failures = [
        index
        for index in shared
        if not any(_first_trial(records[index]) for records in arms.values())
    ]

    payload = {
        "format": "payne_zero_multi_arm_comparison_v1",
        "date": "2026-08-17",
        "question": (
            "Three structural rewrites of the emulator-free warm start span a "
            "factor of 2.7 in offline held-out temperature error. Does the "
            "solver see that spread?"
        ),
        "policy": {
            "formula_arms": "one trial, 15 iterations",
            "production": "up to two trials; first-trial reported for the paired test",
            "stars": len(shared),
            "note": (
                "Paired: every arm is seeded from the same drawn stars and "
                "differs only in the constants that produced (m, T)."
            ),
        },
        "power": {
            "min_discordant_for_p_below_0_05": MIN_DISCORDANT_FOR_SIGNIFICANCE,
            "expected_discordant_at_60_stars": 2.7,
            "paired_stars_for_80_percent_power": 290,
            "consequence": (
                "A difference here is reportable as 'not worse', never as "
                "'better'. Check can_reach_significance before reading any p."
            ),
        },
        "endpoint_note": (
            "Two endpoints, with very different power. The binary convergence "
            "outcome depends only on the discordant pairs and cannot reach "
            "significance at sixty stars. The first-trial iteration count uses "
            "every concordant star and can, so it is the endpoint that carries "
            "the result here."
        ),
        "arms": summary,
        "paired_comparisons": comparisons,
        "confound_sensitivity": {
            "note": (
                "The analytic arm is the original H2 with no monotone "
                "projection, so on stars whose raw profile needs repair it "
                "differs from every other formula arm by the projection as "
                "well as by the thing under test. Re-run without them."
            ),
            "projection_touched_stars": sorted(touched),
            "iterations": sensitivity,
        },
        "failed_by_every_arm": shared_failures,
        "reproducer": "PYTHONPATH=. python3 -m experiments.analytic_initializer.run_multi_arm_comparison",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(shared)} shared stars\n")
    print(
        "%-11s %-7s %-9s %-16s %-8s %s"
        % ("arm", "floats", "offline T", "first-trial", "median it", "Wilson 95%")
    )
    for name, entry in summary.items():
        print(
            "%-11s %-7s %-9s %2d/%-13d %-8s [%.2f, %.2f]"
            % (
                name,
                entry.get("stored_floats") or "-",
                f"{entry['temperature_p95']:.4f}" if entry.get("temperature_p95") else "-",
                entry["first_trial_converged"],
                entry["stars"],
                entry["median_iterations_on_first_trial"],
                *entry["first_trial_wilson95"],
            )
        )
    print(
        "\n%-26s | %-20s %-8s %-6s | %-22s %s"
        % ("paired", "convergence split", "p", "sig?", "iterations (A-B)", "Wilcoxon p")
    )
    for name, entry in comparisons.items():
        convergence, iterations = entry["convergence"], entry["iterations"]
        print(
            "%-26s | %-20s %-8.4f %-6s | %-22s %s"
            % (
                name,
                f"{convergence['left_only']}/{convergence['right_only']} of {convergence['discordant']}",
                convergence["exact_mcnemar_p"],
                convergence["can_reach_significance"],
                (
                    f"{iterations['median_left']} vs {iterations['median_right']}"
                    f"  mean {iterations['mean_difference']:+.2f}"
                    if iterations.get("comparable")
                    else "-"
                ),
                (
                    f"{iterations['wilcoxon_signed_rank_p']:.2e}"
                    if iterations.get("wilcoxon_signed_rank_p") is not None
                    else "-"
                ),
            )
        )
    if sensitivity:
        print("\n%-26s %-26s %s" % ("sensitivity (iterations)", "all stars", "projection-touched removed"))
        for name, entry in sensitivity.items():
            fmt = lambda e: (  # noqa: E731
                f"n={e['stars']} {e['mean_difference']:+.2f} "
                f"p={e.get('wilcoxon_signed_rank_p', float('nan')):.1e}"
            )
            print("%-26s %-26s %s" % (name, fmt(entry["all_stars"]),
                                      fmt(entry["projection_touched_removed"])))
        print(f"  (touched: {sorted(touched)})")
    print(f"\nfailed by every arm: {shared_failures}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
