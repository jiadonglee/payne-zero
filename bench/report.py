"""Aggregate reference-solver records into the Stage 0 baseline report.

The headline numbers are the *tail* and the *retry rate*, not the mean. The
production policy floors every solve at three iterations
(``minimum_iterations_before_convergence=3``, ``cli.py:427``) and caps each trial
at fifteen, so a mean iteration count cannot fall below three however good the
initializer becomes. The cost that an initializer can actually remove lives in
the upper percentiles and in the second trial, which is a whole extra
fifteen-iteration budget.

Usage::

    python -m bench.report runs/baseline/records.jsonl
    python -m bench.report runs/baseline/records.jsonl --json runs/baseline/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .labels import LABEL_FIELDS


PERCENTILES = (50, 75, 90, 95, 99)


def load_records(path: Path | str) -> list[dict]:
    """Read a records.jsonl written by bench.run_reference."""

    lines = Path(path).read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {f"p{p}": float("nan") for p in PERCENTILES}
    return {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}


def summarize(records: list[dict]) -> dict:
    """Compute the baseline summary from a record list."""

    total = len(records)
    if total == 0:
        raise ValueError("no records to summarize")

    converged = [r for r in records if r["converged"]]
    retried = [r for r in records if r["needed_retry"]]
    failed = [r for r in records if not r["converged"]]

    # Iterations of the trial that converged: what a better initializer shortens.
    winning = np.array(
        [r["converging_trial_iterations"] for r in converged], dtype=float
    )
    # Every iteration burned including failed trials: the true wall-clock driver.
    burned = np.array([r["total_iterations"] for r in records], dtype=float)
    seconds = np.array([r["seconds"] for r in records], dtype=float)

    summary = {
        "star_count": total,
        "converged_count": len(converged),
        "converged_fraction": len(converged) / total,
        "retry_count": len(retried),
        "retry_fraction": len(retried) / total,
        "failure_count": len(failed),
        "failure_fraction": len(failed) / total,
        "converging_trial_iterations": {
            "mean": float(winning.mean()) if winning.size else float("nan"),
            **_percentiles(winning),
            "max": float(winning.max()) if winning.size else float("nan"),
            "histogram": _histogram(winning),
        },
        "total_iterations_including_retries": {
            "mean": float(burned.mean()),
            **_percentiles(burned),
            "max": float(burned.max()),
        },
        "seconds_per_star": {
            "mean": float(seconds.mean()),
            **_percentiles(seconds),
            "total_hours": float(seconds.sum() / 3600.0),
        },
        "stage_seconds": _stage_seconds(records),
        "contraction": _contraction(records),
        "tail_labels": _tail_labels(records),
        "headroom": _headroom(records),
    }
    return summary


def _histogram(values: np.ndarray) -> dict[str, int]:
    if values.size == 0:
        return {}
    counts: dict[str, int] = {}
    for value in np.sort(np.unique(values)):
        counts[str(int(value))] = int((values == value).sum())
    return counts


def _finite(value) -> float | None:
    """Return a usable float, or None.

    Diagnostics from a diverged solve arrive as ``null`` (see
    ``run_reference._as_plain``), and a missing key is equally possible for an
    older record, so both are folded into the same "unusable" answer here.
    """

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _iterations(record: dict) -> list[dict]:
    """Per-iteration diagnostics of the trial that converged, else the last."""

    for trial in record["trials"]:
        if trial["converged"]:
            return trial["diagnostics"].get("iteration_timings", [])
    if record["trials"]:
        return record["trials"][-1]["diagnostics"].get("iteration_timings", [])
    return []


def _stage_seconds(records: list[dict]) -> dict:
    """Mean seconds per stage, split by first versus later iterations.

    The first iteration pays the line-selection and catalog cost, so averaging
    it together with later ones hides where the time actually goes.
    """

    stages = (
        "population_seconds",
        "opacity_seconds",
        "transfer_seconds",
        "finalization_seconds",
        "remap_seconds",
    )
    first: dict[str, list[float]] = {s: [] for s in stages}
    later: dict[str, list[float]] = {s: [] for s in stages}
    for record in records:
        for step in _iterations(record):
            bucket = first if step.get("iteration") == 1 else later
            for stage in stages:
                value = _finite(step.get(stage))
                if value is not None:
                    bucket[stage].append(value)
    return {
        "first_iteration_mean": {
            s: float(np.mean(v)) if v else float("nan") for s, v in first.items()
        },
        "later_iteration_mean": {
            s: float(np.mean(v)) if v else float("nan") for s, v in later.items()
        },
    }


def _contraction(records: list[dict]) -> dict:
    """Distribution of the per-iteration residual contraction ratio.

    ``q_k = r_{k+1} / r_k`` on the deep-layer relative temperature change, which
    is the quantity the stopping rule actually tests (``convergence.py:60``).
    A learned initializer is supposed to push these below one and keep them
    there; measuring them first says how much room there is.
    """

    ratios: list[float] = []
    first_residual: list[float] = []
    non_monotonic = 0
    trajectories = 0
    for record in records:
        steps = _iterations(record)
        series = [
            value
            for value in (
                _finite(s.get("deep_layer_relative_temperature_change")) for s in steps
            )
            if value is not None
        ]
        if not series:
            continue
        trajectories += 1
        first_residual.append(series[0])
        pairs = list(zip(series[:-1], series[1:]))
        if any(after > before for before, after in pairs):
            non_monotonic += 1
        ratios.extend(after / before for before, after in pairs if before > 0.0)

    array = np.array(ratios, dtype=float)
    residual = np.array(first_residual, dtype=float)

    # A contraction ratio is multiplicative, and its denominator can be a
    # residual near zero, so an arithmetic mean is dominated by a handful of
    # enormous values and says nothing about typical behaviour. The geometric
    # mean is the per-iteration factor that actually compounds over a solve.
    positive = array[np.isfinite(array) & (array > 0.0)]
    geometric_mean = (
        float(np.exp(np.log(positive).mean())) if positive.size else float("nan")
    )
    return {
        "trajectory_count": trajectories,
        "non_monotonic_fraction": (
            non_monotonic / trajectories if trajectories else float("nan")
        ),
        "q_ratio": {
            "geometric_mean": geometric_mean,
            "max": float(array.max()) if array.size else float("nan"),
            **_percentiles(array),
        },
        "first_iteration_residual": {
            "mean": float(residual.mean()) if residual.size else float("nan"),
            **_percentiles(residual),
        },
    }


def _headroom(records: list[dict]) -> dict:
    """How many iterations an ideal initializer could actually remove.

    The floor is three iterations, so a star that already converges in three has
    no headroom at all. This counts the recoverable iterations directly rather
    than inferring them from a mean.
    """

    floor = 3
    recoverable = 0
    total_burned = 0
    stars_at_floor = 0
    for record in records:
        burned = record["total_iterations"]
        total_burned += burned
        if record["converged"]:
            if record["converging_trial_iterations"] == floor and not record["needed_retry"]:
                stars_at_floor += 1
        recoverable += max(0, burned - floor)
    return {
        "iteration_floor": floor,
        "stars_already_at_floor": stars_at_floor,
        "stars_already_at_floor_fraction": stars_at_floor / len(records),
        "total_iterations_burned": total_burned,
        "recoverable_iterations": recoverable,
        "recoverable_fraction": recoverable / total_burned if total_burned else 0.0,
    }


def _tail_labels(records: list[dict], quantile: float = 0.9) -> dict:
    """Label ranges of the slowest decile, versus the whole set.

    Answers "where does the tail live" without assuming a shape: it reports the
    mean of each label among expensive stars next to the population mean.
    """

    burned = np.array([r["total_iterations"] for r in records], dtype=float)
    if burned.size == 0:
        return {}
    threshold = float(np.quantile(burned, quantile))
    tail = [r for r in records if r["total_iterations"] >= threshold]
    if not tail:
        return {"threshold_iterations": threshold, "tail_count": 0}

    out = {
        "threshold_iterations": threshold,
        "tail_count": len(tail),
        "tail_fraction": len(tail) / len(records),
        "labels": {},
    }
    for field in LABEL_FIELDS:
        everything = np.array([r["labels"][field] for r in records], dtype=float)
        subset = np.array([r["labels"][field] for r in tail], dtype=float)
        out["labels"][field] = {
            "population_mean": float(everything.mean()),
            "tail_mean": float(subset.mean()),
            "tail_min": float(subset.min()),
            "tail_max": float(subset.max()),
        }
    return out


def format_report(summary: dict) -> str:
    """Render the summary as a readable text block."""

    lines = []
    add = lines.append

    add("=" * 68)
    add("Stage 0 baseline: unmodified payne_zero_atmosphere production solve")
    add("=" * 68)
    add("")
    add(f"stars                  {summary['star_count']}")
    add(
        f"converged              {summary['converged_count']} "
        f"({summary['converged_fraction']:.1%})"
    )
    add(
        f"needed a retry         {summary['retry_count']} "
        f"({summary['retry_fraction']:.1%})"
    )
    add(
        f"failed outright        {summary['failure_count']} "
        f"({summary['failure_fraction']:.1%})"
    )
    add("")

    winning = summary["converging_trial_iterations"]
    add("iterations of the converging trial")
    add(
        f"  mean {winning['mean']:.2f}   "
        + "   ".join(f"{k} {winning[k]:.0f}" for k in ("p50", "p90", "p95", "p99"))
        + f"   max {winning['max']:.0f}"
    )
    if winning["histogram"]:
        add("  histogram  " + "  ".join(f"{k}:{v}" for k, v in winning["histogram"].items()))
    add("")

    burned = summary["total_iterations_including_retries"]
    add("total iterations burned per star (retries included)")
    add(
        f"  mean {burned['mean']:.2f}   "
        + "   ".join(f"{k} {burned[k]:.0f}" for k in ("p50", "p90", "p95", "p99"))
        + f"   max {burned['max']:.0f}"
    )
    add("")

    head = summary["headroom"]
    add("headroom above the three-iteration floor")
    add(
        f"  already at the floor   {head['stars_already_at_floor']} "
        f"({head['stars_already_at_floor_fraction']:.1%}) - no headroom at all"
    )
    add(
        f"  recoverable iterations {head['recoverable_iterations']} of "
        f"{head['total_iterations_burned']} ({head['recoverable_fraction']:.1%})"
    )
    add("")

    contraction = summary["contraction"]
    q = contraction["q_ratio"]
    residual = contraction["first_iteration_residual"]
    add("residual contraction on the deep-layer dT/T that the stopping rule tests")
    add(
        f"  q = r_next/r     geomean {q['geometric_mean']:.3f}   "
        + "   ".join(f"{k} {q[k]:.3f}" for k in ("p50", "p90", "p95"))
        + f"   max {q['max']:.3g}"
    )
    add(f"  non-monotonic    {contraction['non_monotonic_fraction']:.1%} of trajectories")
    add(
        f"  first residual   median {residual['p50']:.3e}  "
        f"p90 {residual['p90']:.3e}   (threshold 5.0e-04)"
    )
    add("")

    seconds = summary["seconds_per_star"]
    add("wall clock")
    add(
        f"  seconds/star     mean {seconds['mean']:.1f}   "
        + "   ".join(f"{k} {seconds[k]:.1f}" for k in ("p50", "p90", "p99"))
    )
    add(f"  total            {seconds['total_hours']:.2f} h")
    stages = summary["stage_seconds"]
    add("  mean seconds per stage")
    for name, bucket in (
        ("first iteration", stages["first_iteration_mean"]),
        ("later iterations", stages["later_iteration_mean"]),
    ):
        rendered = "  ".join(
            f"{k.replace('_seconds', '')} {v:.2f}" for k, v in bucket.items()
        )
        add(f"    {name:17s} {rendered}")
    add("")

    tail = summary.get("tail_labels") or {}
    if tail.get("tail_count"):
        add(
            f"where the tail lives ({tail['tail_count']} stars at "
            f">= {tail['threshold_iterations']:.0f} iterations)"
        )
        add(f"    {'label':24s} {'population':>12s} {'tail mean':>12s}  tail range")
        for field, stats in tail["labels"].items():
            add(
                f"    {field:24s} {stats['population_mean']:12.3f} "
                f"{stats['tail_mean']:12.3f}  "
                f"[{stats['tail_min']:.2f}, {stats['tail_max']:.2f}]"
            )
        add("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.report", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("records", type=Path, help="records.jsonl from bench.run_reference")
    parser.add_argument("--json", type=Path, help="also write the summary as JSON")
    args = parser.parse_args(argv)

    summary = summarize(load_records(args.records))
    print(format_report(summary))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"summary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
