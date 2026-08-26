"""Consolidate the existing baseline `summary.json` files into one table.

Part 1 of the reduced-state project needs "one machine-readable benchmark
table". The four baseline runs already exist (`runs/baseline_local/`,
`runs/baseline_cluster/{iid,boundary,hard}/`) and already carry every metric
asked for -- contraction, non-monotonic fraction, iteration percentiles,
failure/retry fraction, wall time -- via `bench/report.py`. This script does
not recompute anything; it reads the four `summary.json` files and reshapes
them into one flat table, tagged by slice, matching `bench.report`'s field
names verbatim.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_SLICES = {
    "box": "runs/baseline_local/summary.json",
    "iid": "runs/baseline_cluster/iid/summary.json",
    "boundary": "runs/baseline_cluster/boundary/summary.json",
    "hard": "runs/baseline_cluster/hard/summary.json",
}


def _flatten_slice(slice_name: str, summary: dict) -> dict:
    iters = summary["converging_trial_iterations"]
    total_iters = summary["total_iterations_including_retries"]
    seconds = summary["seconds_per_star"]
    contraction = summary["contraction"]
    q_ratio = contraction["q_ratio"]
    headroom = summary["headroom"]
    return {
        "slice": slice_name,
        "star_count": summary["star_count"],
        "converged_fraction": summary["converged_fraction"],
        "retry_fraction": summary["retry_fraction"],
        "failure_fraction": summary["failure_fraction"],
        "iterations_mean": iters["mean"],
        "iterations_p50": iters["p50"],
        "iterations_p90": iters["p90"],
        "iterations_p95": iters["p95"],
        "iterations_p99": iters["p99"],
        "iterations_max": iters["max"],
        "total_iterations_incl_retries_mean": total_iters["mean"],
        "total_iterations_incl_retries_p90": total_iters["p90"],
        "seconds_per_star_mean": seconds["mean"],
        "seconds_per_star_p50": seconds["p50"],
        "seconds_per_star_p90": seconds["p90"],
        "total_wall_hours": seconds["total_hours"],
        "contraction_geomean_q": q_ratio["geometric_mean"],
        "contraction_p50_q": q_ratio["p50"],
        "contraction_p90_q": q_ratio["p90"],
        "non_monotonic_fraction": contraction["non_monotonic_fraction"],
        "first_iteration_residual_mean": contraction["first_iteration_residual"]["mean"],
        "first_iteration_residual_p90": contraction["first_iteration_residual"]["p90"],
        "recoverable_fraction": headroom["recoverable_fraction"],
        "stars_already_at_floor_fraction": headroom["stars_already_at_floor_fraction"],
    }


def consolidate(slices: dict[str, Path], repo_root: Path) -> list[dict]:
    rows = []
    for slice_name, rel_path in slices.items():
        path = repo_root / rel_path if not Path(rel_path).is_absolute() else Path(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"missing baseline summary for slice {slice_name!r}: {path}")
        summary = json.loads(path.read_text())
        rows.append(_flatten_slice(slice_name, summary))
    return rows


def write_table(rows: list[dict], out_json: Path, out_csv: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"slices": rows}, indent=2))
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    out_json = args.out_json or (args.repo_root / "results" / "baseline_metrics.json")
    out_csv = args.out_csv or (args.repo_root / "results" / "baseline_metrics.csv")

    rows = consolidate(DEFAULT_SLICES, args.repo_root)
    write_table(rows, out_json, out_csv)
    print(f"wrote {out_json}")
    print(f"wrote {out_csv}")
    for row in rows:
        print(
            f"  {row['slice']:>10s}  n={row['star_count']:5d}  "
            f"converged={row['converged_fraction']:.3f}  "
            f"iters(mean/p90)={row['iterations_mean']:.2f}/{row['iterations_p90']:.0f}  "
            f"geomean_q={row['contraction_geomean_q']:.3f}  "
            f"nonmono={row['non_monotonic_fraction']:.3f}  "
            f"recoverable={row['recoverable_fraction']:.3f}"
        )


if __name__ == "__main__":
    main()
