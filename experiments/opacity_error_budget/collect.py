"""Collect the decimation ladder into one citable result file.

Reads every ``summary.json`` under a run directory and emits the error budget:
per star, the displacement of the stride-N converged atmosphere from that same
star's stride-1 atmosphere, measured with the solver's own
``deep_layer_relative_temperature_change`` and read against the production
``5e-4`` convergence threshold.

Timings are deliberately NOT summarised. The Sun was run twice in this campaign
with bit-identical physics and wall times differing by 1.68-2.41x, with the
ratio itself varying by stride, so no speedup from these runs can carry a
claim. See ``timing_reproducibility`` in
``results/opacity_decimation/sun_smoke_20260818.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

STRIDE_KEY = "opacity_frequency_grid_stride"
DEEP_LAYER_THRESHOLD = 5.0e-4
NAME = re.compile(r"row(\d+)_(\w+?)_t([\d.]+)_g([-\d.]+)_m([-\d.]+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def largest_free_stride(displacement_by_stride: dict[int, float]) -> int:
    """Largest stride whose whole prefix stays inside the convergence threshold.

    A stride only counts as free if every coarser-than-1 stride below it is also
    free, so a lucky single point cannot certify a ladder it does not support.
    """

    free = [
        stride
        for stride in (1, 2, 4, 8)
        if all(
            displacement_by_stride.get(lower, float("inf")) < DEEP_LAYER_THRESHOLD
            for lower in (2, 4, 8)
            if lower <= stride
        )
    ]
    return max(free) if free else 1


def collect_star(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text())
    rows = summary["strides"]
    match = NAME.match(summary_path.parent.name)
    labels = json.loads((summary_path.parent / "labels.json").read_text())

    displacement = {
        row[STRIDE_KEY]: float(
            row.get("versus_stride_one", {}).get(
                "deep_layer_relative_temperature_change", 0.0
            )
        )
        for row in rows
    }
    iterations = {row[STRIDE_KEY]: int(row["iterations_to_convergence"]) for row in rows}

    # Displacement grows close to linearly in the stride; the slope is a more
    # portable summary than any single stride's value, and it predicts s*.
    strides = np.array([s for s in (2, 4, 8) if s in displacement], dtype=float)
    values = np.array([displacement[int(s)] for s in strides], dtype=float)
    slope = float(np.linalg.lstsq(strides[:, None], values[:, None], rcond=None)[0][0, 0])

    return {
        "row": int(match.group(1)) if match else None,
        "stratum": match.group(2) if match else None,
        "labels": labels,
        "iterations_by_stride": iterations,
        "iteration_count_changed_by_stride": len(set(iterations.values())) > 1,
        "displacement_by_stride": displacement,
        "displacement_over_threshold_by_stride": {
            s: v / DEEP_LAYER_THRESHOLD for s, v in displacement.items()
        },
        "displacement_slope_per_unit_stride": slope,
        "largest_free_stride": largest_free_stride(displacement),
        "summary_sha256": _sha256(summary_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs/opacity_error_budget"))
    parser.add_argument(
        "--out", type=Path, default=Path("results/opacity_error_budget/ladder.json")
    )
    args = parser.parse_args(argv)

    stars = [collect_star(p) for p in sorted(args.runs.glob("*/summary.json"))]
    free = [s["largest_free_stride"] for s in stars]
    slopes = [s["displacement_slope_per_unit_stride"] for s in stars]

    result = {
        "what": (
            "Frequency-grid decimation error budget: how much kappa_nu frequency "
            "resolution the converged atmosphere needs."
        ),
        "metric": (
            "convergence.deep_layer_relative_temperature_change between the "
            "stride-N and stride-1 converged temperature, against the production "
            f"threshold {DEEP_LAYER_THRESHOLD}."
        ),
        "star_count": len(stars),
        "largest_free_stride_histogram": {
            str(v): free.count(v) for v in sorted(set(free))
        },
        "displacement_slope_per_unit_stride": {
            "median": float(np.median(slopes)),
            "min": float(np.min(slopes)),
            "max": float(np.max(slopes)),
        },
        "stars": stars,
        "limits": [
            "Temperature yardstick only. The pre-registered criterion is spectral "
            "(3.44e-3 median fixed-point width) and needs synthesis from the "
            "reduced state through reduced_state/reconstruct.py; not run.",
            "No timing is reported: wall clock was not reproducible in this "
            "campaign (same star, bit-identical physics, 1.68-2.41x apart).",
            "Stratification is by effective temperature, which the data does not "
            "support as the axis that separates the budget.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}  ({len(stars)} stars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
