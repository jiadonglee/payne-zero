"""The continuity harness: does refining the depth grid remove the drift?

Runs three measurements over the converged truth corpus and writes one JSON
summary:

1. **Seed residual over the whole corpus.** ``log10(kappa0*m0/tau0)`` in closed
   form for all 52,199 stars, plus how the solver's iteration count varies with
   it.
2. **Refinement scan.** The solver's own quadrature re-run at 1x .. 16x grid
   density on a sample, reported at the top layers.
3. **Population of the failures.** Labels of the stars whose seed residual
   exceeds 0.1 dex.

Usage::

    python -m continuity.run_harness --out runs/continuity
    python -m continuity.run_harness --out runs/continuity --sample 2000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .closure import (
    COLUMN_MASS,
    ROSSELAND_OPACITY,
    closure_residual_at_refinement,
    load_corpus,
    residual_to_column_mass_drift,
    seed_residual,
)

DEFAULT_CORPUS = Path(
    "source_data_files/atmosphere_emulator/five_label/strict_truth_52199.npz"
)
DEFAULT_REFINEMENTS = (1, 2, 4, 8, 16)
TOP_LAYERS = 5


def _quantiles(values: np.ndarray) -> dict:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "median": float(np.median(absolute)),
        "p90": float(np.percentile(absolute, 90)),
        "p95": float(np.percentile(absolute, 95)),
        "p99": float(np.percentile(absolute, 99)),
        "max": float(np.max(absolute)),
    }


def seed_survey(profiles, tau, iterations, labels) -> dict:
    """Seed residual for every star, and iteration count binned by it."""

    residual = seed_residual(
        profiles[:, :, COLUMN_MASS], profiles[:, :, ROSSELAND_OPACITY], tau
    )
    absolute = np.abs(residual)

    teff = np.array([row["effective_temperature"] for row in labels])
    logg = np.array([row["log_surface_gravity"] for row in labels])
    metallicity = np.array([row["metallicity"] for row in labels])

    bins = []
    edges = [0.0, 1e-3, 1e-2, 1e-1, np.inf]
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (absolute >= low) & (absolute < high)
        if not selected.any():
            continue
        bins.append(
            {
                "range_dex": [low, None if np.isinf(high) else high],
                "count": int(selected.sum()),
                "iterations_median": float(np.median(iterations[selected])),
                "iterations_mean": float(iterations[selected].mean()),
                "effective_temperature_mean": float(teff[selected].mean()),
                "log_surface_gravity_mean": float(logg[selected].mean()),
                "metallicity_mean": float(metallicity[selected].mean()),
            }
        )

    return {
        "star_count": int(len(absolute)),
        "residual_dex": _quantiles(residual),
        "fraction_above_0.01_dex": float((absolute > 0.01).mean()),
        "fraction_above_0.10_dex": float((absolute > 0.10).mean()),
        "iterations_by_residual_bin": bins,
    }


def refinement_scan(profiles, tau, sample, refinements, seed) -> dict:
    """Re-run the solver's closure at increasing grid density on a sample."""

    rng = np.random.default_rng(seed)
    chosen = rng.choice(profiles.shape[0], size=min(sample, profiles.shape[0]),
                        replace=False)

    results = {}
    for factor in refinements:
        top = []
        for index in chosen:
            residual = closure_residual_at_refinement(
                profiles[index, :, COLUMN_MASS],
                profiles[index, :, ROSSELAND_OPACITY],
                tau[index],
                factor,
            )
            top.append(residual[:TOP_LAYERS])
        top = np.asarray(top)
        drift = residual_to_column_mass_drift(top[:, 1:]) * 100.0
        results[str(factor)] = {
            "grid_points": 1 + (profiles.shape[1] - 1) * factor,
            "top_layer_residual_dex": _quantiles(top),
            "column_mass_drift_percent": _quantiles(drift),
        }
    return {"sample": int(len(chosen)), "seed": int(seed), "by_refinement": results}


def failure_population(profiles, tau, iterations, labels, threshold=0.1) -> dict:
    """Where in label space the broken-seed stars sit."""

    residual = np.abs(
        seed_residual(
            profiles[:, :, COLUMN_MASS], profiles[:, :, ROSSELAND_OPACITY], tau
        )
    )
    broken = residual > threshold
    teff = np.array([row["effective_temperature"] for row in labels])
    logg = np.array([row["log_surface_gravity"] for row in labels])
    metallicity = np.array([row["metallicity"] for row in labels])

    def _describe(mask):
        return {
            "count": int(mask.sum()),
            "effective_temperature_mean": float(teff[mask].mean()),
            "log_surface_gravity_mean": float(logg[mask].mean()),
            "metallicity_mean": float(metallicity[mask].mean()),
            "iterations_median": float(np.median(iterations[mask])),
        }

    return {
        "threshold_dex": threshold,
        "broken": _describe(broken),
        "rest": _describe(~broken),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=400,
                        help="stars in the refinement scan (it is the slow part)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--refinements",
        type=int,
        nargs="+",
        default=list(DEFAULT_REFINEMENTS),
    )
    args = parser.parse_args(argv)

    profiles, tau, iterations, labels = load_corpus(args.corpus)
    print(f"corpus: {profiles.shape[0]} atmospheres, {profiles.shape[1]} layers")

    summary = {
        "corpus": str(args.corpus),
        "seed_survey": seed_survey(profiles, tau, iterations, labels),
        "refinement_scan": refinement_scan(
            profiles, tau, args.sample, args.refinements, args.seed
        ),
        "failure_population": failure_population(profiles, tau, iterations, labels),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {path}")

    scan = summary["refinement_scan"]["by_refinement"]
    print("\nrefinement scan — top-layer closure residual (dex):")
    for factor, block in scan.items():
        stats = block["top_layer_residual_dex"]
        print(
            f"  x{factor:<3} ({block['grid_points']:>5} points)  "
            f"median {stats['median']:.5f}  p99 {stats['p99']:.5f}  "
            f"max {stats['max']:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
