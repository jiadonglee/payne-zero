"""Fast solver diagnostic for the H2 analytic cool-star initializer.

This companion to ``cool_star_h2_targeted`` uses the analytic bridge's
positive provisional four fields, so it can answer the solver-basin question
without waiting for the exact pressure-synchronization materialization.  It
is diagnostic only; the exact-reconstruction runner remains the acceptance
test.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.labels import StellarLabels
from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.no_emulator_bridge import analytic_seed_model
from experiments.analytic_initializer.profile_initializer import (
    fit_analytic_profile_parameters,
    load_analytic_profile_parameters,
    predict_analytic_reduced_state,
)

from .cool_star_step_test import TrackSpec, _solve_attempt


MANIFESTS = (
    Path("results/reconstruction_metrics.json"),
    Path("results/sealed_solver_subset_20260808.json"),
    Path("results/sealed_audit_20260808.json"),
    Path("results/sealed_audit_20260811.json"),
    Path("results/sealed_initializer_holdout_20260812.json"),
    Path("results/initializer_calibration_20260812.json"),
    Path("results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, default=3500.0)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, default=None)
    parser.add_argument("--iteration-cap", type=int, default=30)
    args = parser.parse_args()

    corpus = load_strict_truth(DEFAULT_CORPUS)
    if args.parameters is None:
        excluded, used_manifests = collect_excluded_indices(
            MANIFESTS, corpus_size=corpus.size
        )
        split = make_split(corpus.size, excluded=excluded, seed=20260816)
        parameters = fit_analytic_profile_parameters(
            corpus, split, degree=3, components=5
        )
    else:
        split = None
        used_manifests = []
        parameters = load_analytic_profile_parameters(args.parameters)
    labels = np.asarray([[args.target, 5.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    mass, temperature, log_opacity = predict_analytic_reduced_state(
        labels, corpus.tau, parameters
    )
    track = TrackSpec(log_surface_gravity=5.0, metallicity=0.0)
    stellar_labels = track.labels(args.target)
    started = time.perf_counter()
    record: dict[str, object] = {
        "format": "payne_zero_cool_star_h2_raw_targeted_v1",
        "method": "analytic_h2_raw_seed",
        "source_temperature": 4000.0,
        "target_temperature": args.target,
        "formula": "low_rank_hopf_and_positive_opacity_profile",
        "corpus": str(corpus.path),
        "fit_split_seed": None if split is None else split.seed,
        "fit_excluded_count": None if split is None else int(split.excluded.size),
        "fit_manifests": used_manifests,
        "analytic_input": {
            "mass_min": float(mass[0].min()),
            "mass_max": float(mass[0].max()),
            "temperature_min": float(temperature[0].min()),
            "temperature_max": float(temperature[0].max()),
            "log_opacity_min": float(log_opacity[0].min()),
            "log_opacity_max": float(log_opacity[0].max()),
            "uses_target_emulator": False,
        },
    }
    try:
        seed = analytic_seed_model(
            labels[0], mass[0], temperature[0], log_opacity[0], corpus.tau
        )
        solver_record, _state = _solve_attempt(
            track=track,
            method="analytic_h2_raw_seed",
            schedule="direct",
            source_temperature=4000.0,
            target_labels=stellar_labels,
            initial_atmosphere=seed,
            product_dir=args.out_root / "products" / "analytic_h2_raw_seed",
            iteration_cap=args.iteration_cap,
        )
        record.update(solver_record)
    except Exception as exc:  # noqa: BLE001
        record.update(
            {
                "status": "initializer_or_solver_exception",
                "converged": False,
                "survives_solver": False,
                "survives": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    record["wall_seconds"] = float(time.perf_counter() - started)
    output = args.out_root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
