"""Test the fitted analytic H2 reduced-state initializer at cool targets.

The H2 closure is a low-rank, label-conditioned formula for the Hopf
temperature residual and the positive Rosseland-opacity profile.  It produces
only ``(m,T)`` at inference time; Payne-Zero reconstructs the other fields
with its exact physics path.  This targeted runner keeps the expensive
cool-star question separate from the long multi-arm pilot.

Example
-------
python -m experiments.reduced_state_emulator.cool_star_h2_targeted \
    --targets 3900 3800 3750 3700 3600 3500 \
    --out-root runs/cool_star_h2_targeted
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.profile_initializer import (
    fit_analytic_profile_parameters,
    load_analytic_profile_parameters,
    predict_analytic_reduced_state,
)
from bench.labels import StellarLabels

from .cool_star_step_test import (
    TrackSpec,
    _reconstruct_from_mt,
    _solve_attempt,
)


MANIFESTS = (
    Path("results/reconstruction_metrics.json"),
    Path("results/sealed_solver_subset_20260808.json"),
    Path("results/sealed_audit_20260808.json"),
    Path("results/sealed_audit_20260811.json"),
    Path("results/sealed_initializer_holdout_20260812.json"),
    Path("results/initializer_calibration_20260812.json"),
    Path("results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json"),
)


def _fit_h2_parameters(parameters_path: Path | None = None):
    corpus = load_strict_truth(DEFAULT_CORPUS)
    if parameters_path is not None:
        return (
            corpus,
            load_analytic_profile_parameters(parameters_path),
            None,
            [],
        )
    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=20260816)
    parameters = fit_analytic_profile_parameters(
        corpus, split, degree=3, components=5
    )
    return corpus, parameters, split, used_manifests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", type=float, default=[3500.0])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, default=None)
    parser.add_argument("--iteration-cap", type=int, default=30)
    parser.add_argument("--logg", type=float, default=5.0)
    parser.add_argument("--metallicity", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--vmic", type=float, default=1.0)
    args = parser.parse_args()

    targets = tuple(float(value) for value in args.targets)
    if any(value <= 0.0 for value in targets):
        raise SystemExit("target temperatures must be positive")

    corpus, parameters, split, used_manifests = _fit_h2_parameters(args.parameters)
    labels_array = np.asarray(
        [
            [
                value,
                args.logg,
                args.metallicity,
                args.alpha,
                args.vmic,
            ]
            for value in targets
        ],
        dtype=np.float64,
    )
    masses, temperatures, log_opacity = predict_analytic_reduced_state(
        labels_array, corpus.tau, parameters
    )

    track = TrackSpec(
        log_surface_gravity=args.logg,
        metallicity=args.metallicity,
        alpha_enhancement=args.alpha,
        microturbulence_km_s=args.vmic,
    )
    records: list[dict[str, object]] = []
    for index, target in enumerate(targets):
        labels = track.labels(target)
        started = time.perf_counter()
        record: dict[str, object] = {
            "method": "analytic_h2_target_reduced",
            "schedule": "direct",
            "source_temperature": 4000.0,
            "target_temperature": target,
            "formula": "low_rank_hopf_and_positive_opacity_profile",
            "corpus": str(corpus.path),
            "corpus_size": corpus.size,
            "fit_split_seed": None if split is None else split.seed,
            "fit_excluded_count": None if split is None else int(split.excluded.size),
            "fit_manifests": used_manifests,
            "analytic_input": {
                "mass_min": float(np.min(masses[index])),
                "mass_max": float(np.max(masses[index])),
                "temperature_min": float(np.min(temperatures[index])),
                "temperature_max": float(np.max(temperatures[index])),
                "log_opacity_min": float(np.min(log_opacity[index])),
                "log_opacity_max": float(np.max(log_opacity[index])),
                "uses_target_emulator": False,
            },
        }
        try:
            seed = _reconstruct_from_mt(
                labels,
                masses[index],
                temperatures[index],
            )
            solver_record, _state = _solve_attempt(
                track=track,
                method="analytic_h2_target_reduced",
                schedule="direct",
                source_temperature=4000.0,
                target_labels=labels,
                initial_atmosphere=seed,
                product_dir=args.out_root / "products" / "analytic_h2_target_reduced",
                iteration_cap=args.iteration_cap,
            )
            record.update(solver_record)
        except Exception as exc:  # noqa: BLE001 - preserve one target per row
            record.update(
                {
                    "status": "initializer_or_solver_exception",
                    "converged": False,
                    "survives_solver": False,
                    "survives": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        record["wall_seconds_including_setup"] = float(time.perf_counter() - started)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    output = {
        "format": "payne_zero_cool_star_h2_targeted_v1",
        "track": track.as_json(),
        "targets": list(targets),
        "iteration_cap": args.iteration_cap,
        "formula": "low_rank_hopf_and_positive_opacity_profile",
        "formula_coefficient_count": parameters.coefficient_count,
        "formula_basis_value_count": parameters.basis_value_count,
        "records": records,
    }
    output_path = args.out_root / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
