"""Run a small real-solver smoke test for the standalone H2 profile formula."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

# This must happen before importing the production package on macOS.
from bench import environment as _environment  # noqa: F401

from bench.run_reference import _solver_config
from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.no_emulator_bridge import analytic_seed_model
from experiments.analytic_initializer.profile_initializer import (
    fit_analytic_profile_parameters,
    predict_analytic_reduced_state,
)
from payne_zero_atmosphere.runner import run_atmosphere_model


OUTPUT = Path("results/analytic_initializer/h2_solver_smoke12.json")
MANIFESTS = (
    Path("results/reconstruction_metrics.json"),
    Path("results/sealed_solver_subset_20260808.json"),
    Path("results/sealed_audit_20260808.json"),
    Path("results/sealed_audit_20260811.json"),
    Path("results/sealed_initializer_holdout_20260812.json"),
    Path("results/initializer_calibration_20260812.json"),
    Path("results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json"),
)


def _choose_indices(labels: np.ndarray, available: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    selected: list[int] = []
    for low, high in ((4000.0, 5500.0), (5500.0, 7500.0), (7500.0, 10500.0)):
        candidates = available[
            (labels[available, 0] >= low) & (labels[available, 0] < high)
        ]
        if candidates.size < 4:
            raise RuntimeError(f"not enough validation stars in {low:g}--{high:g} K")
        selected.extend(generator.choice(candidates, size=4, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def main() -> None:
    corpus = load_strict_truth(DEFAULT_CORPUS)
    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=20260816)
    indices = _choose_indices(corpus.labels, split.validation, seed=20260816)
    parameters = fit_analytic_profile_parameters(
        corpus, split, degree=3, components=5
    )
    mass, temperature, log_opacity = predict_analytic_reduced_state(
        corpus.labels[indices], corpus.tau, parameters
    )

    records: list[dict[str, object]] = []
    for row, corpus_index in enumerate(indices):
        labels = corpus.labels[corpus_index]
        start = time.perf_counter()
        record: dict[str, object] = {
            "corpus_index": int(corpus_index),
            "effective_temperature": float(labels[0]),
            "log_surface_gravity": float(labels[1]),
            "metallicity": float(labels[2]),
            "alpha_enhancement": float(labels[3]),
            "microturbulence_km_s": float(labels[4]),
        }
        try:
            seed = analytic_seed_model(
                labels,
                mass[row],
                temperature[row],
                log_opacity[row],
                corpus.tau,
            )
            result = run_atmosphere_model(
                _solver_config(
                    seed,
                    iterations_per_trial=15,
                    structured_atmosphere_path=None,
                    debug_state_path=None,
                )
            )
            record.update(
                {
                    "converged": bool(result.converged),
                    "iterations_completed": int(result.iterations_completed),
                    "finite_final_state": bool(
                        all(
                            np.all(np.isfinite(value))
                            for value in (
                                result.atmosphere.column_mass,
                                result.atmosphere.temperature,
                                result.atmosphere.gas_pressure,
                                result.atmosphere.electron_density,
                                result.atmosphere.rosseland_opacity,
                            )
                        )
                    ),
                    "deep_layer_relative_temperature_change": float(
                        result.diagnostics["deep_layer_relative_temperature_change"]
                    ),
                    "seconds": float(time.perf_counter() - start),
                }
            )
        except Exception as exc:  # noqa: BLE001 - the smoke report records failures
            record.update(
                {
                    "converged": False,
                    "iterations_completed": 0,
                    "finite_final_state": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "seconds": float(time.perf_counter() - start),
                }
            )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    result = {
        "candidate": "H2_standalone_low_rank_hopf_and_opacity_profile",
        "status": "smoke_only_not_production",
        "corpus": str(corpus.path),
        "split_seed": split.seed,
        "excluded_count": int(excluded.size),
        "excluded_manifests": used_manifests,
        "degree": 3,
        "components": 5,
        "coefficient_count": parameters.coefficient_count,
        "basis_value_count": parameters.basis_value_count,
        "records": records,
        "converged_count": int(sum(bool(item["converged"]) for item in records)),
        "finite_count": int(sum(bool(item["finite_final_state"]) for item in records)),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("converged_count", "finite_count", "coefficient_count", "basis_value_count")}, sort_keys=True))
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
