"""Localize the H2 analytic-initializer error before committing to a deep closure.

The solver funnel only records whether a star converged, so it cannot say *where*
the analytic initial state is wrong.  This probe answers that offline, with no
solver calls: it splits the predicted-versus-truth temperature error into the
surface band and the deep convergence window that the production stop criterion
actually watches (layers 39..layers-5), and it correlates the deep error with a
Schwarzschild-style radiative/convective diagnostic computed on the truth
profiles.  The output decides whether a radiative/convective entropy closure is
the right next target or a mis-attribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.deep_diagnostics import (
    bin_by_effective_temperature,
    convective_diagnostics,
    deep_window,
    error_bands,
)
from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.profile_initializer import (
    fit_analytic_profile_parameters,
    predict_analytic_reduced_state,
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

# Stars with a recorded real-solver outcome from the same H2 parameters.
FUNNEL_NONCONVERGED = (11206, 13265, 33356)
FUNNEL_HARD_TAIL = (34042,)
SMOKE_CONVERGED = (
    2891, 6896, 7811, 10082, 10313, 22134,
    25100, 25948, 27946, 35654, 38262, 41936,
)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=3000)
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/analytic_initializer/deep_closure_localization.json"),
    )
    args = parser.parse_args(argv)

    corpus = load_strict_truth(DEFAULT_CORPUS)
    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=20260816)
    parameters = fit_analytic_profile_parameters(corpus, split, degree=3, components=5)

    start, stop = deep_window(corpus.layers)

    def _error_bands(indices: np.ndarray) -> dict[str, np.ndarray]:
        mass, temperature, log_opacity = predict_analytic_reduced_state(
            corpus.labels[indices], corpus.tau, parameters
        )
        return error_bands(
            corpus,
            indices,
            mass=mass,
            temperature=temperature,
            log_opacity=log_opacity,
        )

    # --- per-star rows for every index with a recorded real-solver outcome ---
    solver_indices = np.asarray(
        FUNNEL_NONCONVERGED + FUNNEL_HARD_TAIL + SMOKE_CONVERGED, dtype=np.int64
    )
    outcome = (
        ["funnel_nonconverged_finite"] * len(FUNNEL_NONCONVERGED)
        + ["funnel_hard_tail_terminated"] * len(FUNNEL_HARD_TAIL)
        + ["smoke_converged"] * len(SMOKE_CONVERGED)
    )
    bands = _error_bands(solver_indices)
    solver_rows = []
    for row, index in enumerate(solver_indices):
        solver_rows.append(
            {
                "corpus_index": int(index),
                "solver_outcome": outcome[row],
                "effective_temperature": float(corpus.labels[index, 0]),
                "log_surface_gravity": float(corpus.labels[index, 1]),
                "metallicity": float(corpus.labels[index, 2]),
                "alpha_enhancement": float(corpus.labels[index, 3]),
                "microturbulence_km_s": float(corpus.labels[index, 4]),
                "temperature_surface_dex": float(bands["temperature_surface"][row]),
                "temperature_deep_dex": float(bands["temperature_deep"][row]),
                "mass_surface_dex": float(bands["mass_surface"][row]),
                "mass_deep_dex": float(bands["mass_deep"][row]),
                "opacity_deep_dex": float(bands["opacity_deep"][row]),
            }
        )

    # --- population sweep over the held-out validation split ---
    generator = np.random.default_rng(args.sample_seed)
    sample = generator.choice(split.validation, args.sample_count, replace=False)
    sample_bands = _error_bands(sample)
    deep_error = sample_bands["temperature_deep"]
    _subadiabatic, onset = convective_diagnostics(corpus, sample)
    effective_temperature = corpus.labels[sample, 0]
    bin_rows = bin_by_effective_temperature(effective_temperature, deep_error, onset)

    top_mask = deep_error >= np.quantile(deep_error, 0.95)
    has_convection = onset >= 0
    offset = (
        sample_bands["temperature_deep_argmax_layer"][has_convection]
        - onset[has_convection]
    )
    top_offset = (
        sample_bands["temperature_deep_argmax_layer"][has_convection & top_mask]
        - onset[has_convection & top_mask]
    )

    result = {
        "format": "payne_zero_analytic_initializer_deep_closure_localization_v1",
        "candidate": "H2_standalone_low_rank_hopf_and_opacity_profile",
        "status": "offline_diagnostic_no_solver_calls",
        "question": (
            "Is the H2 initializer error concentrated in the deep radiative/"
            "convective closure, as the execution log asserts?"
        ),
        "corpus": {
            "path": str(corpus.path),
            "sha256": file_sha256(corpus.path),
            "star_count": int(corpus.size),
            "layers": int(corpus.layers),
        },
        "excluded_manifests": used_manifests,
        "split": {
            "seed": int(split.seed),
            "train_count": int(split.train.size),
            "validation_count": int(split.validation.size),
            "excluded_count": int(excluded.size),
        },
        "deep_window": {
            "start_layer": start,
            "stop_layer": stop,
            "note": (
                "This is the band the production convergence stop watches; the "
                "threshold is maximum_deep_layer_relative_temperature_change=5.0e-4."
            ),
        },
        "stars_with_recorded_solver_outcome": solver_rows,
        "validation_population": {
            "sample_count": int(sample.size),
            "sample_seed": int(args.sample_seed),
            "temperature_surface_dex_p50": float(
                np.median(sample_bands["temperature_surface"])
            ),
            "temperature_surface_dex_p95": float(
                np.quantile(sample_bands["temperature_surface"], 0.95)
            ),
            "temperature_deep_dex_p50": float(np.median(deep_error)),
            "temperature_deep_dex_p95": float(np.quantile(deep_error, 0.95)),
            "temperature_deep_dex_p99": float(np.quantile(deep_error, 0.99)),
            "by_effective_temperature": bin_rows,
            "deep_error_peak_layer_relative_to_convective_onset": {
                "all_stars_with_convection_count": int(has_convection.sum()),
                "median_layer_offset": float(np.median(offset)),
                "within_five_layers_fraction": float((np.abs(offset) <= 5).mean()),
                "top_five_percent_error_count": int(len(top_offset)),
                "top_five_percent_median_layer_offset": float(np.median(top_offset)),
                "top_five_percent_within_five_layers_fraction": float(
                    (np.abs(top_offset) <= 5).mean()
                ),
            },
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
