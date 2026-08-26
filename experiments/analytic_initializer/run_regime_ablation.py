"""Separate a segmentation artefact from missing physics in the H2 deep error.

``results/analytic_initializer/deep_closure_localization.json`` shows the H2
deep-band temperature error peaking between 7000 and 8000 K, where a deep
convection zone stops being universal.  Two very different causes predict that
same peak:

* the formula is missing a radiative/convective closure, so a smooth low-rank
  map has to average over a bifurcation it cannot represent; or
* the fit is segmented at hard 5500/7500 K seams -- and the 7500 K seam sits
  inside that very transition, so each regime is fitted to a mixture.

This probe varies only the segmentation and the basis size, never the physics.
Whatever error survives all four configurations is the part a closure would
have to earn.  No solver calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.deep_diagnostics import (
    bin_by_effective_temperature,
    convective_diagnostics,
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

# The two bins where the localization probe found the deep error peak.
PEAK_BINS = ((7000, 7500), (7500, 8000))
# The quiet bin immediately below the transition, used as the target level: a
# closure that works should bring the peak bins down to roughly this.
REFERENCE_BIN = (6500, 7000)

CONFIGURATIONS = (
    {
        "name": "baseline",
        "rationale": "current H2: hard 5500/7500 K seams, degree 3, five modes",
        "components": 5,
        "regime_boundaries": (5500.0, 7500.0),
        "smoothing_width_K": 0.0,
    },
    {
        "name": "shifted",
        "rationale": "seams moved clear of the transition, everything else fixed",
        "components": 5,
        "regime_boundaries": (6300.0, 8700.0),
        "smoothing_width_K": 0.0,
    },
    {
        "name": "smooth",
        "rationale": "same seams and same fitted constants, blended at predict time",
        "components": 5,
        "regime_boundaries": (5500.0, 7500.0),
        "smoothing_width_K": 250.0,
    },
    {
        "name": "capacity_8",
        "rationale": "more depth modes, segmentation untouched",
        "components": 8,
        "regime_boundaries": (5500.0, 7500.0),
        "smoothing_width_K": 0.0,
    },
    {
        "name": "capacity_12",
        "rationale": "more depth modes still, segmentation untouched",
        "components": 12,
        "regime_boundaries": (5500.0, 7500.0),
        "smoothing_width_K": 0.0,
    },
)


def _bin_lookup(rows: list[dict[str, object]]) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (int(row["effective_temperature_low"]), int(row["effective_temperature_high"])): row
        for row in rows
    }


def _evaluate(
    corpus,
    split,
    sample: np.ndarray,
    onset: np.ndarray,
    configuration: dict,
) -> dict[str, object]:
    parameters = fit_analytic_profile_parameters(
        corpus,
        split,
        degree=3,
        components=int(configuration["components"]),
        regime_boundaries=tuple(configuration["regime_boundaries"]),
        smoothing_width_K=float(configuration["smoothing_width_K"]),
    )
    mass, temperature, log_opacity = predict_analytic_reduced_state(
        corpus.labels[sample], corpus.tau, parameters
    )
    bands = error_bands(
        corpus, sample, mass=mass, temperature=temperature, log_opacity=log_opacity
    )
    deep_error = bands["temperature_deep"]
    bin_rows = bin_by_effective_temperature(
        corpus.labels[sample, 0], deep_error, onset
    )
    lookup = _bin_lookup(bin_rows)
    peak = [float(lookup[key]["temperature_deep_dex_p95"]) for key in PEAK_BINS]
    return {
        "name": configuration["name"],
        "rationale": configuration["rationale"],
        "components": int(configuration["components"]),
        "regime_boundaries": [float(value) for value in configuration["regime_boundaries"]],
        "smoothing_width_K": float(configuration["smoothing_width_K"]),
        "coefficient_count": parameters.coefficient_count,
        "basis_value_count": parameters.basis_value_count,
        "stored_constant_count": parameters.coefficient_count + parameters.basis_value_count,
        "temperature_surface_dex_p95": float(
            np.quantile(bands["temperature_surface"], 0.95)
        ),
        "temperature_deep_dex_p50": float(np.median(deep_error)),
        "temperature_deep_dex_p95": float(np.quantile(deep_error, 0.95)),
        "mass_deep_dex_p95": float(np.quantile(bands["mass_deep"], 0.95)),
        "peak_bin_temperature_deep_dex_p95": peak,
        "worst_peak_bin_temperature_deep_dex_p95": max(peak),
        "reference_bin_temperature_deep_dex_p95": float(
            lookup[REFERENCE_BIN]["temperature_deep_dex_p95"]
        ),
        "by_effective_temperature": bin_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=3000)
    parser.add_argument("--sample-seed", type=int, default=7)
    parser.add_argument("--constant-budget", type=int, default=600)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/analytic_initializer/regime_ablation.json"),
    )
    args = parser.parse_args(argv)

    corpus = load_strict_truth(DEFAULT_CORPUS)
    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=20260816)

    # Same seed and same draw as the localization probe, so the baseline row
    # here must reproduce that artifact's numbers rather than merely resemble
    # them.
    generator = np.random.default_rng(args.sample_seed)
    sample = generator.choice(split.validation, args.sample_count, replace=False)
    _subadiabatic, onset = convective_diagnostics(corpus, sample)

    rows = [
        _evaluate(corpus, split, sample, onset, configuration)
        for configuration in CONFIGURATIONS
    ]
    baseline = rows[0]
    reference = baseline["reference_bin_temperature_deep_dex_p95"]
    for row in rows:
        row["peak_over_reference_ratio"] = float(
            row["worst_peak_bin_temperature_deep_dex_p95"] / reference
        )
        row["within_constant_budget"] = bool(
            row["stored_constant_count"] <= args.constant_budget
        )

    # Thresholds fixed before the run so the verdict cannot be read off the
    # numbers after the fact.
    segmentation_rows = [row for row in rows if row["name"] in ("shifted", "smooth")]
    capacity_rows = [row for row in rows if row["name"].startswith("capacity")]
    best_segmentation = min(
        row["worst_peak_bin_temperature_deep_dex_p95"] for row in segmentation_rows
    )
    best_capacity = min(
        row["worst_peak_bin_temperature_deep_dex_p95"] for row in capacity_rows
    )
    worst_overall = min(row["worst_peak_bin_temperature_deep_dex_p95"] for row in rows)
    gate = {
        "repaired_threshold_dex": 0.03,
        "physics_indicated_threshold_dex": 0.06,
        "best_segmentation_peak_dex": float(best_segmentation),
        "best_capacity_peak_dex": float(best_capacity),
        "best_any_configuration_peak_dex": float(worst_overall),
        "segmentation_explains_the_peak": bool(best_segmentation <= 0.03),
        "capacity_explains_the_peak": bool(best_capacity <= 0.03),
        "physics_closure_indicated": bool(worst_overall > 0.06),
    }

    result = {
        "gate_b": gate,
        "format": "payne_zero_analytic_initializer_regime_ablation_v1",
        "candidate": "H2_standalone_low_rank_hopf_and_opacity_profile",
        "status": "offline_diagnostic_no_solver_calls",
        "question": (
            "Is the 7000-8000 K deep-error peak caused by the hard temperature "
            "seams or by a missing radiative/convective closure?"
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
        "sample_count": int(sample.size),
        "sample_seed": int(args.sample_seed),
        "peak_bins": [list(item) for item in PEAK_BINS],
        "reference_bin": list(REFERENCE_BIN),
        "constant_budget": int(args.constant_budget),
        "configurations": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"{'configuration':14s} {'constants':>10s} {'peak p95':>9s} "
        f"{'ref p95':>8s} {'ratio':>6s}"
    )
    for row in rows:
        print(
            f"{row['name']:14s} {row['stored_constant_count']:10d} "
            f"{row['worst_peak_bin_temperature_deep_dex_p95']:9.4f} "
            f"{row['reference_bin_temperature_deep_dex_p95']:8.4f} "
            f"{row['peak_over_reference_ratio']:6.2f}"
        )
    print(json.dumps(gate, indent=2, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
