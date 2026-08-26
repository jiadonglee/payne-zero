"""Run the held-out H3 local-opacity closure probe."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
    load_strict_truth,
    make_split,
)
from .physical_opacity import (
    fit_local_opacity_parameters,
    integrate_self_consistent_mass,
    predict_local_log_opacity,
    profile_invariants,
)
from .profile_closure import integrate_mass_from_opacity


DEFAULT_OUTPUT = Path("results/analytic_initializer/h3_local_opacity_probe.json")
DEFAULT_EXCLUSION_MANIFESTS = (
    Path("results/reconstruction_metrics.json"),
    Path("results/sealed_solver_subset_20260808.json"),
    Path("results/sealed_audit_20260808.json"),
    Path("results/sealed_audit_20260811.json"),
    Path("results/sealed_initializer_holdout_20260812.json"),
    Path("results/initializer_calibration_20260812.json"),
    Path("results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json"),
)


def _score(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    absolute = np.abs(residual)
    return {
        "rmse_dex": float(np.sqrt(np.mean(residual**2))),
        "p50_dex": float(np.percentile(absolute, 50.0)),
        "p95_dex": float(np.percentile(absolute, 95.0)),
        "max_dex": float(np.max(absolute)),
    }


def main() -> None:
    corpus = load_strict_truth(DEFAULT_CORPUS)
    excluded, used_manifests = collect_excluded_indices(
        DEFAULT_EXCLUSION_MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(
        corpus.size,
        excluded=excluded,
        validation_fraction=0.2,
        seed=20260816,
    )
    generator = np.random.default_rng(20260816)
    validation_stars = np.sort(
        generator.choice(split.validation, size=min(1200, split.validation.size), replace=False)
    )
    labels = corpus.labels[validation_stars]
    temperature = corpus.temperature[validation_stars]
    pressure = corpus.gas_pressure[validation_stars]
    truth_opacity = corpus.rosseland_opacity[validation_stars]
    truth_mass = corpus.column_mass[validation_stars]

    results: dict[str, object] = {
        "candidate": "H3_local_positive_log_opacity_then_hydrostatic_mass_integral",
        "corpus": str(corpus.path),
        "corpus_sha256": file_sha256(corpus.path),
        "seed": 20260816,
        "train_count": int(split.train.size),
        "validation_count": int(split.validation.size),
        "validation_probe_star_count": int(validation_stars.size),
        "excluded_count": int(excluded.size),
        "excluded_manifests": used_manifests,
        "temperature_boundaries_K": [5500.0, 7500.0],
        "smooth_width_K": 250.0,
        "fits": {},
    }

    for degree in (2, 3):
        parameters, fit_metrics = fit_local_opacity_parameters(
            corpus,
            split,
            degree=degree,
            max_training_rows=120_000,
            seed=20260816,
        )
        fit_result: dict[str, object] = {
            "fit": fit_metrics,
            "term_count": int(parameters.term_count),
            "coefficient_count": int(parameters.coefficient_count),
            "constants_if_three_smooth_regimes": int(3 * parameters.term_count),
            "hard": {},
            "smooth": {},
        }
        for smooth_name, smooth in (("hard", False), ("smooth", True)):
            predicted_log_opacity = predict_local_log_opacity(
                labels, temperature, pressure, corpus.tau, parameters, smooth=smooth
            )
            opacity_score = _score(np.log10(truth_opacity), predicted_log_opacity)
            mass_from_truth_pressure = integrate_mass_from_opacity(
                corpus.tau, predicted_log_opacity
            )
            mass_score = _score(np.log10(truth_mass), np.log10(mass_from_truth_pressure))
            self_consistent_mass, self_consistent_log_opacity = integrate_self_consistent_mass(
                labels, temperature, corpus.tau, parameters, iterations=6
            )
            self_score = _score(np.log10(truth_mass), np.log10(self_consistent_mass))
            fit_result[smooth_name] = {
                "opacity": opacity_score,
                "mass_using_truth_P_T": mass_score,
                "mass_using_P_equals_gm_and_truth_T": self_score,
                "invariants": profile_invariants(self_consistent_mass, temperature),
                "self_consistent_opacity": _score(
                    np.log10(truth_opacity), self_consistent_log_opacity
                ),
            }
        results["fits"][str(degree)] = fit_result

    output = DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
