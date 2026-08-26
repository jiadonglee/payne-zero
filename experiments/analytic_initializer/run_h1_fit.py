"""Fit and evaluate the first effective-opacity analytic candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .candidates import (
    build_h1_reduced_state,
    fit_scalar_opacity_parameters,
)
from .discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
    load_strict_truth,
    make_split,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "analytic_initializer" / "h1_effective_opacity.json"
MANIFESTS = (
    REPO_ROOT / "results" / "reconstruction_metrics.json",
    REPO_ROOT / "results" / "sealed_solver_subset_20260808.json",
    REPO_ROOT / "results" / "sealed_audit_20260808.json",
    REPO_ROOT / "results" / "sealed_audit_20260811.json",
    REPO_ROOT / "results" / "sealed_initializer_holdout_20260812.json",
    REPO_ROOT / "results" / "initializer_calibration_20260812.json",
    REPO_ROOT / "results" / "four_initializer_benchmark_expanded_20260814" / "expanded200_manifest.json",
)


def _serialize_parameters(parameters) -> dict[str, object]:
    return {
        "degree": parameters.degree,
        "exponents": parameters.exponents.tolist(),
        "feature_center": parameters.feature_center.tolist(),
        "feature_scale": parameters.feature_scale.tolist(),
        "coefficients": parameters.coefficients.tolist(),
        "regime_names": list(parameters.regime_names),
    }


def run(*, corpus_path: Path, output_path: Path) -> dict[str, object]:
    corpus = load_strict_truth(corpus_path)
    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded)
    parameters, metrics = fit_scalar_opacity_parameters(corpus, split, degree=2)
    validation_labels = corpus.labels[split.validation]
    mass, temperature, opacity = build_h1_reduced_state(
        validation_labels,
        corpus.tau,
        parameters=parameters,
    )
    truth_mass = corpus.column_mass[split.validation]
    truth_temperature = corpus.temperature[split.validation]
    mass_error = np.abs(np.log10(mass) - np.log10(truth_mass))
    temperature_error = np.abs(
        np.log10(temperature) - np.log10(truth_temperature)
    )
    result: dict[str, object] = {
        "format": "payne_zero_analytic_initializer_h1_v1",
        "candidate": "eddington_temperature_plus_piecewise_effective_opacity",
        "corpus": {
            "path": str(corpus.path),
            "sha256": file_sha256(corpus.path),
            "star_count": corpus.size,
        },
        "excluded_manifests": used_manifests,
        "split": {
            "seed": split.seed,
            "train_count": int(split.train.size),
            "validation_count": int(split.validation.size),
            "excluded_count": int(split.excluded.size),
        },
        "fit_metrics": metrics,
        "profile_metrics": {
            "temperature_dex_p50": float(np.percentile(temperature_error, 50.0)),
            "temperature_dex_p95": float(np.percentile(temperature_error, 95.0)),
            "temperature_dex_max": float(np.max(temperature_error)),
            "column_mass_dex_p50": float(np.percentile(mass_error, 50.0)),
            "column_mass_dex_p95": float(np.percentile(mass_error, 95.0)),
            "column_mass_dex_max": float(np.max(mass_error)),
            "effective_opacity_quantiles": [
                float(value) for value in np.percentile(opacity, [1, 50, 99])
            ],
        },
        "parameters": _serialize_parameters(parameters),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    result = run(corpus_path=args.corpus, output_path=args.out)
    print(json.dumps(result["fit_metrics"], indent=2))
    print(json.dumps(result["profile_metrics"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
