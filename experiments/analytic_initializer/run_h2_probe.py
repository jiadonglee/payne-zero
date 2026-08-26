"""Measure whether a predicted opacity profile can recover column mass."""

from __future__ import annotations

import argparse
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
from .profile_closure import (
    fit_profile_closure,
    integrate_mass_from_opacity,
    predict_profile_closure,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "analytic_initializer" / "h2_opacity_profile_probe.json"
MANIFESTS = (
    REPO_ROOT / "results" / "reconstruction_metrics.json",
    REPO_ROOT / "results" / "sealed_solver_subset_20260808.json",
    REPO_ROOT / "results" / "sealed_audit_20260808.json",
    REPO_ROOT / "results" / "sealed_audit_20260811.json",
    REPO_ROOT / "results" / "sealed_initializer_holdout_20260812.json",
    REPO_ROOT / "results" / "initializer_calibration_20260812.json",
    REPO_ROOT / "results" / "four_initializer_benchmark_expanded_20260814" / "expanded200_manifest.json",
)


def run(*, corpus_path: Path, output_path: Path) -> dict[str, object]:
    corpus = load_strict_truth(corpus_path)
    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded)
    target = np.log10(corpus.rosseland_opacity)
    records: list[dict[str, object]] = []

    for degree, components in ((2, 3), (3, 3), (3, 5), (3, 8)):
        parameters = fit_profile_closure(
            corpus,
            split,
            target=target,
            degree=degree,
            components=components,
        )
        predicted_log_kappa = predict_profile_closure(
            corpus.labels[split.validation],
            corpus.tau,
            parameters,
        )
        predicted_mass = integrate_mass_from_opacity(
            corpus.tau,
            predicted_log_kappa,
        )
        truth_log_kappa = target[split.validation]
        truth_mass = corpus.column_mass[split.validation]
        kappa_error = np.abs(predicted_log_kappa - truth_log_kappa)
        mass_error = np.abs(np.log10(predicted_mass) - np.log10(truth_mass))
        records.append(
            {
                "degree": degree,
                "components": components,
                "term_count": parameters.term_count,
                "opacity_dex_p95": float(np.percentile(kappa_error, 95.0)),
                "opacity_dex_max": float(np.max(kappa_error)),
                "mass_dex_p50": float(np.percentile(mass_error, 50.0)),
                "mass_dex_p95": float(np.percentile(mass_error, 95.0)),
                "mass_dex_max": float(np.max(mass_error)),
            }
        )

    # This is the physical closure floor: exact kappa with the same numerical
    # integral, before any approximation to opacity is introduced.
    exact_mass = integrate_mass_from_opacity(
        corpus.tau,
        target[split.validation],
    )
    exact_error = np.abs(np.log10(exact_mass) - np.log10(corpus.column_mass[split.validation]))
    result: dict[str, object] = {
        "format": "payne_zero_analytic_initializer_h2_opacity_profile_probe_v1",
        "candidate": "regimewise_low_rank_log_kappa_tau_then_mass_integral",
        "status": "diagnostic_not_final_formula",
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
        "exact_kappa_integration_floor": {
            "mass_dex_p50": float(np.percentile(exact_error, 50.0)),
            "mass_dex_p95": float(np.percentile(exact_error, 95.0)),
            "mass_dex_max": float(np.max(exact_error)),
        },
        "candidates": records,
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
    print(json.dumps(result["exact_kappa_integration_floor"], indent=2))
    for candidate in result["candidates"]:
        print(json.dumps(candidate, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
