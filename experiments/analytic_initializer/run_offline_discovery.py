"""Run the first offline low-rank discovery pass for the analytic initializer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .discovery import (
    DEFAULT_CORPUS,
    TARGET_FIELDS,
    collect_excluded_indices,
    file_sha256,
    fit_low_rank_surrogate,
    load_strict_truth,
    make_split,
    normalized_targets,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "analytic_initializer" / "offline_discovery.json"
DEFAULT_MANIFESTS = (
    REPO_ROOT / "results" / "reconstruction_metrics.json",
    REPO_ROOT / "results" / "sealed_solver_subset_20260808.json",
    REPO_ROOT / "results" / "sealed_audit_20260808.json",
    REPO_ROOT / "results" / "sealed_audit_20260811.json",
    REPO_ROOT / "results" / "sealed_initializer_holdout_20260812.json",
    REPO_ROOT / "results" / "initializer_calibration_20260812.json",
    REPO_ROOT / "results" / "four_initializer_benchmark_expanded_20260814" / "expanded200_manifest.json",
)


def run(*, corpus_path: Path, output_path: Path, seed: int = 20260816) -> dict:
    corpus = load_strict_truth(corpus_path)
    excluded, manifests = collect_excluded_indices(
        DEFAULT_MANIFESTS,
        corpus_size=corpus.size,
    )
    split = make_split(corpus.size, excluded=excluded, seed=seed)
    targets = normalized_targets(corpus)

    summary: dict[str, object] = {
        "format": "payne_zero_analytic_initializer_offline_discovery_v1",
        "corpus": {
            "path": str(corpus.path),
            "sha256": file_sha256(corpus.path),
            "star_count": corpus.size,
            "layer_count": corpus.layers,
            "tau_min": float(np.min(corpus.tau)),
            "tau_max": float(np.max(corpus.tau)),
        },
        "excluded_manifests": manifests,
        "split": {
            "seed": int(split.seed),
            "train_count": int(split.train.size),
            "validation_count": int(split.validation.size),
            "excluded_count": int(split.excluded.size),
        },
        "targets": {},
    }

    for target_name in TARGET_FIELDS:
        values = targets[target_name]
        target_summary = {
            "global_quantiles": [
                float(value) for value in np.percentile(values, [0.0, 1.0, 50.0, 99.0, 100.0])
            ],
            "fits": [],
        }
        for components in (3, 5, 8):
            for degree in (1, 2, 3):
                target_summary["fits"].append(
                    fit_low_rank_surrogate(
                        values,
                        corpus.labels,
                        split.train,
                        split.validation,
                        components=components,
                        degree=degree,
                    )
                )
        summary["targets"][target_name] = target_summary

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args(argv)
    summary = run(corpus_path=args.corpus, output_path=args.out, seed=args.seed)
    print(json.dumps(summary["split"], indent=2))
    for name, payload in summary["targets"].items():
        best = min(payload["fits"], key=lambda item: item["absolute_error_p95"])
        print(
            f"{name}: best degree={best['degree']} components={best['components']} "
            f"terms={best['term_count']} r2={best['r2']:.5f} "
            f"p95={best['absolute_error_p95']:.5g}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
