"""Decisive Gate-1 test: is the deep-tail residual a smooth label function?


The entropy-closure plan asks for a <=600-constant formula that turns labels
into (m, T).  Earlier work showed a per-star optimizer on the deep 7000-8000 K
residual can reach ~0.015 dex p95 for a fixed-base tanh family (a good
representation), and the local-oracle veto test recorded the family's failure.
This runner measures the binding constraint: whether a *shared* label-conditioned
map of that same family can carry the deep correction to held-out stars.  The
answer, per the plan's pre-registered rule, decides whether any compact
no-emulator formula can pass Gate 1 for the peak bins.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from experiments.analytic_initializer.deep_diagnostics import deep_window
from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
    label_features,
    load_strict_truth,
    make_split,
    polynomial_exponents,
    polynomial_features,
)
from experiments.analytic_initializer.profile_initializer import (
    load_analytic_profile_parameters,
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

H2_PARAMETERS = Path("results/analytic_initializer/h2_profile_parameters_v1.npz")
FIXED_KNOTS = np.asarray((58.0, 62.0, 66.0, 70.0, 74.0))


def fixed_base(layers: int) -> np.ndarray:
    """(layers, 7): const + linear(l-60) + 5 fixed-width tanh modes."""

    depth = np.arange(layers, dtype=np.float64)
    return np.column_stack(
        [np.ones(layers), depth - 60.0]
        + [np.tanh((depth - knot) / 2.0) for knot in FIXED_KNOTS]
    )


def shared_map(corpus, split, target, degree):
    """Return (center, scale, exponents, W) fitting label-poly x fixed-base."""

    base = fixed_base(corpus.layers)
    features = label_features(corpus.labels)
    center = features[split.train].mean(axis=0)
    scale = np.maximum(features[split.train].std(axis=0), 1.0e-12)
    normalized = (features - center) / scale
    exponents = polynomial_exponents(features.shape[1], degree)
    design, _, _ = polynomial_features(
        normalized[split.train], exponents, center=np.zeros(5), scale=np.ones(5)
    )
    row_coefficients = np.linalg.lstsq(
        base, target[split.train].T, rcond=None
    )[0].T  # (N_train, 7)
    weight = np.linalg.lstsq(design, row_coefficients, rcond=None)[0]  # (Q, 7)
    return center, scale, exponents, weight


def score(corpus, split, target, baseline, weight, base, center, scale, exponents, indices):
    """Return per-row deep temperature error (in dex) on ``indices``."""

    features = label_features(corpus.labels[indices])
    normalized = (features - center) / np.maximum(scale, 1.0e-12)
    design, _, _ = polynomial_features(
        normalized, exponents, center=np.zeros(5), scale=np.ones(5)
    )
    correction = (design @ weight) @ base.T
    start, stop = deep_window(corpus.layers)
    predicted_log_temperature = np.log10(baseline[indices]) + correction
    error = np.abs(
        predicted_log_temperature - np.log10(corpus.temperature[indices])
    )
    return error[:, start:stop].max(axis=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/analytic_initializer/deep_tail_smoothness.json"),
    )
    parser.add_argument(
        "--band-sample",
        type=int,
        default=3000,
        help="cap on the 7000-8000 K validation sample reported per band",
    )
    args = parser.parse_args(argv)

    corpus = load_strict_truth(DEFAULT_CORPUS)
    excluded, used = collect_excluded_indices(MANIFESTS, corpus_size=corpus.size)
    split = make_split(corpus.size, excluded=excluded, seed=20260816)

    parameters = load_analytic_profile_parameters(H2_PARAMETERS)
    _, baseline, _ = predict_analytic_reduced_state(
        corpus.labels, corpus.tau, parameters
    )
    target = np.log10(corpus.temperature) - np.log10(baseline)

    base = fixed_base(corpus.layers)
    center, scale, exponents, weight = shared_map(corpus, split, target, args.degree)

    indices = split.validation
    deep_error = score(
        corpus, split, target, baseline, weight, base,
        center, scale, exponents, indices,
    )
    teff = corpus.labels[indices, 0]
    band = (teff >= 7000.0) & (teff < 8000.0)
    band_indices = indices[band]
    rng = np.random.default_rng(7)
    keep = rng.choice(
        band_indices.size, size=min(args.band_sample, int(band_indices.size)), replace=False
    )
    band_deep = deep_error[band][keep]
    band_teff = teff[band][keep]

    def _summary(values, temperatures):
        out = {"n": int(values.size)}
        for label in ("p50", "p95", "max"):
            fn = {"p50": lambda a: float(np.median(a)),
                  "p95": lambda a: float(np.percentile(a, 95.0)),
                  "max": lambda a: float(np.max(a))}[label]
            out[label] = fn(values)
        for low, high in ((7000.0, 7500.0), (7500.0, 8000.0)):
            sub = (temperatures >= low) & (temperatures < high)
            out[f"{int(low)}-{int(high)}"] = {
                "n": int(sub.sum()),
                "p50": float(np.median(values[sub])),
                "p95": float(np.percentile(values[sub], 95.0)),
            }
        return out

    artifact = {
        "format": "payne_zero_deep_tail_smoothness_v1",
        "question": (
            "Is the deep 7000-8000 K convective-tail residual a smooth function "
            "of the five labels within the 600-constant budget?"
        ),
        "corpus_path": str(DEFAULT_CORPUS),
        "corpus_sha256": file_sha256(DEFAULT_CORPUS),
        "split_seed": int(split.seed),
        "excluded_manifests_used": [str(path) for path in used],
        "excluded_star_count": int(excluded.size),
        "validation_count": int(split.validation.size),
        "degree": args.degree,
        "constant_count": int(exponents.shape[0] * 7),
        "result_all_validation": _summary(deep_error, teff),
        "result_7000_8000_band": _summary(band_deep, band_teff),
        "targets": {"gate1_deep_p95": 0.020, "plan_oracle_veto_threshold_dex": 0.015},
        "reproducer": "experiments/analytic_initializer/run_deep_tail_smoothness.py",
        "date": "2026-08-16",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
