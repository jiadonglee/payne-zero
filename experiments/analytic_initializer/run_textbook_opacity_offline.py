"""Offline validation of the named-constant textbook opacity candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.profile_closure import integrate_mass_from_opacity
from experiments.analytic_initializer.textbook_opacity import (
    DEFAULT_TEXTBOOK_CONSTANTS,
    textbook_opacity_components,
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


def _metrics(residual: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(residual, dtype=np.float64))
    return {
        "p50_dex": float(np.percentile(absolute, 50.0)),
        "p95_dex": float(np.percentile(absolute, 95.0)),
        "max_dex": float(np.max(absolute)),
        "rmse_dex": float(np.sqrt(np.mean(np.asarray(residual) ** 2))),
    }


def _band_masks(teff: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(teff, dtype=np.float64)
    return {
        "cool_below_6000K": values < 6000.0,
        "known_transition_5500_7000K": (values >= 5500.0) & (values < 7000.0),
        "middle_6000_10000K": (values >= 6000.0) & (values < 10000.0),
        "hot_at_least_10000K": values >= 10000.0,
    }


def _profile_metrics(residual: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    pooled = _metrics(residual.reshape(-1))
    per_star_p50 = np.percentile(np.abs(residual), 50.0, axis=1)
    per_star_p95 = np.percentile(np.abs(residual), 95.0, axis=1)
    result: dict[str, object] = {
        "pooled": pooled,
        "star_p50_dex_quantiles": [
            float(value) for value in np.percentile(per_star_p50, [50.0, 95.0, 100.0])
        ],
        "star_p95_dex_quantiles": [
            float(value) for value in np.percentile(per_star_p95, [50.0, 95.0, 100.0])
        ],
    }
    for name, mask in _band_masks(labels[:, 0]).items():
        if not np.any(mask):
            continue
        result[name] = _metrics(residual[mask].reshape(-1))
        result[f"{name}_star_count"] = int(np.sum(mask))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/analytic_initializer/textbook_opacity_offline_validation.json"),
    )
    args = parser.parse_args(argv)

    corpus = load_strict_truth(args.corpus)
    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=20260816)
    indices = split.validation
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        indices = indices[: int(args.limit)]

    labels = corpus.labels[indices]
    temperature = corpus.temperature[indices]
    pressure = corpus.gas_pressure[indices]
    truth_opacity = corpus.rosseland_opacity[indices]
    components = textbook_opacity_components(labels, temperature, pressure)
    prediction = components["total"]
    opacity_residual = np.log10(prediction) - np.log10(truth_opacity)

    # This is still an offline control: it uses the true P and T to isolate
    # opacity/integration error, never the self-consistent pressure loop.
    integrated_mass = integrate_mass_from_opacity(
        corpus.tau, np.log10(prediction)
    )
    mass_residual = np.log10(integrated_mass) - np.log10(corpus.column_mass[indices])

    component_fraction = {
        name: float(np.median(components[name] / prediction))
        for name in components
        if name != "total"
    }
    result = {
        "candidate": "named_constant_saha_hminus_hbf_kramers_es",
        "status": "offline_only_not_production",
        "corpus": str(corpus.path),
        "validation_indices": [int(index) for index in indices],
        "split_seed": split.seed,
        "excluded_count": int(excluded.size),
        "excluded_manifests": used_manifests,
        "constants": {
            name: value
            for name, value in DEFAULT_TEXTBOOK_CONSTANTS.__dict__.items()
        },
        "component_median_fraction_of_total": component_fraction,
        "nonfinite_count": int(
            np.sum(~np.isfinite(prediction)) + np.sum(~np.isfinite(integrated_mass))
        ),
        "opacity_profile_metrics": _profile_metrics(opacity_residual, labels),
        "mass_profile_metrics_using_true_P_T": _profile_metrics(mass_residual, labels),
        "offline_gate": {
            "cool_p95_max_dex": 0.30,
            "middle_p95_max_dex": 0.50,
            "transition_band_reported_separately": True,
            "pass": bool(
                _metrics(opacity_residual[labels[:, 0] < 6000.0].reshape(-1))["p95_dex"]
                <= 0.30
                and _metrics(
                    opacity_residual[
                        (labels[:, 0] >= 6000.0) & (labels[:, 0] < 10000.0)
                    ].reshape(-1)
                )["p95_dex"]
                <= 0.50
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "opacity": result["opacity_profile_metrics"],
                "mass_using_true_P_T": result["mass_profile_metrics_using_true_P_T"],
                "offline_gate": result["offline_gate"],
            },
            sort_keys=True,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
