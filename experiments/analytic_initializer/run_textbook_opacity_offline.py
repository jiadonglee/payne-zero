"""Offline validation of the named-constant textbook opacity candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
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
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot summarize an empty or non-finite residual set")
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "signed_median_dex": float(np.median(values)),
        "signed_mean_dex": float(np.mean(values)),
        "positive_fraction": float(np.mean(values > 0.0)),
        "p50_dex": float(np.percentile(absolute, 50.0)),
        "p95_dex": float(np.percentile(absolute, 95.0)),
        "max_dex": float(np.max(absolute)),
        "rmse_dex": float(np.sqrt(np.mean(values**2))),
    }


def _band_masks(teff: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(teff, dtype=np.float64)
    return {
        "cool_below_6000K": values < 6000.0,
        "known_transition_5500_7000K": (values >= 5500.0) & (values < 7000.0),
        "middle_6000_10000K": (values >= 6000.0) & (values < 10000.0),
        "hot_at_least_10000K": values >= 10000.0,
    }


def _profile_metrics(
    residual: np.ndarray,
    labels: np.ndarray,
    *,
    layer_mask: np.ndarray | None = None,
) -> dict[str, object]:
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("residual must have shape (N, layers)")
    if layer_mask is None:
        valid_layers = np.ones_like(values, dtype=bool)
    else:
        valid_layers = np.asarray(layer_mask, dtype=bool)
        if valid_layers.shape != values.shape:
            raise ValueError("layer_mask must match residual shape")
    if not np.any(valid_layers):
        return {
            "pooled": None,
            "star_p50_dex_quantiles": [],
            "star_p95_dex_quantiles": [],
            "profile_count": 0,
        }
    pooled = _metrics(values[valid_layers])
    per_star_p50 = []
    per_star_p95 = []
    for row, mask in zip(values, valid_layers, strict=True):
        row_values = row[mask]
        if row_values.size == 0:
            continue
        per_star_p50.append(float(np.percentile(np.abs(row_values), 50.0)))
        per_star_p95.append(float(np.percentile(np.abs(row_values), 95.0)))
    per_star_p50_array = np.asarray(per_star_p50, dtype=np.float64)
    per_star_p95_array = np.asarray(per_star_p95, dtype=np.float64)
    result: dict[str, object] = {
        "pooled": pooled,
        "star_p50_dex_quantiles": [
            float(value)
            for value in np.percentile(per_star_p50_array, [50.0, 95.0, 100.0])
        ],
        "star_p95_dex_quantiles": [
            float(value)
            for value in np.percentile(per_star_p95_array, [50.0, 95.0, 100.0])
        ],
        "profile_count": int(per_star_p50_array.size),
    }
    for name, mask in _band_masks(labels[:, 0]).items():
        band_layers = mask[:, None] & valid_layers
        if not np.any(band_layers):
            continue
        result[name] = _metrics(values[band_layers])
        result[f"{name}_star_count"] = int(
            np.sum(mask & np.any(valid_layers, axis=1))
        )
    return result


def _component_diagnostics(
    components: dict[str, np.ndarray],
    labels: np.ndarray,
    layer_mask: np.ndarray,
) -> dict[str, object]:
    """Summarize the physical component that carries each temperature band."""

    total = np.asarray(components["total"], dtype=np.float64)
    names = tuple(name for name in components if name != "total")
    fractions = {
        name: np.asarray(components[name], dtype=np.float64) / total
        for name in names
    }
    result: dict[str, object] = {}
    for band, star_mask in _band_masks(labels[:, 0]).items():
        mask = star_mask[:, None] & layer_mask
        if not np.any(mask):
            continue
        medians = {
            name: float(np.median(fractions[name][mask])) for name in names
        }
        dominant = max(medians, key=medians.get)
        result[band] = {
            "star_count": int(np.sum(star_mask)),
            "layer_count": int(np.sum(mask)),
            "dominant_component": dominant,
            "dominant_median_fraction": medians[dominant],
            "component_median_fraction": medians,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/analytic_initializer/textbook_opacity_v2_offline_validation.json"
        ),
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
    applicable_layers = temperature >= 2500.0
    out_of_domain_layers = ~applicable_layers

    # This is still an offline control: it uses the true P and T to isolate
    # opacity/integration error, never the self-consistent pressure loop.
    integrated_mass = integrate_mass_from_opacity(
        corpus.tau, np.log10(prediction)
    )
    mass_residual = np.log10(integrated_mass) - np.log10(corpus.column_mass[indices])

    component_fraction = {
        name: float(np.median((components[name] / prediction)[applicable_layers]))
        for name in components
        if name != "total"
    }
    band_summary = []
    component_diagnostics = _component_diagnostics(
        components, labels, applicable_layers
    )
    for band, star_mask in _band_masks(labels[:, 0]).items():
        if not np.any(star_mask):
            continue
        opacity_metrics = _metrics(
            opacity_residual[star_mask[:, None] & applicable_layers]
        )
        component_summary = component_diagnostics.get(band, {})
        band_summary.append(
            {
                "band": band,
                "star_count": int(np.sum(star_mask)),
                "signed_median_dex": opacity_metrics["signed_median_dex"],
                "signed_mean_dex": opacity_metrics["signed_mean_dex"],
                "abs_p95_dex": opacity_metrics["p95_dex"],
                "dominant_component": component_summary.get(
                    "dominant_component", "none"
                ),
                "dominant_median_fraction": component_summary.get(
                    "dominant_median_fraction", 0.0
                ),
            }
        )
    cool_gate = _metrics(
        opacity_residual[(labels[:, 0] < 6000.0)[:, None] & applicable_layers]
    )
    middle_gate = _metrics(
        opacity_residual[
            ((labels[:, 0] >= 6000.0) & (labels[:, 0] < 10000.0))[:, None]
            & applicable_layers
        ]
    )
    result = {
        "candidate": "named_constant_saha_hminus_gray_hbf_kramers_ff_window_v2",
        "status": "offline_only_not_production",
        "corpus": str(corpus.path),
        "corpus_sha256": file_sha256(corpus.path),
        "validation_star_count": int(indices.size),
        "validation_layer_count": int(indices.size * corpus.layers),
        "validation_indices": [int(index) for index in indices],
        "split_seed": split.seed,
        "excluded_count": int(excluded.size),
        "excluded_manifests": used_manifests,
        "constants": {
            name: value
            for name, value in DEFAULT_TEXTBOOK_CONSTANTS.__dict__.items()
        },
        "component_median_fraction_of_total": component_fraction,
        "component_diagnostics_by_band": component_diagnostics,
        "band_summary": band_summary,
        "applicability": {
            "temperature_floor_K": 2500.0,
            "excluded_layer_count": int(np.sum(out_of_domain_layers)),
            "applicable_layer_count": int(np.sum(applicable_layers)),
            "excluded_layer_fraction": float(np.mean(out_of_domain_layers)),
            "excluded_layers_reported_separately": True,
        },
        "nonfinite_count": int(
            np.sum(~np.isfinite(prediction)) + np.sum(~np.isfinite(integrated_mass))
        ),
        "opacity_profile_metrics": _profile_metrics(
            opacity_residual, labels, layer_mask=applicable_layers
        ),
        "opacity_out_of_domain_metrics": _profile_metrics(
            opacity_residual, labels, layer_mask=out_of_domain_layers
        ),
        "mass_profile_metrics_using_true_P_T": _profile_metrics(
            mass_residual, labels, layer_mask=applicable_layers
        ),
        "mass_out_of_domain_metrics_using_true_P_T": _profile_metrics(
            mass_residual, labels, layer_mask=out_of_domain_layers
        ),
        "offline_gate": {
            "cool_p95_max_dex": 0.30,
            "middle_p95_max_dex": 0.50,
            "transition_band_reported_separately": True,
            "low_temperature_layers_excluded_from_gate_below_K": 2500.0,
            "pass": bool(
                cool_gate["p95_dex"] <= 0.30
                and middle_gate["p95_dex"] <= 0.50
            ),
            "cool_observed_p95_dex": cool_gate["p95_dex"],
            "middle_observed_p95_dex": middle_gate["p95_dex"],
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
