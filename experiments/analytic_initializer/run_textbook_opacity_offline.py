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
    textbook_opacity_window_components,
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


LAYER_TEMPERATURE_BINS = (
    ("3200_4000K", 3200.0, 4000.0),
    ("4000_5000K", 4000.0, 5000.0),
    ("5000_6000K", 5000.0, 6000.0),
    ("6000_7000K", 6000.0, 7000.0),
    ("7000_8000K", 7000.0, 8000.0),
    ("8000_10000K", 8000.0, 10000.0),
    ("10000_15000K", 10000.0, 15000.0),
    ("at_least_15000K", 15000.0, np.inf),
)


def _teff_layer_temperature_summary(
    residual: np.ndarray,
    labels: np.ndarray,
    temperature: np.ndarray,
    layer_mask: np.ndarray,
) -> list[dict[str, object]]:
    """Report signed and absolute errors in the requested two-dimensional bins."""

    rows: list[dict[str, object]] = []
    for teff_band, teff_mask in _band_masks(labels[:, 0]).items():
        for layer_band, lower, upper in LAYER_TEMPERATURE_BINS:
            mask = (
                teff_mask[:, None]
                & layer_mask
                & (temperature >= lower)
                & (temperature < upper)
            )
            if not np.any(mask):
                continue
            metrics = _metrics(residual[mask])
            rows.append(
                {
                    "teff_band": teff_band,
                    "layer_temperature_band": layer_band,
                    "layer_temperature_lower_K": lower,
                    "layer_temperature_upper_K": None if np.isinf(upper) else upper,
                    **metrics,
                }
            )
    return rows


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
    *,
    component_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Summarize the physical component that carries each temperature band."""

    names = (
        tuple(component_names)
        if component_names is not None
        else tuple(name for name in components if name != "total")
    )
    component_sum = np.maximum(
        sum(np.asarray(components[name], dtype=np.float64) for name in names),
        1.0e-30,
    )
    fractions = {
        name: np.asarray(components[name], dtype=np.float64) / component_sum
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


def _window_diagnostics(
    components: dict[str, np.ndarray],
    labels: np.ndarray,
    layer_mask: np.ndarray,
) -> dict[str, object]:
    weights = np.asarray(components["window_weights"], dtype=np.float64)
    opacity = np.asarray(components["window_opacity"], dtype=np.float64)
    result: dict[str, object] = {}
    for band, star_mask in _band_masks(labels[:, 0]).items():
        mask = star_mask[:, None] & layer_mask
        if not np.any(mask):
            continue
        result[band] = {
            "star_count": int(np.sum(star_mask)),
            "layer_count": int(np.sum(mask)),
            "median_rosseland_weight": [
                float(np.median(weights[:, :, index][mask]))
                for index in range(opacity.shape[-1])
            ],
            "median_window_opacity_cm2_per_g": [
                float(np.median(opacity[:, :, index][mask]))
                for index in range(opacity.shape[-1])
            ],
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.out is None:
        args.out = Path(
            "results/analytic_initializer/"
            f"textbook_opacity_{args.version}_offline_validation.json"
        )

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
    if args.version == "v3":
        components = textbook_opacity_window_components(labels, temperature, pressure)
        component_names = (
            "hminus_boundfree",
            "hminus_freefree",
            "hydrogen_paschen_boundfree",
            "hydrogen_balmer_boundfree",
            "kramers_freefree",
            "electron_scattering",
            "hydrogen_rayleigh_scattering",
        )
        temperature_floor = 3200.0
    else:
        components = textbook_opacity_components(labels, temperature, pressure)
        component_names = tuple(name for name in components if name != "total")
        temperature_floor = 2500.0
    prediction = components["total"]
    opacity_residual = np.log10(prediction) - np.log10(truth_opacity)
    applicable_layers = temperature >= temperature_floor
    out_of_domain_layers = ~applicable_layers

    # This is still an offline control: it uses the true P and T to isolate
    # opacity/integration error, never the self-consistent pressure loop.
    integrated_mass = integrate_mass_from_opacity(
        corpus.tau, np.log10(prediction)
    )
    mass_residual = np.log10(integrated_mass) - np.log10(corpus.column_mass[indices])

    component_fraction = {
        name: float(
            np.median(
                (
                    components[name]
                    / np.maximum(
                        sum(components[item] for item in component_names),
                        1.0e-30,
                    )
                )[applicable_layers]
            )
        )
        for name in component_names
    }
    band_summary = []
    component_diagnostics = _component_diagnostics(
        components,
        labels,
        applicable_layers,
        component_names=component_names,
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
    mass_cool_gate = _metrics(
        mass_residual[(labels[:, 0] < 6000.0)[:, None] & applicable_layers]
    )
    mass_middle_gate = _metrics(
        mass_residual[
            ((labels[:, 0] >= 6000.0) & (labels[:, 0] < 10000.0))[:, None]
            & applicable_layers
        ]
    )
    formal_opacity_pass = bool(
        cool_gate["p95_dex"] <= 0.30 and middle_gate["p95_dex"] <= 0.50
    )
    bridge_allowance_applies = bool(
        cool_gate["p95_dex"] <= 0.40 and middle_gate["p95_dex"] <= 0.60
    )
    bridge_pass = bool(
        bridge_allowance_applies
        and mass_cool_gate["p95_dex"] <= 0.20
        and mass_middle_gate["p95_dex"] <= 0.20
    )
    two_dimensional_summary = _teff_layer_temperature_summary(
        opacity_residual, labels, temperature, applicable_layers
    )
    result = {
        "candidate": (
            "named_constant_saha_hminus_gray_hbf_kramers_ff_window_v3"
            if args.version == "v3"
            else "named_constant_saha_hminus_gray_hbf_kramers_ff_window_v2"
        ),
        "version": args.version,
        "status": (
            "offline_fail_stop"
            if args.version == "v3" and not (formal_opacity_pass or bridge_pass)
            else "offline_only_not_production"
        ),
        "decision": (
            "FAIL_STOP"
            if args.version == "v3" and not (formal_opacity_pass or bridge_pass)
            else "legacy_v2_record"
        ),
        "next_registered_stage": (
            "blocked_before_ode_temperature_ablation_smoke_and_funnel"
            if args.version == "v3" and not (formal_opacity_pass or bridge_pass)
            else "not_applicable_to_legacy_v2_record"
        ),
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
        "component_median_fraction_of_component_sum": component_fraction,
        "component_diagnostics_by_band": component_diagnostics,
        "band_summary": band_summary,
        "teff_x_layer_temperature_summary": two_dimensional_summary,
        "window_diagnostics_by_band": (
            _window_diagnostics(components, labels, applicable_layers)
            if args.version == "v3"
            else None
        ),
        "applicability": {
            "temperature_floor_K": temperature_floor,
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
            "low_temperature_layers_excluded_from_gate_below_K": temperature_floor,
            "formal_opacity_pass": formal_opacity_pass,
            "bridge_allowance_max_excess_dex": 0.10,
            "bridge_mass_p95_max_dex": 0.20,
            "bridge_allowance_applies": bridge_allowance_applies,
            "bridge_pass": bridge_pass,
            "pass": bool(formal_opacity_pass or bridge_pass),
            "cool_observed_p95_dex": cool_gate["p95_dex"],
            "middle_observed_p95_dex": middle_gate["p95_dex"],
            "cool_mass_observed_p95_dex": mass_cool_gate["p95_dex"],
            "middle_mass_observed_p95_dex": mass_middle_gate["p95_dex"],
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
