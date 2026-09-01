"""Run the preregistered full-corpus v4 node-level offline validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
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
from experiments.analytic_initializer.run_textbook_opacity_offline import (
    MANIFESTS,
    _band_masks,
    _metrics,
    _profile_metrics,
    _teff_layer_temperature_summary,
)
from experiments.analytic_initializer.textbook_opacity import (
    DEFAULT_TEXTBOOK_CONSTANTS,
    textbook_opacity_node_components,
)


COMPONENT_NAMES = (
    "hminus_boundfree",
    "hminus_freefree",
    "hydrogen_boundfree",
    "hydrogen_freefree",
    "electron_scattering",
    "hydrogen_rayleigh_scattering",
)
TEMPERATURE_FLOOR_K = 3200.0


def _component_diagnostics(
    component_fraction: dict[str, np.ndarray],
    labels: np.ndarray,
    layer_mask: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for band, star_mask in _band_masks(labels[:, 0]).items():
        mask = star_mask[:, None] & layer_mask
        if not np.any(mask):
            continue
        medians = {
            name: float(np.median(values[mask]))
            for name, values in component_fraction.items()
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


def _batch_prediction(
    labels: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    prediction = np.empty_like(temperature, dtype=np.float64)
    component_fraction = {
        name: np.empty_like(temperature, dtype=np.float64)
        for name in COMPONENT_NAMES
    }
    for start in range(0, labels.shape[0], int(batch_size)):
        stop = min(labels.shape[0], start + int(batch_size))
        components = textbook_opacity_node_components(
            labels[start:stop],
            temperature[start:stop],
            pressure[start:stop],
        )
        weights = components["node_weights"]
        total = components["total"]
        prediction[start:stop] = 1.0 / np.sum(
            weights / np.maximum(total, 1.0e-30), axis=(-2, -1)
        )
        arithmetic_total = np.maximum(
            np.sum(weights * total, axis=(-2, -1)),
            1.0e-30,
        )
        for name in COMPONENT_NAMES:
            component_fraction[name][start:stop] = np.sum(
                weights * components[name], axis=(-2, -1)
            ) / arithmetic_total
        print(
            f"processed {stop}/{labels.shape[0]} validation stars",
            flush=True,
        )
    return prediction, component_fraction


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/analytic_initializer/textbook_opacity_v4_offline_validation.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    corpus = load_strict_truth(args.corpus)
    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=20260816)
    indices = split.validation
    if args.limit is not None:
        indices = indices[: int(args.limit)]

    labels = corpus.labels[indices]
    temperature = corpus.temperature[indices]
    pressure = corpus.gas_pressure[indices]
    truth_opacity = corpus.rosseland_opacity[indices]
    prediction, component_fraction = _batch_prediction(
        labels,
        temperature,
        pressure,
        batch_size=args.batch_size,
    )
    opacity_residual = np.log10(prediction) - np.log10(truth_opacity)
    applicable_layers = temperature >= TEMPERATURE_FLOOR_K
    out_of_domain_layers = ~applicable_layers

    integrated_mass = integrate_mass_from_opacity(
        corpus.tau,
        np.log10(prediction),
    )
    mass_residual = np.log10(integrated_mass) - np.log10(corpus.column_mass[indices])

    band_masks = _band_masks(labels[:, 0])
    cool_mask = band_masks["cool_below_6000K"][:, None] & applicable_layers
    middle_mask = band_masks["middle_6000_10000K"][:, None] & applicable_layers
    cool_gate = _metrics(opacity_residual[cool_mask])
    middle_gate = _metrics(opacity_residual[middle_mask])
    mass_cool_gate = _metrics(mass_residual[cool_mask])
    mass_middle_gate = _metrics(mass_residual[middle_mask])

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
    promotion_stage_pass = bool(formal_opacity_pass or bridge_pass)

    band_summary = []
    component_diagnostics = _component_diagnostics(
        component_fraction,
        labels,
        applicable_layers,
    )
    for band, star_mask in band_masks.items():
        if not np.any(star_mask):
            continue
        mask = star_mask[:, None] & applicable_layers
        opacity_metrics = _metrics(opacity_residual[mask])
        mass_metrics = _metrics(mass_residual[mask])
        component_summary = component_diagnostics.get(band, {})
        band_summary.append(
            {
                "band": band,
                "star_count": int(np.sum(star_mask)),
                "signed_median_dex": opacity_metrics["signed_median_dex"],
                "signed_mean_dex": opacity_metrics["signed_mean_dex"],
                "abs_p95_dex": opacity_metrics["p95_dex"],
                "mass_abs_p95_dex": mass_metrics["p95_dex"],
                "dominant_component": component_summary.get(
                    "dominant_component", "none"
                ),
                "dominant_median_fraction": component_summary.get(
                    "dominant_median_fraction", 0.0
                ),
            }
        )

    result = {
        "candidate": "named_constant_saha_node_frequency_john_hminus_hydrogenic_v4",
        "version": "v4",
        "status": "offline_pass_to_bridge" if promotion_stage_pass else "offline_fail_stop",
        "decision": "PROCEED_TO_ODE_AND_SMOKE" if promotion_stage_pass else "FAIL_STOP",
        "next_registered_stage": (
            "ode_temperature_ablation_smoke_and_funnel"
            if promotion_stage_pass
            else "blocked_before_ode_temperature_ablation_smoke_and_funnel"
        ),
        "corpus": str(corpus.path),
        "corpus_sha256": file_sha256(corpus.path),
        "validation_star_count": int(indices.size),
        "validation_layer_count": int(indices.size * corpus.layers),
        "validation_indices": [int(index) for index in indices],
        "split_seed": split.seed,
        "excluded_count": int(excluded.size),
        "excluded_manifests": used_manifests,
        "constants": asdict(DEFAULT_TEXTBOOK_CONSTANTS),
        "node_quadrature": {
            "window_count": 5,
            "nodes_per_window": 32,
            "upper_u_truncation": 100.0,
            "frequency_synthesis": "monochromatic components evaluated at every node",
        },
        "component_median_fraction_of_weighted_arithmetic_sum": {
            name: float(np.median(values[applicable_layers]))
            for name, values in component_fraction.items()
        },
        "component_diagnostics_by_band": component_diagnostics,
        "band_summary": band_summary,
        "teff_x_layer_temperature_summary": _teff_layer_temperature_summary(
            opacity_residual,
            labels,
            temperature,
            applicable_layers,
        ),
        "applicability": {
            "temperature_floor_K": TEMPERATURE_FLOOR_K,
            "excluded_layer_count": int(np.sum(out_of_domain_layers)),
            "applicable_layer_count": int(np.sum(applicable_layers)),
            "excluded_layer_fraction": float(np.mean(out_of_domain_layers)),
            "excluded_layers_reported_separately": True,
            "molecular_opacity_added": False,
        },
        "nonfinite_count": int(
            np.sum(~np.isfinite(prediction))
            + np.sum(~np.isfinite(integrated_mass))
        ),
        "opacity_profile_metrics_against_stored_total_kappa_R": _profile_metrics(
            opacity_residual,
            labels,
            layer_mask=applicable_layers,
        ),
        "opacity_out_of_domain_metrics_against_stored_total_kappa_R": _profile_metrics(
            opacity_residual,
            labels,
            layer_mask=out_of_domain_layers,
        ),
        "mass_profile_metrics_using_true_P_T": _profile_metrics(
            mass_residual,
            labels,
            layer_mask=applicable_layers,
        ),
        "mass_out_of_domain_metrics_using_true_P_T": _profile_metrics(
            mass_residual,
            labels,
            layer_mask=out_of_domain_layers,
        ),
        "offline_gate": {
            "stored_target": "total Rosseland opacity; sanity replay confirmed line contribution",
            "cool_p95_max_dex": 0.30,
            "middle_p95_max_dex": 0.50,
            "transition_band_reported_separately": True,
            "low_temperature_layers_excluded_from_gate_below_K": TEMPERATURE_FLOOR_K,
            "formal_opacity_pass_against_total_target": formal_opacity_pass,
            "bridge_allowance_max_excess_dex": 0.10,
            "bridge_mass_p95_max_dex": 0.20,
            "bridge_allowance_applies": bridge_allowance_applies,
            "bridge_pass": bridge_pass,
            "pass_to_next_stage": promotion_stage_pass,
            "cool_observed_p95_dex": cool_gate["p95_dex"],
            "middle_observed_p95_dex": middle_gate["p95_dex"],
            "cool_mass_observed_p95_dex": mass_cool_gate["p95_dex"],
            "middle_mass_observed_p95_dex": mass_middle_gate["p95_dex"],
            "line_floor_does_not_relax_mass_bridge": True,
        },
        "target_interpretation": {
            "v4_candidate": "continuum_only",
            "stored_kappa_R": "line_plus_continuum production target",
            "total_target_comparison_is_continuum_comparability_diagnostic": True,
            "continuum_reachability_sanity_result": "results/analytic_initializer/textbook_opacity_v4_sanity_20260826.json",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "opacity_against_total": result[
                    "opacity_profile_metrics_against_stored_total_kappa_R"
                ],
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
