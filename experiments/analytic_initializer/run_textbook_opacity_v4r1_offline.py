"""Run the preregistered full-development v4r1 offline validation."""

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
from experiments.analytic_initializer.run_textbook_opacity_v4r1_diagnostics import (
    COMPONENT_NAMES,
    _fraction_metrics,
    _rosseland_diagnostics,
)
from experiments.analytic_initializer.textbook_opacity import (
    DEFAULT_TEXTBOOK_CONSTANTS,
    V4R1_FORMAL_TEMPERATURE_FLOOR_K,
    WINDOW_NAMES,
    saha_electron_diagnostics_v4r1,
    textbook_opacity_node_components_v4r1,
)


TEMPERATURE_FLOOR_K = V4R1_FORMAL_TEMPERATURE_FLOOR_K
DIAGNOSTIC_RESULT = Path(
    "results/analytic_initializer/textbook_opacity_v4r1_diagnostics_20260827.json"
)


def _batch_prediction(
    labels: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    prediction = np.empty_like(temperature, dtype=np.float64)
    component_sensitivity = {
        name: np.empty_like(temperature, dtype=np.float64)
        for name in COMPONENT_NAMES
    }
    inverse_window_fraction = {
        name: np.empty_like(temperature, dtype=np.float64)
        for name in WINDOW_NAMES
    }
    electron_density = np.empty_like(temperature, dtype=np.float64)
    for start in range(0, labels.shape[0], int(batch_size)):
        stop = min(labels.shape[0], start + int(batch_size))
        local_labels = labels[start:stop]
        local_temperature = temperature[start:stop]
        local_pressure = pressure[start:stop]
        components = textbook_opacity_node_components_v4r1(
            local_labels,
            local_temperature,
            local_pressure,
        )
        weights = components["node_weights"]
        total = np.maximum(components["total"], 1.0e-30)
        prediction[start:stop] = 1.0 / np.sum(
            weights / total,
            axis=(-2, -1),
        )
        _, window_fraction, sensitivity = _rosseland_diagnostics(components)
        for index, name in enumerate(COMPONENT_NAMES):
            component_sensitivity[name][start:stop] = sensitivity[..., index]
        for index, name in enumerate(WINDOW_NAMES):
            inverse_window_fraction[name][start:stop] = window_fraction[..., index]
        electron_density[start:stop] = saha_electron_diagnostics_v4r1(
            local_labels,
            local_temperature,
            local_pressure,
        )["electron_density_cm3"]
        print(f"processed {stop}/{labels.shape[0]} validation stars", flush=True)
    return (
        prediction,
        component_sensitivity,
        inverse_window_fraction,
        electron_density,
    )


def _fraction_summary_by_band(
    values: dict[str, np.ndarray],
    labels: np.ndarray,
    layer_mask: np.ndarray,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for band, star_mask in _band_masks(labels[:, 0]).items():
        mask = star_mask[:, None] & layer_mask
        if not np.any(mask):
            continue
        result[band] = {
            "star_count": int(np.sum(star_mask)),
            "layer_count": int(np.sum(mask)),
            "metrics": {
                name: _fraction_metrics(array[mask])
                for name, array in values.items()
            },
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=DIAGNOSTIC_RESULT,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/analytic_initializer/"
            "textbook_opacity_v4r1_offline_validation_20260827.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if not args.diagnostics.is_file():
        raise SystemExit(
            f"required preregistered diagnostic result is missing: {args.diagnostics}"
        )

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
    (
        prediction,
        component_sensitivity,
        inverse_window_fraction,
        electron_density,
    ) = _batch_prediction(
        labels,
        temperature,
        pressure,
        batch_size=args.batch_size,
    )
    opacity_residual = np.log10(prediction) - np.log10(truth_opacity)
    electron_density_residual = np.log10(
        electron_density / corpus.electron_density[indices]
    )
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
    mass_bridge_pass = bool(
        mass_cool_gate["p95_dex"] <= 0.20
        and mass_middle_gate["p95_dex"] <= 0.20
    )
    pass_to_next_stage = bool(formal_opacity_pass and mass_bridge_pass)

    sensitivity_by_band = _fraction_summary_by_band(
        component_sensitivity,
        labels,
        applicable_layers,
    )
    window_by_band = _fraction_summary_by_band(
        inverse_window_fraction,
        labels,
        applicable_layers,
    )
    band_summary = []
    for band, star_mask in band_masks.items():
        if not np.any(star_mask):
            continue
        mask = star_mask[:, None] & applicable_layers
        opacity_metrics = _metrics(opacity_residual[mask])
        mass_metrics = _metrics(mass_residual[mask])
        sensitivity = sensitivity_by_band[band]["metrics"]
        dominant_component = max(
            sensitivity,
            key=lambda name: sensitivity[name]["median"],
        )
        band_summary.append(
            {
                "band": band,
                "star_count": int(np.sum(star_mask)),
                "signed_median_dex": opacity_metrics["signed_median_dex"],
                "signed_mean_dex": opacity_metrics["signed_mean_dex"],
                "abs_p95_dex": opacity_metrics["p95_dex"],
                "mass_abs_p95_dex": mass_metrics["p95_dex"],
                "dominant_rosseland_sensitivity_component": dominant_component,
                "dominant_component_median_log_sensitivity": sensitivity[
                    dominant_component
                ]["median"],
            }
        )

    status = (
        "offline_pass_next_stage_qualified"
        if pass_to_next_stage
        else "offline_fail_stop"
    )
    decision = "QUALIFIED_BUT_STOP_AFTER_OFFLINE" if pass_to_next_stage else "FAIL_STOP"
    result = {
        "schema_version": 1,
        "candidate": (
            "agss09_ground_partition_saha_john_hminus_freefree_v4r1"
        ),
        "version": "v4r1",
        "status": status,
        "decision": decision,
        "next_registered_stage": (
            "qualified_but_not_run_in_this_task"
            if pass_to_next_stage
            else "blocked_before_ode_temperature_ablation_smoke_and_funnel"
        ),
        "historical_v4_result_preserved": (
            "results/analytic_initializer/"
            "textbook_opacity_v4_offline_validation.json"
        ),
        "diagnostic_result": str(args.diagnostics),
        "diagnostic_result_sha256": file_sha256(args.diagnostics),
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
            "window_count": len(WINDOW_NAMES),
            "nodes_per_window": 32,
            "upper_u_truncation": 100.0,
            "frequency_synthesis": (
                "monochromatic components evaluated at every node"
            ),
        },
        "v4r1_repairs": {
            "electron_donors": ["Na", "K", "Ca", "Mg", "Fe", "Al", "Si"],
            "saha_partition_factor": "fixed ground-term 2 U_II / U_I",
            "solar_abundance_convention": "AGSS09",
            "alpha_scaled_donors": ["Mg", "Si", "Ca"],
            "neutral_mean_molecular_weight": 1.30,
            "hminus_freefree": "John coefficient times n_H times electron pressure over rho",
            "extra_stimulated_emission_on_hminus_freefree": False,
            "stored_electron_density_used_by_candidate": False,
        },
        "band_summary": band_summary,
        "component_log_sensitivity_by_band": sensitivity_by_band,
        "inverse_opacity_window_fraction_by_band": window_by_band,
        "teff_x_layer_temperature_summary": _teff_layer_temperature_summary(
            opacity_residual,
            labels,
            temperature,
            applicable_layers,
        ),
        "electron_density_metrics_against_stored": _profile_metrics(
            electron_density_residual,
            labels,
            layer_mask=applicable_layers,
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
            + np.sum(~np.isfinite(electron_density))
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
            "stored_target": (
                "total Rosseland opacity; historical sanity replay confirmed "
                "line contribution"
            ),
            "cool_p95_max_dex": 0.30,
            "middle_p95_max_dex": 0.50,
            "bridge_mass_p95_max_dex": 0.20,
            "bridge_allowance_max_excess_dex": 0.10,
            "bridge_allowance_applies": bridge_allowance_applies,
            "formal_opacity_pass_against_total_target": formal_opacity_pass,
            "mass_bridge_pass": mass_bridge_pass,
            "pass_to_next_stage": pass_to_next_stage,
            "cool_observed_p95_dex": cool_gate["p95_dex"],
            "middle_observed_p95_dex": middle_gate["p95_dex"],
            "cool_mass_observed_p95_dex": mass_cool_gate["p95_dex"],
            "middle_mass_observed_p95_dex": mass_middle_gate["p95_dex"],
            "all_registered_gates_required": True,
            "line_floor_does_not_relax_mass_bridge": True,
        },
        "scope_boundary": {
            "production_solver_changed": False,
            "ode_run": False,
            "smoke_run": False,
            "funnel_run": False,
            "sealed_holdout_opened": False,
            "stop_after_offline_even_if_qualified": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "offline_gate": result["offline_gate"]}))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
