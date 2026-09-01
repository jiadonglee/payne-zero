"""Run the preregistered v4r5 ground-anchored H I bound-free offline validation."""

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
    _fraction_metrics,
    _rosseland_diagnostics,
)
from experiments.analytic_initializer.run_textbook_opacity_v4r1_offline import (
    _fraction_summary_by_band,
)
from experiments.analytic_initializer.textbook_opacity import (
    COMPONENT_NAMES_V4R5,
    DEFAULT_TEXTBOOK_CONSTANTS,
    V4R1_FORMAL_TEMPERATURE_FLOOR_K,
    V4R5_FORMAL_TEMPERATURE_FLOOR_K,
    WINDOW_NAMES,
    textbook_opacity_node_components_v4r5,
)


PREVIOUS_FLOOR_K = V4R1_FORMAL_TEMPERATURE_FLOOR_K
TEMPERATURE_FLOOR_K = V4R5_FORMAL_TEMPERATURE_FLOOR_K
V4R3_OFFLINE = Path(
    "results/analytic_initializer/textbook_opacity_v4r3_offline_validation_20260827.json"
)
HOT_FLAG_ABLATION = Path(
    "results/analytic_initializer/textbook_opacity_v4r4_hot_flag_ablation_20260828.json"
)
MOLECULE_ABLATION = Path(
    "results/analytic_initializer/textbook_opacity_v4r1_molecule_ablation_20260827.json"
)

REGISTERED_LAYER_SLICES = (
    ("8000_15000K", 8000.0, 15000.0),
    ("15000_22000K", 15000.0, 22000.0),
    ("22000_30000K", 22000.0, 30000.0),
    ("at_least_30000K", 30000.0, np.inf),
    ("at_least_15000K", 15000.0, np.inf),
)
CONTROL_SLICE_NAME = "8000_15000K"
HOT_TAIL_SLICE_NAME = "at_least_30000K"
CONTROL_MEDIAN_TOLERANCE_DEX = 0.02


def _batch_prediction(
    labels: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    *,
    batch_size: int,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
]:
    prediction = np.empty_like(temperature, dtype=np.float64)
    component_sensitivity = {
        name: np.empty_like(temperature, dtype=np.float64)
        for name in COMPONENT_NAMES_V4R5
    }
    inverse_window_fraction = {
        name: np.empty_like(temperature, dtype=np.float64)
        for name in WINDOW_NAMES
    }
    electron_density = np.empty_like(temperature, dtype=np.float64)
    mean_molecular_weight = np.empty_like(temperature, dtype=np.float64)
    for start in range(0, labels.shape[0], int(batch_size)):
        stop = min(labels.shape[0], start + int(batch_size))
        local_labels = labels[start:stop]
        local_temperature = temperature[start:stop]
        local_pressure = pressure[start:stop]
        components = textbook_opacity_node_components_v4r5(
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
        _, window_fraction, sensitivity = _rosseland_diagnostics(
            components,
            component_names=COMPONENT_NAMES_V4R5,
        )
        for index, name in enumerate(COMPONENT_NAMES_V4R5):
            component_sensitivity[name][start:stop] = sensitivity[..., index]
        for index, name in enumerate(WINDOW_NAMES):
            inverse_window_fraction[name][start:stop] = window_fraction[..., index]
        electron_density[start:stop] = components["electron_density_cm3"]
        mean_molecular_weight[start:stop] = components["mean_molecular_weight"]
        print(f"processed {stop}/{labels.shape[0]} stars", flush=True)
    return (
        prediction,
        component_sensitivity,
        inverse_window_fraction,
        electron_density,
        mean_molecular_weight,
    )


def _slice_metrics(
    residual: np.ndarray,
    temperature: np.ndarray,
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for name, lower, upper in REGISTERED_LAYER_SLICES:
        mask = (temperature >= lower) & (temperature < upper)
        if not np.any(mask):
            continue
        rows[name] = _metrics(residual[mask])
    return rows


def _hot_grid_falsification(
    corpus,
    ablation: dict[str, object],
    *,
    ablation_path: Path,
    batch_size: int,
) -> dict[str, object]:
    references = ablation["references"]
    indices = np.asarray(
        [int(row["corpus_index"]) for row in references], dtype=np.int64
    )
    labels = corpus.labels[indices]
    temperature = corpus.temperature[indices]
    pressure = corpus.gas_pressure[indices]
    stored_temperature = np.asarray(
        [row["temperature_K"] for row in references], dtype=np.float64
    )
    if stored_temperature.shape != temperature.shape:
        raise ValueError("hot-grid stored temperatures do not match the corpus")
    if np.max(np.abs(temperature - stored_temperature)) > 1.0e-6:
        raise ValueError("hot-grid corpus temperatures drifted from the ablation JSON")
    production = np.asarray(
        [row["production_continuum_baseline"] for row in references],
        dtype=np.float64,
    )
    v4r3_opacity = np.asarray(
        [row["v4r3_rosseland_opacity"] for row in references], dtype=np.float64
    )
    (
        prediction,
        _,
        _,
        _,
        _,
    ) = _batch_prediction(
        labels,
        temperature,
        pressure,
        batch_size=batch_size,
    )
    v4r5_residual = np.log10(prediction) - np.log10(production)
    v4r3_residual = np.log10(v4r3_opacity) - np.log10(production)
    v4r5_slices = _slice_metrics(v4r5_residual, temperature)
    v4r3_slices = _slice_metrics(v4r3_residual, temperature)
    control_v4r5 = v4r5_slices[CONTROL_SLICE_NAME]["signed_median_dex"]
    control_v4r3 = v4r3_slices[CONTROL_SLICE_NAME]["signed_median_dex"]
    hot_v4r5 = v4r5_slices[HOT_TAIL_SLICE_NAME]["signed_median_dex"]
    hot_v4r3 = v4r3_slices[HOT_TAIL_SLICE_NAME]["signed_median_dex"]
    control_stable = bool(
        abs(control_v4r5 - control_v4r3) <= CONTROL_MEDIAN_TOLERANCE_DEX
    )
    hot_increased_toward_zero = bool(hot_v4r5 > hot_v4r3)
    hypothesis_holds = bool(control_stable and hot_increased_toward_zero)
    return {
        "reference_count": int(indices.size),
        "reference_indices": [int(index) for index in indices],
        "ablation_result": str(ablation_path),
        "control_slice": CONTROL_SLICE_NAME,
        "hot_tail_slice": HOT_TAIL_SLICE_NAME,
        "control_median_tolerance_dex": CONTROL_MEDIAN_TOLERANCE_DEX,
        "v4r3_minus_production_continuum": v4r3_slices,
        "v4r5_minus_production_continuum": v4r5_slices,
        "control_signed_median_v4r3_dex": control_v4r3,
        "control_signed_median_v4r5_dex": control_v4r5,
        "hot_tail_signed_median_v4r3_dex": hot_v4r3,
        "hot_tail_signed_median_v4r5_dex": hot_v4r5,
        "control_stable": control_stable,
        "hot_tail_increased_algebraically": hot_increased_toward_zero,
        "hypothesis_holds": hypothesis_holds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--v4r3-offline", type=Path, default=V4R3_OFFLINE)
    parser.add_argument("--hot-flag-ablation", type=Path, default=HOT_FLAG_ABLATION)
    parser.add_argument("--molecule-ablation", type=Path, default=MOLECULE_ABLATION)
    parser.add_argument(
        "--hot-grid-only",
        action="store_true",
        help="Evaluate only the frozen 20-star production-continuum grid.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/analytic_initializer/"
            "textbook_opacity_v4r5_offline_validation_20260828.json"
        ),
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.hot_grid_only:
        required = (args.hot_flag_ablation,)
    else:
        required = (args.v4r3_offline, args.hot_flag_ablation, args.molecule_ablation)
    for path in required:
        if not path.is_file():
            raise SystemExit(f"required prior artifact is missing: {path}")

    corpus = load_strict_truth(args.corpus)
    ablation = json.loads(args.hot_flag_ablation.read_text(encoding="utf-8"))
    hot_grid = _hot_grid_falsification(
        corpus,
        ablation,
        ablation_path=args.hot_flag_ablation,
        batch_size=args.batch_size,
    )
    if args.hot_grid_only:
        result = {
            "schema_version": 1,
            "candidate": "ground_anchored_hydrogen_boundfree_v4r5",
            "version": "v4r5",
            "status": (
                "hot_grid_hypothesis_holds"
                if hot_grid["hypothesis_holds"]
                else "hot_grid_hypothesis_fails"
            ),
            "decision": (
                "HYPOTHESIS_HOLD" if hot_grid["hypothesis_holds"] else "HYPOTHESIS_FAIL"
            ),
            "hot_grid_falsification": hot_grid,
            "scope_boundary": {
                "production_solver_changed": False,
                "full_offline_run": False,
                "ode_run": False,
                "funnel_run": False,
                "sealed_holdout_opened": False,
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
                    "decision": result["decision"],
                    "hot_grid_falsification": {
                        key: hot_grid[key]
                        for key in (
                            "control_signed_median_v4r3_dex",
                            "control_signed_median_v4r5_dex",
                            "hot_tail_signed_median_v4r3_dex",
                            "hot_tail_signed_median_v4r5_dex",
                            "hypothesis_holds",
                        )
                    },
                }
            )
        )
        print(f"wrote {args.out}")
        return 0

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
        mean_molecular_weight,
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
    newly_excluded_layers = (temperature >= PREVIOUS_FLOOR_K) & (
        temperature < TEMPERATURE_FLOOR_K
    )
    historical_excluded_layers = temperature < PREVIOUS_FLOOR_K
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
    molecule_ablation = json.loads(args.molecule_ablation.read_text(encoding="utf-8"))
    mu_applicable = _fraction_metrics(mean_molecular_weight[applicable_layers])
    result = {
        "schema_version": 1,
        "candidate": "ground_anchored_hydrogen_boundfree_v4r5",
        "version": "v4r5",
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
        "historical_v4r3_result_preserved": str(args.v4r3_offline),
        "historical_v4r3_result_sha256": file_sha256(args.v4r3_offline),
        "hot_flag_ablation_result": str(args.hot_flag_ablation),
        "hot_flag_ablation_result_sha256": file_sha256(args.hot_flag_ablation),
        "molecule_ablation_result": str(args.molecule_ablation),
        "molecule_ablation_result_sha256": file_sha256(args.molecule_ablation),
        "molecule_ablation_verdict": molecule_ablation["decision"]["verdict"],
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
        "v4r5_change": {
            "hydrogen_boundfree_level_closure": "ground_anchored_boltzmann",
            "ten_level_partition_renormalization": False,
            "hydrogen_boundfree_edge_cross_section_power": float(
                DEFAULT_TEXTBOOK_CONSTANTS.hydrogen_boundfree_edge_cross_section_power
            ),
            "heii_added": False,
            "helium_added_as_saha_donor": False,
            "karzas_tables_used": False,
            "h2plus_continuum_added": True,
            "heminus_continuum_added": True,
            "molecular_band_opacity_added": False,
            "stored_electron_density_used_by_candidate": False,
            "temperature_floor_K": TEMPERATURE_FLOOR_K,
            "john_hminus_coefficients_changed": False,
        },
        "hot_grid_falsification": hot_grid,
        "registered_layer_slices_against_stored_total_kappa_R": _slice_metrics(
            opacity_residual, temperature
        ),
        "band_summary": band_summary,
        "component_log_sensitivity_by_band": sensitivity_by_band,
        "inverse_opacity_window_fraction_by_band": window_by_band,
        "mean_molecular_weight_on_domain": mu_applicable,
        "teff_x_layer_temperature_summary": _teff_layer_temperature_summary(
            opacity_residual,
            labels,
            temperature,
            applicable_layers,
        ),
        "teff_x_layer_temperature_summary_3200_to_4000K": (
            _teff_layer_temperature_summary(
                opacity_residual,
                labels,
                temperature,
                newly_excluded_layers,
            )
        ),
        "electron_density_metrics_against_stored": _profile_metrics(
            electron_density_residual,
            labels,
            layer_mask=applicable_layers,
        ),
        "applicability": {
            "previous_temperature_floor_K": PREVIOUS_FLOOR_K,
            "temperature_floor_K": TEMPERATURE_FLOOR_K,
            "excluded_layer_count": int(np.sum(out_of_domain_layers)),
            "applicable_layer_count": int(np.sum(applicable_layers)),
            "excluded_layer_fraction": float(np.mean(out_of_domain_layers)),
            "excluded_3200_to_4000K_layer_count": int(np.sum(newly_excluded_layers)),
            "excluded_below_3200K_layer_count": int(
                np.sum(historical_excluded_layers)
            ),
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
        "opacity_metrics_3200_to_4000K_against_stored_total_kappa_R": _profile_metrics(
            opacity_residual,
            labels,
            layer_mask=newly_excluded_layers,
        ),
        "opacity_metrics_below_3200K_against_stored_total_kappa_R": _profile_metrics(
            opacity_residual,
            labels,
            layer_mask=historical_excluded_layers,
        ),
        "mass_profile_metrics_using_true_P_T": _profile_metrics(
            mass_residual,
            labels,
            layer_mask=applicable_layers,
        ),
        "mass_metrics_3200_to_4000K_using_true_P_T": _profile_metrics(
            mass_residual,
            labels,
            layer_mask=newly_excluded_layers,
        ),
        "mass_metrics_below_3200K_using_true_P_T": _profile_metrics(
            mass_residual,
            labels,
            layer_mask=historical_excluded_layers,
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
            "mass_integral_still_includes_layers_below_floor": True,
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
    print(
        json.dumps(
            {
                "decision": decision,
                "offline_gate": result["offline_gate"],
                "hot_grid_hypothesis_holds": hot_grid["hypothesis_holds"],
            }
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
