"""Run the preregistered v4r1 electron and Rosseland-window diagnostics."""

from __future__ import annotations

# Must precede any Numba import.
from bench import environment as _environment  # noqa: F401

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
from experiments.analytic_initializer.run_textbook_opacity_v4_sanity import (
    LAYER_TEMPERATURE_BANDS,
    MANIFESTS,
    METALLICITY_BANDS,
    TEFF_BANDS,
    _atmosphere_from_corpus,
    _config,
    _metrics,
    _reference_indices,
    _rosseland_from_slabs,
)
from experiments.analytic_initializer.textbook_opacity import (
    DEFAULT_TEXTBOOK_CONSTANTS,
    WINDOW_NAMES,
    _textbook_opacity_node_components_v4r1_with_electron_density_oracle,
    saha_electron_diagnostics,
    saha_electron_diagnostics_v4r1,
    textbook_opacity_node_components_v4r1,
)
from payne_zero_atmosphere.runner import (
    prepare_opacity_state,
    prepare_population_state,
)
from payne_zero_atmosphere.source_catalogs import (
    molecular_equilibrium_catalog_path,
    source_line_paths,
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


def _rosseland_mean(components: dict[str, np.ndarray]) -> np.ndarray:
    return 1.0 / np.sum(
        components["node_weights"] / np.maximum(components["total"], 1.0e-30),
        axis=(-2, -1),
    )


def _rosseland_diagnostics(
    components: dict[str, np.ndarray],
    component_names: tuple[str, ...] = COMPONENT_NAMES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = components["node_weights"]
    total = np.maximum(components["total"], 1.0e-30)
    names = tuple(component_names)
    raw_window_weight = np.sum(weights, axis=-1)
    inverse_by_window = np.sum(weights / total, axis=-1)
    inverse_window_fraction = inverse_by_window / np.maximum(
        np.sum(inverse_by_window, axis=-1, keepdims=True),
        1.0e-300,
    )
    inverse_total = np.sum(weights / total, axis=(-2, -1))
    component_log_sensitivity = np.stack(
        [
            np.sum(
                weights * components[name] / total**2,
                axis=(-2, -1),
            )
            / np.maximum(inverse_total, 1.0e-300)
            for name in names
        ],
        axis=-1,
    )
    return raw_window_weight, inverse_window_fraction, component_log_sensitivity


def _fraction_metrics(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("fraction metrics require at least one finite value")
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _teff_layer_metallicity_summary(
    labels: np.ndarray,
    temperature: np.ndarray,
    values: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for teff_name, teff_lower, teff_upper in TEFF_BANDS:
        teff_mask = (labels[:, 0] >= teff_lower) & (labels[:, 0] < teff_upper)
        for layer_name, layer_lower, layer_upper in LAYER_TEMPERATURE_BANDS:
            layer_mask = (temperature >= layer_lower) & (temperature < layer_upper)
            for metal_name, metal_lower, metal_upper in METALLICITY_BANDS:
                metal_mask = labels[:, 2] >= metal_lower
                if metal_upper < 0.5:
                    metal_mask &= labels[:, 2] < metal_upper
                else:
                    metal_mask &= labels[:, 2] <= metal_upper
                mask = teff_mask[:, None] & metal_mask[:, None] & layer_mask
                if not np.any(mask):
                    continue
                row: dict[str, object] = {
                    "teff_band": teff_name,
                    "layer_temperature_band": layer_name,
                    "metallicity_band": metal_name,
                    "star_count": int(np.sum(teff_mask & metal_mask)),
                    "layer_count": int(np.sum(mask)),
                }
                row.update({name: _metrics(array[mask]) for name, array in values.items()})
                rows.append(row)
    return rows


def _layer_fraction_summary(
    labels: np.ndarray,
    temperature: np.ndarray,
    raw_window_weight: np.ndarray,
    inverse_window_fraction: np.ndarray,
    component_log_sensitivity: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for teff_name, teff_lower, teff_upper in TEFF_BANDS:
        star_mask = (labels[:, 0] >= teff_lower) & (labels[:, 0] < teff_upper)
        for layer_name, layer_lower, layer_upper in LAYER_TEMPERATURE_BANDS:
            mask = (
                star_mask[:, None]
                & (temperature >= layer_lower)
                & (temperature < layer_upper)
            )
            if not np.any(mask):
                continue
            rows.append(
                {
                    "teff_band": teff_name,
                    "layer_temperature_band": layer_name,
                    "layer_count": int(np.sum(mask)),
                    "raw_rosseland_weight_fraction": {
                        name: _fraction_metrics(raw_window_weight[..., index][mask])
                        for index, name in enumerate(WINDOW_NAMES)
                    },
                    "inverse_opacity_window_fraction": {
                        name: _fraction_metrics(
                            inverse_window_fraction[..., index][mask]
                        )
                        for index, name in enumerate(WINDOW_NAMES)
                    },
                    "component_log_sensitivity": {
                        name: _fraction_metrics(
                            component_log_sensitivity[..., index][mask]
                        )
                        for index, name in enumerate(COMPONENT_NAMES)
                    },
                }
            )
    return rows


def _prepare_production_continuum(
    corpus,
    index: int,
    *,
    stride: int,
    line_paths: dict[str, Path],
    molecule_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    atmosphere = _atmosphere_from_corpus(corpus, index, lines_enabled=False)
    config = _config(
        atmosphere,
        stride=stride,
        line_paths=line_paths,
        molecule_path=molecule_path,
    )
    population = prepare_population_state(config)
    opacity = prepare_opacity_state(config, population_state=population)
    return (
        _rosseland_from_slabs(opacity),
        np.asarray(opacity.population_state.runtime_state.mass_density, dtype=np.float64),
        {
            "frequency_count": int(opacity.opacity_frequency_hz.size),
            "positive_line_cells": int(
                np.count_nonzero(
                    opacity.line_opacity.line_mass_absorption_coefficient > 0.0
                )
            ),
            "line_catalog_selected_count": int(
                0
                if opacity.selected_line_catalog is None
                else opacity.selected_line_catalog.line_count
            ),
        },
    )


def run_diagnostics(
    *,
    corpus_path: Path,
    output_path: Path,
    split_seed: int = 20260816,
    stride: int = 16,
    batch_size: int = 256,
) -> dict[str, object]:
    if stride < 1 or batch_size < 1:
        raise ValueError("stride and batch_size must be positive")
    corpus = load_strict_truth(corpus_path)
    excluded, excluded_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=split_seed)
    indices = split.validation
    labels = corpus.labels[indices]
    temperature = corpus.temperature[indices]
    pressure = corpus.gas_pressure[indices]

    old_electron_density = np.empty_like(temperature)
    v4r1_electron_density = np.empty_like(temperature)
    charge_balance_residual = np.empty_like(temperature)
    for start in range(0, indices.size, int(batch_size)):
        stop = min(indices.size, start + int(batch_size))
        old_state = saha_electron_diagnostics(
            labels[start:stop],
            temperature[start:stop],
            pressure[start:stop],
        )
        new_state = saha_electron_diagnostics_v4r1(
            labels[start:stop],
            temperature[start:stop],
            pressure[start:stop],
        )
        old_electron_density[start:stop] = old_state["electron_density_cm3"]
        v4r1_electron_density[start:stop] = new_state["electron_density_cm3"]
        charge_balance_residual[start:stop] = new_state[
            "charge_balance_relative_residual"
        ]
    stored_electron_density = corpus.electron_density[indices]
    old_ne_residual = np.log10(old_electron_density / stored_electron_density)
    v4r1_ne_residual = np.log10(v4r1_electron_density / stored_electron_density)
    applicable = temperature >= TEMPERATURE_FLOOR_K

    references = _reference_indices(corpus, split.validation)
    line_paths = source_line_paths()
    molecule_path = molecular_equilibrium_catalog_path()
    reference_rows: list[dict[str, object]] = []
    reference_labels: list[np.ndarray] = []
    reference_temperature: list[np.ndarray] = []
    raw_weights: list[np.ndarray] = []
    inverse_fractions: list[np.ndarray] = []
    sensitivities: list[np.ndarray] = []
    candidate_minus_continuum: list[np.ndarray] = []
    oracle_minus_continuum: list[np.ndarray] = []
    density_residuals: list[np.ndarray] = []
    frequency_counts: list[int] = []
    for ordinal, index in enumerate(references, start=1):
        local_labels = corpus.labels[index : index + 1]
        local_temperature = corpus.temperature[index : index + 1]
        local_pressure = corpus.gas_pressure[index : index + 1]
        candidate_components = textbook_opacity_node_components_v4r1(
            local_labels,
            local_temperature,
            local_pressure,
        )
        oracle_components = (
            _textbook_opacity_node_components_v4r1_with_electron_density_oracle(
                local_labels,
                local_temperature,
                local_pressure,
                corpus.electron_density[index : index + 1],
            )
        )
        candidate_opacity = _rosseland_mean(candidate_components)[0]
        oracle_opacity = _rosseland_mean(oracle_components)[0]
        continuum, production_density, replay = _prepare_production_continuum(
            corpus,
            index,
            stride=stride,
            line_paths=line_paths,
            molecule_path=molecule_path,
        )
        raw_weight, inverse_fraction, sensitivity = _rosseland_diagnostics(
            candidate_components
        )
        candidate_residual = np.log10(candidate_opacity / continuum)
        oracle_residual = np.log10(oracle_opacity / continuum)
        analytic_density = saha_electron_diagnostics_v4r1(
            local_labels,
            local_temperature,
            local_pressure,
        )["rho_g_cm3"][0]
        density_residual = np.log10(analytic_density / production_density)
        reference_labels.append(local_labels[0])
        reference_temperature.append(local_temperature[0])
        raw_weights.append(raw_weight[0])
        inverse_fractions.append(inverse_fraction[0])
        sensitivities.append(sensitivity[0])
        candidate_minus_continuum.append(candidate_residual)
        oracle_minus_continuum.append(oracle_residual)
        density_residuals.append(density_residual)
        frequency_counts.append(replay["frequency_count"])
        reference_rows.append(
            {
                "reference_ordinal": ordinal,
                "corpus_index": int(index),
                "slug": str(corpus.slugs[index]),
                "labels": [float(value) for value in local_labels[0]],
                "temperature_K": [float(value) for value in local_temperature[0]],
                "v4r1_electron_density_cm3": [
                    float(value)
                    for value in saha_electron_diagnostics_v4r1(
                        local_labels,
                        local_temperature,
                        local_pressure,
                    )["electron_density_cm3"][0]
                ],
                "stored_electron_density_cm3": [
                    float(value) for value in corpus.electron_density[index]
                ],
                "v4r1_rosseland_opacity": [
                    float(value) for value in candidate_opacity
                ],
                "oracle_ne_rosseland_opacity": [
                    float(value) for value in oracle_opacity
                ],
                "production_continuum_rosseland_opacity": [
                    float(value) for value in continuum
                ],
                "v4r1_minus_continuum_dex": [
                    float(value) for value in candidate_residual
                ],
                "oracle_ne_minus_continuum_dex": [
                    float(value) for value in oracle_residual
                ],
                "fixed_mu_density_minus_production_dex": [
                    float(value) for value in density_residual
                ],
                "raw_rosseland_weight_fraction": raw_weight[0].tolist(),
                "inverse_opacity_window_fraction": inverse_fraction[0].tolist(),
                "component_log_sensitivity": sensitivity[0].tolist(),
                "production_replay": replay,
            }
        )
        print(
            f"[{ordinal:02d}/{len(references):02d}] index={index} "
            f"Teff={local_labels[0, 0]:.0f} [M/H]={local_labels[0, 2]:+.2f} "
            f"continuum_p95={np.percentile(np.abs(candidate_residual), 95.0):.3f}",
            flush=True,
        )

    ref_labels = np.asarray(reference_labels, dtype=np.float64)
    ref_temperature = np.asarray(reference_temperature, dtype=np.float64)
    raw_weight_array = np.asarray(raw_weights, dtype=np.float64)
    inverse_fraction_array = np.asarray(inverse_fractions, dtype=np.float64)
    sensitivity_array = np.asarray(sensitivities, dtype=np.float64)
    candidate_residual_array = np.asarray(
        candidate_minus_continuum, dtype=np.float64
    )
    oracle_residual_array = np.asarray(oracle_minus_continuum, dtype=np.float64)
    density_residual_array = np.asarray(density_residuals, dtype=np.float64)

    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "textbook_opacity_v4r1_d1_d2_diagnostics",
        "version": "v4r1",
        "status": "diagnostic_only",
        "corpus": str(corpus.path),
        "corpus_sha256": file_sha256(corpus.path),
        "split": {
            "seed": int(split.seed),
            "validation_count": int(indices.size),
            "excluded_count": int(excluded.size),
            "excluded_manifests": excluded_manifests,
            "sealed_rows_read": False,
        },
        "constants": asdict(DEFAULT_TEXTBOOK_CONSTANTS),
        "d1_electron_density": {
            "candidate_uses_stored_electron_density": False,
            "stored_electron_density_used_only_in_oracle_diagnostic": True,
            "fixed_neutral_mean_molecular_weight": float(
                DEFAULT_TEXTBOOK_CONSTANTS.neutral_mean_molecular_weight
            ),
            "overall_applicable": {
                "historical_v4_minus_stored_dex": _metrics(
                    old_ne_residual[applicable]
                ),
                "v4r1_minus_stored_dex": _metrics(v4r1_ne_residual[applicable]),
            },
            "teff_x_layer_temperature_x_metallicity": (
                _teff_layer_metallicity_summary(
                    labels,
                    temperature,
                    {
                        "historical_v4_minus_stored_ne_dex": old_ne_residual,
                        "v4r1_minus_stored_ne_dex": v4r1_ne_residual,
                    },
                )
            ),
            "charge_balance_maximum_absolute_relative_residual": float(
                np.max(np.abs(charge_balance_residual))
            ),
            "fixed_mu_density_minus_production_reference_summary_dex": _metrics(
                density_residual_array
            ),
            "reference_continuum_comparison": {
                "v4r1_minus_production_continuum_dex": _metrics(
                    candidate_residual_array
                ),
                "stored_ne_oracle_minus_production_continuum_dex": _metrics(
                    oracle_residual_array
                ),
            },
        },
        "d2_rosseland_windows": {
            "window_names": list(WINDOW_NAMES),
            "component_names": list(COMPONENT_NAMES),
            "inverse_window_definition": (
                "sum_nodes_in_window(w/kappa_nu) / sum_all_nodes(w/kappa_nu)"
            ),
            "component_log_sensitivity_definition": (
                "sum(w*kappa_component/kappa_total^2) / sum(w/kappa_total)"
            ),
            "production_continuum_stride": int(stride),
            "production_frequency_count_min": int(np.min(frequency_counts)),
            "production_frequency_count_max": int(np.max(frequency_counts)),
            "teff_x_layer_temperature_summary": _layer_fraction_summary(
                ref_labels,
                ref_temperature,
                raw_weight_array,
                inverse_fraction_array,
                sensitivity_array,
            ),
        },
        "reference_grid": {
            "teff_bands": [list(row) for row in TEFF_BANDS],
            "metallicity_bands": [list(row) for row in METALLICITY_BANDS],
            "reference_count": len(references),
            "reference_indices": [int(index) for index in references],
            "selection": (
                "nearest validation row to each Teff x metallicity cell midpoint"
            ),
        },
        "references": reference_rows,
        "scope_boundary": {
            "production_solver_changed": False,
            "temperature_iteration_run": False,
            "ode_run": False,
            "smoke_run": False,
            "funnel_run": False,
            "sealed_holdout_opened": False,
            "d3_grid_run": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/analytic_initializer/"
            "textbook_opacity_v4r1_diagnostics_20260827.json"
        ),
    )
    parser.add_argument("--split-seed", type=int, default=20260816)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    run_diagnostics(
        corpus_path=args.corpus,
        output_path=args.out,
        split_seed=args.split_seed,
        stride=args.stride,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
