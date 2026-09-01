"""Run the fixed-reference v4 line-blanketing and continuum-reachability check.

This is a diagnostic before any v4 full-corpus gate.  It replays the exact
production continuum and line-opacity preparation on a fixed, manifest-aware
20-star development sample, then evaluates the same Rosseland inverse-opacity
sum once with lines and once with lines disabled.  It does not run a
temperature iteration, alter the production solver, or inspect sealed rows.

The v4 candidate is evaluated separately on the truth ``(T, P)`` layers.  The
comparison to the line-free replay estimates how much of the residual against
the stored corpus ``kappa_R`` can be a line-blanketing floor rather than a
missing continuum component.
"""

from __future__ import annotations

# Must precede any Numba import.
from bench import environment as _environment  # noqa: F401

import argparse
import json
import re
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.textbook_opacity import (
    textbook_rosseland_opacity_v4,
)
from payne_zero_atmosphere.atmosphere_io import parse_atmosphere_deck
from payne_zero_atmosphere.config import (
    AtmosphereConfig,
    AtmosphereInput,
    AtmosphereOutput,
    DEFAULT_OPACITY_FLAGS,
)
from payne_zero_atmosphere.runner import (
    prepare_opacity_state,
    prepare_population_state,
)
from payne_zero_atmosphere.source_catalogs import (
    molecular_equilibrium_catalog_path,
    source_line_paths,
)
from payne_zero_atmosphere.warm_start import format_warm_start_deck


MANIFESTS = (
    Path("results/reconstruction_metrics.json"),
    Path("results/sealed_solver_subset_20260808.json"),
    Path("results/sealed_audit_20260808.json"),
    Path("results/sealed_audit_20260811.json"),
    Path("results/sealed_initializer_holdout_20260812.json"),
    Path("results/initializer_calibration_20260812.json"),
    Path("results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json"),
)

TEFF_BANDS = (
    ("4000_5000K", 4000.0, 5000.0),
    ("5000_6000K", 5000.0, 6000.0),
    ("6000_7000K", 6000.0, 7000.0),
    ("7000_9000K", 7000.0, 9000.0),
    ("9000_11000K", 9000.0, 11000.0),
)
METALLICITY_BANDS = (
    ("metal_poor", -2.5, -1.5),
    ("metal_intermediate", -1.5, -0.5),
    ("metal_solar", -0.5, 0.25),
    ("metal_rich", 0.25, 0.5),
)
LAYER_TEMPERATURE_BANDS = (
    ("3200_4000K", 3200.0, 4000.0),
    ("4000_5000K", 4000.0, 5000.0),
    ("5000_6000K", 5000.0, 6000.0),
    ("6000_7000K", 6000.0, 7000.0),
    ("7000_10000K", 7000.0, 10000.0),
    ("10000_15000K", 10000.0, 15000.0),
    ("at_least_15000K", 15000.0, np.inf),
)


def _metrics(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "signed_median_dex": float("nan"),
            "signed_mean_dex": float("nan"),
            "positive_fraction": float("nan"),
            "p50_abs_dex": float("nan"),
            "p95_abs_dex": float("nan"),
            "maximum_abs_dex": float("nan"),
        }
    return {
        "count": int(array.size),
        "signed_median_dex": float(np.median(array)),
        "signed_mean_dex": float(np.mean(array)),
        "positive_fraction": float(np.mean(array > 0.0)),
        "p50_abs_dex": float(np.percentile(np.abs(array), 50.0)),
        "p95_abs_dex": float(np.percentile(np.abs(array), 95.0)),
        "maximum_abs_dex": float(np.max(np.abs(array))),
    }


def _reference_indices(corpus, validation: np.ndarray) -> list[int]:
    """Select one deterministic development row per Teff x [M/H] cell."""

    selected: list[int] = []
    for _, teff_lower, teff_upper in TEFF_BANDS:
        teff_mask = (corpus.labels[:, 0] >= teff_lower) & (
            corpus.labels[:, 0] < teff_upper
        )
        for _, metallicity_lower, metallicity_upper in METALLICITY_BANDS:
            metal_mask = (corpus.labels[:, 2] >= metallicity_lower) & (
                (corpus.labels[:, 2] < metallicity_upper)
                if metallicity_upper < 0.5
                else (corpus.labels[:, 2] <= metallicity_upper)
            )
            candidates = np.intersect1d(
                validation,
                np.flatnonzero(teff_mask & metal_mask),
                assume_unique=False,
            )
            candidates = np.asarray(
                [index for index in candidates if int(index) not in selected],
                dtype=np.int64,
            )
            if candidates.size == 0:
                raise RuntimeError(
                    "fixed v4 reference grid has an empty validation cell: "
                    f"Teff={teff_lower:g}--{teff_upper:g}, "
                    f"[M/H]={metallicity_lower:g}--{metallicity_upper:g}"
                )
            teff_midpoint = 0.5 * (teff_lower + teff_upper)
            metal_midpoint = 0.5 * (metallicity_lower + metallicity_upper)
            score = (
                np.abs(corpus.labels[candidates, 0] - teff_midpoint)
                / (teff_upper - teff_lower)
                + np.abs(corpus.labels[candidates, 2] - metal_midpoint)
                / (metallicity_upper - metallicity_lower)
            )
            order = np.lexsort((candidates, score))
            selected.append(int(candidates[order[0]]))
    return selected


def _atmosphere_from_corpus(
    corpus,
    index: int,
    *,
    lines_enabled: bool,
    opacity_flag_overrides: dict[int, int] | None = None,
):
    labels = np.asarray(corpus.labels[index], dtype=np.float64)
    table = np.zeros((corpus.layers, 9), dtype=np.float64)
    table[:, 0] = corpus.column_mass[index]
    table[:, 1] = corpus.temperature[index]
    table[:, 2] = corpus.gas_pressure[index]
    table[:, 3] = corpus.electron_density[index]
    table[:, 4] = corpus.rosseland_opacity[index]
    table[:, 6] = labels[4]
    deck = format_warm_start_deck(
        effective_temperature=labels[0],
        log_surface_gravity=labels[1],
        layer_table=table,
        metallicity=labels[2],
        alpha_enhancement=labels[3],
        title=f"v4 line blanketing sanity index {index}",
    )
    atmosphere = parse_atmosphere_deck(deck, source="v4 sanity reference")
    flags = [int(value) for value in re.findall(r"-?\d+", atmosphere.metadata["opacity_flags"])]
    if len(flags) < 20:
        flags = list(DEFAULT_OPACITY_FLAGS)
    else:
        flags = flags[-20:]
    flags[14] = int(bool(lines_enabled))
    flags[16] = int(bool(lines_enabled))
    if opacity_flag_overrides:
        for flag_index, flag_value in opacity_flag_overrides.items():
            flags[int(flag_index)] = int(flag_value)
    atmosphere.metadata["opacity_flags"] = "OPACITY IFOP " + " ".join(
        str(value) for value in flags
    )
    return atmosphere


def _config(
    atmosphere,
    *,
    stride: int,
    line_paths: dict[str, Path],
    molecule_path: Path,
    enable_molecules: bool = True,
):
    return AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=atmosphere,
            molecules_path=molecule_path,
            **line_paths,
        ),
        outputs=AtmosphereOutput(),
        iterations=1,
        enable_molecules=bool(enable_molecules),
        enable_convection=False,
        enable_convergence_stop=False,
        opacity_frequency_grid_stride=int(stride),
    )


def _rosseland_from_slabs(opacity_state) -> np.ndarray:
    """Apply the production mode-2/mode-3 Rosseland algebra to opacity slabs."""

    setup = opacity_state.population_state.setup
    atmosphere = setup.atmosphere
    frequency = np.asarray(opacity_state.opacity_frequency_hz, dtype=np.float64)
    temperature = np.asarray(atmosphere.temperature, dtype=np.float64)
    h_over_kt = np.asarray(atmosphere.h_over_kt, dtype=np.float64)
    exponential = np.exp(-frequency[:, None] * h_over_kt[None, :])
    stimulated = np.maximum(1.0 - exponential, 1.0e-300)
    planck = (
        1.47439e-2
        * (frequency[:, None] / 1.0e15) ** 3
        * exponential
        / stimulated
    )
    source_derivative = (
        planck
        * frequency[:, None]
        * h_over_kt[None, :]
        / (temperature[None, :] * stimulated)
    )
    total = (
        np.asarray(opacity_state.continuum_absorption, dtype=np.float64)
        + np.asarray(opacity_state.continuum_scattering, dtype=np.float64)
        + np.asarray(
            opacity_state.line_opacity.line_mass_absorption_coefficient,
            dtype=np.float64,
        )
    )
    accumulator = np.sum(
        source_derivative
        / np.maximum(total.T, 1.0e-300)
        * np.asarray(opacity_state.frequency_weights, dtype=np.float64)[:, None],
        axis=0,
    )
    return 4.0 * (5.6697e-5 / 3.14159) * temperature**3 / np.maximum(
        accumulator, 1.0e-300
    )


def _prepare_replayed_mean(
    corpus,
    index: int,
    *,
    lines_enabled: bool,
    stride: int,
    line_paths: dict[str, Path],
    molecule_path: Path,
    enable_molecules: bool = True,
    opacity_flag_overrides: dict[int, int] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    atmosphere = _atmosphere_from_corpus(
        corpus,
        index,
        lines_enabled=lines_enabled,
        opacity_flag_overrides=opacity_flag_overrides,
    )
    flags = [
        int(value)
        for value in re.findall(r"-?\d+", atmosphere.metadata["opacity_flags"])
    ][-20:]
    config = _config(
        atmosphere,
        stride=stride,
        line_paths=line_paths,
        molecule_path=molecule_path,
        enable_molecules=enable_molecules,
    )
    population = prepare_population_state(config)
    opacity = prepare_opacity_state(config, population_state=population)
    return _rosseland_from_slabs(opacity), {
        "frequency_count": int(opacity.opacity_frequency_hz.size),
        "positive_line_cells": int(
            np.count_nonzero(opacity.line_opacity.line_mass_absorption_coefficient > 0.0)
        ),
        "line_catalog_selected_count": int(
            0
            if opacity.selected_line_catalog is None
            else opacity.selected_line_catalog.line_count
        ),
        "molecules_enabled": int(bool(enable_molecules)),
        "population_molecules_enabled": int(
            bool(population.setup.molecules_enabled)
        ),
        "opacity_flags": flags,
    }


def _layer_summary(
    labels: np.ndarray,
    temperature: np.ndarray,
    values: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for teff_name, teff_lower, teff_upper in TEFF_BANDS:
        star_mask = (labels[:, 0] >= teff_lower) & (labels[:, 0] < teff_upper)
        for layer_name, layer_lower, layer_upper in LAYER_TEMPERATURE_BANDS:
            layer_mask = (
                star_mask[:, None]
                & (temperature >= layer_lower)
                & (temperature < layer_upper)
            )
            if not np.any(layer_mask):
                continue
            record: dict[str, object] = {
                "teff_band": teff_name,
                "layer_temperature_band": layer_name,
                "layer_temperature_lower_K": layer_lower,
                "layer_temperature_upper_K": (
                    None if np.isinf(layer_upper) else layer_upper
                ),
            }
            for name, array in values.items():
                record[name] = _metrics(array[layer_mask])
            rows.append(record)
    return rows


def _metallicity_summary(
    labels: np.ndarray,
    temperature: np.ndarray,
    values: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cool_transition = (
        ((labels[:, 0] >= 4000.0) & (labels[:, 0] < 7000.0))[:, None]
        & (temperature >= 4000.0)
        & (temperature < 7000.0)
    )
    for name, lower, upper in METALLICITY_BANDS:
        mask = cool_transition & (labels[:, 2, None] >= lower)
        if upper < 0.5:
            mask &= labels[:, 2, None] < upper
        else:
            mask &= labels[:, 2, None] <= upper
        if not np.any(mask):
            continue
        record: dict[str, object] = {
            "metallicity_band": name,
            "metallicity_lower": lower,
            "metallicity_upper": upper,
        }
        for value_name, array in values.items():
            record[value_name] = _metrics(array[mask])
        rows.append(record)
    return rows


def run_sanity(
    *,
    corpus_path: Path,
    output_path: Path,
    sample_count: int = 20,
    stride: int = 16,
    split_seed: int = 20260816,
) -> dict[str, object]:
    if int(sample_count) != len(TEFF_BANDS) * len(METALLICITY_BANDS):
        raise ValueError(
            "the registered v4 reference design is exactly "
            f"{len(TEFF_BANDS) * len(METALLICITY_BANDS)} points"
        )
    if int(stride) < 1:
        raise ValueError("stride must be positive")
    corpus = load_strict_truth(corpus_path)
    excluded, excluded_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=split_seed)
    references = _reference_indices(corpus, split.validation)
    line_paths = source_line_paths()
    molecule_path = molecular_equilibrium_catalog_path()

    profile_rows: list[dict[str, object]] = []
    frequency_counts: list[int] = []
    positive_line_cells: list[int] = []
    selected_line_counts: list[int] = []
    for ordinal, index in enumerate(references, start=1):
        labels = corpus.labels[index]
        temperature = corpus.temperature[index]
        pressure = corpus.gas_pressure[index]
        truth = corpus.rosseland_opacity[index]
        v4 = textbook_rosseland_opacity_v4(
            labels[None, :], temperature[None, :], pressure[None, :]
        )[0]
        with_lines, line_diagnostics = _prepare_replayed_mean(
            corpus,
            index,
            lines_enabled=True,
            stride=stride,
            line_paths=line_paths,
            molecule_path=molecule_path,
        )
        continuum_only, continuum_diagnostics = _prepare_replayed_mean(
            corpus,
            index,
            lines_enabled=False,
            stride=stride,
            line_paths=line_paths,
            molecule_path=molecule_path,
        )
        line_effect = np.log10(with_lines / np.maximum(continuum_only, 1.0e-300))
        v4_minus_truth = np.log10(v4 / np.maximum(truth, 1.0e-300))
        v4_minus_continuum = np.log10(v4 / np.maximum(continuum_only, 1.0e-300))
        profile_rows.append(
            {
                "reference_ordinal": ordinal,
                "corpus_index": int(index),
                "slug": str(corpus.slugs[index]),
                "labels": [float(value) for value in labels],
                "temperature_K": [float(value) for value in temperature],
                "truth_rosseland_opacity": [float(value) for value in truth],
                "v4_rosseland_opacity": [float(value) for value in v4],
                "line_plus_continuum_rosseland_opacity": [
                    float(value) for value in with_lines
                ],
                "continuum_only_rosseland_opacity": [
                    float(value) for value in continuum_only
                ],
                "line_effect_dex": [float(value) for value in line_effect],
                "v4_minus_truth_dex": [float(value) for value in v4_minus_truth],
                "v4_minus_continuum_dex": [
                    float(value) for value in v4_minus_continuum
                ],
                "line_replay": line_diagnostics,
                "continuum_replay": continuum_diagnostics,
            }
        )
        frequency_counts.append(line_diagnostics["frequency_count"])
        positive_line_cells.append(line_diagnostics["positive_line_cells"])
        selected_line_counts.append(line_diagnostics["line_catalog_selected_count"])
        print(
            f"[{ordinal:02d}/{len(references):02d}] index={index} "
            f"Teff={labels[0]:.0f} [M/H]={labels[2]:+.2f} "
            f"line_effect_p95={np.percentile(np.abs(line_effect), 95.0):.3f} dex",
            flush=True,
        )

    labels = np.asarray([row["labels"] for row in profile_rows], dtype=np.float64)
    temperature = np.asarray(
        [row["temperature_K"] for row in profile_rows], dtype=np.float64
    )
    line_effect = np.asarray(
        [row["line_effect_dex"] for row in profile_rows], dtype=np.float64
    )
    v4_minus_truth = np.asarray(
        [row["v4_minus_truth_dex"] for row in profile_rows], dtype=np.float64
    )
    v4_minus_continuum = np.asarray(
        [row["v4_minus_continuum_dex"] for row in profile_rows], dtype=np.float64
    )
    summary_values = {
        "line_effect": line_effect,
        "v4_minus_total_truth": v4_minus_truth,
        "v4_minus_continuum_replay": v4_minus_continuum,
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "textbook_opacity_v4_line_blanketing_sanity",
        "status": "diagnostic_only",
        "corpus": {
            "path": str(corpus.path),
            "size": corpus.size,
            "layers": corpus.layers,
            "stored_kappa_R_provenance": "fully_converged_solver_output",
        },
        "split": {
            "seed": int(split.seed),
            "validation_count": int(split.validation.size),
            "excluded_count": int(excluded.size),
            "excluded_manifests": excluded_manifests,
            "reference_count": len(references),
            "reference_indices": references,
        },
        "replay_contract": {
            "production_continuum_path": True,
            "production_line_path": True,
            "molecules_enabled": True,
            "line_flags_with_lines": {"ifop_15_selected": 1, "ifop_17_detailed": 1},
            "line_flags_without_lines": {"ifop_15_selected": 0, "ifop_17_detailed": 0},
            "opacity_frequency_grid_stride": int(stride),
            "rosseland_algebra": "same mode-2 inverse-opacity sum and mode-3 normalization",
            "temperature_iteration": False,
            "sealed_rows_read": False,
        },
        "reference_grid": {
            "teff_bands": [list(row) for row in TEFF_BANDS],
            "metallicity_bands": [list(row) for row in METALLICITY_BANDS],
            "selection": "nearest remaining validation row to the cell midpoint; corpus index tie-break",
        },
        "overall": {
            name: _metrics(array) for name, array in summary_values.items()
        },
        "metallicity_summary_4000_7000K": _metallicity_summary(
            labels,
            temperature,
            summary_values,
        ),
        "teff_x_layer_temperature_summary": _layer_summary(
            labels,
            temperature,
            summary_values,
        ),
        "replay_runtime_summary": {
            "frequency_count_min": int(np.min(frequency_counts)),
            "frequency_count_max": int(np.max(frequency_counts)),
            "positive_line_cells_total": int(np.sum(positive_line_cells)),
            "selected_line_catalog_count_min": int(np.min(selected_line_counts)),
            "selected_line_catalog_count_max": int(np.max(selected_line_counts)),
        },
        "references": profile_rows,
        "interpretation": {
            "line_effect_is_measured": True,
            "stored_kappa_R_is_total_opacity_target": True,
            "v4_is_continuum_only": True,
            "formal_gate_decision": "deferred_until_preregistered_v4_after_sanity",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/analytic_initializer/textbook_opacity_v4_sanity_20260826.json"
        ),
    )
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=20260816)
    args = parser.parse_args()
    run_sanity(
        corpus_path=args.corpus,
        output_path=args.out,
        sample_count=args.sample_count,
        stride=args.stride,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()
