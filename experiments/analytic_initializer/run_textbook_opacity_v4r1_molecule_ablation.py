"""Run the preregistered v4r1 molecule-on/off continuum ablation."""

from __future__ import annotations

# Must precede any Numba import.
from bench import environment as _environment  # noqa: F401

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
from experiments.analytic_initializer.run_textbook_opacity_v4_sanity import (
    LAYER_TEMPERATURE_BANDS,
    MANIFESTS,
    METALLICITY_BANDS,
    TEFF_BANDS,
    _layer_summary,
    _metrics,
    _prepare_replayed_mean,
    _reference_indices,
)
from experiments.analytic_initializer.textbook_opacity import (
    textbook_rosseland_opacity_v4r1,
)
from experiments.analytic_initializer.textbook_opacity_v4r1_molecule_verdict import (
    CONTROL_LAYER,
    PRIMARY_LAYER,
    decide_molecule_ablation,
)
from payne_zero_atmosphere.source_catalogs import (
    molecular_equilibrium_catalog_path,
    source_line_paths,
)


def _slice_metrics(
    temperature: np.ndarray,
    values: dict[str, np.ndarray],
    *,
    lower: float,
    upper: float,
    star_mask: np.ndarray | None = None,
) -> dict[str, dict[str, float | int]]:
    layer_mask = (temperature >= lower) & (temperature < upper)
    if star_mask is not None:
        layer_mask = star_mask[:, None] & layer_mask
    return {name: _metrics(array[layer_mask]) for name, array in values.items()}


def run_molecule_ablation(
    *,
    corpus_path: Path,
    output_path: Path,
    split_seed: int = 20260816,
    stride: int = 16,
) -> dict[str, object]:
    if stride < 1:
        raise ValueError("stride must be positive")
    corpus = load_strict_truth(corpus_path)
    excluded, excluded_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=split_seed)
    references = _reference_indices(corpus, split.validation)
    line_paths = source_line_paths()
    molecule_path = molecular_equilibrium_catalog_path()

    reference_labels: list[np.ndarray] = []
    reference_temperature: list[np.ndarray] = []
    molecular_effect: list[np.ndarray] = []
    v4r1_minus_atomic: list[np.ndarray] = []
    v4r1_minus_molecular: list[np.ndarray] = []
    identity_error: list[np.ndarray] = []
    reference_rows: list[dict[str, object]] = []
    frequency_counts: list[int] = []

    for ordinal, index in enumerate(references, start=1):
        labels = corpus.labels[index]
        temperature = corpus.temperature[index]
        pressure = corpus.gas_pressure[index]
        v4r1 = textbook_rosseland_opacity_v4r1(
            labels[None, :], temperature[None, :], pressure[None, :]
        )[0]
        molecules_on, on_replay = _prepare_replayed_mean(
            corpus,
            index,
            lines_enabled=False,
            stride=stride,
            line_paths=line_paths,
            molecule_path=molecule_path,
            enable_molecules=True,
        )
        molecules_off, off_replay = _prepare_replayed_mean(
            corpus,
            index,
            lines_enabled=False,
            stride=stride,
            line_paths=line_paths,
            molecule_path=molecule_path,
            enable_molecules=False,
        )
        if on_replay["population_molecules_enabled"] != 1:
            raise RuntimeError("molecules-on replay did not enable molecules")
        if off_replay["population_molecules_enabled"] != 0:
            raise RuntimeError("molecules-off replay did not disable molecules")
        if (
            on_replay["positive_line_cells"] != 0
            or off_replay["positive_line_cells"] != 0
        ):
            raise RuntimeError("molecule ablation must keep line opacity off")
        effect = np.log10(molecules_on / np.maximum(molecules_off, 1.0e-300))
        minus_atomic = np.log10(v4r1 / np.maximum(molecules_off, 1.0e-300))
        minus_molecular = np.log10(v4r1 / np.maximum(molecules_on, 1.0e-300))
        identity = minus_atomic - (minus_molecular + effect)
        reference_labels.append(np.asarray(labels, dtype=np.float64))
        reference_temperature.append(np.asarray(temperature, dtype=np.float64))
        molecular_effect.append(effect)
        v4r1_minus_atomic.append(minus_atomic)
        v4r1_minus_molecular.append(minus_molecular)
        identity_error.append(identity)
        frequency_counts.extend(
            [on_replay["frequency_count"], off_replay["frequency_count"]]
        )
        reference_rows.append(
            {
                "reference_ordinal": ordinal,
                "corpus_index": int(index),
                "slug": str(corpus.slugs[index]),
                "labels": [float(value) for value in labels],
                "temperature_K": [float(value) for value in temperature],
                "v4r1_rosseland_opacity": [float(value) for value in v4r1],
                "production_continuum_molecules_on": [
                    float(value) for value in molecules_on
                ],
                "production_continuum_molecules_off": [
                    float(value) for value in molecules_off
                ],
                "molecular_effect_dex": [float(value) for value in effect],
                "v4r1_minus_atomic_dex": [float(value) for value in minus_atomic],
                "v4r1_minus_molecular_dex": [
                    float(value) for value in minus_molecular
                ],
                "molecules_on_replay": on_replay,
                "molecules_off_replay": off_replay,
            }
        )
        print(
            f"[{ordinal:02d}/{len(references):02d}] index={index} "
            f"Teff={labels[0]:.0f} [M/H]={labels[2]:+.2f}",
            flush=True,
        )

    labels_array = np.asarray(reference_labels, dtype=np.float64)
    temperature_array = np.asarray(reference_temperature, dtype=np.float64)
    values = {
        "molecular_effect": np.asarray(molecular_effect, dtype=np.float64),
        "v4r1_minus_atomic": np.asarray(v4r1_minus_atomic, dtype=np.float64),
        "v4r1_minus_molecular": np.asarray(v4r1_minus_molecular, dtype=np.float64),
    }
    identity_array = np.asarray(identity_error, dtype=np.float64)
    primary_metrics = _slice_metrics(
        temperature_array,
        values,
        lower=PRIMARY_LAYER[1],
        upper=PRIMARY_LAYER[2],
    )
    if int(primary_metrics["molecular_effect"]["count"]) < 1:
        raise RuntimeError("primary 3200-4000 K slice is empty")
    control_metrics = _slice_metrics(
        temperature_array,
        values,
        lower=CONTROL_LAYER[1],
        upper=CONTROL_LAYER[2],
    )
    cool_primary_metrics = _slice_metrics(
        temperature_array,
        values,
        lower=PRIMARY_LAYER[1],
        upper=PRIMARY_LAYER[2],
        star_mask=labels_array[:, 0] < 6000.0,
    )
    metallicity_primary_slice: list[dict[str, object]] = []
    for name, lower, upper in METALLICITY_BANDS:
        metal_metrics = _slice_metrics(
            temperature_array,
            values,
            lower=PRIMARY_LAYER[1],
            upper=PRIMARY_LAYER[2],
            star_mask=(
                (labels_array[:, 2] >= lower)
                & (
                    (labels_array[:, 2] < upper)
                    if upper < 0.5
                    else (labels_array[:, 2] <= upper)
                )
            ),
        )
        if int(metal_metrics["molecular_effect"]["count"]) < 1:
            continue
        metallicity_primary_slice.append(
            {
                "metallicity_band": name,
                "metallicity_lower": lower,
                "metallicity_upper": upper,
                "metrics": metal_metrics,
            }
        )
    decision = decide_molecule_ablation(primary_metrics)
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "textbook_opacity_v4r1_molecule_ablation",
        "version": "v4r1",
        "status": "diagnostic_only",
        "corpus": str(corpus.path),
        "corpus_sha256": file_sha256(corpus.path),
        "split": {
            "seed": int(split.seed),
            "validation_count": int(split.validation.size),
            "excluded_count": int(excluded.size),
            "excluded_manifests": excluded_manifests,
            "sealed_rows_read": False,
        },
        "replay": {
            "line_flags_15_and_17": 0,
            "stride": int(stride),
            "temperature_iteration_run": False,
            "production_frequency_count_min": int(np.min(frequency_counts)),
            "production_frequency_count_max": int(np.max(frequency_counts)),
            "log_identity_max_abs_dex": float(np.max(np.abs(identity_array))),
        },
        "primary_slice": {
            "name": PRIMARY_LAYER[0],
            "temperature_lower_K": PRIMARY_LAYER[1],
            "temperature_upper_K": PRIMARY_LAYER[2],
            "metrics": primary_metrics,
            "cool_teff_below_6000K_metrics": cool_primary_metrics,
        },
        "control_slice": {
            "name": CONTROL_LAYER[0],
            "temperature_lower_K": CONTROL_LAYER[1],
            "temperature_upper_K": CONTROL_LAYER[2],
            "metrics": control_metrics,
        },
        "teff_x_layer_temperature_summary": _layer_summary(
            labels_array, temperature_array, values
        ),
        "metallicity_primary_slice": metallicity_primary_slice,
        "decision": decision,
        "reference_grid": {
            "teff_bands": [list(row) for row in TEFF_BANDS],
            "metallicity_bands": [list(row) for row in METALLICITY_BANDS],
            "layer_temperature_bands": [
                [name, lower, None if np.isinf(upper) else upper]
                for name, lower, upper in LAYER_TEMPERATURE_BANDS
            ],
            "reference_count": len(references),
            "reference_indices": [int(index) for index in references],
        },
        "references": reference_rows,
        "scope_boundary": {
            "production_solver_changed": False,
            "v4r1_candidate_changed": False,
            "gates_changed": False,
            "ode_run": False,
            "smoke_run": False,
            "funnel_run": False,
            "sealed_holdout_opened": False,
            "v4r2_constructed": False,
        },
        "next_registered_stage": (
            "register_domain_or_named_molecular_term"
            if decision["verdict"] == "MOLECULAR_CONTINUUM_DOMINATES"
            else (
                "diagnose_atomic_infrared_continuum"
                if decision["verdict"] == "ATOMIC_IR_REMAINS"
                else "do_not_construct_v4r2_until_verdict_is_unmixed"
            )
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "primary_slice": primary_metrics}))
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
            "textbook_opacity_v4r1_molecule_ablation_20260827.json"
        ),
    )
    parser.add_argument("--split-seed", type=int, default=20260816)
    parser.add_argument("--stride", type=int, default=16)
    args = parser.parse_args()
    run_molecule_ablation(
        corpus_path=args.corpus,
        output_path=args.out,
        split_seed=args.split_seed,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
