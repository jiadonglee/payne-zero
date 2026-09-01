"""Run the preregistered v4r4 hot-layer production-flag ablation."""

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
    textbook_rosseland_opacity_v4r3,
)
from experiments.analytic_initializer.textbook_opacity_v4r4_hot_flag_verdict import (
    CONTROL_LAYER,
    FLAG_KNOCKOUTS,
    PRIMARY_LAYER,
    decide_hot_flag_ablation,
)
from payne_zero_atmosphere.source_catalogs import (
    molecular_equilibrium_catalog_path,
    source_line_paths,
)


FLAG_NAMES = {
    0: "H_bf_ff",
    4: "He_I",
    5: "He_II",
    8: "C_Mg_Al_Si_Fe_plus_CIA",
    9: "lukewarm_metals",
    10: "hot_metals",
}


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


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


def _assert_hot_flag_replay(
    replay: dict[str, object],
    *,
    knockout: int | None,
) -> None:
    flags = [int(value) for value in replay["opacity_flags"]]
    if flags[14] != 0 or flags[16] != 0:
        raise RuntimeError("hot-flag ablation must keep line flags 14 and 16 off")
    if int(replay["positive_line_cells"]) != 0:
        raise RuntimeError("hot-flag ablation must keep line opacity off")
    if int(replay["population_molecules_enabled"]) != 0:
        raise RuntimeError("hot-flag ablation must keep molecules off")
    if knockout is not None and flags[int(knockout)] != 0:
        raise RuntimeError(f"knockout of flag {knockout} did not take")


def _next_registered_stage(decision: dict) -> str:
    verdict = str(decision["verdict"])
    if verdict == "HYDROGEN_CONTINUUM_MISMATCH":
        return "register_hydrogen_boundfree_karzas_or_balmer_repair"
    if verdict == "HELIUM_NEUTRAL_CONTINUUM":
        return "register_helium_neutral_continuum_repair"
    if verdict == "HOT_METAL_CONTINUUM":
        return "do_not_license_line_haze"
    return "inconclusive_do_not_construct_new_opacity_law"


def _jsonable_decision(decision: dict) -> dict[str, object]:
    payload = dict(decision)
    payload["flag_effects_signed_median_dex"] = {
        str(flag): float(value)
        for flag, value in decision["flag_effects_signed_median_dex"].items()
    }
    return payload


def _json_safe(value):
    if isinstance(value, dict):
        return {
            str(key) if isinstance(key, (int, np.integer)) else key: _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def run_hot_flag_ablation(
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
    flag_effects = {flag: [] for flag in FLAG_KNOCKOUTS}
    v4r3_minus_base_rows: list[np.ndarray] = []
    reference_rows: list[dict[str, object]] = []
    frequency_counts: list[int] = []

    for ordinal, index in enumerate(references, start=1):
        labels = corpus.labels[index]
        temperature = corpus.temperature[index]
        pressure = corpus.gas_pressure[index]
        v4r3 = textbook_rosseland_opacity_v4r3(
            labels[None, :], temperature[None, :], pressure[None, :]
        )[0]
        baseline, baseline_replay = _prepare_replayed_mean(
            corpus,
            index,
            lines_enabled=False,
            stride=stride,
            line_paths=line_paths,
            molecule_path=molecule_path,
            enable_molecules=False,
        )
        _assert_hot_flag_replay(baseline_replay, knockout=None)
        knockout_opacity: dict[int, np.ndarray] = {}
        knockout_replays: dict[str, dict[str, object]] = {}
        knockout_effects: dict[int, np.ndarray] = {}
        for flag in FLAG_KNOCKOUTS:
            knocked, replay = _prepare_replayed_mean(
                corpus,
                index,
                lines_enabled=False,
                stride=stride,
                line_paths=line_paths,
                molecule_path=molecule_path,
                enable_molecules=False,
                opacity_flag_overrides={int(flag): 0},
            )
            _assert_hot_flag_replay(replay, knockout=int(flag))
            effect = np.log10(baseline / np.maximum(knocked, 1.0e-300))
            knockout_opacity[int(flag)] = knocked
            knockout_replays[str(flag)] = replay
            knockout_effects[int(flag)] = effect
            flag_effects[int(flag)].append(effect)
            frequency_counts.append(int(replay["frequency_count"]))
        minus_base = np.log10(v4r3 / np.maximum(baseline, 1.0e-300))
        reference_labels.append(np.asarray(labels, dtype=np.float64))
        reference_temperature.append(np.asarray(temperature, dtype=np.float64))
        v4r3_minus_base_rows.append(minus_base)
        frequency_counts.append(int(baseline_replay["frequency_count"]))
        reference_rows.append(
            {
                "reference_ordinal": ordinal,
                "corpus_index": int(index),
                "slug": str(corpus.slugs[index]),
                "labels": [float(value) for value in labels],
                "temperature_K": [float(value) for value in temperature],
                "v4r3_rosseland_opacity": [float(value) for value in v4r3],
                "production_continuum_baseline": [
                    float(value) for value in baseline
                ],
                "production_continuum_flag_off": {
                    str(flag): [float(value) for value in knockout_opacity[flag]]
                    for flag in FLAG_KNOCKOUTS
                },
                "flag_effect_dex": {
                    str(flag): [float(value) for value in knockout_effects[flag]]
                    for flag in FLAG_KNOCKOUTS
                },
                "v4r3_minus_base_dex": [float(value) for value in minus_base],
                "baseline_replay": baseline_replay,
                "knockout_replays": knockout_replays,
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
        f"flag_{flag}_effect": np.asarray(flag_effects[flag], dtype=np.float64)
        for flag in FLAG_KNOCKOUTS
    }
    values["v4r3_minus_base"] = np.asarray(v4r3_minus_base_rows, dtype=np.float64)
    primary_metrics = _slice_metrics(
        temperature_array,
        values,
        lower=PRIMARY_LAYER[1],
        upper=PRIMARY_LAYER[2],
    )
    if int(primary_metrics["v4r3_minus_base"]["count"]) < 1:
        raise RuntimeError("primary T >= 15000 K slice is empty")
    control_metrics = _slice_metrics(
        temperature_array,
        values,
        lower=CONTROL_LAYER[1],
        upper=CONTROL_LAYER[2],
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
        if int(metal_metrics["v4r3_minus_base"]["count"]) < 1:
            continue
        metallicity_primary_slice.append(
            {
                "metallicity_band": name,
                "metallicity_lower": lower,
                "metallicity_upper": upper,
                "metrics": metal_metrics,
            }
        )
    primary_effects = {
        int(flag): primary_metrics[f"flag_{flag}_effect"]
        for flag in FLAG_KNOCKOUTS
    }
    decision = decide_hot_flag_ablation(
        primary_effects, primary_metrics["v4r3_minus_base"]
    )
    jsonable_decision = _jsonable_decision(decision)
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "textbook_opacity_v4r4_hot_flag_ablation",
        "version": "v4r4_hot_flag_ablation",
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
            "molecules_enabled": 0,
            "stride": int(stride),
            "temperature_iteration_run": False,
            "knockouts": [int(flag) for flag in FLAG_KNOCKOUTS],
            "knockout_names": {
                str(flag): FLAG_NAMES[flag] for flag in FLAG_KNOCKOUTS
            },
            "production_frequency_count_min": int(np.min(frequency_counts)),
            "production_frequency_count_max": int(np.max(frequency_counts)),
        },
        "primary_slice": {
            "name": PRIMARY_LAYER[0],
            "temperature_lower_K": PRIMARY_LAYER[1],
            "temperature_upper_K": _finite_or_none(PRIMARY_LAYER[2]),
            "metrics": primary_metrics,
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
        "decision": jsonable_decision,
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
            "default_initializer_changed": False,
            "v4r3_candidate_changed": False,
            "v4r4_physics_changed": False,
            "gates_changed": False,
            "ode_run": False,
            "smoke_run": False,
            "funnel_run": False,
            "sealed_holdout_opened": False,
            "historical_v4_v4r1_v4r2_v4r3_opacity_changed": False,
        },
        "next_registered_stage": _next_registered_stage(decision),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            _json_safe(
                {
                    "decision": jsonable_decision,
                    "primary_slice": primary_metrics,
                }
            ),
            allow_nan=False,
        )
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
            "textbook_opacity_v4r4_hot_flag_ablation_20260828.json"
        ),
    )
    parser.add_argument("--split-seed", type=int, default=20260816)
    parser.add_argument("--stride", type=int, default=16)
    args = parser.parse_args()
    run_hot_flag_ablation(
        corpus_path=args.corpus,
        output_path=args.out,
        split_seed=args.split_seed,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
