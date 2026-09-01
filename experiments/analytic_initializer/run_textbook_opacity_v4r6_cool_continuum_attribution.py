"""Preregistered B2 attribution of the cool 3200-4000 K continuum miss.

Diagnostic only.  Reuses the frozen 20-star production-continuum ablation
grid and adds exactly two new production replays per row: IFOP(2) (H2+) off
and IFOP(7) (He-minus) off, with otherwise identical settings to the stored
ablation (stride-16 grid, lines off via IFOP(15)=IFOP(17)=0, molecules off,
no temperature iteration).  Candidate v4r5/v4r6 node components are evaluated
on the same stored ``(P, T)`` rows so candidate H2+/He- magnitudes sit next
to the production knockout effects.

This runner does not change production opacity, the candidate, any gate, or
the mass integral, and it does not run the solver, ODE, funnel, or sealed
holdout.  Production calls must run on the remote Linux host, never from the
macOS .venv.
"""

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
    MANIFESTS,
    _metrics,
    _prepare_replayed_mean,
    _reference_indices,
)
from experiments.analytic_initializer.run_textbook_opacity_v4r4_hot_flag_ablation import (
    _assert_hot_flag_replay,
)
from experiments.analytic_initializer.run_textbook_opacity_v4r5_cool_mass_decomposition import (
    _rosseland_diagnostics,
)
from experiments.analytic_initializer.textbook_opacity import (
    COMPONENT_NAMES_V4R5,
    textbook_opacity_node_components_v4r5,
    textbook_opacity_node_components_v4r6,
)
from payne_zero_atmosphere.source_catalogs import (
    molecular_equilibrium_catalog_path,
    source_line_paths,
)


HOT_FLAG_ABLATION = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r4_hot_flag_ablation_20260828.json"
)
REGISTERED_OUTPUT = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r6_cool_continuum_attribution_20260828.json"
)

# 0-based Python indices into the 20-element OPACITY IFOP vector:
# index 1 = IFOP(2) H2+, index 6 = IFOP(7) He-minus.
NEW_KNOCKOUTS = {1: "IFOP2_H2plus", 6: "IFOP7_He_minus"}
STORED_KNOCKOUTS = (0, 4, 5, 8, 9, 10)

COOL_TEFF_MAX_K = 6000.0
SLICE_LOWER_K = 3200.0
SLICE_UPPER_K = 4000.0
EXPECTED_COOL_STAR_COUNT = 8
EXPECTED_SLICE_LAYER_COUNT = 199
TEMPERATURE_DRIFT_TOLERANCE_K = 1.0e-6
BASELINE_REPRODUCE_MEDIAN_ABS_MAX_DEX = 1.0e-6

H2PLUS_IMPLEMENTATION_MIN_DEX = 0.03
HEMINUS_IMPLEMENTATION_MIN_DEX = 0.03
NULL_KNOCKOUT_MAX_DEX = 0.01
BASE_FRACTION_MIN = 0.90
BASE_HMINUS_SENSITIVITY_MIN = 0.80
BASE_GAP_SIGNED_MEDIAN_MAX_DEX = -0.05

HOT_FLAG_ABLATION_SHA256 = (
    "c136b076d5f135733e4d7e43081d2ed8040f3586b0f4cbd01283628dda613b66"
)
COOL_MASS_DECOMPOSITION_SHA256 = (
    "6115c8c78c3ab583fac2fa47b964224f0346588713ac55a39ced54bad3c0bcf1"
)


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


def _subset_rosseland(
    components: dict[str, np.ndarray], exclude: tuple[str, ...]
) -> np.ndarray:
    """Rosseland mean of the candidate with named components removed."""

    total = np.asarray(components["total"], dtype=np.float64)
    for name in exclude:
        total = total - np.asarray(components[name], dtype=np.float64)
    weights = components["node_weights"]
    return 1.0 / np.sum(weights / np.maximum(total, 1.0e-30), axis=(-2, -1))


def _median_or_none(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return float(np.median(array))


def _decide(
    *,
    sanity_failures: list[str],
    ifop2_signed_median_dex: float,
    ifop7_signed_median_dex: float,
    implied_base_fraction_median: float,
    candidate_hminus_sensitivity_median: float,
    base_gap_signed_median_dex: float,
) -> dict[str, object]:
    """Apply the registered verdict rule for the cool-continuum attribution."""

    primary = (ifop2_signed_median_dex, ifop7_signed_median_dex)
    if sanity_failures or not all(np.isfinite(value) for value in primary):
        return {
            "verdict": "INCONCLUSIVE",
            "sanity_failures": list(sanity_failures),
            "named_base_component": None,
            "base_naming_reason": "not_evaluated",
        }
    m2 = float(ifop2_signed_median_dex)
    m7 = float(ifop7_signed_median_dex)
    named = None
    naming_reason = "not_applicable"
    if m2 >= H2PLUS_IMPLEMENTATION_MIN_DEX:
        verdict = "H2PLUS_IMPLEMENTATION"
    elif m7 >= HEMINUS_IMPLEMENTATION_MIN_DEX:
        verdict = "HEMINUS_IMPLEMENTATION"
    elif m2 < NULL_KNOCKOUT_MAX_DEX and m7 < NULL_KNOCKOUT_MAX_DEX:
        verdict = "BASE_CONTINUUM"
        isolates = (
            float(implied_base_fraction_median) >= BASE_FRACTION_MIN
            and float(candidate_hminus_sensitivity_median)
            >= BASE_HMINUS_SENSITIVITY_MIN
            and float(base_gap_signed_median_dex) <= BASE_GAP_SIGNED_MEDIAN_MAX_DEX
        )
        if isolates:
            named = (
                "production H-minus free-free/bound-free implementation "
                "(versus candidate John 1988); electron-scattering and "
                "Rayleigh terms are not separately knocked out"
            )
            naming_reason = "bookkeeping_isolates_hminus_base"
        else:
            naming_reason = "bookkeeping_does_not_isolate_a_single_base_component"
    else:
        verdict = "UNRESOLVED"
    return {
        "verdict": verdict,
        "sanity_failures": [],
        "named_base_component": named,
        "base_naming_reason": naming_reason,
        "thresholds": {
            "h2plus_implementation_min_dex": H2PLUS_IMPLEMENTATION_MIN_DEX,
            "heminus_implementation_min_dex": HEMINUS_IMPLEMENTATION_MIN_DEX,
            "null_knockout_max_dex": NULL_KNOCKOUT_MAX_DEX,
            "base_fraction_min": BASE_FRACTION_MIN,
            "base_hminus_sensitivity_min": BASE_HMINUS_SENSITIVITY_MIN,
            "base_gap_signed_median_max_dex": BASE_GAP_SIGNED_MEDIAN_MAX_DEX,
        },
    }


def run_attribution(
    *,
    corpus_path: Path,
    ablation_path: Path,
    output_path: Path,
    split_seed: int = 20260816,
    stride: int = 16,
) -> dict[str, object]:
    if stride < 1:
        raise ValueError("stride must be positive")
    corpus = load_strict_truth(corpus_path)
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    stored_references = ablation["references"]
    stored_indices = [int(row["corpus_index"]) for row in stored_references]

    excluded, excluded_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=split_seed)
    recomputed_indices = [int(index) for index in _reference_indices(corpus, split.validation)]
    indices_match = recomputed_indices == stored_indices

    indices = np.asarray(stored_indices, dtype=np.int64)
    labels = np.asarray(corpus.labels[indices], dtype=np.float64)
    temperature = np.asarray(corpus.temperature[indices], dtype=np.float64)
    pressure = np.asarray(corpus.gas_pressure[indices], dtype=np.float64)
    stored_temperature = np.asarray(
        [row["temperature_K"] for row in stored_references], dtype=np.float64
    )
    stored_baseline = np.asarray(
        [row["production_continuum_baseline"] for row in stored_references],
        dtype=np.float64,
    )
    temperature_drift_max_K = float(
        np.max(np.abs(temperature - stored_temperature))
    )

    cool = labels[:, 0] < COOL_TEFF_MAX_K
    slice_mask = (
        cool[:, None] & (temperature >= SLICE_LOWER_K) & (temperature < SLICE_UPPER_K)
    )
    slice_layer_count = int(np.sum(slice_mask))
    cool_star_count = int(np.sum(cool))

    line_paths = source_line_paths()
    molecule_path = molecular_equilibrium_catalog_path()

    replayed_baseline = np.empty_like(stored_baseline)
    knockout_opacity = {
        flag: np.empty_like(stored_baseline) for flag in NEW_KNOCKOUTS
    }
    frequency_counts: list[int] = []
    reference_rows: list[dict[str, object]] = []
    for ordinal, index in enumerate(stored_indices, start=1):
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
        replayed_baseline[ordinal - 1] = baseline
        frequency_counts.append(int(baseline_replay["frequency_count"]))
        row: dict[str, object] = {
            "reference_ordinal": ordinal,
            "corpus_index": int(index),
            "baseline_replay": baseline_replay,
            "knockout_replays": {},
        }
        for flag, name in NEW_KNOCKOUTS.items():
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
            knockout_opacity[flag][ordinal - 1] = knocked
            frequency_counts.append(int(replay["frequency_count"]))
            row["knockout_replays"][name] = replay
        reference_rows.append(row)
        print(
            f"[{ordinal:02d}/{len(stored_indices):02d}] index={index} "
            f"Teff={labels[ordinal - 1, 0]:.0f} replays=3",
            flush=True,
        )

    baseline_diff = np.log10(np.maximum(replayed_baseline, 1.0e-300)) - np.log10(
        np.maximum(stored_baseline, 1.0e-300)
    )
    baseline_diff_slice = np.abs(baseline_diff[slice_mask])
    baseline_reproduce_median_abs_dex = float(np.median(baseline_diff_slice))
    baseline_reproduce_max_abs_dex = float(np.max(baseline_diff_slice))

    knockout_effects = {
        flag: np.log10(
            np.maximum(replayed_baseline, 1.0e-300)
            / np.maximum(knockout_opacity[flag], 1.0e-300)
        )
        for flag in NEW_KNOCKOUTS
    }

    components_v4r5 = textbook_opacity_node_components_v4r5(
        labels, temperature, pressure
    )
    components_v4r6 = textbook_opacity_node_components_v4r6(
        labels, temperature, pressure
    )
    _, sensitivity_v4r5 = _rosseland_diagnostics(components_v4r5)
    _, sensitivity_v4r6 = _rosseland_diagnostics(components_v4r6)
    v4r5_total = _subset_rosseland(components_v4r5, ())
    v4r6_total = _subset_rosseland(components_v4r6, ())
    v4r5_without_h2plus = _subset_rosseland(components_v4r5, ("h2plus",))
    v4r5_without_heminus = _subset_rosseland(components_v4r5, ("heminus",))
    v4r5_base = _subset_rosseland(components_v4r5, ("h2plus", "heminus"))

    candidate_h2plus_effect = np.log10(
        v4r5_total / np.maximum(v4r5_without_h2plus, 1.0e-300)
    )
    candidate_heminus_effect = np.log10(
        v4r5_total / np.maximum(v4r5_without_heminus, 1.0e-300)
    )
    candidate_h2plus_heminus_effect = np.log10(
        v4r5_total / np.maximum(v4r5_base, 1.0e-300)
    )
    v4r5_minus_production = np.log10(
        v4r5_total / np.maximum(stored_baseline, 1.0e-300)
    )
    v4r6_minus_production = np.log10(
        v4r6_total / np.maximum(stored_baseline, 1.0e-300)
    )

    implied_flag_contribution = np.zeros_like(stored_baseline)
    for flag in STORED_KNOCKOUTS:
        flag_off = np.asarray(
            [row["production_continuum_flag_off"][str(flag)] for row in stored_references],
            dtype=np.float64,
        )
        implied_flag_contribution += stored_baseline - flag_off
    for flag in NEW_KNOCKOUTS:
        implied_flag_contribution += stored_baseline - knockout_opacity[flag]
    implied_base = np.maximum(
        stored_baseline - implied_flag_contribution, 1.0e-300
    )
    implied_base_fraction = implied_base / np.maximum(stored_baseline, 1.0e-300)
    base_gap = np.log10(v4r5_base / implied_base)

    component_index = {name: i for i, name in enumerate(COMPONENT_NAMES_V4R5)}
    slice_sensitivity_v4r5 = {
        name: _median_or_none(sensitivity_v4r5[..., component_index[name]][slice_mask])
        for name in COMPONENT_NAMES_V4R5
    }
    slice_sensitivity_v4r6 = {
        name: _median_or_none(sensitivity_v4r6[..., component_index[name]][slice_mask])
        for name in COMPONENT_NAMES_V4R5
    }
    hminus_sensitivity_median = float(
        (slice_sensitivity_v4r5["hminus_freefree"] or 0.0)
        + (slice_sensitivity_v4r5["hminus_boundfree"] or 0.0)
    )
    implied_base_fraction_median = float(np.median(implied_base_fraction[slice_mask]))

    slice_metrics = {
        "ifop2_h2plus_knockout_effect": _metrics(knockout_effects[1][slice_mask]),
        "ifop7_heminus_knockout_effect": _metrics(knockout_effects[6][slice_mask]),
        "candidate_v4r5_h2plus_subset_effect": _metrics(
            candidate_h2plus_effect[slice_mask]
        ),
        "candidate_v4r5_heminus_subset_effect": _metrics(
            candidate_heminus_effect[slice_mask]
        ),
        "candidate_v4r5_h2plus_heminus_subset_effect": _metrics(
            candidate_h2plus_heminus_effect[slice_mask]
        ),
        "v4r5_minus_production_continuum": _metrics(v4r5_minus_production[slice_mask]),
        "v4r6_minus_production_continuum_control": _metrics(
            v4r6_minus_production[slice_mask]
        ),
        "candidate_base_minus_implied_production_base": _metrics(base_gap[slice_mask]),
    }

    sanity_failures: list[str] = []
    if not indices_match:
        sanity_failures.append("reference_indices_drift_from_stored_ablation")
    if temperature_drift_max_K > TEMPERATURE_DRIFT_TOLERANCE_K:
        sanity_failures.append("corpus_temperature_drift")
    if cool_star_count != EXPECTED_COOL_STAR_COUNT:
        sanity_failures.append(f"cool_star_count_{cool_star_count}")
    if slice_layer_count != EXPECTED_SLICE_LAYER_COUNT:
        sanity_failures.append(f"slice_layer_count_{slice_layer_count}")
    if baseline_reproduce_median_abs_dex >= BASELINE_REPRODUCE_MEDIAN_ABS_MAX_DEX:
        sanity_failures.append("baseline_replay_does_not_reproduce_stored_baseline")

    decision = _decide(
        sanity_failures=sanity_failures,
        ifop2_signed_median_dex=slice_metrics["ifop2_h2plus_knockout_effect"][
            "signed_median_dex"
        ],
        ifop7_signed_median_dex=slice_metrics["ifop7_heminus_knockout_effect"][
            "signed_median_dex"
        ],
        implied_base_fraction_median=implied_base_fraction_median,
        candidate_hminus_sensitivity_median=hminus_sensitivity_median,
        base_gap_signed_median_dex=slice_metrics[
            "candidate_base_minus_implied_production_base"
        ]["signed_median_dex"],
    )

    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "textbook_opacity_v4r6_cool_continuum_attribution",
        "status": "diagnostic_only",
        "question": (
            "which production continuum component carries the -0.0668 dex "
            "signed-median gap (v4r5 vs production continuum, lines off) on "
            "the 20-star cool 3200-4000 K slice"
        ),
        "corpus": str(corpus.path),
        "corpus_sha256": file_sha256(corpus.path),
        "hot_flag_ablation_result": str(ablation_path),
        "hot_flag_ablation_sha256": file_sha256(ablation_path),
        "cited_artifacts": {
            "hot_flag_ablation_sha256": HOT_FLAG_ABLATION_SHA256,
            "cool_mass_decomposition_sha256": COOL_MASS_DECOMPOSITION_SHA256,
        },
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
            "new_knockouts": {
                str(flag): name for flag, name in NEW_KNOCKOUTS.items()
            },
            "flag_index_convention": (
                "0-based Python index into the 20-element OPACITY IFOP "
                "vector; index 1 = IFOP(2) H2+, index 6 = IFOP(7) He-minus"
            ),
            "production_frequency_count_min": int(np.min(frequency_counts)),
            "production_frequency_count_max": int(np.max(frequency_counts)),
        },
        "slice_of_record": {
            "cool_teff_max_K": COOL_TEFF_MAX_K,
            "layer_temperature_lower_K": SLICE_LOWER_K,
            "layer_temperature_upper_K": SLICE_UPPER_K,
            "cool_star_count": cool_star_count,
            "expected_cool_star_count": EXPECTED_COOL_STAR_COUNT,
            "slice_layer_count": slice_layer_count,
            "expected_slice_layer_count": EXPECTED_SLICE_LAYER_COUNT,
            "reference_indices_stored": stored_indices,
            "reference_indices_recomputed": recomputed_indices,
            "reference_indices_match": bool(indices_match),
            "temperature_drift_max_K": temperature_drift_max_K,
        },
        "baseline_reproduction": {
            "median_abs_diff_dex": baseline_reproduce_median_abs_dex,
            "max_abs_diff_dex": baseline_reproduce_max_abs_dex,
            "tolerance_median_abs_dex": BASELINE_REPRODUCE_MEDIAN_ABS_MAX_DEX,
            "pass": bool(
                baseline_reproduce_median_abs_dex
                < BASELINE_REPRODUCE_MEDIAN_ABS_MAX_DEX
            ),
        },
        "slice_metrics": slice_metrics,
        "candidate_component_log_sensitivity_median_v4r5": slice_sensitivity_v4r5,
        "candidate_component_log_sensitivity_median_v4r6_control": (
            slice_sensitivity_v4r6
        ),
        "base_bookkeeping": {
            "implied_base_fraction_median": implied_base_fraction_median,
            "implied_base_definition": (
                "kappa_all_on minus sum over the 8 knocked-out flags (6 "
                "stored + IFOP(2) + IFOP(7)) of (kappa_all_on - kappa_flag_off); "
                "linear Rosseland bookkeeping"
            ),
            "candidate_base_definition": (
                "v4r5 subset Rosseland without h2plus and heminus"
            ),
            "candidate_hminus_freefree_plus_boundfree_sensitivity_median": (
                hminus_sensitivity_median
            ),
            "candidate_electron_scattering_sensitivity_median": (
                slice_sensitivity_v4r5["electron_scattering"]
            ),
            "candidate_hydrogen_rayleigh_sensitivity_median": (
                slice_sensitivity_v4r5["hydrogen_rayleigh_scattering"]
            ),
        },
        "decision": decision,
        "references": reference_rows,
        "scope_boundary": {
            "production_solver_changed": False,
            "textbook_opacity_py_edited": False,
            "candidate_changed": False,
            "gates_changed": False,
            "new_opacity_version_implemented": False,
            "ode_run": False,
            "smoke_run": False,
            "funnel_run": False,
            "sealed_holdout_opened": False,
            "production_opacity_evaluated_on_macos": False,
        },
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
                    "decision": decision,
                    "slice_metrics": slice_metrics,
                    "baseline_reproduction": result["baseline_reproduction"],
                    "slice_of_record": result["slice_of_record"],
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
    parser.add_argument("--hot-flag-ablation", type=Path, default=HOT_FLAG_ABLATION)
    parser.add_argument("--out", type=Path, default=REGISTERED_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=20260816)
    parser.add_argument("--stride", type=int, default=16)
    args = parser.parse_args()
    run_attribution(
        corpus_path=args.corpus,
        ablation_path=args.hot_flag_ablation,
        output_path=args.out,
        split_seed=args.split_seed,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
