"""Preregistered v4r6 middle-band mass slice by layer temperature.

This runner does not change v4r6, the mass integral, or any gate.  It only
reproduces the middle-band true-(P, T) mass failure and slices the
wholly-in-domain increment residual by layer temperature.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Mapping

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
)
from experiments.analytic_initializer.run_textbook_opacity_v4r5_cool_mass_decomposition import (
    _jsonify,
    _metrics_or_none,
    _p95,
    blend_log_opacity_by_temperature,
    local_increment_residual,
    log_mass_residual,
    wholly_in_domain_increment_mask,
)
from experiments.analytic_initializer.textbook_opacity import (
    DEFAULT_TEXTBOOK_CONSTANTS,
    V4R6_FORMAL_TEMPERATURE_FLOOR_K,
    textbook_opacity_node_components_v4r6,
)


TEMPERATURE_FLOOR_K = V4R6_FORMAL_TEMPERATURE_FLOOR_K
MASS_GATE_LIMIT_DEX = 0.20
EXPECTED_V4R6_MIDDLE_MASS_P95_DEX = 0.20741398414881243
EXPECTED_MIDDLE_INCREMENT_P95_DEX = 0.22927611663282257
REPRODUCE_SURFACE_TOLERANCE_DEX = 0.002
REPRODUCE_INCREMENT_TOLERANCE_DEX = 0.01
SANITY_TRUTH_KAPPA_P95_MAX_DEX = 0.05
CARRIER_SHARE_MIN = 0.70
EXPECTED_VALIDATION_STAR_COUNT = 10228
SPLIT_SEED = 20260816
LAYER_TEMPERATURE_BINS = (
    ("4000_6000K", 4000.0, 6000.0),
    ("6000_8000K", 6000.0, 8000.0),
    ("8000_10000K", 8000.0, 10000.0),
    ("10000_15000K", 10000.0, 15000.0),
    ("15000_22000K", 15000.0, 22000.0),
    ("22000_30000K", 22000.0, 30000.0),
    ("at_least_30000K", 30000.0, np.inf),
)
LINE_FLOOR_BIN_NAMES = ("8000_10000K", "10000_15000K")
DEEP_HOT_BIN_NAMES = ("15000_22000K", "22000_30000K")
V4R6_OFFLINE = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r6_offline_validation_20260828.json"
)
COOL_MASS_DECOMPOSITION = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r5_cool_mass_decomposition_20260828.json"
)
REGISTERED_OUTPUT = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r6_midmass_slice_20260828.json"
)


def _batch_prediction(
    labels: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    prediction = np.empty_like(temperature, dtype=np.float64)
    for start in range(0, labels.shape[0], int(batch_size)):
        stop = min(labels.shape[0], start + int(batch_size))
        components = textbook_opacity_node_components_v4r6(
            labels[start:stop],
            temperature[start:stop],
            pressure[start:stop],
        )
        weights = components["node_weights"]
        total = np.maximum(components["total"], 1.0e-30)
        prediction[start:stop] = 1.0 / np.sum(weights / total, axis=(-2, -1))
        print(f"processed {stop}/{labels.shape[0]} stars", flush=True)
    return prediction


def _bin_stats(
    residual: np.ndarray,
    star_mask: np.ndarray,
    bin_layer_mask: np.ndarray,
    extra_mask: np.ndarray | None = None,
) -> dict[str, dict[str, float] | None]:
    rows: dict[str, dict[str, float] | None] = {}
    for name, lower, upper in LAYER_TEMPERATURE_BINS:
        mask = (
            star_mask[:, None]
            & (bin_layer_mask >= lower)
            & (bin_layer_mask < upper)
        )
        if extra_mask is not None:
            mask &= extra_mask
        rows[name] = _metrics_or_none(residual[mask])
    return rows


def _carrier_shares(
    bin_rows: Mapping[str, Mapping[str, float] | None],
    *,
    mass_limit_dex: float = MASS_GATE_LIMIT_DEX,
) -> dict[str, object]:
    """Over-limit excess per bin, weighted by layer fraction, and the best
    contiguous bin interval."""

    names = [name for name, _, _ in LAYER_TEMPERATURE_BINS]
    count = np.zeros(len(names))
    excess = np.zeros(len(names))
    for index, name in enumerate(names):
        row = bin_rows.get(name)
        if row is None:
            continue
        count[index] = float(row["count"])
        excess[index] = max(float(row["p95_dex"]) - float(mass_limit_dex), 0.0)
    total_count = float(np.sum(count))
    contribution = excess * (count / total_count) if total_count > 0.0 else excess * 0.0
    total_contribution = float(np.sum(contribution))
    share = (
        contribution / total_contribution
        if total_contribution > 0.0
        else np.zeros_like(contribution)
    )
    best_interval = None
    best_share = 0.0
    for lo in range(len(names)):
        for hi in range(lo, len(names)):
            interval_share = float(np.sum(share[lo : hi + 1]))
            if interval_share > best_share:
                best_share = interval_share
                best_interval = (lo, hi)
    return {
        "bin_names": names,
        "increment_count_total": int(total_count),
        "excess_over_limit_dex": {
            name: float(excess[i]) for i, name in enumerate(names)
        },
        "layer_fraction": {
            name: (float(count[i] / total_count) if total_count > 0.0 else 0.0)
            for i, name in enumerate(names)
        },
        "excess_share": {name: float(share[i]) for i, name in enumerate(names)},
        "best_contiguous_interval": (
            None
            if best_interval is None
            else [names[best_interval[0]], names[best_interval[1]]]
        ),
        "best_contiguous_interval_share": float(best_share),
    }


def decide_middle_mass_slice(
    *,
    surface_p95_dex: float,
    hybrid_p95_dex: float,
    in_domain_increment_p95_dex: float,
    truth_kappa_p95_dex: float,
    carrier: Mapping[str, object],
    expected_surface_p95_dex: float | None = EXPECTED_V4R6_MIDDLE_MASS_P95_DEX,
    expected_increment_p95_dex: float | None = EXPECTED_MIDDLE_INCREMENT_P95_DEX,
    reproduce_surface_tolerance_dex: float = REPRODUCE_SURFACE_TOLERANCE_DEX,
    reproduce_increment_tolerance_dex: float = REPRODUCE_INCREMENT_TOLERANCE_DEX,
    mass_limit_dex: float = MASS_GATE_LIMIT_DEX,
    sanity_truth_kappa_p95_max_dex: float = SANITY_TRUTH_KAPPA_P95_MAX_DEX,
    carrier_share_min: float = CARRIER_SHARE_MIN,
) -> dict[str, object]:
    """Return the preregistered middle-mass slice verdict."""

    values = {
        "surface_p95_dex": float(surface_p95_dex),
        "hybrid_p95_dex": float(hybrid_p95_dex),
        "in_domain_increment_p95_dex": float(in_domain_increment_p95_dex),
        "truth_kappa_p95_dex": float(truth_kappa_p95_dex),
    }
    finite = all(np.isfinite(value) for value in values.values())
    truth_sanity_pass = bool(
        values["truth_kappa_p95_dex"] <= sanity_truth_kappa_p95_max_dex
    )
    reproduced_surface = True
    if expected_surface_p95_dex is not None:
        reproduced_surface = bool(
            abs(values["surface_p95_dex"] - float(expected_surface_p95_dex))
            <= float(reproduce_surface_tolerance_dex)
        )
    reproduced_increment = True
    if expected_increment_p95_dex is not None:
        reproduced_increment = bool(
            abs(
                values["in_domain_increment_p95_dex"]
                - float(expected_increment_p95_dex)
            )
            <= float(reproduce_increment_tolerance_dex)
        )
    gate_still_fails = bool(values["surface_p95_dex"] > mass_limit_dex)
    increments_nonempty = int(carrier["increment_count_total"]) > 0
    best_interval = carrier["best_contiguous_interval"]
    best_share = float(carrier["best_contiguous_interval_share"])
    carrier_names = (
        None if best_interval is None else list(best_interval)
    )
    if carrier_names is not None:
        lo_name, hi_name = carrier_names
        names = [name for name, _, _ in LAYER_TEMPERATURE_BINS]
        span = names[names.index(lo_name) : names.index(hi_name) + 1]
    else:
        span = []
    interval_in_line_floor = bool(span) and all(
        name in LINE_FLOOR_BIN_NAMES for name in span
    )
    interval_in_deep_hot = bool(span) and all(
        name in DEEP_HOT_BIN_NAMES for name in span
    )
    hybrid_pass = bool(values["hybrid_p95_dex"] <= mass_limit_dex)
    increment_pass = bool(values["in_domain_increment_p95_dex"] <= mass_limit_dex)
    inconclusive_reason = None
    if not finite:
        verdict = "INCONCLUSIVE"
        inconclusive_reason = "non_finite_primary_p95"
    elif not truth_sanity_pass:
        verdict = "INCONCLUSIVE"
        inconclusive_reason = "stored_kappa_integral_does_not_recover_mass"
    elif not reproduced_surface:
        verdict = "INCONCLUSIVE"
        inconclusive_reason = "did_not_reproduce_v4r6_middle_mass_p95"
    elif not reproduced_increment:
        verdict = "INCONCLUSIVE"
        inconclusive_reason = "did_not_reproduce_middle_increment_p95"
    elif not gate_still_fails:
        verdict = "INCONCLUSIVE"
        inconclusive_reason = "middle_mass_gate_already_passes"
    elif not increments_nonempty:
        verdict = "INCONCLUSIVE"
        inconclusive_reason = "no_wholly_in_domain_increments"
    elif hybrid_pass and increment_pass:
        verdict = "SURFACE_COLUMN"
    elif best_share >= float(carrier_share_min) and interval_in_line_floor:
        verdict = "LINE_FLOOR_8_15KK"
    elif best_share >= float(carrier_share_min) and interval_in_deep_hot:
        verdict = "DEEP_HOT_15_30KK"
    else:
        verdict = "MIXED"
    return {
        "verdict": verdict,
        "inconclusive_reason": inconclusive_reason,
        "hybrid_pass": hybrid_pass,
        "increment_pass": increment_pass,
        "truth_kappa_sanity_pass": truth_sanity_pass,
        "reproduced_v4r6_middle_mass_p95": reproduced_surface,
        "reproduced_middle_increment_p95": reproduced_increment,
        "gate_still_fails": gate_still_fails,
        "mass_limit_dex": float(mass_limit_dex),
        "carrier_share_min": float(carrier_share_min),
        "best_contiguous_interval": carrier_names,
        "best_contiguous_interval_share": best_share,
        "best_interval_in_line_floor_bins": interval_in_line_floor,
        "best_interval_in_deep_hot_bins": interval_in_deep_hot,
        **values,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--v4r6-offline", type=Path, default=V4R6_OFFLINE)
    parser.add_argument(
        "--cool-mass-decomposition", type=Path, default=COOL_MASS_DECOMPOSITION
    )
    parser.add_argument("--out", type=Path, default=REGISTERED_OUTPUT)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    for path in (args.v4r6_offline, args.cool_mass_decomposition):
        if not path.is_file():
            raise SystemExit(f"required prior artifact is missing: {path}")

    corpus = load_strict_truth(args.corpus)
    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=SPLIT_SEED)
    indices = split.validation
    if args.limit is not None:
        indices = indices[: int(args.limit)]
    labels = corpus.labels[indices]
    temperature = corpus.temperature[indices]
    pressure = corpus.gas_pressure[indices]
    stored_opacity = corpus.rosseland_opacity[indices]
    stored_mass = corpus.column_mass[indices]
    print(
        f"evaluating v4r6 on {int(indices.size)} validation stars",
        flush=True,
    )
    prediction = _batch_prediction(
        labels,
        temperature,
        pressure,
        batch_size=args.batch_size,
    )
    log_prediction = np.log10(prediction)
    log_stored = np.log10(stored_opacity)
    surface_mass = integrate_mass_from_opacity(corpus.tau, log_prediction)
    truth_kappa_mass = integrate_mass_from_opacity(corpus.tau, log_stored)
    hybrid_mass = integrate_mass_from_opacity(
        corpus.tau,
        blend_log_opacity_by_temperature(
            log_prediction,
            log_stored,
            temperature,
            TEMPERATURE_FLOOR_K,
        ),
    )
    increment = local_increment_residual(surface_mass, stored_mass)
    mass_residual = log_mass_residual(surface_mass, stored_mass)

    middle_stars = _band_masks(labels[:, 0])["middle_6000_10000K"]
    middle_gate = middle_stars[:, None] & (temperature >= TEMPERATURE_FLOOR_K)
    in_domain_increments = wholly_in_domain_increment_mask(
        temperature, TEMPERATURE_FLOOR_K
    )
    middle_increment_mask = middle_stars[:, None] & in_domain_increments

    surface_metrics = _metrics_or_none(mass_residual[middle_gate])
    hybrid_metrics = _metrics_or_none(
        log_mass_residual(hybrid_mass, stored_mass)[middle_gate]
    )
    increment_metrics = _metrics_or_none(increment[middle_increment_mask])
    truth_kappa_metrics = _metrics_or_none(
        log_mass_residual(truth_kappa_mass, stored_mass)[middle_gate]
    )

    increment_by_bin = _bin_stats(
        increment,
        middle_stars,
        temperature,
        extra_mask=in_domain_increments,
    )
    cumulative_by_bin = _bin_stats(mass_residual, middle_stars, temperature)
    carrier = _carrier_shares(increment_by_bin)

    full_split = (
        args.limit is None and int(indices.size) == EXPECTED_VALIDATION_STAR_COUNT
    )
    decision = decide_middle_mass_slice(
        surface_p95_dex=_p95(surface_metrics),
        hybrid_p95_dex=_p95(hybrid_metrics),
        in_domain_increment_p95_dex=_p95(increment_metrics),
        truth_kappa_p95_dex=_p95(truth_kappa_metrics),
        carrier=carrier,
        expected_surface_p95_dex=(
            EXPECTED_V4R6_MIDDLE_MASS_P95_DEX if full_split else None
        ),
        expected_increment_p95_dex=(
            EXPECTED_MIDDLE_INCREMENT_P95_DEX if full_split else None
        ),
    )

    result = {
        "schema_version": 1,
        "candidate": "v4r6_midmass_layer_temperature_slice",
        "version": "v4r6",
        "decision": decision["verdict"],
        "slice_decision": decision,
        "carrier_analysis": carrier,
        "increment_bin_assignment": "upper_endpoint_layer_temperature",
        "v4r6_offline_result": str(args.v4r6_offline),
        "v4r6_offline_result_sha256": file_sha256(args.v4r6_offline),
        "cool_mass_decomposition_result": str(args.cool_mass_decomposition),
        "cool_mass_decomposition_result_sha256": file_sha256(
            args.cool_mass_decomposition
        ),
        "corpus": str(corpus.path),
        "corpus_sha256": file_sha256(corpus.path),
        "validation_star_count": int(indices.size),
        "validation_layer_count": int(indices.size * corpus.layers),
        "full_registered_split": full_split,
        "split_seed": split.seed,
        "excluded_count": int(excluded.size),
        "excluded_manifests": used_manifests,
        "constants": asdict(DEFAULT_TEXTBOOK_CONSTANTS),
        "middle_star_count": int(np.sum(middle_stars)),
        "middle_gate_layer_count": int(np.sum(middle_gate)),
        "middle_in_domain_increment_count": int(np.sum(middle_increment_mask)),
        "nonfinite_count": int(
            np.sum(~np.isfinite(prediction)) + np.sum(~np.isfinite(surface_mass))
        ),
        "middle_gate_mass": {
            "surface_started_v4r6": surface_metrics,
            "hybrid_stored_kappa_below_4000K": hybrid_metrics,
            "wholly_in_domain_increment": increment_metrics,
            "stored_kappa_integral_sanity": truth_kappa_metrics,
        },
        "increment_residual_by_layer_temperature_bin": increment_by_bin,
        "cumulative_mass_residual_by_layer_temperature_bin": cumulative_by_bin,
        "scope_boundary": {
            "production_solver_changed": False,
            "textbook_opacity_py_edited": False,
            "mass_integral_start_changed": False,
            "gates_changed": False,
            "new_opacity_version_implemented": False,
            "ode_run": False,
            "funnel_run": False,
            "sealed_holdout_opened": False,
            "production_opacity_called": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(_jsonify(result), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": decision["verdict"],
                "surface_p95_dex": decision["surface_p95_dex"],
                "hybrid_p95_dex": decision["hybrid_p95_dex"],
                "in_domain_increment_p95_dex": decision[
                    "in_domain_increment_p95_dex"
                ],
                "truth_kappa_p95_dex": decision["truth_kappa_p95_dex"],
                "best_contiguous_interval": decision["best_contiguous_interval"],
                "best_contiguous_interval_share": decision[
                    "best_contiguous_interval_share"
                ],
            }
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
