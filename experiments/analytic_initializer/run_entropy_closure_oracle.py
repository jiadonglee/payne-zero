"""Pre-registered Gate-0 oracle for the no-emulator entropy-closure family.

The plan in ``notes/initializer_improvement_plan_20260812.md`` fixes, before any
fit, that the per-star oracle must optimize the closure's free parameters on
held-out stars and that a 7000-8000 K deep p95 above 0.015 dex vetoes the whole
family.  This runner measures that ceiling exactly: truth colour mass, opacity
and pressure, plus the H2 radiative branch, so the only approximation left is
the convective gradient family itself.  If the family cannot reach 0.015 dex
even with truth physics and a per-star optimizer, no label-conditioned fit of
it can.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analytic_initializer.deep_diagnostics import (
    CONVECTIVE_FLUX_DEFICIT,
    deep_window,
)
from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.profile_initializer import (
    load_analytic_profile_parameters,
    predict_analytic_reduced_state,
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

H2_PARAMETERS = Path("results/analytic_initializer/h2_profile_parameters_v1.npz")


def log10_relu(x: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(x, 1.0e-300))


def load_star_bundle(corpus, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Truth physics plus the H2 radiative branch for the closure oracle."""

    parameters = load_analytic_profile_parameters(H2_PARAMETERS)
    _, radiative_temperature, _ = predict_analytic_reduced_state(
        corpus.labels[indices], corpus.tau, parameters
    )
    return {
        "pressure": corpus.gas_pressure[indices],
        "truth_temperature": corpus.temperature[indices],
        "opacity": corpus.rosseland_opacity[indices],
        "radiative_temperature": np.maximum(radiative_temperature, 1.0e-12),
        "effective_temperature": corpus.labels[indices, 0],
        "gravity": 10.0 ** corpus.labels[indices, 1],
    }


def radiative_gradient(pressure, radiative_temperature) -> np.ndarray:
    return np.gradient(np.log(radiative_temperature), axis=1) / np.gradient(
        np.log(pressure), axis=1
    )


def make_predictor(bundle, start, stop):
    pressure = bundle["pressure"]
    logP = np.log10(pressure)
    dlnP = np.gradient(np.log(pressure), axis=1)
    grad_rad = radiative_gradient(pressure, bundle["radiative_temperature"])
    truth_ln = np.log(bundle["truth_temperature"])

    def _log_profile(v, row):
        # v = (grad_ad_slope_c0, grad_ad_logP_slope_c1, entropy_jump_as,
        #      onset_layer, switch_width_dex, logT_offset)
        c0, c1, a_s, onset_l, width, offset = v
        onset_l = int(round(max(0.0, min(pressure.shape[1] - 2, onset_l))))
        width = max(abs(width), 1.0e-4)
        lp0 = logP[row, onset_l]
        switch = 1.0 / (1.0 + np.exp(-(logP[row] - lp0) / width))
        grad_ad = c0 + c1 * (logP[row] - lp0)
        grad_eff = grad_rad[row] * (1.0 - switch) + grad_ad * switch + a_s * switch * (
            1.0 - switch
        )
        logT = np.zeros_like(logP[row])
        logT[0] = np.log(bundle["radiative_temperature"][row, 0])
        logT[1:] = logT[0] + np.cumsum(grad_eff[:-1] * dlnP[row][:-1])
        return logT + offset

    def deep_error(v, row):
        profile = _log_profile(v, row)
        return float(np.abs(profile[start:stop] - truth_ln[row, start:stop]).max())

    return deep_error


BOUNDS = (
    (0.05, 0.45),  # c0: adiabatic gradient level
    (-0.50, 0.50),  # c1: logP slope of adiabatic gradient
    (-0.30, 0.50),  # a_s: entropy-jump bump amplitude
    (10.0, 75.0),  # onset layer
    (0.02, 0.60),  # switch width in dex
    (-0.05, 0.05),  # logT offset
)


def run_oracle(corpus, validation, count, seed) -> dict:
    teff = corpus.labels[validation, 0]
    mask = (teff >= 7000.0) & (teff < 8000.0)
    candidates = validation[mask]
    rng = np.random.default_rng(seed)
    chosen = candidates[rng.permutation(candidates.size)[:count]]

    bundle = load_star_bundle(corpus, chosen)
    start, stop = deep_window(corpus.layers)
    deep_error = make_predictor(bundle, start, stop)

    rows = []
    for row, index in enumerate(chosen):
        result = differential_evolution(
            lambda v: deep_error(v, row),
            BOUNDS,
            seed=11 + row,
            maxiter=400,
            popsize=6,
            tol=1.0e-8,
            polish=True,
            workers=1,
        )
        rows.append(
            {
                "corpus_index": int(index),
                "effective_temperature": float(bundle["effective_temperature"][row]),
                "best_deep_error_dex": float(result.fun),
                "c0": float(result.x[0]),
                "c1": float(result.x[1]),
                "a_s": float(result.x[2]),
                "onset_layer": float(result.x[3]),
                "switch_width_dex": float(result.x[4]),
                "logT_offset": float(result.x[5]),
                "solver_outcome": "oracle_per_star",
            }
        )
        if (row + 1) % 15 == 0:
            print(f"oracle row {row + 1}/{chosen.size}", flush=True)

    errors = np.asarray([row["best_deep_error_dex"] for row in rows])
    teffs = np.asarray([row["effective_temperature"] for row in rows])
    summary = {
        "n": int(errors.size),
        "deep_dex_p50": float(np.median(errors)),
        "deep_dex_p95": float(np.percentile(errors, 95.0)),
        "deep_dex_max": float(np.max(errors)),
    }
    for low, high in ((7000.0, 7500.0), (7500.0, 8000.0)):
        band = (teffs >= low) & (teffs < high)
        summary[f"{int(low)}-{int(high)}"] = {
            "n": int(band.sum()),
            "deep_dex_p50": float(np.median(errors[band])),
            "deep_dex_p95": float(np.percentile(errors[band], 95.0)),
        }
    return summary, rows


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-count", type=int, default=45)
    parser.add_argument("--oracle-seed", type=int, default=7)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/analytic_initializer/entropy_closure_oracle.json"),
    )
    args = parser.parse_args(argv)

    corpus = load_strict_truth(DEFAULT_CORPUS)
    excluded, used = collect_excluded_indices(MANIFESTS, corpus_size=corpus.size)
    split = make_split(corpus.size, excluded=excluded, seed=20260816)

    summary, rows = run_oracle(
        corpus, split.validation, args.oracle_count, args.oracle_seed
    )

    veto_threshold = 0.015
    pass_band = summary["7000-7500"]["deep_dex_p95"] <= veto_threshold and (
        summary["7500-8000"]["deep_dex_p95"] <= veto_threshold
    )

    artifact = {
        "format": "payne_zero_entropy_closure_oracle_v1",
        "corpus_path": str(DEFAULT_CORPUS),
        "corpus_sha256": file_sha256(DEFAULT_CORPUS),
        "split_seed": int(split.seed),
        "excluded_manifests_used": [str(path) for path in used],
        "excluded_star_count": int(excluded.size),
        "validation_star_count": int(split.validation.size),
        "h2_parameters_used": str(H2_PARAMETERS),
        "truth_physics_used": [
            "column_mass_fixed_to_truth",
            "opacity_fixed_to_truth",
            "pressure_fixed_to_truth",
        ],
        "closure_family": (
            "logistic_switch_mixing_rad_and_ad_plus_entropy_bump; "
            "grad_ad = c0 + c1*logP; free onset,width,offset"
        ),
        "free_parameters_per_star": 6,
        "pre_registered_veto_threshold_dex": veto_threshold,
        "gate_1_deep_p95_target_dex": 0.020,
        "offline_best_case": summary,
        "status": ("family_vetoed_by_pre_registered_oracle" if not pass_band else "passes"),
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"status: {artifact['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
