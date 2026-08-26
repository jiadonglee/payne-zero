"""Pre-registered Gate-0 oracle for the v2 dual-crossing entropy closure.

Plan: ``notes/analytic_initializer_repair_plan_v2.md``.  Two oracles on
held-out 7000--8000 K stars with truth mass/opacity/pressure/radiative
branch fixed, so only the convective-gradient family is tested:

(a) *representation oracle*: per-star optimize gamma_ad, A_enter, A_exit and
    BOTH crossing locations.  Failure to reach 0.015 dex deep p95 in both
    7000--7500 and 7500--8000 vetoes the family outright.

(b) *physics-trigger oracle*: crossings are located by the model's own
    Schwarzschild scan on truth physics, optimizing only gamma_ad,
    A_enter, A_exit.  Its failure activates the exit-logP alternative.

If (a) passes and (b) fails, representation is not the limit: the family
cannot self-locate the boundaries, and the pre-registered response is the
587-constant exit-logP variant.  Retuning after the fact is forbidden.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analytic_initializer.deep_diagnostics import deep_window
from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.entropy_closure_v2 import (
    SWITCH_WIDTH_DEX,
    schwarzschild_crossings,
    dual_crossing_gradient,
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

VETO_THRESHOLD_DEX = 0.015


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


def _ln_truth(bundle, row):
    return np.log(bundle["truth_temperature"][row])


def _log_p(bundle, row):
    return np.log10(np.maximum(bundle["pressure"][row], 1.0e-300))


def _grad_rad(bundle, row):
    # d ln T / d ln P, matching the v1 oracle convention (not log10).
    d_ln_p = np.gradient(_log_p(bundle, row)) * np.log(10.0)
    return np.gradient(np.log(bundle["radiative_temperature"][row])) / d_ln_p


def _dln_p(log_p):
    return np.gradient(log_p) * np.log(10.0)


def _integrate(log_p, gradient, ln_surface):
    ln_t = np.empty_like(gradient)
    ln_t[0] = float(ln_surface)
    dln_p = _dln_p(log_p)
    ln_t[1:] = ln_t[0] + np.cumsum(gradient[:-1] * dln_p[:-1])
    return ln_t


def make_representation_predictor(bundle, row, start, stop):
    """Optimize gamma_ad, A_enter, A_exit and both crossing layer locations."""

    log_p = _log_p(bundle, row)
    grad_rad = _grad_rad(bundle, row)
    truth_ln = _ln_truth(bundle, row)
    n_layers = int(log_p.size)
    ln_surface = float(np.log(bundle["radiative_temperature"][row, 0]))

    def _predict(v):
        gamma_ad, a_enter, a_exit, enter_layer, exit_layer = v
        enter_layer = int(round(max(0.0, min(n_layers - 2, enter_layer))))
        exit_layer = int(round(max(enter_layer + 1.0, min(n_layers, exit_layer))))
        lp_enter = float(log_p[enter_layer])
        lp_exit = float(log_p[exit_layer]) if exit_layer < n_layers else float(log_p[-1]) + 3.0
        gradient, _, _ = dual_crossing_gradient(
            log_p, lp_enter, lp_exit, grad_rad,
            np.full_like(log_p, gamma_ad), a_enter, a_exit,
            width_dex=SWITCH_WIDTH_DEX,
        )
        return _integrate(log_p, gradient, ln_surface)

    def _error(v):
        profile = _predict(v)
        return float(np.abs(profile[start:stop] - truth_ln[start:stop]).max())

    return _error


def make_trigger_predictor(bundle, row, start, stop):
    """Physics trigger: crossings from the model's own Schwarzschild scan."""

    log_p = _log_p(bundle, row)
    grad_rad = _grad_rad(bundle, row)
    truth_ln = _ln_truth(bundle, row)
    n_layers = int(log_p.size)
    ln_surface = float(np.log(bundle["radiative_temperature"][row, 0]))

    def _cross(gamma_ad):
        return schwarzschild_crossings(
            log_p, grad_rad, np.full_like(log_p, gamma_ad)
        )

    def _error(v):
        gamma_ad, a_enter, a_exit = v
        enter_layer, exit_layer = _cross(gamma_ad)
        if enter_layer < 0:
            gradient = grad_rad
        else:
            lp_enter = float(log_p[enter_layer])
            lp_exit = (
                float(log_p[exit_layer])
                if exit_layer < n_layers
                else float(log_p[-1]) + 3.0
            )
            gradient, _, _ = dual_crossing_gradient(
                log_p, lp_enter, lp_exit, grad_rad,
                np.full_like(log_p, gamma_ad), a_enter, a_exit,
                width_dex=SWITCH_WIDTH_DEX,
            )
        profile = _integrate(log_p, gradient, ln_surface)
        return float(np.abs(profile[start:stop] - truth_ln[start:stop]).max())

    return _error


REPRESENTATION_BOUNDS = (
    (0.10, 0.45),    # gamma_ad
    (0.00, 0.50),    # a_enter
    (-0.50, 0.50),   # a_exit
    (10.0, 75.0),    # enter layer
    (15.0, 80.0),    # exit layer
)

TRIGGER_BOUNDS = (
    (0.10, 0.45),   # gamma_ad
    (0.00, 0.50),   # a_enter
    (-0.50, 0.50),  # a_exit
)


def _optimize(fun, bounds, row):
    return differential_evolution(
        fun, bounds, seed=101 + row, maxiter=400, popsize=6,
        tol=1.0e-8, polish=True, workers=1,
    )


def run_oracle(corpus, validation, count, seed) -> dict:
    teff = corpus.labels[validation, 0]
    mask = (teff >= 7000.0) & (teff < 8000.0)
    candidates = validation[mask]
    rng = np.random.default_rng(seed)
    chosen = candidates[rng.permutation(candidates.size)[:count]]
    bundle = load_star_bundle(corpus, chosen)
    start, stop = deep_window(corpus.layers)

    rows = []
    for row, index in enumerate(chosen):
        repr_fun = make_representation_predictor(bundle, row, start, stop)
        phys_fun = make_trigger_predictor(bundle, row, start, stop)
        repr_res = _optimize(repr_fun, REPRESENTATION_BOUNDS, row)
        phys_res = _optimize(phys_fun, TRIGGER_BOUNDS, row)
        rows.append(
            {
                "corpus_index": int(index),
                "effective_temperature": float(bundle["effective_temperature"][row]),
                "representation_deep_error_dex": float(repr_res.fun),
                "representation_gamma_ad": float(repr_res.x[0]),
                "representation_a_enter": float(repr_res.x[1]),
                "representation_a_exit": float(repr_res.x[2]),
                "representation_enter_layer": float(repr_res.x[3]),
                "representation_exit_layer": float(repr_res.x[4]),
                "physics_deep_error_dex": float(phys_res.fun),
                "physics_gamma_ad": float(phys_res.x[0]),
                "physics_a_enter": float(phys_res.x[1]),
                "physics_a_exit": float(phys_res.x[2]),
                "solver_outcome": "oracle_per_star",
            }
        )
        if (row + 1) % 15 == 0:
            print(f"oracle row {row + 1}/{chosen.size}", flush=True)

    return chosen, bundle, rows



def summarize(errors, teffs) -> dict:
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
    return summary


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-count", type=int, default=45)
    parser.add_argument("--oracle-seed", type=int, default=7)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/analytic_initializer/entropy_closure_v2_oracle.json"),
    )
    args = parser.parse_args(argv)

    corpus = load_strict_truth(DEFAULT_CORPUS)
    excluded, used = collect_excluded_indices(MANIFESTS, corpus_size=corpus.size)
    split = make_split(corpus.size, excluded=excluded, seed=20260816)

    chosen, bundle, rows = run_oracle(
        corpus, split.validation, args.oracle_count, args.oracle_seed
    )
    teffs = np.asarray([row["effective_temperature"] for row in rows])
    repr_errors = np.asarray([row["representation_deep_error_dex"] for row in rows])
    phys_errors = np.asarray([row["physics_deep_error_dex"] for row in rows])

    repr_summary = summarize(repr_errors, teffs)
    phys_summary = summarize(phys_errors, teffs)

    repr_pass = (
        repr_summary["7000-7500"]["deep_dex_p95"] <= VETO_THRESHOLD_DEX
        and repr_summary["7500-8000"]["deep_dex_p95"] <= VETO_THRESHOLD_DEX
    )
    phys_pass = (
        phys_summary["7000-7500"]["deep_dex_p95"] <= VETO_THRESHOLD_DEX
        and phys_summary["7500-8000"]["deep_dex_p95"] <= VETO_THRESHOLD_DEX
    )
    if not repr_pass:
        status = "family_vetoed_by_representation_oracle"
    elif not phys_pass:
        status = "physics_trigger_failed_activate_exit_logp_alt"
    else:
        status = "passes"

    artifact = {
        "format": "payne_zero_entropy_closure_v2_oracle",
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
            "dual_crossing: enter switch (convection on), exit switch "
            "(convection off), gamma_ad constant, bumps a_enter/a_exit"
        ),
        "representation_free_parameters_per_star": 5,
        "physics_free_parameters_per_star": 3,
        "pre_registered_veto_threshold_dex": VETO_THRESHOLD_DEX,
        "representation": repr_summary,
        "physics_trigger": phys_summary,
        "status": status,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "representation": repr_summary,
                      "physics_trigger": phys_summary}, indent=2))
    return 0 if repr_pass and phys_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
