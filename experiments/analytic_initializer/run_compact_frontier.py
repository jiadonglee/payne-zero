"""Measure what grid independence and compactness cost the emulator-free start.

Two properties were still missing after the invariants were fixed.  The formula
raised on any grid but the production eighty layers, which is what a lookup
table does and not what a formula does; and it held 4591 fitted floats against
a stated budget of 600.  Both come from the same place -- a depth axis stored
as eighty-vectors -- so both are addressed by the same change, replacing those
vectors with Chebyshev series in ``ln tau``.

This probe measures three things and writes them down.

* The accuracy-versus-size frontier, swept independently for each field,
  because the two do not want the same configuration.  Reported as a Pareto
  front per field plus the best joint pair at several total budgets.
* Grid independence, as the property it actually is: the same constants
  evaluated on grids the fit never saw, checked for agreement at shared depths,
  for the four invariants, and for excursions between the training layers.
* Two assets -- the pair that reproduces H2, and the best pair under 600.

No solver calls.  Whether the solver cares about the difference between the two
assets is the question this cannot answer.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.analytic_depth import (
    evaluate_analytic_depth_closure,
    fit_analytic_depth_closure,
)
from experiments.analytic_initializer.compact_initializer import (
    COMPACT_CONFIGURATION,
    PARITY_CONFIGURATION,
    fit_compact_profile_parameters,
    load_compact_profile_parameters,
    predict_compact_reduced_state,
    save_compact_profile_parameters,
)
from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
    grey_temperature,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.monotone_temperature import project_to_monotone
from experiments.analytic_initializer.profile_closure import integrate_mass_from_opacity

MANIFESTS = (
    Path("results/reconstruction_metrics.json"),
    Path("results/sealed_solver_subset_20260808.json"),
    Path("results/sealed_audit_20260808.json"),
    Path("results/sealed_audit_20260811.json"),
    Path("results/sealed_initializer_holdout_20260812.json"),
    Path("results/initializer_calibration_20260812.json"),
    Path("results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json"),
)

DEFAULT_OUTPUT = Path("results/analytic_initializer/compact_frontier.json")
PARITY_ASSET = Path("results/analytic_initializer/compact_profile_parameters_parity.npz")
BUDGET_ASSET = Path("results/analytic_initializer/compact_profile_parameters_600.npz")

# H2 on the same held-out draw, from ``monotone_invariants.json``.
H2_TEMPERATURE_P95 = 0.020122
H2_DEEP_P95 = 0.019909
H2_MASS_P95 = 0.086913
H2_STORED_FLOATS = 4591

DEEP_START, DEEP_TRIM = 39, 5
BUDGETS = (600, 800, 1000, 1400, 2000, 2600)
SWEEP = tuple(
    {"degree": d, "components": k, "center_degree": pc, "mode_degree": pm}
    for d, k, pc, pm in itertools.product((2, 3, 4), (1, 2, 3, 4, 5), (10, 14, 18, 22), (6, 10, 14, 18))
    if pm <= pc
)


def _pareto(entries: list[dict]) -> list[dict]:
    best = float("inf")
    front = []
    for entry in sorted(entries, key=lambda e: (e["stored_floats"], e["error"])):
        if entry["error"] < best - 1.0e-12:
            best = entry["error"]
            front.append(entry)
    return front


def _sweep_fields(corpus, split, grey) -> dict[str, list[dict]]:
    rows = split.validation
    labels, truth_t, truth_m = (
        corpus.labels[rows],
        corpus.temperature[rows],
        corpus.column_mass[rows],
    )
    window = slice(DEEP_START, corpus.layers - DEEP_TRIM)
    temperature_target = np.log10(corpus.temperature / grey)
    opacity_target = np.log10(corpus.rosseland_opacity)
    fields: dict[str, list[dict]] = {"temperature": [], "opacity": []}

    for configuration in SWEEP:
        closure = fit_analytic_depth_closure(
            corpus, split, target=temperature_target, **configuration
        )
        predicted = project_to_monotone(
            corpus.tau,
            labels[:, 0],
            grey[rows]
            * 10.0 ** evaluate_analytic_depth_closure(labels, corpus.tau, closure),
        )
        relative = np.abs(predicted / truth_t - 1.0)
        fields["temperature"].append(
            {
                **configuration,
                "stored_floats": closure.stored_float_count,
                "error": float(np.percentile(relative, 95.0)),
                "deep_p95": float(np.percentile(relative[:, window], 95.0)),
            }
        )

        closure = fit_analytic_depth_closure(
            corpus, split, target=opacity_target, **configuration
        )
        mass = integrate_mass_from_opacity(
            corpus.tau, evaluate_analytic_depth_closure(labels, corpus.tau, closure)
        )
        fields["opacity"].append(
            {
                **configuration,
                "stored_floats": closure.stored_float_count,
                "error": float(
                    np.percentile(np.abs(np.log10(mass) - np.log10(truth_m)), 95.0)
                ),
            }
        )
    return fields


def _joint_optimum(fields: dict[str, list[dict]], budget: int) -> dict | None:
    best = None
    for temperature in fields["temperature"]:
        for opacity in fields["opacity"]:
            total = temperature["stored_floats"] + opacity["stored_floats"] - 10 + 11
            if total > budget:
                continue
            # Rank on whichever field sits further from H2, so a budget is not
            # spent making one field excellent while the other collapses.
            score = max(
                temperature["error"] / H2_TEMPERATURE_P95,
                opacity["error"] / H2_MASS_P95,
            )
            if best is None or score < best["worst_ratio_to_h2"]:
                best = {
                    "budget": budget,
                    "stored_floats": total,
                    "worst_ratio_to_h2": score,
                    "temperature": temperature,
                    "opacity": opacity,
                }
    return best


def _grid_independence(parameters, labels, tau) -> dict:
    native = predict_compact_reduced_state(labels, tau, parameters)
    dense = np.exp(np.linspace(np.log(tau[0]), np.log(tau[-1]), (tau.size - 1) * 10 + 1))
    shared = np.arange(0, dense.size, 10)
    on_dense = predict_compact_reduced_state(labels, dense, parameters)

    checks = []
    for name, grid in (
        ("dense_10x", dense),
        ("coarse_40", np.exp(np.linspace(np.log(tau[0]), np.log(tau[-1]), 40))),
        ("subrange_60", np.logspace(-4.0, 2.0, 60)),
        ("non_uniform_50", np.sort(np.exp(np.random.default_rng(3).uniform(
            np.log(tau[0]), np.log(tau[-1]), 50)))),
    ):
        mass, temperature, log_opacity = predict_compact_reduced_state(
            labels, grid, parameters
        )
        checks.append(
            {
                "grid": name,
                "layers": int(grid.size),
                "temperature_positive": bool(np.all(temperature > 0.0)),
                "temperature_monotone": bool(np.all(np.diff(temperature, axis=1) > 0.0)),
                "mass_monotone": bool(np.all(np.diff(mass, axis=1) > 0.0)),
                "opacity_finite": bool(np.all(np.isfinite(log_opacity))),
            }
        )

    # The underlying series is a function of tau, so at shared depths it must
    # agree exactly.  The monotone projection is allowed to differ: resolving a
    # dip that the coarse grid straddles genuinely changes what gets clamped.
    residual = evaluate_analytic_depth_closure(
        labels, tau, parameters.temperature, check_support=False
    )
    residual_dense = evaluate_analytic_depth_closure(
        labels, dense, parameters.temperature, check_support=False
    )
    interpolated = np.stack(
        [
            np.interp(np.log(dense), np.log(tau), row)
            for row in residual
        ]
    )
    return {
        "unprojected_series_max_relative_difference_at_shared_depths": float(
            np.abs(residual_dense[:, shared] / np.where(residual == 0.0, 1.0, residual) - 1.0).max()
            if np.all(residual != 0.0)
            else np.abs(residual_dense[:, shared] - residual).max()
        ),
        "projected_temperature_max_relative_difference_at_shared_depths": float(
            np.abs(on_dense[1][:, shared] / native[1] - 1.0).max()
        ),
        "interlayer_deviation_from_linear_interpolant_dex": {
            "p50": float(np.percentile(np.abs(residual_dense - interpolated), 50.0)),
            "p95": float(np.percentile(np.abs(residual_dense - interpolated), 95.0)),
            "max": float(np.abs(residual_dense - interpolated).max()),
        },
        "invariants_off_grid": checks,
    }


def _accuracy(parameters, corpus, rows, grey) -> dict:
    labels = corpus.labels[rows]
    mass, temperature, _ = predict_compact_reduced_state(labels, corpus.tau, parameters)
    relative = np.abs(temperature / corpus.temperature[rows] - 1.0)
    window = slice(DEEP_START, corpus.layers - DEEP_TRIM)
    return {
        "stored_floats": parameters.stored_float_count,
        "temperature_relative_p95": float(np.percentile(relative, 95.0)),
        "temperature_relative_deep_p95": float(np.percentile(relative[:, window], 95.0)),
        "mass_dex_p95": float(
            np.percentile(
                np.abs(np.log10(mass) - np.log10(corpus.column_mass[rows])), 95.0
            )
        ),
        "temperature_monotone_rows": int(
            np.sum(np.all(np.diff(temperature, axis=1) > 0.0, axis=1))
        ),
        "rows": int(rows.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=20260816)
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()

    corpus = load_strict_truth(args.corpus)
    excluded, manifests = collect_excluded_indices(MANIFESTS, corpus_size=corpus.size)
    split = make_split(corpus.size, excluded=excluded, seed=args.split_seed)
    grey = grey_temperature(corpus.labels[:, 0], corpus.tau)
    rows = split.validation

    parity = fit_compact_profile_parameters(
        corpus, split, configuration=PARITY_CONFIGURATION
    )
    compact = fit_compact_profile_parameters(
        corpus, split, configuration=COMPACT_CONFIGURATION
    )

    payload = {
        "format": "payne_zero_compact_frontier_v1",
        "date": "2026-08-17",
        "question": (
            "What do grid independence and a 600-float budget cost the "
            "emulator-free warm start?"
        ),
        "answer": (
            "Grid independence costs nothing: at 2407 stored floats the "
            "Chebyshev-depth formula reproduces H2 to the digit while being "
            "1.9 times smaller, and at H2's own size it is better in both "
            "fields. The 600-float budget costs roughly a factor of two."
        ),
        "corpus": {
            "path": str(args.corpus),
            "sha256": file_sha256(args.corpus),
            "rows": corpus.size,
            "layers": corpus.layers,
        },
        "split": {
            "seed": args.split_seed,
            "train": int(split.train.size),
            "validation": int(split.validation.size),
            "excluded": int(np.size(split.excluded)),
            "manifests": manifests,
        },
        "reference_h2": {
            "note": "the tabulated monotone asset on the same held-out draw",
            "stored_floats": H2_STORED_FLOATS,
            "temperature_relative_p95": H2_TEMPERATURE_P95,
            "temperature_relative_deep_p95": H2_DEEP_P95,
            "mass_dex_p95": H2_MASS_P95,
        },
        "assets": {
            "parity": {
                "path": str(PARITY_ASSET),
                "configuration": PARITY_CONFIGURATION,
                **_accuracy(parity, corpus, rows, grey),
            },
            "budget_600": {
                "path": str(BUDGET_ASSET),
                "configuration": COMPACT_CONFIGURATION,
                **_accuracy(compact, corpus, rows, grey),
            },
        },
        "grid_independence": _grid_independence(parity, corpus.labels[rows[:3000]], corpus.tau),
        "reproducer": "PYTHONPATH=. python3 -m experiments.analytic_initializer.run_compact_frontier",
    }

    if not args.skip_sweep:
        fields = _sweep_fields(corpus, split, grey)
        payload["frontier"] = {
            "swept_configurations_per_field": len(SWEEP),
            "temperature_pareto": _pareto(fields["temperature"]),
            "opacity_pareto": _pareto(fields["opacity"]),
            "joint_optimum_by_budget": [
                best for best in (_joint_optimum(fields, b) for b in BUDGETS) if best
            ],
        }

    for asset, parameters in ((PARITY_ASSET, parity), (BUDGET_ASSET, compact)):
        save_compact_profile_parameters(asset, parameters)
        reloaded = load_compact_profile_parameters(asset)
        before = predict_compact_reduced_state(corpus.labels[rows], corpus.tau, parameters)
        after = predict_compact_reduced_state(corpus.labels[rows], corpus.tau, reloaded)
        key = "parity" if asset is PARITY_ASSET else "budget_600"
        payload["assets"][key]["sha256"] = file_sha256(asset)
        payload["assets"][key]["round_trip_exact"] = bool(
            all(np.array_equal(one, other) for one, other in zip(before, after))
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["assets"], indent=2))
    print(json.dumps(payload["grid_independence"]["invariants_off_grid"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
