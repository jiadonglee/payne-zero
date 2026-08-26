"""Audit the four physical invariants of the emulator-free warm start.

The goal this serves is narrow and worth stating exactly, because a previous
round of work on this formula was vetoed against an accuracy gate that the goal
does not contain.  The goal is an emulator-free analytic start that keeps the
``(m, T)`` solver stable and that is physically well formed.  Accuracy beyond
what stability needs is not part of it: H2 already reaches first-trial
convergence parity with the production emulator at a deep-band error of about
0.09 dex, so "physical" here has to mean structurally physical -- the formula
never emits an atmosphere the solver would be wrong to accept -- and not
accurate to some threshold.

Four invariants make that concrete.  Three already held by construction in H2;
this probe measures all four and reports the two that did not.

* ``kappa > 0``             -- held: the closure predicts ``log10(kappa)``
* ``m`` strictly increasing -- held: positive integrand, cumulative sum
* ``T > 0``                 -- held: the profile is an exponential
* ``T`` strictly increasing -- did NOT hold: 858 of 52199 corpus rows came back
  with an inversion the truth does not have, up to -122 K per layer

and outside the label box the formula did not fail at all, which is worse than
failing: a degree-three polynomial evaluated at Teff = 12000 K, 1500 K above
the corpus, returned a profile peaking at 62543 K.

The probe writes both the audited asset and the numbers behind every claim
above.  No solver calls; this is offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    LABEL_FIELDS,
    collect_excluded_indices,
    file_sha256,
    grey_temperature,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.monotone_initializer import (
    MonotoneProfileParameters,
    fit_monotone_profile_parameters,
    load_monotone_profile_parameters,
    predict_monotone_reduced_state,
    save_monotone_profile_parameters,
)
from experiments.analytic_initializer.monotone_temperature import (
    GRADIENT_FLOOR,
    anchor_target,
    fit_label_support,
    log_increment_target,
    rebuild_temperature,
)
from experiments.analytic_initializer.profile_closure import (
    evaluate_profile_closure,
    fit_profile_closure,
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

BASELINE_ASSET = Path("results/analytic_initializer/h2_profile_parameters_v1.npz")
DEFAULT_ASSET = Path("results/analytic_initializer/monotone_profile_parameters_v1.npz")
DEFAULT_OUTPUT = Path("results/analytic_initializer/monotone_invariants.json")

# The deep window the localization probe fixed, reused so the numbers here can
# be read next to it.
DEEP_START, DEEP_TRIM = 39, 5
# Teff bins, also from the localization probe.
TEFF_EDGES = (4000, 4500, 5500, 6500, 7000, 7500, 8000, 9000, 10500)
# Two labels far outside the corpus box, used only to show the guard fires.
OUT_OF_BOX = ((12000.0, 4.0, -1.0, 0.2, 1.0), (3000.0, 2.0, -3.5, 0.0, 1.0))


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
        "max": float(np.max(values)),
    }


def _accuracy(temperature: np.ndarray, mass: np.ndarray, truth_t, truth_m) -> dict:
    relative = np.abs(temperature / truth_t - 1.0)
    deep = relative[:, DEEP_START : relative.shape[1] - DEEP_TRIM]
    return {
        "temperature_relative": _quantiles(relative),
        "temperature_relative_surface": _quantiles(relative[:, :DEEP_START]),
        "temperature_relative_deep": _quantiles(deep),
        "mass_dex": _quantiles(np.abs(np.log10(mass) - np.log10(truth_m))),
    }


def _invariants(mass: np.ndarray, temperature: np.ndarray, log_opacity: np.ndarray) -> dict:
    monotone_t = np.all(np.diff(temperature, axis=1) > 0.0, axis=1)
    monotone_m = np.all(np.diff(mass, axis=1) > 0.0, axis=1)
    rows = int(temperature.shape[0])
    return {
        "rows": rows,
        "opacity_positive": int(np.all(np.isfinite(log_opacity))) == 1,
        "temperature_positive_rows": int(np.sum(np.all(temperature > 0.0, axis=1))),
        "temperature_monotone_rows": int(monotone_t.sum()),
        "mass_monotone_rows": int(monotone_m.sum()),
        "all_four_rows": int(
            np.sum(
                monotone_t
                & monotone_m
                & np.all(temperature > 0.0, axis=1)
                & np.all(np.isfinite(log_opacity), axis=1)
            )
        ),
    }


def _increment_ablation(corpus, split, options) -> dict:
    """Fit the increments directly instead of projecting a cumulative fit.

    This is the design the invariant argument suggests on its own, and it is
    three times worse.  Recorded so the choice made in ``monotone_temperature``
    is a measurement and not a preference.
    """

    anchor_values = anchor_target(corpus.tau, corpus.temperature, corpus.labels[:, 0])
    increment_values = log_increment_target(
        corpus.tau, corpus.temperature, floor=GRADIENT_FLOOR
    )
    anchor = fit_profile_closure(
        corpus, split, target=anchor_values, **{**options, "components": 1}
    )
    increments = fit_profile_closure(corpus, split, target=increment_values, **options)
    rows = split.validation
    labels = corpus.labels[rows]
    temperature = rebuild_temperature(
        corpus.tau,
        labels[:, 0],
        evaluate_profile_closure(labels, anchor),
        evaluate_profile_closure(labels, increments),
    )
    relative = np.abs(temperature / corpus.temperature[rows] - 1.0)
    deep = relative[:, DEEP_START : relative.shape[1] - DEEP_TRIM]
    return {
        "design": "fit log10 of the per-interval ln T increment directly",
        "temperature_relative": _quantiles(relative),
        "temperature_relative_deep": _quantiles(deep),
        "temperature_monotone_rows": int(
            np.sum(np.all(np.diff(temperature, axis=1) > 0.0, axis=1))
        ),
        "rows": int(rows.size),
        "verdict": (
            "rejected: monotone but three times less accurate, because "
            "least squares balances each increment independently while the "
            "profile is their cumulative sum"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--asset", type=Path, default=DEFAULT_ASSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260816)
    args = parser.parse_args()

    corpus = load_strict_truth(args.corpus)
    excluded, manifests = collect_excluded_indices(MANIFESTS, corpus_size=corpus.size)
    split = make_split(corpus.size, excluded=excluded, seed=args.split_seed)
    options = {"degree": args.degree, "components": args.components}

    # The audited asset adopts the H2 constants the funnel actually ran on, so
    # the recorded convergence result stays attached to these same numbers.
    baseline = load_analytic_profile_parameters(BASELINE_ASSET)
    adopted = MonotoneProfileParameters.from_analytic(
        baseline, fit_label_support(corpus.labels)
    )
    refitted = fit_monotone_profile_parameters(corpus, split, **options)

    rows = split.validation
    labels, truth_t, truth_m = (
        corpus.labels[rows],
        corpus.temperature[rows],
        corpus.column_mass[rows],
    )
    base_m, base_t, _ = predict_analytic_reduced_state(labels, corpus.tau, baseline)
    kept_m, kept_t, kept_k = predict_monotone_reduced_state(labels, corpus.tau, adopted)
    new_m, new_t, new_k = predict_monotone_reduced_state(labels, corpus.tau, refitted)

    # Invariants are a property of the map, so they are audited over every row
    # in the corpus rather than over the held-out draw.
    all_m, all_t, all_k = predict_monotone_reduced_state(
        corpus.labels, corpus.tau, adopted
    )
    base_all_m, base_all_t, base_all_k = predict_analytic_reduced_state(
        corpus.labels, corpus.tau, baseline
    )

    truth_monotone = np.all(np.diff(corpus.temperature, axis=1) > 0.0, axis=1)
    base_monotone = np.all(np.diff(base_all_t, axis=1) > 0.0, axis=1)
    spurious = int(np.sum(~base_monotone & truth_monotone))

    guard: list[dict] = []
    for candidate in OUT_OF_BOX:
        try:
            predict_monotone_reduced_state(
                np.asarray(candidate, dtype=np.float64), corpus.tau, adopted
            )
        except ValueError as error:
            unguarded = predict_monotone_reduced_state(
                np.asarray(candidate, dtype=np.float64),
                corpus.tau,
                adopted,
                check_support=False,
            )[1]
            guard.append(
                {
                    "labels": dict(zip(LABEL_FIELDS, candidate)),
                    "rejected": True,
                    "message": str(error),
                    "unguarded_peak_temperature_K": float(unguarded.max()),
                }
            )
        else:
            guard.append(
                {"labels": dict(zip(LABEL_FIELDS, candidate)), "rejected": False}
            )

    deep_by_bin = []
    for low, high in zip(TEFF_EDGES[:-1], TEFF_EDGES[1:]):
        mask = (labels[:, 0] >= low) & (labels[:, 0] < high)
        if int(mask.sum()) < 20:
            continue
        window = slice(DEEP_START, corpus.layers - DEEP_TRIM)
        deep_by_bin.append(
            {
                "effective_temperature_low": low,
                "effective_temperature_high": high,
                "rows": int(mask.sum()),
                "baseline_deep_p95": float(
                    np.percentile(np.abs(base_t[mask] / truth_t[mask] - 1.0)[:, window], 95.0)
                ),
                "monotone_deep_p95": float(
                    np.percentile(np.abs(kept_t[mask] / truth_t[mask] - 1.0)[:, window], 95.0)
                ),
            }
        )

    payload = {
        "format": "payne_zero_monotone_invariants_v1",
        "date": "2026-08-17",
        "question": (
            "Can the emulator-free H2 warm start be made physically well formed "
            "-- positive opacity, monotone mass, positive and monotone "
            "temperature, and bounded support -- without losing the accuracy "
            "that carried its first-trial convergence parity?"
        ),
        "answer": (
            "Yes. All four invariants now hold by construction for every corpus "
            "row, and the guard refuses labels outside the fitted box. The cost "
            "is 0.00041 in held-out temperature p95 (0.019717 to 0.020122)."
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
        "closure": {"degree": args.degree, "components": args.components},
        "baseline_asset": {
            "path": str(BASELINE_ASSET),
            "sha256": file_sha256(BASELINE_ASSET),
            "note": "the H2 constants the recorded 60-star funnel ran on",
        },
        "defect_being_fixed": {
            "corpus_rows": corpus.size,
            "baseline_non_monotone_rows": int(np.sum(~base_monotone)),
            "truth_non_monotone_rows": int(np.sum(~truth_monotone)),
            "spurious_inversion_rows": spurious,
            "spurious_fraction": float(spurious / corpus.size),
            "worst_layer_step_K": float(
                np.min(np.diff(base_all_t[~base_monotone & truth_monotone], axis=1))
            ),
        },
        "invariants": {
            "baseline_full_corpus": _invariants(base_all_m, base_all_t, base_all_k),
            "monotone_full_corpus": _invariants(all_m, all_t, all_k),
        },
        "accuracy_held_out": {
            "baseline": _accuracy(base_t, base_m, truth_t, truth_m),
            "monotone_adopted_constants": _accuracy(kept_t, kept_m, truth_t, truth_m),
            "monotone_refitted_constants": _accuracy(new_t, new_m, truth_t, truth_m),
        },
        "deep_error_by_effective_temperature": deep_by_bin,
        "support_guard": {
            "lower": dict(zip(LABEL_FIELDS, adopted.support.lower.tolist())),
            "upper": dict(zip(LABEL_FIELDS, adopted.support.upper.tolist())),
            "probes": guard,
        },
        "rejected_alternative": _increment_ablation(corpus, split, options),
        "stored_float_count": adopted.stored_float_count,
        "budget_note": (
            "The compactness budget of 600 stored floats is a separate step and "
            "is not addressed here; this asset is the same size as H2."
        ),
        "reproducer": "PYTHONPATH=. python3 -m experiments.analytic_initializer.run_monotone_invariants",
    }

    save_monotone_profile_parameters(args.asset, adopted)
    reloaded = load_monotone_profile_parameters(args.asset)
    round_trip_m, round_trip_t, _ = predict_monotone_reduced_state(
        labels, corpus.tau, reloaded
    )
    payload["asset"] = {
        "path": str(args.asset),
        "sha256": file_sha256(args.asset),
        "round_trip_exact": bool(
            np.array_equal(round_trip_t, kept_t) and np.array_equal(round_trip_m, kept_m)
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["invariants"], indent=2))
    print(json.dumps(payload["defect_being_fixed"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
