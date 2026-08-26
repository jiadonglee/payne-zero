"""Does a physically motivated depth basis beat a generic one?  No, and why.

The depth axis of the compact formula is a Chebyshev series in ``ln tau``,
which is a generic basis for smooth functions and carries no physics.  There is
an obvious physical alternative for the temperature half, and it is not a
guess: the target ``log10(T / T_grey)`` is algebraically a Hopf function in
disguise.  Inverting ``T^4 = (3/4) Teff^4 (tau + q(tau))`` over the corpus
recovers ``q`` with a median of 0.567 at tau = 0.013 against the classical
0.580 there, so the outermost layers really are the textbook Hopf function.
Deeper it runs above grey -- 0.719 against 0.624 at tau = 0.237, 0.816 against
0.677 at tau = 0.75 -- which is line blanketing warming the atmosphere above
the grey solution, not a fitting artefact.  Deep, ``q`` turns negative and
diverges -- convection has taken over and the grey parameterization is
meaningless there -- which is the honest reason a single generic series needs
degree twenty: it is bridging two different physical regimes.

So the hypothesis: give the basis the grey solution explicitly, as columns of
the exact form ``(1/4) log10((tau + q) / (tau + 2/3))``, and let Chebyshev
handle only what is left.  These columns decay to 6e-5 by tau = 1000, so they
touch the surface and leave the deep alone by construction.

The hypothesis is correct about representation and irrelevant in practice.  At
matched basis dimension the physical columns improve the representation floor
by up to 1.9 times.  End to end at matched budget they are worth 0.5 percent,
and about 6 percent of stored floats at fixed accuracy.

The third measurement here explains the gap and is the useful one.  Splitting
the held-out temperature error by stage:

    depth basis alone             p95 0.00136 dex
    + rank-5 truncation           p95 0.00449 dex
    + label polynomial (actual)   p95 0.00856 dex

The depth basis contributes sixteen percent of the final error.  A perfect
depth axis would move p95 from 0.00856 to about 0.00845.  Rank truncation and
the label-to-amplitude polynomial own the rest, and that is where an
improvement has to come from -- physical or otherwise.

Recorded as a negative result with its reproducer, in the same spirit as the
vetoed entropy closures.  No solver calls.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.special import expn

from experiments.analytic_initializer.candidates import temperature_regimes
from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    file_sha256,
    grey_temperature,
    label_features,
    load_strict_truth,
    make_split,
    polynomial_exponents,
    polynomial_features,
)
from experiments.analytic_initializer.monotone_temperature import project_to_monotone

MANIFESTS = (
    Path("results/reconstruction_metrics.json"),
    Path("results/sealed_solver_subset_20260808.json"),
    Path("results/sealed_audit_20260808.json"),
    Path("results/sealed_audit_20260811.json"),
    Path("results/sealed_initializer_holdout_20260812.json"),
    Path("results/initializer_calibration_20260812.json"),
    Path("results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json"),
)
DEFAULT_OUTPUT = Path("results/analytic_initializer/hopf_basis_probe.json")

REGIME_BOUNDARIES = (5500.0, 7500.0)
DEEP_START, DEEP_TRIM = 39, 5
# expn overflows to zero well before this; the clamp only silences the warning.
_EXPN_CLAMP = 700.0


def grey_hopf_column(tau: np.ndarray, q: np.ndarray | float) -> np.ndarray:
    """``log10(T / T_grey)`` for a grey atmosphere with Hopf parameter ``q``.

    Exact, not fitted: this is the temperature profile the transfer equation
    gives for a grey atmosphere, written as a residual against the Eddington
    approximation ``q = 2/3``.  It tends to zero deep, so a column of this form
    can only carry surface structure.
    """

    depth = np.asarray(tau, dtype=np.float64)
    return 0.25 * np.log10((depth + q) / (depth + 2.0 / 3.0))


def classical_hopf_parameter(tau: np.ndarray) -> np.ndarray:
    """The textbook Hopf function, 0.577 at the surface to 0.710 deep."""

    return 0.710 - 0.133 * np.exp(-1.85 * np.asarray(tau, dtype=np.float64))


def physical_columns(tau: np.ndarray, variant: str) -> np.ndarray:
    """Build one candidate physical block."""

    depth = np.asarray(tau, dtype=np.float64)
    classical = grey_hopf_column(depth, classical_hopf_parameter(depth))
    kernels = [expn(2, np.minimum(depth, _EXPN_CLAMP)), expn(3, np.minimum(depth, _EXPN_CLAMP))]
    blocks = {
        "classical_hopf": [classical],
        "hopf_plus_1q": [classical, grey_hopf_column(depth, 0.20)],
        "hopf_plus_2q": [classical, grey_hopf_column(depth, 0.15), grey_hopf_column(depth, 0.95)],
        "hopf_plus_5q": [classical]
        + [grey_hopf_column(depth, q) for q in (0.10, 0.30, 0.577, 0.710, 0.90)],
        "hopf_plus_kernels": [classical, *kernels],
        "hopf_2q_kernels": [
            classical,
            grey_hopf_column(depth, 0.15),
            grey_hopf_column(depth, 0.95),
            *kernels,
        ],
    }
    if variant not in blocks:
        raise ValueError(f"unknown physical basis variant: {variant}")
    return np.column_stack(blocks[variant])


def _coordinate(tau: np.ndarray) -> np.ndarray:
    log_tau = np.log(np.asarray(tau, dtype=np.float64))
    return 2.0 * (log_tau - log_tau.min()) / (log_tau.max() - log_tau.min()) - 1.0


def _design(tau: np.ndarray, variant: str | None, dimension: int) -> np.ndarray:
    """A basis of exactly ``dimension`` columns, physical block first."""

    if variant is None:
        return np.polynomial.chebyshev.chebvander(_coordinate(tau), dimension - 1)
    block = physical_columns(tau, variant)
    remaining = dimension - block.shape[1]
    if remaining < 1:
        raise ValueError("dimension is too small for this physical block")
    return np.column_stack(
        [block, np.polynomial.chebyshev.chebvander(_coordinate(tau), remaining - 1)]
    )


def _floor(design: np.ndarray, target: np.ndarray) -> dict:
    coefficients, *_ = np.linalg.lstsq(design, target.T, rcond=None)
    error = np.abs((design @ coefficients).T - target)
    return {
        "dimension": int(design.shape[1]),
        "p50": float(np.percentile(error, 50.0)),
        "p95": float(np.percentile(error, 95.0)),
        "surface_p95": float(np.percentile(error[:, :DEEP_START], 95.0)),
        "deep_p95": float(np.percentile(error[:, DEEP_START : error.shape[1] - DEEP_TRIM], 95.0)),
    }


def _fit_and_score(corpus, split, target, variant, degree, components, center_dim, mode_dim):
    """Run the real pipeline with a swapped depth basis and score it."""

    tau, rows = corpus.tau, split.validation
    center_design = _design(tau, variant, center_dim)
    mode_design = _design(tau, variant, mode_dim)
    features = label_features(corpus.labels)
    center = features[split.train].mean(axis=0)
    scale = np.maximum(features[split.train].std(axis=0), 1.0e-12)
    exponents = polynomial_exponents(5, degree)
    train, _, _ = polynomial_features(
        ((features - center) / scale)[split.train], exponents, center=np.zeros(5), scale=np.ones(5)
    )
    holdout, _, _ = polynomial_features(
        (features[rows] - center) / scale, exponents, center=np.zeros(5), scale=np.ones(5)
    )
    regimes = temperature_regimes(corpus.labels, boundaries=REGIME_BOUNDARIES)

    prediction = np.zeros((rows.size, tau.size))
    stored = 2 + 10 + 2 + 1
    for regime in range(3):
        mask = regimes[split.train] == regime
        values = target[split.train][mask]
        mean_profile = center_design @ np.linalg.lstsq(
            center_design, values.mean(axis=0), rcond=None
        )[0]
        residual = (
            mode_design
            @ np.linalg.lstsq(mode_design, (values - mean_profile).T, rcond=None)[0]
        ).T
        basis = np.linalg.svd(residual, full_matrices=False)[2][:components]
        amplitudes = np.linalg.lstsq(train[mask], residual @ basis.T, rcond=None)[0]
        stored += center_dim + components * mode_dim + exponents.shape[0] * components
        selected = regimes[rows] == regime
        if selected.any():
            prediction[selected] = mean_profile + (holdout[selected] @ amplitudes) @ basis
    return stored, prediction


def _error_budget(corpus, split, target, center_dim, mode_dim, degree, components) -> dict:
    """Split the held-out error into depth, rank and label-map contributions."""

    tau, rows = corpus.tau, split.validation
    center_design = _design(tau, None, center_dim)
    mode_design = _design(tau, None, mode_dim)
    features = label_features(corpus.labels)
    center = features[split.train].mean(axis=0)
    scale = np.maximum(features[split.train].std(axis=0), 1.0e-12)
    exponents = polynomial_exponents(5, degree)
    train, _, _ = polynomial_features(
        ((features - center) / scale)[split.train], exponents, center=np.zeros(5), scale=np.ones(5)
    )
    holdout, _, _ = polynomial_features(
        (features[rows] - center) / scale, exponents, center=np.zeros(5), scale=np.ones(5)
    )
    regimes = temperature_regimes(corpus.labels, boundaries=REGIME_BOUNDARIES)
    stages = {name: np.zeros((rows.size, tau.size)) for name in ("depth", "rank", "label")}

    for regime in range(3):
        mask = regimes[split.train] == regime
        values = target[split.train][mask]
        mean_profile = center_design @ np.linalg.lstsq(
            center_design, values.mean(axis=0), rcond=None
        )[0]
        residual = (
            mode_design
            @ np.linalg.lstsq(mode_design, (values - mean_profile).T, rcond=None)[0]
        ).T
        basis = np.linalg.svd(residual, full_matrices=False)[2][:components]
        amplitudes = np.linalg.lstsq(train[mask], residual @ basis.T, rcond=None)[0]
        selected = regimes[rows] == regime
        if not selected.any():
            continue
        held = target[rows][selected]
        # Each star represented as well as the depth basis allows.
        stages["depth"][selected] = (
            center_design @ np.linalg.lstsq(center_design, held.T, rcond=None)[0]
        ).T
        # Plus rank truncation, with per-star amplitudes: an oracle label map.
        oracle = (
            mode_design
            @ np.linalg.lstsq(mode_design, (held - mean_profile).T, rcond=None)[0]
        ).T
        stages["rank"][selected] = mean_profile + (oracle @ basis.T) @ basis
        # Plus amplitudes from the labels: what the formula actually does.
        stages["label"][selected] = (
            mean_profile + (holdout[selected] @ amplitudes) @ basis
        )

    held = target[rows]
    budget, previous = [], 0.0
    for name in ("depth", "rank", "label"):
        p95 = float(np.percentile(np.abs(stages[name] - held), 95.0))
        budget.append(
            {
                "stage": name,
                "cumulative_p95_dex": p95,
                "added_by_this_stage_dex": p95 - previous,
                "share_of_final": None,
            }
        )
        previous = p95
    for entry in budget:
        entry["share_of_final"] = entry["added_by_this_stage_dex"] / budget[-1]["cumulative_p95_dex"]
    return {"stages": budget, "center_dimension": center_dim, "mode_dimension": mode_dim}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=20260816)
    args = parser.parse_args()

    corpus = load_strict_truth(args.corpus)
    excluded, manifests = collect_excluded_indices(MANIFESTS, corpus_size=corpus.size)
    split = make_split(corpus.size, excluded=excluded, seed=args.split_seed)
    tau, rows = corpus.tau, split.validation
    grey = grey_temperature(corpus.labels[:, 0], tau)
    target = np.log10(corpus.temperature / grey)
    truth = corpus.temperature[rows]
    labels = corpus.labels[rows]

    # The implied Hopf function, which is what motivated the hypothesis.
    implied = (tau[None, :] + 2.0 / 3.0) * 10.0 ** (4.0 * target) - tau[None, :]
    hopf_recovery = [
        {
            "tau": float(tau[j]),
            "implied_q_p50": float(np.percentile(implied[:, j], 50.0)),
            "implied_q_p16": float(np.percentile(implied[:, j], 16.0)),
            "implied_q_p84": float(np.percentile(implied[:, j], 84.0)),
            "classical_q": float(classical_hopf_parameter(tau[j])),
        }
        for j in (0, 20, 40, 50, 54, 60, 70, 79)
    ]

    floors = []
    for dimension in (9, 13, 17, 21):
        floors.append({"variant": "chebyshev", **_floor(_design(tau, None, dimension), target)})
        for variant in (
            "classical_hopf",
            "hopf_plus_1q",
            "hopf_plus_2q",
            "hopf_plus_5q",
            "hopf_plus_kernels",
            "hopf_2q_kernels",
        ):
            block = physical_columns(tau, variant)
            if dimension - block.shape[1] < 2:
                continue
            floors.append(
                {
                    "variant": variant,
                    "physical_columns": int(block.shape[1]),
                    "physical_block_condition_number": float(np.linalg.cond(block)),
                    "physical_block_max_at_tau_1000": float(np.abs(block[-1]).max()),
                    **_floor(_design(tau, variant, dimension), target),
                }
            )

    end_to_end = []
    window = slice(DEEP_START, corpus.layers - DEEP_TRIM)
    for variant, degree, components, center_dim, mode_dim in itertools.product(
        (None, "hopf_2q_kernels"), (2, 3), (2, 3, 4, 5), (9, 13, 17, 23), (7, 11, 15, 19)
    ):
        if mode_dim > center_dim:
            continue
        if variant is not None and min(center_dim, mode_dim) < 7:
            continue
        stored, prediction = _fit_and_score(
            corpus, split, target, variant, degree, components, center_dim, mode_dim
        )
        predicted = project_to_monotone(tau, labels[:, 0], grey[rows] * 10.0**prediction)
        relative = np.abs(predicted / truth - 1.0)
        end_to_end.append(
            {
                "variant": "chebyshev" if variant is None else variant,
                "degree": degree,
                "components": components,
                "center_dimension": center_dim,
                "mode_dimension": mode_dim,
                "stored_floats": stored,
                "temperature_relative_p95": float(np.percentile(relative, 95.0)),
                "temperature_relative_deep_p95": float(np.percentile(relative[:, window], 95.0)),
            }
        )

    best_at_budget = []
    for limit in (300, 400, 550, 800, 1200):
        entry = {"budget": limit}
        for variant in ("chebyshev", "hopf_2q_kernels"):
            candidates = [
                e
                for e in end_to_end
                if e["variant"] == variant and e["stored_floats"] <= limit
            ]
            if candidates:
                entry[variant] = min(candidates, key=lambda e: e["temperature_relative_p95"])
        if "chebyshev" in entry and "hopf_2q_kernels" in entry:
            entry["hopf_gain"] = (
                1.0
                - entry["hopf_2q_kernels"]["temperature_relative_p95"]
                / entry["chebyshev"]["temperature_relative_p95"]
            )
        best_at_budget.append(entry)

    payload = {
        "format": "payne_zero_hopf_basis_probe_v1",
        "date": "2026-08-17",
        "question": (
            "Does replacing part of the generic Chebyshev depth basis with the "
            "exact grey/Hopf temperature profile improve the emulator-free "
            "warm start?"
        ),
        "answer": (
            "No, in the only sense that matters. The physical columns do improve "
            "the depth representation floor by up to 1.9 times at matched "
            "dimension, but the depth basis contributes only 16 percent of the "
            "held-out error, so end to end the gain is 0.5 percent and about 6 "
            "percent of stored floats. Rank truncation and the label-to-"
            "amplitude polynomial own 37 and 48 percent respectively, and that "
            "is where any real improvement has to come from."
        ),
        "verdict": "not adopted: real but immaterial; the optimized term is not binding",
        "corpus": {"path": str(args.corpus), "sha256": file_sha256(args.corpus), "rows": corpus.size},
        "split": {
            "seed": args.split_seed,
            "train": int(split.train.size),
            "validation": int(split.validation.size),
            "manifests": manifests,
        },
        "hopf_recovery": {
            "note": (
                "Inverting the target through T^4 = (3/4) Teff^4 (tau + q) "
                "recovers the textbook Hopf function at the surface and "
                "diverges deep, where convection replaces the grey solution."
            ),
            "rows": hopf_recovery,
        },
        "representation_floor_dex": floors,
        "end_to_end": end_to_end,
        "best_at_budget": best_at_budget,
        "error_budget": _error_budget(corpus, split, target, 23, 19, 3, 5),
        "reproducer": "PYTHONPATH=. python3 -m experiments.analytic_initializer.run_hopf_basis_probe",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["error_budget"], indent=2))
    for entry in best_at_budget:
        if "hopf_gain" in entry:
            print(
                "budget %5d: chebyshev %.5f  hopf %.5f  gain %+.2f%%"
                % (
                    entry["budget"],
                    entry["chebyshev"]["temperature_relative_p95"],
                    entry["hopf_2q_kernels"]["temperature_relative_p95"],
                    100.0 * entry["hopf_gain"],
                )
            )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
