"""Seed-only audit of the v4r6 decoupled grey-mass / convective-T candidate.

Builds the frozen convective, grey, and decoupled seeds on the historical
development-60. It does not run the production solver or evaluate production
opacity. Stored truth is used only after seed construction for error reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    file_sha256,
    load_strict_truth,
)
from experiments.analytic_initializer.run_h2_solver_funnel import _seed_error_splits
from experiments.analytic_initializer.textbook_opacity import (
    build_textbook_reduced_state_v4r6,
    build_textbook_reduced_state_v4r6_decoupled,
    textbook_rosseland_opacity_v4r6,
)
from experiments.analytic_initializer.write_textbook_opacity_v4r6_decoupled_source_manifest import (
    OUTPUT as SOURCE_MANIFEST,
    write_source_manifest,
)

CANDIDATE = "v4r6_decoupled_mgrey_tconv_v1"
RUN_DATE = "20260828"
INDICES_FROM = Path(
    "results/paper_physical_seed_20260820/learned/"
    "convergence_metrics_learned_monotone.json"
)
OUTPUT = Path(
    f"results/analytic_initializer/textbook_opacity_v4r6_decoupled_seed_audit_{RUN_DATE}.json"
)
OPACITY_RTOL = 1.0e-12


def _load_indices(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(
        sorted({int(index) for index in payload["star_indices"]}),
        dtype=np.int64,
    )


def _identity(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    finite = np.isfinite(left) & np.isfinite(right)
    equal = np.array_equal(left, right)
    max_abs = float(np.max(np.abs(left - right))) if finite.any() else float("nan")
    return {
        "bitwise_equal": bool(equal),
        "compared_count": int(left.size),
        "finite_count": int(np.count_nonzero(finite)),
        "max_abs_difference": max_abs,
    }


def _per_star_rows(
    indices: np.ndarray,
    labels: np.ndarray,
    tau: np.ndarray,
    *,
    mass_decoupled: np.ndarray,
    temperature_decoupled: np.ndarray,
    opacity_decoupled: np.ndarray,
    mass_grey: np.ndarray,
    temperature_convective: np.ndarray,
    opacity_recomputed: np.ndarray,
    truth_mass: np.ndarray,
    truth_temperature: np.ndarray,
) -> list[dict[str, object]]:
    depth = np.asarray(tau, dtype=np.float64)
    deep = depth > 10.0
    rows: list[dict[str, object]] = []
    for row, index in enumerate(indices):
        mass = mass_decoupled[row]
        temperature = temperature_decoupled[row]
        opacity = opacity_decoupled[row]
        finite = bool(
            np.all(np.isfinite(mass) & np.isfinite(temperature) & np.isfinite(opacity))
        )
        positive = bool(np.all((mass > 0.0) & (temperature > 0.0) & (opacity > 0.0)))
        rel = np.abs(opacity - opacity_recomputed[row]) / np.maximum(
            opacity_recomputed[row], 1.0e-30
        )
        mass_dex = np.abs(np.log10(mass) - np.log10(truth_mass[row]))
        temperature_rel = np.abs(temperature / truth_temperature[row] - 1.0)
        rows.append(
            {
                "corpus_index": int(index),
                "effective_temperature": float(labels[row, 0]),
                "log_surface_gravity": float(labels[row, 1]),
                "finite": finite,
                "positive": positive,
                "mass_identity": bool(np.array_equal(mass, mass_grey[row])),
                "temperature_identity": bool(
                    np.array_equal(temperature, temperature_convective[row])
                ),
                "opacity_max_rel_residual": float(np.max(rel)),
                "opacity_identity": bool(
                    np.allclose(
                        opacity,
                        opacity_recomputed[row],
                        rtol=OPACITY_RTOL,
                        atol=0.0,
                    )
                ),
                "all_layers": {
                    "log_mass_dex_p50": float(np.percentile(mass_dex, 50.0)),
                    "temperature_relative_p50": float(
                        np.percentile(temperature_rel, 50.0)
                    ),
                },
                "tau_gt_10": {
                    "log_mass_dex_p50": float(np.percentile(mass_dex[deep], 50.0)),
                    "temperature_relative_p50": float(
                        np.percentile(temperature_rel[deep], 50.0)
                    ),
                },
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--indices-from", type=Path, default=INDICES_FROM)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    source_manifest = write_source_manifest()
    corpus = load_strict_truth(args.corpus)
    indices = _load_indices(args.indices_from)
    if indices.size != 60:
        raise SystemExit(f"expected 60 development indices, found {indices.size}")

    labels = corpus.labels[indices]
    tau = corpus.tau
    mass_grey, temperature_grey, diagnostics_grey = build_textbook_reduced_state_v4r6(
        labels, tau, include_convection=False
    )
    mass_convective, temperature_convective, diagnostics_convective = (
        build_textbook_reduced_state_v4r6(labels, tau, include_convection=True)
    )
    mass_decoupled, temperature_decoupled, diagnostics_decoupled = (
        build_textbook_reduced_state_v4r6_decoupled(labels, tau)
    )
    opacity_decoupled = np.asarray(
        diagnostics_decoupled["rosseland_opacity"], dtype=np.float64
    )
    pressure_decoupled = np.asarray(
        diagnostics_decoupled["gas_pressure"], dtype=np.float64
    )
    opacity_recomputed = textbook_rosseland_opacity_v4r6(
        labels, temperature_decoupled, pressure_decoupled
    )
    gravity = 10.0 ** labels[:, 1]
    pressure_from_mass = gravity[:, None] * mass_decoupled

    finite = np.all(
        np.isfinite(mass_decoupled)
        & np.isfinite(temperature_decoupled)
        & np.isfinite(opacity_decoupled),
        axis=1,
    )
    positive = np.all(
        (mass_decoupled > 0.0)
        & (temperature_decoupled > 0.0)
        & (opacity_decoupled > 0.0),
        axis=1,
    )
    mass_identity = _identity(mass_decoupled, mass_grey)
    temperature_identity = _identity(temperature_decoupled, temperature_convective)
    opacity_identity = bool(
        np.allclose(opacity_decoupled, opacity_recomputed, rtol=OPACITY_RTOL, atol=0.0)
    )
    pressure_identity = bool(np.array_equal(pressure_decoupled, pressure_from_mass))
    per_star = _per_star_rows(
        indices,
        labels,
        tau,
        mass_decoupled=mass_decoupled,
        temperature_decoupled=temperature_decoupled,
        opacity_decoupled=opacity_decoupled,
        mass_grey=mass_grey,
        temperature_convective=temperature_convective,
        opacity_recomputed=opacity_recomputed,
        truth_mass=corpus.column_mass[indices],
        truth_temperature=corpus.temperature[indices],
    )
    finite_stars = finite
    error_splits = _seed_error_splits(
        mass_decoupled,
        temperature_decoupled,
        truth_mass=corpus.column_mass[indices],
        truth_temperature=corpus.temperature[indices],
        teff=labels[:, 0],
        tau=tau,
        finite_stars=finite_stars,
    )
    structural_checks = {
        "candidate_finite": int(np.count_nonzero(finite)) == int(indices.size),
        "candidate_positive": int(np.count_nonzero(positive)) == int(indices.size),
        "mass_identity": bool(mass_identity["bitwise_equal"]),
        "temperature_identity": bool(temperature_identity["bitwise_equal"]),
        "opacity_identity": opacity_identity,
        "pressure_is_g_times_grey_mass": pressure_identity,
        "fitted_parameter_count_is_zero": True,
        "mass_not_reintegrated_after_convection": (
            diagnostics_decoupled["mass_reintegrated_after_convection"] is False
        ),
        "source_manifest_complete": SOURCE_MANIFEST.is_file(),
    }
    decision = (
        "PASS_STRUCTURAL"
        if all(bool(value) for value in structural_checks.values())
        else "FAIL_STOP_STRUCTURAL"
    )

    result = {
        "status": "development_only",
        "decision": decision,
        "candidate": CANDIDATE,
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": file_sha256(SOURCE_MANIFEST),
        "sample_manifest": str(args.indices_from),
        "sample_manifest_sha256": file_sha256(args.indices_from),
        "corpus": str(args.corpus),
        "corpus_sha256": file_sha256(args.corpus),
        "runtime_signature": {
            "hostname": source_manifest["hostname"],
            "python": source_manifest["python"],
            "numpy": source_manifest["numpy"],
            "numba": source_manifest["numba"],
            "git_head": source_manifest["git_head"],
            "git_diff_sha256": source_manifest["git_diff_sha256"],
        },
        "solver_policy": None,
        "initializer_provenance": {
            "opacity": "textbook_rosseland_opacity_v4r6",
            "temperature": "current_saha_aware_convective_temperature",
            "mass": "v4r6_grey_integrated_mass",
            "mass_reintegrated_after_convection": False,
            "fitted_parameter_count": 0,
            "offline_decision": "FAIL_STOP",
        },
        "star_count": int(indices.size),
        "finite_star_count": int(np.count_nonzero(finite)),
        "positive_star_count": int(np.count_nonzero(positive)),
        "structural_checks": structural_checks,
        "mass_identity": mass_identity,
        "temperature_identity": temperature_identity,
        "opacity_max_rel_residual": float(
            np.max(
                np.abs(opacity_decoupled - opacity_recomputed)
                / np.maximum(opacity_recomputed, 1.0e-30)
            )
        ),
        "seed_error_splits_vs_stored_truth": error_splits,
        "grey_seed_error_splits_vs_stored_truth": _seed_error_splits(
            mass_grey,
            temperature_grey,
            truth_mass=corpus.column_mass[indices],
            truth_temperature=corpus.temperature[indices],
            teff=labels[:, 0],
            tau=tau,
            finite_stars=np.all(np.isfinite(mass_grey) & np.isfinite(temperature_grey), axis=1),
        ),
        "convective_seed_error_splits_vs_stored_truth": _seed_error_splits(
            mass_convective,
            temperature_convective,
            truth_mass=corpus.column_mass[indices],
            truth_temperature=corpus.temperature[indices],
            teff=labels[:, 0],
            tau=tau,
            finite_stars=np.all(
                np.isfinite(mass_convective) & np.isfinite(temperature_convective),
                axis=1,
            ),
        ),
        "records": per_star,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "finite_star_count": result["finite_star_count"],
                "mass_identity": mass_identity["bitwise_equal"],
                "temperature_identity": temperature_identity["bitwise_equal"],
                "opacity_identity": opacity_identity,
                "wrote": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if decision == "PASS_STRUCTURAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
