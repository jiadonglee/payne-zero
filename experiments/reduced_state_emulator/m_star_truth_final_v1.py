"""Relax the v3 validation stars' truth and emulator endpoints to the frozen rule.

For each of the nine M-giant validation stars, both the emulator-solver
endpoint and the truth product are continued thirty iterations with the
stopping rule released, products saved every five.  Each arm's final state is
the last checkpoint whose trailing segment satisfies the frozen stability
rule; the TiO gate is then evaluated between the two relaxed endpoints (and,
for the record, between the unrelaxed endpoints as measured in v3).
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from emulator_v1_2.gates.compare_spectra import (
    _absolute_stats,
    _continuum_scaled_stats,
    _load_spectrum_npz,
    _relative_stats,
    _synthesize_one,
)

from .cool_star_step_test import (
    TrackSpec,
    _clone_atmosphere,
    _reconstruct_from_mt,
    _set_single_thread_environment,
)
from .m_star_bootstrap_v1 import _load_mt, _write_json
from .m_star_stop_rule_v1 import (
    MASS_STABLE_P95_DEX,
    SEGMENT,
    TEMPERATURE_STABLE_P95,
    TIO_STABLE_LIMIT,
    STABLE_SEGMENTS_REQUIRED,
    WINDOW_NM,
    RESOLUTION,
    _labels_for,
    _segment_flux_pass,
    _segment_stable,
)

from bench.run_reference import _solver_config  # noqa: E402
from payne_zero_atmosphere.runner import run_atmosphere_model  # noqa: E402
from payne_zero_atmosphere.synthesis_bridge import (  # noqa: E402
    save_product_structured_atmosphere,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_truth_final_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_CORPUS = (
    REPO_ROOT / "results" / "m_star_cool_corpus_mgiant_v1" / "cool_truth_corpus.npz"
)
DEFAULT_TRUTH_DIR = (
    REPO_ROOT
    / "results"
    / "m_star_emulator_mgiant_v3"
    / "spectral_stage"
    / "products"
    / "production_six_field"
)
DEFAULT_CANDIDATE_DIR = (
    REPO_ROOT
    / "results"
    / "m_star_emulator_mgiant_v3"
    / "candidate_validation"
    / "products"
)
DEFAULT_FLUX_GATE = REPO_ROOT / "results" / "m_star_emulator_v1" / "flux_gate.json"
EXTRA_ITERATIONS = 30


def _continue_released(
    *,
    start_atmosphere,
    arm_dir: Path,
    extra_iterations: int,
) -> None:
    """Continue with the stop released, products every five iterations."""

    arm_dir.mkdir(parents=True, exist_ok=True)
    save_product_structured_atmosphere(
        _clone_atmosphere(start_atmosphere),
        arm_dir / "iter_0000.npz",
        device="cpu",
        dtype="float64",
    )

    residual_handle = (arm_dir / "iterations.jsonl").open("w")

    def hook(iteration_index, setup, step):
        flux_error = np.asarray(
            step.remapped.finalization.temperature_correction_result.flux_error_percent,
            dtype=np.float64,
        )
        residual_handle.write(
            json.dumps(
                {
                    "iteration": int(iteration_index),
                    "flux_error_p95_percent": float(
                        np.percentile(np.abs(flux_error), 95.0)
                    ),
                    "flux_error_median_percent": float(
                        np.percentile(np.abs(flux_error), 50.0)
                    ),
                    "flux_error_max_percent": float(np.max(np.abs(flux_error))),
                },
                sort_keys=True,
            )
            + "\n"
        )
        residual_handle.flush()
        if int(iteration_index) % 5 == 0:
            save_product_structured_atmosphere(
                _clone_atmosphere(step.remapped.atmosphere),
                arm_dir / f"iter_{int(iteration_index):04d}.npz",
                device="cpu",
                dtype="float64",
            )
        return {"iteration": int(iteration_index)}

    config = dataclasses.replace(
        _solver_config(
            _clone_atmosphere(start_atmosphere),
            iterations_per_trial=int(extra_iterations),
            structured_atmosphere_path=None,
            debug_state_path=None,
        ),
        enable_convergence_stop=False,
    )
    run_atmosphere_model(config, after_iteration_hook=hook)


def _spectrum_cached(product: Path, spectrum_path: Path) -> Path:
    if not spectrum_path.is_file():
        _synthesize_one(
            product,
            spectrum_path,
            wavelength_start_nm=WINDOW_NM[0],
            wavelength_end_nm=WINDOW_NM[1],
            resolution=RESOLUTION,
            molecular_lines=True,
            device=None,
            dtype="float64",
        )
    return spectrum_path


def _frozen_final_iteration(arm_dir: Path, gate: dict) -> int | None:
    """Last checkpoint whose trailing segment satisfies the frozen rule."""

    checkpoints = sorted(
        int(path.stem.split("_")[1])
        for path in arm_dir.glob("iter_*.npz")
    )
    spectra: dict[int, dict] = {}
    consecutive = 0
    first_stable = None
    for k_end in checkpoints:
        if k_end - SEGMENT < 0 or (k_end - SEGMENT) not in checkpoints:
            continue
        end = np.load(arm_dir / f"iter_{k_end:04d}.npz", allow_pickle=False)
        start = np.load(
            arm_dir / f"iter_{k_end - SEGMENT:04d}.npz", allow_pickle=False
        )
        temperature_start = np.asarray(start["temperature"], dtype=np.float64)
        temperature_end = np.asarray(end["temperature"], dtype=np.float64)
        mass_start = np.asarray(start["column_mass"], dtype=np.float64)
        mass_end = np.asarray(end["column_mass"], dtype=np.float64)
        spectrum_start = _load_spectrum_npz(
            _spectrum_cached(
                arm_dir / f"iter_{k_end - SEGMENT:04d}.npz",
                arm_dir / f"spectrum_k{k_end - SEGMENT:04d}.npz"
            )
        )
        spectrum_end = _load_spectrum_npz(
            _spectrum_cached(
                arm_dir / f"iter_{k_end:04d}.npz",
                arm_dir / f"spectrum_k{k_end:04d}.npz"
            )
        )
        changes = {
            "tiO": {
                "normalized_flux": _absolute_stats(
                    spectrum_end["normalized_flux"],
                    spectrum_start["normalized_flux"],
                )["max"],
                "flux_total": _continuum_scaled_stats(
                    spectrum_end["flux_total"],
                    spectrum_start["flux_total"],
                    spectrum_start["flux_continuum"],
                )["max"],
                "flux_continuum": _relative_stats(
                    spectrum_end["flux_continuum"],
                    spectrum_start["flux_continuum"],
                )["max"],
            },
            "temperature_p95": float(
                np.percentile(
                    np.abs(temperature_end - temperature_start)
                    / temperature_start,
                    95.0,
                )
            ),
            "mass_dex_p95": float(
                np.percentile(
                    np.abs(np.log10(mass_end) - np.log10(mass_start)), 95.0
                )
            ),
        }
        rows = [
            json.loads(line)
            for line in _residual_rows(arm_dir)
        ]
        flux_pass = True
        thresholds = gate["thresholds"]
        for k in range(k_end - SEGMENT + 1, k_end + 1):
            row = next((r for r in rows if r["iteration"] == k), None)
            if row is None:
                flux_pass = False
                break
            if not (
                row["flux_error_p95_percent"]
                <= float(thresholds["p95_absolute_flux_error_percent"])
                and row["flux_error_median_percent"]
                <= float(thresholds["median_absolute_flux_error_percent"])
                and row["flux_error_max_percent"]
                <= float(thresholds["maximum_absolute_flux_error_percent"])
            ):
                flux_pass = False
                break
        if _segment_stable(changes, flux_pass):
            consecutive += 1
            if consecutive >= STABLE_SEGMENTS_REQUIRED and first_stable is None:
                first_stable = k_end
        else:
            consecutive = 0
    return first_stable


def _residual_rows(arm_dir: Path) -> list[str]:
    """The released continuation writes no residual log; per-iteration flux
    residuals are taken from the solver-side records when present."""

    path = arm_dir / "iterations.jsonl"
    if path.is_file():
        return path.read_text().splitlines()
    return []


def _gate_metrics(spectrum_a: dict, spectrum_b: dict) -> dict[str, float]:
    return {
        "normalized_flux": _absolute_stats(
            spectrum_a["normalized_flux"], spectrum_b["normalized_flux"]
        )["max"],
        "flux_total": _continuum_scaled_stats(
            spectrum_a["flux_total"],
            spectrum_b["flux_total"],
            spectrum_b["flux_continuum"],
        )["max"],
        "flux_continuum": _relative_stats(
            spectrum_a["flux_continuum"], spectrum_b["flux_continuum"]
        )["max"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--truth-dir", type=Path, default=DEFAULT_TRUTH_DIR)
    parser.add_argument(
        "--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR
    )
    parser.add_argument("--flux-gate", type=Path, default=DEFAULT_FLUX_GATE)
    args = parser.parse_args(argv)
    _set_single_thread_environment()
    gate = json.loads(args.flux_gate.read_text())

    corpus = load_validation_rows(args.corpus)
    probe_dir = args.result_root / "checkpoints"
    spectra_dir = args.result_root / "spectra"
    probe_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"campaign": CAMPAIGN, "stars": {}}
    for node_id, labels_row, truth_rel in corpus:
        teff = float(labels_row[0])
        star = f"t{int(teff):04d}"
        labels = _labels_for(teff, float(labels_row[1]), float(labels_row[2]))
        truth_product = Path(truth_rel)
        if not truth_product.is_absolute():
            truth_product = REPO_ROOT / truth_product
        candidate_product = args.candidate_dir / Path(truth_rel).name
        star_root = probe_dir / star
        star_report: dict[str, Any] = {
            "node_id": node_id,
            "candidate_product": str(candidate_product),
            "truth_product": str(truth_product),
        }
        for arm, product in (
            ("candidate", candidate_product),
            ("truth", truth_product),
        ):
            arm_dir = star_root / arm
            if not (arm_dir / "iter_0030.npz").is_file():
                mass, temperature = _load_mt(product)
                start = _reconstruct_from_mt(labels, mass, temperature)
                _continue_released(
                    start_atmosphere=start,
                    arm_dir=arm_dir,
                    extra_iterations=EXTRA_ITERATIONS,
                )
            star_report[arm] = {
                "frozen_iteration": _frozen_final_iteration(arm_dir, gate)
            }
        report["stars"][star] = star_report
        print(f"{star}: {json.dumps(star_report)}", flush=True)

    for star, star_report in report["stars"].items():
        metrics = {}
        for arm in ("candidate", "truth"):
            frozen = star_report[arm]["frozen_iteration"]
            fallback = 30 if frozen is None else frozen
            product = (
                args.result_root
                / "checkpoints"
                / star
                / arm
                / f"iter_{fallback:04d}.npz"
            )
            spectrum = _load_spectrum_npz(
                _synthesize_one(
                    product,
                    spectra_dir / f"{star}_{arm}_final.npz",
                    wavelength_start_nm=WINDOW_NM[0],
                    wavelength_end_nm=WINDOW_NM[1],
                    resolution=RESOLUTION,
                    molecular_lines=True,
                    device=None,
                    dtype="float64",
                )
            )
            metrics[arm] = spectrum
            star_report[arm]["final_product"] = str(product)
        star_report["gate_relaxed"] = _gate_metrics(
            metrics["candidate"], metrics["truth"]
        )
        original_candidate = _load_spectrum_npz(
            _spectrum_cached(
                Path(star_report["candidate_product"]),
                spectra_dir / f"{star}_candidate_v3.npz",
            )
        )
        original_truth = _load_spectrum_npz(
            _spectrum_cached(
                Path(star_report["truth_product"]),
                spectra_dir / f"{star}_truth_v3.npz",
            )
        )
        star_report["gate_v3"] = _gate_metrics(
            original_candidate, original_truth
        )

    passed = sum(
        1
        for star_report in report["stars"].values()
        if max(star_report["gate_relaxed"].values()) <= 5.0e-3
    )
    report["summary"] = {
        "gate_bar": 5.0e-3,
        "passed_relaxed": passed,
        "stars": len(report["stars"]),
    }
    _write_json(args.result_root / "truth_final.json", report)
    print(json.dumps(report["summary"], indent=2))
    for star, star_report in report["stars"].items():
        print(
            star,
            "relaxed:",
            json.dumps(star_report["gate_relaxed"], sort_keys=True),
            "| v3:",
            json.dumps(star_report["gate_v3"], sort_keys=True),
        )
    return 0


def load_validation_rows(corpus: Path):
    """Validation rows as (node_id, labels_row, truth product path)."""

    with np.load(corpus, allow_pickle=False) as data:
        roles = np.asarray(data["roles"]).astype(str)
        index = np.flatnonzero(roles == "validation")
        node_ids = np.asarray(data["node_ids"]).astype(str)[index]
        labels = np.asarray(data["labels"], dtype=np.float64)[index]
        products = np.asarray(data["source_product_paths"]).astype(str)[index]
    return list(zip(node_ids, labels, products))


if __name__ == "__main__":
    raise SystemExit(main())
