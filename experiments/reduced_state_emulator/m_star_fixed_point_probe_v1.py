"""Fixed-point continuation probe at the two spectrally-failing M giants.

For ``t3850`` and ``t3900`` (TiO-gate misses) plus ``t3950`` as the control,
continue both the emulator endpoint and the truth atmosphere for ten extra
iterations with the stopping rule released (physics and thresholds untouched),
recording checkpoints at iterations 0, 1, 3, 5, 10: cross-arm temperature and
column-mass differences, each arm's own flux residual and drift, and the
TiO-window spectral error against the original truth.  The three outcomes
this separates: loose stopping (errors fall), genuine fixed-point offset
(errors stay flat), an unstable truth reference (both arms wander).
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
from .train_mstar_physical_v1 import load_cool_corpus

from bench.labels import StellarLabels  # noqa: E402
from bench.run_reference import _solver_config  # noqa: E402
from payne_zero_atmosphere.runner import run_atmosphere_model  # noqa: E402
from payne_zero_atmosphere.synthesis_bridge import (  # noqa: E402
    save_product_structured_atmosphere,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_fixed_point_probe_v1"
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

STARS = (3850.0, 3900.0, 3950.0)
DEFAULT_CHECKPOINTS = (0, 1, 3, 5, 10, 15, 20)
DEFAULT_EXTRA_ITERATIONS = 20
WINDOW_NM = (665.0, 667.0)
RESOLUTION = 20000.0


def _product_name(teff: float, labels_row: np.ndarray) -> str:
    return (
        f"t{teff:07.1f}_g{labels_row[1]:+05.2f}_m{labels_row[2]:+05.2f}"
        f"_a+0.00_x{labels_row[4]:.2f}.npz"
    )


def _labels_from_row(row: np.ndarray) -> StellarLabels:
    track = TrackSpec(
        log_surface_gravity=float(row[1]),
        metallicity=float(row[2]),
        alpha_enhancement=0.0,
        carbon_enhancement=0.0,
        microturbulence_km_s=float(row[4]),
    )
    return track.labels(float(row[0]))


def _continue_with_hook(
    *,
    labels: StellarLabels,
    initial_atmosphere,
    checkpoint_dir: Path,
    extra_iterations: int,
    checkpoints: tuple[int, ...],
) -> dict[str, Any]:
    """Run ``extra_iterations`` updates from ``initial_atmosphere``.

    The stopping rule is released; the iteration hook appends one JSONL line
    per iteration (flux residual, update size, cumulative drift) and writes a
    structured product at each checkpoint iteration so the checkpoint spectra
    can be synthesized from the same fixed-column path as the gate.
    """

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start_temperature = np.asarray(
        initial_atmosphere.temperature, dtype=np.float64
    )
    start_column_mass = np.asarray(
        initial_atmosphere.column_mass, dtype=np.float64
    )
    save_product_structured_atmosphere(
        _clone_atmosphere(initial_atmosphere),
        checkpoint_dir / "iter_0000.npz",
        device="cpu",
        dtype="float64",
    )
    residual_path = checkpoint_dir / "iterations.jsonl"
    residual_handle = residual_path.open("w")

    previous_temperature = start_temperature.copy()

    def hook(iteration_index, setup, step):
        nonlocal previous_temperature
        post_temperature = np.asarray(
            step.remapped.atmosphere.temperature, dtype=np.float64
        )
        post_column_mass = np.asarray(
            step.remapped.atmosphere.column_mass, dtype=np.float64
        )
        flux_error = np.asarray(
            step.remapped.finalization.temperature_correction_result.flux_error_percent,
            dtype=np.float64,
        )
        update = np.abs(post_temperature - previous_temperature) / previous_temperature
        record = {
            "iteration": int(iteration_index),
            "flux_error_p95_percent": float(np.percentile(np.abs(flux_error), 95.0)),
            "flux_error_median_percent": float(np.percentile(np.abs(flux_error), 50.0)),
            "flux_error_max_percent": float(np.max(np.abs(flux_error))),
            "update_temperature_relative_max": float(np.max(update)),
            "update_temperature_relative_p95": float(np.percentile(update, 95.0)),
            "drift_temperature_relative_max": float(
                np.max(np.abs(post_temperature - start_temperature) / start_temperature)
            ),
            "drift_column_mass_dex_max": float(
                np.max(
                    np.abs(
                        np.log10(post_column_mass) - np.log10(start_column_mass)
                    )
                )
            ),
        }
        residual_handle.write(json.dumps(record, sort_keys=True) + "\n")
        residual_handle.flush()
        if int(iteration_index) in checkpoints:
            save_product_structured_atmosphere(
                _clone_atmosphere(step.remapped.atmosphere),
                checkpoint_dir / f"iter_{int(iteration_index):04d}.npz",
                device="cpu",
                dtype="float64",
            )
        previous_temperature = post_temperature
        return record

    config = dataclasses.replace(
        _solver_config(
            _clone_atmosphere(initial_atmosphere),
            iterations_per_trial=int(extra_iterations),
            structured_atmosphere_path=None,
            debug_state_path=None,
        ),
        enable_convergence_stop=False,
    )
    result = run_atmosphere_model(config, after_iteration_hook=hook)
    residual_handle.close()
    return {
        "iterations_completed": int(result.iterations_completed),
        "converged": bool(result.converged),
        "residual_path": str(residual_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--truth-dir", type=Path, default=DEFAULT_TRUTH_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--extra-iterations", type=int, default=DEFAULT_EXTRA_ITERATIONS
    )
    parser.add_argument(
        "--checkpoints",
        default=",".join(str(k) for k in DEFAULT_CHECKPOINTS),
    )
    args = parser.parse_args(argv)
    checkpoints = tuple(sorted(int(k) for k in args.checkpoints.split(",")))
    _set_single_thread_environment()
    args.result_root.mkdir(parents=True, exist_ok=True)

    corpus = load_cool_corpus(args.corpus)
    probe_dir = args.result_root / "checkpoints"
    synthesis_dir = args.result_root / "spectra"
    probe_dir.mkdir(parents=True, exist_ok=True)
    synthesis_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "extra_iterations": int(args.extra_iterations),
        "checkpoints": list(checkpoints),
        "window_nm": list(WINDOW_NM),
        "stars": {},
    }
    for teff in STARS:
        index = None
        for probe_index, node in enumerate(corpus["node_ids"]):
            track_id, _, temperature = str(node).rpartition("_t")
            if abs(float(temperature) - teff) < 0.5 and str(node).startswith(
                "g+2.00_m+0.00"
            ):
                index = probe_index
                break
        if index is None:
            raise SystemExit(f"FAIL_STOP: no corpus row for t{int(teff)}")
        labels_row = corpus["labels"][index]
        node_id = str(corpus["node_ids"][index])
        labels = _labels_from_row(labels_row)
        truth_product = args.truth_dir / _product_name(teff, labels_row)
        candidate_product = args.candidate_dir / _product_name(teff, labels_row)
        for product in (truth_product, candidate_product):
            if not product.is_file():
                raise SystemExit(f"FAIL_STOP: missing {product}")
        star_root = probe_dir / f"t{int(teff):04d}"
        star_report = {"node_id": node_id, "arms": {}}

        for arm, product in (
            ("candidate", candidate_product),
            ("truth", truth_product),
        ):
            mass, temperature = _load_mt(product)
            start = _reconstruct_from_mt(labels, mass, temperature)
            star_report["arms"][arm] = _continue_with_hook(
                labels=labels,
                initial_atmosphere=start,
                checkpoint_dir=star_root / arm,
                extra_iterations=args.extra_iterations,
                checkpoints=checkpoints,
            )
        report["stars"][f"t{int(teff):04d}"] = star_report
        print(f"t{int(teff)}: continuations done", flush=True)

    for teff in STARS:
        star_root = probe_dir / f"t{int(teff):04d}"
        star_report = report["stars"][f"t{int(teff):04d}"]
        spectra: dict[tuple[str, int], dict] = {}
        for arm in ("candidate", "truth"):
            for k in CHECKPOINTS:
                spectrum_path = (
                    synthesis_dir / f"t{int(teff):04d}_{arm}_k{k:02d}.npz"
                )
                if not spectrum_path.is_file():
                    _synthesize_one(
                        star_root / arm / f"iter_{k:04d}.npz",
                        spectrum_path,
                        wavelength_start_nm=WINDOW_NM[0],
                        wavelength_end_nm=WINDOW_NM[1],
                        resolution=RESOLUTION,
                        molecular_lines=True,
                        device=None,
                        dtype="float64",
                    )
                spectra[(arm, k)] = _load_spectrum_npz(spectrum_path)

        star_report["spectral"] = {}
        for k in checkpoints:
            cross = {
                "normalized_flux": _absolute_stats(
                    spectra[("candidate", k)]["normalized_flux"],
                    spectra[("truth", 0)]["normalized_flux"],
                ),
                "flux_total": _continuum_scaled_stats(
                    spectra[("candidate", k)]["flux_total"],
                    spectra[("truth", 0)]["flux_total"],
                    spectra[("truth", 0)]["flux_continuum"],
                ),
                "flux_continuum": _relative_stats(
                    spectra[("candidate", k)]["flux_continuum"],
                    spectra[("truth", 0)]["flux_continuum"],
                ),
            }
            live = {
                "normalized_flux": _absolute_stats(
                    spectra[("candidate", k)]["normalized_flux"],
                    spectra[("truth", k)]["normalized_flux"],
                ),
                "flux_total": _continuum_scaled_stats(
                    spectra[("candidate", k)]["flux_total"],
                    spectra[("truth", k)]["flux_total"],
                    spectra[("truth", k)]["flux_continuum"],
                ),
                "flux_continuum": _relative_stats(
                    spectra[("candidate", k)]["flux_continuum"],
                    spectra[("truth", k)]["flux_continuum"],
                ),
            }
            drift = {
                "normalized_flux": _absolute_stats(
                    spectra[("truth", k)]["normalized_flux"],
                    spectra[("truth", 0)]["normalized_flux"],
                ),
            }
            star_report["spectral"][f"k{k}"] = {
                "candidate_vs_original_truth": cross,
                "candidate_vs_live_truth": live,
                "truth_drift_from_original": drift,
            }
        print(f"t{int(teff)}: spectra done", flush=True)

    for teff in STARS:
        star_report = report["stars"][f"t{int(teff):04d}"]
        star_report["profile"] = {}
        starts = {}
        states = {}
        for arm in ("candidate", "truth"):
            start_data = np.load(
                probe_dir / f"t{int(teff):04d}" / arm / "iter_0000.npz",
                allow_pickle=False,
            )
            starts[arm] = np.asarray(
                start_data["temperature"], dtype=np.float64
            )
            states[arm] = {}
            for k in checkpoints:
                data = np.load(
                    probe_dir / f"t{int(teff):04d}" / arm / f"iter_{k:04d}.npz",
                    allow_pickle=False,
                )
                states[arm][k] = {
                    "temperature": np.asarray(
                        data["temperature"], dtype=np.float64
                    ),
                    "column_mass": np.asarray(
                        data["column_mass"], dtype=np.float64
                    ),
                }
        for k in checkpoints:
            candidate, truth = states["candidate"][k], states["truth"][k]
            star_report["profile"][f"k{k}"] = {
                "cross_arm_temperature_relative_p95": float(
                    np.percentile(
                        np.abs(candidate["temperature"] - truth["temperature"])
                        / truth["temperature"],
                        95.0,
                    )
                ),
                "cross_arm_column_mass_dex_p95": float(
                    np.percentile(
                        np.abs(
                            np.log10(candidate["column_mass"])
                            - np.log10(truth["column_mass"])
                        ),
                        95.0,
                    )
                ),
                "candidate_drift_temperature_relative_max": float(
                    np.max(
                        np.abs(candidate["temperature"] - starts["candidate"])
                        / starts["candidate"]
                    )
                ),
                "truth_drift_temperature_relative_max": float(
                    np.max(
                        np.abs(truth["temperature"] - starts["truth"])
                        / starts["truth"]
                    )
                ),
            }

    _write_json(args.result_root / "probe.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "stars"}, indent=2))
    print(f"wrote {args.result_root / 'probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
