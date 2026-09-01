"""Evaluate a cool-star emulator directly inside the unchanged ATLAS solver.

Only opened validation rows from ``cool_truth_corpus.npz`` are used.  The
three training seeds are combined by the coordinate-wise median in physical
``(m,T)`` space, reconstructed into a six-field seed, and solved with the
unchanged 30-iteration policy.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np

from reduced_state.emulator import load_physical_checkpoint, predict_physical_state

from .cool_star_step_test import (
    ITERATION_CAP,
    PRIMARY_ITERATION_CAP,
    TrackSpec,
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from .m_star_bootstrap_v1 import (
    PATH_COLUMN_MASS_P95_DEX_LIMIT,
    PATH_TEMPERATURE_P95_LIMIT,
    _passes_flux_gate,
    _product_consistency,
    _write_json,
)
from .train_mstar_physical_v1 import load_cool_corpus


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COOL_CORPUS = (
    REPO_ROOT / "results" / "m_star_emulator_v1" / "cool_truth_corpus.npz"
)
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "m_star_emulator_v1"
DEFAULT_FLUX_GATE = (
    REPO_ROOT / "results" / "m_star_emulator_v1" / "flux_gate.json"
)
DEFAULT_OUT = (
    REPO_ROOT / "results" / "m_star_emulator_v1" / "candidate_validation"
)

PROFILE_TEMPERATURE_P95_LIMIT = 3.0e-3
PROFILE_MASS_P95_DEX_LIMIT = 7.7e-3
SOLVER_CONVERGENCE_FRACTION_MIN = 0.95
CLASS_CONVERGENCE_FRACTION_MIN = 0.90
WITHIN_15_FRACTION_MIN = 0.80


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _track_from_labels(labels: np.ndarray) -> TrackSpec:
    return TrackSpec(
        log_surface_gravity=float(labels[1]),
        metallicity=float(labels[2]),
        alpha_enhancement=float(labels[3]),
        carbon_enhancement=0.0,
        microturbulence_km_s=float(labels[4]),
    )


def _worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        node_id,
        labels_array,
        predicted_mass,
        predicted_temperature,
        truth_product,
        result_root_text,
        flux_gate,
        iteration_cap,
        campaign,
    ) = payload
    _set_single_thread_environment()
    labels_array = np.asarray(labels_array, dtype=np.float64)
    track = _track_from_labels(labels_array)
    labels = track.labels(float(labels_array[0]))
    stellar_class = "dwarf" if labels.log_surface_gravity >= 3.5 else "giant"
    try:
        initial = _reconstruct_from_mt(
            labels,
            np.asarray(predicted_mass, dtype=np.float64),
            np.asarray(predicted_temperature, dtype=np.float64),
        )
        record, _state = _solve_attempt(
            track=track,
            method=f"{campaign}_direct",
            schedule="candidate_direct",
            source_temperature=None,
            target_labels=labels,
            initial_atmosphere=initial,
            product_dir=Path(result_root_text) / "products",
            iteration_cap=int(iteration_cap),
        )
    except Exception as exc:  # noqa: BLE001 - validation outcome
        record = {
            "survives_solver": False,
            "iterations": None,
            "product_path": None,
            "state_quality": {"valid": False},
            "status": "candidate_initialization_or_solver_exception",
            "error": f"{type(exc).__name__}: {exc}",
        }
    iterations = record.get("iterations")
    flux = _passes_flux_gate(record, flux_gate)
    truth_consistency = _product_consistency(
        truth_product,
        record.get("product_path"),
    )
    return {
        "node_id": str(node_id),
        "class": stellar_class,
        "labels": labels.as_kwargs(),
        "solver": record,
        "solver_converged": bool(record.get("survives_solver")),
        "within_15_iterations": bool(
            record.get("survives_solver")
            and iterations is not None
            and int(iterations) <= PRIMARY_ITERATION_CAP
        ),
        "flux_gate": flux,
        "truth_consistency": truth_consistency,
        "passes": bool(
            record.get("survives_solver")
            and flux["passes"]
            and truth_consistency["passes"]
        ),
    }


def profile_metrics(
    predicted_mass: np.ndarray,
    predicted_temperature: np.ndarray,
    truth_mass: np.ndarray,
    truth_temperature: np.ndarray,
) -> dict[str, Any]:
    temperature_error = np.abs(predicted_temperature - truth_temperature) / truth_temperature
    mass_error = np.abs(np.log10(predicted_mass) - np.log10(truth_mass))
    return {
        "temperature_relative": {
            "median": float(np.median(temperature_error)),
            "p95": float(np.percentile(temperature_error, 95.0)),
            "max": float(np.max(temperature_error)),
        },
        "column_mass_dex": {
            "median": float(np.median(mass_error)),
            "p95": float(np.percentile(mass_error, 95.0)),
            "max": float(np.max(mass_error)),
        },
        "monotonicity_violations": int(
            np.any(np.diff(predicted_mass, axis=1) <= 0.0, axis=1).sum()
        ),
    }


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cool-corpus", type=Path, default=DEFAULT_COOL_CORPUS)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--seeds", default="20260831,20260901,20260902")
    parser.add_argument("--flux-gate", type=Path, default=DEFAULT_FLUX_GATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iteration-cap", type=int, default=ITERATION_CAP)
    parser.add_argument("--campaign", default="m_star_emulator_v1")
    args = parser.parse_args(argv)

    cool = load_cool_corpus(args.cool_corpus)
    validation = np.flatnonzero(cool["roles"] == "validation")
    if not len(validation):
        raise SystemExit("no opened cool validation rows")
    labels = cool["labels"][validation]
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    masses = []
    temperatures = []
    checkpoints = []
    for seed in seeds:
        checkpoint = args.checkpoint_dir / f"checkpoint_mstar_seed{seed}.pt"
        model, standardization, meta = load_physical_checkpoint(checkpoint)
        if meta.get("sealed_cool_rows_loaded") is not False:
            raise SystemExit(f"checkpoint {checkpoint} lacks the sealed-row exclusion")
        mass, temperature = predict_physical_state(model, standardization, labels)
        masses.append(mass)
        temperatures.append(temperature)
        checkpoints.append(
            {
                "seed": seed,
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
            }
        )
    predicted_mass = np.median(np.stack(masses, axis=0), axis=0)
    predicted_temperature = np.median(np.stack(temperatures, axis=0), axis=0)
    profile = profile_metrics(
        predicted_mass,
        predicted_temperature,
        cool["column_mass"][validation],
        cool["temperature"][validation],
    )
    flux_gate = json.loads(args.flux_gate.read_text())
    if not flux_gate.get("frozen"):
        raise SystemExit("flux gate is not frozen")
    args.out.mkdir(parents=True, exist_ok=True)

    payloads = [
        (
            cool["node_ids"][index],
            cool["labels"][index],
            predicted_mass[row],
            predicted_temperature[row],
            str(cool["source_product_paths"][index]),
            str(args.out),
            flux_gate,
            int(args.iteration_cap),
            str(args.campaign),
        )
        for row, index in enumerate(validation)
    ]
    if args.workers <= 1:
        records = [_worker(payload) for payload in payloads]
    else:
        # The parent performs three torch forwards before solver workers are
        # created. Spawn avoids inheriting an initialized OpenMP state.
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            records = list(pool.map(_worker, payloads))

    total = len(records)
    converged = sum(row["solver_converged"] for row in records)
    within_15 = sum(row["within_15_iterations"] for row in records)
    flux_pass = sum(row["flux_gate"]["passes"] for row in records)
    truth_pass = sum(row["truth_consistency"]["passes"] for row in records)
    class_metrics = {}
    for stellar_class in ("giant", "dwarf"):
        selected = [row for row in records if row["class"] == stellar_class]
        class_metrics[stellar_class] = {
            "count": len(selected),
            "converged": sum(row["solver_converged"] for row in selected),
            "convergence_fraction": _fraction(
                sum(row["solver_converged"] for row in selected),
                len(selected),
            ),
        }
    gates = {
        "profile_temperature_p95": (
            profile["temperature_relative"]["p95"]
            <= PROFILE_TEMPERATURE_P95_LIMIT
        ),
        "profile_mass_p95": (
            profile["column_mass_dex"]["p95"] <= PROFILE_MASS_P95_DEX_LIMIT
        ),
        "profile_monotonic": profile["monotonicity_violations"] == 0,
        "solver_convergence": (
            _fraction(converged, total) >= SOLVER_CONVERGENCE_FRACTION_MIN
        ),
        "each_class_convergence": all(
            values["count"] > 0
            and values["convergence_fraction"] >= CLASS_CONVERGENCE_FRACTION_MIN
            for values in class_metrics.values()
        ),
        "within_15": _fraction(within_15, total) >= WITHIN_15_FRACTION_MIN,
        "flux_quality": flux_pass == total,
        "truth_fixed_point": truth_pass == total,
    }
    summary = {
        "campaign": str(args.campaign),
        "status": "pass_opened_validation" if all(gates.values()) else "fail_opened_validation",
        "cool_corpus": str(args.cool_corpus),
        "cool_corpus_sha256": _sha256(args.cool_corpus),
        "flux_gate": str(args.flux_gate),
        "flux_gate_sha256": _sha256(args.flux_gate),
        "checkpoints": checkpoints,
        "ensemble_policy": "coordinate-wise median of three physical-state predictions",
        "validation_count": total,
        "profile_metrics": profile,
        "solver_converged": converged,
        "solver_convergence_fraction": _fraction(converged, total),
        "within_15": within_15,
        "within_15_fraction": _fraction(within_15, total),
        "flux_pass": flux_pass,
        "truth_consistency_pass": truth_pass,
        "class_metrics": class_metrics,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
        "records": records,
        "sealed_cool_rows_loaded": False,
        "production_routing_changed": False,
        "existing_sealed_holdout_opened": False,
    }
    _write_json(args.out / "validation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
