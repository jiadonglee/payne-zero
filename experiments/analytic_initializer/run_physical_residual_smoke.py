"""Run the bounded real-solver smoke test for the residual initializer."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import queue as queue_module
import time
from pathlib import Path

import numpy as np

from bench import environment as _environment  # noqa: F401

from bench.run_reference import _atmosphere_is_finite, _solver_config
from experiments.analytic_initializer.discovery import DEFAULT_CORPUS, load_strict_truth
from experiments.analytic_initializer.no_emulator_bridge import analytic_seed_model
from experiments.analytic_initializer.physical_residual_initializer import (
    physical_residual_seed,
    resample_residual_seed,
)
from payne_zero_atmosphere.runner import run_atmosphere_model


DEFAULT_INDICES = (2891, 6896)
DEFAULT_OUTPUT = Path("results/analytic_initializer/physical_residual_smoke2.json")


def _labels_record(labels: np.ndarray, index: int) -> dict[str, object]:
    names = (
        "effective_temperature",
        "log_surface_gravity",
        "metallicity",
        "alpha_enhancement",
        "microturbulence_km_s",
    )
    return {
        "corpus_index": int(index),
        **{name: float(labels[position]) for position, name in enumerate(names)},
    }


def _solve_one(labels: np.ndarray, index: int, *, seed_only: bool) -> dict[str, object]:
    start = time.perf_counter()
    record = _labels_record(labels, index)
    try:
        seed_start = time.perf_counter()
        coarse = physical_residual_seed(labels)
        tau, mass, temperature, opacity = resample_residual_seed(coarse)
        record.update(
            {
                "seed_seconds": float(time.perf_counter() - seed_start),
                "seed_diagnostics": coarse.diagnostics,
                "seed_finite": bool(
                    all(
                        np.all(np.isfinite(values))
                        for values in (mass, temperature, opacity)
                    )
                ),
                "seed_monotone_mass": bool(np.all(np.diff(mass) > 0.0)),
                "seed_monotone_temperature": bool(
                    np.all(np.diff(temperature) > 0.0)
                ),
                "seed_temperature_range": [
                    float(np.min(temperature)),
                    float(np.max(temperature)),
                ],
                "seed_mass_range": [float(np.min(mass)), float(np.max(mass))],
            }
        )
        if seed_only:
            record.update(
                {
                    "solver_outcome": "not_run",
                    "converged": False,
                    "finite_final_state": None,
                    "iterations_completed": None,
                }
            )
        else:
            seed = analytic_seed_model(
                labels, mass, temperature, np.log10(opacity), tau
            )
            solver_start = time.perf_counter()
            result = run_atmosphere_model(
                _solver_config(
                    seed,
                    iterations_per_trial=15,
                    structured_atmosphere_path=None,
                    debug_state_path=None,
                )
            )
            record.update(
                {
                    "solver_seconds": float(time.perf_counter() - solver_start),
                    "solver_outcome": (
                        "converged" if result.converged else "not_converged"
                    ),
                    "converged": bool(result.converged),
                    "finite_final_state": bool(
                        _atmosphere_is_finite(result.atmosphere)
                    ),
                    "iterations_completed": int(result.iterations_completed),
                    "deep_layer_relative_temperature_change": float(
                        result.diagnostics["deep_layer_relative_temperature_change"]
                    ),
                }
            )
    except Exception as exc:  # noqa: BLE001 - each row is an auditable result
        record.update(
            {
                "solver_outcome": "error",
                "converged": False,
                "finite_final_state": False,
                "iterations_completed": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    record["seconds"] = float(time.perf_counter() - start)
    return record


def _worker(payload: tuple[np.ndarray, int, bool], result_queue) -> None:
    labels, index, seed_only = payload
    result_queue.put(_solve_one(labels, index, seed_only=seed_only))


def _run_with_timeout(
    labels: np.ndarray,
    index: int,
    *,
    timeout: float,
    seed_only: bool,
) -> dict[str, object]:
    if seed_only:
        return _solve_one(labels, index, seed_only=True)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_worker, args=((labels, index, False), result_queue))
    process.start()
    try:
        return result_queue.get(timeout=float(timeout))
    except queue_module.Empty:
        return {
            **_labels_record(labels, index),
            "solver_outcome": "timeout",
            "converged": False,
            "finite_final_state": False,
            "iterations_completed": None,
            "seconds": float(timeout),
        }
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=15.0)
        result_queue.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed-only", action="store_true")
    parser.add_argument("--indices", type=int, nargs="+", default=list(DEFAULT_INDICES))
    args = parser.parse_args(argv)

    corpus = load_strict_truth(DEFAULT_CORPUS)
    records: list[dict[str, object]] = []
    jsonl_path = args.out.with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for position, index in enumerate(args.indices, start=1):
            record = _run_with_timeout(
                corpus.labels[int(index)],
                int(index),
                timeout=float(args.timeout),
                seed_only=bool(args.seed_only),
            )
            records.append(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"[{position}/{len(args.indices)}] index={index} "
                f"outcome={record['solver_outcome']} "
                f"iters={record['iterations_completed']}",
                flush=True,
            )

    result = {
        "candidate": "low_dimensional_physical_residual_continuum_only",
        "status": "seed_only" if args.seed_only else "smoke_not_production",
        "corpus": str(corpus.path),
        "star_indices": [int(index) for index in args.indices],
        "star_count": len(records),
        "converged_count": int(sum(bool(row.get("converged")) for row in records)),
        "finite_count": int(
            sum(row.get("finite_final_state") is True for row in records)
        ),
        "timeout_count": int(
            sum(row.get("solver_outcome") == "timeout" for row in records)
        ),
        "records": records,
        "gate": {
            "development_60": "blocked_until_a_bounded_smoke_passes",
            "sealed_holdout": "closed",
            "note": "prototype residual reduction is not a production result",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("converged_count", "finite_count", "timeout_count")}, sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
