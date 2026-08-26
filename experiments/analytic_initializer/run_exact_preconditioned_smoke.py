"""Compare direct grey solving with a one-exact-iteration grey preconditioner."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import queue as queue_module
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from bench import environment as _environment  # noqa: F401

from bench.run_reference import _atmosphere_is_finite, _solver_config
from experiments.analytic_initializer.discovery import DEFAULT_CORPUS, load_strict_truth
from experiments.analytic_initializer.no_emulator_bridge import analytic_seed_model
from experiments.analytic_initializer.run_physical_handoff_audit import _grey_seed
from payne_zero_atmosphere.runner import run_atmosphere_model


DEFAULT_INDICES = (2891, 6896)
DEFAULT_ARMS = ("direct_grey", "preconditioned_grey")
DEFAULT_OUTPUT = Path(
    "results/analytic_initializer/exact_preconditioned_smoke.json"
)


def _label_record(labels: np.ndarray, index: int, arm: str) -> dict[str, object]:
    names = (
        "effective_temperature",
        "log_surface_gravity",
        "metallicity",
        "alpha_enhancement",
        "microturbulence_km_s",
    )
    return {
        "corpus_index": int(index),
        "arm": str(arm),
        **{name: float(labels[position]) for position, name in enumerate(names)},
    }


def _run_exact(seed, *, iterations: int, temperature_correction_damping: float):
    config = _solver_config(
        seed,
        iterations_per_trial=int(iterations),
        structured_atmosphere_path=None,
        debug_state_path=None,
    )
    return run_atmosphere_model(
        replace(
            config,
            temperature_correction_damping=float(temperature_correction_damping),
        )
    )


def _state_info(atmosphere) -> dict[str, object]:
    finite = bool(_atmosphere_is_finite(atmosphere))
    positive = bool(
        np.all(atmosphere.temperature > 0.0)
        and np.all(atmosphere.column_mass > 0.0)
    )
    return {
        "finite": finite,
        "positive_temperature_mass": positive,
        "temperature_range": [
            float(np.min(atmosphere.temperature)),
            float(np.max(atmosphere.temperature)),
        ],
        "mass_range": [
            float(np.min(atmosphere.column_mass)),
            float(np.max(atmosphere.column_mass)),
        ],
    }


def _solve_one(
    labels: np.ndarray,
    index: int,
    arm: str,
    *,
    production_iterations: int,
    temperature_correction_damping: float,
) -> dict[str, object]:
    start = time.perf_counter()
    record = _label_record(labels, index, arm)
    record["temperature_correction_damping"] = float(temperature_correction_damping)
    try:
        _tau, grey_seed, grey_info = _grey_seed(labels)
        record["grey_seed"] = grey_info
        if arm == "direct_grey":
            solver_start = time.perf_counter()
            result = _run_exact(
                grey_seed,
                iterations=int(production_iterations),
                temperature_correction_damping=float(temperature_correction_damping),
            )
            record.update(
                {
                    "production_seconds": float(time.perf_counter() - solver_start),
                    "production_iterations": int(result.iterations_completed),
                    "converged": bool(result.converged),
                    "finite_final_state": bool(_atmosphere_is_finite(result.atmosphere)),
                    "final_state": _state_info(result.atmosphere),
                    "diagnostics": result.diagnostics,
                }
            )
        elif arm == "preconditioned_grey":
            first_start = time.perf_counter()
            first = _run_exact(
                grey_seed,
                iterations=1,
                temperature_correction_damping=float(temperature_correction_damping),
            )
            first_seconds = float(time.perf_counter() - first_start)
            record["preconditioner_seconds"] = first_seconds
            record["preconditioner_iterations"] = int(first.iterations_completed)
            record["preconditioner_state"] = _state_info(first.atmosphere)
            if not (
                _atmosphere_is_finite(first.atmosphere)
                and np.all(first.atmosphere.temperature > 0.0)
                and np.all(first.atmosphere.column_mass > 0.0)
            ):
                record.update(
                    {
                        "converged": False,
                        "finite_final_state": bool(_atmosphere_is_finite(first.atmosphere)),
                        "production_iterations": 0,
                        "solver_outcome": "invalid_preconditioner_state",
                    }
                )
            else:
                production_start = time.perf_counter()
                result = _run_exact(
                    first.atmosphere,
                    iterations=int(production_iterations),
                    temperature_correction_damping=float(
                        temperature_correction_damping
                    ),
                )
                record.update(
                    {
                        "production_seconds": float(
                            time.perf_counter() - production_start
                        ),
                        "production_iterations": int(result.iterations_completed),
                        "converged": bool(result.converged),
                        "finite_final_state": bool(
                            _atmosphere_is_finite(result.atmosphere)
                        ),
                        "final_state": _state_info(result.atmosphere),
                        "diagnostics": result.diagnostics,
                        "total_solver_iterations": int(
                            first.iterations_completed + result.iterations_completed
                        ),
                    }
                )
        else:
            raise ValueError(f"unknown arm: {arm}")
        record.setdefault("solver_outcome", "converged" if record["converged"] else "not_converged")
    except Exception as exc:  # noqa: BLE001 - the failure is an audit result
        record.update(
            {
                "solver_outcome": "error",
                "converged": False,
                "finite_final_state": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    record["seconds"] = float(time.perf_counter() - start)
    return record


def _worker(payload: tuple[np.ndarray, int, str, int, float], result_queue) -> None:
    labels, index, arm, production_iterations, temperature_correction_damping = payload
    result_queue.put(
        _solve_one(
            labels,
            index,
            arm,
            production_iterations=int(production_iterations),
            temperature_correction_damping=float(temperature_correction_damping),
        )
    )


def _run_with_timeout(
    labels: np.ndarray,
    index: int,
    arm: str,
    *,
    timeout: float,
    production_iterations: int,
    temperature_correction_damping: float,
) -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_worker,
        args=(
            (
                labels,
                index,
                arm,
                int(production_iterations),
                float(temperature_correction_damping),
            ),
            result_queue,
        ),
    )
    process.start()
    try:
        return result_queue.get(timeout=float(timeout))
    except queue_module.Empty:
        return {
            **_label_record(labels, index, arm),
            "solver_outcome": "timeout",
            "converged": False,
            "finite_final_state": False,
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
    parser.add_argument("--indices", type=int, nargs="+", default=list(DEFAULT_INDICES))
    parser.add_argument("--arms", nargs="+", choices=DEFAULT_ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--production-iterations", type=int, default=15)
    parser.add_argument("--temperature-correction-damping", type=float, default=1.0)
    args = parser.parse_args(argv)

    corpus = load_strict_truth(DEFAULT_CORPUS)
    records: list[dict[str, object]] = []
    jsonl_path = args.out.with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index in args.indices:
            for arm in args.arms:
                record = _run_with_timeout(
                    corpus.labels[int(index)],
                    int(index),
                    str(arm),
                    timeout=float(args.timeout),
                    production_iterations=int(args.production_iterations),
                    temperature_correction_damping=float(
                        args.temperature_correction_damping
                    ),
                )
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                print(
                    f"index={index} arm={arm} outcome={record['solver_outcome']} "
                    f"seconds={record['seconds']:.1f}",
                    flush=True,
                )

    result = {
        "candidate": "one_exact_iteration_grey_preconditioner",
        "status": "paired_real_solver_smoke",
        "indices": [int(index) for index in args.indices],
        "arms": [str(arm) for arm in args.arms],
        "production_iterations": int(args.production_iterations),
        "temperature_correction_damping": float(args.temperature_correction_damping),
        "records": records,
        "development_60": "blocked_until_paired_smoke_passes",
        "sealed_holdout": "closed",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
