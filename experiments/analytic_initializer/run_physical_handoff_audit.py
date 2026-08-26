"""Audit the 32-to-80-layer handoff with one exact line-opacity iteration.

This is deliberately smaller than a convergence smoke test.  It compares a
plain grey seed with the low-dimensional physical-residual seed on the same
stars, runs exactly one unchanged production iteration, and records whether
the 80-layer population, continuum, selected-line slab, and remapped state are
finite.  It is intended for the CPU cluster because the line catalog is large.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import replace
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
from payne_zero_atmosphere.runner import _run_atmosphere_model


DEFAULT_INDICES = (2891, 6896)
DEFAULT_ARMS = ("grey", "physical_residual")
DEFAULT_OUTPUT = Path("results/analytic_initializer/physical_handoff_audit.json")
_LOG_TAU_START = -6.875
_LOG_TAU_STEP = 0.125
_REFERENCE_OPACITY = 0.34


def _grey_seed(labels: np.ndarray):
    values = np.asarray(labels, dtype=np.float64)
    tau = 10.0 ** (_LOG_TAU_START + _LOG_TAU_STEP * np.arange(80))
    temperature = values[0] * (0.75 * (tau + 2.0 / 3.0)) ** 0.25
    mass = tau / _REFERENCE_OPACITY
    return tau, analytic_seed_model(
        values,
        mass,
        temperature,
        np.full(tau.size, np.log10(_REFERENCE_OPACITY)),
        tau,
    ), {
        "seed_source": "analytic_grey",
        "seed_mass_range": [float(mass.min()), float(mass.max())],
        "seed_temperature_range": [float(temperature.min()), float(temperature.max())],
        "seed_monotone_mass": bool(np.all(np.diff(mass) > 0.0)),
        "seed_monotone_temperature": bool(np.all(np.diff(temperature) > 0.0)),
    }


def _physical_residual_seed(labels: np.ndarray, *, max_nfev: int):
    result = physical_residual_seed(labels, max_nfev=int(max_nfev))
    tau, mass, temperature, opacity = resample_residual_seed(result)
    return tau, analytic_seed_model(
        labels,
        mass,
        temperature,
        np.log10(opacity),
        tau,
    ), {
        "seed_source": "low_dimensional_physical_residual",
        "seed_diagnostics": result.diagnostics,
        "seed_mass_range": [float(mass.min()), float(mass.max())],
        "seed_temperature_range": [float(temperature.min()), float(temperature.max())],
        "seed_monotone_mass": bool(np.all(np.diff(mass) > 0.0)),
        "seed_monotone_temperature": bool(np.all(np.diff(temperature) > 0.0)),
    }


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


def _handoff_hook(iteration_index, setup, step):
    """Capture finite/range checks directly at the exact opacity boundary."""

    opacity = step.opacity
    population = opacity.population_state
    runtime = population.runtime_state
    line = np.asarray(opacity.line_opacity.line_mass_absorption_coefficient)
    continuum_absorption = np.asarray(opacity.continuum_absorption)
    continuum_scattering = np.asarray(opacity.continuum_scattering)
    def _array_summary(values: np.ndarray) -> dict[str, object]:
        finite = np.isfinite(values)
        return {
            "shape": list(values.shape),
            "finite": bool(np.all(finite)),
            "positive_count": int(np.count_nonzero(values > 0.0)),
            "min": float(np.nanmin(values)) if values.size else None,
            "max": float(np.nanmax(values)) if values.size else None,
        }

    return {
        "iteration": int(iteration_index),
        "layer_count": int(population.setup.atmosphere.layers),
        "frequency_count": int(opacity.opacity_frequency_hz.size),
        "selected_line_count": int(opacity.line_opacity.selected_line_count),
        "line_slab": _array_summary(line),
        "continuum_absorption": _array_summary(continuum_absorption),
        "continuum_scattering": _array_summary(continuum_scattering),
        "temperature": _array_summary(np.asarray(population.setup.atmosphere.temperature)),
        "mass_density": _array_summary(np.asarray(runtime.mass_density)),
        "electron_density": _array_summary(np.asarray(runtime.electron_density)),
    }


def _run_one(
    labels: np.ndarray,
    index: int,
    arm: str,
    *,
    max_nfev: int,
    temperature_correction_damping: float,
) -> dict[str, object]:
    start = time.perf_counter()
    record = _label_record(labels, index, arm)
    record["temperature_correction_damping"] = float(temperature_correction_damping)
    try:
        seed_start = time.perf_counter()
        if arm == "grey":
            tau, seed, seed_info = _grey_seed(labels)
        elif arm == "physical_residual":
            tau, seed, seed_info = _physical_residual_seed(
                labels,
                max_nfev=int(max_nfev),
            )
        else:
            raise ValueError(f"unknown arm: {arm}")
        record.update(seed_info)
        record["seed_seconds"] = float(time.perf_counter() - seed_start)

        config = replace(
            _solver_config(
                seed,
                iterations_per_trial=1,
                structured_atmosphere_path=None,
                debug_state_path=None,
            ),
            temperature_correction_damping=float(temperature_correction_damping),
        )
        solver_start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = _run_atmosphere_model(
                config,
                after_iteration_hook=_handoff_hook,
            )
        record.update(
            {
                "solver_seconds": float(time.perf_counter() - solver_start),
                "solver_outcome": "one_iteration_complete",
                "iterations_completed": int(result.iterations_completed),
                "converged": bool(result.converged),
                "finite_final_state": bool(_atmosphere_is_finite(result.atmosphere)),
                "positive_temperature_mass": bool(
                    np.all(result.atmosphere.temperature > 0.0)
                    and np.all(result.atmosphere.column_mass > 0.0)
                ),
                "handoff_state_acceptable": bool(
                    _atmosphere_is_finite(result.atmosphere)
                    and np.all(result.atmosphere.temperature > 0.0)
                    and np.all(result.atmosphere.column_mass > 0.0)
                ),
                "final_temperature_range": [
                    float(np.min(result.atmosphere.temperature)),
                    float(np.max(result.atmosphere.temperature)),
                ],
                "final_mass_range": [
                    float(np.min(result.atmosphere.column_mass)),
                    float(np.max(result.atmosphere.column_mass)),
                ],
                "solver_diagnostics": result.diagnostics,
                "warning_count": len(caught),
                "warning_types": sorted({type(item.message).__name__ for item in caught}),
            }
        )
    except Exception as exc:  # noqa: BLE001 - the failure is the audit result
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--indices", type=int, nargs="+", default=list(DEFAULT_INDICES))
    parser.add_argument("--arms", nargs="+", choices=DEFAULT_ARMS, default=list(DEFAULT_ARMS))
    parser.add_argument("--max-nfev", type=int, default=8)
    parser.add_argument("--temperature-correction-damping", type=float, default=1.0)
    args = parser.parse_args(argv)

    corpus = load_strict_truth(DEFAULT_CORPUS)
    records: list[dict[str, object]] = []
    jsonl_path = args.out.with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index in args.indices:
            for arm in args.arms:
                record = _run_one(
                    corpus.labels[int(index)],
                    int(index),
                    str(arm),
                    max_nfev=int(args.max_nfev),
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
        "candidate": "physical_residual_32_to_80_handoff_audit",
        "status": "one_exact_iteration_audit",
        "indices": [int(index) for index in args.indices],
        "arms": [str(arm) for arm in args.arms],
        "temperature_correction_damping": float(args.temperature_correction_damping),
        "records": records,
        "development_60": "blocked_until_handoff_audit_is_finite_and_paired",
        "sealed_holdout": "closed",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
