"""One-step local M-dwarf response and true trial-residual diagnostic.

The campaign uses only the existing A (solar-metallicity) and D
(``[M/H]=-0.5``) tomography snapshots.  A single deep layer is used by
default so that the first result remains small and inspectable; ``--max-layers
3`` adds the two next largest deep residual layers.  Every state evaluation
runs one exact opacity, transfer, EOS, and convection iteration with opacity
lagging disabled.

The runner's correction diagnostics describe the input state ``x``.  For each
returned state ``G(x)`` this script runs a second exact physical evaluation and
records its raw and smoothed total-flux residual, which is the diagnostic for
``R(G(x))``.  The alpha trials use the one-step direction and re-evaluate each
candidate independently.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import dataclasses
import json
from pathlib import Path
import time
import traceback
from typing import Any

import numpy as np

from bench.labels import StellarLabels
from bench.run_reference import _solver_config
from payne_zero_atmosphere.runner import run_atmosphere_model

from .cool_star_step_test import (
    _clone_atmosphere,
    _reconstruct_from_mt,
    _set_single_thread_environment,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_local_iteration_response_20260919"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
TOMOGRAPHY_ROOT = REPO_ROOT / "results" / "m_star_iteration_tomography_v1"
EPSILON = 1.0e-3
ALPHAS = (1.0, 0.5, 0.25)

CASES = {
    "A": {
        "track_slug": "g+4.50_m+0.00_a+0.00_c+0.00_x1.00",
        "temperature_K": 3500.0,
    },
    "D": {
        "track_slug": "g+4.50_m-0.50_a+0.00_c+0.00_x1.00",
        "temperature_K": 3600.0,
    },
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")


def _latest_snapshot(track_slug: str, teff: float) -> Path:
    root = (
        TOMOGRAPHY_ROOT
        / "cases"
        / "dwarf"
        / track_slug
        / f"t{int(teff):04d}"
        / "iterations"
        / "primary"
    )
    paths = sorted(root.glob("iter_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no tomography snapshot in {root}")
    return paths[-1]


def _load_case(case_id: str) -> dict[str, Any]:
    spec = CASES[case_id]
    case_root = (
        TOMOGRAPHY_ROOT
        / "cases"
        / "dwarf"
        / spec["track_slug"]
        / f"t{int(spec['temperature_K']):04d}"
    )
    case_json_path = case_root / "case.json"
    case_json = json.loads(case_json_path.read_text())
    labels_payload = case_json["labels"]
    labels = StellarLabels(
        effective_temperature=float(labels_payload["effective_temperature"]),
        log_surface_gravity=float(labels_payload["log_surface_gravity"]),
        metallicity=float(labels_payload["metallicity"]),
        alpha_enhancement=float(labels_payload["alpha_enhancement"]),
        microturbulence_km_s=float(labels_payload["microturbulence_km_s"]),
    )
    snapshot = _latest_snapshot(spec["track_slug"], spec["temperature_K"])
    with np.load(snapshot, allow_pickle=False) as data:
        required = (
            "log_tau_standard",
            "temperature_post",
            "gas_pressure_post",
            "column_mass_post",
            "flux_error_percent",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{snapshot} missing {missing}")
        arrays = {key: np.asarray(data[key], dtype=np.float64) for key in required}
        snapshot_iteration = int(data["iteration"])
    deep = np.flatnonzero(arrays["log_tau_standard"] >= 1.5)
    order = np.argsort(np.abs(arrays["flux_error_percent"][deep]))[::-1]
    ranked = deep[order]
    return {
        "case_id": case_id,
        "track_slug": spec["track_slug"],
        "target_temperature_K": float(spec["temperature_K"]),
        "case_json_path": str(case_json_path.resolve()),
        "snapshot_path": str(snapshot.resolve()),
        "snapshot_iteration": snapshot_iteration,
        "labels": labels,
        "label_payload": {key: float(value) for key, value in labels_payload.items()},
        "arrays": arrays,
        "ranked_deep_layers": [int(value) for value in ranked],
    }


def _state_from_mt(
    labels: StellarLabels,
    column_mass: np.ndarray,
    temperature: np.ndarray,
    gas_pressure: np.ndarray | None = None,
):
    atmosphere = _reconstruct_from_mt(labels, column_mass, temperature)
    atmosphere.column_mass[:] = np.asarray(column_mass, dtype=np.float64)
    atmosphere.temperature[:] = np.asarray(temperature, dtype=np.float64)
    if gas_pressure is not None:
        atmosphere.gas_pressure[:] = np.asarray(gas_pressure, dtype=np.float64)
    return atmosphere


def _array_summary(values: np.ndarray, indices: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "selected": {str(i): float(array[i]) for i in indices},
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
    }


def _selected_state(atmosphere, indices: list[int]) -> dict[str, Any]:
    return {
        "temperature_K": {str(i): float(atmosphere.temperature[i]) for i in indices},
        "column_mass_g_cm2": {str(i): float(atmosphere.column_mass[i]) for i in indices},
        "gas_pressure_dyn_cm2": {str(i): float(atmosphere.gas_pressure[i]) for i in indices},
    }


def _physics_rows(
    *,
    step,
    effective_temperature: float,
    indices: list[int],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    correction = step.remapped.finalization.temperature_correction_result
    h_rad = np.asarray(
        step.transfer.temperature_correction_state.integrated_eddington_flux,
        dtype=np.float64,
    ).copy()
    h_conv_smooth = np.asarray(correction.convective_flux, dtype=np.float64).copy()
    convection = step.remapped.finalization.convection_result
    if convection is None:
        h_conv_raw = h_conv_smooth.copy()
    else:
        h_conv_raw = np.asarray(convection.raw_convective_flux, dtype=np.float64).copy()
    target_h = 5.6697e-5 / 12.5664 * float(effective_temperature) ** 4
    r_raw = (h_rad + h_conv_raw - target_h) / max(target_h, 1.0e-300)
    r_smooth = (h_rad + h_conv_smooth - target_h) / max(target_h, 1.0e-300)
    reported = np.asarray(correction.flux_error_percent, dtype=np.float64) / 100.0
    rows = {
        "target_integrated_eddington_flux_H": float(target_h),
        "units": "F and convective_flux are integrated Eddington flux H=F/(4*pi)",
        "raw": {
            "Hrad": _array_summary(h_rad, indices),
            "Hconv": _array_summary(h_conv_raw, indices),
            "R": _array_summary(r_raw, indices),
        },
        "smoothed": {
            "Hrad": _array_summary(h_rad, indices),
            "Hconv": _array_summary(h_conv_smooth, indices),
            "R": _array_summary(r_smooth, indices),
        },
        "runner_flux_error_percent_over_100": _array_summary(reported, indices),
        "max_abs_difference_runner_vs_smoothed_R": float(
            np.max(np.abs(reported - r_smooth))
        ),
        "all_layer": {
            "raw_p95_abs_R": float(np.percentile(np.abs(r_raw), 95.0)),
            "raw_max_abs_R": float(np.max(np.abs(r_raw))),
            "smoothed_p95_abs_R": float(np.percentile(np.abs(r_smooth), 95.0)),
            "smoothed_max_abs_R": float(np.max(np.abs(r_smooth))),
        },
    }
    arrays = {
        "Hrad": h_rad,
        "Hconv_raw": h_conv_raw,
        "Hconv_smoothed": h_conv_smooth,
        "R_raw": r_raw,
        "R_smoothed": r_smooth,
        "runner_R_smoothed": reported,
    }
    return rows, arrays


def _physics_rows_from_arrays(
    *,
    h_rad: np.ndarray,
    h_conv_raw: np.ndarray,
    h_conv_smooth: np.ndarray,
    reported: np.ndarray,
    effective_temperature: float,
    indices: list[int],
) -> dict[str, Any]:
    """Rebuild the compact physics record from a saved trial NPZ."""

    target_h = 5.6697e-5 / 12.5664 * float(effective_temperature) ** 4
    r_raw = (h_rad + h_conv_raw - target_h) / max(target_h, 1.0e-300)
    r_smooth = (h_rad + h_conv_smooth - target_h) / max(target_h, 1.0e-300)
    return {
        "target_integrated_eddington_flux_H": float(target_h),
        "units": "F and convective_flux are integrated Eddington flux H=F/(4*pi)",
        "raw": {
            "Hrad": _array_summary(h_rad, indices),
            "Hconv": _array_summary(h_conv_raw, indices),
            "R": _array_summary(r_raw, indices),
        },
        "smoothed": {
            "Hrad": _array_summary(h_rad, indices),
            "Hconv": _array_summary(h_conv_smooth, indices),
            "R": _array_summary(r_smooth, indices),
        },
        "runner_flux_error_percent_over_100": _array_summary(reported, indices),
        "max_abs_difference_runner_vs_smoothed_R": float(
            np.max(np.abs(reported - r_smooth))
        ),
        "all_layer": {
            "raw_p95_abs_R": float(np.percentile(np.abs(r_raw), 95.0)),
            "raw_max_abs_R": float(np.max(np.abs(r_raw))),
            "smoothed_p95_abs_R": float(np.percentile(np.abs(r_smooth), 95.0)),
            "smoothed_max_abs_R": float(np.max(np.abs(r_smooth))),
        },
    }


def _selected_state_from_arrays(
    temperature: np.ndarray,
    column_mass: np.ndarray,
    gas_pressure: np.ndarray,
    indices: list[int],
) -> dict[str, Any]:
    return {
        "temperature_K": {str(i): float(temperature[i]) for i in indices},
        "column_mass_g_cm2": {str(i): float(column_mass[i]) for i in indices},
        "gas_pressure_dyn_cm2": {str(i): float(gas_pressure[i]) for i in indices},
    }


def _resume_array_path(result_root: Path, run_id: str) -> Path | None:
    """Find an existing trial NPZ without treating a missing file as success."""

    roots = (Path(result_root), Path(result_root) / "parallel_d")
    for root in roots:
        candidate = root / "arrays" / f"{run_id}.npz"
        if candidate.is_file():
            return candidate
    return None


def _resume_protocol_compatible(
    result_root: Path,
    requested_protocol: dict[str, Any],
) -> tuple[bool, str]:
    """Check the protocol fields that determine whether an NPZ is reusable."""

    protocol_path = Path(result_root) / "protocol.json"
    if not protocol_path.is_file():
        return False, f"missing protocol manifest: {protocol_path}"
    try:
        existing = json.loads(protocol_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"cannot read protocol manifest {protocol_path}: {exc}"

    mismatches: list[str] = []
    existing_epsilon = existing.get("epsilon")
    requested_epsilon = requested_protocol.get("epsilon")
    if existing_epsilon is None or not np.isclose(
        float(existing_epsilon), float(requested_epsilon), rtol=0.0, atol=1.0e-15
    ):
        mismatches.append(
            f"epsilon={existing_epsilon!r} (requested {requested_epsilon!r})"
        )
    existing_alphas = existing.get("alpha_trials")
    requested_alphas = requested_protocol.get("alpha_trials")
    if (
        existing_alphas is None
        or len(existing_alphas) != len(requested_alphas)
        or not np.allclose(
            np.asarray(existing_alphas, dtype=np.float64),
            np.asarray(requested_alphas, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-15,
        )
    ):
        mismatches.append(
            f"alpha_trials={existing_alphas!r} (requested {requested_alphas!r})"
        )
    if existing.get("max_layers") != requested_protocol.get("max_layers"):
        mismatches.append(
            f"max_layers={existing.get('max_layers')!r} "
            f"(requested {requested_protocol.get('max_layers')!r})"
        )
    existing_opacity_lagging = existing.get("opacity_lagging")
    if "opacity_lagging" not in existing or bool(existing_opacity_lagging) is not bool(
        requested_protocol.get("opacity_lagging")
    ):
        mismatches.append(
            f"opacity_lagging={existing_opacity_lagging!r} "
            f"(requested {requested_protocol.get('opacity_lagging')!r})"
        )
    if mismatches:
        return False, "; ".join(mismatches)
    # The cases list is intentionally not compared: A and D can be resumed
    # from separate/partial runs under the same numerical protocol.
    return True, "matching epsilon, alpha_trials, max_layers, opacity_lagging"


def _record_from_saved_arrays(
    *,
    array_path: Path,
    run_id: str,
    effective_temperature: float,
    indices: list[int],
) -> dict[str, Any]:
    """Recover a completed trial without inventing timing or solver flags."""

    required = (
        "temperature",
        "column_mass",
        "gas_pressure",
        "output_temperature",
        "output_column_mass",
        "output_gas_pressure",
        "Hrad",
        "Hconv_raw",
        "Hconv_smoothed",
        "runner_R_smoothed",
    )
    with np.load(array_path, allow_pickle=False) as data:
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{array_path} missing saved trial fields: {missing}")
        temperature = np.asarray(data["temperature"], dtype=np.float64)
        column_mass = np.asarray(data["column_mass"], dtype=np.float64)
        gas_pressure = np.asarray(data["gas_pressure"], dtype=np.float64)
        output_temperature = np.asarray(data["output_temperature"], dtype=np.float64)
        output_column_mass = np.asarray(data["output_column_mass"], dtype=np.float64)
        output_gas_pressure = np.asarray(data["output_gas_pressure"], dtype=np.float64)
        h_rad = np.asarray(data["Hrad"], dtype=np.float64)
        h_conv_raw = np.asarray(data["Hconv_raw"], dtype=np.float64)
        h_conv_smooth = np.asarray(data["Hconv_smoothed"], dtype=np.float64)
        reported = np.asarray(data["runner_R_smoothed"], dtype=np.float64)
    arrays = (
        temperature,
        column_mass,
        gas_pressure,
        output_temperature,
        output_column_mass,
        output_gas_pressure,
        h_rad,
        h_conv_raw,
        h_conv_smooth,
        reported,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError(f"saved trial {array_path} contains non-finite values")
    return {
        "run_id": run_id,
        "seconds": None,
        "opacity_lagging": False,
        "iteration_cap": 1,
        "error": None,
        "resumed_from_npz": True,
        "solver_converged_flag": None,
        "iterations_completed": None,
        "diagnostics": None,
        "iteration_index": None,
        "input_state": _selected_state_from_arrays(
            temperature, column_mass, gas_pressure, indices
        ),
        "output_state": _selected_state_from_arrays(
            output_temperature,
            output_column_mass,
            output_gas_pressure,
            indices,
        ),
        "physics": _physics_rows_from_arrays(
            h_rad=h_rad,
            h_conv_raw=h_conv_raw,
            h_conv_smooth=h_conv_smooth,
            reported=reported,
            effective_temperature=effective_temperature,
            indices=indices,
        ),
        "timing": None,
        "arrays_path": str(array_path.resolve()),
    }


def _run_exact_state(
    *,
    atmosphere,
    labels: StellarLabels,
    effective_temperature: float,
    run_id: str,
    indices: list[int],
    result_root: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Run one exact physical iteration and capture R(input), output state, and G."""

    if resume:
        array_path = _resume_array_path(result_root, run_id)
        if array_path is not None:
            record = _record_from_saved_arrays(
                array_path=array_path,
                run_id=run_id,
                effective_temperature=effective_temperature,
                indices=indices,
            )
            _write_json(result_root / "records" / f"{run_id}.json", record)
            print(f"[{run_id}] resumed from {array_path}", flush=True)
            return record

    capture: dict[str, Any] = {}

    def hook(iteration_index, setup, step):
        physics, physics_arrays = _physics_rows(
            step=step,
            effective_temperature=effective_temperature,
            indices=indices,
        )
        capture["iteration_index"] = int(iteration_index)
        capture["input_state"] = _selected_state(setup.atmosphere, indices)
        capture["output_state"] = _selected_state(step.remapped.atmosphere, indices)
        capture["input_arrays"] = {
            "temperature": np.asarray(setup.atmosphere.temperature, dtype=np.float64).copy(),
            "column_mass": np.asarray(setup.atmosphere.column_mass, dtype=np.float64).copy(),
            "gas_pressure": np.asarray(setup.atmosphere.gas_pressure, dtype=np.float64).copy(),
        }
        capture["output_arrays"] = {
            "temperature": np.asarray(step.remapped.atmosphere.temperature, dtype=np.float64).copy(),
            "column_mass": np.asarray(step.remapped.atmosphere.column_mass, dtype=np.float64).copy(),
            "gas_pressure": np.asarray(step.remapped.atmosphere.gas_pressure, dtype=np.float64).copy(),
        }
        capture["physics"] = physics
        capture["physics_arrays"] = physics_arrays
        capture["timing"] = dict(step.timing)
        return {"captured_input_residual": True}

    started = time.perf_counter()
    error = None
    result = None
    try:
        config = _solver_config(
            _clone_atmosphere(atmosphere),
            iterations_per_trial=1,
            structured_atmosphere_path=None,
            debug_state_path=None,
        )
        config = dataclasses.replace(
            config,
            enable_opacity_lagging=False,
            opacity_recompute_interval=1,
            flux_residual_guided_damping=False,
            require_improving_flux_residual=False,
        )
        result = run_atmosphere_model(config, after_iteration_hook=hook)
    except Exception as exc:  # one failed trial is recorded, not hidden
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    seconds = time.perf_counter() - started
    record: dict[str, Any] = {
        "run_id": run_id,
        "seconds": float(seconds),
        "opacity_lagging": False,
        "iteration_cap": 1,
        "error": error,
    }
    if result is not None and capture:
        record.update(
            {
                "solver_converged_flag": bool(result.converged),
                "iterations_completed": int(result.iterations_completed),
                "diagnostics": {
                    "p95_absolute_flux_error_percent": float(
                        result.diagnostics["p95_absolute_flux_error_percent"]
                    ),
                    "maximum_absolute_flux_error_percent": float(
                        result.diagnostics["maximum_absolute_flux_error_percent"]
                    ),
                    "total_seconds": float(result.diagnostics["total_seconds"]),
                },
                "iteration_index": capture["iteration_index"],
                "input_state": capture["input_state"],
                "output_state": capture["output_state"],
                "physics": capture["physics"],
                "timing": capture["timing"],
            }
        )
        arrays = {
            **capture["input_arrays"],
            **{f"output_{key}": value for key, value in capture["output_arrays"].items()},
            **capture["physics_arrays"],
        }
        array_path = result_root / "arrays" / f"{run_id}.npz"
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(array_path, **arrays)
        record["arrays_path"] = str(array_path.resolve())
    elif error is None:
        record["error"] = "runner returned without an iteration hook capture"
    _write_json(result_root / "records" / f"{run_id}.json", record)
    print(
        f"[{run_id}] {seconds:.1f}s "
        f"status={'ok' if record.get('error') is None else 'failed'}",
        flush=True,
    )
    return record


def _state_for_trial(labels, record: dict[str, Any], *, alpha: float | None = None):
    arrays_path = record.get("arrays_path")
    if not arrays_path:
        raise RuntimeError(f"run {record.get('run_id')} has no output state")
    with np.load(arrays_path, allow_pickle=False) as data:
        if alpha is None:
            temperature = np.asarray(data["output_temperature"], dtype=np.float64)
            column_mass = np.asarray(data["output_column_mass"], dtype=np.float64)
            # Every true-residual candidate, including alpha=1, goes through
            # the same (m,T)->full-state map.  This avoids interpreting a
            # pressure-reconstruction difference as damping response.
            return _state_from_mt(labels, column_mass, temperature)
        base_temperature = np.asarray(data["temperature"], dtype=np.float64)
        base_column_mass = np.asarray(data["column_mass"], dtype=np.float64)
        out_temperature = np.asarray(data["output_temperature"], dtype=np.float64)
        out_column_mass = np.asarray(data["output_column_mass"], dtype=np.float64)
    temperature = base_temperature + float(alpha) * (out_temperature - base_temperature)
    column_mass = base_column_mass + float(alpha) * (out_column_mass - base_column_mass)
    return _state_from_mt(labels, column_mass, temperature)


def _record_layer_differences(
    *,
    plus: dict[str, Any],
    minus: dict[str, Any],
    baseline: dict[str, Any],
    indices: list[int],
    kind: str,
) -> dict[str, Any]:
    output_rows: dict[str, Any] = {}
    input_rows: dict[str, Any] = {}
    for index in indices:
        key = str(index)
        p_in = plus["input_state"]["temperature_K" if kind == "temperature" else "column_mass_g_cm2"][key]
        m_in = minus["input_state"]["temperature_K" if kind == "temperature" else "column_mass_g_cm2"][key]
        p_out = plus["output_state"]["temperature_K" if kind == "temperature" else "column_mass_g_cm2"][key]
        m_out = minus["output_state"]["temperature_K" if kind == "temperature" else "column_mass_g_cm2"][key]
        base_in = baseline["input_state"]["temperature_K" if kind == "temperature" else "column_mass_g_cm2"][key]
        scale = max(abs(float(base_in)), 1.0e-300)
        input_rows[key] = {
            "plus_minus_relative": float((p_in - m_in) / (2.0 * scale)),
            "plus": float(p_in),
            "minus": float(m_in),
        }
        output_rows[key] = {
            "G_output_central_difference": float((p_out - m_out) / (2.0 * EPSILON * scale)),
            "plus": float(p_out),
            "minus": float(m_out),
        }
    return {"kind": kind, "input": input_rows, "output": output_rows}


def _record_true_residual_response(
    *,
    plus: dict[str, Any],
    minus: dict[str, Any],
    indices: list[int],
) -> dict[str, Any]:
    """Central difference of R(G(x)) after the extra physical evaluation."""

    rows: dict[str, Any] = {}
    plus_eval = plus.get("true_R_of_one_step_state")
    minus_eval = minus.get("true_R_of_one_step_state")
    if not plus_eval or not minus_eval:
        return {"available": False, "reason": "one or both true residual evaluations failed"}
    for index in indices:
        key = str(index)
        plus_smooth = plus_eval["physics"]["smoothed"]["R"]["selected"][key]
        minus_smooth = minus_eval["physics"]["smoothed"]["R"]["selected"][key]
        plus_raw = plus_eval["physics"]["raw"]["R"]["selected"][key]
        minus_raw = minus_eval["physics"]["raw"]["R"]["selected"][key]
        rows[key] = {
            "dR_smooth_drelative_input": float(
                (plus_smooth - minus_smooth) / (2.0 * EPSILON)
            ),
            "dR_raw_drelative_input": float(
                (plus_raw - minus_raw) / (2.0 * EPSILON)
            ),
            "plus_R_smooth": float(plus_smooth),
            "minus_R_smooth": float(minus_smooth),
            "plus_R_raw": float(plus_raw),
            "minus_R_raw": float(minus_raw),
        }
    return {"available": True, "selected": rows}


def _residual_metric(record: dict[str, Any]) -> float | None:
    physics = record.get("physics")
    if not isinstance(physics, dict):
        return None
    selected = physics.get("smoothed", {}).get("R", {}).get("selected", {})
    values = [abs(float(value)) for value in selected.values() if np.isfinite(value)]
    return None if not values else float(max(values))


def _alpha_acceptance(
    *,
    baseline: dict[str, Any],
    alpha_trials: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Choose alpha using the newly evaluated deep-layer smoothed residual."""

    baseline_metric = _residual_metric(baseline)
    candidate_metrics = {
        alpha: _residual_metric(record) for alpha, record in alpha_trials.items()
    }
    finite_candidates = {
        alpha: value
        for alpha, value in candidate_metrics.items()
        if value is not None and np.isfinite(value)
    }
    if baseline_metric is None or not finite_candidates:
        return {
            "metric": "max absolute smoothed R on selected deep layers",
            "baseline_metric": baseline_metric,
            "candidate_metrics": candidate_metrics,
            "accepted_alpha": None,
            "accepted": False,
            "reason": "missing true residual evaluation",
        }
    best_alpha = min(finite_candidates, key=finite_candidates.get)
    accepted = bool(finite_candidates[best_alpha] < baseline_metric)
    return {
        "metric": "max absolute smoothed R on selected deep layers",
        "baseline_metric": float(baseline_metric),
        "candidate_metrics": candidate_metrics,
        "accepted_alpha": float(best_alpha) if accepted else None,
        "accepted": accepted,
        "reason": (
            "candidate reduced the re-evaluated residual"
            if accepted
            else "no tested candidate reduced the re-evaluated residual"
        ),
    }


def _case_complete(case_result: dict[str, Any], max_layers: int) -> bool:
    baseline = case_result.get("baseline")
    if not isinstance(baseline, dict) or baseline.get("error") is not None:
        return False
    alpha_trials = case_result.get("alpha_trials", {})
    if set(alpha_trials) != {str(value) for value in ALPHAS}:
        return False
    if any(
        not isinstance(record, dict)
        or record.get("error") is not None
        or not record.get("arrays_path")
        or not record.get("physics")
        for record in alpha_trials.values()
    ):
        return False
    local = case_result.get("local_perturbations", {})
    selected = [
        str(row.get("layer_index"))
        for row in case_result.get("selected_layers", [])
        if isinstance(row, dict)
    ]
    if len(selected) != int(max_layers):
        return False
    for kind in ("temperature", "column_mass"):
        for layer in selected:
            row = local.get(kind, {}).get(layer)
            if not isinstance(row, dict) or row.get("true_residual_response", {}).get("available") is not True:
                return False
            for sign in ("plus", "minus"):
                trial = row.get(sign, {})
                true_r = trial.get("true_R_of_one_step_state")
                if (
                    trial.get("one_step", {}).get("error") is not None
                    or not isinstance(true_r, dict)
                    or true_r.get("error") is not None
                    or not true_r.get("arrays_path")
                ):
                    return False
    return True


def _g_update_rows(record: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    """Return the one-step G update at the selected layers."""

    arrays_path = record.get("arrays_path")
    if not arrays_path:
        return {}
    with np.load(arrays_path, allow_pickle=False) as data:
        input_temperature = np.asarray(data["temperature"], dtype=np.float64)
        input_column_mass = np.asarray(data["column_mass"], dtype=np.float64)
        output_temperature = np.asarray(data["output_temperature"], dtype=np.float64)
        output_column_mass = np.asarray(data["output_column_mass"], dtype=np.float64)
    return {
        "temperature_relative_update": {
            str(index): float(
                (output_temperature[index] - input_temperature[index])
                / max(abs(input_temperature[index]), 1.0e-300)
            )
            for index in indices
        },
        "column_mass_relative_update": {
            str(index): float(
                (output_column_mass[index] - input_column_mass[index])
                / max(abs(input_column_mass[index]), 1.0e-300)
            )
            for index in indices
        },
    }


def _run_case(
    case_id: str,
    *,
    max_layers: int,
    result_root: Path,
    baseline_only: bool,
    resume: bool = False,
) -> dict[str, Any]:
    case = _load_case(case_id)
    arrays = case["arrays"]
    indices = [int(value) for value in case["ranked_deep_layers"][:max_layers]]
    labels = case["labels"]
    base_atmosphere = _state_from_mt(
        labels,
        arrays["column_mass_post"],
        arrays["temperature_post"],
        arrays["gas_pressure_post"],
    )
    case_result: dict[str, Any] = {
        "case_id": case_id,
        "track_slug": case["track_slug"],
        "target_temperature_K": case["target_temperature_K"],
        "case_json_path": case["case_json_path"],
        "snapshot_path": case["snapshot_path"],
        "snapshot_iteration": case["snapshot_iteration"],
        "labels": case["label_payload"],
        "selected_layers": [
            {
                "layer_index": int(index),
                "log_tau_standard": float(arrays["log_tau_standard"][index]),
                "temperature_K": float(arrays["temperature_post"][index]),
                "column_mass_g_cm2": float(arrays["column_mass_post"][index]),
                "gas_pressure_dyn_cm2": float(arrays["gas_pressure_post"][index]),
                "snapshot_flux_error_percent": float(arrays["flux_error_percent"][index]),
            }
            for index in indices
        ],
        "epsilon": EPSILON,
        "baseline": None,
        "local_perturbations": {},
        "alpha_trials": {},
    }
    baseline = _run_exact_state(
        atmosphere=base_atmosphere,
        labels=labels,
        effective_temperature=case["target_temperature_K"],
        run_id=f"{case_id.lower()}_baseline",
        indices=indices,
        result_root=result_root,
        resume=resume,
    )
    case_result["baseline"] = baseline
    case_result["baseline"]["G_update_selected"] = _g_update_rows(baseline, indices)
    if baseline.get("error") is not None or baseline_only:
        return case_result

    # Alpha trials use the baseline one-step direction.  alpha=1 is the exact
    # G(x) state; every alpha is nonetheless evaluated by a fresh full physics run.
    for alpha in ALPHAS:
        state = _state_for_trial(labels, baseline, alpha=None if alpha == 1.0 else alpha)
        trial = _run_exact_state(
            atmosphere=state,
            labels=labels,
            effective_temperature=case["target_temperature_K"],
            run_id=f"{case_id.lower()}_alpha_{str(alpha).replace('.', 'p')}",
            indices=indices,
            result_root=result_root,
            resume=resume,
        )
        case_result["alpha_trials"][str(alpha)] = trial
    case_result["alpha_acceptance"] = _alpha_acceptance(
        baseline=baseline,
        alpha_trials=case_result["alpha_trials"],
    )

    for kind in ("temperature", "column_mass"):
        case_result["local_perturbations"][kind] = {}
        for perturb_index in indices:
            signed: dict[str, dict[str, Any]] = {}
            for sign in ("plus", "minus"):
                temperature = arrays["temperature_post"].copy()
                column_mass = arrays["column_mass_post"].copy()
                factor = 1.0 + EPSILON * (1.0 if sign == "plus" else -1.0)
                if kind == "temperature":
                    temperature[perturb_index] *= factor
                else:
                    column_mass[perturb_index] *= factor
                perturbed = _state_from_mt(labels, column_mass, temperature)
                first = _run_exact_state(
                    atmosphere=perturbed,
                    labels=labels,
                    effective_temperature=case["target_temperature_K"],
                    run_id=f"{case_id.lower()}_{kind}_l{perturb_index}_{sign}",
                    indices=indices,
                    result_root=result_root,
                    resume=resume,
                )
                signed[sign] = {
                    "perturb_layer_index": int(perturb_index),
                    "input_relative_step": float(
                        EPSILON * (1.0 if sign == "plus" else -1.0)
                    ),
                    "one_step": first,
                }
                if first.get("error") is None:
                    output_state = _state_for_trial(labels, first, alpha=None)
                    reevaluated = _run_exact_state(
                        atmosphere=output_state,
                        labels=labels,
                        effective_temperature=case["target_temperature_K"],
                        run_id=f"{case_id.lower()}_{kind}_l{perturb_index}_{sign}_trueR",
                        indices=indices,
                        result_root=result_root,
                        resume=resume,
                    )
                    signed[sign]["true_R_of_one_step_state"] = reevaluated
            if (
                signed.get("plus", {}).get("one_step", {}).get("error") is None
                and signed.get("minus", {}).get("one_step", {}).get("error") is None
            ):
                case_result["local_perturbations"][kind][str(perturb_index)] = {
                    "plus": signed["plus"],
                    "minus": signed["minus"],
                    "central_response": _record_layer_differences(
                        plus=signed["plus"]["one_step"],
                        minus=signed["minus"]["one_step"],
                        baseline=baseline,
                        indices=indices,
                        kind=kind,
                    ),
                    "true_residual_response": _record_true_residual_response(
                        plus=signed["plus"],
                        minus=signed["minus"],
                        indices=indices,
                    ),
                }
            else:
                case_result["local_perturbations"][kind][str(perturb_index)] = signed
    return case_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--max-layers", type=int, choices=(1, 3), default=1)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse finite saved NPZ trials only after protocol verification",
    )
    args = parser.parse_args(argv)
    _set_single_thread_environment()
    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "campaign": CAMPAIGN,
        "cases": list(args.cases),
        "epsilon": EPSILON,
        "alpha_trials": list(ALPHAS),
        "max_layers": int(args.max_layers),
        "layer_selection": "largest absolute snapshot flux residual among log_tau_standard >= 1.5",
        "one_step": "production runner with one iteration",
        "opacity_lagging": False,
        "residual": "R=(Hrad+Hconv-target_H)/target_H, raw and smoothed Hconv recorded",
        "baseline_only": bool(args.baseline_only),
        "resume": bool(args.resume),
    }
    resume_allowed = False
    resume_reason = "resume not requested"
    if args.resume:
        resume_allowed, resume_reason = _resume_protocol_compatible(
            result_root, protocol
        )
        if not resume_allowed:
            raise ValueError(
                "--resume requires a matching existing protocol.json; "
                f"{resume_reason}"
            )
    protocol["resume_protocol_verified"] = bool(resume_allowed)
    protocol["resume_protocol_reason"] = resume_reason
    _write_json(result_root / "protocol.json", protocol)
    summary: dict[str, Any] = {
        "protocol": protocol,
        "cases": {},
        "status": "running",
    }
    _write_json(result_root / "summary.json", summary)
    for case_id in args.cases:
        try:
            summary["cases"][case_id] = _run_case(
                case_id,
                max_layers=int(args.max_layers),
                result_root=result_root,
                baseline_only=bool(args.baseline_only),
                resume=resume_allowed,
            )
        except Exception as exc:  # preserve a machine-readable partial result
            summary["cases"][case_id] = {
                "case_id": case_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            }
        _write_json(result_root / "summary.json", summary)
    summary["status"] = "complete"
    complete_cases = [
        _case_complete(summary["cases"].get(case_id, {}), int(args.max_layers))
        for case_id in args.cases
    ]
    if all(complete_cases):
        summary["status"] = "complete"
    elif any(
        isinstance(summary["cases"].get(case_id), dict)
        and summary["cases"][case_id].get("baseline")
        for case_id in args.cases
    ):
        summary["status"] = "partial"
    else:
        summary["status"] = "error"
    _write_json(result_root / "summary.json", summary)
    print(f"wrote {result_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
