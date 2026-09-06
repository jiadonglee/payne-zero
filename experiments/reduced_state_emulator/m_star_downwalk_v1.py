"""Downward fine-step walk on the [M/H]-1.0 dwarf track: 4000 -> 3600 K.

Walks from the certified scale-out 4000 K product through 25 K waypoints
(halving to 12.5 K on failure) to the closed 3600 K node. Waypoints need
solver convergence and a finite state under the unchanged production
solver; the 3600 K target certifies under the full pipeline standard --
phase-aware primary, strict phase-aware self-restart, frozen flux gate
on both legs, path consistency, and the phase guard on both legs.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from . import m_star_pipeline as pipeline
from .cool_star_step_test import (
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from . import m_star_bootstrap_v1r2_marcs100 as base
from .m_star_bootstrap_v1 import (
    _annotate_record,
    _load_mt,
    _passes_flux_gate,
    _product_consistency,
    _sha256,
    _write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_downwalk_v1"
TARGET_TEMPERATURE_K = 3600.0
TRACK_SLUG = "g+4.50_m-1.00_a+0.00_c+0.00_x1.00"
CHAIN_SEED_PRODUCTS_DIR = (
    REPO_ROOT
    / "results"
    / "m_star_pipeline_scaleout_v1"
    / "cases"
    / "dwarf_g+4.50_m-1.00_t4000"
    / "cap60"
    / "products"
    / "primary"
)
CHAIN_SEED_TEMPERATURE_K = 4000.0
INITIAL_STEP_K = 25.0
MINIMUM_STEP_K = 12.5
WAYPOINT_CAP = 60
TARGET_CAP = 120
TEMPERATURE_TOLERANCE_K = 1.0e-9
PHASE_AWARE = {"require_improving_flux_residual": True}
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_GATE_PATH = (
    REPO_ROOT / "results" / "m_star_iteration_tomography_v1" / "flux_gate.json"
)
PREREGISTRATION_PATH = (
    REPO_ROOT / "notes" / "m_star_downwalk_v1_preregistration_20260906.md"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _phase_guard(record: dict[str, Any] | None) -> bool | None:
    if not record:
        return None
    value = (
        (record.get("solver_diagnostics") or {})
        .get("final_diagnostics", {})
        .get("flux_residual_improving_at_stop")
    )
    return None if value is None else bool(value)


def run_walk(args: argparse.Namespace) -> dict[str, Any]:
    _set_single_thread_environment()
    result_root = Path(args.result_root)
    tomography_root = Path(args.tomography_root)
    gate = _read_json(Path(args.gate_path))
    chain_products = sorted(CHAIN_SEED_PRODUCTS_DIR.glob("*.npz"))
    if not chain_products:
        raise FileNotFoundError(f"no chain-seed product in {CHAIN_SEED_PRODUCTS_DIR}")
    chain_product = chain_products[0]
    protocol = {
        "campaign": CAMPAIGN,
        "preregistration": str(PREREGISTRATION_PATH),
        "track": TRACK_SLUG,
        "target_temperature_K": TARGET_TEMPERATURE_K,
        "chain_seed_product": str(chain_product),
        "step_policy": {
            "initial_step_K": INITIAL_STEP_K,
            "minimum_step_K": MINIMUM_STEP_K,
            "waypoint_cap": WAYPOINT_CAP,
            "target_cap": TARGET_CAP,
            "target_phase_aware": True,
        },
        "flux_gate_source": {
            "campaign": gate.get("campaign"),
            "gate_hash": gate.get("gate_hash"),
            "thresholds": gate.get("thresholds"),
        },
    }
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    _write_json(result_root / "protocol.json", protocol)

    track_payload = pipeline.track_payload(
        stellar_class="dwarf",
        log_surface_gravity=4.5,
        metallicity=-1.0,
        microturbulence_km_s=pipeline.MICROTURBULENCE["dwarf"],
    )
    track = base._track_from_payload(track_payload)
    case_root = (
        result_root / "cases" / f"dwarf_g+4.50_m-1.00_t{int(TARGET_TEMPERATURE_K):04d}"
    )
    start_mass, start_profile = _load_mt(chain_product)

    steps: list[dict[str, Any]] = []
    current_temperature = float(CHAIN_SEED_TEMPERATURE_K)
    current_mass = np.asarray(start_mass, dtype=np.float64)
    current_profile = np.asarray(start_profile, dtype=np.float64)
    step_size = float(INITIAL_STEP_K)
    primary: dict[str, Any] | None = None
    restart: dict[str, Any] | None = None
    reached = False
    stopped_reason: str | None = None
    started = time.perf_counter()

    while abs(current_temperature - TARGET_TEMPERATURE_K) > TEMPERATURE_TOLERANCE_K:
        proposed = max(TARGET_TEMPERATURE_K, current_temperature - step_size)
        is_target = abs(proposed - TARGET_TEMPERATURE_K) <= TEMPERATURE_TOLERANCE_K
        labels = track.labels(proposed)
        try:
            seed_atmosphere = _reconstruct_from_mt(
                labels, current_mass, current_profile
            )
        except Exception as exc:  # noqa: BLE001 - a failed step is an outcome
            steps.append(
                {
                    "status": "initialization_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "source_temperature_K": current_temperature,
                    "target_temperature_K": proposed,
                    "requested_step_K": step_size,
                    "is_target_node": is_target,
                }
            )
            record = None
        else:
            record, state = _solve_attempt(
                track=track,
                method="downwalk_waypoint" if not is_target else "downwalk_target",
                schedule="downwalk",
                source_temperature=current_temperature,
                target_labels=labels,
                initial_atmosphere=seed_atmosphere,
                product_dir=(
                    case_root / "products" / "primary"
                    if is_target
                    else case_root / "products" / "continuation"
                ),
                iteration_cap=WAYPOINT_CAP if not is_target else TARGET_CAP,
                maximum_all_layer_relative_temperature_change=(
                    pipeline.STRICT_ALL_LAYER_LIMIT
                ),
                config_overrides=dict(PHASE_AWARE) if is_target else None,
            )
            record.update(
                {
                    "requested_step_K": step_size,
                    "is_target_node": is_target,
                }
            )
            steps.append(record)
        survives = bool(record is not None and record.get("survives_solver"))

        if survives and is_target:
            primary = record
            reached = True
            break
        if survives:
            current_temperature = proposed
            current_mass = np.asarray(state.column_mass, dtype=np.float64)
            current_profile = np.asarray(state.temperature, dtype=np.float64)
            continue
        if step_size > MINIMUM_STEP_K:
            step_size = max(step_size / 2.0, MINIMUM_STEP_K)
            continue
        stopped_reason = "minimum_step_failed"
        break

    restart: dict[str, Any] | None = None
    if primary is not None and primary.get("survives_solver") and primary.get(
        "product_path"
    ):
        labels = track.labels(TARGET_TEMPERATURE_K)
        solved_m, solved_t = _load_mt(primary["product_path"])
        restart_seed = _reconstruct_from_mt(labels, solved_m, solved_t)
        restart, _restart_state = _solve_attempt(
            track=track,
            method="downwalk_strict_self_restart",
            schedule="independent_self_restart",
            source_temperature=TARGET_TEMPERATURE_K,
            target_labels=labels,
            initial_atmosphere=restart_seed,
            product_dir=case_root / "products" / "restart",
            iteration_cap=TARGET_CAP,
            maximum_all_layer_relative_temperature_change=(
                pipeline.STRICT_ALL_LAYER_LIMIT
            ),
            config_overrides=dict(PHASE_AWARE),
        )
        restart = _annotate_record(
            restart,
            track_payload=track_payload,
            role="train",
            node_id=f"{TRACK_SLUG}_t{int(TARGET_TEMPERATURE_K)}",
        )
        primary = _annotate_record(
            primary,
            track_payload=track_payload,
            role="train",
            node_id=f"{TRACK_SLUG}_t{int(TARGET_TEMPERATURE_K)}",
        )

    primary_flux = (
        _passes_flux_gate(primary, gate)
        if primary
        else {"passes": False, "metrics": {}}
    )
    restart_flux = (
        _passes_flux_gate(restart, gate) if restart else {"passes": False, "metrics": {}}
    )
    consistency = (
        _product_consistency(
            primary.get("product_path"),
            None if restart is None else restart.get("product_path"),
        )
        if primary
        else {"available": False, "passes": False}
    )
    guard_primary = _phase_guard(primary)
    guard_restart = _phase_guard(restart)
    eligible = bool(
        reached
        and primary is not None
        and primary.get("survives_solver")
        and restart is not None
        and restart.get("survives_solver")
        and primary.get("state_quality", {}).get("valid")
        and restart.get("state_quality", {}).get("valid")
        and primary_flux["passes"]
        and restart_flux["passes"]
        and consistency["passes"]
        and guard_primary is True
        and guard_restart is True
    )
    reasons: list[str] = []
    if not reached:
        reasons.append(stopped_reason or "target_not_reached")
    else:
        if restart is None or not restart.get("survives_solver"):
            reasons.append("self_restart")
        if not primary_flux["passes"]:
            reasons.append("primary_flux_gate")
        if not restart_flux["passes"]:
            reasons.append("restart_flux_gate")
        if not consistency["passes"]:
            reasons.append("path_consistency")
        if guard_primary is not True:
            reasons.append("primary_phase_guard")
        if guard_restart is not True:
            reasons.append("restart_phase_guard")

    output = {
        "campaign": CAMPAIGN,
        "protocol_hash": protocol["protocol_hash"],
        "candidate_id": f"{TRACK_SLUG}_t{int(TARGET_TEMPERATURE_K)}",
        "class": "dwarf",
        "track": track_payload,
        "temperature_K": TARGET_TEMPERATURE_K,
        "labels": track.labels(TARGET_TEMPERATURE_K).as_kwargs(),
        "continuation": {
            "mode": "reduced_rematerialized",
            "direction": "downward_from_4000",
            "initial_step_K": INITIAL_STEP_K,
            "minimum_step_K": MINIMUM_STEP_K,
            "final_step_K": step_size,
            "steps": steps,
            "reached_target": reached,
            "chain_seed": {
                "product_path": str(chain_product),
                "product_sha256": _sha256(chain_product),
                "temperature_K": CHAIN_SEED_TEMPERATURE_K,
            },
        },
        "primary": primary,
        "restart": restart,
        "primary_flux_gate": primary_flux,
        "restart_flux_gate": restart_flux,
        "path_consistency": consistency,
        "phase_guard": {"primary": guard_primary, "restart": guard_restart},
        "training_eligible": eligible,
        "status": "training_eligible" if eligible else "ineligible",
        "failure_reason": None if eligible else ",".join(reasons),
        "seconds": float(time.perf_counter() - started),
    }
    _write_json(case_root / "case.json", output)
    print(
        f"downwalk: eligible={eligible} reached={reached} "
        f"reason={output['failure_reason']} steps={len(steps)} "
        f"seconds={output['seconds']:.0f}"
    )
    for step in steps:
        p95 = (
            (step.get("solver_diagnostics") or {})
            .get("final_diagnostics", {})
            .get("p95_absolute_flux_error_percent")
        )
        print(
            f"   {step.get('status')} {step.get('source_temperature_K')}->"
            f"{step.get('target_temperature_K')} step={step.get('requested_step_K')} "
            f"iters={step.get('iterations')} "
            f"p95={round(p95, 2) if p95 is not None else '—'}"
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--gate-path", default=str(DEFAULT_GATE_PATH))
    parser.add_argument(
        "--tomography-root",
        default=str(REPO_ROOT / "results" / "m_star_iteration_tomography_v1"),
    )
    args = parser.parse_args(argv)
    run_walk(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
