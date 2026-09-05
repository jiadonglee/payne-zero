"""Fine-step continuation to close the 3200 K node on the rich dwarf track.

Two pincer arms, both under the unchanged production solver: `down` walks
from the gated tomography 3400 K product in 25 K steps (halving to 12.5 K
on failure), `up` walks from the gated 3300 K product. Waypoints need
solver convergence and a finite state only; the 3200 K target needs the
full certification -- primary plus same-path strict self-restart, the
frozen flux gate on both legs, path consistency, and the new
certification phase guard: both legs must have stopped on a
non-worsening p95 flux error.
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

from .cool_star_step_test import (
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from . import m_star_bootstrap_v1r2_marcs100 as base
from . import m_star_iteration_tomography_v1 as tomography
from .m_star_bootstrap_v1 import (
    _annotate_record,
    _load_mt,
    _passes_flux_gate,
    _product_consistency,
    _run_workers,
    _sha256,
    _write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_continuation_fine_step_v1"
ITERATION_CAP = base.ITERATION_CAP
STRICT_ALL_LAYER_LIMIT = base.STRICT_ALL_LAYER_LIMIT
TRACK_SLUG = "g+4.50_m+0.00_a+0.00_c+0.00_x1.00"
TARGET_TEMPERATURE_K = 3200.0
INITIAL_STEP_K = 25.0
MINIMUM_STEP_K = 12.5
TEMPERATURE_TOLERANCE_K = 1.0e-9
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_TOMOGRAPHY_ROOT = REPO_ROOT / "results" / "m_star_iteration_tomography_v1"
DEFAULT_INTERP_ROOT = tomography.DEFAULT_INTERP_ROOT
DEFAULT_V1R2_ROOT = tomography.DEFAULT_V1R2_ROOT
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "notes"
    / "m_star_continuation_fine_step_v1_preregistration_20260905.md"
)

ARMS = {
    "down": {"seed_candidate_id": f"{TRACK_SLUG}_t3400", "seed_temperature_K": 3400.0},
    "up": {"seed_candidate_id": f"{TRACK_SLUG}_t3300", "seed_temperature_K": 3300.0},
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _arm_chain_seed(
    tomography_root: Path, seed_candidate_id: str, seed_temperature_K: float
):
    product = tomography._history_product(
        tomography_root, TRACK_SLUG, seed_temperature_K
    )
    if product is None:
        raise FileNotFoundError(f"tomography product for {seed_candidate_id} missing")
    seed_mass, seed_profile = _load_mt(product)
    track_payload = _read_json(
        tomography._case_dir(tomography_root, TRACK_SLUG, seed_temperature_K)
        / "case.json"
    )["seed"]["track"]
    return seed_mass, seed_profile, dict(track_payload), product


def _step_attempt(
    *,
    track,
    labels,
    initial_atmosphere,
    source_temperature: float,
    target_temperature: float,
    case_root: Path,
    product_subdir: str,
    is_target: bool,
):
    return _solve_attempt(
        track=track,
        method="fine_step_continuation" if not is_target else "fine_step_continuation_target",
        schedule="fine_step_walk",
        source_temperature=source_temperature,
        target_labels=labels,
        initial_atmosphere=initial_atmosphere,
        product_dir=case_root / "products" / product_subdir,
        iteration_cap=ITERATION_CAP,
        maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
    )


def _phase_guard(record: dict[str, Any] | None) -> bool | None:
    if not record:
        return None
    value = (
        (record.get("solver_diagnostics") or {})
        .get("final_diagnostics", {})
        .get("flux_residual_improving_at_stop")
    )
    return None if value is None else bool(value)


def _walk_arm(
    arm: str,
    *,
    result_root: Path,
    tomography_root: Path,
    flux_gate: dict[str, Any],
    protocol_hash: str,
) -> dict[str, Any]:
    _set_single_thread_environment()
    spec = ARMS[arm]
    seed_mass, seed_profile, track_payload, seed_product = _arm_chain_seed(
        tomography_root, spec["seed_candidate_id"], spec["seed_temperature_K"]
    )
    track = base._track_from_payload(track_payload)

    case_root = result_root / "cases" / "dwarf" / TRACK_SLUG / f"t{int(TARGET_TEMPERATURE_K):04d}" / arm
    steps: list[dict[str, Any]] = []
    current_temperature = float(spec["seed_temperature_K"])
    current_mass = seed_mass
    current_profile = seed_profile
    step_size = float(INITIAL_STEP_K)
    primary: dict[str, Any] | None = None
    restart: dict[str, Any] | None = None
    reached = False
    stopped_reason: str | None = None

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
            record, state = _step_attempt(
                track=track,
                labels=labels,
                initial_atmosphere=seed_atmosphere,
                source_temperature=current_temperature,
                target_temperature=proposed,
                case_root=case_root,
                product_subdir="primary" if is_target else "continuation",
                is_target=is_target,
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
    if primary is not None and primary.get("survives_solver") and primary.get("product_path"):
        labels = track.labels(TARGET_TEMPERATURE_K)
        solved_m, solved_t = _load_mt(primary["product_path"])
        restart_seed = _reconstruct_from_mt(labels, solved_m, solved_t)
        restart, _restart_state = _solve_attempt(
            track=track,
            method="fine_step_strict_self_restart",
            schedule="independent_self_restart",
            source_temperature=TARGET_TEMPERATURE_K,
            target_labels=labels,
            initial_atmosphere=restart_seed,
            product_dir=case_root / "products" / "restart",
            iteration_cap=ITERATION_CAP,
            maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
        )
        restart = _annotate_record(
            restart,
            track_payload=track_payload,
            role="train",
            node_id=f"{TRACK_SLUG}_t{int(TARGET_TEMPERATURE_K)}",
        )

    primary_flux = _passes_flux_gate(primary, flux_gate) if primary else {
        "passes": False,
        "metrics": {},
    }
    restart_flux = (
        _passes_flux_gate(restart, flux_gate) if restart else {"passes": False, "metrics": {}}
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
        "protocol_hash": protocol_hash,
        "arm": arm,
        "candidate_id": f"{TRACK_SLUG}_t{int(TARGET_TEMPERATURE_K)}",
        "class": "dwarf",
        "track": track_payload,
        "temperature_K": TARGET_TEMPERATURE_K,
        "labels": track.labels(TARGET_TEMPERATURE_K).as_kwargs(),
        "continuation": {
            "mode": "reduced_rematerialized",
            "initial_step_K": INITIAL_STEP_K,
            "minimum_step_K": MINIMUM_STEP_K,
            "final_step_K": step_size,
            "steps": steps,
            "reached_target": reached,
            "chain_seed": {
                "candidate_id": spec["seed_candidate_id"],
                "temperature_K": spec["seed_temperature_K"],
                "product_path": str(seed_product),
                "product_sha256": _sha256(seed_product),
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
    }
    _write_json(case_root / "case.json", output)
    return output


def _arm_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        arm,
        result_root_text,
        tomography_root_text,
        flux_gate,
        protocol_hash,
    ) = payload
    try:
        return _walk_arm(
            arm,
            result_root=Path(result_root_text),
            tomography_root=Path(tomography_root_text),
            flux_gate=flux_gate,
            protocol_hash=protocol_hash,
        )
    except Exception as exc:  # noqa: BLE001 - a failed arm is an outcome
        return {
            "campaign": CAMPAIGN,
            "protocol_hash": protocol_hash,
            "arm": arm,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    result_root = Path(args.result_root)
    gate = _read_json(Path(args.tomography_root) / "flux_gate.json")
    protocol = {
        "campaign": CAMPAIGN,
        "preregistration": str(PREREGISTRATION_PATH),
        "arms": ARMS,
        "target": {"track": TRACK_SLUG, "temperature_K": TARGET_TEMPERATURE_K},
        "step_policy": {
            "initial_step_K": INITIAL_STEP_K,
            "minimum_step_K": MINIMUM_STEP_K,
        },
        "certification": {
            "phase_guard_required": True,
            "iteration_cap": int(ITERATION_CAP),
            "maximum_all_layer_relative_temperature_change": float(
                STRICT_ALL_LAYER_LIMIT
            ),
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
    _write_json(result_root / "flux_gate.json", gate)
    print(f"protocol hash {protocol['protocol_hash']}")
    return protocol


def run_campaign(args: argparse.Namespace) -> int:
    result_root = Path(args.result_root)
    protocol = _read_json(result_root / "protocol.json")
    gate = _read_json(result_root / "flux_gate.json")
    payloads = [
        (
            arm,
            str(result_root),
            str(Path(args.tomography_root)),
            gate,
            protocol["protocol_hash"],
        )
        for arm in ARMS
    ]
    started = time.perf_counter()
    records = _run_workers(_arm_worker, payloads, workers=int(args.workers))
    for record in sorted(records, key=lambda row: row.get("arm", "")):
        print(
            "{arm}: eligible={elig} reached={reached} reason={reason}".format(
                arm=record.get("arm"),
                elig=record.get("training_eligible"),
                reached=(record.get("continuation") or {}).get("reached_target"),
                reason=record.get("failure_reason") or record.get("error"),
            )
        )
    print(f"wall seconds {time.perf_counter() - started:.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    common.add_argument("--tomography-root", default=str(DEFAULT_TOMOGRAPHY_ROOT))
    common.add_argument("--interp-root", default=str(DEFAULT_INTERP_ROOT))
    common.add_argument("--v1r2-root", default=str(DEFAULT_V1R2_ROOT))
    common.add_argument("--workers", type=int, default=2)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("protocol", parents=[common])
    sub.add_parser("run", parents=[common])
    args = parser.parse_args(argv)

    if args.stage == "protocol":
        run_protocol(args)
        return 0
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
