"""Drive every uncertified scale-out node to certification.

Same pipeline, same physics, same frozen gate; the only additions are
the two certification-strategy tools the campaigns already built:

- ``require_improving_flux_residual`` with a 120-iteration budget, so a
  stop is only accepted when the p95 flux error is not worsening;
- continuation seed routing, so a node whose MARCS-seeded solve
  diverges is approached from its nearest converged neighbour in
  25 -> 12.5 -> 6.25 K steps.

Nodes: giant 4000/2.5 (restart budget), dwarfs 3800/[M/H]-1.0 and
4000/[M/H]0 (phase-aware legs), dwarf 3600/[M/H]-1.0 (continuation
from the converged 3800/[M/H]-1.0 product), dwarf 3500/[M/H]0
(continuation seed from the gated 3600 K product), and the giant
3750/2.5 cap scan at 30/120 whose node id collided in scaleout v1.
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
from . import m_star_iteration_tomography_v1 as tomography
from .cool_star_step_test import (
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from . import m_star_bootstrap_v1r2_marcs100 as base
from .m_star_bootstrap_v1 import (
    _load_mt,
    _run_workers,
    _write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_pipeline_complete_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_GATE_PATH = (
    REPO_ROOT / "results" / "m_star_iteration_tomography_v1" / "flux_gate.json"
)
DEFAULT_SCALEOUT_ROOT = REPO_ROOT / "results" / "m_star_pipeline_scaleout_v1"
DEFAULT_TOMOGRAPHY_ROOT = REPO_ROOT / "results" / "m_star_iteration_tomography_v1"
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "notes"
    / "m_star_pipeline_complete_v1_preregistration_20260905.md"
)

PHASE_AWARE = {"require_improving_flux_residual": True}
CERTIFICATION_CAP = 120
WAYPOINT_INITIAL_STEP_K = 25.0
WAYPOINT_MINIMUM_STEP_K = 6.25
TEMPERATURE_TOLERANCE_K = 1.0e-9

GIANT_TRACK = dict(
    stellar_class="giant",
    log_surface_gravity=2.5,
    metallicity=0.0,
)
DWARF_TRACK_M1 = dict(
    stellar_class="dwarf",
    log_surface_gravity=4.5,
    metallicity=-1.0,
)
DWARF_TRACK_M0 = dict(
    stellar_class="dwarf",
    log_surface_gravity=4.5,
    metallicity=0.0,
)

NODES = [
    {
        "node_id": "complete_giant_g+2.50_m+0.00_t4000",
        "arm": "phase_aware",
        "track": dict(GIANT_TRACK),
        "temperature_K": 4000.0,
        "caps": (1500,),
    },
    {
        "node_id": "complete_dwarf_g+4.50_m-1.00_t3800",
        "arm": "phase_aware",
        "track": dict(DWARF_TRACK_M1),
        "temperature_K": 3800.0,
        "caps": (CERTIFICATION_CAP,),
    },
    {
        "node_id": "complete_dwarf_g+4.50_m+0.00_t4000",
        "arm": "phase_aware",
        "track": dict(DWARF_TRACK_M0),
        "temperature_K": 4000.0,
        "caps": (CERTIFICATION_CAP,),
    },
    {
        "node_id": "complete_dwarf_g+4.50_m-1.00_t3600",
        "arm": "continuation_from_3800",
        "track": dict(DWARF_TRACK_M1),
        "temperature_K": 3600.0,
        "caps": (CERTIFICATION_CAP,),
    },
    {
        "node_id": "complete_dwarf_g+4.50_m+0.00_t3500",
        "arm": "continuation_from_3600",
        "track": dict(DWARF_TRACK_M0),
        "temperature_K": 3500.0,
        "caps": (CERTIFICATION_CAP,),
    },
    {
        "node_id": "capscan_giant_g+2.50_m+0.00_t3750",
        "arm": "cap_scan",
        "track": dict(GIANT_TRACK),
        "temperature_K": 3750.0,
        "caps": (30, 120),
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _waypoint_chain(
    *,
    labels_of,
    start_mass: np.ndarray,
    start_profile: np.ndarray,
    start_temperature: float,
    target_temperature: float,
    track_payload: dict[str, Any],
    case_root: Path,
    initial_step: float,
    minimum_step: float,
):
    """Walk from a converged product to the target in shrinking steps.

    Returns (mass, profile, steps, reached, stopped_reason).
    """

    track = base._track_from_payload(track_payload)
    steps: list[dict[str, Any]] = []
    current_temperature = float(start_temperature)
    current_mass = np.asarray(start_mass, dtype=np.float64)
    current_profile = np.asarray(start_profile, dtype=np.float64)
    step_size = float(initial_step)
    while abs(current_temperature - target_temperature) > TEMPERATURE_TOLERANCE_K:
        proposed = max(target_temperature, current_temperature - step_size)
        labels = labels_of(proposed)
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
                }
            )
            record = None
            state = None
        else:
            record, state = _solve_attempt(
                track=track,
                method="fine_step_waypoint",
                schedule="continuation_walk",
                source_temperature=current_temperature,
                target_labels=labels,
                initial_atmosphere=seed_atmosphere,
                product_dir=case_root / "products" / "continuation",
                iteration_cap=120,
                maximum_all_layer_relative_temperature_change=(
                    pipeline.STRICT_ALL_LAYER_LIMIT
                ),
            )
            steps.append(
                {
                    "status": "solver_pass" if record.get("survives_solver") else "solver_fail",
                    "source_temperature_K": current_temperature,
                    "target_temperature_K": proposed,
                    "requested_step_K": step_size,
                    "iterations": record.get("iterations"),
                    "p95_absolute_flux_error_percent": (
                        (record.get("solver_diagnostics") or {})
                        .get("final_diagnostics", {})
                        .get("p95_absolute_flux_error_percent")
                    ),
                }
            )
        if record is not None and record.get("survives_solver"):
            current_temperature = proposed
            current_mass = np.asarray(state.column_mass, dtype=np.float64)
            current_profile = np.asarray(state.temperature, dtype=np.float64)
            continue
        if step_size <= minimum_step:
            return current_mass, current_profile, steps, False, "minimum_step_failed"
        step_size = max(step_size / 2.0, minimum_step)
    return (
        current_mass,
        current_profile,
        steps,
        True,
        None,
    )


def _case_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        node,
        result_root_text,
        marcs_grid_text,
        scaleout_root_text,
        tomography_root_text,
        v1r2_root_text,
        gate,
        protocol_hash,
    ) = payload
    _set_single_thread_environment()
    result_root = Path(result_root_text)
    case_path = result_root / "cases" / f"{node['node_id']}_complete.json"
    if case_path.is_file():
        existing = _read_json(case_path)
        if existing.get("status") == "complete":
            return existing

    scaleout_root = Path(scaleout_root_text)
    tomography_root = Path(tomography_root_text)
    track = pipeline.track_payload(
        **node["track"],
        microturbulence_km_s=pipeline.MICROTURBULENCE[
            node["track"]["stellar_class"]
        ],
    )
    labels = pipeline.labels_for(track, node["temperature_K"])
    attempts: dict[str, Any] = {}
    notes: dict[str, Any] = {}
    try:
        if node["arm"] == "continuation_from_3800":
            waypoint_dir = (
                scaleout_root
                / "cases"
                / "dwarf_g+4.50_m-1.00_t3800"
                / "cap60"
                / "products"
                / "primary"
            )
            waypoint_products = sorted(waypoint_dir.glob("*.npz"))
            if not waypoint_products:
                raise FileNotFoundError(f"no waypoint product in {waypoint_dir}")
            waypoint_product = waypoint_products[0]
            start_mass, start_profile = _load_mt(waypoint_product)
            chain = _waypoint_chain(
                labels_of=lambda t: pipeline.labels_for(track, t),
                start_mass=start_mass,
                start_profile=start_profile,
                start_temperature=3800.0,
                target_temperature=float(node["temperature_K"]),
                track_payload=track,
                case_root=result_root / "cases" / node["node_id"],
                initial_step=WAYPOINT_INITIAL_STEP_K,
                minimum_step=WAYPOINT_MINIMUM_STEP_K,
            )
            notes["waypoint_chain"] = {
                "steps": chain[2],
                "reached": chain[3],
                "stopped_reason": chain[4],
            }
            if not chain[3]:
                output = {
                    "campaign": CAMPAIGN,
                    "protocol_hash": protocol_hash,
                    **node,
                    "notes": notes,
                    "status": "closed",
                    "failure_reason": chain[4],
                }
                _write_json(case_path, output)
                return output
            mass, profile = chain[0], chain[1]
            seed = _reconstruct_from_mt(labels, mass, profile)
            seed_kind = "continuation_waypoint"
        elif node["arm"] == "continuation_from_3600":
            donor_dir = (
                Path(v1r2_root_text)
                / "cases"
                / "dwarf"
                / "g+4.50_m+0.00_a+0.00_c+0.00_x1.00"
                / "t3600"
                / "products"
                / "primary"
            )
            donor_products = sorted(donor_dir.glob("*.npz"))
            if not donor_products:
                raise FileNotFoundError(f"no gated 3600 K product in {donor_dir}")
            donor = donor_products[0]
            seed = pipeline.continuation_seed(donor, labels)
            seed_kind = f"continuation_seed:{donor.name}"
        else:
            seed = pipeline.marcs_seed(labels, marcs_grid=Path(marcs_grid_text))
            seed_kind = "marcs_native"

        for cap in node["caps"]:
            attempts[str(cap)] = pipeline.certified_solve(
                labels=labels,
                track=track,
                candidate_id=node["node_id"],
                initial_atmosphere=seed,
                product_dir=(
                    result_root / "cases" / node["node_id"] / f"cap{cap}"
                ),
                flux_gate=gate,
                iteration_cap=int(cap),
                solver_overrides=dict(PHASE_AWARE),
            )
        output = {
            "campaign": CAMPAIGN,
            "protocol_hash": protocol_hash,
            **node,
            "seed_kind": seed_kind,
            "attempts": attempts,
            "notes": notes,
            "eligible_by_cap": {
                cap: bool(block["training_eligible"])
                for cap, block in attempts.items()
            },
            "status": "complete",
        }
    except Exception as exc:  # noqa: BLE001 - a failed node is an outcome
        output = {
            "campaign": CAMPAIGN,
            "protocol_hash": protocol_hash,
            **node,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write_json(case_path, output)
    return output


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    result_root = Path(args.result_root)
    gate = _read_json(Path(args.gate_path))
    protocol = {
        "campaign": CAMPAIGN,
        "preregistration": str(PREREGISTRATION_PATH),
        "nodes": [node["node_id"] for node in NODES],
        "certification_cap": CERTIFICATION_CAP,
        "strategy": {
            "phase_aware_stop": PHASE_AWARE,
            "waypoint_steps": [
                WAYPOINT_INITIAL_STEP_K,
                WAYPOINT_MINIMUM_STEP_K,
            ],
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
    print(f"protocol hash {protocol['protocol_hash']}")
    return protocol


def run_campaign(args: argparse.Namespace) -> int:
    result_root = Path(args.result_root)
    protocol = _read_json(result_root / "protocol.json")
    gate = _read_json(Path(args.gate_path))
    payloads = [
        (
            node,
            str(result_root),
            str(Path(args.marcs_grid)),
            str(Path(args.scaleout_root)),
            str(Path(args.tomography_root)),
            str(Path(args.v1r2_root)),
            gate,
            protocol["protocol_hash"],
        )
        for node in NODES
    ]
    started = time.perf_counter()
    records = _run_workers(_case_worker, payloads, workers=int(args.workers))
    for record in sorted(records, key=lambda row: row["node_id"]):
        line = f"{record['node_id']}: status={record.get('status')}"
        if record.get("status") == "complete":
            for cap, block in sorted(
                record.get("attempts", {}).items(), key=lambda kv: int(kv[0])
            ):
                primary = block.get("primary") or {}
                guard = block.get("phase_guard") or {}
                p95 = (
                    (block.get("primary_flux_gate") or {})
                    .get("metrics", {})
                    .get("p95_absolute_flux_error_percent", {})
                    .get("value")
                )
                line += (
                    f" | cap{cap}: elig={block.get('training_eligible')} "
                    f"iters={primary.get('iterations')} "
                    f"p95={round(p95, 2) if p95 is not None else '—'} "
                    f"guard={guard.get('primary')}/{guard.get('restart')}"
                )
        else:
            line += f" reason={record.get('failure_reason') or record.get('error')}"
        print(line)
    print(f"wall seconds {time.perf_counter() - started:.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    common.add_argument(
        "--marcs-grid", default=str(pipeline.DEFAULT_MARCS_GRID)
    )
    common.add_argument("--gate-path", default=str(DEFAULT_GATE_PATH))
    common.add_argument("--scaleout-root", default=str(DEFAULT_SCALEOUT_ROOT))
    common.add_argument(
        "--tomography-root", default=str(DEFAULT_TOMOGRAPHY_ROOT)
    )
    common.add_argument(
        "--v1r2-root",
        default=str(REPO_ROOT / "results" / "m_star_emulator_v1r2_marcs100"),
    )
    common.add_argument("--workers", type=int, default=6)
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
