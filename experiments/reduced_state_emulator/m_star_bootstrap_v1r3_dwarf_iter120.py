"""Recover cap-limited v1r2 dwarf nodes with a 120-iteration ceiling."""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import json
from pathlib import Path
import time
from typing import Any

from . import m_star_bootstrap_v1r2_marcs100 as base
from .marcs_h5 import inspect_marcs_grid


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_emulator_v1r3_dwarf_iter120"
ITERATION_CAP = 120
TARGET_NEW_DWARFS = 24
EXPECTED_PARENT_ELIGIBLE_DWARFS = 26
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_PARENT_ROOT = (
    REPO_ROOT / "results" / "m_star_emulator_v1r2_marcs100"
)
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "notes"
    / "m_star_emulator_v1r3_dwarf_iter120_preregistration_20260903.md"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _select_rescue_candidates(
    parent_protocol: dict[str, Any],
    parent_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates_by_id = {
        str(row["candidate_id"]): row
        for row in parent_protocol["split"]["new_train_candidates"]
    }
    selected: list[dict[str, Any]] = []
    for record in parent_records:
        primary = record.get("primary") or {}
        if (
            record.get("class") == "dwarf"
            and not bool(record.get("training_eligible"))
            and "primary_solver"
            in str(record.get("failure_reason") or "").split(",")
            and int(primary.get("iterations") or 0) == 60
            and bool(primary.get("state_quality", {}).get("valid"))
        ):
            selected.append(candidates_by_id[str(record["candidate_id"])])
    return sorted(selected, key=lambda row: int(row["priority"]))


def protocol_payload(
    result_root: Path,
    *,
    parent_root: Path,
    marcs_grid: Path,
) -> dict[str, Any]:
    parent_protocol = _read_json(parent_root / "protocol.json")
    parent_status = _read_json(parent_root / "status_dwarf.json")
    parent_gate = _read_json(parent_root / "flux_gate.json")
    if parent_protocol.get("campaign") != "m_star_emulator_v1r2_marcs100":
        raise ValueError("unexpected parent campaign")
    if (
        int(parent_status.get("attempted_count", -1)) != 108
        or int(parent_status.get("eligible_count", -1))
        != EXPECTED_PARENT_ELIGIBLE_DWARFS
        or not bool(parent_status.get("pool_exhausted"))
    ):
        raise ValueError("parent dwarf campaign is not at the frozen FAIL_STOP")
    if not parent_gate.get("frozen") or parent_gate.get("thresholds_refit"):
        raise ValueError("parent flux gate is not frozen")

    schema = inspect_marcs_grid(
        marcs_grid,
        verify_sha256=True,
        expected_sha256=parent_protocol["marcs_seed"]["sha256"],
    )
    parent_records = base.load_case_records(parent_root)
    rescue_candidates = _select_rescue_candidates(
        parent_protocol,
        parent_records,
    )
    payload: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "status": "preregistered_before_heavy_solver",
        "source": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": base._sha256(Path(__file__)),
            "preregistration": str(PREREGISTRATION_PATH),
            "preregistration_sha256": (
                base._sha256(PREREGISTRATION_PATH)
                if PREREGISTRATION_PATH.is_file()
                else None
            ),
        },
        "parent": {
            "campaign": parent_protocol["campaign"],
            "result_root": str(parent_root.resolve()),
            "protocol_hash": parent_protocol["protocol_hash"],
            "status_dwarf_sha256": base._sha256(
                parent_root / "status_dwarf.json"
            ),
            "existing_eligible_dwarf_count": (
                EXPECTED_PARENT_ELIGIBLE_DWARFS
            ),
        },
        "marcs_seed": {
            "path": str(schema.path),
            "sha256": schema.sha256,
            "fields_passed_to_payne_zero": [
                "column_mass",
                "temperature",
            ],
            "is_training_target": False,
        },
        "rescue_pool": {
            "selection": (
                "v1r2 ineligible dwarf nodes that reached iteration 60 "
                "with a valid finite terminal six-field state"
            ),
            "candidate_count": len(rescue_candidates),
            "candidates": rescue_candidates,
            "priority": "retain the frozen v1r2 candidate priority",
        },
        "target": {
            "new_eligible_dwarfs": TARGET_NEW_DWARFS,
            "combined_eligible_dwarfs": (
                EXPECTED_PARENT_ELIGIBLE_DWARFS + TARGET_NEW_DWARFS
            ),
            "batch_overshoot": "reserve only",
        },
        "solver": {
            "seed": "same-node native MARCS (m,T)",
            "truth_source": "terminal Payne-Zero ATLAS atmosphere",
            "continuation": False,
            "iteration_cap": ITERATION_CAP,
            "maximum_all_layer_relative_temperature_change": (
                base.STRICT_ALL_LAYER_LIMIT
            ),
            "independent_restart": (
                "strict self-restart from terminal ATLAS (m,T)"
            ),
            "only_change_from_parent_attempt": (
                "primary and self-restart iteration ceiling 60 -> 120"
            ),
        },
        "training_eligibility": dict(
            parent_protocol["training_eligibility"]
        ),
        "imported_flux_gate": {
            "parent_gate_hash": parent_gate["gate_hash"],
            "thresholds": dict(parent_gate["thresholds"]),
            "thresholds_refit": False,
        },
        "boundaries": {
            "parent_results_mutated": False,
            "training_run": False,
            "candidate_validation_run": False,
            "sealed_holdout_opened": False,
            "production_routing_changed": False,
            "korg_run": False,
            "marcs_is_training_target": False,
        },
        "paths": {
            "result_root": str(result_root.resolve()),
            "protocol": str((result_root / "protocol.json").resolve()),
            "flux_gate": str((result_root / "flux_gate.json").resolve()),
            "cases": str((result_root / "cases").resolve()),
        },
    }
    payload["protocol_hash"] = base._hash_payload(payload)
    return payload


def _case_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    base.CAMPAIGN = CAMPAIGN
    base.ITERATION_CAP = ITERATION_CAP
    return base._case_worker(payload)


def rescue_status(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    write: bool = True,
) -> dict[str, Any]:
    candidates = protocol["rescue_pool"]["candidates"]
    records = base.load_case_records(result_root)
    by_id = {str(row["candidate_id"]): row for row in records}
    eligible = sorted(
        (row for row in records if bool(row.get("training_eligible"))),
        key=lambda row: int(row["priority"]),
    )
    pending = [
        row
        for row in candidates
        if str(row["candidate_id"]) not in by_id
    ]
    payload = {
        "campaign": CAMPAIGN,
        "protocol_hash": protocol["protocol_hash"],
        "candidate_count": len(candidates),
        "attempted_count": len(records),
        "eligible_count": len(eligible),
        "pending_count": len(pending),
        "target_new_eligible": TARGET_NEW_DWARFS,
        "target_reached": len(eligible) >= TARGET_NEW_DWARFS,
        "pool_exhausted": not pending and len(eligible) < TARGET_NEW_DWARFS,
        "combined_dwarf_eligible_count": (
            EXPECTED_PARENT_ELIGIBLE_DWARFS
            + min(len(eligible), TARGET_NEW_DWARFS)
        ),
        "eligible_candidate_ids": [
            row["candidate_id"] for row in eligible
        ],
        "pending_candidates": pending,
    }
    payload["status_hash"] = base._hash_payload(payload)
    if write:
        base._write_json(result_root / "status.json", payload)
    return payload


def run_until_target(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    flux_gate: dict[str, Any],
    marcs_grid: Path,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    while True:
        status = rescue_status(result_root, protocol=protocol)
        if status["target_reached"]:
            (result_root / "RESCUE_TARGET_REACHED").touch()
            return status
        if status["pool_exhausted"]:
            (result_root / "RESCUE_POOL_EXHAUSTED").touch()
            return status
        pending = status["pending_candidates"][:batch_size]
        started = time.perf_counter()
        outputs = base._run_workers(
            _case_worker,
            [
                (
                    candidate,
                    str(result_root),
                    str(marcs_grid),
                    protocol["marcs_seed"]["sha256"],
                    flux_gate,
                    protocol["protocol_hash"],
                )
                for candidate in pending
            ],
            workers=min(workers, len(pending)),
        )
        print(
            json.dumps(
                {
                    "attempted_in_batch": len(outputs),
                    "eligible_in_batch": sum(
                        bool(row.get("training_eligible"))
                        for row in outputs
                    ),
                    "seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("protocol", "import-gate", "run", "status"),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--marcs-grid", type=Path, default=DEFAULT_MARCS_GRID)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args(argv)

    args.result_root.mkdir(parents=True, exist_ok=True)
    protocol_path = args.result_root / "protocol.json"
    if protocol_path.is_file():
        protocol = _read_json(protocol_path)
        if protocol.get("campaign") != CAMPAIGN:
            raise SystemExit("existing protocol belongs to another campaign")
    else:
        protocol = protocol_payload(
            args.result_root,
            parent_root=args.parent_root,
            marcs_grid=args.marcs_grid,
        )
        base._write_json(protocol_path, protocol)

    if args.stage == "protocol":
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return 0

    gate_path = args.result_root / "flux_gate.json"
    if args.stage == "import-gate":
        base.CAMPAIGN = CAMPAIGN
        gate = base.import_flux_gate(
            args.result_root,
            protocol=protocol,
            flux_parent_root=args.parent_root,
        )
        print(json.dumps(gate, indent=2, sort_keys=True))
        return 0
    if not gate_path.is_file():
        raise SystemExit("run import-gate before the rescue")
    flux_gate = _read_json(gate_path)

    if args.stage == "run":
        status = run_until_target(
            args.result_root,
            protocol=protocol,
            flux_gate=flux_gate,
            marcs_grid=args.marcs_grid,
            batch_size=args.batch_size,
            workers=args.workers,
        )
    else:
        status = rescue_status(args.result_root, protocol=protocol)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["target_reached"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
