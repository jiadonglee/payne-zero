"""Fill the open giant-track gaps inside the 3500-4000 K safe zone.

The v1r2 campaign stopped at its 50-row giant quota with 53 eligible giants.
The certified cold-star inventory then identified the standard giant nodes on
open train tracks between 3500 K and 4000 K that still have no eligible
product.  This campaign runs exactly those 17 nodes with the unchanged
truth-generation machinery: same-node native MARCS ``(m,T)`` seed, cap 60,
strict self-restart leg, and the same imported frozen flux gate.  No quota,
no dwarf candidates, and no re-adjudication of any earlier campaign record.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .cool_star_step_test import (
    TrackSpec,
    _marcs_diagnostics,
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from .marcs_h5 import inspect_marcs_grid, load_marcs_node
from .m_star_bootstrap_v1 import (
    FLUX_METRICS,
    PATH_COLUMN_MASS_P95_DEX_LIMIT,
    PATH_TEMPERATURE_P95_LIMIT,
    _annotate_record,
    _hash_payload,
    _load_mt,
    _passes_flux_gate,
    _product_consistency,
    _run_workers,
    _sha256,
    _write_json,
)
from .m_star_bootstrap_v1r2_marcs100 import (
    STRICT_ALL_LAYER_LIMIT,
    _read_json,
    _track_from_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_giant_supplement_v1"
ITERATION_CAP = 60
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
DEFAULT_FLUX_PARENT_ROOT = REPO_ROOT / "results" / "m_star_emulator_v1"

MICROTURBULENCE = 2.0

# The open-track gap list emitted by m_star_cold_library_inventory_v1
# (results/m_star_cold_library_inventory_v1/inventory.json).
GAP_NODES = (
    (0.5, 0.0, 3750.0),
    (0.5, 0.0, 3800.0),
    (0.5, 0.0, 3900.0),
    (0.5, -0.5, 3750.0),
    (0.5, -0.5, 3800.0),
    (0.5, -0.5, 3900.0),
    (0.5, -1.0, 3500.0),
    (0.5, -1.0, 3750.0),
    (0.5, -1.0, 3800.0),
    (0.5, -1.0, 3900.0),
    (1.5, 0.5, 3750.0),
    (1.5, -1.0, 3900.0),
    (1.5, -1.0, 4000.0),
    (2.5, 0.0, 3800.0),
    (2.5, -0.5, 3500.0),
    (2.5, -0.5, 3800.0),
    (2.5, -1.0, 3800.0),
)


def build_candidates() -> list[dict[str, Any]]:
    candidates = []
    for priority, (logg, metallicity, teff) in enumerate(GAP_NODES):
        track = TrackSpec(
            log_surface_gravity=float(logg),
            metallicity=float(metallicity),
            alpha_enhancement=0.0,
            carbon_enhancement=0.0,
            microturbulence_km_s=MICROTURBULENCE,
        )
        candidates.append(
            {
                "candidate_id": f"{track.track_id}_t{int(teff):04d}",
                "priority": priority,
                "temperature_K": float(teff),
                "class": "giant",
                "role": "train",
                "track": track.as_json(),
            }
        )
    return candidates


def protocol_payload(
    result_root: Path,
    *,
    marcs_grid: Path,
    flux_parent_root: Path,
) -> dict[str, Any]:
    schema = inspect_marcs_grid(marcs_grid, verify_sha256=True)
    flux_gate_path = flux_parent_root / "flux_gate.json"
    if not flux_gate_path.is_file():
        raise FileNotFoundError(flux_gate_path)
    parent_gate = _read_json(flux_gate_path)
    if not parent_gate.get("frozen") or parent_gate.get("status") != "pass":
        raise ValueError("parent flux gate is not a frozen pass")
    if set(parent_gate.get("thresholds", {})) != set(FLUX_METRICS):
        raise ValueError("parent flux gate does not contain the expected metrics")

    candidates = build_candidates()
    payload = {
        "campaign": CAMPAIGN,
        "status": "preregistered_before_heavy_solver",
        "source": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__)),
            "inventory": str(
                REPO_ROOT
                / "results"
                / "m_star_cold_library_inventory_v1"
                / "inventory.json"
            ),
        },
        "marcs_seed": {
            "path": str(schema.path),
            "sha256": schema.sha256,
            "native_nodes_only": True,
            "depth_coordinate": "log_mass",
            "fields_passed_to_payne_zero": ["column_mass", "temperature"],
            "is_training_target": False,
        },
        "grid": {
            "candidates": candidates,
            "candidate_count": len(candidates),
            "selection": "open train tracks missing standard nodes in 3500-4000 K",
            "alpha_enhancement": 0.0,
            "carbon_enhancement": 0.0,
            "microturbulence_km_s": MICROTURBULENCE,
        },
        "split": {
            "unit": "single (logg, metallicity, Teff) node",
            "new_train_candidates": candidates,
            "new_validation_tracks_run": False,
            "new_sealed_tracks_run": False,
        },
        "quota": {
            "target_total_training_rows": len(candidates),
            "pool_exhaustion": "not applicable; the gap list is fixed",
        },
        "solver": {
            "seed": "same-node native MARCS (m,T)",
            "truth_source": "terminal Payne-Zero ATLAS atmosphere",
            "independent_nodes": True,
            "continuation": False,
            "iteration_cap": ITERATION_CAP,
            "maximum_all_layer_relative_temperature_change": (
                STRICT_ALL_LAYER_LIMIT
            ),
            "independent_restart": (
                "strict self-restart from terminal ATLAS (m,T)"
            ),
        },
        "training_eligibility": {
            "primary_and_restart_must_converge": True,
            "finite_positive_monotone_six_field_state": True,
            "primary_and_restart_must_pass_imported_flux_gate": True,
            "restart_temperature_relative_p95_max": (
                PATH_TEMPERATURE_P95_LIMIT
            ),
            "restart_column_mass_p95_dex_max": (
                PATH_COLUMN_MASS_P95_DEX_LIMIT
            ),
            "failed_or_unselected_rows_retained": True,
        },
        "imported_flux_gate": {
            "path": str(flux_gate_path),
            "sha256": _sha256(flux_gate_path),
            "gate_hash": parent_gate["gate_hash"],
            "thresholds": parent_gate["thresholds"],
            "thresholds_refit": False,
        },
        "boundaries": {
            "production_routing_changed": False,
            "existing_sealed_holdout_opened": False,
            "earlier_campaign_records_mutated": False,
            "marcs_is_training_target": False,
        },
        "paths": {
            "result_root": str(result_root),
            "protocol": str(result_root / "protocol.json"),
            "flux_gate": str(result_root / "flux_gate.json"),
            "cases": str(result_root / "cases"),
        },
    }
    payload["protocol_hash"] = _hash_payload(payload)
    return payload


def import_flux_gate(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    flux_parent_root: Path,
) -> dict[str, Any]:
    parent_path = flux_parent_root / "flux_gate.json"
    parent = _read_json(parent_path)
    if not parent.get("frozen") or parent.get("status") != "pass":
        raise ValueError("parent flux gate is not a frozen pass")
    payload = {
        "campaign": CAMPAIGN,
        "status": "pass",
        "frozen": True,
        "protocol_hash": protocol["protocol_hash"],
        "thresholds": dict(parent["thresholds"]),
        "source": {
            "path": str(parent_path),
            "sha256": _sha256(parent_path),
            "gate_hash": parent["gate_hash"],
        },
        "thresholds_refit": False,
    }
    payload["gate_hash"] = _hash_payload(payload)
    _write_json(result_root / "flux_gate.json", payload)
    return payload


def _case_path(result_root: Path, candidate: dict[str, Any]) -> Path:
    return (
        result_root
        / "cases"
        / str(candidate["class"])
        / str(candidate["track"]["track_id"])
        / f"t{int(candidate['temperature_K']):04d}"
        / "case.json"
    )


def _failed_case(
    candidate: dict[str, Any],
    *,
    protocol_hash: str,
    error: BaseException | str,
) -> dict[str, Any]:
    message = (
        f"{type(error).__name__}: {error}"
        if isinstance(error, BaseException)
        else str(error)
    )
    return {
        "campaign": CAMPAIGN,
        "protocol_hash": protocol_hash,
        **candidate,
        "marcs_seed": None,
        "primary": None,
        "restart": None,
        "primary_flux_gate": {"passes": False, "metrics": {}},
        "restart_flux_gate": {"passes": False, "metrics": {}},
        "path_consistency": {"available": False, "passes": False},
        "training_eligible": False,
        "selected_for_training": False,
        "status": "failed_before_or_during_solver",
        "failure_reason": message,
    }


def _case_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        candidate,
        result_root_text,
        marcs_grid_text,
        marcs_sha256,
        flux_gate,
        protocol_hash,
    ) = payload
    _set_single_thread_environment()
    result_root = Path(result_root_text)
    case_path = _case_path(result_root, candidate)
    if case_path.is_file():
        return _read_json(case_path)

    track_payload = candidate["track"]
    track = _track_from_payload(track_payload)
    labels = track.labels(float(candidate["temperature_K"]))
    case_root = case_path.parent
    try:
        node = load_marcs_node(
            Path(marcs_grid_text),
            labels,
            carbon_enhancement=float(track.carbon_enhancement),
            verify_sha256=False,
            expected_sha256=None,
        )
        seed = _reconstruct_from_mt(
            labels,
            node.reduced_column_mass,
            node.reduced_temperature,
        )
        primary, _primary_state = _solve_attempt(
            track=track,
            method="native_marcs_same_node_strict_primary",
            schedule="independent_marcs_target",
            source_temperature=None,
            target_labels=labels,
            initial_atmosphere=seed,
            product_dir=case_root / "products" / "primary",
            iteration_cap=ITERATION_CAP,
            maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
        )
        primary = _annotate_record(
            primary,
            track_payload=track_payload,
            role="train",
            node_id=str(candidate["candidate_id"]),
        )
        restart: dict[str, Any] | None = None
        if primary.get("survives_solver") and primary.get("product_path"):
            solved_m, solved_t = _load_mt(primary["product_path"])
            restart_seed = _reconstruct_from_mt(labels, solved_m, solved_t)
            restart, _restart_state = _solve_attempt(
                track=track,
                method="strict_self_restart_from_atlas_mt",
                schedule="independent_self_restart",
                source_temperature=float(candidate["temperature_K"]),
                target_labels=labels,
                initial_atmosphere=restart_seed,
                product_dir=case_root / "products" / "restart",
                iteration_cap=ITERATION_CAP,
                maximum_all_layer_relative_temperature_change=(
                    STRICT_ALL_LAYER_LIMIT
                ),
            )
            restart = _annotate_record(
                restart,
                track_payload=track_payload,
                role="train",
                node_id=str(candidate["candidate_id"]),
            )

        primary_flux = _passes_flux_gate(primary, flux_gate)
        restart_flux = (
            {"passes": False, "metrics": {}}
            if restart is None
            else _passes_flux_gate(restart, flux_gate)
        )
        consistency = _product_consistency(
            primary.get("product_path"),
            None if restart is None else restart.get("product_path"),
        )
        eligible = bool(
            primary.get("survives_solver")
            and restart is not None
            and restart.get("survives_solver")
            and primary.get("state_quality", {}).get("valid")
            and restart.get("state_quality", {}).get("valid")
            and primary_flux["passes"]
            and restart_flux["passes"]
            and consistency["passes"]
        )
        reasons: list[str] = []
        if not primary.get("survives_solver"):
            reasons.append("primary_solver")
        if restart is None or not restart.get("survives_solver"):
            reasons.append("self_restart")
        if not primary_flux["passes"]:
            reasons.append("primary_flux_gate")
        if not restart_flux["passes"]:
            reasons.append("restart_flux_gate")
        if not consistency["passes"]:
            reasons.append("path_consistency")
        output = {
            "campaign": CAMPAIGN,
            "protocol_hash": protocol_hash,
            **candidate,
            "labels": labels.as_kwargs(),
            "marcs_seed": {
                "source_sha256": marcs_sha256,
                "native_indices": list(node.indices),
                "fields_used": ["column_mass", "temperature"],
                "is_training_target": False,
                "diagnostics": _marcs_diagnostics(node),
            },
            "primary": primary,
            "restart": restart,
            "primary_flux_gate": primary_flux,
            "restart_flux_gate": restart_flux,
            "path_consistency": consistency,
            "training_eligible": eligible,
            "selected_for_training": False,
            "status": "training_eligible" if eligible else "ineligible",
            "failure_reason": None if eligible else ",".join(reasons),
        }
    except Exception as exc:  # noqa: BLE001 - retained campaign outcome
        output = _failed_case(
            candidate, protocol_hash=protocol_hash, error=exc
        )
    case_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(case_path, output)
    return output


def campaign_status(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    write: bool = True,
) -> dict[str, Any]:
    candidates = protocol["grid"]["candidates"]
    records = [
        _read_json(path)
        for path in sorted((result_root / "cases").glob("*/*/t*/case.json"))
    ]
    by_id = {str(row["candidate_id"]): row for row in records}
    eligible = [row for row in records if bool(row.get("training_eligible"))]
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
        "complete": not pending,
        "eligible_candidate_ids": [
            row["candidate_id"]
            for row in sorted(eligible, key=lambda item: int(item["priority"]))
        ],
        "ineligible": [
            {
                "candidate_id": str(row["candidate_id"]),
                "failure_reason": row.get("failure_reason"),
            }
            for row in records
            if not bool(row.get("training_eligible"))
        ],
    }
    payload["status_hash"] = _hash_payload(payload)
    if write:
        _write_json(result_root / "status.json", payload)
    return payload


def run_gap(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    flux_gate: dict[str, Any],
    marcs_grid: Path,
    workers: int,
) -> dict[str, Any]:
    candidates = protocol["grid"]["candidates"]
    pending = [
        candidate
        for candidate in candidates
        if not _case_path(result_root, candidate).is_file()
    ]
    if not pending:
        return campaign_status(result_root, protocol=protocol)
    started = time.perf_counter()
    outputs = _run_workers(
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
                    bool(row.get("training_eligible")) for row in outputs
                ),
                "seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return campaign_status(result_root, protocol=protocol)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("protocol", "import-gate", "run", "status"),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--marcs-grid", type=Path, default=DEFAULT_MARCS_GRID)
    parser.add_argument(
        "--flux-parent-root",
        type=Path,
        default=DEFAULT_FLUX_PARENT_ROOT,
    )
    parser.add_argument("--workers", type=int, default=6)
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
            marcs_grid=args.marcs_grid,
            flux_parent_root=args.flux_parent_root,
        )
        _write_json(protocol_path, protocol)

    if args.stage == "protocol":
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return 0

    gate_path = args.result_root / "flux_gate.json"
    if args.stage == "import-gate":
        gate = import_flux_gate(
            args.result_root,
            protocol=protocol,
            flux_parent_root=args.flux_parent_root,
        )
        print(json.dumps(gate, indent=2, sort_keys=True))
        return 0
    if not gate_path.is_file():
        raise SystemExit("run import-gate before truth generation")
    flux_gate = _read_json(gate_path)
    if not flux_gate.get("frozen") or flux_gate.get("thresholds_refit"):
        raise SystemExit("FAIL_STOP: imported flux gate is not frozen")

    if args.stage == "run":
        status = run_gap(
            args.result_root,
            protocol=protocol,
            flux_gate=flux_gate,
            marcs_grid=args.marcs_grid,
            workers=args.workers,
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        if status["complete"]:
            (args.result_root / "GAP_RUN_COMPLETE").touch()
        return 0

    if args.stage == "status":
        status = campaign_status(args.result_root, protocol=protocol)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"unhandled stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
