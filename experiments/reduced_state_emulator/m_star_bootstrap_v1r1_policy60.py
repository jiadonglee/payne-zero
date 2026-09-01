"""Versioned 60-iteration follow-up to the M-star emulator v1 campaign.

This runner preserves the v1 track split and every frozen truth-admission gate.
It changes only the solver iteration ceiling from 30 to 60.  A separate
strict-settling diagnostic is available for the two 4000 K training dwarfs
that converged in v1 but failed primary/restart path consistency.  Diagnostic
products are never admitted to the v1r1 training corpus.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .cool_star_step_test import (
    TrackSpec,
    _production_atmosphere,
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from .m_star_bootstrap_v1 import (
    FLUX_METRICS,
    _annotate_record,
    _hash_payload,
    _load_mt,
    _passes_flux_gate,
    _product_consistency,
    _run_workers,
    _selected_tracks,
    _sha256,
    _spine_worker,
    _write_json,
    build_cool_corpus,
    summarize_open_spine,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_emulator_v1r1_policy60"
PARENT_CAMPAIGN = "m_star_emulator_v1"
ITERATION_CAP = 60
STRICT_ALL_LAYER_LIMIT = 5.0e-4
DEFAULT_PARENT_ROOT = REPO_ROOT / "results" / PARENT_CAMPAIGN
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_SETTLING_ROOT = (
    REPO_ROOT / "results" / "m_star_emulator_v1r1_settling_diagnostic"
)
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "notes"
    / "m_star_emulator_v1r1_policy60_preregistration_20260831.md"
)
SETTLING_LOGG = (4.5, 5.0)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _parent_inputs(parent_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = parent_root / "protocol.json"
    gate_path = parent_root / "flux_gate.json"
    if not protocol_path.is_file() or not gate_path.is_file():
        raise FileNotFoundError(
            f"parent protocol/gate missing under {parent_root}"
        )
    protocol = _read_json(protocol_path)
    gate = _read_json(gate_path)
    if protocol.get("campaign") != PARENT_CAMPAIGN:
        raise ValueError(f"unexpected parent campaign: {protocol.get('campaign')}")
    if not gate.get("frozen") or gate.get("status") != "pass":
        raise ValueError("parent flux gate is not a frozen pass")
    if set(gate.get("thresholds", {})) != set(FLUX_METRICS):
        raise ValueError("parent flux gate does not contain the expected metrics")
    return protocol, gate


def protocol_payload(result_root: Path, parent_root: Path) -> dict[str, Any]:
    parent_protocol, parent_gate = _parent_inputs(parent_root)
    payload = {
        "campaign": CAMPAIGN,
        "status": "preregistered_before_heavy_solver",
        "source": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__)),
            "preregistration": str(PREREGISTRATION_PATH),
            "preregistration_sha256": (
                _sha256(PREREGISTRATION_PATH)
                if PREREGISTRATION_PATH.is_file()
                else None
            ),
        },
        "parent": {
            "campaign": PARENT_CAMPAIGN,
            "result_root": str(parent_root),
            "protocol_hash": parent_protocol["protocol_hash"],
            "protocol_sha256": _sha256(parent_root / "protocol.json"),
            "flux_gate_hash": parent_gate["gate_hash"],
            "flux_gate_sha256": _sha256(parent_root / "flux_gate.json"),
        },
        "change_from_parent": {
            "only_truth_campaign_change": "iteration cap 30 -> 60",
            "track_split_changed": False,
            "flux_gate_changed": False,
            "path_consistency_gate_changed": False,
            "stopping_rule_changed": False,
        },
        "grid": copy.deepcopy(parent_protocol["grid"]),
        "split": copy.deepcopy(parent_protocol["split"]),
        "reference_flux_gate": {
            **copy.deepcopy(parent_protocol["reference_flux_gate"]),
            "thresholds_imported_from_parent": True,
            "thresholds_refit": False,
            "source_gate_hash": parent_gate["gate_hash"],
        },
        "solver": {
            **copy.deepcopy(parent_protocol["solver"]),
            "iteration_cap": ITERATION_CAP,
            "within_15_iterations_recorded": True,
            "stopping_rule": "unchanged production solver stopping rule",
        },
        "training_eligibility": copy.deepcopy(
            parent_protocol["training_eligibility"]
        ),
        "strict_settling_diagnostic": {
            "separate_from_training_corpus": True,
            "logg": list(SETTLING_LOGG),
            "effective_temperature_K": 4000.0,
            "iteration_cap": ITERATION_CAP,
            "maximum_all_layer_relative_temperature_change": (
                STRICT_ALL_LAYER_LIMIT
            ),
            "truth_admission_effect": "none",
        },
        "boundaries": {
            "production_routing_changed": False,
            "existing_sealed_holdout_opened": False,
            "new_sealed_tracks_run": False,
            "marcs_is_training_target": False,
            "korg_run": False,
        },
        "paths": {
            "result_root": str(result_root),
            "protocol": str(result_root / "protocol.json"),
            "flux_gate": str(result_root / "flux_gate.json"),
            "solar_spine": str(result_root / "solar_spine"),
            "cool_corpus": str(result_root / "cool_truth_corpus.npz"),
        },
    }
    payload["protocol_hash"] = _hash_payload(payload)
    return payload


def import_parent_flux_gate(
    result_root: Path,
    *,
    parent_root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    _parent_protocol, parent_gate = _parent_inputs(parent_root)
    payload = {
        "campaign": CAMPAIGN,
        "status": "pass",
        "frozen": True,
        "protocol_hash": protocol["protocol_hash"],
        "thresholds": copy.deepcopy(parent_gate["thresholds"]),
        "source": {
            "campaign": PARENT_CAMPAIGN,
            "path": str(parent_root / "flux_gate.json"),
            "gate_hash": parent_gate["gate_hash"],
            "sha256": _sha256(parent_root / "flux_gate.json"),
        },
        "thresholds_refit": False,
    }
    payload["gate_hash"] = _hash_payload(payload)
    _write_json(result_root / "flux_gate.json", payload)
    return payload


def _calibration_worker(
    payload: tuple[dict[str, Any], str, str],
) -> dict[str, Any]:
    track_payload, result_root_text, protocol_hash = payload
    _set_single_thread_environment()
    track = TrackSpec(
        log_surface_gravity=float(track_payload["log_surface_gravity"]),
        metallicity=float(track_payload["metallicity"]),
        alpha_enhancement=float(track_payload["alpha_enhancement"]),
        carbon_enhancement=float(track_payload["carbon_enhancement"]),
        microturbulence_km_s=float(track_payload["microturbulence_km_s"]),
    )
    track_root = Path(result_root_text) / "reference" / track.track_id
    records = []
    for temperature in (4500.0, 4000.0):
        labels = track.labels(temperature)
        node_id = f"{track.track_id}_t{int(temperature):04d}"
        try:
            initial = _production_atmosphere(labels)
            record, _state = _solve_attempt(
                track=track,
                method="production_reference_direct_policy60",
                schedule="reference_direct_policy60",
                source_temperature=None,
                target_labels=labels,
                initial_atmosphere=initial,
                product_dir=track_root / "products",
                iteration_cap=ITERATION_CAP,
            )
        except Exception as exc:  # noqa: BLE001 - retained campaign outcome
            record = {
                "method": "production_reference_direct_policy60",
                "schedule": "reference_direct_policy60",
                "labels": labels.as_kwargs(),
                "target_temperature": temperature,
                "survives_solver": False,
                "status": "initializer_or_solver_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "iterations": None,
                "product_path": None,
            }
        records.append(
            _annotate_record(
                record,
                track_payload=track_payload,
                role=str(track_payload["role"]),
                node_id=node_id,
            )
        )
    output = {
        "campaign": CAMPAIGN,
        "track": track_payload,
        "protocol_hash": protocol_hash,
        "iteration_cap": ITERATION_CAP,
        "records": records,
    }
    _write_json(track_root / "reference.json", output)
    return output


def _settling_worker(
    payload: tuple[dict[str, Any], str, dict[str, Any], str],
) -> dict[str, Any]:
    track_payload, output_root_text, flux_gate, protocol_hash = payload
    _set_single_thread_environment()
    track = TrackSpec(
        log_surface_gravity=float(track_payload["log_surface_gravity"]),
        metallicity=float(track_payload["metallicity"]),
        alpha_enhancement=float(track_payload["alpha_enhancement"]),
        carbon_enhancement=float(track_payload["carbon_enhancement"]),
        microturbulence_km_s=float(track_payload["microturbulence_km_s"]),
    )
    labels = track.labels(4000.0)
    track_root = Path(output_root_text) / track.track_id
    primary, _primary_state = _solve_attempt(
        track=track,
        method="strict_all_layer_primary",
        schedule="diagnostic_4000K",
        source_temperature=None,
        target_labels=labels,
        initial_atmosphere=_production_atmosphere(labels),
        product_dir=track_root / "products" / "primary",
        iteration_cap=ITERATION_CAP,
        maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
    )
    primary = _annotate_record(
        primary,
        track_payload=track_payload,
        role=str(track_payload["role"]),
        node_id=f"{track.track_id}_t4000",
    )
    restart = None
    if primary.get("survives_solver") and primary.get("product_path"):
        solved_m, solved_t = _load_mt(primary["product_path"])
        restart, _restart_state = _solve_attempt(
            track=track,
            method="strict_all_layer_self_restart",
            schedule="diagnostic_4000K_self_restart",
            source_temperature=4000.0,
            target_labels=labels,
            initial_atmosphere=_reconstruct_from_mt(labels, solved_m, solved_t),
            product_dir=track_root / "products" / "restart",
            iteration_cap=ITERATION_CAP,
            maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
        )
        restart = _annotate_record(
            restart,
            track_payload=track_payload,
            role=str(track_payload["role"]),
            node_id=f"{track.track_id}_t4000",
        )
    consistency = _product_consistency(
        primary.get("product_path"),
        None if restart is None else restart.get("product_path"),
    )
    primary_flux = _passes_flux_gate(primary, flux_gate)
    restart_flux = (
        {"passes": False, "metrics": {}}
        if restart is None
        else _passes_flux_gate(restart, flux_gate)
    )
    diagnostic_pass = bool(
        primary.get("survives_solver")
        and restart is not None
        and restart.get("survives_solver")
        and primary_flux["passes"]
        and restart_flux["passes"]
        and consistency["passes"]
    )
    output = {
        "campaign": "m_star_emulator_v1r1_settling_diagnostic",
        "protocol_hash": protocol_hash,
        "track": track_payload,
        "temperature_K": 4000.0,
        "iteration_cap": ITERATION_CAP,
        "maximum_all_layer_relative_temperature_change": STRICT_ALL_LAYER_LIMIT,
        "primary": primary,
        "restart": restart,
        "primary_flux_gate": primary_flux,
        "restart_flux_gate": restart_flux,
        "path_consistency": consistency,
        "diagnostic_pass": diagnostic_pass,
        "admitted_to_training": False,
    }
    _write_json(track_root / "diagnostic.json", output)
    return output


def _write_summary(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    roles: set[str],
) -> dict[str, Any]:
    summary = summarize_open_spine(
        result_root,
        protocol=protocol,
        roles=roles,
    )
    summary["campaign"] = CAMPAIGN
    summary["parent_campaign"] = PARENT_CAMPAIGN
    summary["iteration_cap"] = ITERATION_CAP
    summary.pop("summary_hash", None)
    summary["summary_hash"] = _hash_payload(summary)
    _write_json(result_root / "open_spine_summary.json", summary)
    return summary


def _parse_roles(text: str) -> set[str]:
    roles = {item.strip() for item in text.split(",") if item.strip()}
    unknown = roles - {"train", "validation", "sealed"}
    if unknown or not roles:
        raise ValueError(f"invalid roles: {sorted(unknown or roles)}")
    return roles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "protocol",
            "import-gate",
            "calibrate",
            "spine",
            "summary",
            "build-corpus",
            "settling-diagnostic",
            "all-open",
        ),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--settling-root", type=Path, default=DEFAULT_SETTLING_ROOT)
    parser.add_argument("--roles", default="train,validation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-sealed", action="store_true")
    args = parser.parse_args(argv)

    roles = _parse_roles(args.roles)
    if "sealed" in roles and not args.run_sealed:
        raise SystemExit("sealed tracks require the explicit --run-sealed flag")
    args.result_root.mkdir(parents=True, exist_ok=True)
    protocol_path = args.result_root / "protocol.json"
    if protocol_path.is_file():
        protocol = _read_json(protocol_path)
        if protocol.get("campaign") != CAMPAIGN:
            raise SystemExit("existing protocol belongs to another campaign")
    else:
        protocol = protocol_payload(args.result_root, args.parent_root)
        _write_json(protocol_path, protocol)

    if args.stage == "protocol":
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return 0

    flux_gate_path = args.result_root / "flux_gate.json"
    if args.stage in {"import-gate", "all-open"}:
        flux_gate = import_parent_flux_gate(
            args.result_root,
            parent_root=args.parent_root,
            protocol=protocol,
        )
        print(json.dumps(flux_gate, indent=2, sort_keys=True))
        if args.stage == "import-gate":
            return 0
    else:
        if not flux_gate_path.is_file():
            raise SystemExit("run import-gate before this stage")
        flux_gate = _read_json(flux_gate_path)
        if not flux_gate.get("frozen"):
            raise SystemExit("FAIL_STOP: imported flux gate is not frozen")

    selected = _selected_tracks(protocol, roles)
    if args.stage in {"calibrate", "all-open"}:
        started = time.perf_counter()
        payloads = [
            (row, str(args.result_root), protocol["protocol_hash"])
            for row in selected
        ]
        outputs = _run_workers(
            _calibration_worker,
            payloads,
            workers=args.workers,
        )
        print(
            f"reference tracks={len(outputs)} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
        if args.stage == "calibrate":
            return 0

    if args.stage in {"spine", "all-open"}:
        started = time.perf_counter()
        payloads = [
            (
                row,
                str(args.result_root),
                ITERATION_CAP,
                flux_gate,
                protocol["protocol_hash"],
            )
            for row in selected
        ]
        outputs = _run_workers(_spine_worker, payloads, workers=args.workers)
        print(
            f"spine tracks={len(outputs)} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
        if args.stage == "spine":
            return 0

    if args.stage == "settling-diagnostic":
        args.settling_root.mkdir(parents=True, exist_ok=True)
        diagnostic_tracks = [
            row
            for row in selected
            if row["class"] == "dwarf"
            and float(row["log_surface_gravity"]) in SETTLING_LOGG
        ]
        outputs = _run_workers(
            _settling_worker,
            [
                (
                    row,
                    str(args.settling_root),
                    flux_gate,
                    protocol["protocol_hash"],
                )
                for row in diagnostic_tracks
            ],
            workers=min(args.workers, len(diagnostic_tracks)),
        )
        summary = {
            "campaign": "m_star_emulator_v1r1_settling_diagnostic",
            "protocol_hash": protocol["protocol_hash"],
            "result_count": len(outputs),
            "pass_count": sum(bool(row["diagnostic_pass"]) for row in outputs),
            "admitted_to_training": False,
            "results": outputs,
        }
        summary["summary_hash"] = _hash_payload(summary)
        _write_json(args.settling_root / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.stage in {"summary", "all-open"}:
        summary = _write_summary(
            args.result_root,
            protocol=protocol,
            roles=roles,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.stage == "summary":
            return 0

    if args.stage in {"build-corpus", "all-open"}:
        corpus = build_cool_corpus(
            args.result_root,
            protocol=protocol,
            roles=roles,
            allow_sealed=args.run_sealed,
        )
        print(json.dumps(corpus, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"unhandled stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
