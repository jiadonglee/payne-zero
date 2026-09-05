"""Warm-start failed v1r2 dwarfs from same-track gated ATLAS ``(m,T)``.

Donors are training-eligible dwarf nodes from the immutable v1r2 campaign.
Only column mass and temperature are mixed; Payne-Zero reconstructs the other
fields.  Solver settings and admission gates are copied from v1r2.  Cross-track
kNN is out of scope for this arm.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
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
from .m_star_bootstrap_v1 import (
    _annotate_record,
    _hash_payload,
    _load_mt,
    _passes_flux_gate,
    _product_consistency,
    _run_workers,
    _sha256,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_interpolated_mt_seed_v1"
ITERATION_CAP = base.ITERATION_CAP
STRICT_ALL_LAYER_LIMIT = base.STRICT_ALL_LAYER_LIMIT
EXPECTED_PARENT_ELIGIBLE_DWARFS = 26
EXPECTED_PARENT_FAILED_DWARFS = 82
EXPECTED_SAME_TRACK_FAILURES = 58
EXPECTED_NO_DONOR_FAILURES = 24
TARGET_NEW_DWARFS = 24
EASY_DELTA_T_MAX = 100.0
HARD_DELTA_T_MIN = 200.0
HARD_DELTA_T_MAX = 300.0
PROBE_PER_BIN = 3
SCALE_EASY_MIN_ELIGIBLE = 2
LOG_TEFF_POWER = 1.0
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_PARENT_ROOT = REPO_ROOT / "results" / "m_star_emulator_v1r2_marcs100"
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "notes"
    / "m_star_interpolated_mt_seed_v1_preregistration_20260903.md"
)
EXPECTED_PROBE_IDS = (
    "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3500",
    "g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3800",
    "g+5.00_m+0.50_a+0.00_c+0.00_x1.00_t3300",
    "g+4.50_m-1.00_a+0.00_c+0.00_x1.00_t3500",
    "g+5.00_m+0.00_a+0.00_c+0.00_x1.00_t3500",
    "g+5.50_m+0.50_a+0.00_c+0.00_x1.00_t3500",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def track_key(payload: dict[str, Any]) -> tuple[float, float, float]:
    track = payload["track"] if "track" in payload else payload
    return (
        float(track["log_surface_gravity"]),
        float(track["metallicity"]),
        float(track["microturbulence_km_s"]),
    )


def select_same_track_donor_indices(
    target_teff: float,
    donor_teffs: np.ndarray,
) -> dict[str, Any]:
    temperatures = np.asarray(donor_teffs, dtype=np.float64)
    if temperatures.size == 0:
        raise ValueError("same-track interpolation requires at least one donor")
    matches = np.where(np.isclose(temperatures, float(target_teff)))[0]
    if matches.size:
        index = int(matches[0])
        return {
            "kind": "exact",
            "indices": [index],
            "outside_convex_hull": False,
        }
    cooler = np.where(temperatures < float(target_teff))[0]
    hotter = np.where(temperatures > float(target_teff))[0]
    hull_min = float(np.min(temperatures))
    hull_max = float(np.max(temperatures))
    outside = bool(
        float(target_teff) < hull_min or float(target_teff) > hull_max
    )
    if cooler.size and hotter.size:
        left = int(cooler[np.argmax(temperatures[cooler])])
        right = int(hotter[np.argmin(temperatures[hotter])])
        return {
            "kind": "bracketed",
            "indices": [left, right],
            "outside_convex_hull": False,
        }
    nearest = int(np.argmin(np.abs(temperatures - float(target_teff))))
    return {
        "kind": "one_sided",
        "indices": [nearest],
        "outside_convex_hull": True,
    }


def _mix_log_mt(
    donor_reduced: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Log mix of ``(m, T)`` used by ``interpolated_reduced_state``."""

    log_mass = np.log10(np.maximum(donor_reduced[:, :, 0], 1.0e-300))
    log_temperature = np.log10(np.maximum(donor_reduced[:, :, 1], 1.0e-300))
    return (
        10.0 ** (weights @ log_mass),
        10.0 ** (weights @ log_temperature),
    )


def interpolate_same_track_mt(
    target_labels: dict[str, float],
    donor_labels: list[dict[str, float]],
    donor_reduced: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    donor_teffs = np.asarray(
        [float(row["effective_temperature"]) for row in donor_labels],
        dtype=np.float64,
    )
    selection = select_same_track_donor_indices(
        float(target_labels["effective_temperature"]),
        donor_teffs,
    )
    indices = selection["indices"]
    subset = np.asarray(donor_reduced, dtype=np.float64)[indices]
    if len(indices) == 1:
        weights = np.ones(1, dtype=np.float64)
    else:
        log_teff = np.log(
            np.asarray([donor_teffs[index] for index in indices], dtype=np.float64)
        )
        target_log = np.log(float(target_labels["effective_temperature"]))
        fraction = (target_log - log_teff[0]) / (log_teff[1] - log_teff[0])
        weights = np.asarray([1.0 - fraction, fraction], dtype=np.float64)
    column_mass, temperature = _mix_log_mt(subset, weights)
    nearest_index = int(
        np.argmin(np.abs(donor_teffs - float(target_labels["effective_temperature"])))
    )
    diagnostics = {
        "neighbours": len(indices),
        "power": LOG_TEFF_POWER,
        "weights": [float(value) for value in weights],
        "kind": selection["kind"],
        "outside_convex_hull": bool(selection["outside_convex_hull"]),
        "donor_effective_temperatures": [
            float(donor_teffs[index]) for index in indices
        ],
        "nearest_donor_effective_temperature": float(donor_teffs[nearest_index]),
        "delta_t_K": float(
            abs(
                donor_teffs[nearest_index]
                - float(target_labels["effective_temperature"])
            )
        ),
    }
    return column_mass, temperature, diagnostics


def probe_bin(delta_t_K: float) -> str:
    gap = float(delta_t_K)
    if gap <= EASY_DELTA_T_MAX:
        return "easy"
    if HARD_DELTA_T_MIN <= gap <= HARD_DELTA_T_MAX:
        return "hard"
    return "other"


def _dwarf_records(parent_root: Path) -> list[dict[str, Any]]:
    return [
        row
        for row in base.load_case_records(parent_root)
        if row.get("class") == "dwarf"
    ]


def _candidate_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": record["candidate_id"],
        "priority": int(record["priority"]),
        "temperature_K": float(record["temperature_K"]),
        "class": "dwarf",
        "role": record.get("role", "train"),
        "track": dict(record["track"]),
        "labels": dict(record["labels"]),
    }


def _resolve_parent_product(parent_root: Path, record: dict[str, Any]) -> Path:
    recorded = Path(record["primary"]["product_path"])
    if recorded.is_file():
        return recorded
    case_path = base._case_path(parent_root, record)
    local = case_path.parent / "products" / "primary" / recorded.name
    if not local.is_file():
        raise FileNotFoundError(local)
    return local


def _flux_metric(record: dict[str, Any] | None, name: str) -> float | None:
    if not record:
        return None
    gate = record.get("primary_flux_gate") or {}
    item = (gate.get("metrics") or {}).get(name) or {}
    value = item.get("value")
    if value is None:
        diagnostics = (
            (record.get("primary") or {})
            .get("solver_diagnostics", {})
            .get("final_diagnostics", {})
        )
        value = diagnostics.get(name)
    if value is None:
        return None
    return float(value)


def build_neighbor_table(
    parent_root: Path,
    *,
    parent_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = (
        _dwarf_records(parent_root) if parent_records is None else parent_records
    )
    donors = [row for row in records if bool(row.get("training_eligible"))]
    failures = [row for row in records if not bool(row.get("training_eligible"))]
    donors_by_track: dict[tuple[float, float, float], list[dict[str, Any]]] = {}
    for row in donors:
        donors_by_track.setdefault(track_key(row), []).append(row)

    rows: list[dict[str, Any]] = []
    for record in failures:
        key = track_key(record)
        track_donors = sorted(
            donors_by_track.get(key, []),
            key=lambda item: float(item["temperature_K"]),
        )
        teff = float(record["temperature_K"])
        if not track_donors:
            interpolation = {
                "kind": "no_same_track_donor",
                "donor_candidate_ids": [],
                "donor_effective_temperatures": [],
                "nearest_donor_id": None,
                "nearest_donor_effective_temperature": None,
                "delta_t_K": None,
                "outside_convex_hull": True,
                "probe_bin": "none",
            }
        else:
            donor_teffs = np.asarray(
                [float(item["temperature_K"]) for item in track_donors],
                dtype=np.float64,
            )
            selection = select_same_track_donor_indices(teff, donor_teffs)
            chosen = [track_donors[index] for index in selection["indices"]]
            nearest_index = int(np.argmin(np.abs(donor_teffs - teff)))
            nearest = track_donors[nearest_index]
            delta = abs(float(nearest["temperature_K"]) - teff)
            interpolation = {
                "kind": selection["kind"],
                "donor_candidate_ids": [item["candidate_id"] for item in chosen],
                "donor_effective_temperatures": [
                    float(item["temperature_K"]) for item in chosen
                ],
                "nearest_donor_id": nearest["candidate_id"],
                "nearest_donor_effective_temperature": float(
                    nearest["temperature_K"]
                ),
                "delta_t_K": float(delta),
                "outside_convex_hull": bool(selection["outside_convex_hull"]),
                "probe_bin": probe_bin(delta),
            }
        rows.append(
            {
                **_candidate_payload(record),
                "interpolation": interpolation,
                "parent": {
                    "training_eligible": False,
                    "failure_reason": record.get("failure_reason"),
                    "primary_iterations": (record.get("primary") or {}).get(
                        "iterations"
                    ),
                    "p95_absolute_flux_error_percent": _flux_metric(
                        record, "p95_absolute_flux_error_percent"
                    ),
                },
            }
        )
    rows.sort(key=lambda item: int(item["priority"]))
    same_track = [
        row
        for row in rows
        if row["interpolation"]["kind"] != "no_same_track_donor"
    ]
    return {
        "campaign": CAMPAIGN,
        "parent_campaign": "m_star_emulator_v1r2_marcs100",
        "donor_count": len(donors),
        "failed_count": len(failures),
        "same_track_count": len(same_track),
        "no_donor_count": len(rows) - len(same_track),
        "bracketed_count": sum(
            row["interpolation"]["kind"] == "bracketed" for row in same_track
        ),
        "one_sided_count": sum(
            row["interpolation"]["kind"] == "one_sided" for row in same_track
        ),
        "outside_convex_hull_count": sum(
            bool(row["interpolation"]["outside_convex_hull"]) for row in same_track
        ),
        "easy_count": sum(
            row["interpolation"]["probe_bin"] == "easy" for row in same_track
        ),
        "hard_count": sum(
            row["interpolation"]["probe_bin"] == "hard" for row in same_track
        ),
        "rows": rows,
    }


def select_probe_candidates(table: dict[str, Any]) -> list[dict[str, Any]]:
    same_track = [
        row
        for row in table["rows"]
        if row["interpolation"]["kind"] != "no_same_track_donor"
    ]
    selected: list[dict[str, Any]] = []
    for bin_name in ("easy", "hard"):
        pool = [
            row
            for row in same_track
            if row["interpolation"]["probe_bin"] == bin_name
        ]
        pool.sort(key=lambda item: int(item["priority"]))
        selected.extend(pool[:PROBE_PER_BIN])
    return selected


def select_scale_candidates(
    table: dict[str, Any],
    *,
    exclude_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    skip = exclude_ids or set()
    return [
        row
        for row in table["rows"]
        if row["interpolation"]["kind"] != "no_same_track_donor"
        and str(row["candidate_id"]) not in skip
    ]


def probe_decision(records: list[dict[str, Any]]) -> dict[str, Any]:
    easy = [row for row in records if row.get("probe_bin") == "easy"]
    eligible = sum(bool(row.get("training_eligible")) for row in easy)
    scale = eligible >= SCALE_EASY_MIN_ELIGIBLE
    return {
        "easy_count": len(easy),
        "easy_eligible_count": int(eligible),
        "scale": bool(scale),
        "decision": (
            "scale_same_track_failures"
            if scale
            else "stop_switch_to_continuation"
        ),
        "next_if_stop": (
            "experiments/reduced_state_emulator/cool_star_adaptive.py"
        ),
    }


def protocol_payload(
    result_root: Path,
    *,
    parent_root: Path,
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

    parent_records = _dwarf_records(parent_root)
    table = build_neighbor_table(parent_root, parent_records=parent_records)
    if (
        table["donor_count"] != EXPECTED_PARENT_ELIGIBLE_DWARFS
        or table["failed_count"] != EXPECTED_PARENT_FAILED_DWARFS
        or table["same_track_count"] != EXPECTED_SAME_TRACK_FAILURES
        or table["no_donor_count"] != EXPECTED_NO_DONOR_FAILURES
    ):
        raise ValueError("parent neighbor counts drifted from the frozen v1r2 table")
    probe = select_probe_candidates(table)
    probe_ids = tuple(row["candidate_id"] for row in probe)
    if probe_ids != EXPECTED_PROBE_IDS:
        raise ValueError(f"frozen probe identities drifted: {probe_ids}")
    scale_candidates = select_scale_candidates(table)
    payload: dict[str, Any] = {
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
            "campaign": parent_protocol["campaign"],
            "result_root": str(parent_root.resolve()),
            "protocol_hash": parent_protocol["protocol_hash"],
            "status_dwarf_sha256": _sha256(parent_root / "status_dwarf.json"),
            "existing_eligible_dwarf_count": EXPECTED_PARENT_ELIGIBLE_DWARFS,
        },
        "donors": {
            "class": "dwarf",
            "training_eligible_only": True,
            "giants_excluded": True,
            "ungated_excluded": True,
            "count": table["donor_count"],
        },
        "interpolation": {
            "fields": ["column_mass", "temperature"],
            "mix": "log10 (m,T) linear in log Teff; same log mix as interpolated_reduced_state",
            "power": LOG_TEFF_POWER,
            "same_track_only": True,
            "cross_track_knn": False,
            "one_sided_copies_nearest_donor": True,
            "six_field_interpolation": False,
        },
        "neighbor_table": {
            "failed_count": table["failed_count"],
            "same_track_count": table["same_track_count"],
            "no_donor_count": table["no_donor_count"],
            "bracketed_count": table["bracketed_count"],
            "one_sided_count": table["one_sided_count"],
            "outside_convex_hull_count": table["outside_convex_hull_count"],
            "easy_count": table["easy_count"],
            "hard_count": table["hard_count"],
        },
        "probe": {
            "selection": (
                "first 3 same-track failures with |ΔT|<=100 K and first 3 "
                "with |ΔT| in 200-300 K, in frozen v1r2 priority"
            ),
            "candidate_ids": list(EXPECTED_PROBE_IDS),
            "candidates": probe,
        },
        "scale_pool": {
            "selection": "all same-track dwarf failures, including the probe",
            "candidate_count": len(scale_candidates),
            "candidates": scale_candidates,
            "gated_on": "easy probe eligible count >= 2",
        },
        "target": {
            "new_eligible_dwarfs": TARGET_NEW_DWARFS,
            "combined_eligible_dwarfs": (
                EXPECTED_PARENT_ELIGIBLE_DWARFS + TARGET_NEW_DWARFS
            ),
            "batch_overshoot": "reserve only",
        },
        "solver": {
            "seed": "same-track gated ATLAS (m,T), reconstructed",
            "truth_source": "terminal Payne-Zero ATLAS atmosphere",
            "continuation": False,
            "iteration_cap": ITERATION_CAP,
            "maximum_all_layer_relative_temperature_change": (
                STRICT_ALL_LAYER_LIMIT
            ),
            "independent_restart": (
                "strict self-restart from terminal ATLAS (m,T)"
            ),
            "only_change_from_parent_attempt": (
                "replace same-node MARCS (m,T) with same-track ATLAS (m,T)"
            ),
        },
        "training_eligibility": dict(parent_protocol["training_eligibility"]),
        "imported_flux_gate": {
            "parent_gate_hash": parent_gate["gate_hash"],
            "thresholds": dict(parent_gate["thresholds"]),
            "thresholds_refit": False,
        },
        "boundaries": {
            "parent_results_mutated": False,
            "v1r3_results_mutated": False,
            "training_run": False,
            "candidate_validation_run": False,
            "sealed_holdout_opened": False,
            "production_routing_changed": False,
            "korg_run": False,
            "marcs_is_training_target": False,
            "cross_track_knn": False,
        },
        "paths": {
            "result_root": str(result_root.resolve()),
            "protocol": str((result_root / "protocol.json").resolve()),
            "flux_gate": str((result_root / "flux_gate.json").resolve()),
            "cases": str((result_root / "cases").resolve()),
            "neighbor_table": str(
                (result_root / "neighbor_table.json").resolve()
            ),
        },
    }
    payload["protocol_hash"] = _hash_payload(payload)
    return payload


def import_flux_gate(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    parent_root: Path,
) -> dict[str, Any]:
    parent_path = parent_root / "flux_gate.json"
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


def write_neighbor_table(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    parent_root: Path,
) -> dict[str, Any]:
    table = build_neighbor_table(parent_root)
    table["protocol_hash"] = protocol["protocol_hash"]
    table["probe_candidate_ids"] = list(protocol["probe"]["candidate_ids"])
    _write_json(result_root / "neighbor_table.json", table)
    lines = [
        "# Same-track neighbor table",
        "",
        f"Donors: {table['donor_count']}. Failed dwarfs: {table['failed_count']}.",
        f"Same-track starts: {table['same_track_count']}. No donor: {table['no_donor_count']}.",
        f"Bracketed interpolations: {table['bracketed_count']}. One-sided copies: {table['one_sided_count']}.",
        f"Outside donor hull: {table['outside_convex_hull_count']}. Easy: {table['easy_count']}. Hard: {table['hard_count']}.",
        "",
        "| priority | candidate | Teff | nearest donor | ΔT | kind | hull | bin |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in table["rows"]:
        item = row["interpolation"]
        nearest = item["nearest_donor_effective_temperature"]
        delta = item["delta_t_K"]
        lines.append(
            "| {priority} | `{cid}` | {teff:.0f} | {nearest} | {delta} | {kind} | {hull} | {bin} |".format(
                priority=int(row["priority"]),
                cid=row["candidate_id"],
                teff=float(row["temperature_K"]),
                nearest="—" if nearest is None else f"{nearest:.0f}",
                delta="—" if delta is None else f"{delta:.0f}",
                kind=item["kind"],
                hull=item["outside_convex_hull"],
                bin=item["probe_bin"],
            )
        )
    (result_root / "neighbor_table.md").write_text("\n".join(lines) + "\n")
    return table


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
        "probe_bin": candidate.get("interpolation", {}).get("probe_bin"),
        "interpolated_seed": None,
        "primary": None,
        "restart": None,
        "primary_flux_gate": {"passes": False, "metrics": {}},
        "restart_flux_gate": {"passes": False, "metrics": {}},
        "path_consistency": {"available": False, "passes": False},
        "training_eligible": False,
        "status": "failed_before_or_during_solver",
        "failure_reason": message,
    }


def _case_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        candidate,
        result_root_text,
        parent_root_text,
        flux_gate,
        protocol_hash,
    ) = payload
    _set_single_thread_environment()
    result_root = Path(result_root_text)
    parent_root = Path(parent_root_text)
    case_path = base._case_path(result_root, candidate)
    if case_path.is_file():
        return _read_json(case_path)

    track_payload = candidate["track"]
    track = base._track_from_payload(track_payload)
    labels = track.labels(float(candidate["temperature_K"]))
    case_root = case_path.parent
    interpolation = candidate["interpolation"]
    try:
        donor_records = []
        for donor_id, donor_teff in zip(
            interpolation["donor_candidate_ids"],
            interpolation["donor_effective_temperatures"],
        ):
            donor_candidate = {
                "class": "dwarf",
                "track": track_payload,
                "temperature_K": float(donor_teff),
                "candidate_id": donor_id,
            }
            donor_records.append(
                _read_json(base._case_path(parent_root, donor_candidate))
            )
        donor_labels = [dict(row["labels"]) for row in donor_records]
        donor_reduced = []
        for row in donor_records:
            mass, temperature = _load_mt(_resolve_parent_product(parent_root, row))
            donor_reduced.append(np.stack([mass, temperature], axis=-1))
        column_mass, temperature, mix = interpolate_same_track_mt(
            dict(labels.as_kwargs()),
            donor_labels,
            np.stack(donor_reduced, axis=0),
        )
        seed = _reconstruct_from_mt(labels, column_mass, temperature)
        primary, _primary_state = _solve_attempt(
            track=track,
            method="same_track_interpolated_atlas_mt_strict_primary",
            schedule="independent_interpolated_target",
            source_temperature=interpolation.get(
                "nearest_donor_effective_temperature"
            ),
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
                maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
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
            "probe_bin": interpolation.get("probe_bin"),
            "interpolated_seed": {
                "fields_used": ["column_mass", "temperature"],
                "is_training_target": False,
                **mix,
                "donor_candidate_ids": interpolation["donor_candidate_ids"],
            },
            "primary": primary,
            "restart": restart,
            "primary_flux_gate": primary_flux,
            "restart_flux_gate": restart_flux,
            "path_consistency": consistency,
            "training_eligible": eligible,
            "status": "training_eligible" if eligible else "ineligible",
            "failure_reason": None if eligible else ",".join(reasons),
        }
    except Exception as exc:  # noqa: BLE001 - retained campaign outcome
        output = _failed_case(
            candidate,
            protocol_hash=protocol_hash,
            error=exc,
        )
    _write_json(case_path, output)
    return output


def load_case_records(result_root: Path) -> list[dict[str, Any]]:
    return [
        _read_json(path)
        for path in sorted((result_root / "cases").glob("*/*/t*/case.json"))
    ]


def _write_probe_table(
    result_root: Path,
    records: list[dict[str, Any]],
) -> None:
    lines = [
        "# Six-star interpolated (m,T) probe",
        "",
        "| bin | candidate | ΔT | hull | kind | eligible | iterations | flux p95 | v1r2 MARCS iterations | v1r2 MARCS flux p95 |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    order = {candidate_id: index for index, candidate_id in enumerate(EXPECTED_PROBE_IDS)}
    for record in sorted(
        records, key=lambda row: order.get(str(row["candidate_id"]), 99)
    ):
        interpolation = record.get("interpolation") or {}
        primary = record.get("primary") or {}
        parent = record.get("parent") or {}
        flux = _flux_metric(record, "p95_absolute_flux_error_percent")
        delta = interpolation.get("delta_t_K")
        lines.append(
            "| {bin} | `{cid}` | {delta} | {hull} | {kind} | {elig} | {iters} | {flux} | {parent_iters} | {parent_flux} |".format(
                bin=record.get("probe_bin") or interpolation.get("probe_bin"),
                cid=record["candidate_id"],
                delta="—" if delta is None else f"{float(delta):.0f}",
                hull=interpolation.get("outside_convex_hull"),
                kind=interpolation.get("kind"),
                elig=bool(record.get("training_eligible")),
                iters=primary.get("iterations") if primary else "—",
                flux="—" if flux is None else f"{flux:.3g}",
                parent_iters=parent.get("primary_iterations") or "—",
                parent_flux=(
                    "—"
                    if parent.get("p95_absolute_flux_error_percent") is None
                    else f"{float(parent['p95_absolute_flux_error_percent']):.3g}"
                ),
            )
        )
    (result_root / "probe_table.md").write_text("\n".join(lines) + "\n")


def campaign_status(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    write: bool = True,
) -> dict[str, Any]:
    records = load_case_records(result_root)
    by_id = {str(row["candidate_id"]): row for row in records}
    probe_ids = [str(row) for row in protocol["probe"]["candidate_ids"]]
    probe_records = [by_id[item] for item in probe_ids if item in by_id]
    scale_ids = [
        str(row["candidate_id"]) for row in protocol["scale_pool"]["candidates"]
    ]
    eligible = sorted(
        (row for row in records if bool(row.get("training_eligible"))),
        key=lambda row: int(row["priority"]),
    )
    decision = (
        probe_decision(probe_records)
        if len(probe_records) == len(probe_ids)
        else None
    )
    pending_probe = [item for item in probe_ids if item not in by_id]
    pending_scale = [
        row
        for row in protocol["scale_pool"]["candidates"]
        if str(row["candidate_id"]) not in by_id
    ]
    payload = {
        "campaign": CAMPAIGN,
        "protocol_hash": protocol["protocol_hash"],
        "probe_attempted_count": len(probe_records),
        "probe_pending_count": len(pending_probe),
        "probe_eligible_count": sum(
            bool(row.get("training_eligible")) for row in probe_records
        ),
        "probe_complete": not pending_probe,
        "decision": decision,
        "attempted_count": len(records),
        "eligible_count": len(eligible),
        "scale_pending_count": len(pending_scale),
        "target_new_eligible": TARGET_NEW_DWARFS,
        "target_reached": len(eligible) >= TARGET_NEW_DWARFS,
        "eligible_candidate_ids": [row["candidate_id"] for row in eligible],
        "pending_probe_ids": pending_probe,
        "pending_scale_candidates": pending_scale,
    }
    payload["status_hash"] = _hash_payload(payload)
    if write:
        _write_json(result_root / "status.json", payload)
        if probe_records:
            _write_probe_table(result_root, probe_records)
    return payload


def _run_candidates(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    flux_gate: dict[str, Any],
    parent_root: Path,
    candidates: list[dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    if not candidates:
        return []
    started = time.perf_counter()
    outputs = _run_workers(
        _case_worker,
        [
            (
                candidate,
                str(result_root),
                str(parent_root),
                flux_gate,
                protocol["protocol_hash"],
            )
            for candidate in candidates
        ],
        workers=min(max(int(workers), 1), len(candidates)),
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
    return outputs


def run_probe(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    flux_gate: dict[str, Any],
    parent_root: Path,
    workers: int,
) -> dict[str, Any]:
    _run_candidates(
        result_root,
        protocol=protocol,
        flux_gate=flux_gate,
        parent_root=parent_root,
        candidates=protocol["probe"]["candidates"],
        workers=workers,
    )
    status = campaign_status(result_root, protocol=protocol)
    (result_root / "PROBE_COMPLETE").touch()
    decision = status["decision"] or {}
    if decision.get("scale"):
        (result_root / "PROBE_SCALE").touch()
    else:
        (result_root / "PROBE_STOP").touch()
        (result_root / "STOP_SWITCH_TO_CONTINUATION").touch()
    return status


def run_scale(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    flux_gate: dict[str, Any],
    parent_root: Path,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    status = campaign_status(result_root, protocol=protocol)
    decision = status.get("decision") or {}
    if not status.get("probe_complete"):
        raise SystemExit("run the six-star probe before scaling")
    if not decision.get("scale"):
        raise SystemExit(
            "easy probe recovered fewer than 2/3; stop and use adaptive continuation"
        )
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    while True:
        status = campaign_status(result_root, protocol=protocol)
        if status["target_reached"]:
            (result_root / "SCALE_TARGET_REACHED").touch()
            return status
        if not status["pending_scale_candidates"]:
            (result_root / "SCALE_POOL_EXHAUSTED").touch()
            return status
        pending = status["pending_scale_candidates"][:batch_size]
        _run_candidates(
            result_root,
            protocol=protocol,
            flux_gate=flux_gate,
            parent_root=parent_root,
            candidates=pending,
            workers=workers,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "protocol",
            "import-gate",
            "neighbor-table",
            "probe",
            "status",
            "scale",
        ),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
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
            parent_root=args.parent_root,
        )
        _write_json(protocol_path, protocol)

    if args.stage == "protocol":
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return 0

    if args.stage == "neighbor-table":
        table = write_neighbor_table(
            args.result_root,
            protocol=protocol,
            parent_root=args.parent_root,
        )
        print(
            json.dumps(
                {key: table[key] for key in table if key != "rows"},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    gate_path = args.result_root / "flux_gate.json"
    if args.stage == "import-gate":
        gate = import_flux_gate(
            args.result_root,
            protocol=protocol,
            parent_root=args.parent_root,
        )
        print(json.dumps(gate, indent=2, sort_keys=True))
        return 0
    if not gate_path.is_file():
        raise SystemExit("run import-gate before the solver stages")
    flux_gate = _read_json(gate_path)

    if args.stage == "probe":
        status = run_probe(
            args.result_root,
            protocol=protocol,
            flux_gate=flux_gate,
            parent_root=args.parent_root,
            workers=args.workers,
        )
    elif args.stage == "scale":
        status = run_scale(
            args.result_root,
            protocol=protocol,
            flux_gate=flux_gate,
            parent_root=args.parent_root,
            batch_size=args.batch_size,
            workers=args.workers,
        )
    else:
        status = campaign_status(args.result_root, protocol=protocol)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
