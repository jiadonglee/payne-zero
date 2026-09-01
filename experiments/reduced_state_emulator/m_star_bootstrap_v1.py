"""Build the first ATLAS-only cool-star truth corpus for a two-field emulator.

The campaign is intentionally staged:

1. solve a 4000/4500 K reference panel and freeze flux-quality thresholds;
2. continue each open track from 4000 to 3000 K in 50 K steps;
3. independently restart every candidate from its final ``(m, T)``;
4. admit only path-consistent, flux-clean ATLAS solutions to the corpus.

MARCS is not a training target.  The released production initializer supplies
the reference/anchor starts; every stored target is the terminal Payne-Zero
ATLAS atmosphere.  Sealed tracks are present in the preregistered manifest but
are never run without the explicit ``--run-sealed`` flag.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

import numpy as np

from .cool_star_step_test import (
    ITERATION_CAP,
    PRIMARY_ITERATION_CAP,
    TrackSpec,
    _atmosphere_quality,
    _production_atmosphere,
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "m_star_emulator_v1"
DEFAULT_EXISTING_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)

REFERENCE_TEMPERATURES = (4500.0, 4000.0)
SPINE_TEMPERATURES = tuple(float(value) for value in range(4000, 2999, -50))
GIANT_LOGG = (0.5, 1.0, 1.5, 2.0, 2.5)
DWARF_LOGG = (4.5, 4.75, 5.0, 5.25, 5.5)
FLUX_METRICS = (
    "median_absolute_flux_error_percent",
    "p95_absolute_flux_error_percent",
    "maximum_absolute_flux_error_percent",
)
FLUX_REFERENCE_MULTIPLIER = 1.25
MINIMUM_REFERENCE_SUCCESSES = 12
PATH_TEMPERATURE_P95_LIMIT = 3.0e-3
PATH_COLUMN_MASS_P95_DEX_LIMIT = 7.7e-3

# Ten full temperature tracks cannot realize 70/15/15 while retaining both
# luminosity classes in validation and sealed sets.  This balanced 60/20/20
# split is fixed before any spine calculation and avoids adjacent-temperature
# leakage across roles.
FIXED_TRACK_ROLES = {
    ("giant", 0.5): "train",
    ("giant", 1.0): "sealed",
    ("giant", 1.5): "train",
    ("giant", 2.0): "validation",
    ("giant", 2.5): "train",
    ("dwarf", 4.5): "train",
    ("dwarf", 4.75): "validation",
    ("dwarf", 5.0): "train",
    ("dwarf", 5.25): "sealed",
    ("dwarf", 5.5): "train",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_tracks() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stellar_class, gravities, microturbulence in (
        ("giant", GIANT_LOGG, 2.0),
        ("dwarf", DWARF_LOGG, 1.0),
    ):
        for logg in gravities:
            track = TrackSpec(
                log_surface_gravity=float(logg),
                metallicity=0.0,
                alpha_enhancement=0.0,
                carbon_enhancement=0.0,
                microturbulence_km_s=microturbulence,
            )
            rows.append(
                {
                    **track.as_json(),
                    "class": stellar_class,
                    "role": FIXED_TRACK_ROLES[(stellar_class, float(logg))],
                }
            )
    return rows


def protocol_payload(result_root: Path) -> dict[str, Any]:
    tracks = build_tracks()
    payload = {
        "campaign": "m_star_emulator_v1",
        "status": "preregistered_before_heavy_solver",
        "created_for": "ATLAS-only cool-star truth generation and two-field emulator training",
        "source": {
            "git_head": _git_head(),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__)),
            "existing_corpus": str(DEFAULT_EXISTING_CORPUS),
            "existing_corpus_sha256": (
                _sha256(DEFAULT_EXISTING_CORPUS)
                if DEFAULT_EXISTING_CORPUS.is_file()
                else None
            ),
        },
        "grid": {
            "temperature_K": list(SPINE_TEMPERATURES),
            "temperature_step_K": 50.0,
            "giant_logg": list(GIANT_LOGG),
            "dwarf_logg": list(DWARF_LOGG),
            "metallicity": 0.0,
            "alpha_enhancement": 0.0,
            "microturbulence_km_s": {"giant": 2.0, "dwarf": 1.0},
            "target_node_count": len(SPINE_TEMPERATURES) * len(tracks),
        },
        "split": {
            "unit": "complete temperature track",
            "reason": "prevent adjacent-temperature leakage; preserve both classes in validation and sealed sets",
            "realized_fraction": {"train": 0.60, "validation": 0.20, "sealed": 0.20},
            "sealed_not_run_by_default": True,
            "tracks": tracks,
        },
        "reference_flux_gate": {
            "temperatures_K": list(REFERENCE_TEMPERATURES),
            "roles": ["train", "validation"],
            "metrics": list(FLUX_METRICS),
            "minimum_successful_reference_solves": MINIMUM_REFERENCE_SUCCESSES,
            "threshold_formula": "1.25 times the maximum successful reference value for each metric",
            "multiplier": FLUX_REFERENCE_MULTIPLIER,
            "must_be_frozen_before_spine": True,
        },
        "solver": {
            "initializer_for_4000_and_4500_K": "released production initializer; extrapolation allowed only to construct the seed",
            "truth_source": "terminal Payne-Zero ATLAS atmosphere",
            "continuation": "4000 to 3000 K in 50 K reduced/rematerialized (m,T) steps",
            "independent_restart": "every candidate restarted from its own final (m,T)",
            "iteration_cap": ITERATION_CAP,
            "within_15_iterations_recorded": True,
            "stopping_rule": "unchanged production solver stopping rule",
        },
        "training_eligibility": {
            "primary_and_restart_must_converge": True,
            "finite_positive_monotone_six_field_state": True,
            "primary_and_restart_must_pass_frozen_flux_gate": True,
            "restart_temperature_relative_p95_max": PATH_TEMPERATURE_P95_LIMIT,
            "restart_column_mass_p95_dex_max": PATH_COLUMN_MASS_P95_DEX_LIMIT,
            "failed_or_ineligible_rows_retained": True,
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


def _track_from_payload(payload: dict[str, Any]) -> TrackSpec:
    return TrackSpec(
        log_surface_gravity=float(payload["log_surface_gravity"]),
        metallicity=float(payload["metallicity"]),
        alpha_enhancement=float(payload["alpha_enhancement"]),
        carbon_enhancement=float(payload["carbon_enhancement"]),
        microturbulence_km_s=float(payload["microturbulence_km_s"]),
    )


def _final_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("solver_diagnostics", {}).get("final_diagnostics", {})


def _annotate_record(
    record: dict[str, Any],
    *,
    track_payload: dict[str, Any],
    role: str,
    node_id: str,
) -> dict[str, Any]:
    record = _jsonable(dict(record))
    record["node_id"] = node_id
    record["split_role"] = role
    record["class"] = track_payload["class"]
    iterations = record.get("iterations")
    record["within_15_iterations"] = bool(
        record.get("survives_solver")
        and iterations is not None
        and int(iterations) <= PRIMARY_ITERATION_CAP
    )
    return record


def _calibration_worker(payload: tuple[dict[str, Any], str, int, str]) -> dict[str, Any]:
    track_payload, result_root_text, iteration_cap, protocol_hash = payload
    _set_single_thread_environment()
    track = _track_from_payload(track_payload)
    role = str(track_payload["role"])
    track_root = Path(result_root_text) / "reference" / track.track_id
    records: list[dict[str, Any]] = []
    for temperature in REFERENCE_TEMPERATURES:
        labels = track.labels(temperature)
        node_id = f"{track.track_id}_t{int(temperature):04d}"
        try:
            initial = _production_atmosphere(labels)
            record, _state = _solve_attempt(
                track=track,
                method="production_reference_direct",
                schedule="reference_direct",
                source_temperature=None,
                target_labels=labels,
                initial_atmosphere=initial,
                product_dir=track_root / "products",
                iteration_cap=iteration_cap,
            )
        except Exception as exc:  # noqa: BLE001 - retained campaign outcome
            record = {
                "method": "production_reference_direct",
                "schedule": "reference_direct",
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
                role=role,
                node_id=node_id,
            )
        )
    output = {
        "track": track_payload,
        "protocol_hash": protocol_hash,
        "records": records,
    }
    _write_json(track_root / "reference.json", output)
    return output


def freeze_flux_gate(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    roles: set[str],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for track_payload in protocol["split"]["tracks"]:
        if track_payload["role"] not in roles:
            continue
        path = result_root / "reference" / track_payload["track_id"] / "reference.json"
        if not path.is_file():
            continue
        records.extend(json.loads(path.read_text())["records"])
    successful = [
        row
        for row in records
        if row.get("survives_solver")
        and all(
            np.isfinite(_final_diagnostics(row).get(metric, np.nan))
            for metric in FLUX_METRICS
        )
    ]
    status = (
        "pass"
        if len(successful) >= MINIMUM_REFERENCE_SUCCESSES
        else "fail_stop_insufficient_reference_solves"
    )
    thresholds = {}
    if status == "pass":
        for metric in FLUX_METRICS:
            maximum = max(float(_final_diagnostics(row)[metric]) for row in successful)
            thresholds[metric] = FLUX_REFERENCE_MULTIPLIER * maximum
    payload = {
        "campaign": "m_star_emulator_v1",
        "status": status,
        "frozen": status == "pass",
        "protocol_hash": protocol["protocol_hash"],
        "roles": sorted(roles),
        "reference_attempt_count": len(records),
        "reference_success_count": len(successful),
        "minimum_reference_success_count": MINIMUM_REFERENCE_SUCCESSES,
        "multiplier": FLUX_REFERENCE_MULTIPLIER,
        "thresholds": thresholds,
        "source_records": [
            {
                "node_id": row["node_id"],
                "split_role": row["split_role"],
                "iterations": row.get("iterations"),
                "metrics": {
                    metric: _final_diagnostics(row).get(metric)
                    for metric in FLUX_METRICS
                },
            }
            for row in successful
        ],
    }
    payload["gate_hash"] = _hash_payload(payload)
    _write_json(result_root / "flux_gate.json", payload)
    return payload


def _load_mt(product_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(product_path, allow_pickle=False) as data:
        return (
            np.asarray(data["column_mass"], dtype=np.float64),
            np.asarray(data["temperature"], dtype=np.float64),
        )


def _product_consistency(
    first_product: str | Path | None,
    second_product: str | Path | None,
) -> dict[str, Any]:
    if not first_product or not second_product:
        return {"available": False, "passes": False}
    try:
        first_m, first_t = _load_mt(first_product)
        second_m, second_t = _load_mt(second_product)
        temperature_relative = np.abs(second_t - first_t) / first_t
        mass_dex = np.abs(np.log10(second_m) - np.log10(first_m))
        result = {
            "available": True,
            "temperature_relative": {
                "median": float(np.median(temperature_relative)),
                "p95": float(np.percentile(temperature_relative, 95.0)),
                "max": float(np.max(temperature_relative)),
            },
            "column_mass_dex": {
                "median": float(np.median(mass_dex)),
                "p95": float(np.percentile(mass_dex, 95.0)),
                "max": float(np.max(mass_dex)),
            },
        }
        result["passes"] = bool(
            result["temperature_relative"]["p95"] <= PATH_TEMPERATURE_P95_LIMIT
            and result["column_mass_dex"]["p95"]
            <= PATH_COLUMN_MASS_P95_DEX_LIMIT
        )
        return result
    except Exception as exc:  # noqa: BLE001 - diagnostic failure is explicit
        return {
            "available": False,
            "passes": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _passes_flux_gate(record: dict[str, Any], flux_gate: dict[str, Any]) -> dict[str, Any]:
    diagnostics = _final_diagnostics(record)
    metrics: dict[str, Any] = {}
    passes = bool(record.get("survives_solver"))
    for metric, threshold in flux_gate["thresholds"].items():
        value = diagnostics.get(metric)
        metric_pass = bool(
            value is not None
            and np.isfinite(value)
            and float(value) <= float(threshold)
        )
        metrics[metric] = {
            "value": value,
            "threshold": threshold,
            "passes": metric_pass,
        }
        passes = passes and metric_pass
    return {"passes": bool(passes), "metrics": metrics}


def _blocked_case(
    track_payload: dict[str, Any],
    temperature: float,
    reason: str,
) -> dict[str, Any]:
    track = _track_from_payload(track_payload)
    node_id = f"{track.track_id}_t{int(temperature):04d}"
    return {
        "node_id": node_id,
        "track_id": track.track_id,
        "split_role": track_payload["role"],
        "class": track_payload["class"],
        "labels": track.labels(temperature).as_kwargs(),
        "temperature_K": temperature,
        "primary": None,
        "restart": None,
        "path_consistency": {"available": False, "passes": False},
        "primary_flux_gate": {"passes": False, "metrics": {}},
        "restart_flux_gate": {"passes": False, "metrics": {}},
        "training_eligible": False,
        "status": "blocked_by_previous_ineligible_step",
        "failure_reason": reason,
    }


def _spine_worker(
    payload: tuple[dict[str, Any], str, int, dict[str, Any], str],
) -> dict[str, Any]:
    track_payload, result_root_text, iteration_cap, flux_gate, protocol_hash = payload
    _set_single_thread_environment()
    result_root = Path(result_root_text)
    track = _track_from_payload(track_payload)
    role = str(track_payload["role"])
    track_root = result_root / "solar_spine" / track.track_id
    reference_path = (
        result_root / "reference" / track.track_id / "reference.json"
    )
    if not reference_path.is_file():
        raise FileNotFoundError(f"missing reference record: {reference_path}")
    reference = json.loads(reference_path.read_text())
    anchor = next(
        (
            row
            for row in reference["records"]
            if float(row.get("target_temperature", np.nan)) == 4000.0
        ),
        None,
    )
    cases: list[dict[str, Any]] = []
    current_m: np.ndarray | None = None
    current_t: np.ndarray | None = None
    block_reason: str | None = None

    for temperature in SPINE_TEMPERATURES:
        labels = track.labels(temperature)
        node_id = f"{track.track_id}_t{int(temperature):04d}"
        if block_reason is not None:
            cases.append(_blocked_case(track_payload, temperature, block_reason))
            continue

        if temperature == 4000.0:
            primary = anchor
            if primary is None or not primary.get("survives_solver"):
                block_reason = "4000 K reference anchor did not converge"
                cases.append(_blocked_case(track_payload, temperature, block_reason))
                continue
            try:
                current_m, current_t = _load_mt(primary["product_path"])
            except Exception as exc:  # noqa: BLE001
                block_reason = f"cannot load 4000 K anchor product: {type(exc).__name__}: {exc}"
                cases.append(_blocked_case(track_payload, temperature, block_reason))
                continue
            primary = _annotate_record(
                primary,
                track_payload=track_payload,
                role=role,
                node_id=node_id,
            )
            primary["method"] = "production_reference_anchor"
        else:
            if current_m is None or current_t is None:
                block_reason = "previous step did not provide an eligible (m,T) state"
                cases.append(_blocked_case(track_payload, temperature, block_reason))
                continue
            initial = _reconstruct_from_mt(labels, current_m, current_t)
            primary, primary_state = _solve_attempt(
                track=track,
                method="atlas_reduced_continuation_50K",
                schedule="4000_to_3000_by_50K",
                source_temperature=temperature + 50.0,
                target_labels=labels,
                initial_atmosphere=initial,
                product_dir=track_root / "products" / "primary",
                iteration_cap=iteration_cap,
            )
            primary = _annotate_record(
                primary,
                track_payload=track_payload,
                role=role,
                node_id=node_id,
            )
            if primary.get("survives_solver") and primary_state is not None:
                current_m = np.asarray(primary_state.column_mass, dtype=np.float64)
                current_t = np.asarray(primary_state.temperature, dtype=np.float64)

        restart: dict[str, Any] | None = None
        if primary.get("survives_solver") and primary.get("product_path"):
            try:
                solved_m, solved_t = _load_mt(primary["product_path"])
                restart_initial = _reconstruct_from_mt(labels, solved_m, solved_t)
                restart, _restart_state = _solve_attempt(
                    track=track,
                    method="atlas_self_restart_from_mt",
                    schedule="independent_self_restart",
                    source_temperature=temperature,
                    target_labels=labels,
                    initial_atmosphere=restart_initial,
                    product_dir=track_root / "products" / "restart",
                    iteration_cap=iteration_cap,
                )
                restart = _annotate_record(
                    restart,
                    track_payload=track_payload,
                    role=role,
                    node_id=node_id,
                )
            except Exception as exc:  # noqa: BLE001
                restart = {
                    "node_id": node_id,
                    "split_role": role,
                    "class": track_payload["class"],
                    "survives_solver": False,
                    "status": "restart_exception",
                    "error": f"{type(exc).__name__}: {exc}",
                    "product_path": None,
                }

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
        training_eligible = bool(
            primary.get("survives_solver")
            and restart is not None
            and restart.get("survives_solver")
            and primary.get("state_quality", {}).get("valid")
            and restart.get("state_quality", {}).get("valid")
            and primary_flux["passes"]
            and restart_flux["passes"]
            and consistency["passes"]
        )
        case = {
            "node_id": node_id,
            "track_id": track.track_id,
            "split_role": role,
            "class": track_payload["class"],
            "labels": labels.as_kwargs(),
            "temperature_K": temperature,
            "primary": primary,
            "restart": restart,
            "path_consistency": consistency,
            "primary_flux_gate": primary_flux,
            "restart_flux_gate": restart_flux,
            "training_eligible": training_eligible,
            "status": "training_eligible" if training_eligible else "ineligible",
            "failure_reason": None,
        }
        if not training_eligible:
            reasons = []
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
            case["failure_reason"] = ",".join(reasons) or "independent_quality_gate"
            block_reason = f"{node_id} failed training eligibility: {case['failure_reason']}"
        else:
            current_m, current_t = _load_mt(primary["product_path"])
        cases.append(_jsonable(case))

    output = {
        "track": track_payload,
        "protocol_hash": protocol_hash,
        "flux_gate_hash": flux_gate["gate_hash"],
        "cases": cases,
        "eligible_count": sum(bool(case["training_eligible"]) for case in cases),
        "attempted_count": sum(case["primary"] is not None for case in cases),
    }
    _write_json(track_root / "track.json", output)
    _write_jsonl(track_root / "records.jsonl", cases)
    return output


def _run_workers(
    worker: Any,
    payloads: list[tuple[Any, ...]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [worker(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(worker, payloads))


def _selected_tracks(protocol: dict[str, Any], roles: set[str]) -> list[dict[str, Any]]:
    return [
        row
        for row in protocol["split"]["tracks"]
        if str(row["role"]) in roles
    ]


def build_cool_corpus(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    roles: set[str],
    allow_sealed: bool,
) -> dict[str, Any]:
    if "sealed" in roles and not allow_sealed:
        raise ValueError("sealed corpus rows require --run-sealed")
    rows: list[dict[str, Any]] = []
    for track_payload in _selected_tracks(protocol, roles):
        path = result_root / "solar_spine" / track_payload["track_id"] / "track.json"
        if not path.is_file():
            continue
        track = json.loads(path.read_text())
        rows.extend(case for case in track["cases"] if case["training_eligible"])
    rows.sort(key=lambda row: (row["split_role"], row["track_id"], -row["temperature_K"]))
    if not rows:
        raise RuntimeError("no training-eligible cool-star rows are available")

    labels = []
    column_mass = []
    temperature = []
    flux_metrics = []
    for row in rows:
        values = row["labels"]
        labels.append(
            [
                values["effective_temperature"],
                values["log_surface_gravity"],
                values["metallicity"],
                values["alpha_enhancement"],
                values["microturbulence_km_s"],
            ]
        )
        mass, temp = _load_mt(row["primary"]["product_path"])
        column_mass.append(mass)
        temperature.append(temp)
        diagnostics = _final_diagnostics(row["primary"])
        flux_metrics.append([diagnostics[metric] for metric in FLUX_METRICS])

    output_path = result_root / "cool_truth_corpus.npz"
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            labels=np.asarray(labels, dtype=np.float64),
            label_fields=np.asarray(
                [
                    "effective_temperature",
                    "log_surface_gravity",
                    "metallicity",
                    "alpha_enhancement",
                    "microturbulence_km_s",
                ],
                dtype="U40",
            ),
            column_mass=np.asarray(column_mass, dtype=np.float64),
            temperature=np.asarray(temperature, dtype=np.float64),
            roles=np.asarray([row["split_role"] for row in rows], dtype="U16"),
            track_ids=np.asarray([row["track_id"] for row in rows], dtype="U80"),
            node_ids=np.asarray([row["node_id"] for row in rows], dtype="U96"),
            source_product_paths=np.asarray(
                [row["primary"]["product_path"] for row in rows], dtype="U512"
            ),
            flux_metric_fields=np.asarray(FLUX_METRICS, dtype="U64"),
            flux_metrics=np.asarray(flux_metrics, dtype=np.float64),
            protocol_hash=np.asarray([protocol["protocol_hash"]], dtype="U64"),
            flux_gate_hash=np.asarray(
                [json.loads((result_root / "flux_gate.json").read_text())["gate_hash"]],
                dtype="U64",
            ),
        )
    temporary.replace(output_path)
    summary = {
        "status": "complete",
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "row_count": len(rows),
        "role_counts": {
            role: sum(row["split_role"] == role for row in rows)
            for role in sorted({row["split_role"] for row in rows})
        },
        "class_counts": {
            stellar_class: sum(row["class"] == stellar_class for row in rows)
            for stellar_class in ("giant", "dwarf")
        },
        "protocol_hash": protocol["protocol_hash"],
        "sealed_rows_included": "sealed" in roles,
    }
    _write_json(result_root / "cool_truth_corpus.json", summary)
    return summary


def summarize_open_spine(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    roles: set[str],
) -> dict[str, Any]:
    tracks = []
    cases = []
    for track_payload in _selected_tracks(protocol, roles):
        path = result_root / "solar_spine" / track_payload["track_id"] / "track.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        tracks.append(payload)
        cases.extend(payload["cases"])
    summary = {
        "campaign": "m_star_emulator_v1",
        "roles": sorted(roles),
        "track_count": len(tracks),
        "planned_node_count": len(_selected_tracks(protocol, roles))
        * len(SPINE_TEMPERATURES),
        "attempted_node_count": sum(case["primary"] is not None for case in cases),
        "primary_solver_converged": sum(
            bool(case["primary"] and case["primary"].get("survives_solver"))
            for case in cases
        ),
        "training_eligible_count": sum(
            bool(case["training_eligible"]) for case in cases
        ),
        "within_15_count": sum(
            bool(case["training_eligible"])
            and bool(case["primary"].get("within_15_iterations"))
            for case in cases
        ),
        "first_ineligible_by_track": {
            track["track"]["track_id"]: next(
                (
                    case["node_id"]
                    for case in track["cases"]
                    if not case["training_eligible"]
                ),
                None,
            )
            for track in tracks
        },
        "production_routing_changed": False,
        "existing_sealed_holdout_opened": False,
        "new_sealed_tracks_run": "sealed" in roles,
        "korg_run": False,
        "protocol_hash": protocol["protocol_hash"],
    }
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
            "calibrate",
            "freeze-gate",
            "spine",
            "build-corpus",
            "summary",
            "all-open",
        ),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--roles", default="train,validation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iteration-cap", type=int, default=ITERATION_CAP)
    parser.add_argument("--run-sealed", action="store_true")
    args = parser.parse_args(argv)

    roles = _parse_roles(args.roles)
    if "sealed" in roles and not args.run_sealed:
        raise SystemExit("sealed tracks require the explicit --run-sealed flag")
    args.result_root.mkdir(parents=True, exist_ok=True)
    protocol_path = args.result_root / "protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text())
    else:
        protocol = protocol_payload(args.result_root)
        _write_json(protocol_path, protocol)

    selected = _selected_tracks(protocol, roles)
    if args.stage == "protocol":
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return 0

    if args.stage in {"calibrate", "all-open"}:
        started = time.perf_counter()
        payloads = [
            (
                row,
                str(args.result_root),
                int(args.iteration_cap),
                protocol["protocol_hash"],
            )
            for row in selected
        ]
        outputs = _run_workers(_calibration_worker, payloads, workers=args.workers)
        print(
            f"reference tracks={len(outputs)} seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
        if args.stage == "calibrate":
            return 0

    if args.stage in {"freeze-gate", "all-open"}:
        flux_gate = freeze_flux_gate(
            args.result_root,
            protocol=protocol,
            roles=roles,
        )
        print(json.dumps(flux_gate, indent=2, sort_keys=True))
        if not flux_gate["frozen"]:
            raise SystemExit("FAIL_STOP: reference flux gate could not be frozen")
        if args.stage == "freeze-gate":
            return 0
    else:
        flux_gate_path = args.result_root / "flux_gate.json"
        if not flux_gate_path.is_file():
            raise SystemExit("run freeze-gate before spine generation")
        flux_gate = json.loads(flux_gate_path.read_text())
        if not flux_gate.get("frozen"):
            raise SystemExit("FAIL_STOP: flux gate is not frozen")

    if args.stage in {"spine", "all-open"}:
        started = time.perf_counter()
        payloads = [
            (
                row,
                str(args.result_root),
                int(args.iteration_cap),
                flux_gate,
                protocol["protocol_hash"],
            )
            for row in selected
        ]
        outputs = _run_workers(_spine_worker, payloads, workers=args.workers)
        print(
            f"spine tracks={len(outputs)} seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
        if args.stage == "spine":
            return 0

    if args.stage in {"build-corpus", "all-open"}:
        corpus = build_cool_corpus(
            args.result_root,
            protocol=protocol,
            roles=roles,
            allow_sealed=args.run_sealed,
        )
        print(json.dumps(corpus, indent=2, sort_keys=True))
        if args.stage == "build-corpus":
            return 0

    summary = summarize_open_spine(
        args.result_root,
        protocol=protocol,
        roles=roles,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
