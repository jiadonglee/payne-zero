"""Build 100 balanced cool-star ATLAS truths from native MARCS ``(m,T)`` seeds.

This is a prospective follow-up to the immutable v1 and v1r1 FAIL_STOP
campaigns.  Every candidate is an independent native MARCS grid node.  MARCS
provides only the initial column-mass and temperature profiles; Payne-Zero
reconstructs the other fields and the terminal ATLAS atmosphere is the only
training target.

The train pool is fixed before any solve.  Giant and dwarf workers may run on
different hosts because every case has a disjoint output path.  Each class
stops after at least 50 eligible rows are available.  Corpus construction then
takes the first 50 eligible rows in the frozen priority order, retaining any
batch overshoot as unused reserve evidence.
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
    TrackSpec,
    _marcs_diagnostics,
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from .marcs_h5 import (
    EXPECTED_MARCS_SHA256,
    inspect_marcs_grid,
    load_marcs_node,
)
from .m_star_bootstrap_v1 import (
    FLUX_METRICS,
    PATH_COLUMN_MASS_P95_DEX_LIMIT,
    PATH_TEMPERATURE_P95_LIMIT,
    _annotate_record,
    _final_diagnostics,
    _hash_payload,
    _load_mt,
    _passes_flux_gate,
    _product_consistency,
    _run_workers,
    _sha256,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_emulator_v1r2_marcs100"
TARGET_PER_CLASS = 50
ITERATION_CAP = 60
STRICT_ALL_LAYER_LIMIT = 5.0e-4
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
DEFAULT_FLUX_PARENT_ROOT = REPO_ROOT / "results" / "m_star_emulator_v1"
DEFAULT_VALIDATION_CORPUS = (
    REPO_ROOT
    / "results"
    / "m_star_emulator_v1r1_policy60"
    / "cool_truth_corpus.npz"
)
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "notes"
    / "m_star_emulator_v1r2_marcs100_preregistration_20260831.md"
)

# The order is intentionally space-filling rather than monotonic.  If the
# quota is met early, the selected rows still span the cool interval.
TEMPERATURE_PRIORITY = (
    4000.0,
    3500.0,
    3000.0,
    3800.0,
    3300.0,
    3900.0,
    3600.0,
    3200.0,
    3750.0,
    3400.0,
    3700.0,
    3100.0,
)
METALLICITIES = (-1.0, -0.5, 0.0, 0.5)
CLASS_GRAVITIES = {
    "giant": (0.5, 1.5, 2.5),
    "dwarf": (4.5, 5.0, 5.5),
}
MICROTURBULENCE = {"giant": 2.0, "dwarf": 1.0}

# Complete (class, logg, metallicity) tracks are assigned prospectively.
# Validation and sealed tracks are listed for provenance but are not run by the
# truth-generation command.  Open validation is imported from v1r1.
VALIDATION_TRACKS = {
    ("giant", 1.5, -0.5),
    ("giant", 2.5, 0.5),
    ("dwarf", 4.5, 0.5),
    ("dwarf", 5.5, -0.5),
}
SEALED_TRACKS = {
    ("giant", 0.5, 0.5),
    ("dwarf", 5.0, -1.0),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _track_role(stellar_class: str, logg: float, metallicity: float) -> str:
    key = (stellar_class, float(logg), float(metallicity))
    if key in VALIDATION_TRACKS:
        return "validation"
    if key in SEALED_TRACKS:
        return "sealed"
    return "train"


def build_tracks() -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    for stellar_class in ("giant", "dwarf"):
        for logg in CLASS_GRAVITIES[stellar_class]:
            for metallicity in METALLICITIES:
                track = TrackSpec(
                    log_surface_gravity=float(logg),
                    metallicity=float(metallicity),
                    alpha_enhancement=0.0,
                    carbon_enhancement=0.0,
                    microturbulence_km_s=MICROTURBULENCE[stellar_class],
                )
                tracks.append(
                    {
                        **track.as_json(),
                        "class": stellar_class,
                        "role": _track_role(
                            stellar_class,
                            float(logg),
                            float(metallicity),
                        ),
                    }
                )
    return tracks


def build_candidates(
    tracks: list[dict[str, Any]] | None = None,
    *,
    role: str = "train",
) -> list[dict[str, Any]]:
    tracks = build_tracks() if tracks is None else tracks
    candidates: list[dict[str, Any]] = []
    for stellar_class in ("giant", "dwarf"):
        selected_tracks = sorted(
            (
                track
                for track in tracks
                if track["class"] == stellar_class and track["role"] == role
            ),
            key=lambda row: (
                float(row["metallicity"]),
                float(row["log_surface_gravity"]),
            ),
        )
        priority = 0
        for temperature in TEMPERATURE_PRIORITY:
            for track in selected_tracks:
                candidates.append(
                    {
                        "candidate_id": (
                            f"{track['track_id']}_t{int(temperature):04d}"
                        ),
                        "priority": priority,
                        "temperature_K": float(temperature),
                        "class": stellar_class,
                        "role": role,
                        "track": track,
                    }
                )
                priority += 1
    return candidates


def protocol_payload(
    result_root: Path,
    *,
    marcs_grid: Path,
    flux_parent_root: Path,
    validation_corpus: Path,
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
    if not validation_corpus.is_file():
        raise FileNotFoundError(validation_corpus)

    tracks = build_tracks()
    candidates = build_candidates(tracks)
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
        "marcs_seed": {
            "path": str(schema.path),
            "sha256": schema.sha256,
            "native_nodes_only": True,
            "depth_coordinate": "log_mass",
            "fields_passed_to_payne_zero": ["column_mass", "temperature"],
            "other_native_fields_are_diagnostics_only": True,
            "is_training_target": False,
        },
        "grid": {
            "temperature_priority_K": list(TEMPERATURE_PRIORITY),
            "metallicity": list(METALLICITIES),
            "alpha_enhancement": 0.0,
            "carbon_enhancement": 0.0,
            "giant_logg": list(CLASS_GRAVITIES["giant"]),
            "dwarf_logg": list(CLASS_GRAVITIES["dwarf"]),
            "microturbulence_km_s": dict(MICROTURBULENCE),
        },
        "split": {
            "unit": "complete (class, logg, metallicity) track",
            "tracks": tracks,
            "new_train_candidates": candidates,
            "new_train_candidate_count": len(candidates),
            "new_validation_tracks_run": False,
            "new_sealed_tracks_run": False,
            "opened_validation_import": str(validation_corpus),
            "opened_validation_import_sha256": _sha256(validation_corpus),
        },
        "quota": {
            "target_total_training_rows": 2 * TARGET_PER_CLASS,
            "target_per_class": {
                "giant": TARGET_PER_CLASS,
                "dwarf": TARGET_PER_CLASS,
            },
            "selection": (
                "first 50 eligible rows per class in frozen priority order"
            ),
            "batch_overshoot": "retained as reserve; not admitted to training",
            "pool_exhaustion": "FAIL_STOP",
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
            "new_sealed_tracks_run": False,
            "korg_run": False,
            "marcs_is_training_target": False,
            "v1_and_v1r1_results_mutated": False,
        },
        "paths": {
            "result_root": str(result_root),
            "protocol": str(result_root / "protocol.json"),
            "flux_gate": str(result_root / "flux_gate.json"),
            "cases": str(result_root / "cases"),
            "cool_corpus": str(result_root / "cool_truth_corpus.npz"),
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


def _track_from_payload(payload: dict[str, Any]) -> TrackSpec:
    return TrackSpec(
        log_surface_gravity=float(payload["log_surface_gravity"]),
        metallicity=float(payload["metallicity"]),
        alpha_enhancement=float(payload["alpha_enhancement"]),
        carbon_enhancement=float(payload["carbon_enhancement"]),
        microturbulence_km_s=float(payload["microturbulence_km_s"]),
    )


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
        schema = inspect_marcs_grid(
            Path(marcs_grid_text),
            verify_sha256=False,
            expected_sha256=None,
        )
        node = load_marcs_node(
            Path(marcs_grid_text),
            labels,
            carbon_enhancement=float(track.carbon_enhancement),
            verify_sha256=False,
            expected_sha256=None,
            schema=schema,
            depth_coordinate="log_mass",
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


def class_status(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    stellar_class: str,
) -> dict[str, Any]:
    candidates = [
        row
        for row in protocol["split"]["new_train_candidates"]
        if row["class"] == stellar_class
    ]
    records = [
        row
        for row in load_case_records(result_root)
        if row["class"] == stellar_class
    ]
    by_id = {str(row["candidate_id"]): row for row in records}
    eligible = [
        row for row in records if bool(row.get("training_eligible"))
    ]
    pending = [
        row
        for row in candidates
        if str(row["candidate_id"]) not in by_id
    ]
    return {
        "campaign": CAMPAIGN,
        "class": stellar_class,
        "target": TARGET_PER_CLASS,
        "candidate_count": len(candidates),
        "attempted_count": len(records),
        "eligible_count": len(eligible),
        "pending_count": len(pending),
        "quota_reached": len(eligible) >= TARGET_PER_CLASS,
        "pool_exhausted": not pending and len(eligible) < TARGET_PER_CLASS,
        "eligible_candidate_ids": [
            row["candidate_id"]
            for row in sorted(eligible, key=lambda item: int(item["priority"]))
        ],
        "pending_candidates": pending,
    }


def aggregate_status(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    write: bool = True,
) -> dict[str, Any]:
    classes = {
        stellar_class: class_status(
            result_root,
            protocol=protocol,
            stellar_class=stellar_class,
        )
        for stellar_class in ("giant", "dwarf")
    }
    payload = {
        "campaign": CAMPAIGN,
        "protocol_hash": protocol["protocol_hash"],
        "classes": classes,
        "quota_reached": all(
            values["quota_reached"] for values in classes.values()
        ),
        "pool_exhausted": any(
            values["pool_exhausted"] for values in classes.values()
        ),
        "attempted_count": sum(
            values["attempted_count"] for values in classes.values()
        ),
        "eligible_count": sum(
            values["eligible_count"] for values in classes.values()
        ),
    }
    payload["status_hash"] = _hash_payload(payload)
    if write:
        _write_json(result_root / "status.json", payload)
    return payload


def run_class_until_quota(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    flux_gate: dict[str, Any],
    marcs_grid: Path,
    stellar_class: str,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    if stellar_class not in {"giant", "dwarf"}:
        raise ValueError("stellar_class must be giant or dwarf")
    if batch_size <= 0 or workers <= 0:
        raise ValueError("batch_size and workers must be positive")
    class_marker = result_root / f"{stellar_class.upper()}_QUOTA_REACHED"
    failure_marker = result_root / f"{stellar_class.upper()}_QUOTA_FAILED"
    while True:
        status = class_status(
            result_root,
            protocol=protocol,
            stellar_class=stellar_class,
        )
        _write_json(result_root / f"status_{stellar_class}.json", status)
        if status["quota_reached"]:
            class_marker.touch()
            return status
        if status["pool_exhausted"]:
            failure_marker.touch()
            raise RuntimeError(
                f"{stellar_class} candidate pool exhausted at "
                f"{status['eligible_count']}/{TARGET_PER_CLASS}"
            )
        pending = status["pending_candidates"][:batch_size]
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
                    "class": stellar_class,
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


def select_training_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for stellar_class in ("giant", "dwarf"):
        eligible = sorted(
            (
                row
                for row in records
                if row["class"] == stellar_class
                and row["role"] == "train"
                and bool(row.get("training_eligible"))
            ),
            key=lambda row: int(row["priority"]),
        )
        if len(eligible) < TARGET_PER_CLASS:
            raise RuntimeError(
                f"{stellar_class} quota not reached: "
                f"{len(eligible)}/{TARGET_PER_CLASS}"
            )
        selected.extend(eligible[:TARGET_PER_CLASS])
    return selected


def build_cool_corpus(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    validation_corpus: Path,
) -> dict[str, Any]:
    records = load_case_records(result_root)
    selected = select_training_records(records)
    selected_ids = {str(row["candidate_id"]) for row in selected}
    for row in records:
        if str(row["candidate_id"]) in selected_ids:
            row["selected_for_training"] = True
            _write_json(
                _case_path(result_root, row),
                row,
            )

    train_labels: list[list[float]] = []
    train_mass: list[np.ndarray] = []
    train_temperature: list[np.ndarray] = []
    train_flux: list[list[float]] = []
    for row in selected:
        values = row["labels"]
        train_labels.append(
            [
                values["effective_temperature"],
                values["log_surface_gravity"],
                values["metallicity"],
                values["alpha_enhancement"],
                values["microturbulence_km_s"],
            ]
        )
        mass, temperature = _load_mt(row["primary"]["product_path"])
        train_mass.append(mass)
        train_temperature.append(temperature)
        diagnostics = _final_diagnostics(row["primary"])
        train_flux.append([diagnostics[field] for field in FLUX_METRICS])

    with np.load(validation_corpus, allow_pickle=False) as parent:
        parent_roles = np.asarray(parent["roles"]).astype(str)
        if "sealed" in set(parent_roles):
            raise ValueError("validation parent corpus contains sealed rows")
        validation_index = np.flatnonzero(parent_roles == "validation")
        if not len(validation_index):
            raise ValueError("validation parent corpus has no validation rows")
        validation_labels = np.asarray(
            parent["labels"][validation_index],
            dtype=np.float64,
        )
        validation_mass = np.asarray(
            parent["column_mass"][validation_index],
            dtype=np.float64,
        )
        validation_temperature = np.asarray(
            parent["temperature"][validation_index],
            dtype=np.float64,
        )
        validation_track_ids = np.asarray(
            parent["track_ids"][validation_index]
        ).astype(str)
        validation_node_ids = np.asarray(
            parent["node_ids"][validation_index]
        ).astype(str)
        validation_products = np.asarray(
            parent["source_product_paths"][validation_index]
        ).astype(str)
        validation_flux = np.asarray(
            parent["flux_metrics"][validation_index],
            dtype=np.float64,
        )

    labels = np.vstack(
        [np.asarray(train_labels, dtype=np.float64), validation_labels]
    )
    column_mass = np.vstack(
        [np.asarray(train_mass, dtype=np.float64), validation_mass]
    )
    temperature = np.vstack(
        [np.asarray(train_temperature, dtype=np.float64), validation_temperature]
    )
    flux_metrics = np.vstack(
        [np.asarray(train_flux, dtype=np.float64), validation_flux]
    )
    roles = np.asarray(
        ["train"] * len(selected)
        + ["validation"] * len(validation_index),
        dtype="U16",
    )
    track_ids = np.asarray(
        [row["track"]["track_id"] for row in selected]
        + validation_track_ids.tolist(),
        dtype="U96",
    )
    node_ids = np.asarray(
        [row["candidate_id"] for row in selected]
        + validation_node_ids.tolist(),
        dtype="U128",
    )
    source_products = np.asarray(
        [row["primary"]["product_path"] for row in selected]
        + validation_products.tolist(),
        dtype="U512",
    )
    source_campaigns = np.asarray(
        [CAMPAIGN] * len(selected)
        + ["m_star_emulator_v1r1_policy60"] * len(validation_index),
        dtype="U64",
    )
    flux_gate = _read_json(result_root / "flux_gate.json")
    output_path = result_root / "cool_truth_corpus.npz"
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            labels=labels,
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
            column_mass=column_mass,
            temperature=temperature,
            roles=roles,
            track_ids=track_ids,
            node_ids=node_ids,
            source_product_paths=source_products,
            source_campaigns=source_campaigns,
            flux_metric_fields=np.asarray(FLUX_METRICS, dtype="U64"),
            flux_metrics=flux_metrics,
            protocol_hash=np.asarray(
                [protocol["protocol_hash"]],
                dtype="U64",
            ),
            flux_gate_hash=np.asarray([flux_gate["gate_hash"]], dtype="U64"),
        )
    temporary.replace(output_path)
    train_labels_array = labels[roles == "train"]
    validation_labels_array = labels[roles == "validation"]
    summary = {
        "campaign": CAMPAIGN,
        "status": "complete",
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "row_count": int(len(labels)),
        "role_counts": {
            "train": int(np.sum(roles == "train")),
            "validation": int(np.sum(roles == "validation")),
        },
        "role_class_counts": {
            "train": {
                "giant": int(np.sum(train_labels_array[:, 1] < 3.5)),
                "dwarf": int(np.sum(train_labels_array[:, 1] >= 3.5)),
            },
            "validation": {
                "giant": int(np.sum(validation_labels_array[:, 1] < 3.5)),
                "dwarf": int(np.sum(validation_labels_array[:, 1] >= 3.5)),
            },
        },
        "selected_training_candidate_ids": [
            row["candidate_id"]
            for row in sorted(
                selected,
                key=lambda item: (item["class"], int(item["priority"])),
            )
        ],
        "eligible_reserve_count": int(
            sum(bool(row.get("training_eligible")) for row in records)
            - len(selected)
        ),
        "validation_import": str(validation_corpus),
        "validation_import_sha256": _sha256(validation_corpus),
        "protocol_hash": protocol["protocol_hash"],
        "flux_gate_hash": flux_gate["gate_hash"],
        "sealed_rows_included": False,
        "marcs_is_training_target": False,
    }
    _write_json(result_root / "cool_truth_corpus.json", summary)
    (result_root / "CORPUS_READY").touch()
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "protocol",
            "import-gate",
            "class-until-quota",
            "status",
            "build-corpus",
        ),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--marcs-grid", type=Path, default=DEFAULT_MARCS_GRID)
    parser.add_argument(
        "--flux-parent-root",
        type=Path,
        default=DEFAULT_FLUX_PARENT_ROOT,
    )
    parser.add_argument(
        "--validation-corpus",
        type=Path,
        default=DEFAULT_VALIDATION_CORPUS,
    )
    parser.add_argument(
        "--stellar-class",
        choices=("giant", "dwarf"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
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
            validation_corpus=args.validation_corpus,
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

    if args.stage == "class-until-quota":
        if args.stellar_class is None:
            raise SystemExit("--stellar-class is required")
        status = run_class_until_quota(
            args.result_root,
            protocol=protocol,
            flux_gate=flux_gate,
            marcs_grid=args.marcs_grid,
            stellar_class=args.stellar_class,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    if args.stage == "status":
        status = aggregate_status(
            args.result_root,
            protocol=protocol,
        )
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    if args.stage == "build-corpus":
        status = aggregate_status(
            args.result_root,
            protocol=protocol,
        )
        if not status["quota_reached"]:
            raise SystemExit(
                "FAIL_STOP: 50 giant and 50 dwarf eligible rows are required"
            )
        summary = build_cool_corpus(
            args.result_root,
            protocol=protocol,
            validation_corpus=args.validation_corpus,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"unhandled stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
