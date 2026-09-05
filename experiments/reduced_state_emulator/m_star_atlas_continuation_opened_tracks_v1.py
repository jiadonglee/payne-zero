"""Small-step ATLAS continuation along the two interpolation-opened dwarf tracks.

The interpolated ``(m,T)`` arm opened two dwarf tracks past the v1r2 cold
edge: log g=4.5/[M/H]=0 down to 3500 and 3300 K, and log g=4.5/[M/H]=-0.5
down to 3800, 3750 and 3700 K.  This arm starts from the newly gated
Payne-Zero/ATLAS ``(m,T)`` products on those tracks and walks to the next
frozen v1r2 grid node with at most 50 K steps (25 K after one backtrack).
Only ``(m,T)`` is carried; Payne-Zero rematerializes the other four fields
through the exact physical reconstruction path.  Solver settings and
admission gates are copied from v1r2 unchanged.

Intermediate continuation waypoints that are not v1r2 grid nodes only need
solver convergence and a finite valid six-field state.  Every v1r2 node that
should count toward the corpus must pass the full v1r2 eligibility.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .cool_star_adaptive import backtrack_step, proposed_temperature
from .cool_star_step_test import (
    TrackSpec,
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
from .m_star_interpolated_mt_seed_v1 import _flux_metric, _read_json


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_atlas_continuation_opened_tracks_v1"
ITERATION_CAP = base.ITERATION_CAP
STRICT_ALL_LAYER_LIMIT = base.STRICT_ALL_LAYER_LIMIT
INITIAL_STEP_K = 50.0
MINIMUM_STEP_K = 25.0
TEMPERATURE_TOLERANCE_K = 1.0e-9
SOURCE_CAMPAIGN = "m_star_interpolated_mt_seed_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_PARENT_ROOT = REPO_ROOT / "results" / "m_star_emulator_v1r2_marcs100"
DEFAULT_INTERP_ROOT = REPO_ROOT / "results" / SOURCE_CAMPAIGN
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "notes"
    / "m_star_atlas_continuation_opened_tracks_v1_preregistration_20260904.md"
)

TRACK_A = TrackSpec(
    log_surface_gravity=4.5,
    metallicity=0.0,
).as_json()
TRACK_B = TrackSpec(
    log_surface_gravity=4.5,
    metallicity=-0.5,
).as_json()

# Frozen seed identities from the interpolation arm.  The sha256 values are
# the gated primary products on Garching at plan time; the remaining
# interpolation scale candidates (3100 K extrapolations) cannot change them.
SEEDS = {
    "probe_a": {
        "candidate_id": "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3500",
        "temperature_K": 3500.0,
        "product_sha256": (
            "295fa503b4296414a408040d48e3c646d0be27304f33aaa8dafe22077232b306"
        ),
    },
    "probe_b": {
        "candidate_id": "g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3700",
        "temperature_K": 3700.0,
        "product_sha256": (
            "ef75d3245a5462e381c0a343dd007b66e67cee79aedc3d03bf9ca1960e245889"
        ),
    },
    "track_a_backup": {
        "candidate_id": "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3300",
        "temperature_K": 3300.0,
        "product_sha256": (
            "54790c865451ec25704a4b380f2a56173f1715ca9f2b4b9e2f3b3f54644cb2b5"
        ),
    },
}
PROBES = {
    "A": {
        "track": TRACK_A,
        "seed": "probe_a",
        "target_temperature_K": 3400.0,
    },
    "B": {
        "track": TRACK_B,
        "seed": "probe_b",
        "target_temperature_K": 3600.0,
    },
}
WALK_TARGETS = {
    "A": (3200.0, 3100.0, 3000.0),
    "B": (3500.0, 3400.0, 3300.0, 3200.0, 3100.0, 3000.0),
}
GRID_TEMPERATURES = frozenset(float(value) for value in base.TEMPERATURE_PRIORITY)


def track_spec(track: dict[str, Any]) -> TrackSpec:
    return TrackSpec(
        log_surface_gravity=float(track["log_surface_gravity"]),
        metallicity=float(track["metallicity"]),
        alpha_enhancement=float(track["alpha_enhancement"]),
        carbon_enhancement=float(track["carbon_enhancement"]),
        microturbulence_km_s=float(track["microturbulence_km_s"]),
    )


def candidate_id(track: dict[str, Any], temperature_K: float) -> str:
    return f"{track_spec(track).track_id}_t{int(temperature_K):04d}"


def track_label(track: dict[str, Any]) -> str:
    return (
        f"g{float(track['log_surface_gravity']):+04.1f}"
        f"_m{float(track['metallicity']):+04.1f}"
    )


def probe_candidate_id(name: str) -> str:
    probe = PROBES[name]
    return candidate_id(probe["track"], probe["target_temperature_K"])


def nominal_steps(
    source_temperature: float,
    target_temperature: float,
    initial_step: float,
) -> list[float]:
    """Waypoints of an uninterrupted continuation, never overshooting."""

    current = float(source_temperature)
    target = float(target_temperature)
    step = float(initial_step)
    waypoints: list[float] = []
    while abs(current - target) > TEMPERATURE_TOLERANCE_K:
        current = proposed_temperature(current, target, step)
        waypoints.append(current)
    return waypoints


def probe_walk_decision(
    probe_eligible: dict[str, bool],
) -> dict[str, Any]:
    """Apply the frozen 2/2, 1/2, 0/2 probe decision rules."""

    tracks: dict[str, Any] = {}
    for name, probe in PROBES.items():
        opened = bool(probe_eligible.get(name))
        tracks[name] = {
            "track": dict(probe["track"]),
            "probe_eligible": opened,
            "opened": opened,
            "seed_temperature_K": (
                float(probe["target_temperature_K"]) if opened else None
            ),
            "targets_K": (
                [float(value) for value in WALK_TARGETS[name]] if opened else []
            ),
        }
    count = sum(bool(value) for value in probe_eligible.values())
    return {
        "campaign": CAMPAIGN,
        "rule": (
            "2/2 probes eligible: walk both; 1/2: walk the passing track "
            "only; 0/2: stop, smaller steps or cross-track neighbours are "
            "out of scope"
        ),
        "probe_eligible_count": count,
        "decision": (
            "walk_both" if count == 2 else "walk_single" if count == 1 else "stop"
        ),
        "tracks": tracks,
    }


def validate_walk_decision(
    decision: dict[str, Any],
    *,
    protocol: dict[str, Any],
) -> None:
    """Refuse any walk outside the frozen probe-derived plan."""

    if decision.get("campaign") != CAMPAIGN:
        raise ValueError("walk decision belongs to another campaign")
    if decision.get("protocol_hash") != protocol["protocol_hash"]:
        raise ValueError("walk decision was not written against this protocol")
    flags = {
        name: bool(decision["tracks"][name].get("probe_eligible"))
        for name in PROBES
    }
    expected = probe_walk_decision(flags)
    if decision.get("decision") != expected["decision"]:
        raise ValueError("walk decision contradicts the probe outcome")
    for name, track in decision["tracks"].items():
        expected_track = expected["tracks"][name]
        if bool(track["opened"]) != expected_track["opened"]:
            raise ValueError(f"track {name} opening drifted from the probe outcome")
        if [float(value) for value in track["targets_K"]] != expected_track[
            "targets_K"
        ]:
            raise ValueError(f"track {name} targets drifted from the frozen walk")
        for value in track["targets_K"]:
            if float(value) not in GRID_TEMPERATURES:
                raise ValueError(
                    f"track {name} target {value} is not a v1r2 grid node"
                )


def seed_stem(candidate_id_value: str) -> str:
    return str(candidate_id_value).split("_t")[0]


def _case_dir(root: Path, candidate_id_value: str) -> Path:
    return (
        root
        / "cases"
        / "dwarf"
        / seed_stem(candidate_id_value)
        / f"t{int(candidate_id_value.split('_t')[-1]):04d}"
    )


def resolve_interp_product(
    interp_root: Path,
    candidate_id_value: str,
) -> Path:
    """Locate an interpolation-arm primary product for a gated dwarf node."""

    case_dir = _case_dir(interp_root, candidate_id_value)
    record = _read_json(case_dir / "case.json")
    if not bool(record.get("training_eligible")):
        raise ValueError(
            f"{candidate_id_value} is not a gated interpolation-arm product"
        )
    return _resolve_product(record, case_dir)


def _resolve_product(record: dict[str, Any], case_dir: Path) -> Path:
    recorded = Path(record["primary"]["product_path"])
    if recorded.is_file():
        return recorded
    local = case_dir / "products" / "primary" / recorded.name
    if not local.is_file():
        raise FileNotFoundError(local)
    return local


def _parent_record(
    parent_root: Path,
    track: dict[str, Any],
    temperature: float,
) -> dict[str, Any] | None:
    target_id = candidate_id(track, temperature)
    for record in base.load_case_records(parent_root):
        if str(record.get("candidate_id")) == target_id:
            return record
    return None


def _interp_control(
    interp_root: Path,
    track: dict[str, Any],
    temperature: float,
) -> dict[str, Any]:
    interp_id = candidate_id(track, temperature)
    record = _read_json(_case_dir(interp_root, interp_id) / "case.json")
    donor = (record.get("interpolated_seed") or {}).get("donor_candidate_ids")
    flux = _flux_metric(record, "p95_absolute_flux_error_percent")
    return {
        "campaign": SOURCE_CAMPAIGN,
        "candidate_id": interp_id,
        "training_eligible": bool(record.get("training_eligible")),
        "donor_candidate_ids": donor,
        "iterations": (record.get("primary") or {}).get("iterations"),
        "p95_absolute_flux_error_percent": flux,
        "summary": (
            "{donor}, {iters} iters, flux p95 {flux}".format(
                donor=(donor or ["—"])[0],
                iters=(record.get("primary") or {}).get("iterations") or "—",
                flux="—" if flux is None else f"{flux:.3g}",
            )
        ),
    }


def _continuation_case(
    candidate: dict[str, Any],
    *,
    seed: dict[str, Any],
    seed_product: Path,
    expected_seed_sha256: str | None,
    flux_gate: dict[str, Any],
    protocol_hash: str,
    result_root: Path,
) -> dict[str, Any]:
    _set_single_thread_environment()
    case_path = base._case_path(result_root, candidate)
    if case_path.is_file():
        return _read_json(case_path)
    case_root = case_path.parent

    seed_sha256 = _sha256(seed_product)
    if expected_seed_sha256 is not None and seed_sha256 != expected_seed_sha256:
        raise ValueError(
            f"seed product {seed_product} drifted from the frozen sha256"
        )
    seed_mass, seed_profile = _load_mt(seed_product)

    track_payload = candidate["track"]
    track = track_spec(track_payload)
    target_temperature = float(candidate["temperature_K"])
    steps: list[dict[str, Any]] = []
    current_temperature = float(seed["temperature_K"])
    current_mass = seed_mass
    current_profile = seed_profile
    step_size = float(INITIAL_STEP_K)
    primary: dict[str, Any] | None = None
    restart: dict[str, Any] | None = None
    primary_flux: dict[str, Any] | None = None
    restart_flux: dict[str, Any] | None = None
    consistency: dict[str, Any] | None = None
    reached = False
    stopped_reason: str | None = None

    while abs(current_temperature - target_temperature) > TEMPERATURE_TOLERANCE_K:
        proposed = proposed_temperature(
            current_temperature, target_temperature, step_size
        )
        is_target = abs(proposed - target_temperature) <= TEMPERATURE_TOLERANCE_K
        labels = track.labels(proposed)
        record: dict[str, Any] | None = None
        state = None
        try:
            seed_atmosphere = _reconstruct_from_mt(
                labels,
                current_mass,
                current_profile,
            )
        except Exception as exc:  # noqa: BLE001 - a failed step is an outcome
            steps.append(
                {
                    "status": "initialization_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "requested_step_K": step_size,
                    "source_temperature_K": current_temperature,
                    "target_temperature_K": proposed,
                    "is_target_node": is_target,
                }
            )
        else:
            record, state = _solve_attempt(
                track=track,
                method="atlas_continuation_rematerialized",
                schedule="atlas_continuation",
                source_temperature=current_temperature,
                target_labels=labels,
                initial_atmosphere=seed_atmosphere,
                product_dir=(
                    case_root / "products" / "primary"
                    if is_target
                    else case_root / "products" / "continuation"
                ),
                iteration_cap=ITERATION_CAP,
                maximum_all_layer_relative_temperature_change=(
                    STRICT_ALL_LAYER_LIMIT
                ),
            )
            record.update(
                {
                    "requested_step_K": step_size,
                    "is_target_node": is_target,
                }
            )
            record = _annotate_record(
                record,
                track_payload=track_payload,
                role="train",
                node_id=str(candidate["candidate_id"]),
            )
            steps.append(record)

        survives = bool(record is not None and record.get("survives_solver"))
        if survives and not is_target:
            current_temperature = proposed
            current_mass = np.asarray(state.column_mass, dtype=np.float64)
            current_profile = np.asarray(state.temperature, dtype=np.float64)
            continue

        if survives and is_target:
            primary = record
            solved_mass, solved_temperature = _load_mt(primary["product_path"])
            restart_seed = _reconstruct_from_mt(
                labels,
                solved_mass,
                solved_temperature,
            )
            restart, _restart_state = _solve_attempt(
                track=track,
                method="strict_self_restart_from_atlas_mt",
                schedule="independent_self_restart",
                source_temperature=target_temperature,
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
            restart_flux = _passes_flux_gate(restart, flux_gate)
            consistency = _product_consistency(
                primary.get("product_path"),
                restart.get("product_path"),
            )
            eligible = bool(
                restart.get("survives_solver")
                and restart.get("state_quality", {}).get("valid")
                and primary_flux["passes"]
                and restart_flux["passes"]
                and consistency["passes"]
            )
            if eligible:
                reached = True
                break

        narrowed = backtrack_step(step_size, MINIMUM_STEP_K)
        if narrowed >= step_size - TEMPERATURE_TOLERANCE_K:
            stopped_reason = "minimum_step_failed"
            break
        step_size = narrowed

    accepted = [
        float(entry["target_temperature_K"])
        for entry in steps
        if entry.get("survives_solver")
        and entry.get("target_temperature_K") is not None
    ]
    eligible = bool(reached)
    reasons: list[str] = []
    if not reached:
        reasons.append(stopped_reason or "target_not_reached")
    else:
        if not restart.get("survives_solver"):
            reasons.append("self_restart")
        if not primary_flux["passes"]:
            reasons.append("primary_flux_gate")
        if not restart_flux["passes"]:
            reasons.append("restart_flux_gate")
        if not consistency["passes"]:
            reasons.append("path_consistency")
    output: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "protocol_hash": protocol_hash,
        **candidate,
        "labels": track.labels(target_temperature).as_kwargs(),
        "continuation": {
            "mode": "reduced_rematerialized",
            "fields_carried": ["column_mass", "temperature"],
            "seed": {
                "campaign": seed["campaign"],
                "role": seed["role"],
                "candidate_id": seed["candidate_id"],
                "temperature_K": float(seed["temperature_K"]),
                "product_path": str(seed_product),
                "product_sha256": seed_sha256,
            },
            "initial_step_K": INITIAL_STEP_K,
            "minimum_step_K": MINIMUM_STEP_K,
            "final_step_K": step_size,
            "steps": steps,
            "accepted_temperatures_K": accepted,
            "reached_target": reached,
        },
        "primary": primary,
        "restart": restart,
        "primary_flux_gate": primary_flux or {"passes": False, "metrics": {}},
        "restart_flux_gate": restart_flux or {"passes": False, "metrics": {}},
        "path_consistency": consistency
        or {"available": False, "passes": False},
        "training_eligible": eligible,
        "status": "training_eligible" if eligible else "ineligible",
        "failure_reason": None if eligible else ",".join(reasons),
    }
    _write_json(case_path, output)
    return output


def _probe_candidate(parent_root: Path, name: str) -> dict[str, Any]:
    probe = PROBES[name]
    track = probe["track"]
    temperature = float(probe["target_temperature_K"])
    parent_record = _parent_record(parent_root, track, temperature)
    if parent_record is None:
        raise ValueError(f"{candidate_id(track, temperature)} is not a v1r2 grid node")
    return {
        "candidate_id": candidate_id(track, temperature),
        "priority": int(parent_record["priority"]),
        "temperature_K": temperature,
        "class": "dwarf",
        "role": "train",
        "track": dict(track),
        "probe": name,
    }


def _probe_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        name,
        result_root_text,
        parent_root_text,
        interp_root_text,
        flux_gate,
        protocol_hash,
    ) = payload
    result_root = Path(result_root_text)
    parent_root = Path(parent_root_text)
    interp_root = Path(interp_root_text)
    candidate = _probe_candidate(parent_root, name)
    seed_spec = SEEDS[PROBES[name]["seed"]]
    seed_product = resolve_interp_product(
        interp_root,
        str(seed_spec["candidate_id"]),
    )
    output = _continuation_case(
        candidate,
        seed={
            "campaign": SOURCE_CAMPAIGN,
            "role": f"probe_{name.lower()}_seed",
            **seed_spec,
        },
        seed_product=seed_product,
        expected_seed_sha256=str(seed_spec["product_sha256"]),
        flux_gate=flux_gate,
        protocol_hash=protocol_hash,
        result_root=result_root,
    )
    output["control"] = _interp_control(
        interp_root,
        candidate["track"],
        float(candidate["temperature_K"]),
    )
    _write_json(base._case_path(result_root, candidate), output)
    return output


def _chain_seed(
    result_root: Path,
    seed_candidate_id: str,
    target_temperature: float,
) -> tuple[dict[str, Any], Path]:
    """The just-gated cell of this track's chain, warmer than the target."""

    case_dir = _case_dir(result_root, seed_candidate_id)
    record = _read_json(case_dir / "case.json")
    if not bool(record.get("training_eligible")):
        raise ValueError(
            f"walk seed {seed_candidate_id} has not passed full eligibility"
        )
    seed_temperature = float(record["temperature_K"])
    if seed_temperature <= target_temperature:
        raise ValueError(
            f"walk seed {seed_temperature} K is not warmer than the target"
        )
    seed = {
        "campaign": CAMPAIGN,
        "role": "walk_chain_seed",
        "candidate_id": seed_candidate_id,
        "temperature_K": seed_temperature,
    }
    return seed, _resolve_product(record, case_dir)


def _track_walk_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        name,
        targets,
        seed_candidate_id,
        seed_product_sha256,
        result_root_text,
        parent_root_text,
        interp_root_text,
        flux_gate,
        protocol_hash,
    ) = payload
    result_root = Path(result_root_text)
    parent_root = Path(parent_root_text)
    interp_root = Path(interp_root_text)
    track = PROBES[name]["track"]
    outcomes: list[dict[str, Any]] = []
    for index, temperature in enumerate(targets):
        temperature = float(temperature)
        parent_record = _parent_record(parent_root, track, temperature)
        if parent_record is None:
            raise ValueError(
                f"{candidate_id(track, temperature)} is not a v1r2 grid node"
            )
        candidate = {
            "candidate_id": candidate_id(track, temperature),
            "priority": int(parent_record["priority"]),
            "temperature_K": temperature,
            "class": "dwarf",
            "role": "train",
            "track": dict(track),
            "probe": name,
        }
        seed, seed_product = _chain_seed(
            result_root,
            seed_candidate_id,
            temperature,
        )
        output = _continuation_case(
            candidate,
            seed=seed,
            seed_product=seed_product,
            expected_seed_sha256=(
                str(seed_product_sha256) if index == 0 else None
            ),
            flux_gate=flux_gate,
            protocol_hash=protocol_hash,
            result_root=result_root,
        )
        output["control"] = _interp_control(interp_root, track, temperature)
        _write_json(base._case_path(result_root, candidate), output)
        outcomes.append(output)
        if not bool(output.get("training_eligible")):
            break
        seed_candidate_id = str(candidate["candidate_id"])
        seed_product_sha256 = None
    return {"track": name, "outcomes": outcomes}


def load_case_records(result_root: Path) -> list[dict[str, Any]]:
    return [
        _read_json(path)
        for path in sorted((result_root / "cases").glob("*/*/t*/case.json"))
    ]


def protocol_payload(
    result_root: Path,
    *,
    parent_root: Path,
    interp_root: Path,
) -> dict[str, Any]:
    parent_protocol = _read_json(parent_root / "protocol.json")
    parent_status = _read_json(parent_root / "status_dwarf.json")
    parent_gate = _read_json(parent_root / "flux_gate.json")
    if parent_protocol.get("campaign") != base.CAMPAIGN:
        raise ValueError("unexpected parent campaign")
    if (
        int(parent_status.get("attempted_count", -1)) != 108
        or int(parent_status.get("eligible_count", -1)) != 26
        or not bool(parent_status.get("pool_exhausted"))
    ):
        raise ValueError("parent dwarf campaign is not at the frozen FAIL_STOP")
    if not parent_gate.get("frozen") or parent_gate.get("thresholds_refit"):
        raise ValueError("parent flux gate is not frozen")

    interp_status = _read_json(interp_root / "status.json")
    if not (interp_root / "SCALE_POOL_EXHAUSTED").is_file():
        raise ValueError("interpolation arm scale has not drained its pool")
    if (
        not bool(interp_status.get("probe_complete"))
        or int(interp_status.get("attempted_count", -1)) != 58
        or int(interp_status.get("eligible_count", -1)) < 6
        or interp_status.get("pending_scale_candidates")
    ):
        raise ValueError("interpolation arm closeout state drifted")

    probes: dict[str, Any] = {}
    for name, probe in PROBES.items():
        track = probe["track"]
        target_temperature = float(probe["target_temperature_K"])
        target_id = candidate_id(track, target_temperature)
        parent_record = _parent_record(parent_root, track, target_temperature)
        if parent_record is None:
            raise ValueError(f"{target_id} is not a v1r2 grid node")
        if bool(parent_record.get("training_eligible")):
            raise ValueError(f"{target_id} already passed in v1r2")
        interp_record = _read_json(
            _case_dir(interp_root, target_id) / "case.json"
        )
        if bool(interp_record.get("training_eligible")):
            raise ValueError(
                f"{target_id} already passed in the interpolation arm"
            )
        seed_spec = SEEDS[probe["seed"]]
        seed_product = resolve_interp_product(
            interp_root,
            str(seed_spec["candidate_id"]),
        )
        probes[name] = {
            "track": dict(track),
            "seed": {
                **seed_spec,
                "campaign": SOURCE_CAMPAIGN,
                "product_path": str(seed_product),
                "product_sha256_verified": _sha256(seed_product),
            },
            "target": {
                "candidate_id": target_id,
                "temperature_K": target_temperature,
                "v1r2_failure_reason": parent_record.get("failure_reason"),
                "interp_control": _interp_control(
                    interp_root,
                    track,
                    target_temperature,
                ),
            },
        }

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
            "campaign": parent_protocol["campaign"],
            "result_root": str(parent_root.resolve()),
            "protocol_hash": parent_protocol["protocol_hash"],
            "existing_eligible_dwarf_count": 26,
        },
        "seed_campaign": {
            "campaign": SOURCE_CAMPAIGN,
            "result_root": str(interp_root.resolve()),
            "attempted_count": int(interp_status["attempted_count"]),
            "eligible_count": int(interp_status["eligible_count"]),
            "pool_drained": True,
        },
        "continuation": {
            "mode": "reduced_rematerialized",
            "fields_carried": ["column_mass", "temperature"],
            "initial_step_K": INITIAL_STEP_K,
            "minimum_step_K": MINIMUM_STEP_K,
            "backtrack": "halve on failure, floor at 25 K, no smaller step",
            "same_track_only": True,
            "cross_track_knn": False,
            "marcs_seed_used": False,
            "intermediate_waypoints": (
                "solver convergence plus finite valid six-field state only"
            ),
            "v1r2_grid_nodes": "full v1r2 eligibility required",
            "walk_targets": {
                name: [float(value) for value in targets]
                for name, targets in WALK_TARGETS.items()
            },
            "walk_gate": "walk_decision.json written by the probe stage",
        },
        "probe": {
            "selection": (
                "one next v1r2 node per opened track: A 3500->3400, "
                "B 3700->3600"
            ),
            "probes": probes,
        },
        "seeds": {
            name: {**spec, "campaign": SOURCE_CAMPAIGN}
            for name, spec in SEEDS.items()
        },
        "solver": {
            "seed": "gated ATLAS (m,T) from the same track, rematerialized",
            "truth_source": "terminal Payne-Zero ATLAS atmosphere",
            "iteration_cap": ITERATION_CAP,
            "maximum_all_layer_relative_temperature_change": (
                STRICT_ALL_LAYER_LIMIT
            ),
            "independent_restart": (
                "strict self-restart from terminal ATLAS (m,T)"
            ),
            "only_change_from_parent_attempt": (
                "replace the MARCS seed with a gated same-track ATLAS (m,T) "
                "approached in <=50 K continuation steps"
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
            "interpolation_results_mutated": False,
            "training_run": False,
            "candidate_validation_run": False,
            "sealed_holdout_opened": False,
            "production_routing_changed": False,
            "korg_run": False,
            "marcs_is_training_target": False,
            "cross_track_knn": False,
            "full_carry_used": False,
        },
        "paths": {
            "result_root": str(result_root.resolve()),
            "protocol": str((result_root / "protocol.json").resolve()),
            "flux_gate": str((result_root / "flux_gate.json").resolve()),
            "cases": str((result_root / "cases").resolve()),
            "walk_decision": str((result_root / "walk_decision.json").resolve()),
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


def _steps_summary(record: dict[str, Any]) -> str:
    continuation = record.get("continuation") or {}
    used = sorted(
        {
            float(entry["requested_step_K"])
            for entry in continuation.get("steps", [])
            if entry.get("requested_step_K") is not None
        },
        reverse=True,
    )
    return "->".join(f"{value:g}" for value in used) if used else "—"


def _attempted_steps(record: dict[str, Any]) -> int:
    continuation = record.get("continuation") or {}
    return sum(
        1
        for entry in continuation.get("steps", [])
        if entry.get("requested_step_K") is not None
    )


def _table_row(record: dict[str, Any]) -> str:
    seed = (record.get("continuation") or {}).get("seed") or {}
    primary = record.get("primary") or {}
    flux = _flux_metric(record, "p95_absolute_flux_error_percent")
    seed_temperature = seed.get("temperature_K")
    return (
        "| {steps} | {att} | {seed} | {iters} | {flux} | {control} |".format(
            steps=_steps_summary(record),
            att=_attempted_steps(record),
            seed=(
                "—"
                if seed_temperature is None
                else f"{float(seed_temperature):.0f}"
            ),
            iters=primary.get("iterations") or "—",
            flux="—" if flux is None else f"{flux:.3g}",
            control=(record.get("control") or {}).get("summary", "—"),
        )
    )


def write_probe_table(
    result_root: Path,
    *,
    records: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Two-probe ATLAS continuation",
        "",
        "| probe | track | target | eligible | steps K | attempts"
        " | seed Teff | primary iters | flux p95 | interp control |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in ("A", "B"):
        record = records.get(name)
        if record is None:
            continue
        lines.append(
            "| {probe} | `{track}` | {target:.0f} | {elig} | {row} |".format(
                probe=name,
                track=track_label(record["track"]),
                target=float(record["temperature_K"]),
                elig=bool(record.get("training_eligible")),
                row=_table_row(record),
            )
        )
    (result_root / "probe_table.md").write_text("\n".join(lines) + "\n")


def _write_walk_table(
    result_root: Path,
    *,
    records: list[dict[str, Any]],
) -> None:
    lines = [
        "# Track walk",
        "",
        "| track | candidate | eligible | steps K | attempts | seed Teff"
        " | primary iters | flux p95 | interp control |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in sorted(
        records,
        key=lambda row: (track_label(row["track"]), -float(row["temperature_K"])),
    ):
        lines.append(
            "| `{track}` | `{cid}` | {elig} | {row} |".format(
                track=track_label(record["track"]),
                cid=record["candidate_id"],
                elig=bool(record.get("training_eligible")),
                row=_table_row(record),
            )
        )
    (result_root / "walk_table.md").write_text("\n".join(lines) + "\n")


def campaign_status(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    write: bool = True,
) -> dict[str, Any]:
    records = load_case_records(result_root)
    by_id = {str(row["candidate_id"]): row for row in records}
    probe_ids = {name: probe_candidate_id(name) for name in PROBES}
    probe_records = {
        name: by_id[candidate]
        for name, candidate in probe_ids.items()
        if candidate in by_id
    }
    walk_records = [
        row for row in records if str(row["candidate_id"]) not in set(probe_ids.values())
    ]
    decision_path = result_root / "walk_decision.json"
    decision = _read_json(decision_path) if decision_path.is_file() else None
    walk_by_track: dict[str, Any] = {}
    for name, targets in WALK_TARGETS.items():
        target_ids = {
            candidate_id(PROBES[name]["track"], value) for value in targets
        }
        rows = sorted(
            (
                row
                for row in walk_records
                if str(row["candidate_id"]) in target_ids
            ),
            key=lambda row: float(row["temperature_K"]),
            reverse=True,
        )
        opened = bool(decision and decision["tracks"][name]["opened"])
        eligible_ids = [
            row["candidate_id"]
            for row in rows
            if bool(row.get("training_eligible"))
        ]
        stopped = bool(
            opened
            and rows
            and not bool(rows[-1].get("training_eligible"))
            and len(rows) < len(targets)
        )
        walk_by_track[name] = {
            "opened": opened,
            "attempted_count": len(rows),
            "eligible_count": len(eligible_ids),
            "eligible_candidate_ids": eligible_ids,
            "stopped_after_failure": stopped,
            "complete": opened
            and len(rows) == len(targets)
            and all(bool(row.get("training_eligible")) for row in rows),
            "candidate_ids": [row["candidate_id"] for row in rows],
        }
    payload = {
        "campaign": CAMPAIGN,
        "protocol_hash": protocol["protocol_hash"],
        "probe_attempted_count": len(probe_records),
        "probe_eligible_count": sum(
            bool(row.get("training_eligible")) for row in probe_records.values()
        ),
        "probe_complete": len(probe_records) == len(PROBES),
        "decision": (decision or {}).get("decision"),
        "walk": walk_by_track,
        "new_eligible_dwarfs": sum(
            bool(row.get("training_eligible")) for row in records
        ),
        "candidate_ids": [row["candidate_id"] for row in records],
    }
    payload["status_hash"] = _hash_payload(payload)
    if write:
        _write_json(result_root / "status.json", payload)
        if probe_records:
            write_probe_table(result_root, records=probe_records)
        if walk_records:
            _write_walk_table(result_root, records=walk_records)
    return payload


def run_probe(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    flux_gate: dict[str, Any],
    parent_root: Path,
    interp_root: Path,
    workers: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    _run_workers(
        _probe_worker,
        [
            (
                name,
                str(result_root),
                str(parent_root),
                str(interp_root),
                flux_gate,
                protocol["protocol_hash"],
            )
            for name in PROBES
        ],
        workers=min(max(int(workers), 1), len(PROBES)),
    )
    status = campaign_status(result_root, protocol=protocol)
    probe_records = {
        name: _read_json(
            base._case_path(
                result_root,
                {
                    "class": "dwarf",
                    "track": PROBES[name]["track"],
                    "temperature_K": PROBES[name]["target_temperature_K"],
                    "candidate_id": probe_candidate_id(name),
                },
            )
        )
        for name in PROBES
    }
    print(
        json.dumps(
            {
                "attempted_in_batch": len(probe_records),
                "eligible_in_batch": sum(
                    bool(row.get("training_eligible"))
                    for row in probe_records.values()
                ),
                "seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    decision = probe_walk_decision(
        {
            name: bool(record.get("training_eligible"))
            for name, record in probe_records.items()
        }
    )
    decision["protocol_hash"] = protocol["protocol_hash"]
    for name, track in decision["tracks"].items():
        if track["opened"]:
            record = probe_records[name]
            track["seed_candidate_id"] = str(record["candidate_id"])
            track["seed_product_sha256"] = _sha256(
                record["primary"]["product_path"]
            )
    validate_walk_decision(decision, protocol=protocol)
    _write_json(result_root / "walk_decision.json", decision)
    (result_root / "PROBE_COMPLETE").touch()
    if decision["decision"] == "stop":
        (result_root / "WALK_STOPPED").touch()
    return campaign_status(result_root, protocol=protocol)


def run_walk(
    result_root: Path,
    *,
    protocol: dict[str, Any],
    flux_gate: dict[str, Any],
    parent_root: Path,
    interp_root: Path,
    workers: int,
) -> dict[str, Any]:
    decision_path = result_root / "walk_decision.json"
    if not decision_path.is_file():
        raise SystemExit(
            "run the probe stage first; the walk needs walk_decision.json"
        )
    decision = _read_json(decision_path)
    validate_walk_decision(decision, protocol=protocol)
    active = []
    for name, track in decision["tracks"].items():
        if not track["opened"]:
            continue
        if not track.get("seed_candidate_id"):
            raise ValueError(f"track {name} decision lacks its chain seed")
        active.append(
            (
                name,
                [float(value) for value in track["targets_K"]],
                str(track["seed_candidate_id"]),
                track.get("seed_product_sha256"),
            )
        )
    if not active:
        raise SystemExit("no track opened; the walk is closed")
    started = time.perf_counter()
    outputs = _run_workers(
        _track_walk_worker,
        [
            (
                name,
                targets,
                seed_candidate_id,
                seed_product_sha256,
                str(result_root),
                str(parent_root),
                str(interp_root),
                flux_gate,
                protocol["protocol_hash"],
            )
            for name, targets, seed_candidate_id, seed_product_sha256 in active
        ],
        workers=min(max(int(workers), 1), len(active)),
    )
    print(
        json.dumps(
            {
                "tracks": len(outputs),
                "new_eligible_in_walk": sum(
                    bool(row.get("training_eligible"))
                    for output in outputs
                    for row in output["outcomes"]
                ),
                "seconds": time.perf_counter() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    status = campaign_status(result_root, protocol=protocol)
    for name, track_status in status["walk"].items():
        if not track_status["opened"]:
            continue
        if track_status["complete"]:
            (result_root / f"TRACK_{name}_COMPLETE").touch()
        elif track_status["stopped_after_failure"]:
            (result_root / f"TRACK_{name}_STOPPED").touch()
    if all(
        track_status["complete"] or track_status["stopped_after_failure"]
        for track_status in status["walk"].values()
        if track_status["opened"]
    ):
        (result_root / "WALK_COMPLETE").touch()
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("protocol", "import-gate", "probe", "walk", "status"),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENT_ROOT)
    parser.add_argument("--interp-root", type=Path, default=DEFAULT_INTERP_ROOT)
    parser.add_argument("--workers", type=int, default=2)
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
            interp_root=args.interp_root,
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
            parent_root=args.parent_root,
        )
        print(json.dumps(gate, indent=2, sort_keys=True))
        return 0
    if not gate_path.is_file():
        raise SystemExit("run import-gate before the solver stages")
    flux_gate = _read_json(gate_path)
    if not flux_gate.get("frozen") or flux_gate.get("thresholds_refit"):
        raise SystemExit("FAIL_STOP: imported flux gate is not frozen")

    if args.stage == "probe":
        status = run_probe(
            args.result_root,
            protocol=protocol,
            flux_gate=flux_gate,
            parent_root=args.parent_root,
            interp_root=args.interp_root,
            workers=args.workers,
        )
    elif args.stage == "walk":
        status = run_walk(
            args.result_root,
            protocol=protocol,
            flux_gate=flux_gate,
            parent_root=args.parent_root,
            interp_root=args.interp_root,
            workers=args.workers,
        )
    else:
        status = campaign_status(args.result_root, protocol=protocol)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
