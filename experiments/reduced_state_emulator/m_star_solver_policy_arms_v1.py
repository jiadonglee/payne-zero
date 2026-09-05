"""Solver-policy arms on the M-star tomography cases and the 3200 K node.

Three opt-in arms against the unchanged production reference (S0, the
existing tomography results): fixed global damping (S1), the
residual-guided step scale (S2), and the step scale plus the
improving-residual companion to the stopping rule (S2S). Physics, the
iteration cap, the strict all-layer limit, the frozen flux gate, the
same-arm strict self-restart, and the path-consistency test are
identical across arms; only the two experimental config fields move.

Cases A-D reuse the tomography campaign's frozen seeds; case E continues
from the gated tomography 3400 K product to 3200 K, the node where the
preregistered walk diverged under production stepping.
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
    _write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_solver_policy_arms_v1"
ITERATION_CAP = base.ITERATION_CAP
STRICT_ALL_LAYER_LIMIT = base.STRICT_ALL_LAYER_LIMIT
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_TOMOGRAPHY_ROOT = REPO_ROOT / "results" / "m_star_iteration_tomography_v1"
DEFAULT_INTERP_ROOT = tomography.DEFAULT_INTERP_ROOT
DEFAULT_V1R2_ROOT = tomography.DEFAULT_V1R2_ROOT
PREREGISTRATION_PATH = (
    REPO_ROOT / "notes" / "m_star_solver_policy_arms_v1_preregistration_20260905.md"
)

ARMS: dict[str, dict[str, Any]] = {
    "S1": {"temperature_correction_damping": 0.5},
    "S2": {"flux_residual_guided_damping": True},
    "S2S": {
        "flux_residual_guided_damping": True,
        "require_improving_flux_residual": True,
    },
}

CONTINUATION_CASE = {
    "candidate_id": "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3200",
    "class": "dwarf",
    "track_slug": "g+4.50_m+0.00_a+0.00_c+0.00_x1.00",
    "temperature_K": 3200.0,
    "seed_source_candidate_id": "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3400",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _arm_case_dir(result_root: Path, track_slug: str, teff: float, arm: str) -> Path:
    return (
        result_root
        / "cases"
        / "dwarf"
        / track_slug
        / f"t{int(teff):04d}"
        / arm
    )


def _candidates() -> list[dict[str, Any]]:
    candidates = []
    for track_slug, teff in tomography.CASES:
        candidates.append(
            {
                "candidate_id": f"{track_slug}_t{int(teff)}",
                "class": "dwarf",
                "track_slug": track_slug,
                "temperature_K": float(teff),
                "seed_source": "tomography_seed",
            }
        )
    candidates.append(dict(CONTINUATION_CASE))
    return candidates


def _case_seed(
    candidate: dict[str, Any],
    *,
    interp_root: Path,
    v1r2_root: Path,
    tomography_root: Path,
):
    """Rebuild the case seed: the frozen tomography seed, or the gated
    3400 K product for the continuation node."""

    if candidate["seed_source"] == "tomography_seed":
        return tomography.build_seed(
            interp_root=interp_root,
            v1r2_root=v1r2_root,
            track_slug=candidate["track_slug"],
            teff=float(candidate["temperature_K"]),
        )
    product = tomography._history_product(
        tomography_root,
        candidate["track_slug"],
        float(candidate["seed_source_candidate_id"].rsplit("_t", 1)[1]),
    )
    if product is None:
        raise FileNotFoundError(
            f"tomography product for {candidate['seed_source_candidate_id']} missing"
        )
    seed_mass, seed_profile = _load_mt(product)
    track = base._track_from_payload(
        _read_json(
            tomography._case_dir(
                tomography_root,
                candidate["track_slug"],
                float(candidate["seed_source_candidate_id"].rsplit("_t", 1)[1]),
            )
            / "case.json"
        )["seed"]["track"]
    )
    labels = track.labels(float(candidate["temperature_K"]))
    seed = _reconstruct_from_mt(labels, seed_mass, seed_profile)
    provenance = {
        "track": dict(_read_json(
            tomography._case_dir(
                tomography_root,
                candidate["track_slug"],
                float(candidate["seed_source_candidate_id"].rsplit("_t", 1)[1]),
            )
            / "case.json"
        )["seed"]["track"]),
        "target_temperature_K": float(candidate["temperature_K"]),
        "labels": labels.as_kwargs(),
        "seed_source_candidate_id": candidate["seed_source_candidate_id"],
        "seed_source_product": str(product),
    }
    return seed, provenance


def _case_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        candidate,
        result_root_text,
        tomography_root_text,
        interp_root_text,
        v1r2_root_text,
        flux_gate,
        protocol_hash,
    ) = payload
    _set_single_thread_environment()
    result_root = Path(result_root_text)
    try:
        seed, provenance = _case_seed(
            candidate,
            interp_root=Path(interp_root_text),
            v1r2_root=Path(v1r2_root_text),
            tomography_root=Path(tomography_root_text),
        )
        track = base._track_from_payload(provenance["track"])
        labels = track.labels(float(candidate["temperature_K"]))

        arms: dict[str, Any] = {}
        for arm, overrides in ARMS.items():
            case_root = _arm_case_dir(
                result_root, candidate["track_slug"], candidate["temperature_K"], arm
            )
            primary, _primary_state = _solve_attempt(
                track=track,
                method=f"{arm}_strict_primary",
                schedule="independent_policy_arm",
                source_temperature=None,
                target_labels=labels,
                initial_atmosphere=seed,
                product_dir=case_root / "products" / "primary",
                iteration_cap=ITERATION_CAP,
                maximum_all_layer_relative_temperature_change=(
                    STRICT_ALL_LAYER_LIMIT
                ),
                config_overrides=dict(overrides),
                after_iteration_hook=tomography.make_tomography_hook(
                    case_root / "iterations" / "primary"
                ),
            )
            primary = _annotate_record(
                primary,
                track_payload=provenance["track"],
                role="train",
                node_id=str(candidate["candidate_id"]),
            )
            restart: dict[str, Any] | None = None
            if primary.get("survives_solver") and primary.get("product_path"):
                solved_m, solved_t = _load_mt(primary["product_path"])
                restart_seed = _reconstruct_from_mt(labels, solved_m, solved_t)
                restart, _restart_state = _solve_attempt(
                    track=track,
                    method=f"{arm}_strict_self_restart",
                    schedule="independent_self_restart",
                    source_temperature=float(candidate["temperature_K"]),
                    target_labels=labels,
                    initial_atmosphere=restart_seed,
                    product_dir=case_root / "products" / "restart",
                    iteration_cap=ITERATION_CAP,
                    maximum_all_layer_relative_temperature_change=(
                        STRICT_ALL_LAYER_LIMIT
                    ),
                    config_overrides=dict(overrides),
                )
                restart = _annotate_record(
                    restart,
                    track_payload=provenance["track"],
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
            arms[arm] = {
                "overrides": dict(overrides),
                "primary": primary,
                "restart": restart,
                "primary_flux_gate": primary_flux,
                "restart_flux_gate": restart_flux,
                "path_consistency": consistency,
                "training_eligible": eligible,
                "status": "training_eligible" if eligible else "ineligible",
                "failure_reason": None if eligible else ",".join(reasons),
            }

        output = {
            "campaign": CAMPAIGN,
            "protocol_hash": protocol_hash,
            **candidate,
            "labels": labels.as_kwargs(),
            "arms": arms,
            "status": "complete",
        }
    except Exception as exc:  # noqa: BLE001 - a failed case is an outcome
        output = {
            "campaign": CAMPAIGN,
            "protocol_hash": protocol_hash,
            **candidate,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    case_path = result_root / "cases" / "dwarf" / f"{candidate['candidate_id']}_arms.json"
    _write_json(case_path, output)
    return output


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    result_root = Path(args.result_root)
    gate = _read_json(Path(args.tomography_root) / "flux_gate.json")
    protocol = {
        "campaign": CAMPAIGN,
        "preregistration": str(PREREGISTRATION_PATH),
        "arms": ARMS,
        "cases": [c["candidate_id"] for c in _candidates()],
        "solver_policy": {
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
    only = args.only
    payloads = []
    for candidate in _candidates():
        if only and only not in candidate["candidate_id"]:
            continue
        payloads.append(
            (
                candidate,
                str(result_root),
                str(Path(args.tomography_root)),
                str(Path(args.interp_root)),
                str(Path(args.v1r2_root)),
                gate,
                protocol["protocol_hash"],
            )
        )
    started = time.perf_counter()
    records = _run_workers(_case_worker, payloads, workers=int(args.workers))
    for record in sorted(records, key=lambda row: row["candidate_id"]):
        for arm, block in sorted((record.get("arms") or {}).items()):
            primary = block.get("primary") or {}
            print(
                "{cid} {arm}: eligible={elig} iters={iters} "
                "p95={p95} reason={reason}".format(
                    cid=record["candidate_id"],
                    arm=arm,
                    elig=block.get("training_eligible"),
                    iters=primary.get("iterations"),
                    p95=(
                        (block.get("primary_flux_gate") or {})
                        .get("metrics", {})
                        .get("p95_absolute_flux_error_percent", {})
                        .get("value")
                    ),
                    reason=block.get("failure_reason"),
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
    common.add_argument("--workers", type=int, default=4)
    common.add_argument(
        "--only",
        default=None,
        help="run only the cases whose candidate_id contains this substring",
    )
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
