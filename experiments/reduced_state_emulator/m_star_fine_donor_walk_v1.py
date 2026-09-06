"""Fine-step donor-jump walk on the [M/H]-1.0 track: 3700 -> 3600 K.

The last untried in-bounds variant: the earlier donor walk used fixed
25 K steps from 3775; the parity verdict showed seed and jump size
decide basin membership, so this walk descends from the converged 3700 K
waypoint product in 12.5 K steps, halving to 6.25 K on failure. Each
step seeds from the previous converged product; a failing step retries
with the deeper-basin donors (3800 K and 4000 K certified products,
3775 K parity product) before the floor applies. Reaching 3600 K
triggers full pipeline certification: phase-aware primary, strict
phase-aware self-restart, frozen flux gate on both legs, path
consistency, and the phase guard on both legs.
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
from .cool_star_step_test import (
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from . import m_star_bootstrap_v1r2_marcs100 as base
from .m_star_bootstrap_v1 import (
    _annotate_record,
    _load_mt,
    _passes_flux_gate,
    _product_consistency,
    _sha256,
    _write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_fine_donor_walk_v1"
TRACK_SLUG = "g+4.50_m-1.00_a+0.00_c+0.00_x1.00"
TARGET_TEMPERATURE_K = 3600.0
START_TEMPERATURE_K = 3700.0
INITIAL_STEP_K = 12.5
MINIMUM_STEP_K = 6.25
CERTIFICATION_CAP = 120
TEMPERATURE_TOLERANCE_K = 1.0e-9
PHASE_AWARE = {"require_improving_flux_residual": True}
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_GATE_PATH = (
    REPO_ROOT / "results" / "m_star_iteration_tomography_v1" / "flux_gate.json"
)
DEFAULT_CHAIN_PRODUCT = (
    REPO_ROOT
    / "results"
    / "m_star_donor_walk_3600_v1"
    / "cases"
    / "dwarf_g+4.50_m-1.00_t3700"
    / "products"
)
DONOR_PRODUCT_DIRS = (
    REPO_ROOT
    / "results"
    / "m_star_pipeline_scaleout_v1"
    / "cases"
    / "dwarf_g+4.50_m-1.00_t3800"
    / "cap60"
    / "products"
    / "primary",
    REPO_ROOT
    / "results"
    / "m_star_pipeline_scaleout_v1"
    / "cases"
    / "dwarf_g+4.50_m-1.00_t4000"
    / "cap60"
    / "products"
    / "primary",
    REPO_ROOT
    / "results"
    / "m_star_parity_divergence_v1"
    / "cases"
    / "step25_to_3775"
    / "products",
)
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "notes"
    / "m_star_fine_donor_walk_v1_preregistration_20260907.md"
)


def _product_in(directory: Path) -> Path | None:
    products = sorted(directory.glob("*.npz"))
    return products[0] if products else None


def run_walk(args: argparse.Namespace) -> dict[str, Any]:
    _set_single_thread_environment()
    result_root = Path(args.result_root)
    gate = _read_json(Path(args.gate_path))
    chain_products = sorted(Path(str(DEFAULT_CHAIN_PRODUCT)).glob("*.npz"))
    if not chain_products:
        raise FileNotFoundError(f"no chain product in {DEFAULT_CHAIN_PRODUCT}")
    chain_product = chain_products[0]

    donor_pool: list[tuple[str, Path]] = [("chain_3700", chain_product)]
    for directory in DONOR_PRODUCT_DIRS:
        product = _product_in(directory)
        if product is not None:
            donor_pool.append((str(directory.parent.parent.name), product))

    track_payload = pipeline.track_payload(
        stellar_class="dwarf",
        log_surface_gravity=4.5,
        metallicity=-1.0,
        microturbulence_km_s=pipeline.MICROTURBULENCE["dwarf"],
    )
    track = base._track_from_payload(track_payload)
    case_root = (
        result_root
        / "cases"
        / f"dwarf_g+4.50_m-1.00_t{int(TARGET_TEMPERATURE_K):04d}"
    )

    steps: list[dict[str, Any]] = []
    current_temperature = float(START_TEMPERATURE_K)
    current_product: Path | None = chain_product
    step_size = float(INITIAL_STEP_K)
    primary: dict[str, Any] | None = None
    reached = False
    stopped_reason: str | None = None
    started = time.perf_counter()

    while abs(current_temperature - TARGET_TEMPERATURE_K) > TEMPERATURE_TOLERANCE_K:
        proposed = max(TARGET_TEMPERATURE_K, current_temperature - step_size)
        is_target = abs(proposed - TARGET_TEMPERATURE_K) <= TEMPERATURE_TOLERANCE_K
        labels = track.labels(proposed)
        record = None
        outcome: dict[str, Any] | None = None
        state = None
        for donor_name, donor_product in (
            [(f"chain_{int(current_temperature)}", current_product)]
            + [(name, product) for name, product in donor_pool]
        ):
            if donor_product is None or not donor_product.is_file():
                continue
            seed_mass, seed_profile = _load_mt(donor_product)
            try:
                seed_atmosphere = _reconstruct_from_mt(
                    labels, seed_mass, seed_profile
                )
            except Exception as exc:  # noqa: BLE001 - a failed step is an outcome
                outcome = {
                    "status": "initialization_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "donor": donor_name,
                }
                continue
            record, state = _solve_attempt(
                track=track,
                method="fine_donor_waypoint" if not is_target else "fine_donor_target",
                schedule="fine_donor_walk",
                source_temperature=current_temperature,
                target_labels=labels,
                initial_atmosphere=seed_atmosphere,
                product_dir=(
                    case_root / "products" / "primary"
                    if is_target
                    else case_root / "products" / "continuation"
                ),
                iteration_cap=120,
                maximum_all_layer_relative_temperature_change=(
                    pipeline.STRICT_ALL_LAYER_LIMIT
                ),
            )
            outcome = {
                "status": (
                    "solver_pass" if record.get("survives_solver") else "solver_fail"
                ),
                "donor": donor_name,
                "iterations": record.get("iterations"),
                "p95_absolute_flux_error_percent": (
                    (record.get("solver_diagnostics") or {})
                    .get("final_diagnostics", {})
                    .get("p95_absolute_flux_error_percent")
                ),
            }
            if record.get("survives_solver"):
                break
        steps.append(
            {
                "target_temperature_K": proposed,
                "requested_step_K": step_size,
                "is_target_node": is_target,
                **(outcome or {"status": "no_donor_product"}),
            }
        )
        if record is not None and record.get("survives_solver"):
            current_temperature = proposed
            current_product = Path(record["product_path"])
            if is_target:
                primary = record
                reached = True
                break
            continue
        if step_size > MINIMUM_STEP_K:
            step_size = max(step_size / 2.0, MINIMUM_STEP_K)
            continue
        stopped_reason = "minimum_step_failed"
        break

    restart: dict[str, Any] | None = None
    certification: dict[str, Any] | None = None
    eligible = False
    if reached and primary is not None and primary.get("product_path"):
        labels = track.labels(TARGET_TEMPERATURE_K)
        solved_m, solved_t = _load_mt(primary["product_path"])
        restart_seed = _reconstruct_from_mt(labels, solved_m, solved_t)
        restart, _restart_state = _solve_attempt(
            track=track,
            method="fine_donor_strict_self_restart",
            schedule="independent_self_restart",
            source_temperature=TARGET_TEMPERATURE_K,
            target_labels=labels,
            initial_atmosphere=restart_seed,
            product_dir=case_root / "products" / "restart",
            iteration_cap=CERTIFICATION_CAP,
            maximum_all_layer_relative_temperature_change=(
                pipeline.STRICT_ALL_LAYER_LIMIT
            ),
            config_overrides=dict(PHASE_AWARE),
        )
        restart = _annotate_record(
            restart,
            track_payload=track_payload,
            role="train",
            node_id=f"{TRACK_SLUG}_t{int(TARGET_TEMPERATURE_K)}",
        )
        primary = _annotate_record(
            primary,
            track_payload=track_payload,
            role="train",
            node_id=f"{TRACK_SLUG}_t{int(TARGET_TEMPERATURE_K)}",
        )
        primary_flux = _passes_flux_gate(primary, gate)
        restart_flux = (
            _passes_flux_gate(restart, gate)
            if restart
            else {"passes": False, "metrics": {}}
        )
        consistency = _product_consistency(
            primary.get("product_path"),
            None if restart is None else restart.get("product_path"),
        )

        def _guard(rec: dict[str, Any] | None) -> bool | None:
            if not rec:
                return None
            value = (
                (rec.get("solver_diagnostics") or {})
                .get("final_diagnostics", {})
                .get("flux_residual_improving_at_stop")
            )
            return None if value is None else bool(value)

        guard_primary, guard_restart = _guard(primary), _guard(restart)
        eligible = bool(
            primary.get("survives_solver")
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
        certification = {
            "primary": primary,
            "restart": restart,
            "primary_flux_gate": primary_flux,
            "restart_flux_gate": restart_flux,
            "path_consistency": consistency,
            "phase_guard": {"primary": guard_primary, "restart": guard_restart},
            "training_eligible": eligible,
            "failure_reason": None if eligible else ",".join(reasons),
        }

    output = {
        "campaign": CAMPAIGN,
        "protocol_hash": hashlib.sha256(
            json.dumps(
                {
                    "campaign": CAMPAIGN,
                    "initial_step_K": INITIAL_STEP_K,
                    "minimum_step_K": MINIMUM_STEP_K,
                    "chain_seed_sha256": _sha256(chain_product),
                    "gate_hash": gate.get("gate_hash"),
                },
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "candidate_id": f"{TRACK_SLUG}_t{int(TARGET_TEMPERATURE_K)}",
        "class": "dwarf",
        "track": track_payload,
        "temperature_K": TARGET_TEMPERATURE_K,
        "steps": steps,
        "reached_target": reached,
        "certification": certification,
        "training_eligible": bool(reached and eligible),
        "status": (
            "training_eligible"
            if reached and eligible
            else ("closed" if not reached else "ineligible")
        ),
        "failure_reason": (
            None
            if reached and eligible
            else (
                (certification or {}).get("failure_reason")
                if reached
                else (stopped_reason or "walk_stopped")
            )
        ),
        "seconds": float(time.perf_counter() - started),
    }
    _write_json(case_root / "case.json", output)
    print(
        f"fine donor walk: reached={reached} eligible={eligible} "
        f"steps={len(steps)} seconds={output['seconds']:.0f}"
    )
    for step in steps:
        print(
            f"   ->{step.get('target_temperature_K')}K {step.get('status')} "
            f"step={step.get('requested_step_K')} donor={step.get('donor')} "
            f"iters={step.get('iterations')} "
            f"p95={step.get('p95_absolute_flux_error_percent')}"
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--gate-path", default=str(DEFAULT_GATE_PATH))
    args = parser.parse_args(argv)
    run_walk(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
