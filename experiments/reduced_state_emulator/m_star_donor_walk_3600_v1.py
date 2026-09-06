"""Donor-jump walk to 3600 K on the [M/H]-1.0 track.

The parity run showed 3775 K converges in 27 iterations when seeded
from the certified 4000 K product directly -- the downwalk divergence
was a property of the accumulated chain state, not of the temperature.
This walk continues from the fresh 3775 K product: each 25 K step seeds
from the previous step's converged product, a diverging step retries
with the deeper-basin donors (3800 K certified, then 4000 K certified),
and the 3600 K target certifies under the full pipeline standard --
phase-aware primary and restart, frozen flux gate on both legs, path
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
CAMPAIGN = "m_star_donor_walk_3600_v1"
TRACK_SLUG = "g+4.50_m-1.00_a+0.00_c+0.00_x1.00"
TARGET_TEMPERATURE_K = 3600.0
STEP_K = 25.0
WALK_CAP = 60
CERTIFICATION_CAP = 120
PHASE_AWARE = {"require_improving_flux_residual": True}
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_GATE_PATH = (
    REPO_ROOT / "results" / "m_star_iteration_tomography_v1" / "flux_gate.json"
)
DEFAULT_PARITY_ROOT = REPO_ROOT / "results" / "m_star_parity_divergence_v1"
DEFAULT_SCALEOUT_ROOT = REPO_ROOT / "results" / "m_star_pipeline_scaleout_v1"
PREREGISTRATION_PATH = (
    REPO_ROOT / "notes" / "m_star_donor_walk_3600_v1_preregistration_20260906.md"
)
WALK_TEMPERATURES = [3750.0, 3725.0, 3700.0, 3675.0, 3650.0, 3625.0, 3600.0]
DONOR_FALLBACKS = (
    (
        DEFAULT_SCALEOUT_ROOT
        / "cases"
        / "dwarf_g+4.50_m-1.00_t3800"
        / "cap60"
        / "products"
        / "primary"
    ),
    DEFAULT_PARITY_ROOT / "cases" / "step25_to_3775" / "products",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _product_in(directory: Path) -> Path | None:
    products = sorted(directory.glob("*.npz"))
    return products[0] if products else None


def run_walk(args: argparse.Namespace) -> dict[str, Any]:
    _set_single_thread_environment()
    result_root = Path(args.result_root)
    gate = _read_json(Path(args.gate_path))
    parity_root = Path(args.parity_root)
    scaleout_root = Path(args.scaleout_root)

    seed_products = sorted(
        (parity_root / "cases" / "step25_to_3775" / "products").glob("*.npz")
    )
    if not seed_products:
        raise FileNotFoundError("3775 K parity product missing")
    seed_product = seed_products[0]

    track_payload = pipeline.track_payload(
        stellar_class="dwarf",
        log_surface_gravity=4.5,
        metallicity=-1.0,
        microturbulence_km_s=pipeline.MICROTURBULENCE["dwarf"],
    )
    track = base._track_from_payload(track_payload)

    steps: list[dict[str, Any]] = []
    latest_product: Path | None = seed_product
    started = time.perf_counter()
    current_temperature = 3775.0
    for temperature in WALK_TEMPERATURES:
        is_target = temperature == TARGET_TEMPERATURE_K
        labels = track.labels(temperature)
        outcome: dict[str, Any] | None = None
        donors_tried: list[str] = []
        record = None
        for donor_dir in (None,) + DONOR_FALLBACKS:
            if donor_dir is None:
                if latest_product is None:
                    continue
                donor_product, donor_name = latest_product, "previous_step"
            else:
                donor_product = _product_in(donor_dir)
                if donor_product is None:
                    continue
                donor_name = str(donor_dir.parent.parent.name)
            donors_tried.append(donor_name)
            seed_mass, seed_profile = _load_mt(donor_product)
            try:
                seed_atmosphere = _reconstruct_from_mt(labels, seed_mass, seed_profile)
            except Exception as exc:  # noqa: BLE001 - a failed step is an outcome
                outcome = {
                    "status": "initialization_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "donor": donor_name,
                }
                continue
            record, state = _solve_attempt(
                track=track,
                method="donor_jump_waypoint",
                schedule="donor_walk",
                source_temperature=current_temperature,
                target_labels=labels,
                initial_atmosphere=seed_atmosphere,
                product_dir=(
                    result_root
                    / "cases"
                    / f"dwarf_g+4.50_m-1.00_t{int(temperature):04d}"
                    / "products"
                ),
                iteration_cap=WALK_CAP,
                maximum_all_layer_relative_temperature_change=(
                    pipeline.STRICT_ALL_LAYER_LIMIT
                ),
            )
            outcome = {
                "status": (
                    "solver_pass" if record.get("survives_solver") else "solver_fail"
                ),
                "donor": donor_name,
                "donor_product": str(donor_product),
                "iterations": record.get("iterations"),
                "p95_absolute_flux_error_percent": (
                    (record.get("solver_diagnostics") or {})
                    .get("final_diagnostics", {})
                    .get("p95_absolute_flux_error_percent")
                ),
            }
            if record.get("survives_solver"):
                if not is_target:
                    latest_product = Path(record["product_path"])
                    current_temperature = temperature
                break
        steps.append(
            {
                "target_temperature_K": temperature,
                "is_target_node": is_target,
                **(outcome or {"status": "no_donor"}),
            }
        )
        if not (record is not None and record.get("survives_solver")):
            break
        current_temperature = temperature
        if is_target:
            break

    reached = bool(
        steps and steps[-1].get("is_target_node")
        and steps[-1].get("status") == "solver_pass"
    )
    primary_record = record if reached else None
    primary = None
    restart = None
    if reached and primary_record is not None and primary_record.get("product_path"):
        labels = track.labels(TARGET_TEMPERATURE_K)
        primary, _primary_state = _solve_attempt(
            track=track,
            method="donor_walk_certified_primary",
            schedule="donor_walk",
            source_temperature=None,
            target_labels=labels,
            initial_atmosphere=_reconstruct_from_mt(
                labels, *_load_mt(primary_record["product_path"])
            ),
            product_dir=(
                result_root
                / "cases"
                / f"dwarf_g+4.50_m-1.00_t{int(TARGET_TEMPERATURE_K):04d}"
                / "certification"
                / "products"
                / "primary"
            ),
            iteration_cap=CERTIFICATION_CAP,
            maximum_all_layer_relative_temperature_change=(
                pipeline.STRICT_ALL_LAYER_LIMIT
            ),
            config_overrides=dict(PHASE_AWARE),
        )
        primary = _annotate_record(
            primary,
            track_payload=track_payload,
            role="train",
            node_id=f"{TRACK_SLUG}_t{int(TARGET_TEMPERATURE_K)}",
        )
        if primary.get("survives_solver") and primary.get("product_path"):
            solved_m, solved_t = _load_mt(primary["product_path"])
            restart_seed = _reconstruct_from_mt(labels, solved_m, solved_t)
            restart, _restart_state = _solve_attempt(
                track=track,
                method="donor_walk_strict_self_restart",
                schedule="independent_self_restart",
                source_temperature=TARGET_TEMPERATURE_K,
                target_labels=labels,
                initial_atmosphere=restart_seed,
                product_dir=(
                    result_root
                    / "cases"
                    / f"dwarf_g+4.50_m-1.00_t{int(TARGET_TEMPERATURE_K):04d}"
                    / "certification"
                    / "products"
                    / "restart"
                ),
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

    if primary is not None:
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
        certification_block = {
            "primary": primary,
            "restart": restart,
            "primary_flux_gate": primary_flux,
            "restart_flux_gate": restart_flux,
            "path_consistency": consistency,
            "phase_guard": {"primary": guard_primary, "restart": guard_restart},
            "training_eligible": eligible,
            "failure_reason": None if eligible else ",".join(reasons),
        }
    else:
        certification_block = None
        eligible = False

    output = {
        "campaign": CAMPAIGN,
        "protocol_hash": hashlib.sha256(
            json.dumps(
                {
                    "campaign": CAMPAIGN,
                    "step_k": STEP_K,
                    "walk_cap": WALK_CAP,
                    "certification_cap": CERTIFICATION_CAP,
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
        "certification": certification_block,
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
                (certification_block or {}).get("failure_reason")
                if reached
                else "walk_stopped_before_target"
            )
        ),
        "seconds": float(time.perf_counter() - started),
    }
    _write_json(
        result_root
        / "cases"
        / f"dwarf_g+4.50_m-1.00_t{int(TARGET_TEMPERATURE_K):04d}"
        / "case.json",
        output,
    )
    print(
        f"donor walk: reached={reached} eligible={output['training_eligible']} "
        f"steps={len(steps)} seconds={output['seconds']:.0f}"
    )
    for step in steps:
        print(
            f"   {step.get('target_temperature_K')}K {step.get('status')} "
            f"donor={step.get('donor')} iters={step.get('iterations')} "
            f"p95={step.get('p95_absolute_flux_error_percent')}"
        )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--gate-path", default=str(DEFAULT_GATE_PATH))
    parser.add_argument("--parity-root", default=str(DEFAULT_PARITY_ROOT))
    parser.add_argument("--scaleout-root", default=str(DEFAULT_SCALEOUT_ROOT))
    args = parser.parse_args(argv)
    run_walk(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
