"""Wall probes: donor-jump walks at the two divergence boundaries.

The parity run showed the [M/H]-1.0 wall below 3800 K was a chain-state
artifact -- 3775 K converges when seeded from a deep-basin product. Two
walls remain: the [M/H]0 corridor at 3325 (diverged from the 3350 chain;
3300 K itself is certified) and the [M/H]-1.0 wall at 3675 (diverged
under three independent donors). This campaign walks both boundaries
with donor-jump seeding -- each step takes the previous converged
product, a diverging step retries with the other certified bracketing
product -- and certifies the far-side grid targets (3200 K on the rich
track, 3600 K on the metal-poor track) under the full pipeline standard.
A wall that holds under donor-jump seeding is real evidence, not a
chain artifact.
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
    _load_mt,
    _passes_flux_gate,
    _product_consistency,
    _annotate_record,
    _write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_wall_probe_v1"
STEP_K = 25.0
WALK_CAP = 60
CERTIFICATION_CAP = 120
PHASE_AWARE = {"require_improving_flux_residual": True}
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_GATE_PATH = (
    REPO_ROOT / "results" / "m_star_iteration_tomography_v1" / "flux_gate.json"
)
TOMOGRAPHY_ROOT = REPO_ROOT / "results" / "m_star_iteration_tomography_v1"
DONOR_WALK_ROOT = REPO_ROOT / "results" / "m_star_donor_walk_3600_v1"
SCALEOUT_ROOT = REPO_ROOT / "results" / "m_star_pipeline_scaleout_v1"

RICH = "g+4.50_m+0.00_a+0.00_c+0.00_x1.00"
POOR = "g+4.50_m-1.00_a+0.00_c+0.00_x1.00"

WALKS = {
    "rich_corridor": {
        "class": "dwarf",
        "log_surface_gravity": 4.5,
        "metallicity": 0.0,
        "target_temperature_K": 3200.0,
        "start_temperature_K": 3300.0,
        "probe_temperature_K": 3325.0,
        # donors: the certified 3300 K and 3400 K tomography products
        "donor_products": [
            TOMOGRAPHY_ROOT / "cases" / "dwarf" / RICH / "t3300" / "products" / "primary",
            TOMOGRAPHY_ROOT / "cases" / "dwarf" / RICH / "t3400" / "products" / "primary",
        ],
    },
    "poor_wall": {
        "class": "dwarf",
        "log_surface_gravity": 4.5,
        "metallicity": -1.0,
        "target_temperature_K": 3600.0,
        "start_temperature_K": 3800.0,
        "probe_temperature_K": 3675.0,
        # donors: the certified 3800 K scale-out product and the 3700 K
        # donor-walk waypoint product
        "donor_products": [
            SCALEOUT_ROOT
            / "cases"
            / "dwarf_g+4.50_m-1.00_t3800"
            / "cap60"
            / "products"
            / "primary",
            DONOR_WALK_ROOT
            / "cases"
            / "dwarf_g+4.50_m-1.00_t3700"
            / "products",
        ],
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _product_in(directory: Path) -> Path | None:
    products = sorted(directory.glob("*.npz"))
    return products[0] if products else None


def _walk(
    name: str,
    spec: dict[str, Any],
    *,
    result_root: Path,
    gate: dict[str, Any],
    protocol_hash: str,
) -> dict[str, Any]:
    track_payload = pipeline.track_payload(
        stellar_class=spec["class"],
        log_surface_gravity=spec["log_surface_gravity"],
        metallicity=spec["metallicity"],
        microturbulence_km_s=pipeline.MICROTURBULENCE[spec["class"]],
    )
    track = base._track_from_payload(track_payload)
    case_root = (
        result_root
        / "cases"
        / f"{spec['class']}_g{spec['log_surface_gravity']:+05.2f}"
        f"_m{spec['metallicity']:+05.2f}_t{int(spec['target_temperature_K']):04d}"
    )
    donor_dirs = [Path(p) for p in spec["donor_products"]]

    steps: list[dict[str, Any]] = []
    # The probe step first: an excursion above the start toward the wall.
    temperatures: list[float] = []
    probe = float(spec["probe_temperature_K"])
    start = float(spec["start_temperature_K"])
    target = float(spec["target_temperature_K"])
    if probe > start:
        temperatures.append(probe)
    t = start
    while t - target > 1.0e-9:
        t = max(target, t - STEP_K)
        temperatures.append(t)

    latest_by_temperature: dict[float, Path] = {}
    current_temperature = start
    primary: dict[str, Any] | None = None
    restart: dict[str, Any] | None = None
    reached = False
    stopped_reason: str | None = None
    started = time.perf_counter()

    for temperature in temperatures:
        is_target = abs(temperature - target) <= 1.0e-9
        labels = track.labels(temperature)
        record = None
        outcome: dict[str, Any] | None = None
        donor_used = None
        donor_candidates: list[Path] = []
        if latest_by_temperature:
            nearest = min(
                latest_by_temperature,
                key=lambda value: abs(value - temperature),
            )
            donor_candidates.append(latest_by_temperature[nearest])
        donor_candidates.extend(d for d in donor_dirs if d.is_dir())
        for donor_dir in donor_candidates:
            donor_product = _product_in(donor_dir)
            if donor_product is None:
                continue
            donor_name = f"{donor_dir.parent.parent.name}/{donor_dir.parent.name}"
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
                method="wall_probe_waypoint" if not is_target else "wall_probe_target",
                schedule="wall_probe",
                source_temperature=current_temperature,
                target_labels=labels,
                initial_atmosphere=seed_atmosphere,
                product_dir=(
                    case_root / "products" / "primary"
                    if is_target
                    else case_root / "products" / "continuation"
                ),
                iteration_cap=WALK_CAP if not is_target else CERTIFICATION_CAP,
                maximum_all_layer_relative_temperature_change=(
                    pipeline.STRICT_ALL_LAYER_LIMIT
                ),
                config_overrides=dict(PHASE_AWARE) if is_target else None,
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
                donor_used = donor_name
                latest_by_temperature[temperature] = Path(record["product_path"])
                if is_target:
                    primary = record
                    reached = True
                else:
                    current_temperature = temperature
                break
        steps.append(
            {
                "target_temperature_K": temperature,
                "is_target_node": is_target,
                **(outcome or {"status": "no_donor_product"}),
            }
        )
        if record is not None and record.get("survives_solver"):
            continue
        stopped_reason = f"diverged_at_{int(temperature)}K"
        break

    restart: dict[str, Any] | None = None
    if primary is not None and primary.get("survives_solver") and primary.get(
        "product_path"
    ):
        labels = track.labels(target)
        solved_m, solved_t = _load_mt(primary["product_path"])
        restart_seed = _reconstruct_from_mt(labels, solved_m, solved_t)
        restart, _restart_state = _solve_attempt(
            track=track,
            method="wall_probe_strict_self_restart",
            schedule="independent_self_restart",
            source_temperature=target,
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
            node_id=f"wall_probe_{name}",
        )
        primary = _annotate_record(
            primary,
            track_payload=track_payload,
            role="train",
            node_id=f"wall_probe_{name}",
        )

    primary_flux = (
        _passes_flux_gate(primary, gate)
        if primary
        else {"passes": False, "metrics": {}}
    )
    restart_flux = (
        _passes_flux_gate(restart, gate) if restart else {"passes": False, "metrics": {}}
    )
    consistency = (
        _product_consistency(
            primary.get("product_path"),
            None if restart is None else restart.get("product_path"),
        )
        if primary
        else {"available": False, "passes": False}
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
        reached
        and primary is not None
        and primary.get("survives_solver")
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
    if not reached:
        reasons.append(stopped_reason or "target_not_reached")
    else:
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

    output = {
        "campaign": CAMPAIGN,
        "protocol_hash": protocol_hash,
        "walk": name,
        "candidate_id": f"wall_probe_{name}_t{int(target)}",
        "class": spec["class"],
        "track": track_payload,
        "temperature_K": target,
        "steps": steps,
        "reached_target": reached,
        "primary": primary,
        "restart": restart,
        "primary_flux_gate": primary_flux,
        "restart_flux_gate": restart_flux,
        "path_consistency": consistency,
        "phase_guard": {"primary": guard_primary, "restart": guard_restart},
        "training_eligible": eligible,
        "status": "training_eligible" if eligible else "closed" if not reached else "ineligible",
        "failure_reason": (
            None
            if eligible
            else (",".join(reasons) if reached else (stopped_reason or "closed"))
        ),
        "seconds": float(time.perf_counter() - started),
    }
    _write_json(case_root / "case.json", output)
    print(
        f"{name}: reached={reached} eligible={eligible} "
        f"reason={output['failure_reason']} steps={len(steps)} "
        f"seconds={output['seconds']:.0f}"
    )
    for step in steps:
        p95 = step.get("p95_absolute_flux_error_percent")
        print(
            f"   {step.get('target_temperature_K')}K {step.get('status')} "
            f"donor={step.get('donor')} iters={step.get('iterations')} "
            f"p95={round(p95, 2) if p95 is not None else '—'}"
        )
    return output


def run_campaign(args: argparse.Namespace) -> int:
    result_root = Path(args.result_root)
    gate = _read_json(Path(args.gate_path))
    protocol = {
        "campaign": CAMPAIGN,
        "step_k": STEP_K,
        "walk_cap": WALK_CAP,
        "certification_cap": CERTIFICATION_CAP,
        "walks": {
            name: {
                "target": spec["target_temperature_K"],
                "probe": spec["probe_temperature_K"],
                "donor_products": [str(p) for p in spec["donor_products"]],
            }
            for name, spec in WALKS.items()
        },
        "flux_gate_source": {
            "campaign": gate.get("campaign"),
            "gate_hash": gate.get("gate_hash"),
        },
    }
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    _write_json(result_root / "protocol.json", protocol)
    print(f"protocol hash {protocol['protocol_hash']}")

    started = time.perf_counter()
    for name, spec in WALKS.items():
        _walk(
            name,
            spec,
            result_root=result_root,
            gate=gate,
            protocol_hash=protocol["protocol_hash"],
        )
    print(f"wall seconds {time.perf_counter() - started:.0f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--gate-path", default=str(DEFAULT_GATE_PATH))
    args = parser.parse_args(argv)
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
