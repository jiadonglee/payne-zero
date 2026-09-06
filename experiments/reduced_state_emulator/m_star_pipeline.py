"""The cold-star solving pipeline, assembled from its certified pieces.

One entry point, ``certified_solve``, runs the exact production solver on
a seed, re-solves it from its terminal ``(m,T)`` as a strict self-restart,
and admits the node only when both legs survive, pass the frozen flux
gate, agree on the fixed point, and stop on a non-worsening p95 flux
error (the certification phase guard). Seeds come from a native MARCS
node (``marcs_seed``) or from any prior converged product
(``continuation_seed``); both carry ``(m, T)`` only and let the
reconstruction rebuild the other four fields.

Everything here composes the pieces that the M-star campaigns
(v1r2, interpolated seeds, tomography, fine-step) already certified
individually; no solver behaviour is configurable or changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .cool_star_step_test import (
    _reconstruct_from_mt,
    _solve_attempt,
)
from . import m_star_bootstrap_v1r2_marcs100 as base
from .marcs_h5 import inspect_marcs_grid, load_marcs_node
from .m_star_bootstrap_v1 import (
    _annotate_record,
    _load_mt,
    _passes_flux_gate,
    _product_consistency,
    _sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
ITERATION_CAP = base.ITERATION_CAP
STRICT_ALL_LAYER_LIMIT = base.STRICT_ALL_LAYER_LIMIT
MICROTURBULENCE = dict(base.MICROTURBULENCE)

_SCHEMA = None


def load_frozen_gate(path: Path) -> dict[str, Any]:
    """Load a frozen flux-gate payload (thresholds + provenance)."""

    return json.loads(Path(path).read_text())


def track_payload(
    *,
    stellar_class: str,
    log_surface_gravity: float,
    metallicity: float,
    microturbulence_km_s: float,
    alpha_enhancement: float = 0.0,
    carbon_enhancement: float = 0.0,
) -> dict[str, Any]:
    """Build the track payload the record annotators expect."""

    from .cool_star_step_test import TrackSpec

    track = TrackSpec(
        log_surface_gravity=float(log_surface_gravity),
        metallicity=float(metallicity),
        alpha_enhancement=float(alpha_enhancement),
        carbon_enhancement=float(carbon_enhancement),
        microturbulence_km_s=float(microturbulence_km_s),
    )
    return {
        **track.as_json(),
        "class": stellar_class,
        "role": "train",
    }


def labels_for(track: dict[str, Any], temperature_k: float):
    return base._track_from_payload(track).labels(float(temperature_k))


def marcs_seed(labels, *, carbon_enhancement: float = 0.0, marcs_grid: Path | None = None):
    """Native same-node MARCS ``(m, T)`` rebuilt into a full atmosphere."""

    global _SCHEMA
    grid = Path(marcs_grid or DEFAULT_MARCS_GRID)
    if _SCHEMA is None:
        _SCHEMA = inspect_marcs_grid(grid, verify_sha256=False, expected_sha256=None)
    node = load_marcs_node(
        grid,
        labels,
        carbon_enhancement=float(carbon_enhancement),
        verify_sha256=False,
        expected_sha256=None,
        schema=_SCHEMA,
        depth_coordinate="log_mass",
    )
    return _reconstruct_from_mt(labels, node.reduced_column_mass, node.reduced_temperature)


def continuation_seed(product_path: Path, labels):
    """A prior converged product's ``(m, T)`` rebuilt into an atmosphere."""

    seed_mass, seed_profile = _load_mt(Path(product_path))
    return _reconstruct_from_mt(labels, seed_mass, seed_profile)


def _phase_guard(record: dict[str, Any] | None) -> bool | None:
    if not record:
        return None
    value = (
        (record.get("solver_diagnostics") or {})
        .get("final_diagnostics", {})
        .get("flux_residual_improving_at_stop")
    )
    return None if value is None else bool(value)


def certified_solve(
    *,
    labels,
    track: dict[str, Any],
    candidate_id: str,
    initial_atmosphere,
    product_dir: Path,
    flux_gate: dict[str, Any],
    iteration_cap: int = ITERATION_CAP,
    require_phase_guard: bool = True,
    solver_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Primary plus strict self-restart under the full admission set.

    Certification: both legs survive the solver with a valid finite state,
    both pass the frozen flux gate, the two products agree (path
    consistency), and -- when ``require_phase_guard`` is on -- both legs
    stopped on a non-worsening p95 flux error. ``solver_overrides`` carries
    certification-strategy solver fields (e.g.
    ``require_improving_flux_residual``) to both legs; physics is never
    overridden here.
    """

    overrides = dict(solver_overrides or {})

    product_dir = Path(product_dir)
    primary, _primary_state = _solve_attempt(
        track=_track(track),
        method="pipeline_strict_primary",
        schedule="independent_pipeline_target",
        source_temperature=None,
        target_labels=labels,
        initial_atmosphere=initial_atmosphere,
        product_dir=product_dir / "products" / "primary",
        iteration_cap=int(iteration_cap),
        maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
        config_overrides=overrides or None,
    )
    primary = _annotate_record(
        primary,
        track_payload=track,
        role="train",
        node_id=str(candidate_id),
    )
    restart: dict[str, Any] | None = None
    if primary.get("survives_solver") and primary.get("product_path"):
        solved_m, solved_t = _load_mt(primary["product_path"])
        restart_seed = _reconstruct_from_mt(labels, solved_m, solved_t)
        restart, _restart_state = _solve_attempt(
            track=_track(track),
            method="pipeline_strict_self_restart",
            schedule="independent_self_restart",
            source_temperature=float(labels.effective_temperature),
            target_labels=labels,
            initial_atmosphere=restart_seed,
            product_dir=product_dir / "products" / "restart",
            iteration_cap=int(iteration_cap),
            maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
            config_overrides=overrides or None,
        )
        restart = _annotate_record(
            restart,
            track_payload=track,
            role="train",
            node_id=str(candidate_id),
        )

    primary_flux = _passes_flux_gate(primary, flux_gate)
    restart_flux = (
        _passes_flux_gate(restart, flux_gate)
        if restart
        else {"passes": False, "metrics": {}}
    )
    consistency = _product_consistency(
        primary.get("product_path"),
        None if restart is None else restart.get("product_path"),
    )
    guard_primary = _phase_guard(primary)
    guard_restart = _phase_guard(restart)
    eligible = bool(
        primary.get("survives_solver")
        and restart is not None
        and restart.get("survives_solver")
        and primary.get("state_quality", {}).get("valid")
        and restart.get("state_quality", {}).get("valid")
        and primary_flux["passes"]
        and restart_flux["passes"]
        and consistency["passes"]
        and (guard_primary is True if require_phase_guard else True)
        and (guard_restart is True if require_phase_guard else True)
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
    if require_phase_guard and guard_primary is not True:
        reasons.append("primary_phase_guard")
    if require_phase_guard and guard_restart is not True:
        reasons.append("restart_phase_guard")

    return {
        "candidate_id": str(candidate_id),
        "labels": labels.as_kwargs(),
        "solver_overrides": overrides,
        "primary": primary,
        "restart": restart,
        "primary_flux_gate": primary_flux,
        "restart_flux_gate": restart_flux,
        "path_consistency": consistency,
        "phase_guard": {"primary": guard_primary, "restart": guard_restart},
        "iteration_cap": int(iteration_cap),
        "training_eligible": eligible,
        "status": "training_eligible" if eligible else "ineligible",
        "failure_reason": None if eligible else ",".join(reasons),
    }


def _track(track: dict[str, Any]):
    return base._track_from_payload(track)
