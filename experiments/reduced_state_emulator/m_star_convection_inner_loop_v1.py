"""COOLTLUSTY-style convective inner-loop arm on the tomography dwarf cases.

The production default is unchanged (``convection_zone_inner_loop_passes=0``).
This campaign only moves solver order: eight inner convective passes between
global radiation iterations, same EOS, opacity, and mixing length.

    python -m experiments.reduced_state_emulator.m_star_convection_inner_loop_v1
    python -m experiments.reduced_state_emulator.m_star_convection_inner_loop_v1 --run-solves

Without ``--run-solves`` the script writes the preregistered protocol and
exits.  It does not launch atmosphere solves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .experiment_identity import config_fingerprint, git_commit, write_arm_status

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_convection_inner_loop_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_TOMOGRAPHY_ROOT = REPO_ROOT / "results" / "m_star_iteration_tomography_v1"
PREREGISTRATION_PATH = (
    REPO_ROOT / "notes" / "m_star_convection_inner_loop_v1_preregistration_20260918.md"
)
# Keep equal to ``convection_inner_loop.DEFAULT_INNER_PASSES``.
INNER_PASSES = 8
TOMOGRAPHY_CASES = (
    ("g+4.50_m+0.00_a+0.00_c+0.00_x1.00", 3500.0),
    ("g+4.50_m+0.00_a+0.00_c+0.00_x1.00", 3400.0),
    ("g+4.50_m+0.00_a+0.00_c+0.00_x1.00", 3300.0),
    ("g+4.50_m-0.50_a+0.00_c+0.00_x1.00", 3600.0),
)
CONTINUATION_CANDIDATE_ID = "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3200"
ITERATION_CAP = 60
STRICT_ALL_LAYER_LIMIT = 5.0e-4

ARM = {
    "S3": {
        "convection_zone_inner_loop_passes": INNER_PASSES,
        "convection_zone_inner_loop_freeze_mask": True,
    }
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _flux_gate_payload(gate: dict[str, Any]) -> dict[str, Any]:
    """``_passes_flux_gate`` expects the full gate object, not the inner map."""

    if "thresholds" in gate:
        return gate
    return {"thresholds": gate}


def _primary_from_artifacts(
    case_root: Path,
    *,
    candidate: dict[str, Any],
    labels,
) -> dict[str, Any] | None:
    """Reuse a crashed primary solve instead of repeating the atmosphere loop."""

    import numpy as np

    iteration_files = sorted(
        (Path(case_root) / "iterations" / "primary").glob("iter_*.npz")
    )
    product_files = sorted(
        (Path(case_root) / "products" / "primary").glob("*.npz")
    )
    if not iteration_files and not product_files:
        return None
    diagnostics: dict[str, float] = {}
    iterations = None
    if iteration_files:
        with np.load(iteration_files[-1], allow_pickle=False) as data:
            iterations = int(data["iteration"])
            diagnostics = {
                "all_layer_relative_temperature_change": float(
                    data["timing_all_layer_relative_temperature_change"]
                ),
                "deep_layer_relative_temperature_change": float(
                    data["timing_deep_layer_relative_temperature_change"]
                ),
                "maximum_absolute_flux_error_percent": float(
                    data["timing_maximum_absolute_flux_error_percent"]
                ),
                "median_absolute_flux_error_percent": float(
                    data["timing_median_absolute_flux_error_percent"]
                ),
                "p95_absolute_flux_error_percent": float(
                    data["timing_p95_absolute_flux_error_percent"]
                ),
            }
    product_path = str(product_files[0]) if product_files else None
    delta_t = diagnostics.get("all_layer_relative_temperature_change")
    solver_converged = bool(
        product_path is not None
        and delta_t is not None
        and float(delta_t) <= float(STRICT_ALL_LAYER_LIMIT)
    )
    return {
        "record_type": "cool_star_step",
        "method": "S3_strict_primary",
        "schedule": "convection_inner_loop",
        "labels": labels.as_kwargs(),
        "source_temperature": None,
        "target_temperature": float(candidate["temperature_K"]),
        "converged": solver_converged,
        "solver_converged": solver_converged,
        "iterations": iterations,
        "product_path": product_path,
        "product_written": bool(product_path),
        "survives_solver": bool(solver_converged and product_path),
        "spectral_pass": None,
        "survives": False,
        "primary_pass": False,
        "recovered_pass": False,
        "status": "solver_pass" if solver_converged else "solver_fail",
        "reused_existing_artifacts": True,
        "solver_diagnostics": {"final_diagnostics": diagnostics},
        "error": None
        if solver_converged
        else "solver did not satisfy its formal convergence criterion",
    }


def _protocol() -> dict[str, Any]:
    return {
        "campaign": CAMPAIGN,
        "preregistration": str(PREREGISTRATION_PATH),
        "arm": ARM,
        "cases": [f"{track}_t{int(teff)}" for track, teff in TOMOGRAPHY_CASES]
        + [CONTINUATION_CANDIDATE_ID],
        "solver_policy": {
            "iteration_cap": int(ITERATION_CAP),
            "maximum_all_layer_relative_temperature_change": float(
                STRICT_ALL_LAYER_LIMIT
            ),
            "mixing_length": 1.25,
            "eos_and_opacity": "production",
            "required_flux_is_target_only": True,
            "mask_released_after_inner_loop": True,
            "not_a_test_that_on_off_chatter_is_the_root_cause": True,
        },
        "reference": "tomography production S0, not re-run here",
        "git_commit": git_commit(REPO_ROOT),
    }


def _run_solves(args: argparse.Namespace, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    from . import m_star_iteration_tomography_v1 as tomography
    from . import m_star_solver_policy_arms_v2 as policy
    from .cool_star_step_test import (
        _reconstruct_from_mt,
        _set_single_thread_environment,
        _solve_attempt,
    )
    from .m_star_bootstrap_v1 import (
        _annotate_record,
        _load_mt,
        _passes_flux_gate,
        _product_consistency,
    )
    from . import m_star_bootstrap_v1r2_marcs100 as base

    _set_single_thread_environment()
    gate = json.loads((Path(args.tomography_root) / "flux_gate.json").read_text())
    flux_gate = _flux_gate_payload(gate)
    outputs: list[dict[str, Any]] = []
    candidates = []
    for track_slug, teff in tomography.CASES:
        candidates.append(
            {
                "candidate_id": f"{track_slug}_t{int(teff)}",
                "track_slug": track_slug,
                "temperature_K": float(teff),
                "seed_source": "tomography_seed",
            }
        )
    candidates.append(dict(policy.CONTINUATION_CASE))
    only = list(getattr(args, "only", None) or [])
    if only:
        candidates = [
            candidate
            for candidate in candidates
            if any(token in str(candidate["candidate_id"]) for token in only)
        ]
        if not candidates:
            raise ValueError(f"no candidates matched --only {only}")
    for candidate in candidates:
        print(
            f"[S3] start {candidate['candidate_id']} seed={candidate['seed_source']}",
            flush=True,
        )
        seed, provenance = policy._case_seed(
            candidate,
            interp_root=Path(args.interp_root),
            v1r2_root=Path(args.v1r2_root),
            tomography_root=Path(args.tomography_root),
        )
        track = base._track_from_payload(provenance["track"])
        labels = track.labels(float(candidate["temperature_K"]))
        case_root = (
            Path(args.result_root)
            / "cases"
            / "dwarf"
            / candidate["track_slug"]
            / f"t{int(candidate['temperature_K']):04d}"
            / "S3"
        )
        existing_case = case_root / "case.json"
        if existing_case.is_file():
            print(
                f"[S3] skip completed {candidate['candidate_id']}",
                flush=True,
            )
            outputs.append(json.loads(existing_case.read_text()))
            continue
        reused = _primary_from_artifacts(
            case_root, candidate=candidate, labels=labels
        )
        if reused is not None:
            print(
                f"[S3] reuse primary {candidate['candidate_id']} "
                f"iters={reused.get('iterations')} "
                f"survives={reused.get('survives_solver')}",
                flush=True,
            )
            primary = reused
        else:
            primary, _state = _solve_attempt(
                track=track,
                method="S3_strict_primary",
                schedule="convection_inner_loop",
                source_temperature=None,
                target_labels=labels,
                initial_atmosphere=seed,
                product_dir=case_root / "products" / "primary",
                iteration_cap=base.ITERATION_CAP,
                maximum_all_layer_relative_temperature_change=base.STRICT_ALL_LAYER_LIMIT,
                config_overrides=dict(ARM["S3"]),
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
        restart = None
        if primary.get("survives_solver") and primary.get("product_path"):
            solved_m, solved_t = _load_mt(primary["product_path"])
            restart_seed = _reconstruct_from_mt(labels, solved_m, solved_t)
            restart, _restart_state = _solve_attempt(
                track=track,
                method="S3_strict_self_restart",
                schedule="independent_self_restart",
                source_temperature=float(candidate["temperature_K"]),
                target_labels=labels,
                initial_atmosphere=restart_seed,
                product_dir=case_root / "products" / "restart",
                iteration_cap=base.ITERATION_CAP,
                maximum_all_layer_relative_temperature_change=base.STRICT_ALL_LAYER_LIMIT,
                config_overrides=dict(ARM["S3"]),
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
        row = {
            **candidate,
            "arm": "S3",
            "primary": primary,
            "restart": restart,
            "primary_flux_gate": primary_flux,
            "restart_flux_gate": restart_flux,
            "path_consistency": consistency,
            "protocol_hash": protocol["protocol_hash"],
        }
        _write_json(case_root / "case.json", row)
        outputs.append(row)
        print(
            f"[S3] done {candidate['candidate_id']} "
            f"primary_flux={primary_flux.get('passes')} "
            f"restart_flux={restart_flux.get('passes')}",
            flush=True,
        )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--tomography-root", type=Path, default=DEFAULT_TOMOGRAPHY_ROOT)
    parser.add_argument(
        "--interp-root",
        type=Path,
        default=REPO_ROOT / "results" / "m_star_interpolated_mt_seed_v1",
    )
    parser.add_argument(
        "--v1r2-root",
        type=Path,
        default=REPO_ROOT / "results" / "m_star_emulator_v1r2_marcs100",
    )
    parser.add_argument(
        "--run-solves",
        action="store_true",
        help="Launch the S3 atmosphere solves. Off by default.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional candidate_id substrings (e.g. t3400 t3600).",
    )
    args = parser.parse_args(argv)

    try:
        protocol = _protocol()
    except Exception as exc:  # noqa: BLE001
        protocol = {
            "campaign": CAMPAIGN,
            "preregistration": str(PREREGISTRATION_PATH),
            "arm": ARM,
            "status": "protocol_partial",
            "error": f"{type(exc).__name__}: {exc}",
            "git_commit": git_commit(REPO_ROOT),
        }
    protocol["protocol_hash"] = config_fingerprint(
        {key: value for key, value in protocol.items() if key != "protocol_hash"}
    )
    result_root = Path(args.result_root)
    solves = None
    if args.run_solves:
        solves = _run_solves(args, protocol)
    payload = {
        **protocol,
        "solves_ran": bool(args.run_solves),
        "solves": solves,
    }
    _write_json(result_root / "summary.json", payload)
    write_arm_status(
        result_root,
        completed=not args.run_solves or solves is not None,
        config_fingerprint=protocol["protocol_hash"],
        campaign=CAMPAIGN,
        solves_ran=bool(args.run_solves),
    )
    print(json.dumps({"summary": str(result_root / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
