"""Per-iteration tomography of the four M-star solver-basin cases.

The interpolated-seed campaign left a sharp pattern on the rich dwarf track
(log g 4.5, [M/H] 0): 3500 K passes every gate, 3400 K formally converges but
fails the flux gate, 3300 K passes again; on the metal-poor track 3700 K
passes and 3600 K fails its primary flux gate. This campaign re-solves those
four cases with an ``after_iteration_hook`` that records the full per-iteration
state -- raw (pre-heuristic) temperature correction, applied correction, flux
error, convective flux ratio, superadiabatic gradient, Rosseland opacity,
electron density, and the per-layer molecular-equilibrium Newton pass count --
so the failure mode (solver oscillation, damped false convergence, or a
physics flux floor) can be read directly off the iteration history.

Solver settings and admission gates are identical to the interpolated arm;
the hook only observes. Seeds are rebuilt from the same one-sided donors
through the same interpolation code and pinned by sha256 in the preregistration.
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
from . import m_star_interpolated_mt_seed_v1 as interp
from .m_star_bootstrap_v1 import (
    _annotate_record,
    _load_mt,
    _passes_flux_gate,
    _product_consistency,
    _run_workers,
    _write_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_iteration_tomography_v1"
ITERATION_CAP = base.ITERATION_CAP
STRICT_ALL_LAYER_LIMIT = base.STRICT_ALL_LAYER_LIMIT
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_INTERP_ROOT = REPO_ROOT / "results" / "m_star_interpolated_mt_seed_v1"
DEFAULT_V1R2_ROOT = REPO_ROOT / "results" / "m_star_emulator_v1r2_marcs100"
PREREGISTRATION_PATH = (
    REPO_ROOT / "notes" / "m_star_iteration_tomography_v1_preregistration_20260905.md"
)

# The four cases, as (track slug, target Teff). Each is a same-track dwarf
# failure of v1r2 that the interpolated arm attempted; A and C passed every
# gate, B and D converged formally but failed the primary flux gate.
CASES = (
    ("g+4.50_m+0.00_a+0.00_c+0.00_x1.00", 3500.0),
    ("g+4.50_m+0.00_a+0.00_c+0.00_x1.00", 3400.0),
    ("g+4.50_m+0.00_a+0.00_c+0.00_x1.00", 3300.0),
    ("g+4.50_m-0.50_a+0.00_c+0.00_x1.00", 3600.0),
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _seed_array_sha256(path: Path) -> str:
    """Hash the seed columns themselves, not the npz container bytes.

    The pressure-synchronization reconstruction is bit-stable per platform
    but differs across BLAS backends (macOS Accelerate vs Linux), so the
    campaign canon is the array content of the machine that runs it.
    """

    with np.load(path, allow_pickle=False) as data:
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(data["column_mass"]).tobytes())
        digest.update(np.ascontiguousarray(data["temperature"]).tobytes())
    return digest.hexdigest()


def _case_dir(result_root: Path, track_slug: str, teff: float) -> Path:
    return result_root / "cases" / "dwarf" / track_slug / f"t{int(teff):04d}"


def _history_product(root: Path, track_slug: str, teff: float) -> Path | None:
    """Local primary product of a historical campaign case, if pulled back."""

    candidates = sorted(
        (root / "cases" / "dwarf" / track_slug / f"t{int(teff):04d}" / "products" / "primary").glob("*.npz")
    )
    return candidates[0] if candidates else None


def build_seed(
    *,
    interp_root: Path,
    v1r2_root: Path,
    track_slug: str,
    teff: float,
) -> tuple[Any, dict[str, Any]]:
    """Rebuild the interpolated-arm seed for one case from its donors."""

    interp_case = _read_json(interp_root / "cases" / "dwarf" / track_slug / f"t{int(teff):04d}" / "case.json")
    interpolation = interp_case["interpolation"]
    track_payload = dict(interp_case["track"])

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
        donor_records.append(_read_json(base._case_path(v1r2_root, donor_candidate)))

    labels = base._track_from_payload(track_payload).labels(float(teff))
    donor_reduced = []
    for row in donor_records:
        mass, temperature = _load_mt(interp._resolve_parent_product(v1r2_root, row))
        donor_reduced.append(np.stack([mass, temperature], axis=-1))
    column_mass, temperature, mix = interp.interpolate_same_track_mt(
        dict(labels.as_kwargs()),
        [dict(row["labels"]) for row in donor_records],
        np.stack(donor_reduced, axis=0),
    )
    seed = _reconstruct_from_mt(labels, column_mass, temperature)
    provenance = {
        "track": track_payload,
        "target_temperature_K": float(teff),
        "labels": labels.as_kwargs(),
        "donor_candidate_ids": list(interpolation["donor_candidate_ids"]),
        "donor_effective_temperatures": [
            float(value) for value in interpolation["donor_effective_temperatures"]
        ],
        "interpolation_kind": interpolation["kind"],
        **mix,
    }
    return seed, provenance


def make_tomography_hook(iterations_dir: Path):
    """Return an ``after_iteration_hook`` writing one NPZ per iteration.

    The pre-iteration temperature is the previous iteration's remapped
    temperature (identical to the iteration input by construction; iteration 1
    uses the setup atmosphere).
    """

    iterations_dir.mkdir(parents=True, exist_ok=True)
    state = {"previous_post_temperature": None}

    def hook(iteration_index, setup, step):
        correction = step.remapped.finalization.temperature_correction_result
        atmosphere = step.remapped.atmosphere
        pre_temperature = (
            setup.atmosphere.temperature
            if state["previous_post_temperature"] is None
            else state["previous_post_temperature"]
        )
        state["previous_post_temperature"] = np.asarray(
            atmosphere.temperature, dtype=np.float64
        )
        payload = {
            "iteration": np.int64(iteration_index),
            "log_tau_working": np.log10(
                np.asarray(
                    step.remapped.finalization.rosseland_optical_depth,
                    dtype=np.float64,
                )
            ),
            "log_tau_standard": np.log10(
                np.asarray(step.remapped.standard_rosseland_optical_depth, dtype=np.float64)
            ),
            "temperature_pre": np.asarray(pre_temperature, dtype=np.float64),
            "temperature_post": np.asarray(atmosphere.temperature, dtype=np.float64),
            "column_mass_post": np.asarray(atmosphere.column_mass, dtype=np.float64),
            "gas_pressure_post": np.asarray(atmosphere.gas_pressure, dtype=np.float64),
            "electron_density_post": np.asarray(
                atmosphere.electron_density, dtype=np.float64
            ),
            "rosseland_opacity_post": np.asarray(
                atmosphere.rosseland_opacity, dtype=np.float64
            ),
            "convective_flux_post": np.asarray(
                atmosphere.convective_flux, dtype=np.float64
            ),
            "raw_temperature_correction": np.asarray(
                step.raw_temperature_correction, dtype=np.float64
            ),
            "applied_temperature_correction": np.asarray(
                correction.temperature_correction, dtype=np.float64
            ),
            "flux_error_percent": np.asarray(
                correction.flux_error_percent, dtype=np.float64
            ),
            "flux_ratio": np.asarray(step.flux_ratio, dtype=np.float64),
            "superadiabatic_gradient": np.asarray(
                step.superadiabatic_gradient, dtype=np.float64
            ),
            "molecular_newton_iterations": np.asarray(
                step.molecular_newton_iterations, dtype=np.int64
            ),
            "molecular_newton_used_lstsq": np.asarray(
                step.molecular_newton_used_lstsq, dtype=np.bool_
            ),
        }
        for key, value in step.timing.items():
            payload[f"timing_{key}"] = value
        np.savez(
            iterations_dir / f"iter_{int(iteration_index):04d}.npz",
            **payload,
        )
        return {"iteration": int(iteration_index), "saved": str(iterations_dir)}

    return hook


def _failed_case(candidate: dict[str, Any], protocol_hash: str, error: Exception) -> dict[str, Any]:
    return {
        "campaign": CAMPAIGN,
        "protocol_hash": protocol_hash,
        **candidate,
        "training_eligible": False,
        "status": "error",
        "failure_reason": "exception",
        "error": f"{type(error).__name__}: {error}",
    }


def _case_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        candidate,
        result_root_text,
        interp_root_text,
        v1r2_root_text,
        flux_gate,
        protocol_hash,
        iteration_cap,
    ) = payload
    _set_single_thread_environment()
    result_root = Path(result_root_text)
    case_root = _case_dir(result_root, candidate["track_slug"], candidate["temperature_K"])
    case_path = case_root / "case.json"
    if case_path.is_file():
        return _read_json(case_path)

    try:
        seed, seed_provenance = build_seed(
            interp_root=Path(interp_root_text),
            v1r2_root=Path(v1r2_root_text),
            track_slug=candidate["track_slug"],
            teff=float(candidate["temperature_K"]),
        )
        seed_path = case_root / "seed.npz"
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            seed_path,
            column_mass=np.asarray(seed.column_mass, dtype=np.float64),
            temperature=np.asarray(seed.temperature, dtype=np.float64),
        )
        track = base._track_from_payload(seed_provenance["track"])
        labels = track.labels(float(candidate["temperature_K"]))

        primary, _primary_state = _solve_attempt(
            track=track,
            method="same_track_interpolated_atlas_mt_strict_primary",
            schedule="independent_interpolated_target",
            source_temperature=seed_provenance.get(
                "nearest_donor_effective_temperature"
            ),
            target_labels=labels,
            initial_atmosphere=seed,
            product_dir=case_root / "products" / "primary",
            iteration_cap=int(iteration_cap),
            maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
            after_iteration_hook=make_tomography_hook(case_root / "iterations" / "primary"),
        )
        primary = _annotate_record(
            primary,
            track_payload=seed_provenance["track"],
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
                iteration_cap=int(iteration_cap),
                maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
                after_iteration_hook=make_tomography_hook(
                    case_root / "iterations" / "restart"
                ),
            )
            restart = _annotate_record(
                restart,
                track_payload=seed_provenance["track"],
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

        history_product = _history_product(
            Path(interp_root_text),
            candidate["track_slug"],
            float(candidate["temperature_K"]),
        )
        parity: dict[str, Any] | None = None
        if history_product is not None and primary.get("product_path"):
            with np.load(history_product, allow_pickle=False) as reference, np.load(
                primary["product_path"], allow_pickle=False
            ) as current:
                if "temperature" in reference.files and "temperature" in current.files:
                    difference = np.abs(
                        np.asarray(current["temperature"], dtype=np.float64)
                        - np.asarray(reference["temperature"], dtype=np.float64)
                    )
                    parity = {
                        "reference_product": str(history_product),
                        "max_abs_temperature_difference_K": float(np.max(difference)),
                    }

        output = {
            "campaign": CAMPAIGN,
            "protocol_hash": protocol_hash,
            "iteration_cap": int(iteration_cap),
            **candidate,
            "labels": labels.as_kwargs(),
            "seed": {
                "path": str(seed_path),
                "sha256": _file_sha256(seed_path),
                "array_sha256": _seed_array_sha256(seed_path),
                **seed_provenance,
            },
            "iteration_tomography": {
                "primary_dir": str(case_root / "iterations" / "primary"),
                "restart_dir": str(case_root / "iterations" / "restart"),
            },
            "primary": primary,
            "restart": restart,
            "primary_flux_gate": primary_flux,
            "restart_flux_gate": restart_flux,
            "path_consistency": consistency,
            "parity_vs_interpolated_campaign": parity,
            "training_eligible": eligible,
            "status": "training_eligible" if eligible else "ineligible",
            "failure_reason": None if eligible else ",".join(reasons),
        }
    except Exception as exc:  # noqa: BLE001 - a failed case is an outcome
        output = _failed_case(candidate, protocol_hash, exc)
    _write_json(case_path, output)
    return output


def load_case_records(result_root: Path) -> list[dict[str, Any]]:
    return [
        _read_json(path)
        for path in sorted((result_root / "cases").glob("*/*/t*/case.json"))
    ]


def _candidates(only: str | None = None) -> list[dict[str, Any]]:
    candidates = [
        {
            "candidate_id": f"{track_slug}_t{int(teff)}",
            "class": "dwarf",
            "track_slug": track_slug,
            "temperature_K": float(teff),
        }
        for track_slug, teff in CASES
    ]
    if only:
        candidates = [row for row in candidates if only in row["candidate_id"]]
    return candidates


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    result_root = Path(args.result_root)
    interp_root = Path(args.interp_root)
    v1r2_root = Path(args.v1r2_root)
    gate = _read_json(interp_root / "flux_gate.json")

    seeds: dict[str, Any] = {}
    for track_slug, teff in CASES:
        seed, provenance = build_seed(
            interp_root=interp_root,
            v1r2_root=v1r2_root,
            track_slug=track_slug,
            teff=teff,
        )
        seed_path = _case_dir(result_root, track_slug, teff) / "seed.npz"
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            seed_path,
            column_mass=np.asarray(seed.column_mass, dtype=np.float64),
            temperature=np.asarray(seed.temperature, dtype=np.float64),
        )
        seeds[f"{track_slug}_t{int(teff)}"] = {
            **provenance,
            "seed_sha256": _file_sha256(seed_path),
            "seed_array_sha256": _seed_array_sha256(seed_path),
        }

    protocol = {
        "campaign": CAMPAIGN,
        "preregistration": str(PREREGISTRATION_PATH),
        "solver_policy": {
            "iteration_cap": int(ITERATION_CAP),
            "maximum_all_layer_relative_temperature_change": float(
                STRICT_ALL_LAYER_LIMIT
            ),
            "method": "same_track_interpolated_atlas_mt_strict_primary",
            "restart_method": "strict_self_restart_from_atlas_mt",
        },
        "cases": [f"{slug}_t{int(teff)}" for slug, teff in CASES],
        "flux_gate_source": {
            "campaign": gate.get("campaign"),
            "gate_hash": gate.get("gate_hash"),
            "thresholds": gate.get("thresholds"),
        },
        "seeds": seeds,
    }
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    _write_json(result_root / "protocol.json", protocol)
    _write_json(result_root / "flux_gate.json", gate)
    for name, seed in seeds.items():
        print(f"{name}: seed array sha256 {seed['seed_array_sha256']}")
        print(f"{name}: seed file  sha256 {seed['seed_sha256']}")
    print(f"protocol hash {protocol['protocol_hash']}")
    return protocol


def run_campaign(args: argparse.Namespace) -> int:
    result_root = Path(args.result_root)
    protocol = _read_json(result_root / "protocol.json")
    gate = _read_json(result_root / "flux_gate.json")
    iteration_cap = int(args.iterations) if int(args.iterations) > 0 else int(
        protocol["solver_policy"]["iteration_cap"]
    )
    payloads = [
        (
            candidate,
            str(result_root),
            str(Path(args.interp_root)),
            str(Path(args.v1r2_root)),
            gate,
            protocol["protocol_hash"],
            iteration_cap,
        )
        for candidate in _candidates(args.only)
    ]
    started = time.perf_counter()
    records = _run_workers(_case_worker, payloads, workers=int(args.workers))
    for record in sorted(records, key=lambda row: row["candidate_id"]):
        print(
            "{candidate_id}: eligible={eligible} status={status} "
            "primary_iters={iters} p95={p95} reason={reason}".format(
                candidate_id=record["candidate_id"],
                eligible=record.get("training_eligible"),
                status=record.get("status"),
                iters=(record.get("primary") or {}).get("iterations"),
                p95=(
                    (record.get("primary_flux_gate") or {})
                    .get("metrics", {})
                    .get("p95_absolute_flux_error_percent", {})
                    .get("value")
                ),
                reason=record.get("failure_reason"),
            )
        )
    print(f"wall seconds {time.perf_counter() - started:.1f}")
    return 0


def run_status(args: argparse.Namespace) -> int:
    result_root = Path(args.result_root)
    records = load_case_records(result_root)
    for record in sorted(records, key=lambda row: row["candidate_id"]):
        primary = record.get("primary") or {}
        print(
            record["candidate_id"],
            "eligible" if record.get("training_eligible") else "ineligible",
            "primary_iters", primary.get("iterations"),
            "failure", record.get("failure_reason"),
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    common.add_argument("--interp-root", default=str(DEFAULT_INTERP_ROOT))
    common.add_argument("--v1r2-root", default=str(DEFAULT_V1R2_ROOT))
    common.add_argument("--workers", type=int, default=4)
    common.add_argument(
        "--only",
        default=None,
        help="run only the cases whose candidate_id contains this substring",
    )
    common.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="override the iteration cap (0 keeps the frozen production cap)",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("protocol", parents=[common])
    sub.add_parser("run", parents=[common])
    sub.add_parser("status", parents=[common])
    args = parser.parse_args(argv)

    if args.stage == "protocol":
        run_protocol(args)
        return 0
    if args.stage == "run":
        return run_campaign(args)
    return run_status(args)


if __name__ == "__main__":
    raise SystemExit(main())
