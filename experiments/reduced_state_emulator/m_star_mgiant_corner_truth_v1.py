"""Generate corner truth rows for the M-giant corpus.

Six cool giant nodes on the open `g1.5` metal-rich and metal-poor tracks -
the two corners the attribution analysis identified as data-thin - are
solved with the unchanged MARCS-seeded machinery (same-node native MARCS
`(m,T)` seed, strict all-layer criterion, cap 60) and then relaxed thirty
iterations under the frozen stop rule.  A row is admitted when the primary
solve survives, both flux gates pass, path consistency holds, and the
released continuation reaches the frozen precision; the admitted (m,T) is
the final relaxed state, not the formal-stop state.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.labels import StellarLabels  # noqa: E402
from bench.run_reference import _solver_config  # noqa: E402
from payne_zero_atmosphere.runner import run_atmosphere_model  # noqa: E402
from payne_zero_atmosphere.synthesis_bridge import (  # noqa: E402
    save_product_structured_atmosphere,
)

from .cool_star_step_test import (
    TrackSpec,
    _clone_atmosphere,
    _marcs_diagnostics,
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from .marcs_h5 import inspect_marcs_grid, load_marcs_node
from .m_star_bootstrap_v1 import (
    _load_mt,
    PATH_COLUMN_MASS_P95_DEX_LIMIT,
    PATH_TEMPERATURE_P95_LIMIT,
    _annotate_record,
    _passes_flux_gate,
    _product_consistency,
    _run_workers,
    _sha256,
    _write_json,
)
from .m_star_stop_rule_v1 import (
    MASS_STABLE_P95_DEX,
    SEGMENT,
    STABLE_SEGMENTS_REQUIRED,
    TEMPERATURE_STABLE_P95,
    TIO_STABLE_LIMIT,
)
from .m_star_stop_rule_v1 import _segment_changes, _segment_flux_pass


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_mgiant_corner_truth_v1"
STRICT_ALL_LAYER_LIMIT = 5.0e-4
ITERATION_CAP = 60
RELAX_ITERATIONS = 30
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
DEFAULT_FLUX_PARENT_ROOT = REPO_ROOT / "results" / "m_star_emulator_v1"
MICROTURBULENCE = 2.0

CORNER_NODES = (
    (1.5, 0.5, 3100.0),
    (1.5, 0.5, 3200.0),
    (1.5, 0.5, 3300.0),
    (1.5, -0.5, 3100.0),
    (1.5, -0.5, 3200.0),
    (1.5, -0.5, 3300.0),
)


def _labels_for(logg: float, metallicity: float, teff: float) -> StellarLabels:
    track = TrackSpec(
        log_surface_gravity=float(logg),
        metallicity=float(metallicity),
        alpha_enhancement=0.0,
        carbon_enhancement=0.0,
        microturbulence_km_s=MICROTURBULENCE,
    )
    return track.labels(float(teff))


def _node_dir(result_root: Path, logg: float, metallicity: float, teff: float) -> Path:
    return (
        result_root
        / "cases"
        / f"g{logg:+05.2f}_m{metallicity:+05.2f}_a+0.00_c+0.00_x{MICROTURBULENCE:.2f}"
        / f"t{int(teff):04d}"
    )


def _relax_with_rule(
    *,
    labels: StellarLabels,
    start_atmosphere,
    arm_dir: Path,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Release the stop for thirty iterations and apply the frozen rule."""

    arm_dir.mkdir(parents=True, exist_ok=True)
    save_product_structured_atmosphere(
        _clone_atmosphere(start_atmosphere),
        arm_dir / "iter_0000.npz",
        device="cpu",
        dtype="float64",
    )
    history: list[np.ndarray] = [np.asarray(start_atmosphere.temperature)]
    residual_handle = (arm_dir / "iterations.jsonl").open("w")

    def hook(iteration_index, setup, step):
        post_temperature = np.asarray(
            step.remapped.atmosphere.temperature, dtype=np.float64
        )
        flux_error = np.asarray(
            step.remapped.finalization.temperature_correction_result.flux_error_percent,
            dtype=np.float64,
        )
        residual_handle.write(
            json.dumps(
                {
                    "iteration": int(iteration_index),
                    "flux_error_p95_percent": float(
                        np.percentile(np.abs(flux_error), 95.0)
                    ),
                    "flux_error_median_percent": float(
                        np.percentile(np.abs(flux_error), 50.0)
                    ),
                    "flux_error_max_percent": float(np.max(np.abs(flux_error))),
                },
                sort_keys=True,
            )
            + "\n"
        )
        residual_handle.flush()
        history.append(post_temperature)
        if int(iteration_index) % SEGMENT == 0:
            save_product_structured_atmosphere(
                _clone_atmosphere(step.remapped.atmosphere),
                arm_dir / f"iter_{int(iteration_index):04d}.npz",
                device="cpu",
                dtype="float64",
            )
        return {"iteration": int(iteration_index)}

    config = dataclasses.replace(
        _solver_config(
            _clone_atmosphere(start_atmosphere),
            iterations_per_trial=int(RELAX_ITERATIONS),
            structured_atmosphere_path=None,
            debug_state_path=None,
        ),
        enable_convergence_stop=False,
    )
    run_atmosphere_model(config, after_iteration_hook=hook)

    spectra_dir = arm_dir / "spectra"
    first_stable = None
    consecutive = 0
    segments = {}
    for k_end in range(SEGMENT, RELAX_ITERATIONS + 1, SEGMENT):
        if not (arm_dir / f"iter_{k_end:04d}.npz").is_file():
            continue
        changes = _segment_changes(arm_dir, k_end, spectra_dir, "relax")
        gate_pass = _segment_flux_pass(arm_dir, k_end, gate)
        stable = _segment_stable(changes, gate_pass)
        segments[f"{k_end - SEGMENT}-{k_end}"] = {
            **changes,
            "gate_pass": gate_pass,
            "stable": stable,
        }
        consecutive = consecutive + 1 if stable else 0
        if consecutive >= STABLE_SEGMENTS_REQUIRED and first_stable is None:
            first_stable = k_end
    return {
        "first_stable_end": first_stable,
        "stable": first_stable is not None,
        "segments": segments,
    }


def _solve_case_worker(payload: tuple) -> dict[str, Any]:
    (
        logg,
        metallicity,
        teff,
        result_root_text,
        marcs_grid_text,
        flux_gate,
    ) = payload
    _set_single_thread_environment()
    result_root = Path(result_root_text)
    case_dir = _node_dir(result_root, logg, metallicity, teff)
    case_path = case_dir / "case.json"
    if case_path.is_file():
        return json.loads(case_path.read_text())
    labels = _labels_for(logg, metallicity, teff)
    candidate_id = (
        f"g{logg:+05.2f}_m{metallicity:+05.2f}_a+0.00_c+0.00"
        f"_x{MICROTURBULENCE:.2f}_t{int(teff):04d}"
    )

    schema = inspect_marcs_grid(
        marcs_grid_text, verify_sha256=False, expected_sha256=None
    )
    node = load_marcs_node(
        Path(marcs_grid_text),
        labels,
        carbon_enhancement=0.0,
        verify_sha256=False,
        expected_sha256=None,
        schema=schema,
    )
    seed = _reconstruct_from_mt(
        labels, node.reduced_column_mass, node.reduced_temperature
    )
    track = TrackSpec(
        log_surface_gravity=float(logg),
        metallicity=float(metallicity),
        alpha_enhancement=0.0,
        carbon_enhancement=0.0,
        microturbulence_km_s=MICROTURBULENCE,
    )
    case_dir.mkdir(parents=True, exist_ok=True)

    primary, _state = _solve_attempt(
        track=track,
        method="native_marcs_same_node_strict_primary",
        schedule="independent_marcs_target",
        source_temperature=None,
        target_labels=labels,
        initial_atmosphere=seed,
        product_dir=case_dir / "products" / "primary",
        iteration_cap=ITERATION_CAP,
        maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
    )
    primary = _annotate_record(
        primary, track_payload=track.as_json(), role="train", node_id=candidate_id
    )
    primary_flux = _passes_flux_gate(primary, flux_gate)

    relax_report: dict[str, Any] | None = None
    final_mass = final_temperature = None
    final_product_rel = None
    if primary.get("survives_solver") and primary.get("product_path"):
        solved_m, solved_t = _load_mt(primary["product_path"])
        continuation_seed = _reconstruct_from_mt(labels, solved_m, solved_t)
        relax_report = _relax_with_rule(
            labels=labels,
            start_atmosphere=continuation_seed,
            arm_dir=case_dir / "relax",
            gate=flux_gate,
        )
        if relax_report["stable"]:
            final_product = (
                case_dir
                / "relax"
                / f"iter_{int(relax_report['first_stable_end']):04d}.npz"
            )
            final_mass, final_temperature = _load_mt(final_product)
            final_product_rel = str(
                final_product.relative_to(result_root)
            )

    output = {
        "campaign": CAMPAIGN,
        "candidate_id": candidate_id,
        "labels": labels.as_kwargs(),
        "admitted": bool(
            relax_report
            and relax_report["stable"]
            and final_mass is not None
            and primary_flux["passes"]
        ),
        "primary": primary,
        "primary_flux_gate": primary_flux,
        "relax": relax_report,
        "final_product": final_product_rel,
        "marcs_seed": {
            "source_sha256": _sha256(Path(marcs_grid_text)),
            "fields_used": ["column_mass", "temperature"],
            "is_training_target": False,
            "diagnostics": _marcs_diagnostics(node),
        },
    }
    _write_json(case_path, output)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--marcs-grid", type=Path, default=DEFAULT_MARCS_GRID)
    parser.add_argument("--flux-parent-root", type=Path, default=DEFAULT_FLUX_PARENT_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    args.result_root.mkdir(parents=True, exist_ok=True)

    gate_path = args.result_root / "flux_gate.json"
    if not gate_path.is_file():
        parent = args.flux_parent_root / "flux_gate.json"
        parent_gate = json.loads(parent.read_text())
        payload = {
            "campaign": CAMPAIGN,
            "status": "pass",
            "frozen": True,
            "thresholds": dict(parent_gate["thresholds"]),
            "source": {"path": str(parent), "gate_hash": parent_gate["gate_hash"]},
            "thresholds_refit": False,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["gate_hash"] = hashlib.sha256(canonical.encode()).hexdigest()
        _write_json(gate_path, payload)
    flux_gate = json.loads(gate_path.read_text())
    if not flux_gate.get("frozen"):
        raise SystemExit("FAIL_STOP: flux gate not frozen")

    payloads = [
        (
            logg,
            metallicity,
            teff,
            str(args.result_root),
            str(args.marcs_grid),
            flux_gate,
        )
        for logg, metallicity, teff in CORNER_NODES
    ]
    outputs = _run_workers(_solve_case_worker, payloads, workers=args.workers)
    admitted = sum(1 for row in outputs if bool(row.get("admitted")))
    print(
        json.dumps(
            {
                "campaign": CAMPAIGN,
                "attempted": len(outputs),
                "admitted": admitted,
                "nodes": [
                    {
                        "candidate_id": row["candidate_id"],
                        "admitted": bool(row.get("admitted")),
                        "stable": bool((row.get("relax") or {}).get("stable")),
                    }
                    for row in sorted(outputs, key=lambda r: r["candidate_id"])
                ],
            },
            indent=2,
        )
    )
    if admitted == len(outputs):
        (args.result_root / "CORNER_RUN_COMPLETE").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
