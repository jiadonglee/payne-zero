"""Resolve the five undetermined dual-path validation points.

Three points (`t3250 g2.0 [M/H]=0`, `t3050 g2.0 [M/H]=+0.5`,
`t3850 g0.5 [M/H]=-0.5`) have no usable reference: their nearest-node MARCS
seed drives the column mass non-positive within a few iterations.  They get a
continuation-seeded reference - the (m,T) of the nearest converged truth
product on the same (logg, [M/H]) track, walked in legs of at most 50 K into
the target, then relaxed under the frozen stop rule.  Two points
(`t3150 g1.5 [M/H]=-0.5`, `t3300 g0.75 [M/H]=+0.5`) have a stable reference
but the emulator arm missed the 30-iteration budget; their emulator arms are
re-solved with a 60-iteration budget against the unchanged reference.

Every arm is judged by the frozen stop rule; a point becomes decidable once
both arms stabilize, and its verdict is the cross-arm spectral comparison at
the frozen iterations against the `5e-3` bar.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from emulator_v1_2.gates.compare_spectra import (
    _absolute_stats,
    _continuum_scaled_stats,
    _load_spectrum_npz,
    _relative_stats,
)

from .cool_star_step_test import (
    TrackSpec,
    _clone_atmosphere,
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from .m_star_bootstrap_v1 import _load_mt, _write_json
from .m_star_stop_rule_v1 import (
    SEGMENT,
    STABLE_SEGMENTS_REQUIRED,
    WINDOW_NM,
    RESOLUTION,
    _labels_for,
    _node_id,
    _segment_changes,
    _segment_flux_pass,
    _segment_stable,
)

STRICT_ALL_LAYER_LIMIT = 5.0e-4

from bench.run_reference import _solver_config  # noqa: E402
from payne_zero_atmosphere.runner import run_atmosphere_model  # noqa: E402
from payne_zero_atmosphere.synthesis_bridge import (  # noqa: E402
    save_product_structured_atmosphere,
)
from reduced_state.emulator import (  # noqa: E402
    load_physical_checkpoint,
    predict_physical_state,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_resolve_undetermined_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_STOP_RULE_ROOT = REPO_ROOT / "results" / "m_star_stop_rule_v1"
DEFAULT_CORPUS = (
    REPO_ROOT / "results" / "m_star_cool_corpus_mgiant_v2" / "cool_truth_corpus.npz"
)
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "m_star_emulator_mgiant_v4"
DEFAULT_FLUX_GATE = REPO_ROOT / "results" / "m_star_emulator_v1" / "flux_gate.json"
ITERATION_CAP = 60
EMULATOR_BUDGET_SLOW = 60
SEEDS = (20260831, 20260901, 20260902)
MICROTURBULENCE = 2.0

# (teff, logg, metallicity, remedy)
TARGETS = (
    (3250.0, 2.0, 0.0, "seeded_reference"),
    (3050.0, 2.0, 0.5, "seeded_reference"),
    (3850.0, 0.5, -0.5, "seeded_reference"),
    (3150.0, 1.5, -0.5, "longer_budget"),
    (3300.0, 0.75, 0.5, "longer_budget"),
)


def _walk_seed_product(
    teff: float, logg: float, metallicity: float, corpus_path: Path
) -> Path | None:
    """Nearest converged truth product on the same (logg, [M/H]) track."""

    with np.load(corpus_path, allow_pickle=False) as data:
        roles = np.asarray(data["roles"]).astype(str)
        labels = np.asarray(data["labels"], dtype=np.float64)
        products = np.asarray(data["source_product_paths"]).astype(str)
    best = None
    best_distance = None
    for row_index in range(len(labels)):
        if abs(labels[row_index, 1] - logg) > 1e-6:
            continue
        if abs(labels[row_index, 2] - metallicity) > 1e-6:
            continue
        distance = abs(float(labels[row_index, 0]) - teff)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = products[row_index]
    if best is None:
        return None
    product = Path(best)
    if not product.is_absolute():
        product = REPO_ROOT / product
    return product if product.is_file() else None


def _released_continuation(
    *,
    start_atmosphere,
    arm_dir: Path,
    cap: int,
) -> dict[str, Any]:
    """Run with the stop released, products every five iterations."""

    arm_dir.mkdir(parents=True, exist_ok=True)
    save_product_structured_atmosphere(
        _clone_atmosphere(start_atmosphere),
        arm_dir / "iter_0000.npz",
        device="cpu",
        dtype="float64",
    )
    residual_handle = (arm_dir / "iterations.jsonl").open("w")

    def hook(iteration_index, setup, step):
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
            iterations_per_trial=int(cap),
            structured_atmosphere_path=None,
            debug_state_path=None,
        ),
        enable_convergence_stop=False,
    )
    result = run_atmosphere_model(config, after_iteration_hook=hook)
    residual_handle.close()
    return {
        "iterations_completed": int(result.iterations_completed),
        "converged": bool(result.converged),
    }


def _frozen_iteration(arm_dir: Path, gate: dict) -> int | None:
    checkpoints = sorted(
        int(path.stem.split("_")[1]) for path in arm_dir.glob("iter_*.npz")
    )
    consecutive = 0
    for k_end in checkpoints:
        if k_end - SEGMENT < 0 or (k_end - SEGMENT) not in checkpoints:
            continue
        changes = _segment_changes(arm_dir, k_end, arm_dir / "spectra", "relax")
        gate_pass = _segment_flux_pass(arm_dir, k_end, gate)
        if _segment_stable(changes, gate_pass):
            consecutive += 1
            if consecutive >= STABLE_SEGMENTS_REQUIRED:
                return k_end
        else:
            consecutive = 0
    return None


def _solve_reference_walk(
    *,
    teff: float,
    logg: float,
    metallicity: float,
    corpus: Path,
    arm_dir: Path,
    gate: dict,
) -> dict[str, Any]:
    """Continuation-seeded reference: walk <=50 K legs from the nearest truth."""

    track = TrackSpec(
        log_surface_gravity=float(logg),
        metallicity=float(metallicity),
        alpha_enhancement=0.0,
        carbon_enhancement=0.0,
        microturbulence_km_s=MICROTURBULENCE,
    )
    anchor_product = _walk_seed_product(teff, logg, metallicity, corpus)
    if anchor_product is None:
        return {"status": "no_anchor"}
    anchor_temperature = float(
        json.loads(
            (REPO_ROOT / "results" / "m_star_cool_corpus_mgiant_v2" / "cool_truth_corpus.json")
        .read_text()
        )["anchor_temperature"]
    ) if False else None
    with np.load(corpus, allow_pickle=False) as data:
        roles = np.asarray(data["roles"]).astype(str)
        labels_all = np.asarray(data["labels"], dtype=np.float64)
        products_all = np.asarray(data["source_product_paths"]).astype(str)
    match = None
    for row_index in range(len(labels_all)):
        if str(products_all[row_index]) == str(anchor_product):
            match = float(labels_all[row_index, 0])
            break
    anchor_temperature = float(match)

    leg_mass, leg_temperature_profile = _load_mt(anchor_product)
    leg_temperature = anchor_temperature
    leg_record = None
    while leg_temperature > teff + 1e-6:
        leg_temperature = max(leg_temperature - 50.0, teff)
        leg_labels = track.labels(leg_temperature)
        leg_seed = _reconstruct_from_mt(
            leg_labels, leg_mass, leg_temperature_profile
        )
        leg_record, _state = _solve_attempt(
            track=track,
            method="continuation_walk_leg",
            schedule="continuation_walk",
            source_temperature=float(teff),
            target_labels=leg_labels,
            initial_atmosphere=leg_seed,
            product_dir=arm_dir / "walk" / f"t{int(leg_temperature):04d}",
            iteration_cap=ITERATION_CAP,
            maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
        )
        if not leg_record.get("survives_solver"):
            break
        leg_mass, leg_temperature_profile = _load_mt(leg_record["product_path"])
    if leg_temperature > teff + 1e-6 or leg_record is None:
        return {"status": "walk_failed", "reached": leg_temperature}
    final_mass, final_temperature = _load_mt(leg_record["product_path"])
    labels = track.labels(teff)
    start = _reconstruct_from_mt(labels, final_mass, final_temperature)
    continuation = _released_continuation(
        start_atmosphere=start,
        arm_dir=arm_dir,
        cap=40,
    )
    return {
        "status": "ok",
        "anchor_temperature": anchor_temperature,
        "walk_completed": True,
        "continuation": continuation,
    }


def _solve_emulator(
    *,
    teff: float,
    logg: float,
    metallicity: float,
    checkpoint_dir: Path,
    arm_dir: Path,
    budget: int,
) -> dict[str, Any]:
    track = TrackSpec(
        log_surface_gravity=float(logg),
        metallicity=float(metallicity),
        alpha_enhancement=0.0,
        carbon_enhancement=0.0,
        microturbulence_km_s=MICROTURBULENCE,
    )
    labels = track.labels(teff)
    labels_row = np.asarray(
        [
            labels.effective_temperature,
            labels.log_surface_gravity,
            labels.metallicity,
            labels.alpha_enhancement,
            labels.microturbulence_km_s,
        ],
        dtype=np.float64,
    )
    masses = []
    temperatures = []
    for seed in SEEDS:
        model, standardization, _meta = load_physical_checkpoint(
            checkpoint_dir / f"checkpoint_mstar_seed{seed}.pt"
        )
        mass, temperature = predict_physical_state(
            model, standardization, labels_row.reshape(1, -1)
        )
        masses.append(np.asarray(mass)[0])
        temperatures.append(np.asarray(temperature)[0])
    median_mass = np.median(np.stack(masses, axis=0), axis=0)
    median_temperature = np.median(np.stack(temperatures, axis=0), axis=0)
    start = _reconstruct_from_mt(labels, median_mass, median_temperature)
    return _released_continuation(
        start_atmosphere=start,
        arm_dir=arm_dir,
        cap=budget,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR
    )
    parser.add_argument("--flux-gate", type=Path, default=DEFAULT_FLUX_GATE)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args(argv)
    gate = json.loads(args.flux_gate.read_text())

    from concurrent.futures import ProcessPoolExecutor

    payloads = []
    for teff, logg, metallicity, remedy in TARGETS:
        node = _node_id(teff, logg, metallicity)
        point_dir = args.result_root / "points" / node
        if remedy == "seeded_reference":
            arms = (("reference_walk", 40),)
        else:
            arms = (("emulator60", EMULATOR_BUDGET_SLOW),)
        payloads.append(
            (
                teff,
                logg,
                metallicity,
                remedy,
                str(point_dir),
                str(args.corpus),
                str(args.checkpoint_dir),
                gate,
                arms,
            )
        )

    def run_one(payload):
        (
            teff,
            logg,
            metallicity,
            remedy,
            point_dir_text,
            corpus_text,
            checkpoint_text,
            gate,
            arms,
        ) = payload
        _set_single_thread_environment()
        point_dir = Path(point_dir_text)
        point_dir.mkdir(parents=True, exist_ok=True)
        summary = {}
        for arm_name, budget in arms:
            arm_dir = point_dir / arm_name
            if (arm_dir / "iterations.jsonl").is_file():
                summary[arm_name] = "already done"
                continue
            if arm_name == "reference_walk":
                summary[arm_name] = _solve_reference_walk(
                    teff=teff,
                    logg=logg,
                    metallicity=metallicity,
                    corpus=Path(corpus_text),
                    arm_dir=arm_dir,
                    gate=gate,
                )
            else:
                summary[arm_name] = _solve_emulator(
                    teff=teff,
                    logg=logg,
                    metallicity=metallicity,
                    checkpoint_dir=Path(checkpoint_text),
                    arm_dir=arm_dir,
                    budget=budget,
                )
        summary["node_id"] = _node_id(teff, logg, metallicity)
        return summary

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for summary in pool.map(run_one, payloads):
            print(json.dumps(summary, sort_keys=True), flush=True)

    # Judge: frozen iteration per arm, cross-arm gate where both sides exist.
    rows = []
    for teff, logg, metallicity, remedy in TARGETS:
        node = _node_id(teff, logg, metallicity)
        point_dir = args.result_root / "points" / node
        row = {
            "node_id": node,
            "teff_K": teff,
            "logg": logg,
            "metallicity": metallicity,
            "remedy": remedy,
        }
        reference_arm = (
            point_dir / "reference_walk"
            if remedy == "seeded_reference"
            else args.stop_rule_root / "validate" / node / "reference"
        )
        emulator_arm = (
            point_dir / "emulator60"
            if remedy == "longer_budget"
            else args.stop_rule_root / "validate" / node / "emulator"
        )
        for name, arm_dir in (
            ("reference", reference_arm),
            ("emulator", emulator_arm),
        ):
            row[f"{name}_frozen"] = _frozen_iteration(arm_dir, gate)
        comparable = (
            row["reference_frozen"] is not None
            and row["emulator_frozen"] is not None
        )
        if comparable:
            reference_product = reference_arm / (
                f"iter_{row['reference_frozen']:04d}.npz"
            )
            emulator_product = emulator_arm / (
                f"iter_{row['emulator_frozen']:04d}.npz"
            )
            spectra_dir = args.result_root / "spectra"
            spectra_dir.mkdir(parents=True, exist_ok=True)

            def spectrum_for(product: Path, tag: str):
                spectrum_path = spectra_dir / f"{tag}.npz"
                if not spectrum_path.is_file():
                    _synthesize_one(
                        product,
                        spectrum_path,
                        wavelength_start_nm=WINDOW_NM[0],
                        wavelength_end_nm=WINDOW_NM[1],
                        resolution=RESOLUTION,
                        molecular_lines=True,
                        device=None,
                        dtype="float64",
                    )
                return _load_spectrum_npz(spectrum_path)

            reference_spectrum = spectrum_for(reference_product, f"{node}_ref")
            emulator_spectrum = spectrum_for(emulator_product, f"{node}_emu")
            row["cross_tio"] = {
                "normalized_flux": _absolute_stats(
                    emulator_spectrum["normalized_flux"],
                    reference_spectrum["normalized_flux"],
                )["max"],
                "flux_total": _continuum_scaled_stats(
                    emulator_spectrum["flux_total"],
                    reference_spectrum["flux_total"],
                    reference_spectrum["flux_continuum"],
                )["max"],
                "flux_continuum": _relative_stats(
                    emulator_spectrum["flux_continuum"],
                    reference_spectrum["flux_continuum"],
                )["max"],
            }
            reference_temperature = np.asarray(
                np.load(reference_product, allow_pickle=False)["temperature"],
                dtype=np.float64,
            )
            emulator_temperature = np.asarray(
                np.load(emulator_product, allow_pickle=False)["temperature"],
                dtype=np.float64,
            )
            row["cross_temperature_p95"] = float(
                np.percentile(
                    np.abs(emulator_temperature - reference_temperature)
                    / reference_temperature,
                    95.0,
                )
            )
            row["verdict"] = (
                "pass"
                if max(row["cross_tio"].values()) <= 5.0e-3
                else "spectral_fail"
            )
        else:
            row["verdict"] = "undetermined"
        rows.append(row)
        print(json.dumps(row, default=str)[:400], flush=True)

    _write_json(args.result_root / "resolution.json", {"rows": rows})
    passed = sum(1 for row in rows if row["verdict"] == "pass")
    print(json.dumps({"passed": passed, "total": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
