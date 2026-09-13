"""Validate the 60-iteration M-giant initializer configuration.

Stage ``t3250-reference`` builds an independent reference for the one
remaining undetermined point (`Teff 3250 K, logg 2.0, [M/H] = 0`): a logg
walk from the converged `g1.5 t3300` corpus truth (legs of 0.25 dex at fixed
temperature), then temperature legs of 25 K into the target, then a released
relaxation judged by the frozen stop rule.  The anchor comes from the corpus,
never from the emulator.

Stage ``six`` runs the six preregistered reproduction points (two nominal,
two cool, two low gravity, distinct metallicities, none used in training or
earlier tests) through both paths: emulator warm start with a 90-iteration
cap (verdicts distinguish stable-within-60 from over-budget) and a reference
arm (nearest-native-MARCS seed, walk-seeded fallback) within the 240-target
budget.  Stage ``evaluate`` assembles the merged eighteen-row table, the
diagnostic figure, and the cost summary.
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
    _synthesize_one,
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
    _continue_arm,
    _labels_for,
    _nearest_marcs_seed,
    _node_id,
    _segment_changes,
    _segment_flux_pass,
    _segment_stable,
)
from .m_star_resolve_undetermined_v1 import (
    EMULATOR_BUDGET_SLOW,
    ITERATION_CAP,
    MICROTURBULENCE,
    SEEDS,
    _frozen_iteration,
)

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
CAMPAIGN = "m_star_sixty_budget_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_CORPUS = (
    REPO_ROOT / "results" / "m_star_cool_corpus_mgiant_v2" / "cool_truth_corpus.npz"
)
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "m_star_emulator_mgiant_v4"
DEFAULT_FLUX_GATE = REPO_ROOT / "results" / "m_star_emulator_v1" / "flux_gate.json"
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
DEFAULT_RESOLVE_ROOT = REPO_ROOT / "results" / "m_star_resolve_undetermined_v1"
DEFAULT_STOP_RULE_ROOT = REPO_ROOT / "results" / "m_star_stop_rule_v1"

REFERENCE_RELAX_CAP = 120
EMULATOR_CAP = 90
WALK_LEG_K = 25.0
WALK_LEG_DEX = 0.25
GATE_BAR = 5.0e-3

T3250 = (3250.0, 2.0, 0.0)
T3250_ANCHOR = (3300.0, 1.5, 0.0)

# Preregistered reproduction points (teff, logg, metallicity, group).
SIX_POINTS = (
    (3675.0, 1.75, 0.5, "nominal"),
    (3525.0, 2.25, -1.0, "nominal"),
    (3175.0, 2.25, 0.5, "cool"),
    (3325.0, 1.5, -0.5, "cool"),
    (3750.0, 0.75, -0.5, "low_gravity"),
    (3400.0, 0.5, 0.5, "low_gravity"),
)


def _track(logg: float, metallicity: float) -> TrackSpec:
    return TrackSpec(
        log_surface_gravity=float(logg),
        metallicity=float(metallicity),
        alpha_enhancement=0.0,
        carbon_enhancement=0.0,
        microturbulence_km_s=MICROTURBULENCE,
    )


def _corpus_product(
    corpus: Path, logg: float, metallicity: float, teff: float
) -> tuple[Path, np.ndarray, np.ndarray] | None:
    with np.load(corpus, allow_pickle=False) as data:
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
            best = row_index
    if best is None:
        return None
    product = Path(products[best])
    if not product.is_absolute():
        product = REPO_ROOT / product
    if not product.is_file():
        return None
    mass, temperature = _load_mt(product)
    return product, mass, temperature


def _walk_leg(
    *,
    track: TrackSpec,
    labels: StellarLabels,
    start_mass: np.ndarray,
    start_temperature: np.ndarray,
    leg_dir: Path,
) -> dict[str, Any] | None:
    seed = _reconstruct_from_mt(labels, start_mass, start_temperature)
    leg, _state = _solve_attempt(
        track=track,
        method="parameter_walk_leg",
        schedule="parameter_walk",
        source_temperature=float(labels.effective_temperature),
        target_labels=labels,
        initial_atmosphere=seed,
        product_dir=leg_dir,
        iteration_cap=ITERATION_CAP,
        maximum_all_layer_relative_temperature_change=5.0e-4,
    )
    if not leg.get("survives_solver"):
        return None
    return leg


def _run_t3250_reference(args: argparse.Namespace) -> dict[str, Any]:
    _set_single_thread_environment()
    gate = json.loads(args.flux_gate.read_text())
    arm_dir = args.result_root / "points" / _node_id(*T3250) / "reference_walk"
    arm_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"legs": []}

    anchor = _corpus_product(
        args.corpus, T3250_ANCHOR[1], T3250_ANCHOR[2], T3250_ANCHOR[0]
    )
    if anchor is None:
        return {"status": "no_anchor"}
    anchor_product, mass, temperature = anchor
    report["anchor"] = str(anchor_product)

    current_mass, current_temperature = mass, temperature
    current_logg = T3250_ANCHOR[1]
    current_teff = T3250_ANCHOR[0]
    while current_logg < T3250[1] - 1e-6:
        current_logg = min(current_logg + WALK_LEG_DEX, T3250[1])
        leg_labels = _labels_for(current_teff, current_logg, T3250[2])
        leg = _walk_leg(
            track=_track(current_logg, T3250[2]),
            labels=leg_labels,
            start_mass=current_mass,
            start_temperature=current_temperature,
            leg_dir=arm_dir / "walk" / f"g{current_logg:+05.2f}_t{int(current_teff):04d}",
        )
        report["legs"].append(
            {"logg": current_logg, "teff": current_teff, "survived": leg is not None}
        )
        if leg is None:
            report["status"] = "logg_walk_failed"
            return report
        current_mass, current_temperature = _load_mt(leg["product_path"])
    while current_teff > T3250[0] + 1e-6:
        current_teff = max(current_teff - WALK_LEG_K, T3250[0])
        leg_labels = _labels_for(current_teff, current_logg, T3250[2])
        leg = _walk_leg(
            track=_track(current_logg, T3250[2]),
            labels=leg_labels,
            start_mass=current_mass,
            start_temperature=current_temperature,
            leg_dir=arm_dir / "walk" / f"g{current_logg:+05.2f}_t{int(current_teff):04d}",
        )
        report["legs"].append(
            {"logg": current_logg, "teff": current_teff, "survived": leg is not None}
        )
        if leg is None:
            report["status"] = "teff_walk_failed"
            return report
        current_mass, current_temperature = _load_mt(leg["product_path"])

    labels = _labels_for(*T3250)
    start = _reconstruct_from_mt(labels, current_mass, current_temperature)
    _continue_arm(
        labels=labels,
        start_atmosphere=start,
        arm_dir=arm_dir,
        cap=REFERENCE_RELAX_CAP,
        gate=gate,
    )
    report["status"] = "ok"
    report["reference_frozen"] = _frozen_iteration(arm_dir, gate)
    return report


def _emulator_seed(
    labels: StellarLabels, checkpoint_dir: Path
) -> Any:
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
    return _reconstruct_from_mt(labels, median_mass, median_temperature)


def _run_point(payload: tuple) -> dict[str, Any]:
    (
        teff,
        logg,
        metallicity,
        group,
        point_dir_text,
        corpus_text,
        checkpoint_text,
        marcs_text,
        gate,
    ) = payload
    _set_single_thread_environment()
    point_dir = Path(point_dir_text)
    point_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels_for(teff, logg, metallicity)
    summary: dict[str, Any] = {"node_id": _node_id(teff, logg, metallicity)}

    emulator_dir = point_dir / "emulator"
    if not (emulator_dir / "iterations.jsonl").is_file():
        start = _emulator_seed(labels, Path(checkpoint_text))
        _continue_arm(
            labels=labels,
            start_atmosphere=start,
            arm_dir=emulator_dir,
            cap=EMULATOR_CAP,
            gate=gate,
        )
    summary["emulator"] = "done"

    reference_dir = point_dir / "reference"
    if not (reference_dir / "iterations.jsonl").is_file():
        reference_dir.mkdir(parents=True, exist_ok=True)
        seed_atmosphere, seed_info = _nearest_marcs_seed(
            labels, Path(marcs_text)
        )
        _write_json(reference_dir / "seed.json", seed_info)
        _continue_arm(
            labels=labels,
            start_atmosphere=seed_atmosphere,
            arm_dir=reference_dir,
            cap=REFERENCE_RELAX_CAP,
            gate=gate,
        )
    if (reference_dir / "diverged.json").is_file():
        # Walk-seeded fallback: temperature legs from the nearest same-track
        # corpus truth into the target.
        if True:
            walk_dir = point_dir / "reference_walk"
            with np.load(Path(corpus_text), allow_pickle=False) as data:
                labels_all = np.asarray(data["labels"], dtype=np.float64)
                products_all = np.asarray(data["source_product_paths"]).astype(str)
            anchor_teff = None
            for row_index in range(len(labels_all)):
                if abs(labels_all[row_index, 1] - logg) > 1e-6:
                    continue
                if abs(labels_all[row_index, 2] - metallicity) > 1e-6:
                    continue
                distance = abs(float(labels_all[row_index, 0]) - teff)
                if anchor_teff is None or distance < anchor_teff[0]:
                    anchor_teff = (distance, float(labels_all[row_index, 0]))
            if anchor_teff is not None:
                anchor = _corpus_product(
                    Path(corpus_text), logg, metallicity, anchor_teff[1]
                )
                if anchor is not None:
                    _anchor_product, leg_mass, leg_temperature = anchor
                    leg_teff = anchor_teff[1]
                    walked = True
                    while leg_teff > teff + 1e-6:
                        leg_teff = max(leg_teff - WALK_LEG_K, teff)
                        leg_labels = _labels_for(leg_teff, logg, metallicity)
                        leg = _walk_leg(
                            track=_track(logg, metallicity),
                            labels=leg_labels,
                            start_mass=leg_mass,
                            start_temperature=leg_temperature,
                            leg_dir=walk_dir / "walk" / f"t{int(leg_teff):04d}",
                        )
                        if leg is None:
                            walked = False
                            break
                        leg_mass, leg_temperature = _load_mt(leg["product_path"])
                    if walked:
                        walk_dir.mkdir(parents=True, exist_ok=True)
                        start = _reconstruct_from_mt(
                            labels, leg_mass, leg_temperature
                        )
                        _continue_arm(
                            labels=labels,
                            start_atmosphere=start,
                            arm_dir=walk_dir,
                            cap=REFERENCE_RELAX_CAP,
                            gate=gate,
                        )
                        summary["reference"] = "walk_seed"
                        return summary
        summary["reference"] = "failed"
        return summary
    summary["reference"] = "marcs_seed"
    return summary


def _spectrum_for(product: Path, spectra_dir: Path, tag: str) -> Any:
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


def _cross_metrics(
    emulator_product: Path,
    reference_product: Path,
    spectra_dir: Path,
    node: str,
) -> dict[str, Any]:
    emulator_spectrum = _spectrum_for(
        emulator_product, spectra_dir, f"{node}_emu"
    )
    reference_spectrum = _spectrum_for(
        reference_product, spectra_dir, f"{node}_ref"
    )
    tiO = {
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
    emulator_arrays = np.load(emulator_product, allow_pickle=False)
    reference_arrays = np.load(reference_product, allow_pickle=False)
    dT = float(
        np.percentile(
            np.abs(
                np.asarray(emulator_arrays["temperature"], dtype=np.float64)
                - np.asarray(reference_arrays["temperature"], dtype=np.float64)
            )
            / np.asarray(reference_arrays["temperature"], dtype=np.float64),
            95.0,
        )
    )
    dlogm = float(
        np.percentile(
            np.abs(
                np.log10(np.asarray(emulator_arrays["column_mass"], dtype=np.float64))
                - np.log10(
                    np.asarray(reference_arrays["column_mass"], dtype=np.float64)
                )
            ),
            95.0,
        )
    )
    return {"tiO": tiO, "temperature_p95": dT, "mass_dex_p95": dlogm}


def _flux_at(arm_dir: Path, iteration: int) -> float | None:
    path = arm_dir / "iterations.jsonl"
    if not path.is_file():
        return None
    best = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["iteration"] <= iteration:
            best = row
    return None if best is None else best["flux_error_p95_percent"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("t3250-reference", "six", "evaluate", "run-all", "extend-relax"),
    )
    parser.add_argument("--arm-dir", type=Path, default=None)
    parser.add_argument("--from-iteration", type=int, default=120)
    parser.add_argument("--to-iteration", type=int, default=240)
    parser.add_argument("--extend-labels", default=None)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR
    )
    parser.add_argument("--flux-gate", type=Path, default=DEFAULT_FLUX_GATE)
    parser.add_argument("--marcs-grid", type=Path, default=DEFAULT_MARCS_GRID)
    parser.add_argument(
        "--stop-rule-root", type=Path, default=DEFAULT_STOP_RULE_ROOT
    )
    parser.add_argument(
        "--resolve-root", type=Path, default=DEFAULT_RESOLVE_ROOT
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    gate = json.loads(args.flux_gate.read_text())

    if args.stage in ("t3250-reference", "run-all"):
        report = _run_t3250_reference(args)
        _write_json(
            args.result_root / "points" / _node_id(*T3250) / "reference_report.json",
            report,
        )
        print(json.dumps(report, sort_keys=True), flush=True)
    if args.stage in ("six", "run-all"):
        from concurrent.futures import ProcessPoolExecutor

        payloads = []
        for teff, logg, metallicity, group in SIX_POINTS:
            payloads.append(
                (
                    teff,
                    logg,
                    metallicity,
                    group,
                    str(args.result_root / "points" / _node_id(teff, logg, metallicity)),
                    str(args.corpus),
                    str(args.checkpoint_dir),
                    str(args.marcs_grid),
                    gate,
                )
            )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for summary in pool.map(_run_point, payloads):
                print(json.dumps(summary, sort_keys=True), flush=True)
    if args.stage == "extend-relax":
        _extend_relax(args, gate)
        return 0
    if args.stage in ("evaluate", "run-all"):
        _evaluate(args, gate)
    return 0


def _extend_relax(args: argparse.Namespace, gate: dict) -> None:
    """Continue a released relaxation from its last checkpoint, appending."""

    _set_single_thread_environment()
    arm_dir = Path(args.arm_dir)
    teff, logg, metallicity = (
        float(value) for value in args.extend_labels.split(",")
    )
    labels = _labels_for(teff, logg, metallicity)
    start_product = arm_dir / f"iter_{args.from_iteration:04d}.npz"
    mass, temperature = _load_mt(start_product)
    start = _reconstruct_from_mt(labels, mass, temperature)
    residual_path = arm_dir / "iterations.jsonl"
    residual_handle = residual_path.open("a")

    def hook(iteration_index, setup, step):
        absolute = int(args.from_iteration) + int(iteration_index)
        flux_error = np.asarray(
            step.remapped.finalization.temperature_correction_result.flux_error_percent,
            dtype=np.float64,
        )
        residual_handle.write(
            json.dumps(
                {
                    "iteration": absolute,
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
        if absolute % 5 == 0:
            save_product_structured_atmosphere(
                _clone_atmosphere(step.remapped.atmosphere),
                arm_dir / f"iter_{absolute:04d}.npz",
                device="cpu",
                dtype="float64",
            )
        return {"iteration": absolute}

    config = dataclasses.replace(
        _solver_config(
            _clone_atmosphere(start),
            iterations_per_trial=int(args.to_iteration - args.from_iteration),
            structured_atmosphere_path=None,
            debug_state_path=None,
        ),
        enable_convergence_stop=False,
    )
    run_atmosphere_model(config, after_iteration_hook=hook)
    residual_handle.close()
    print(
        json.dumps(
            {
                "extended": str(arm_dir),
                "from": args.from_iteration,
                "to": args.to_iteration,
                "reference_frozen": _frozen_iteration(arm_dir, gate),
            }
        )
    )


def _evaluate(args: argparse.Namespace, gate: dict) -> None:
    rows = []
    spectra_dir = args.result_root / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    # The six new points and the t3250 resolution.
    entries = [(teff, logg, metallicity, group) for teff, logg, metallicity, group in SIX_POINTS]
    entries.append((*T3250, "t3250_resolution"))
    for teff, logg, metallicity, group in entries:
        node = _node_id(teff, logg, metallicity)
        point_dir = args.result_root / "points" / node
        emulator_arm = point_dir / "emulator"
        if group == "t3250_resolution":
            emulator_arm = (
                args.resolve_root / "points" / node / "emulator60"
            )
        emulator_frozen = _frozen_iteration(emulator_arm, gate)
        reference_arm = point_dir / "reference"
        if (reference_arm / "diverged.json").is_file() or not (
            (reference_arm / "iterations.jsonl").is_file()
        ):
            if (point_dir / "reference_walk" / "iterations.jsonl").is_file():
                reference_arm = point_dir / "reference_walk"
        reference_frozen = _frozen_iteration(reference_arm, gate)
        row = {
            "node_id": node,
            "teff_K": teff,
            "logg": logg,
            "metallicity": metallicity,
            "group": group,
            "emulator_frozen": emulator_frozen,
            "reference_frozen": reference_frozen,
            "emulator_flux_p95": _flux_at(
                emulator_arm, emulator_frozen or EMULATOR_CAP
            ),
            "reference_flux_p95": _flux_at(
                reference_arm, reference_frozen or REFERENCE_RELAX_CAP
            ),
        }
        if emulator_frozen is not None and reference_frozen is not None:
            metrics = _cross_metrics(
                emulator_arm / f"iter_{emulator_frozen:04d}.npz",
                reference_arm / f"iter_{reference_frozen:04d}.npz",
                spectra_dir,
                node,
            )
            row.update(metrics)
            spectral = max(metrics["tiO"].values()) <= GATE_BAR
            if emulator_frozen <= 60:
                row["verdict"] = "pass" if spectral else "consistency_fail"
                row["budget"] = "within_60"
            else:
                row["verdict"] = "pass" if spectral else "consistency_fail"
                row["budget"] = "over_budget"
        else:
            row["verdict"] = "undetermined"
            row["budget"] = "within_60"
        rows.append(row)

    # The twelve original points, carried forward with their budget tier.
    stop_table = json.loads(
        (args.stop_rule_root / "validation_table.json").read_text()
    )["rows"]
    resolve_rows = {
        row["node_id"]: row
        for row in json.loads(
            (args.resolve_root / "resolution.json").read_text()
        )["rows"]
    }
    t3250_node = _node_id(*T3250)
    for row in stop_table:
        node = row["node_id"]
        if node == t3250_node:
            continue
        if node in resolve_rows and row["verdict"] != "pass":
            resolved = resolve_rows[node]
            rows.append(
                {
                    "node_id": node,
                    "teff_K": resolved["teff_K"],
                    "logg": resolved["logg"],
                    "metallicity": resolved["metallicity"],
                    "group": "resolution",
                    "emulator_frozen": resolved["emulator_frozen"],
                    "reference_frozen": resolved["reference_frozen"],
                    "emulator_flux_p95": None,
                    "reference_flux_p95": None,
                    **(
                        {
                            "tiO": resolved["cross_tio"],
                            "temperature_p95": resolved["cross_temperature_p95"],
                            "mass_dex_p95": None,
                        }
                        if resolved.get("cross_tio")
                        else {}
                    ),
                    "verdict": resolved["verdict"],
                    "budget": (
                        "within_60"
                        if (resolved["emulator_frozen"] or 0) <= 60
                        else "over_budget"
                    ),
                }
            )
        else:
            rows.append(
                {
                    "node_id": node,
                    "teff_K": row["teff_K"],
                    "logg": row["logg"],
                    "metallicity": row["metallicity"],
                    "group": "original",
                    "emulator_frozen": row["emulator_frozen_iteration"],
                    "reference_frozen": row["reference_frozen_iteration"],
                    "emulator_flux_p95": None,
                    "reference_flux_p95": None,
                    **(
                        {
                            "tiO": row["cross_tio"],
                            "temperature_p95": row["cross_structure"]["temperature_p95"],
                            "mass_dex_p95": row["cross_structure"]["mass_dex_p95"],
                        }
                        if row.get("cross_tio")
                        else {}
                    ),
                    "verdict": row["verdict"],
                    "budget": (
                        "within_30"
                        if (row["emulator_frozen_iteration"] or 0) <= 30
                        else "within_60"
                    ),
                }
            )

    _write_json(args.result_root / "merged_table.json", {"rows": rows})
    import csv

    with (args.result_root / "merged_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "node_id",
                "teff_K",
                "logg",
                "metallicity",
                "group",
                "emulator_frozen",
                "reference_frozen",
                "emulator_flux_p95",
                "reference_flux_p95",
                "cross_temperature_p95",
                "cross_mass_dex_p95",
                "tio_normalized_flux",
                "tio_flux_total",
                "tio_flux_continuum",
                "budget",
                "verdict",
            ]
        )
        for row in rows:
            tiO = row.get("tiO") or {}
            writer.writerow(
                [
                    row["node_id"],
                    row["teff_K"],
                    row["logg"],
                    row["metallicity"],
                    row["group"],
                    row["emulator_frozen"],
                    row["reference_frozen"],
                    row["emulator_flux_p95"],
                    row["reference_flux_p95"],
                    row.get("temperature_p95"),
                    row.get("mass_dex_p95"),
                    tiO.get("normalized_flux"),
                    tiO.get("flux_total"),
                    tiO.get("flux_continuum"),
                    row["budget"],
                    row["verdict"],
                ]
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5))
    seen = set()
    for row in rows:
        color = "#1f77b4" if row["group"] in ("nominal", "original") else (
            "#2ca02c" if row["group"] == "cool" else "#9467bd"
        )
        marker = "o"
        if row["verdict"] == "pass":
            label = None
            ax.scatter(
                row["teff_K"],
                row["emulator_frozen"],
                color=color,
                marker=marker,
                s=60,
                zorder=3,
            )
            if row["reference_frozen"]:
                ax.scatter(
                    row["teff_K"],
                    row["reference_frozen"],
                    color=color,
                    marker=marker,
                    s=60,
                    facecolors="none",
                    linewidths=1.5,
                    zorder=3,
                )
        else:
            ax.scatter(row["teff_K"], 260, color="red", marker="x", s=90, zorder=4)
    ax.axhline(30, color="grey", linewidth=1, linestyle="--")
    ax.axhline(60, color="crimson", linewidth=1, linestyle="--")
    ax.text(4015, 32, "30-iteration standard", fontsize=8, ha="right")
    ax.text(4015, 63, "60-iteration budget", fontsize=8, color="crimson", ha="right")
    ax.scatter([], [], color="k", marker="o", s=60, label="emulator (filled) / reference (open)")
    ax.scatter([], [], color="red", marker="x", s=80, label="undetermined")
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("iterations to frozen precision")
    ax.set_yscale("symlog", linthresh=60)
    ax.set_yticks([20, 30, 45, 60, 85, 125, 240])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("60-iteration configuration: iterations to frozen precision")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "m_star_sixty_budget_v1.png", dpi=160)

    passed = sum(1 for row in rows if row["verdict"] == "pass")
    within30 = sum(
        1
        for row in rows
        if row["verdict"] == "pass" and row["budget"] == "within_30"
    )
    within60 = sum(
        1
        for row in rows
        if row["verdict"] == "pass"
        and row["budget"] in ("within_30", "within_60")
    )
    _write_json(
        args.result_root / "summary.json",
        {
            "rows": len(rows),
            "pass": passed,
            "pass_within_30": within30,
            "pass_within_60": within60,
            "undetermined": sum(1 for row in rows if row["verdict"] == "undetermined"),
        },
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "pass": passed,
                "pass_within_30": within30,
                "pass_within_60": within60,
            }
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
