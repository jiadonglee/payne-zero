"""Freeze a numerical stop rule on the probe stars, then validate it on 12 points.

Stage ``probe-continue`` continues the three fixed-point probe stars (both
arms) from their k=20 checkpoints by forty released iterations, saving every
five.  Stage ``freeze`` evaluates the proposed stability rule - two
consecutive five-iteration segments each with all three TiO metric changes
below ``5e-4``, temperature-change p95 below ``1e-4``, column-mass-change
p95 below ``1e-3 dex``, and the frozen flux gate passing - and reports where
each arm first satisfies it.  Stage ``validate-solve`` runs the twelve
preregistered validation points through two paths - emulator warm start and
nearest-native-MARCS initialization - into the same unchanged solver, the
emulator path capped at thirty iterations and the reference path at two
hundred forty.  Stage ``validate-evaluate`` applies the frozen rule
post hoc and emits the twelve-row verdict table.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import csv
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
)
from .marcs_h5 import inspect_marcs_grid, load_marcs_node
from .m_star_bootstrap_v1 import _load_mt, _write_json

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
CAMPAIGN = "m_star_stop_rule_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_K20_ROOT = REPO_ROOT / "results" / "m_star_fixed_point_probe_v1_k20"
DEFAULT_CORPUS = (
    REPO_ROOT / "results" / "m_star_cool_corpus_mgiant_v1" / "cool_truth_corpus.npz"
)
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "m_star_emulator_mgiant_v3"
DEFAULT_FLUX_GATE = (
    REPO_ROOT / "results" / "m_star_emulator_v1" / "flux_gate.json"
)
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
PROBE_SEEDS = ((3850.0, "candidate"), (3850.0, "truth"), (3900.0, "candidate"),
               (3900.0, "truth"), (3950.0, "candidate"), (3950.0, "truth"))
SEGMENT = 5

TIO_STABLE_LIMIT = 5.0e-4
TEMPERATURE_STABLE_P95 = 1.0e-4
MASS_STABLE_P95_DEX = 1.0e-3
ORIGINAL_ALL_LAYER_LIMIT = 5.0e-4
EMULATOR_CAP = 30
REFERENCE_CAP = 240
STABLE_SEGMENTS_REQUIRED = 2
WINDOW_NM = (665.0, 667.0)
RESOLUTION = 20000.0

# Preregistered validation points (teff_K, logg, metallicity).  Giant vmic is
# 2 km/s, alpha = carbon = 0.  Every node avoids the training grid (logg
# 0.5/1.5/2.5 at the twelve standard temperatures), the opened v1r1
# validation stars (g2.0 [M/H]=0 at 3750-4000 K, g4.8), and the three carved
# validation nodes.
VALIDATION_POINTS = (
    (3650.0, 2.0, -0.5),
    (3825.0, 1.5, 0.0),
    (3575.0, 2.5, -1.0),
    (3925.0, 2.0, 0.5),
    (3250.0, 2.0, 0.0),
    (3150.0, 1.5, -0.5),
    (3350.0, 2.5, -1.0),
    (3050.0, 2.0, 0.5),
    (3650.0, 0.75, 0.0),
    (3850.0, 0.5, -0.5),
    (3450.0, 1.0, -1.0),
    (3300.0, 0.75, 0.5),
)
SEEDS = (20260831, 20260901, 20260902)
MICROTURBULENCE = 2.0


def _labels_for(teff: float, logg: float, metallicity: float) -> StellarLabels:
    track = TrackSpec(
        log_surface_gravity=float(logg),
        metallicity=float(metallicity),
        alpha_enhancement=0.0,
        carbon_enhancement=0.0,
        microturbulence_km_s=MICROTURBULENCE,
    )
    return track.labels(float(teff))


def _node_id(teff: float, logg: float, metallicity: float) -> str:
    return (
        f"g{logg:+05.2f}_m{metallicity:+05.2f}_a+0.00_c+0.00"
        f"_x{MICROTURBULENCE:.2f}_t{int(teff):04d}"
    )


def _flux_gate_passes(flux_error: np.ndarray, gate: dict[str, Any]) -> bool:
    metrics = {
        "median_absolute_flux_error_percent": float(
            np.percentile(np.abs(flux_error), 50.0)
        ),
        "p95_absolute_flux_error_percent": float(
            np.percentile(np.abs(flux_error), 95.0)
        ),
        "maximum_absolute_flux_error_percent": float(np.max(np.abs(flux_error))),
    }
    thresholds = gate["thresholds"]
    return all(
        metrics[name] <= float(thresholds[name]) for name in metrics
    )


def _continue_arm(
    *,
    labels: StellarLabels,
    start_atmosphere,
    arm_dir: Path,
    cap: int,
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Run a released-stop continuation, products every five iterations."""

    arm_dir.mkdir(parents=True, exist_ok=True)
    start_temperature = np.asarray(
        initial_temperature(start_atmosphere), dtype=np.float64
    )
    save_product_structured_atmosphere(
        _clone_atmosphere(start_atmosphere),
        arm_dir / "iter_0000.npz",
        device="cpu",
        dtype="float64",
    )
    residual_handle = (arm_dir / "iterations.jsonl").open("w")
    history: list[np.ndarray] = [start_temperature]

    def hook(iteration_index, setup, step):
        post_temperature = np.asarray(
            step.remapped.atmosphere.temperature, dtype=np.float64
        )
        flux_error = np.asarray(
            step.remapped.finalization.temperature_correction_result.flux_error_percent,
            dtype=np.float64,
        )
        reference = history[-1]
        record = {
            "iteration": int(iteration_index),
            "flux_error_p95_percent": float(
                np.percentile(np.abs(flux_error), 95.0)
            ),
            "flux_error_median_percent": float(
                np.percentile(np.abs(flux_error), 50.0)
            ),
            "flux_error_max_percent": float(np.max(np.abs(flux_error))),
            "update_temperature_relative_max": float(
                np.max(np.abs(post_temperature - reference) / reference)
            ),
        }
        residual_handle.write(json.dumps(record, sort_keys=True) + "\n")
        residual_handle.flush()
        history.append(post_temperature)
        if int(iteration_index) % SEGMENT == 0:
            save_product_structured_atmosphere(
                _clone_atmosphere(step.remapped.atmosphere),
                arm_dir / f"iter_{int(iteration_index):04d}.npz",
                device="cpu",
                dtype="float64",
            )
        return record

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


def initial_temperature(atmosphere) -> np.ndarray:
    return np.asarray(atmosphere.temperature, dtype=np.float64)


def _probe_continue(args: argparse.Namespace) -> int:
    _set_single_thread_environment()
    gate = json.loads(args.flux_gate.read_text())
    root = args.result_root / "probe"
    root.mkdir(parents=True, exist_ok=True)
    for teff, arm in PROBE_SEEDS:
        start_product = (
            args.k20_root
            / "checkpoints"
            / f"t{int(teff):04d}"
            / arm
            / "iter_0020.npz"
        )
        if not start_product.is_file():
            raise SystemExit(f"FAIL_STOP: missing {start_product}")
        mass, temperature = _load_mt(start_product)
        labels = _labels_for(teff, 2.0, 0.0)
        start = _reconstruct_from_mt(labels, mass, temperature)
        arm_dir = root / f"t{int(teff):04d}" / arm
        if (arm_dir / "iterations.jsonl").is_file():
            print(f"t{int(teff)} {arm}: already continued", flush=True)
            continue
        record = _continue_arm(
            labels=labels,
            start_atmosphere=start,
            arm_dir=arm_dir,
            cap=args.extra_iterations,
            gate=gate,
        )
        print(f"t{int(teff)} {arm}: {record}", flush=True)
    _write_json(root / "probe_continue.json", {"extra": args.extra_iterations})
    return 0


def _spectrum_for(product: Path, spectra_dir: Path, tag: str) -> Path:
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
    return spectrum_path


def _segment_changes(
    arm_dir: Path, k_end: int, spectra_dir: Path, tag_prefix: str
) -> dict[str, Any]:
    product_end = arm_dir / f"iter_{k_end:04d}.npz"
    product_start = arm_dir / f"iter_{k_end - SEGMENT:04d}.npz"
    end = np.load(product_end, allow_pickle=False)
    start = np.load(product_start, allow_pickle=False)
    temperature_start = np.asarray(start["temperature"], dtype=np.float64)
    temperature_end = np.asarray(end["temperature"], dtype=np.float64)
    mass_start = np.asarray(start["column_mass"], dtype=np.float64)
    mass_end = np.asarray(end["column_mass"], dtype=np.float64)
    spectrum_start = _load_spectrum_npz(
        _spectrum_for(product_start, spectra_dir, f"{tag_prefix}_k{k_end - SEGMENT:04d}")
    )
    spectrum_end = _load_spectrum_npz(
        _spectrum_for(product_end, spectra_dir, f"{tag_prefix}_k{k_end:04d}")
    )
    tiO = {
        "normalized_flux": _absolute_stats(
            spectrum_end["normalized_flux"], spectrum_start["normalized_flux"]
        )["max"],
        "flux_total": _continuum_scaled_stats(
            spectrum_end["flux_total"],
            spectrum_start["flux_total"],
            spectrum_start["flux_continuum"],
        )["max"],
        "flux_continuum": _relative_stats(
            spectrum_end["flux_continuum"], spectrum_start["flux_continuum"]
        )["max"],
    }
    return {
        "tiO": tiO,
        "temperature_p95": float(
            np.percentile(
                np.abs(temperature_end - temperature_start) / temperature_start,
                95.0,
            )
        ),
        "mass_dex_p95": float(
            np.percentile(
                np.abs(np.log10(mass_end) - np.log10(mass_start)), 95.0
            )
        ),
    }


def _segment_flux_pass(arm_dir: Path, k_end: int, gate: dict) -> bool:
    rows = [
        json.loads(line)
        for line in (arm_dir / "iterations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_iteration = {row["iteration"]: row for row in rows}
    segment = [
        by_iteration[k]
        for k in range(k_end - SEGMENT + 1, k_end + 1)
        if k in by_iteration
    ]
    if len(segment) < SEGMENT:
        return False
    return all(
        row["flux_error_p95_percent"] <= float(gate["thresholds"]["p95_absolute_flux_error_percent"])
        and row["flux_error_median_percent"]
        <= float(gate["thresholds"]["median_absolute_flux_error_percent"])
        and row["flux_error_max_percent"]
        <= float(gate["thresholds"]["maximum_absolute_flux_error_percent"])
        for row in segment
    )


def _segment_stable(changes: dict, gate_pass: bool) -> bool:
    return (
        max(changes["tiO"].values()) < TIO_STABLE_LIMIT
        and changes["temperature_p95"] < TEMPERATURE_STABLE_P95
        and changes["mass_dex_p95"] < MASS_STABLE_P95_DEX
        and gate_pass
    )


def _freeze(args: argparse.Namespace) -> int:
    _set_single_thread_environment()
    gate = json.loads(args.flux_gate.read_text())
    root = args.result_root
    spectra_dir = root / "probe" / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    rule = {"segment": SEGMENT, "arms": {}}
    for teff, arm in PROBE_SEEDS:
        arm_dir = root / "probe" / f"t{int(teff):04d}" / arm
        tag = f"t{int(teff):04d}_{arm}"
        segments = {}
        first_stable = None
        consecutive = 0
        for k_end in range(SEGMENT, args.extra_iterations + 1, SEGMENT):
            if not (arm_dir / f"iter_{k_end:04d}.npz").is_file():
                continue
            if not (arm_dir / f"iter_{k_end - SEGMENT:04d}.npz").is_file():
                continue
            changes = _segment_changes(arm_dir, k_end, spectra_dir, tag)
            gate_pass = _segment_flux_pass(arm_dir, k_end, gate)
            stable = _segment_stable(changes, gate_pass)
            segments[f"{k_end - SEGMENT}-{k_end}"] = {**changes, "gate_pass": gate_pass, "stable": stable}
            consecutive = consecutive + 1 if stable else 0
            if consecutive >= STABLE_SEGMENTS_REQUIRED and first_stable is None:
                first_stable = k_end
        rule["arms"][f"t{int(teff):04d}_{arm}"] = {
            "segments": segments,
            "first_stable_end": first_stable,
        }
        print(f"t{int(teff)} {arm}: first stable segment end = {first_stable}", flush=True)
    stable_ends = [
        value["first_stable_end"]
        for value in rule["arms"].values()
        if value["first_stable_end"] is not None
    ]
    rule["frozen_iteration_cap"] = (
        int(max(stable_ends)) if stable_ends else None
    )
    _write_json(root / "stop_rule.json", rule)
    print(json.dumps({k: v for k, v in rule.items() if k != "arms"}, indent=2))
    return 0


def _nearest_marcs_seed(
    labels: StellarLabels, marcs_grid: Path
) -> tuple[Any, dict[str, Any]]:
    schema = inspect_marcs_grid(
        marcs_grid, verify_sha256=False, expected_sha256=None
    )
    requested = {
        "effective_temperature": labels.effective_temperature,
        "log_surface_gravity": labels.log_surface_gravity,
        "metallicity": labels.metallicity,
    }
    snapped = {}
    distances = {}
    for name, value in requested.items():
        values = np.asarray(schema.grid_values[name], dtype=np.float64)
        nearest = values[int(np.argmin(np.abs(values - value)))]
        snapped[name] = float(nearest)
        distances[name] = float(abs(nearest - value))
    snapped_labels = _labels_for(
        snapped["effective_temperature"],
        snapped["log_surface_gravity"],
        snapped["metallicity"],
    )
    node = load_marcs_node(
        marcs_grid,
        snapped_labels,
        carbon_enhancement=0.0,
        verify_sha256=False,
        expected_sha256=None,
        schema=schema,
    )
    # MARCS provides only the starting (m,T) profile; the reconstruction and
    # the solver target carry the requested labels, not the snapped node's.
    atmosphere = _reconstruct_from_mt(
        labels, node.reduced_column_mass, node.reduced_temperature
    )
    return atmosphere, {"snapped": snapped, "distances": distances}


def _emulator_seed(
    labels: StellarLabels, checkpoint_dir: Path
) -> tuple[Any, dict[str, Any]]:
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
    return _reconstruct_from_mt(
        labels, median_mass, median_temperature
    ), {"seeds": list(SEEDS), "policy": "coordinate-wise median"}


def _validate_point_worker(payload: tuple) -> dict[str, Any]:
    (
        teff,
        logg,
        metallicity,
        point_dir_text,
        emulator_cap,
        reference_cap,
        marcs_grid_text,
        checkpoint_dir_text,
        gate,
    ) = payload
    _set_single_thread_environment()
    labels = _labels_for(teff, logg, metallicity)
    node = _node_id(teff, logg, metallicity)
    point_dir = Path(point_dir_text)
    point_dir.mkdir(parents=True, exist_ok=True)
    arms = {
        "emulator": (emulator_cap, "emulator"),
        "reference": (reference_cap, "marcs"),
    }
    summary = {"node_id": node}
    for arm, (cap, kind) in arms.items():
        arm_dir = point_dir / arm
        if (arm_dir / "iterations.jsonl").is_file():
            summary[arm] = "already done"
            continue
        arm_dir.mkdir(parents=True, exist_ok=True)
        if kind == "emulator":
            start, seed_info = _emulator_seed(
                labels, Path(checkpoint_dir_text)
            )
        else:
            start, seed_info = _nearest_marcs_seed(
                labels, Path(marcs_grid_text)
            )
        _write_json(arm_dir / "seed.json", seed_info)
        record = _continue_arm(
            labels=labels,
            start_atmosphere=start,
            arm_dir=arm_dir,
            cap=cap,
            gate=gate,
        )
        summary[arm] = record
    return summary


def _validate_solve(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor

    gate = json.loads(args.flux_gate.read_text())
    root = args.result_root / "validate"
    root.mkdir(parents=True, exist_ok=True)
    payloads = []
    for teff, logg, metallicity in VALIDATION_POINTS:
        node = _node_id(teff, logg, metallicity)
        point_dir = root / node
        payloads.append(
            (
                teff,
                logg,
                metallicity,
                str(point_dir),
                args.emulator_cap,
                args.reference_cap,
                str(args.marcs_grid),
                str(args.checkpoint_dir),
                gate,
            )
        )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for summary in pool.map(_validate_point_worker, payloads):
            print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


def _original_stop_iteration(arm_dir: Path) -> int | None:
    rows = [
        json.loads(line)
        for line in (arm_dir / "iterations.jsonl").read_text().splitlines()
        if line.strip()
    ]
    for row in rows:
        if row["iteration"] >= 3 and (
            row["update_temperature_relative_max"] <= ORIGINAL_ALL_LAYER_LIMIT
        ):
            return int(row["iteration"])
    return None


def _validate_evaluate(args: argparse.Namespace) -> int:
    _set_single_thread_environment()
    gate = json.loads(args.flux_gate.read_text())
    root = args.result_root
    spectra_dir = root / "validate" / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    table = []
    for teff, logg, metallicity in VALIDATION_POINTS:
        node = _node_id(teff, logg, metallicity)
        row = {
            "node_id": node,
            "teff_K": teff,
            "logg": logg,
            "metallicity": metallicity,
        }
        arm_results = {}
        for arm, cap in (("emulator", args.emulator_cap), ("reference", args.reference_cap)):
            arm_dir = root / "validate" / node / arm
            if not (arm_dir / "iterations.jsonl").is_file():
                raise SystemExit(f"FAIL_STOP: missing {arm_dir}")
            k_original = _original_stop_iteration(arm_dir)
            first_stable = None
            consecutive = 0
            previous_changes = None
            for k_end in range(SEGMENT, cap + 1, SEGMENT):
                if not (arm_dir / f"iter_{k_end:04d}.npz").is_file():
                    continue
                if not (arm_dir / f"iter_{k_end - SEGMENT:04d}.npz").is_file():
                    continue
                changes = _segment_changes(
                    arm_dir, k_end, spectra_dir, f"{node}_{arm}"
                )
                gate_pass = _segment_flux_pass(arm_dir, k_end, gate)
                stable = _segment_stable(changes, gate_pass)
                consecutive = consecutive + 1 if stable else 0
                previous_changes = changes
                if consecutive >= STABLE_SEGMENTS_REQUIRED and first_stable is None:
                    first_stable = k_end
                    break
            arm_results[arm] = {
                "original_stop_iteration": k_original,
                "frozen_iteration": first_stable,
                "reached_cap_unstable": first_stable is None,
                "last_segment": previous_changes,
            }
        emulator = arm_results["emulator"]
        reference = arm_results["reference"]
        comparable = (
            emulator["frozen_iteration"] is not None
            and reference["frozen_iteration"] is not None
        )
        if comparable:
            candidate_product = (
                root / "validate" / node / "emulator"
                / f"iter_{emulator['frozen_iteration']:04d}.npz"
            )
            reference_product = (
                root / "validate" / node / "reference"
                / f"iter_{reference['frozen_iteration']:04d}.npz"
            )
            candidate_spectrum = _load_spectrum_npz(
                _spectrum_for(candidate_product, spectra_dir, f"{node}_emulator_final")
            )
            reference_spectrum = _load_spectrum_npz(
                _spectrum_for(reference_product, spectra_dir, f"{node}_reference_final")
            )
            cross_spectrum = {
                "normalized_flux": _absolute_stats(
                    candidate_spectrum["normalized_flux"],
                    reference_spectrum["normalized_flux"],
                )["max"],
                "flux_total": _continuum_scaled_stats(
                    candidate_spectrum["flux_total"],
                    reference_spectrum["flux_total"],
                    reference_spectrum["flux_continuum"],
                )["max"],
                "flux_continuum": _relative_stats(
                    candidate_spectrum["flux_continuum"],
                    reference_spectrum["flux_continuum"],
                )["max"],
            }
            candidate_temperature = np.load(
                candidate_product, allow_pickle=False
            )["temperature"]
            reference_temperature = np.load(
                reference_product, allow_pickle=False
            )["temperature"]
            candidate_mass = np.load(candidate_product, allow_pickle=False)["column_mass"]
            reference_mass = np.load(reference_product, allow_pickle=False)["column_mass"]
            cross_structure = {
                "temperature_p95": float(
                    np.percentile(
                        np.abs(candidate_temperature - reference_temperature)
                        / reference_temperature,
                        95.0,
                    )
                ),
                "mass_dex_p95": float(
                    np.percentile(
                        np.abs(
                            np.log10(candidate_mass) - np.log10(reference_mass)
                        ),
                        95.0,
                    )
                ),
            }
        else:
            cross_spectrum = None
            cross_structure = None
        verdict = "undetermined"
        if comparable:
            within_budget = emulator["frozen_iteration"] <= args.emulator_cap
            spectral = max(cross_spectrum.values()) <= 5.0e-3
            verdict = (
                "pass" if within_budget and spectral else "spectral_fail"
                if within_budget
                else "over_budget"
            )
        row.update(
            {
                "emulator_original_stop": emulator["original_stop_iteration"],
                "emulator_frozen_iteration": emulator["frozen_iteration"],
                "reference_original_stop": reference["original_stop_iteration"],
                "reference_frozen_iteration": reference["frozen_iteration"],
                "reference_unstable": reference["reached_cap_unstable"],
                "cross_structure": cross_structure,
                "cross_tio": cross_spectrum,
                "verdict": verdict,
            }
        )
        table.append(row)
        print(json.dumps(row, default=str)[:400], flush=True)

    output = args.result_root / "validation_table.json"
    _write_json(output, {"campaign": CAMPAIGN, "rows": table})
    with (args.result_root / "validation_table.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "node_id",
                "teff_K",
                "logg",
                "metallicity",
                "emulator_original_stop",
                "emulator_frozen_iteration",
                "reference_original_stop",
                "reference_frozen_iteration",
                "reference_unstable",
                "cross_temperature_p95",
                "cross_mass_dex_p95",
                "tio_normalized_flux",
                "tio_flux_total",
                "tio_flux_continuum",
                "verdict",
            ]
        )
        for row in table:
            cross_spectrum = row["cross_tio"] or {}
            cross_structure = row["cross_structure"] or {}
            writer.writerow(
                [
                    row["node_id"],
                    row["teff_K"],
                    row["logg"],
                    row["metallicity"],
                    row["emulator_original_stop"],
                    row["emulator_frozen_iteration"],
                    row["reference_original_stop"],
                    row["reference_frozen_iteration"],
                    row["reference_unstable"],
                    cross_structure.get("temperature_p95"),
                    cross_structure.get("mass_dex_p95"),
                    cross_spectrum.get("normalized_flux"),
                    cross_spectrum.get("flux_total"),
                    cross_spectrum.get("flux_continuum"),
                    row["verdict"],
                ]
            )
    print(f"wrote {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("probe-continue", "freeze", "validate-solve", "validate-evaluate"),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--k20-root", type=Path, default=DEFAULT_K20_ROOT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--flux-gate", type=Path, default=DEFAULT_FLUX_GATE)
    parser.add_argument("--marcs-grid", type=Path, default=DEFAULT_MARCS_GRID)
    parser.add_argument("--extra-iterations", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--emulator-cap", type=int, default=EMULATOR_CAP)
    parser.add_argument("--reference-cap", type=int, default=REFERENCE_CAP)
    args = parser.parse_args(argv)

    if args.stage == "probe-continue":
        return _probe_continue(args)
    if args.stage == "freeze":
        return _freeze(args)
    if args.stage == "validate-solve":
        return _validate_solve(args)
    if args.stage == "validate-evaluate":
        return _validate_evaluate(args)
    raise AssertionError(f"unhandled stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
