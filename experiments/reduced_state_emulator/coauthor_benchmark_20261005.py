"""Physical restarts on the fixed coauthor test sets, including every failure."""

from __future__ import annotations

from bench import environment as _environment

import argparse
import dataclasses
import json
import multiprocessing
import socket
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bench.run_reference import _solver_config
from payne_zero_atmosphere.atmosphere_io import parse_atmosphere_deck
from payne_zero_atmosphere.runner import run_atmosphere_model
from payne_zero_atmosphere.warm_start import (
    INITIALIZER_OUTPUT_FIELDS, format_warm_start_deck,
)
from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere

FIELDS = (
    "effective_temperature", "log_surface_gravity", "metallicity",
    "alpha_enhancement", "microturbulence_km_s",
)
DATA = {}


def load_data(root: Path, family: str, result: Path) -> dict:
    split = json.loads((result / family / "split.json").read_text())["indices"]
    if family == "direct_abundance":
        paths = sorted((root / "direct_xh_experimental/training_corpora").rglob("*.npz"))
        profiles, stellar, abundances = [], [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as d:
                profiles.append(d["atmosphere_profiles"])
                stellar.append(d["stellar_labels"])
                abundances.append(d["abundance_vectors"])
                tau = d["standard_rosseland_optical_depth"][0].copy()
        stellar, abundances = np.concatenate(stellar), np.concatenate(abundances)
        rows = [dict(effective_temperature=float(s[0]), log_surface_gravity=float(s[1]),
                     metallicity=float(a[26 - 3]), alpha_enhancement=0.0,
                     microturbulence_km_s=float(s[2]),
                     absolute_abundance_offsets={z: float(a[z - 3]) for z in range(3, 100)})
                for s, a in zip(stellar, abundances)]
        profiles = np.concatenate(profiles)
    else:
        filename = "strict_truth_52199.npz" if family == "five_label" else "strict_truth_53824.npz"
        with np.load(root / family / filename, allow_pickle=False) as d:
            rows = [json.loads(str(row)) for row in d["labels_json"]]
            profiles, tau = d["atmosphere_profiles"].copy(), d["standard_rosseland_optical_depth"][0].copy()
        if family == "cno8":
            for row in rows:
                row["absolute_abundance_offsets"] = {
                    z: float(row["metallicity"] + row[field])
                    for z, field in ((6, "carbon_enhancement"), (7, "nitrogen_enhancement"), (8, "oxygen_enhancement"))
                }
    prediction = {}
    for fields in (2, 6):
        with np.load(result / family / f"predicted_{fields}profile_test.npz", allow_pickle=False) as d:
            prediction[fields] = {int(index): profile for index, profile in zip(d["star_indices"], d["profiles"])}
    data = dict(split=split, rows=rows, profiles=profiles, tau=tau, prediction=prediction)
    if family == "five_label":
        labels = np.asarray([[row[field] for field in FIELDS] for row in rows])
        scale = np.asarray([250.0, 0.25, 0.20, 0.10, 0.50])
        donors = np.asarray(split["train"])
        tree = cKDTree(labels[donors] / scale)
        selected, distances = [], []
        targets = labels[split["test"]]
        for sign in (-1, 1):
            offset = targets + sign * np.asarray([250.0, 0.25, 0.20, 0.0, 0.0])
            _, position = tree.query(offset / scale)
            selected.append(donors[position])
            distances.append(labels[donors[position]] - offset)
        data["nodes"] = {int(index): [int(selected[arm][j]) for arm in range(2)]
                         for j, index in enumerate(split["test"])}
        data["node_distances"] = {int(index): [distances[arm][j].tolist() for arm in range(2)]
                                  for j, index in enumerate(split["test"])}
    return data


def atmosphere(profile: np.ndarray, row: dict):
    mass, temperature = profile[:, 0], profile[:, 1]
    if profile.shape[1] == 2:
        pressure = 10.0 ** row["log_surface_gravity"] * mass
        profile = np.column_stack((mass, temperature, pressure,
                                   1.0e-4 * pressure / (1.38054e-16 * temperature),
                                   np.full(80, 0.34), np.zeros(80)))
    table = np.zeros((80, 9))
    table[:, :6], table[:, 6] = profile, row["microturbulence_km_s"] * 1.0e5
    kwargs = {field: float(row[field]) for field in FIELDS[:4]}
    deck = format_warm_start_deck(**kwargs, layer_table=table,
                                 absolute_abundance_offsets=row.get("absolute_abundance_offsets"),
                                 title="Payne Zero matched profile restart")
    seed = parse_atmosphere_deck(deck, source="<matched-profile-restart>")
    # The deck carries chemistry; the numerical restart retains exact labels and profiles.
    return dataclasses.replace(seed,
        **{field: profile[:, i].copy() for i, field in enumerate(INITIALIZER_OUTPUT_FIELDS)},
        metadata={**seed.metadata, "effective_temperature": str(row["effective_temperature"]),
                  "log_surface_gravity": str(row["log_surface_gravity"])})


def save_profiles(path: Path, start, final=None, corrections=None, flux_ratio=None) -> None:
    values = dict(start=np.column_stack([getattr(start, field) for field in INITIALIZER_OUTPUT_FIELDS]))
    if final is not None:
        values["final"] = np.column_stack([getattr(final, field) for field in INITIALIZER_OUTPUT_FIELDS])
    if corrections:
        values["relative_corrections"] = np.stack(corrections)
    if flux_ratio is not None:
        values["final_flux_ratio"] = flux_ratio
    np.savez_compressed(path, **values)


def worker(payload: tuple) -> dict:
    index, arms, output, cap = payload
    row = DATA["rows"][index]
    directory = Path(output) / f"star_{index:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(index=index, labels={field: row[field] for field in FIELDS}, arms={})
    for field in ("carbon_enhancement", "nitrogen_enhancement", "oxygen_enhancement",
                  "absolute_abundance_offsets"):
        if field in row:
            record["labels"][field] = row[field]
    for arm in arms:
        print(json.dumps(dict(index=index, arm=arm, status="started")), flush=True)
        started = time.monotonic()
        cpu = time.process_time()
        entry = {}
        try:
            reconstruction = None
            if arm in ("two_profile", "exact_rebuild"):
                profile = (DATA["prediction"][2][index] if arm == "two_profile"
                           else DATA["profiles"][index, :, :2])
                seed = atmosphere(profile, row)
                reconstruction = reconstruct_full_atmosphere(
                    ReducedAtmosphere(profile[:, 0], profile[:, 1], {field: row[field] for field in FIELDS}),
                    seed=seed, n_synchronizations=None, max_synchronizations=8,
                    pressure_tolerance_dex=1.0e-3,
                )
                start = reconstruction.atmosphere
                entry["rebuild_evaluations"] = reconstruction.n_evaluations
                entry["rebuild_pressure_change_dex"] = reconstruction.pressure_change_dex_by_pass
            elif arm == "six_profile":
                start = atmosphere(DATA["prediction"][6][index], row)
            elif arm.startswith("nearest_"):
                node = int(arm[-1])
                donor = DATA["nodes"][index][node]
                start = atmosphere(DATA["profiles"][donor], row)
                entry["donor_index"] = donor
                entry["donor_labels"] = {field: DATA["rows"][donor][field] for field in FIELDS}
                entry["donor_minus_offset_labels"] = DATA["node_distances"][index][node]
            elif arm == "grey_convective":
                from experiments.analytic_initializer.textbook_opacity import predict_textbook_reduced_state_v4r6_decoupled
                from experiments.analytic_initializer.no_emulator_bridge import analytic_seed_model
                labels = np.asarray([row[field] for field in FIELDS])
                mass, temperature, log_opacity = predict_textbook_reduced_state_v4r6_decoupled(labels, DATA["tau"])
                start = analytic_seed_model(labels, mass[0], temperature[0], log_opacity[0], DATA["tau"])
                start = dataclasses.replace(start, metadata={**start.metadata,
                    "effective_temperature": str(row["effective_temperature"]),
                    "log_surface_gravity": str(row["log_surface_gravity"])})
            else:
                raise ValueError(arm)
            entry["start_cpu_seconds"] = time.process_time() - cpu
            entry["start_seconds"] = time.monotonic() - started
            save_profiles(directory / f"{arm}.npz", start)
            corrections = []
            flux_ratio = None

            def hook(n, setup, step):
                nonlocal flux_ratio
                if step.flux_ratio is not None:
                    flux_ratio = np.asarray(step.flux_ratio).copy()
                if n <= 3 and step.raw_temperature_correction is not None:
                    corrections.append(np.abs(step.raw_temperature_correction / step.opacity.population_state.setup.atmosphere.temperature))
                return None

            config = _solver_config(start, iterations_per_trial=cap,
                                    structured_atmosphere_path=None, debug_state_path=None)
            solve_cpu = time.process_time()
            result = run_atmosphere_model(config, after_iteration_hook=hook)
            entry["solve_cpu_seconds"] = time.process_time() - solve_cpu
            finite = all(np.all(np.isfinite(getattr(result.atmosphere, field))) for field in INITIALIZER_OUTPUT_FIELDS)
            entry.update(converged=bool(result.converged and finite), iterations=int(result.iterations_completed),
                         finite=finite, trajectory=result.diagnostics["iteration_timings"])
            save_profiles(directory / f"{arm}.npz", start, result.atmosphere, corrections, flux_ratio)
        except Exception as exc:
            entry.update(converged=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        entry["total_seconds"] = time.monotonic() - started
        record["arms"][arm] = entry
        print(json.dumps(dict(index=index, arm=arm, status="finished",
                              converged=entry["converged"], iterations=entry.get("iterations"),
                              seconds=entry["total_seconds"], error=entry.get("error"))), flush=True)
    (directory / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def summarize(records: list, family: str, cap: int, expected: int) -> dict:
    report = dict(family=family, host=socket.gethostname(), expected=expected,
                  completed=len(records), iteration_cap=cap, minimum_iterations=3,
                  stop_deep_relative_temperature=5.0e-4, arms={})
    for arm in records[0]["arms"] if records else []:
        entries = [record["arms"][arm] for record in records]
        converged = [entry for entry in entries if entry["converged"]]
        iterations = [entry["iterations"] for entry in converged]
        report["arms"][arm] = dict(total=len(entries), converged=len(converged), failed=len(entries) - len(converged),
            mean_iterations=float(np.mean(iterations)) if iterations else None,
            median_iterations=float(np.median(iterations)) if iterations else None,
            errors=[entry["error"] for entry in entries if "error" in entry])
    return report


def main() -> None:
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--family", choices=("five_label", "cno8", "direct_abundance"), default="five_label")
    parser.add_argument("--result-root", type=Path, default=Path("results/coauthor_validation_20261005"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--arms", default="two_profile,six_profile,exact_rebuild")
    args = parser.parse_args()
    DATA = load_data(args.corpus_root, args.family, args.result_root)
    indices = DATA["split"]["test"][:args.count]
    args.out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    done, pending = [], []
    for index in indices:
        path = args.out / f"star_{index:06d}/record.json"
        if path.exists():
            done.append(json.loads(path.read_text()))
        else:
            pending.append((index, arms, str(args.out), args.iterations))
    (args.out / "summary.json").write_text(
        json.dumps(summarize(done, args.family, args.iterations, len(indices)), indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        futures = [pool.submit(worker, payload) for payload in pending]
        for future in as_completed(futures):
            record = future.result()
            done.append(record)
            report = summarize(done, args.family, args.iterations, len(indices))
            (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(dict(completed=len(done), expected=len(indices), index=record["index"],
                                  results={arm: dict(converged=entry["converged"], iterations=entry.get("iterations"),
                                                     error=entry.get("error")) for arm, entry in record["arms"].items()})), flush=True)


if __name__ == "__main__":
    main()
