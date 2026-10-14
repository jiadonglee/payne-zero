"""Bootstrapping round 1: random CNO labels around the recovered MARCS giants.

Parents are the recovered MARCS giants, sorted by candidate id; every fifth
parent is held out and is not run. Each development parent receives four
random labels. The learned start is the frozen five-label v4 (m,T); the
reference start is the parent's converged (m,T). Both are rebuilt at the
target composition and judged by the existing stability, independent-path,
strict-restart and TiO spectral criteria.
"""
from bench import environment as _environment

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from payne_zero_atmosphere.warm_start import INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH
from reduced_state.emulator import load_physical_checkpoint, predict_physical_state
from coauthor_benchmark_20261005 import FIELDS
from coauthor_mgiant_cno_pilot_20261007 import worker
from coauthor_mgiant_stability_20261007 import evaluate
from coauthor_mgiant_cno_restart_20261007 import worker as strict_restart, spectrum, wait_for

NAMES = (*FIELDS, "carbon_enhancement", "nitrogen_enhancement", "oxygen_enhancement")
SEED = 20261009
PER_PARENT = 4
HOLDOUT_EVERY = 5
OFFSETS = dict(effective_temperature=75.0, log_surface_gravity=0.25, metallicity=0.25)
LIMITS = dict(effective_temperature=(3000.0, 4000.0), log_surface_gravity=(0.5, 2.5),
              metallicity=(-1.0, 0.5))
FREE = dict(alpha_enhancement=(-0.1, 0.1), microturbulence_km_s=(1.5, 2.5),
            carbon_enhancement=(-0.3, 0.1), nitrogen_enhancement=(-0.1, 0.5),
            oxygen_enhancement=(-0.1, 0.1))
TEMPERATURE_BINS = ((3000, 3300), (3300, 3600), (3600, 4001))


def parents(path):
    rows = sorted(json.loads(path.read_text())["rows"], key=lambda r: r["candidate_id"])
    roles = ["holdout" if i % HOLDOUT_EVERY == HOLDOUT_EVERY - 1 else "development"
             for i in range(len(rows))]
    return rows, roles


def draw(rows, roles):
    rng = np.random.default_rng(SEED)
    labels = []
    for parent, role in zip(rows, roles):
        if role != "development":
            continue
        for _ in range(PER_PARENT):
            row = {}
            for field, width in OFFSETS.items():
                lo, hi = LIMITS[field]
                row[field] = float(np.clip(parent["labels"][field] + rng.uniform(-width, width), lo, hi))
            for field, (lo, hi) in FREE.items():
                row[field] = float(rng.uniform(lo, hi))
            row["absolute_abundance_offsets"] = {z: row["metallicity"] + row[f]
                for z, f in zip((6, 7, 8), NAMES[-3:])}
            row["development_group"] = parent["candidate_id"]
            row["reference_product"] = parent["reference_product"]
            labels.append(row)
    return labels


def admit(index, root, gate_path, gate):
    directory = root / f"case_{index:02d}"
    if (directory / "admission.json").exists():
        return
    try:
        evaluate(directory, gate_path)
    except Exception as exc:
        (directory / "spectral_stability.json").write_text(json.dumps(
            dict(arm_errors=dict(evaluation=f"{type(exc).__name__}: {exc}")), indent=2) + "\n")
    strict_restart((index, root, gate))
    record = json.loads((directory / "strict_restart.json").read_text())
    if record["passes"]:
        try:
            a, b = spectrum(directory / "strict_restart.npz"), spectrum(Path(record["reference_product"]))
            metrics = dict(normalized_flux=float(np.max(abs(a[0] - b[0]))),
                flux_total=float(np.max(abs(a[1] - b[1]) / b[2])),
                flux_continuum=float(np.max(abs(a[2] / b[2] - 1))))
            record.update(spectral_metrics=metrics, passes=max(metrics.values()) <= 5e-3)
        except Exception as exc:
            record.update(passes=False, error=f"{type(exc).__name__}: {exc}")
    (directory / "admission.json").write_text(json.dumps(record, indent=2) + "\n")


def outcome(directory):
    """Classify a case by the first criterion it fails."""
    stability = json.loads((directory / "spectral_stability.json").read_text())
    record = json.loads((directory / "admission.json").read_text())
    if record["passes"]:
        return "admitted"
    errors = stability.get("arm_errors") or {}
    if "evaluation" in errors:
        return "evaluation_error"
    arms = stability.get("arms", {})
    if "reference" in errors or arms.get("reference", {}).get("frozen_iteration") is None:
        return "no_reference"
    if "physical" in errors or arms.get("physical", {}).get("frozen_iteration") is None:
        return "learned_not_stable"
    if not stability.get("spectral_consistency"):
        return "paths_disagree"
    return "strict_restart_or_spectrum"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "evaluate", "collect"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parents", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--gate", type=Path, default=Path("flux_gate.json"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="smoke test: first N cases only")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows, roles = parents(args.parents)
    if args.stage == "run":
        labels = draw(rows, roles)[:args.limit]
        setup = dict(expected=len(labels), labels=labels, random_seed=SEED, per_parent=PER_PARENT,
            parents=[dict(candidate_id=r["candidate_id"], labels=r["labels"], role=role)
                     for r, role in zip(rows, roles)],
            holdout_rule=f"sorted candidate id, position % {HOLDOUT_EVERY} == {HOLDOUT_EVERY - 1}",
            offsets=OFFSETS, limits=LIMITS, free_ranges=FREE,
            predictor="frozen five-label v4, target chemistry in reconstruction",
            reference="parent converged (m,T), target chemistry in reconstruction",
            role="bootstrapping round 1, development training; holdout parents not run",
            workers=args.workers, physical_iteration_cap=90, reference_iteration_cap=120)
        (root / "setup.json").write_text(json.dumps(setup, indent=2) + "\n")
        torch.set_num_threads(1)
        values = np.asarray([[row[f] for f in FIELDS] for row in labels])
        checkpoints = sorted(args.checkpoints.glob("checkpoint_mstar_seed*.pt"))
        if len(checkpoints) != 3:
            raise ValueError("the frozen v4 predictor requires three checkpoints")
        predictions = []
        for path in checkpoints:
            model, scaler = load_physical_checkpoint(path)[:2]
            predictions.append(predict_physical_state(model, scaler, values))
        mass, temperature = np.median(np.asarray(predictions), axis=0)
        payloads = []
        for i, row in enumerate(labels):
            with np.load(row["reference_product"]) as d:
                reference = np.column_stack((d["column_mass"], d["temperature"]))
            payloads.append((i, row, (np.column_stack((mass[i], temperature[i])), reference), root))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, payloads))
        return
    wait_for(root / "setup.json")
    setup = json.loads((root / "setup.json").read_text())
    if args.stage == "evaluate":
        gate = json.loads(args.gate.read_text())["thresholds"]
        for index in range(args.shard, setup["expected"], args.shards):
            admit(index, root, args.gate, gate)
        return
    records, labels, profiles, label_rows, groups = [], [], [], [], []
    for index in range(setup["expected"]):
        directory = root / f"case_{index:02d}"
        wait_for(directory / "admission.json")
        row = setup["labels"][index]
        status = outcome(directory)
        records.append(dict(index=index, outcome=status, parent=row["development_group"],
            effective_temperature=row["effective_temperature"],
            log_surface_gravity=row["log_surface_gravity"], metallicity=row["metallicity"],
            carbon_enhancement=row["carbon_enhancement"],
            nitrogen_enhancement=row["nitrogen_enhancement"]))
        if status == "admitted":
            labels.append([row[f] for f in NAMES]); label_rows.append(json.dumps(row))
            groups.append(row["development_group"])
            with np.load(directory / "strict_restart_profiles.npz") as d:
                profiles.append(d["profiles"].copy())
    counts = {s: sum(r["outcome"] == s for r in records) for s in
              ("admitted", "no_reference", "learned_not_stable", "paths_disagree",
               "strict_restart_or_spectrum", "evaluation_error")}
    with_reference = len(records) - counts["no_reference"]
    by_temperature = {f"{lo}-{min(hi, 4000)}": dict(
        cases=sum(lo <= r["effective_temperature"] < hi for r in records),
        admitted=sum(lo <= r["effective_temperature"] < hi and r["outcome"] == "admitted" for r in records),
        no_reference=sum(lo <= r["effective_temperature"] < hi and r["outcome"] == "no_reference" for r in records))
        for lo, hi in TEMPERATURE_BINS}
    np.savez(root / "cno_development_corpus.npz", labels=np.asarray(labels).reshape(-1, 8),
        atmosphere_profiles=np.asarray(profiles).reshape(-1, 80, 6),
        standard_rosseland_optical_depth=np.broadcast_to(INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH, (len(labels), 80)),
        labels_json=np.asarray(label_rows, dtype=str), parent_groups=np.asarray(groups, dtype=str),
        roles=np.asarray(["development_training"] * len(labels)))
    (root / "summary.json").write_text(json.dumps(dict(expected=setup["expected"],
        completed=len(records), counts=counts,
        admitted_fraction=counts["admitted"] / len(records),
        admitted_fraction_with_reference=counts["admitted"] / with_reference if with_reference else None,
        by_temperature=by_temperature, parent_groups=len(set(groups)),
        population="round 1: random labels around development parents", rows=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
