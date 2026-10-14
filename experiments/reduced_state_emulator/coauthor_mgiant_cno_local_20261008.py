"""Twenty-four random development labels near the first accepted CNO batch."""
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
RANGES = ((3700, 3900), (.85, 1.15), (-.9, -.6), (-.05, .05), (1.75, 2.25),
          (-.1, .1), (-.1, .1), (-.1, .1))
GROUP = "t3800_g1.00_m-0.75"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--gate", type=Path, default=Path("flux_gate.json"))
    parser.add_argument("--after", type=Path)
    args = parser.parse_args()
    root, source = args.root.resolve(), args.source.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        rng = np.random.default_rng(20261008)
        values = np.column_stack([rng.uniform(lo, hi, 24) for lo, hi in RANGES])
        rows = [dict(zip(NAMES, map(float, row))) for row in values]
        for row in rows:
            row["absolute_abundance_offsets"] = {z: row["metallicity"] + row[f]
                for z, f in zip((6, 7, 8), NAMES[-3:])}
            row["development_group"] = GROUP
        setup = dict(expected=len(rows), labels=rows, ranges=dict(zip(NAMES, RANGES)),
            random_seed=20261008, role="local random development; no holdout",
            predictor="frozen five-label v4, target chemistry in reconstruction",
            reference="nearest accepted MARCS-derived pilot endpoint in CNO coordinates",
            source=str(source), parent_group=GROUP, workers=8,
            physical_iteration_cap=90, reference_iteration_cap=120)
        (root / "setup.json").write_text(json.dumps(setup, indent=2) + "\n")
        torch.set_num_threads(1)
        predictions = []
        checkpoints = sorted(args.checkpoints.glob("checkpoint_mstar_seed*.pt"))
        if len(checkpoints) != 3:
            raise ValueError("the frozen v4 predictor requires three checkpoints")
        for path in checkpoints:
            model, scaler = load_physical_checkpoint(path)[:2]
            predictions.append(predict_physical_state(model, scaler, values[:, :5]))
        mass, temperature = np.median(np.asarray(predictions), axis=0)
        accepted = [r["index"] for r in json.loads((source / "admission.json").read_text())["rows"] if r["passes"]]
        donors = []
        for index in accepted:
            report = json.loads((source / f"case_{index:02d}" / "spectral_stability.json").read_text())
            n = report["arms"]["reference"]["frozen_iteration"]
            path = source / f"case_{index:02d}" / "reference" / f"iter_{n:04d}.npz"
            with np.load(path) as d:
                donors.append((index, report["labels"], np.column_stack((d["column_mass"], d["temperature"]))))
        payloads = []
        for i, row in enumerate(rows):
            donor = min(donors, key=lambda d: sum((row[f] - d[1][f]) ** 2 for f in NAMES[-3:]))
            row["reference_donor_case"] = donor[0]
            payloads.append((i, row, (np.column_stack((mass[i], temperature[i])), donor[2]), root))
        (root / "setup.json").write_text(json.dumps(setup, indent=2) + "\n")
        with ProcessPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, payloads))
        return
    if args.after:
        wait_for(args.after)
    wait_for(root / "setup.json")
    setup = json.loads((root / "setup.json").read_text())
    gate = json.loads(args.gate.read_text())["thresholds"]
    stability, records, labels, profiles, label_rows = [], [], [], [], []
    for index in range(setup["expected"]):
        directory = root / f"case_{index:02d}"
        evaluate(directory, args.gate)
        stability.append(json.loads((directory / "spectral_stability.json").read_text()))
        strict_restart((index, root, gate))
        record = json.loads((directory / "strict_restart.json").read_text())
        if record["passes"]:
            a, b = spectrum(directory / "strict_restart.npz"), spectrum(Path(record["reference_product"]))
            metrics = dict(normalized_flux=float(np.max(abs(a[0] - b[0]))),
                flux_total=float(np.max(abs(a[1] - b[1]) / b[2])),
                flux_continuum=float(np.max(abs(a[2] / b[2] - 1))))
            record.update(spectral_metrics=metrics, passes=max(metrics.values()) <= 5e-3)
        if record["passes"]:
            row = record["labels"]
            labels.append([row[f] for f in NAMES]); label_rows.append(json.dumps(row))
            with np.load(directory / "strict_restart_profiles.npz") as d:
                profiles.append(d["profiles"].copy())
        records.append(record)
        (root / "partial.json").write_text(json.dumps(dict(expected=setup["expected"],
            completed=len(records), admitted=len(labels), rows=records), indent=2) + "\n")
    np.savez(root / "cno_development_corpus.npz", labels=np.asarray(labels).reshape(-1, 8),
        atmosphere_profiles=np.asarray(profiles).reshape(-1, 80, 6),
        standard_rosseland_optical_depth=np.broadcast_to(INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH, (len(labels), 80)),
        labels_json=np.asarray(label_rows, dtype=str), parent_groups=np.asarray([GROUP] * len(labels)),
        roles=np.asarray(["development_training"] * len(labels)))
    (root / "summary.json").write_text(json.dumps(dict(expected=setup["expected"],
        completed=len(records), admitted=len(labels), population="local random development labels",
        parent_group=GROUP, stability=stability, rows=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
