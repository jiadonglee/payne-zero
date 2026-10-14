"""Hydrostatic-seeded restart of the 3750 K M-giant cap case."""
from bench import environment as _environment

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reduced_state.emulator import load_physical_checkpoint, predict_physical_state
import coauthor_benchmark_20261005 as benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    row = dict(effective_temperature=3750.0, log_surface_gravity=0.75,
               metallicity=-0.50, alpha_enhancement=0.0, microturbulence_km_s=2.0)
    labels = np.asarray([[row[f] for f in benchmark.FIELDS]])
    torch.set_num_threads(1)
    checkpoints = sorted(args.checkpoints.glob("checkpoint_mstar_seed*.pt"))
    if len(checkpoints) != 3:
        raise ValueError(f"expected the three frozen v4 checkpoints, found {len(checkpoints)}")
    predictions = []
    for path in checkpoints:
        model, standardizer = load_physical_checkpoint(path)[:2]
        predictions.append(predict_physical_state(model, standardizer, labels))
    mass, temperature = np.median(np.asarray(predictions), axis=0)
    profile = np.column_stack((mass[0], temperature[0]))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "setup.json").write_text(json.dumps(dict(
        labels=row, checkpoints=[str(p) for p in checkpoints],
        predictor="frozen v4 coordinate-wise physical median", seed="Pgas=g*m",
        rebuild_pass_cap=8, pressure_tolerance_dex=1e-3,
        solver_iteration_cap=90, production_defaults_unchanged=True), indent=2) + "\n")
    benchmark.DATA = dict(rows={0: row}, prediction={2: {0: profile}})
    record = benchmark.worker((0, ["two_profile"], str(args.out), 90))
    print(json.dumps(record["arms"]["two_profile"], indent=2), flush=True)


if __name__ == "__main__":
    main()
