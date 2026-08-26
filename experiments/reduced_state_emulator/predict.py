"""Materialize the learned (m,T) prediction to disk, as its own process.

This is split out of ``run_learned_restart`` for a concrete reason, not tidiness.
Running a torch forward pass in a process that later calls
``ProcessPoolExecutor`` deadlocks on Linux: the default start method is
``fork``, and torch's OpenMP/MKL thread pools do not survive it. The symptom is
silent -- every worker sits in state S with 00:00:00 of CPU time while the
parent waits forever. It cost one 22-minute cluster run to find.

``experiments/reduced_state_parity/run_oracle_parity.py`` never hits this
because its parent process reads truth profiles from an ``.npz`` and only the
*children* ever touch torch (via ``emulator_warm_start_model`` inside
``_seed_atmosphere``). Writing the prediction to an ``.npz`` restores exactly
that property, and makes the prediction an auditable artifact besides.

Usage::

    PYTHONPATH=. .venv/bin/python -m experiments.reduced_state_emulator.predict \\
        --arm monotone
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reduced_state.emulator import (
    load_checkpoint,
    load_corpus,
    predict_reduced_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
DEFAULT_HELD_OUT = REPO_ROOT / "results" / "reconstruction_metrics.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--held-out-from", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reduced_state_emulator",
    )
    parser.add_argument("--arm", default="monotone", choices=("monotone", "direct"))
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    held_out = np.array(
        json.loads(args.held_out_from.read_text())["star_indices"], dtype=np.int64
    )
    labels = corpus["labels"][held_out]

    checkpoint = args.checkpoint_dir / f"checkpoint_{args.arm}.pt"
    model, standardization, meta = load_checkpoint(checkpoint)
    if set(meta.get("held_out", [])) != set(int(i) for i in held_out):
        raise SystemExit(
            "checkpoint's held-out set does not match the evaluation set; "
            "the network may have trained on these stars"
        )

    column_mass, temperature = predict_reduced_state(model, standardization, labels)
    non_monotone = int((np.diff(column_mass, axis=1) <= 0).any(axis=1).sum())
    if non_monotone:
        raise SystemExit(
            f"{non_monotone}/{len(held_out)} predicted column-mass profiles are "
            "non-monotonic; reconstruct.py will reject them"
        )

    out_path = args.checkpoint_dir / f"predicted_{args.arm}.npz"
    np.savez(
        out_path,
        star_indices=held_out,
        labels=labels,
        column_mass=column_mass,
        temperature=temperature,
        truth_column_mass=corpus["column_mass"][held_out],
        truth_temperature=corpus["temperature"][held_out],
    )
    print(
        f"wrote {out_path}  ({len(held_out)} stars, monotone={model.monotone}, "
        f"violations={non_monotone})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
