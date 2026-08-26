"""Control: how far does the solver's own converged product move on its own?

The spectral gate compares the learned arm's converged atmosphere against the
production arm's and asks whether they agree below 5e-3. Two of 57 pairs do not.
That number is uninterpretable without knowing the solver's *intrinsic* width --
if the production initializer, restarted from a start it itself considers
equivalent, moves its own fixed point by a comparable amount, then the two
exceedances are a property of the convergence criterion, not of the learned
initializer.

The production retry policy supplies exactly such a start for free.
``deterministic_initializer_labels`` returns "the exact-label initializer
followed by reproducible neighbors" (`warm_start.py:1012`): index 0 is the
unjittered prediction, index 1 is the label-space neighbor production would
retry from if trial 0 failed to converge. Both are starts production is willing
to ship a result from, so the difference between their converged products is the
solver's own tolerance, measured in the same three metrics.

Index 0 is already on disk as the ``production_six_field`` arm -- that arm calls
``emulator_warm_start_model`` with no ``initializer_label``, which is the index-0
path. Only index 1 has to be solved here.

Warm starts are built inside the workers. Building them in the parent runs a
torch forward pass, and the fork-based pool created afterwards deadlocks --
workers load their catalogs and then freeze with CPU time no longer advancing.

Usage::

    export NUMBA_THREADING_LAYER=workqueue
    PYTHONPATH=. .venv/bin/python -m experiments.reduced_state_emulator.jitter_control \\
        --workers 30 --products-dir runs/reduced_state_emulator/products
"""

from __future__ import annotations

# Must precede any Numba import, matching bench/run_reference.py.
from bench import environment as _environment  # noqa: F401,E402

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench.labels import StellarLabels
from bench.report import summarize
from bench.run_reference import (
    PRODUCTION_INITIALIZER_JITTER_SCALE,
    PRODUCTION_INITIALIZER_SEED,
    PRODUCTION_MAX_TRIALS,
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
LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)
ARM = "production_jitter"


def _worker(payload):
    from payne_zero_atmosphere.warm_start import (
        deterministic_initializer_labels,
        emulator_warm_start_model,
    )
    from reduced_state.restart import run_restart_trial

    label_dict, trial_index, options = payload
    initializers = deterministic_initializer_labels(
        **label_dict,
        max_trials=PRODUCTION_MAX_TRIALS,
        seed=PRODUCTION_INITIALIZER_SEED,
        jitter_scale=PRODUCTION_INITIALIZER_JITTER_SCALE,
        device="cpu",
    )
    atmosphere, _deck = emulator_warm_start_model(
        **label_dict, device="cpu", initializer_label=initializers[trial_index]
    )
    return run_restart_trial(
        StellarLabels(**label_dict), atmosphere, **options
    ).as_json()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--held-out-from", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--trial-index", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument(
        "--products-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "reduced_state_emulator" / "products",
    )
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "reduced_state_emulator",
    )
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args(argv)

    held_out = np.array(
        json.loads(args.held_out_from.read_text())["star_indices"], dtype=np.int64
    )
    with np.load(args.corpus, allow_pickle=False) as data:
        labels_json = [json.loads(str(e)) for e in data["labels_json"][held_out]]
    label_dicts = [
        {field: float(entry[field]) for field in LABEL_FIELDS} for entry in labels_json
    ]
    if args.count is not None:
        label_dicts = label_dicts[: args.count]
    print(
        f"{len(label_dicts)} stars, production initializer trial "
        f"{args.trial_index} (jitter_scale={PRODUCTION_INITIALIZER_JITTER_SCALE}, "
        f"seed={PRODUCTION_INITIALIZER_SEED})",
        flush=True,
    )

    options = {
        "source": ARM,
        "product_dir": str(args.products_dir / ARM),
    }
    payloads = [(d, args.trial_index, options) for d in label_dicts]

    out_path = args.records_dir / ARM / "records.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a") as handle:
        if args.workers <= 1:
            iterator = (_worker(p) for p in payloads)
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers)
            iterator = executor.map(_worker, payloads)
        for index, record in enumerate(iterator, start=1):
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(payloads)}] {record['slug']} "
                f"converged={record['converged']} "
                f"iters={record['converging_trial_iterations']} "
                f"{record['seconds']:.1f}s",
                flush=True,
            )

    summary = {
        "arm": ARM,
        "trial_index": args.trial_index,
        "jitter_scale": PRODUCTION_INITIALIZER_JITTER_SCALE,
        "initializer_seed": PRODUCTION_INITIALIZER_SEED,
        "star_count": len(records),
        ARM: summarize(records),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    path = args.results_dir / "convergence_metrics_production_jitter.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}", flush=True)
    entry = summary[ARM]
    print(
        f"  {ARM:>22s}  converged={entry['converged_fraction']:.3f}  "
        f"iters(mean/p90)={entry['converging_trial_iterations']['mean']:.2f}/"
        f"{entry['converging_trial_iterations']['p90']:.0f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
