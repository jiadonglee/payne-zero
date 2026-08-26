"""The two truth arms, re-run with converged products saved for the spectral gate.

Part 2 established that truth `(m,T)` restarts the solver as well as the full
six-field truth does, measured in iterations. That is the *convergence-rate*
half of the sufficiency claim. This run supplies the other half: the converged
structured atmospheres, so their spectra can be compared. If starting from
`(m,T)` alone and starting from all six truth fields land on spectroscopically
identical answers, `(m,T)` carries the information the solver needs -- stated in
the observable, not in the atmosphere parameterization.

``experiments/reduced_state_parity/run_oracle_parity.py`` cannot be used with
``--products-dir`` for this. Its ``build_restart_pairs`` calls
``emulator_warm_start_model`` in the parent, which initializes torch; the
fork-based pool created afterwards then deadlocks the moment a worker reaches
the solver's structured-product writer. Every atmosphere here is therefore built
inside its worker, and this process imports no torch at module level.

The failure signature, for whoever hits it next: workers load their catalogs
(~5 min CPU, ~15 GB RSS) and then freeze -- ``ps`` shows elapsed time advancing
while ``TIME`` does not, and no product is ever written.

Usage::

    export NUMBA_THREADING_LAYER=workqueue
    PYTHONPATH=. .venv/bin/python -m experiments.reduced_state_emulator.truth_arms \\
        --workers 26 --products-dir runs/reduced_state_emulator/products
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
# Column order of atmosphere_profiles in the corpus.
PROFILE_FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)
N_SYNCHRONIZATIONS = 3
MAX_SYNCHRONIZATIONS = 8
PRESSURE_TOLERANCE_DEX = 1.0e-3


def _reduced_worker(payload):
    """Truth (m,T) -> reconstruct the other four -> restart."""

    from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere
    from reduced_state.restart import run_restart_trial

    profile, label_dict, options = payload
    reduced = ReducedAtmosphere(
        column_mass=profile[:, 0], temperature=profile[:, 1], labels=label_dict
    )
    result = reconstruct_full_atmosphere(
        reduced,
        n_synchronizations=None,
        max_synchronizations=MAX_SYNCHRONIZATIONS,
        pressure_tolerance_dex=PRESSURE_TOLERANCE_DEX,
    )
    return run_restart_trial(
        StellarLabels(**label_dict), result.atmosphere, **options
    ).as_json()


def _full_truth_worker(payload):
    """All six truth fields pinned onto a warm start -> restart.

    Identical construction to ``run_oracle_parity.build_restart_pairs``, moved
    into the worker so the parent never initializes torch.
    """

    import dataclasses

    from payne_zero_atmosphere.warm_start import emulator_warm_start_model
    from reduced_state.restart import run_restart_trial

    profile, label_dict, options = payload
    warm_start, _deck = emulator_warm_start_model(device="cpu", **label_dict)
    atmosphere = dataclasses.replace(
        warm_start,
        **{field: profile[:, index] for index, field in enumerate(PROFILE_FIELDS)},
    )
    return run_restart_trial(
        StellarLabels(**label_dict), atmosphere, **options
    ).as_json()


def run_arm(worker, payloads, *, workers: int, out_path: Path, name: str) -> list:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a") as handle:
        if workers <= 1:
            iterator = (worker(p) for p in payloads)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(worker, payloads)
        for index, record in enumerate(iterator, start=1):
            records.append(record)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                f"[{name} {index}/{len(payloads)}] {record['slug']} "
                f"converged={record['converged']} "
                f"iters={record['converging_trial_iterations']} "
                f"{record['seconds']:.1f}s",
                flush=True,
            )
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--held-out-from", type=Path, default=DEFAULT_HELD_OUT)
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
        profiles = np.asarray(data["atmosphere_profiles"][held_out], dtype=np.float64)
    label_dicts = [
        {field: float(entry[field]) for field in LABEL_FIELDS} for entry in labels_json
    ]
    if args.count is not None:
        label_dicts = label_dicts[: args.count]
        profiles = profiles[: args.count]
    print(f"{len(label_dicts)} held-out stars", flush=True)

    summary = {
        "star_count": len(label_dicts),
        "synchronization_mode": "adaptive",
        "max_synchronizations": MAX_SYNCHRONIZATIONS,
        "pressure_tolerance_dex": PRESSURE_TOLERANCE_DEX,
    }
    for name, worker in (
        ("reduced_state_reconstruction", _reduced_worker),
        ("full_truth_oracle", _full_truth_worker),
    ):
        options = {"source": name, "product_dir": str(args.products_dir / name)}
        payloads = [
            (profiles[i], label_dicts[i], options) for i in range(len(label_dicts))
        ]
        records = run_arm(
            worker, payloads, workers=args.workers,
            out_path=args.records_dir / name / "records.jsonl", name=name,
        )
        summary[name] = summarize(records)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    path = args.results_dir / "convergence_metrics_truth_arms.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}", flush=True)
    for name in ("reduced_state_reconstruction", "full_truth_oracle"):
        entry = summary[name]
        print(
            f"  {name:>30s}  converged={entry['converged_fraction']:.3f}  "
            f"iters(mean/p90)={entry['converging_trial_iterations']['mean']:.2f}/"
            f"{entry['converging_trial_iterations']['p90']:.0f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
