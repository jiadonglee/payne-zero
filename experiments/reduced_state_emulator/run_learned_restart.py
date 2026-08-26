"""Part 4: does a *learned* (m,T) still land in the solver's basin?

Part 2 restarted the solver from reconstructions of **truth** (m,T) and got
100% convergence at mean 3.33 iterations. That is an oracle: it assumes the
profiles are exact. This script substitutes the trained predictor from
``experiments.reduced_state_emulator.train`` and re-runs the identical
pipeline, so the only thing that changed is where (m,T) came from.

Four arms, on the same 60 held-out stars:

===========================  ==========================================
``learned_reduced_state``    trained net -> reconstruct -> restart   (this run)
``production_six_field``     release emulator warm start -> restart  (this run)
``reduced_state_reconstruction``  truth (m,T) -> reconstruct -> restart  (Part 2)
``full_truth_oracle``        truth six fields -> restart             (Part 2)
===========================  ==========================================

The middle two are the comparison that decides deployability: the learned
reduced state has to beat the *shipped* initializer, not just approach the
oracle. Ting's Sec 2.2 basin calibration is the reason this cannot be settled
from profile error alone -- the same marginal errors converged separately and
failed at 30 iterations when coupled, and sign flipped the outcome. Only the
solver answers.

**Run the two arms as two separate invocations.** Doing both in one process
deadlocks the second arm's worker pool: ``production_warm_starts`` runs a torch
forward in the parent, and the fork-based ``ProcessPoolExecutor`` created after
it inherits an unusable OpenMP state. The signature is workers that load their
catalogs (~5 min of CPU, ~15 GB RSS) and then freeze with their CPU time no
longer advancing. Observed twice; do not merge the arms back into one run.

Usage::

    export NUMBA_THREADING_LAYER=workqueue
    PYTHONPATH=. .venv/bin/python -m experiments.reduced_state_emulator.predict --arm monotone
    PYTHONPATH=. .venv/bin/python -m experiments.reduced_state_emulator.run_learned_restart \\
        --arm monotone --workers 30 --skip-production-arm --products-dir runs/.../products
    PYTHONPATH=. .venv/bin/python -m experiments.reduced_state_emulator.run_learned_restart \\
        --production-only --workers 30 --products-dir runs/.../products
"""

from __future__ import annotations

# Must precede any Numba import, matching bench/run_reference.py.
from bench import environment as _environment  # noqa: F401,E402

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench.labels import StellarLabels
from bench.report import summarize
from reduced_state.reconstruct import (
    ReducedAtmosphere,
    ReconstructionConvergenceError,
    reconstruct_full_atmosphere,
)
from reduced_state.restart import run_many_restarts

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
DEFAULT_HELD_OUT = REPO_ROOT / "results" / "reconstruction_metrics.json"
N_SYNCHRONIZATIONS = 3
MAX_SYNCHRONIZATIONS = 8
PRESSURE_TOLERANCE_DEX = 1.0e-3

# Declared here rather than imported from reduced_state.emulator: that module
# imports torch, and a torch forward pass in this process would deadlock the
# fork-based ProcessPoolExecutor below. See experiments/.../predict.py.
LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)


def _label_subset(row: np.ndarray) -> dict:
    return {field: float(value) for field, value in zip(LABEL_FIELDS, row)}


def _reconstruct_worker(payload):
    column_mass, temperature, labels, max_synchronizations = payload
    reduced = ReducedAtmosphere(
        column_mass=column_mass, temperature=temperature, labels=labels
    )
    result = reconstruct_full_atmosphere(
        reduced,
        n_synchronizations=None,
        max_synchronizations=max_synchronizations,
        pressure_tolerance_dex=PRESSURE_TOLERANCE_DEX,
    )
    return result.atmosphere


def reconstruct_predicted(
    column_mass: np.ndarray,
    temperature: np.ndarray,
    labels: np.ndarray,
    *,
    workers: int,
    max_synchronizations: int = MAX_SYNCHRONIZATIONS,
) -> list:
    payloads = [
        (
            column_mass[i],
            temperature[i],
            _label_subset(labels[i]),
            int(max_synchronizations),
        )
        for i in range(len(labels))
    ]
    if workers <= 1:
        return [_reconstruct_worker(p) for p in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_reconstruct_worker, payloads))


def _reconstruct_worker_safe(payload):
    """Return one atmosphere or a serializable per-star reconstruction failure."""

    position, column_mass, temperature, labels, max_synchronizations = payload
    try:
        atmosphere = _reconstruct_worker(
            (column_mass, temperature, labels, max_synchronizations)
        )
    except ReconstructionConvergenceError as exc:
        return {
            "position": int(position),
            "atmosphere": None,
            "failure": {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "pressure_change_dex_by_pass": list(
                    exc.pressure_change_dex_by_pass
                ),
            },
        }
    except Exception as exc:  # preserve unexpected failures for the summary
        return {
            "position": int(position),
            "atmosphere": None,
            "failure": {
                "error_type": type(exc).__name__,
                "error": repr(exc),
                "pressure_change_dex_by_pass": [],
            },
        }
    return {"position": int(position), "atmosphere": atmosphere, "failure": None}


def reconstruct_predicted_safe(
    column_mass: np.ndarray,
    temperature: np.ndarray,
    labels: np.ndarray,
    *,
    workers: int,
    max_synchronizations: int = MAX_SYNCHRONIZATIONS,
) -> tuple[list[tuple[int, object]], list[dict]]:
    """Reconstruct all stars while retaining failures for an auditable report."""

    payloads = [
        (
            i,
            column_mass[i],
            temperature[i],
            _label_subset(labels[i]),
            int(max_synchronizations),
        )
        for i in range(len(labels))
    ]
    if workers <= 1:
        rows = [_reconstruct_worker_safe(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_reconstruct_worker_safe, payloads))
    successes = [
        (int(row["position"]), row["atmosphere"])
        for row in rows
        if row["failure"] is None
    ]
    failures = [
        {"position": int(row["position"]), **row["failure"]}
        for row in rows
        if row["failure"] is not None
    ]
    return successes, failures


def _production_worker(payload):
    """Build the warm start *and* restart from it, entirely inside the worker.

    The warm start must not be built in the parent. ``emulator_warm_start_model``
    runs a torch forward pass, and a fork-based ``ProcessPoolExecutor`` created
    afterwards inherits an unusable OpenMP state: workers load their catalogs
    (~5 min CPU, ~15 GB RSS) and then freeze with CPU time no longer advancing
    while elapsed time keeps climbing. Confirmed by sampling ``ps`` twice 45 s
    apart. This mirrors what the learned arm already does -- its atmospheres are
    built inside ``reconstruct_full_atmosphere``, in the worker.
    """

    from payne_zero_atmosphere.warm_start import emulator_warm_start_model
    from reduced_state.restart import run_restart_trial

    label_dict, options = payload
    atmosphere, _deck = emulator_warm_start_model(device="cpu", **label_dict)
    return run_restart_trial(
        StellarLabels(**label_dict), atmosphere, **options
    ).as_json()


def production_warm_start_profiles(labels: np.ndarray) -> tuple:
    """(m, T) of the shipped initializer, for the profile-error table only.

    Runs in a throwaway subprocess so this process never initializes torch.
    """

    import subprocess
    import sys
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as handle:
        out = Path(handle.name)
    script = (
        "import json,sys,numpy as np\n"
        "from payne_zero_atmosphere.warm_start import emulator_warm_start_model\n"
        "rows=json.loads(sys.argv[1]); m=[];t=[]\n"
        "for r in rows:\n"
        "    a,_=emulator_warm_start_model(device='cpu',**r)\n"
        "    m.append(a.column_mass); t.append(a.temperature)\n"
        "np.savez(sys.argv[2], column_mass=np.array(m), temperature=np.array(t))\n"
    )
    subprocess.run(
        [sys.executable, "-c", script,
         json.dumps([_label_subset(row) for row in labels]), str(out)],
        check=True,
    )
    with np.load(out, allow_pickle=False) as data:
        result = (data["column_mass"], data["temperature"])
    out.unlink()
    return result


def run_production_arm(label_dicts, options, *, workers, out_path):
    """Restart from worker-built production warm starts, streaming records."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payloads = [(d, options) for d in label_dicts]
    records = []
    with out_path.open("a") as handle:
        if workers <= 1:
            iterator = (_production_worker(p) for p in payloads)
        else:
            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(_production_worker, payloads)
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
    return records


def profile_errors(
    predicted_column_mass: np.ndarray,
    predicted_temperature: np.ndarray,
    truth_column_mass: np.ndarray,
    truth_temperature: np.ndarray,
) -> dict:
    temperature_relative = np.abs(
        predicted_temperature - truth_temperature
    ) / truth_temperature
    column_mass_dex = np.abs(
        np.log10(predicted_column_mass) - np.log10(truth_column_mass)
    )
    return {
        "temperature_relative_p50": float(np.median(temperature_relative)),
        "temperature_relative_p95": float(np.percentile(temperature_relative, 95)),
        "column_mass_dex_p50": float(np.median(column_mass_dex)),
        "column_mass_dex_p95": float(np.percentile(column_mass_dex, 95)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--held-out-from", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reduced_state_emulator",
    )
    parser.add_argument(
        "--arm",
        default="monotone",
        choices=("monotone", "direct", "physical_ensemble"),
    )
    parser.add_argument(
        "--prediction",
        type=Path,
        default=None,
        help="explicit prediction artifact; overrides --checkpoint-dir/--arm",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--max-synchronizations",
        type=int,
        default=MAX_SYNCHRONIZATIONS,
        help=(
            "diagnostic pressure-synchronization pass limit; the default "
            "production limit is 8"
        ),
    )
    parser.add_argument(
        "--minimum-iterations-before-convergence",
        type=int,
        default=None,
        help=(
            "diagnostic override for restart trials only; the default production "
            "policy is unchanged"
        ),
    )
    parser.add_argument(
        "--iterations-per-trial",
        type=int,
        default=None,
        help=(
            "diagnostic override for the solver iteration budget; the default "
            "production budget is unchanged"
        ),
    )
    parser.add_argument(
        "--maximum-all-layer-relative-temperature-change",
        type=float,
        default=None,
        help=(
            "diagnostic override that also requires the upper layers to meet "
            "this relative-temperature limit before stopping"
        ),
    )
    parser.add_argument(
        "--write-unconverged-products",
        action="store_true",
        help=(
            "diagnostic only: save the last atmosphere even when the convergence "
            "criterion is not met; never enabled by default"
        ),
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help=(
            "gate/solve only the first N held-out stars. For local pipeline "
            "checks; the reported metrics are only comparable at the full 60."
        ),
    )
    parser.add_argument(
        "--positions",
        default=None,
        help=(
            "diagnostic comma-separated positions in the held-out manifest; "
            "cannot be combined with --count"
        ),
    )
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "reduced_state_emulator",
    )
    parser.add_argument(
        "--skip-production-arm",
        action="store_true",
        help="reuse an earlier production_six_field arm instead of re-solving it",
    )
    parser.add_argument(
        "--products-dir",
        type=Path,
        default=None,
        help=(
            "write each converged structured atmosphere here, per arm. Required "
            "for the spectral gate, which synthesizes from the converged product."
        ),
    )
    parser.add_argument(
        "--production-only",
        action="store_true",
        help=(
            "run only the production six-field baseline. It does not depend on "
            "the trained checkpoint, so it can be launched while training is "
            "still running."
        ),
    )
    args = parser.parse_args(argv)

    if args.production_only and args.skip_production_arm:
        raise SystemExit("--production-only and --skip-production-arm are contradictory")
    if args.iterations_per_trial is not None and args.iterations_per_trial < 1:
        raise SystemExit("--iterations-per-trial must be positive")
    if args.max_synchronizations < 1:
        raise SystemExit("--max-synchronizations must be positive")

    held_out = np.array(
        json.loads(args.held_out_from.read_text())["star_indices"], dtype=np.int64
    )
    with np.load(args.corpus, allow_pickle=False) as data:
        labels_json = [json.loads(str(e)) for e in data["labels_json"][held_out]]
        profiles = np.asarray(data["atmosphere_profiles"][held_out], dtype=np.float64)
    labels = np.array(
        [[entry[field] for field in LABEL_FIELDS] for entry in labels_json],
        dtype=np.float64,
    )
    truth_column_mass = profiles[:, :, 0]
    truth_temperature = profiles[:, :, 1]

    if args.count is not None:
        keep = slice(0, args.count)
        held_out = held_out[keep]
        labels = labels[keep]
        truth_column_mass = truth_column_mass[keep]
        truth_temperature = truth_temperature[keep]
        print(f"LIMITED to the first {len(held_out)} stars (--count)", flush=True)
    selected_positions = None
    if args.positions is not None:
        if args.count is not None:
            raise SystemExit("--positions and --count are mutually exclusive")
        try:
            selected_positions = np.asarray(
                [int(value.strip()) for value in args.positions.split(",")],
                dtype=np.int64,
            )
        except ValueError as exc:
            raise SystemExit("--positions must be comma-separated integers") from exc
        if selected_positions.size == 0 or np.any(selected_positions < 0):
            raise SystemExit("--positions must contain at least one non-negative position")
        if np.any(selected_positions >= len(held_out)):
            raise SystemExit("--positions contains a position outside the held-out manifest")
        held_out = held_out[selected_positions]
        labels = labels[selected_positions]
        truth_column_mass = truth_column_mass[selected_positions]
        truth_temperature = truth_temperature[selected_positions]
        print(
            f"LIMITED to held-out positions {selected_positions.tolist()} "
            "(--positions)",
            flush=True,
        )
    print(f"{len(held_out)} held-out stars", flush=True)

    summary: dict = {
        "arm": args.arm,
        "star_count": int(len(held_out)),
        "star_indices": [int(i) for i in held_out],
        "synchronization_mode": "adaptive",
        "max_synchronizations": int(args.max_synchronizations),
        "pressure_tolerance_dex": PRESSURE_TOLERANCE_DEX,
        "iterations_per_trial": args.iterations_per_trial,
        "minimum_iterations_before_convergence": (
            args.minimum_iterations_before_convergence
        ),
        "maximum_all_layer_relative_temperature_change": (
            args.maximum_all_layer_relative_temperature_change
        ),
        "write_unconverged_products": bool(args.write_unconverged_products),
        "profile_errors": {},
    }

    if not args.production_only:
        prediction_path = args.prediction or (
            args.checkpoint_dir / f"predicted_{args.arm}.npz"
        )
        if not prediction_path.is_file():
            raise SystemExit(
                f"{prediction_path} not found; run "
                f"`python -m experiments.reduced_state_emulator.predict "
                f"--arm {args.arm}` first"
            )
        with np.load(prediction_path, allow_pickle=False) as prediction:
            prediction_positions = (
                selected_positions
                if selected_positions is not None
                else np.arange(len(held_out), dtype=np.int64)
            )
            if not np.array_equal(
                prediction["star_indices"][prediction_positions], held_out
            ):
                raise SystemExit(
                    "prediction artifact was built for a different star set"
                )
            predicted_column_mass = prediction["column_mass"][prediction_positions]
            predicted_temperature = prediction["temperature"][prediction_positions]
        print(f"loaded {prediction_path}", flush=True)
        summary["prediction_artifact"] = str(prediction_path)

        non_monotone = int(
            (np.diff(predicted_column_mass, axis=1) <= 0).any(axis=1).sum()
        )
        if non_monotone:
            raise SystemExit(
                f"{non_monotone}/{len(held_out)} predicted column-mass profiles are "
                "non-monotonic; reconstruct.py will reject them"
            )

        summary["profile_errors"]["learned_reduced_state"] = profile_errors(
            predicted_column_mass,
            predicted_temperature,
            truth_column_mass,
            truth_temperature,
        )

        print("reconstructing from learned (m,T)...", flush=True)
        reconstructed, reconstruction_failures = reconstruct_predicted_safe(
            predicted_column_mass,
            predicted_temperature,
            labels,
            workers=args.workers,
            max_synchronizations=args.max_synchronizations,
        )
        for failure in reconstruction_failures:
            failure["star_index"] = int(held_out[failure["position"]])
        summary["learned_reduced_state_reconstruction"] = {
            "requested_count": int(len(held_out)),
            "synchronized_count": int(len(reconstructed)),
            "failure_count": int(len(reconstruction_failures)),
            "max_synchronizations": int(args.max_synchronizations),
            "pressure_tolerance_dex": PRESSURE_TOLERANCE_DEX,
            "failures": reconstruction_failures,
        }
        if not reconstructed:
            raise SystemExit("no star reached a synchronized reconstructed atmosphere")
        learned_items = [
            (StellarLabels(**_label_subset(labels[position])), atmosphere)
            for position, atmosphere in reconstructed
        ]

        print("restarting the solver from learned reduced state...", flush=True)
        learned_records = run_many_restarts(
            learned_items,
            workers=args.workers,
            out_path=args.records_dir / "learned_reduced_state" / "records.jsonl",
            source="learned_reduced_state",
            iterations_per_trial=(
                args.iterations_per_trial
                if args.iterations_per_trial is not None
                else None
            ),
            write_unconverged_product=args.write_unconverged_products,
            minimum_iterations_before_convergence=(
                args.minimum_iterations_before_convergence
            ),
            maximum_all_layer_relative_temperature_change=(
                args.maximum_all_layer_relative_temperature_change
            ),
            product_dir=(
                args.products_dir / "learned_reduced_state"
                if args.products_dir
                else None
            ),
        )
        summary["learned_reduced_state"] = summarize(learned_records)

    if not args.skip_production_arm:
        print("production warm-start profiles (subprocess)...", flush=True)
        production_m, production_t = production_warm_start_profiles(labels)
        summary["profile_errors"]["production_six_field"] = profile_errors(
            production_m, production_t, truth_column_mass, truth_temperature
        )

        print("restarting the solver from the production initializer...", flush=True)
        options = {
            "source": "production_six_field",
            "iterations_per_trial": (
                args.iterations_per_trial
                if args.iterations_per_trial is not None
                else None
            ),
            "write_unconverged_product": args.write_unconverged_products,
            "minimum_iterations_before_convergence": (
                args.minimum_iterations_before_convergence
            ),
            "maximum_all_layer_relative_temperature_change": (
                args.maximum_all_layer_relative_temperature_change
            ),
            "product_dir": (
                str(args.products_dir / "production_six_field")
                if args.products_dir
                else None
            ),
        }
        production_records = run_production_arm(
            [_label_subset(row) for row in labels],
            options,
            workers=args.workers,
            out_path=args.records_dir / "production_six_field" / "records.jsonl",
        )
        summary["production_six_field"] = summarize(production_records)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stem = "production_baseline" if args.production_only else f"learned_{args.arm}"
    out_path = args.results_dir / f"convergence_metrics_{stem}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", flush=True)

    for tag in ("learned_reduced_state", "production_six_field"):
        entry = summary.get(tag)
        if entry is None:
            continue
        print(
            f"  {tag:>24s}  converged={entry['converged_fraction']:.3f}  "
            f"iters(mean/p90)={entry['converging_trial_iterations']['mean']:.2f}/"
            f"{entry['converging_trial_iterations']['p90']:.0f}  "
            f"geomean_q={entry['contraction']['q_ratio']['geometric_mean']:.3f}  "
            f"non_monotonic={entry['contraction']['non_monotonic_fraction']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
