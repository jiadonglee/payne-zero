"""Measure the unmodified production atmosphere solve, per trial and iteration.

``payne_zero_atmosphere.cli.solve_structured_atmosphere`` runs the same loop but
returns only a path or raises, discarding the per-trial detail that the baseline
needs: which trial converged, how many iterations each one burned, and how the
residual moved. This module mirrors that loop (``cli.py:383-448``) with the
production settings unchanged, and records everything the runner already
computes.

The runner emits per-iteration diagnostics of its own accord
(``runner.py:1784-1802``): deep-layer and all-layer relative temperature change,
median/p95/max absolute flux error, and per-stage timings. Nothing is
recomputed here; this only captures and reshapes them.

Usage::

    python -m bench.run_reference --count 200 --out runs/baseline
    python -m bench.run_reference --labels my_labels.jsonl --out runs/baseline
"""

from __future__ import annotations

# Must precede any Numba import. See bench/environment.py.
from . import environment as _environment  # noqa: F401

import argparse
import dataclasses
import json
import math
import os
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from payne_zero_atmosphere.config import (
    AtmosphereConfig,
    AtmosphereInput,
    AtmosphereOutput,
)
from payne_zero_atmosphere.runner import run_atmosphere_model
from payne_zero_atmosphere.source_catalogs import (
    molecular_equilibrium_catalog_path,
    source_line_paths,
)
from payne_zero_atmosphere.warm_start import (
    deterministic_initializer_labels,
    emulator_warm_start_model,
)

from .labels import StellarLabels, load as load_labels, sample_uniform, save as save_labels


# The production policy, copied from cli.py:414-430 and cli.py:249-252. These
# are the numbers the baseline is defined against; changing one invalidates the
# comparison, so they live here as named constants rather than as defaults
# scattered through the call.
PRODUCTION_ITERATIONS_PER_TRIAL = 15
PRODUCTION_MAX_TRIALS = 2
PRODUCTION_MINIMUM_ITERATIONS = 3
PRODUCTION_REQUIRED_CONSECUTIVE = 1
PRODUCTION_MAXIMUM_DEEP_LAYER_CHANGE = 5.0e-4
PRODUCTION_INITIALIZER_SEED = 20260713
PRODUCTION_INITIALIZER_JITTER_SCALE = 0.01


@dataclass
class TrialRecord:
    """One initializer trial: what it cost and whether it converged."""

    trial_index: int
    initializer_label: dict[str, float] | None
    iterations_completed: int
    converged: bool
    seconds: float
    diagnostics: dict = field(default_factory=dict)
    error: str | None = None

    def as_json(self) -> dict:
        return {
            "trial_index": self.trial_index,
            "initializer_label": self.initializer_label,
            "iterations_completed": self.iterations_completed,
            "converged": self.converged,
            "seconds": self.seconds,
            "diagnostics": self.diagnostics,
            "error": self.error,
        }


@dataclass
class StarRecord:
    """One star's complete solve outcome across all trials."""

    labels: StellarLabels
    trials: list[TrialRecord]
    seconds: float
    warnings: list[str] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return any(trial.converged for trial in self.trials)

    @property
    def total_iterations(self) -> int:
        """Every physical iteration burned, including those of failed trials."""
        return sum(trial.iterations_completed for trial in self.trials)

    @property
    def converging_trial_iterations(self) -> int | None:
        """Iterations used by the trial that converged, or None if none did."""
        for trial in self.trials:
            if trial.converged:
                return trial.iterations_completed
        return None

    @property
    def needed_retry(self) -> bool:
        return len(self.trials) > 1

    def as_json(self) -> dict:
        return {
            "labels": self.labels.as_kwargs(),
            "slug": self.labels.slug,
            "converged": self.converged,
            "needed_retry": self.needed_retry,
            "trials_used": len(self.trials),
            "total_iterations": self.total_iterations,
            "converging_trial_iterations": self.converging_trial_iterations,
            "seconds": self.seconds,
            "warnings": self.warnings,
            "trials": [trial.as_json() for trial in self.trials],
        }


def _solver_config(
    warm_start_atmosphere,
    *,
    iterations_per_trial: int,
    structured_atmosphere_path: Path | None,
    debug_state_path: Path | None,
) -> AtmosphereConfig:
    """Build the production solver configuration for one trial."""

    return AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=warm_start_atmosphere,
            molecules_path=molecular_equilibrium_catalog_path(),
            **source_line_paths(),
        ),
        outputs=AtmosphereOutput(
            structured_atmosphere_path=structured_atmosphere_path,
            debug_state_path=debug_state_path,
        ),
        iterations=iterations_per_trial,
        enable_molecules=True,
        enable_convection=True,
        enable_convergence_stop=True,
        minimum_iterations_before_convergence=PRODUCTION_MINIMUM_ITERATIONS,
        required_consecutive_converged_iterations=PRODUCTION_REQUIRED_CONSECUTIVE,
        maximum_deep_layer_relative_temperature_change=(
            PRODUCTION_MAXIMUM_DEEP_LAYER_CHANGE
        ),
    )


def _atmosphere_is_finite(atmosphere) -> bool:
    """Check every array field of a ``ModelAtmosphere`` for non-finite values.

    The solver's structural convergence stop only inspects relative
    temperature change over a fixed layer window (deep interior); pressure,
    electron density, opacity, and column mass are never checked, so
    ``result.converged`` can be ``True`` while another field has gone
    non-finite. This is an independent, direct check of the full state,
    rather than relying on a downstream proxy such as product serialization
    failing.
    """

    for field_info in dataclasses.fields(atmosphere):
        value = getattr(atmosphere, field_info.name)
        if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
            return False
    return True


def run_star(
    labels: StellarLabels,
    *,
    iterations_per_trial: int = PRODUCTION_ITERATIONS_PER_TRIAL,
    max_trials: int = PRODUCTION_MAX_TRIALS,
    initializer_seed: int = PRODUCTION_INITIALIZER_SEED,
    jitter_scale: float = PRODUCTION_INITIALIZER_JITTER_SCALE,
    trace_dir: Path | None = None,
    keep_product: bool = False,
) -> StarRecord:
    """Solve one star on the production path, recording every trial.

    Mirrors ``cli.py:383-448``: trials stop at the first converged one, and a
    trial that fails to converge is followed by the next jittered initializer.

    ``trace_dir`` turns on the runner's own per-iteration state dump
    (``runner.py:1367``), which is what Stage 1 validates the torch twin
    against. It is off by default because it is large and slow.
    """

    star_start = time.perf_counter()
    initializers = deterministic_initializer_labels(
        **labels.as_kwargs(),
        max_trials=max_trials,
        seed=int(initializer_seed),
        jitter_scale=float(jitter_scale),
        device="cpu",
    )

    trials: list[TrialRecord] = []
    captured_warnings: list[str] = []

    for trial_index, initializer_label in enumerate(initializers):
        trial_start = time.perf_counter()
        structured_path = None
        debug_path = None
        if trace_dir is not None:
            trial_dir = Path(trace_dir) / labels.slug / f"trial_{trial_index:02d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            debug_path = trial_dir / "debug_state.npz"
            if keep_product:
                structured_path = trial_dir / "payne_zero_structured_atmosphere.npz"

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                warm_start_atmosphere, _deck = emulator_warm_start_model(
                    **labels.as_kwargs(),
                    device="cpu",
                    initializer_label=initializer_label,
                )
                result = run_atmosphere_model(
                    _solver_config(
                        warm_start_atmosphere,
                        iterations_per_trial=iterations_per_trial,
                        structured_atmosphere_path=structured_path,
                        debug_state_path=debug_path,
                    )
                )
            captured_warnings.extend(str(w.message) for w in caught)
        except Exception as exc:  # noqa: BLE001 - a failed star is a data point
            trials.append(
                TrialRecord(
                    trial_index=trial_index,
                    initializer_label=_as_plain(initializer_label),
                    iterations_completed=0,
                    converged=False,
                    seconds=time.perf_counter() - trial_start,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )
            continue

        trial_converged = bool(result.converged)
        if trial_converged and not _atmosphere_is_finite(result.atmosphere):
            trial_converged = False
            captured_warnings.append(
                "solver reported convergence but the atmosphere state contains "
                "non-finite values; counted as not converged"
            )
        trials.append(
            TrialRecord(
                trial_index=trial_index,
                initializer_label=_as_plain(initializer_label),
                iterations_completed=int(result.iterations_completed),
                converged=trial_converged,
                seconds=time.perf_counter() - trial_start,
                diagnostics=_as_plain(result.diagnostics),
            )
        )
        if trial_converged:
            break

    return StarRecord(
        labels=labels,
        trials=trials,
        seconds=time.perf_counter() - star_start,
        warnings=captured_warnings,
    )


def _as_plain(value):
    """Convert numpy scalars and Paths so the record is standard JSON.

    Non-finite floats become ``null``. A diverged solve genuinely produces NaN
    flux errors, and ``json.dumps`` would happily write the bare tokens ``NaN``
    and ``Infinity`` — which Python reads back but ``jq``, ``pandas``, and most
    other tools reject. Nothing is lost: ``converged`` already records that the
    star failed, and ``error`` holds any exception.
    """

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return _as_plain(value.item())
    if hasattr(value, "tolist"):
        return _as_plain(value.tolist())
    return str(value)


def _worker(payload):
    """Process-pool entry point: solve one star and return its JSON record."""

    labels, options = payload
    return run_star(labels, **options).as_json()


def run_many(
    labels: list[StellarLabels],
    *,
    workers: int = 1,
    out_path: Path | None = None,
    **options,
) -> list[dict]:
    """Solve a label set, appending each record to ``out_path`` as it lands.

    Records stream to disk so that a run killed partway is still usable — these
    solves take tens of seconds each and a full baseline is hours.
    """

    handle = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        handle = out_path.open("a")

    records: list[dict] = []
    executor = None
    try:
        if workers <= 1:
            iterator = (run_star(item, **options).as_json() for item in labels)
        else:
            from concurrent.futures import ProcessPoolExecutor

            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(_worker, [(item, options) for item in labels])

        for index, record in enumerate(iterator, start=1):
            records.append(record)
            if handle is not None:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
            print(
                f"[{index}/{len(labels)}] {record['slug']} "
                f"converged={record['converged']} "
                f"iters={record['converging_trial_iterations']} "
                f"trials={record['trials_used']} "
                f"{record['seconds']:.1f}s",
                flush=True,
            )
    finally:
        if handle is not None:
            handle.close()
        if executor is not None:
            executor.shutdown()

    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.run_reference",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--count", type=int, help="sample this many labels uniformly")
    source.add_argument("--labels", type=Path, help="JSON or JSON Lines label file")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="label sampling seed")
    parser.add_argument("--workers", type=int, default=1, help="parallel solver processes")
    parser.add_argument(
        "--iterations-per-trial", type=int, default=PRODUCTION_ITERATIONS_PER_TRIAL
    )
    parser.add_argument("--max-trials", type=int, default=PRODUCTION_MAX_TRIALS)
    parser.add_argument(
        "--traces",
        action="store_true",
        help="dump per-iteration solver state for twin validation (large)",
    )
    parser.add_argument(
        "--keep-product",
        action="store_true",
        help="also write the structured atmosphere product for each trial",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.labels is not None:
        labels = load_labels(args.labels)
    else:
        labels = sample_uniform(args.count, seed=args.seed)
    save_labels(labels, out_dir / "labels.jsonl")

    options = {
        "iterations_per_trial": args.iterations_per_trial,
        "max_trials": args.max_trials,
        "trace_dir": (out_dir / "traces") if args.traces else None,
        "keep_product": args.keep_product,
    }

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "star_count": len(labels),
                "seed": args.seed,
                "workers": args.workers,
                "numba_threading_layer": os.environ.get("NUMBA_THREADING_LAYER"),
                "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS"),
                "options": {k: str(v) for k, v in options.items()},
                "production_policy": {
                    "minimum_iterations_before_convergence": PRODUCTION_MINIMUM_ITERATIONS,
                    "required_consecutive_converged_iterations": PRODUCTION_REQUIRED_CONSECUTIVE,
                    "maximum_deep_layer_relative_temperature_change": PRODUCTION_MAXIMUM_DEEP_LAYER_CHANGE,
                    "initializer_seed": PRODUCTION_INITIALIZER_SEED,
                    "initializer_jitter_scale": PRODUCTION_INITIALIZER_JITTER_SCALE,
                },
            },
            indent=2,
        )
        + "\n"
    )

    started = time.perf_counter()
    run_many(
        labels,
        workers=args.workers,
        out_path=out_dir / "records.jsonl",
        **options,
    )
    print(f"done in {time.perf_counter() - started:.1f}s -> {out_dir / 'records.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
