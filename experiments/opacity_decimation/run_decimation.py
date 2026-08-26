"""Experiment A: frequency-grid decimation of the opacity-sampling grid.

The production solve spends 79.1% of its wall time in the opacity stage
(``results/solver_in_loop_k1_hard5_linear/real_solver_comparison.json``: five
iterations, 13.128 s total, 10.379 s opacity), and that cost is linear in the
number of sampled frequencies -- 30000 of them, equally spaced in log10
wavelength at 1e-4 dex. This runs one star at several strides through that grid
and records what each stride costs and what it changes.

The point is the error budget, not the speedup: how much kappa_nu precision the
converged atmosphere actually needs decides whether an opacity emulator is
worth building. The comparison metric is therefore the solver's own yardstick,
``deep_layer_relative_temperature_change`` between the stride-N and stride-1
converged temperatures, read against the 5e-4 convergence threshold.

Everything except the stride comes from ``bench.run_reference``, unchanged: the
production trial policy, the deterministic initializer, and the record format.
The stride is injected by wrapping ``_solver_config``, so no production
constant is restated here.

Usage::

    PYTHONPATH=. .venv/bin/python -m experiments.opacity_decimation.run_decimation \
        --strides 1 2 4 8 --out runs/opacity_decimation
    PYTHONPATH=. .venv/bin/python -m experiments.opacity_decimation.run_decimation \
        --report runs/opacity_decimation
"""

from __future__ import annotations

# Must precede any Numba import. See bench/environment.py.
from bench import environment as _environment  # noqa: F401

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

from bench import run_reference
from bench.labels import StellarLabels
from payne_zero_atmosphere.convergence import (
    deep_layer_relative_temperature_change,
    max_normalized_column_delta,
)


# The star the 79.1% opacity share was measured on
# (results/solver_in_loop_k1_hard5_linear/real_solver_comparison.json).
REFERENCE_STAR = StellarLabels(
    effective_temperature=7650.8,
    log_surface_gravity=1.79,
    metallicity=-1.93,
    alpha_enhancement=0.36,
    microturbulence_km_s=2.06,
)
DEFAULT_STRIDES = (1, 2, 4, 8)
REFERENCE_STRIDE = 1


@contextmanager
def solver_configured_with_stride(stride: int):
    """Run ``bench.run_reference`` with one config field changed, and capture.

    ``run_star`` builds its own config and discards the solved atmosphere, so
    both are intercepted here rather than reimplemented: ``_solver_config`` is
    wrapped so that the production policy is produced first and only the stride
    is replaced on it, and ``run_atmosphere_model`` is wrapped to keep the last
    result, whose atmosphere is the converged state (trials stop at the first
    converged one).
    """

    original_solver_config = run_reference._solver_config
    original_run_atmosphere_model = run_reference.run_atmosphere_model
    captured: dict = {}

    def strided_solver_config(*args, **kwargs):
        return replace(
            original_solver_config(*args, **kwargs),
            opacity_frequency_grid_stride=int(stride),
        )

    def capturing_run_atmosphere_model(config):
        result = original_run_atmosphere_model(config)
        captured["result"] = result
        return result

    run_reference._solver_config = strided_solver_config
    run_reference.run_atmosphere_model = capturing_run_atmosphere_model
    try:
        yield captured
    finally:
        run_reference._solver_config = original_solver_config
        run_reference.run_atmosphere_model = original_run_atmosphere_model


def stride_directory(out_dir: Path, stride: int) -> Path:
    return Path(out_dir) / f"stride_{int(stride):02d}"


def run_one_stride(
    labels: StellarLabels,
    stride: int,
    *,
    out_dir: Path,
    iterations_per_trial: int,
    max_trials: int,
) -> dict:
    """Solve one star at one stride and write its record and converged state."""

    target = stride_directory(out_dir, stride)
    target.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    with solver_configured_with_stride(stride) as captured:
        star = run_reference.run_star(
            labels,
            iterations_per_trial=iterations_per_trial,
            max_trials=max_trials,
        )
    wall_seconds = time.perf_counter() - started

    record = star.as_json()
    record["opacity_frequency_grid_stride"] = int(stride)
    record["wall_seconds"] = float(wall_seconds)
    (target / "record.json").write_text(json.dumps(record, indent=2) + "\n")

    result = captured.get("result")
    if result is not None:
        np.savez(
            target / "converged_state.npz",
            temperature=np.asarray(result.atmosphere.temperature, dtype=np.float64),
            column_mass=np.asarray(result.atmosphere.column_mass, dtype=np.float64),
            opacity_frequency_grid_stride=np.int64(stride),
            iterations_completed=np.int64(result.iterations_completed),
            converged=np.bool_(result.converged),
        )

    for trial in record["trials"]:
        reported = trial.get("diagnostics", {}).get("opacity_frequency_grid_stride")
        if reported is not None and int(reported) != int(stride):
            raise RuntimeError(
                f"solver ran at stride {reported}, not the requested {stride}"
            )
    return record


def _stride_summary(out_dir: Path, stride: int) -> dict | None:
    """Reduce one stride's on-disk output to the row the experiment needs."""

    target = stride_directory(out_dir, stride)
    record_path = target / "record.json"
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())

    converging = next(
        (trial for trial in record["trials"] if trial["converged"]),
        record["trials"][-1] if record["trials"] else None,
    )
    diagnostics = (converging or {}).get("diagnostics", {})
    timings = diagnostics.get("iteration_timings", [])

    opacity_seconds = [float(step["opacity_seconds"]) for step in timings]
    row = {
        "opacity_frequency_grid_stride": int(stride),
        "frequency_count": diagnostics.get("frequency_count"),
        "converged": bool(record["converged"]),
        "trials_used": int(record["trials_used"]),
        "iterations_to_convergence": record["converging_trial_iterations"],
        "deep_layer_relative_temperature_change": diagnostics.get(
            "deep_layer_relative_temperature_change"
        ),
        "opacity_seconds": opacity_seconds,
        "opacity_seconds_total": float(sum(opacity_seconds)),
        "solver_total_seconds": diagnostics.get("total_seconds"),
        "wall_seconds": record.get("wall_seconds"),
        "warnings": record.get("warnings", []),
    }
    if opacity_seconds and row["solver_total_seconds"]:
        row["opacity_fraction_of_total"] = row["opacity_seconds_total"] / float(
            row["solver_total_seconds"]
        )

    state_path = target / "converged_state.npz"
    if state_path.exists():
        with np.load(state_path) as state:
            row["temperature"] = state["temperature"].tolist()
            row["column_mass"] = state["column_mass"].tolist()
    return row


def build_report(out_dir: Path, strides) -> dict:
    """Compare every stride against the stride-1 converged atmosphere."""

    rows = {}
    for stride in strides:
        row = _stride_summary(out_dir, stride)
        if row is not None:
            rows[int(stride)] = row

    reference = rows.get(REFERENCE_STRIDE)
    for stride, row in rows.items():
        if reference is None or "temperature" not in row:
            continue
        reference_temperature = np.asarray(reference["temperature"], dtype=np.float64)
        reference_column_mass = np.asarray(reference["column_mass"], dtype=np.float64)
        temperature = np.asarray(row["temperature"], dtype=np.float64)
        column_mass = np.asarray(row["column_mass"], dtype=np.float64)
        row["versus_stride_one"] = {
            # The solver's own convergence yardstick, so the deviation can be
            # read directly against maximum_deep_layer_relative_temperature_change.
            "deep_layer_relative_temperature_change": (
                deep_layer_relative_temperature_change(
                    reference_temperature, temperature
                )
            ),
            "max_relative_temperature_change": max_normalized_column_delta(
                reference_temperature, temperature, symmetric=True
            ),
            "rms_relative_temperature_change": float(
                np.sqrt(
                    np.mean(
                        ((temperature - reference_temperature) / reference_temperature)
                        ** 2
                    )
                )
            ),
            "max_relative_column_mass_change": max_normalized_column_delta(
                reference_column_mass, column_mass, symmetric=True
            ),
        }
        if reference["opacity_seconds_total"]:
            row["opacity_speedup_versus_stride_one"] = (
                reference["opacity_seconds_total"] / row["opacity_seconds_total"]
                if row["opacity_seconds_total"]
                else None
            )

    return {"strides": [rows[key] for key in sorted(rows)]}


def print_report(report: dict) -> None:
    header = (
        f"{'stride':>6} {'n_freq':>7} {'iters':>6} {'final dT/T':>11} "
        f"{'opacity_s':>10} {'total_s':>9} {'dT/T vs s=1':>12} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in report["strides"]:
        versus = row.get("versus_stride_one", {})
        print(
            f"{row['opacity_frequency_grid_stride']:>6} "
            f"{row['frequency_count'] or 0:>7} "
            f"{row['iterations_to_convergence'] or 0:>6} "
            f"{row['deep_layer_relative_temperature_change'] or float('nan'):>11.3e} "
            f"{row['opacity_seconds_total']:>10.3f} "
            f"{row['solver_total_seconds'] or float('nan'):>9.3f} "
            f"{versus.get('deep_layer_relative_temperature_change', float('nan')):>12.3e} "
            f"{row.get('opacity_speedup_versus_stride_one') or float('nan'):>8.2f}"
        )
        if not row["converged"]:
            print(f"       ^ stride {row['opacity_frequency_grid_stride']} DID NOT CONVERGE")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experiments.opacity_decimation.run_decimation",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/opacity_decimation"),
        help="output directory; one subdirectory per stride",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        default=list(DEFAULT_STRIDES),
        help="opacity-sampling strides to run, in order",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="skip solving; summarize an existing output directory",
    )
    parser.add_argument("--summary", type=Path, default=None, help="summary JSON path")
    parser.add_argument("--effective-temperature", type=float, default=None)
    parser.add_argument("--log-surface-gravity", type=float, default=None)
    parser.add_argument("--metallicity", type=float, default=None)
    parser.add_argument("--alpha-enhancement", type=float, default=None)
    parser.add_argument("--microturbulence-km-s", type=float, default=None)
    parser.add_argument(
        "--iterations-per-trial",
        type=int,
        default=run_reference.PRODUCTION_ITERATIONS_PER_TRIAL,
    )
    parser.add_argument(
        "--max-trials", type=int, default=run_reference.PRODUCTION_MAX_TRIALS
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.report) if args.report is not None else Path(args.out)
    strides = [int(stride) for stride in args.strides]

    if args.report is None:
        overrides = {
            "effective_temperature": args.effective_temperature,
            "log_surface_gravity": args.log_surface_gravity,
            "metallicity": args.metallicity,
            "alpha_enhancement": args.alpha_enhancement,
            "microturbulence_km_s": args.microturbulence_km_s,
        }
        labels = replace(
            REFERENCE_STAR,
            **{key: value for key, value in overrides.items() if value is not None},
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "labels.json").write_text(
            json.dumps(labels.as_kwargs(), indent=2) + "\n"
        )
        print(f"star {labels.slug}, strides {strides}", flush=True)
        # Serial by construction: one solver process peaks near 14 GB.
        for stride in strides:
            print(f"=== stride {stride} ===", flush=True)
            record = run_one_stride(
                labels,
                stride,
                out_dir=out_dir,
                iterations_per_trial=args.iterations_per_trial,
                max_trials=args.max_trials,
            )
            print(
                f"  converged={record['converged']} "
                f"iters={record['converging_trial_iterations']} "
                f"trials={record['trials_used']} "
                f"{record['wall_seconds']:.1f}s",
                flush=True,
            )

    report = build_report(out_dir, strides)
    print_report(report)
    summary_path = Path(args.summary) if args.summary else out_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
