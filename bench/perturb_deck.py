"""Deck-perturbation sensitivity experiment (progress notes §4.1).

Every warm start reaches the solver through the fixed-width deck format, which
quantizes gas pressure, electron density, Rosseland opacity, and radiative
acceleration to four significant digits (``%.3E``, ``warm_start.py:575-579``) —
a worst-case relative half-step of 5e-4, equal to the deep-layer convergence
threshold itself.  The open question: does the iteration count ``N_iter``
actually respond to a perturbation at that scale?  If not, the deck
quantization is a harmless noise floor.  If it does, the solver is chaotically
sensitive at exactly the resolution of its own input format, which bounds what
any initializer improvement can deliver and sets the Gate 1 tolerance.

Design:

- Stars are drawn from the Stage 0 baseline, stratified by trial-0 iteration
  count (fast / mid / slow), restricted to stars whose first trial converged
  so the control ``N_iter`` is a clean number.
- The control (``epsilon = 0``) is re-run inside this harness — twice, to
  confirm run-to-run determinism — rather than reused from the baseline,
  which was collected under mixed thread counts on two machines.
- Perturbed solves multiply the four ``%.3E`` columns by ``1 + epsilon * u``
  with ``u ~ U(-1, 1)`` iid per layer and column — the rounding-noise model of
  the format — applied to the parsed (already quantized) deck, so the solver
  sees exactly a deck that differs at the format's own resolution.
- Every solve runs the production policy with ``max_trials = 1``: the retry
  path exists precisely to mask sensitivity, and the quantity of interest is
  the raw first-trial response.

Usage::

    python -m bench.perturb_deck runs/baseline_local/records.jsonl \
        --out runs/deck_perturbation --workers 32
    python -m bench.perturb_deck --report runs/deck_perturbation/records.jsonl
"""

from __future__ import annotations

# Must precede any Numba import. See bench/environment.py.
from . import environment as _environment  # noqa: F401

import argparse
import dataclasses
import json
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

from payne_zero_atmosphere.runner import run_atmosphere_model
from payne_zero_atmosphere.warm_start import emulator_warm_start_model

from .labels import StellarLabels
from .run_reference import (
    PRODUCTION_ITERATIONS_PER_TRIAL,
    _as_plain,
    _solver_config,
)

# The four columns the deck format writes with %.3E (four significant digits).
# Column mass (%.8E) and temperature (%.1f, ~1e-5 relative) carry far finer
# format resolution and are left untouched.
PERTURBED_FIELDS = (
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)

EPSILONS = (1.0e-4, 5.0e-4, 1.0e-3)
CONTROL_REPS = 2
DEFAULT_REPS = 3
DEFAULT_STARS_PER_STRATUM = 12

# Strata by trial-0 iterations of the baseline record: [3, 5], [6, 9], [10, 15].
STRATUM_BOUNDS = ((3, 5), (6, 9), (10, 15))
STRATUM_NAMES = ("fast", "mid", "slow")


def select_stars(
    records: list[dict], *, per_stratum: int = DEFAULT_STARS_PER_STRATUM, seed: int = 0
) -> list[tuple[str, dict]]:
    """Pick baseline stars stratified by trial-0 iteration count.

    Only stars whose first trial converged are eligible, so the control run
    has a clean iteration count.  Returns ``(stratum_name, labels)`` pairs.
    """

    eligible = []
    for record in records:
        trials = record.get("trials") or []
        if trials and trials[0].get("converged"):
            eligible.append((int(trials[0]["iterations_completed"]), record))

    rng = np.random.default_rng(seed)
    selected: list[tuple[str, dict]] = []
    for name, (low, high) in zip(STRATUM_NAMES, STRATUM_BOUNDS):
        pool = [r for iters, r in eligible if low <= iters <= high]
        order = rng.permutation(len(pool))
        for index in order[:per_stratum]:
            selected.append((name, pool[int(index)]["labels"]))
    return selected


def _job_seed(slug: str, epsilon: float, rep: int) -> int:
    """Deterministic per-job seed, stable across processes and machines."""

    return zlib.crc32(f"{slug}|{epsilon:.3e}|{rep}".encode())


def perturb_atmosphere(model, epsilon: float, seed: int):
    """Multiply the four %.3E columns by ``1 + epsilon * U(-1, 1)`` per layer."""

    rng = np.random.default_rng(seed)
    updates = {}
    for field_name in PERTURBED_FIELDS:
        values = np.asarray(getattr(model, field_name), dtype=np.float64)
        noise = rng.uniform(-1.0, 1.0, size=values.shape)
        updates[field_name] = values * (1.0 + epsilon * noise)
    return dataclasses.replace(model, **updates)


def run_job(payload) -> dict:
    """Solve one (star, epsilon, rep) job on the production path."""

    stratum, labels, epsilon, rep = payload
    labels = StellarLabels(**{k: float(v) for k, v in labels.items()})
    started = time.perf_counter()
    record = {
        "slug": labels.slug,
        "labels": labels.as_kwargs(),
        "stratum": stratum,
        "epsilon": float(epsilon),
        "rep": int(rep),
        "perturbed_fields": list(PERTURBED_FIELDS),
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warm_start, _deck = emulator_warm_start_model(
                **labels.as_kwargs(), device="cpu"
            )
            if epsilon > 0.0:
                warm_start = perturb_atmosphere(
                    warm_start, epsilon, seed=_job_seed(labels.slug, epsilon, rep)
                )
            result = run_atmosphere_model(
                _solver_config(
                    warm_start,
                    iterations_per_trial=PRODUCTION_ITERATIONS_PER_TRIAL,
                    structured_atmosphere_path=None,
                    debug_state_path=None,
                )
            )
        record.update(
            converged=bool(result.converged),
            iterations_completed=int(result.iterations_completed),
            diagnostics=_as_plain(result.diagnostics),
            warnings=[str(w.message) for w in caught],
        )
    except Exception as exc:  # noqa: BLE001 - a failed job is a data point
        record.update(converged=False, iterations_completed=0, error=f"{type(exc).__name__}: {exc}")
    record["seconds"] = time.perf_counter() - started
    return record


def build_jobs(
    stars: list[tuple[str, dict]], *, reps: int = DEFAULT_REPS
) -> list[tuple[str, dict, float, int]]:
    """One control pair plus ``reps`` realizations per epsilon, per star."""

    jobs = []
    for stratum, labels in stars:
        for rep in range(CONTROL_REPS):
            jobs.append((stratum, labels, 0.0, rep))
        for epsilon in EPSILONS:
            for rep in range(reps):
                jobs.append((stratum, labels, epsilon, rep))
    return jobs


def run_jobs(jobs, *, workers: int, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle = out_path.open("a")
    executor = None
    try:
        if workers <= 1:
            iterator = map(run_job, jobs)
        else:
            from concurrent.futures import ProcessPoolExecutor

            executor = ProcessPoolExecutor(max_workers=workers)
            iterator = executor.map(run_job, jobs)
        for index, record in enumerate(iterator, start=1):
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(jobs)}] {record['slug']} eps={record['epsilon']:.0e} "
                f"rep={record['rep']} converged={record['converged']} "
                f"iters={record['iterations_completed']} {record['seconds']:.1f}s",
                flush=True,
            )
    finally:
        handle.close()
        if executor is not None:
            executor.shutdown()


# --- report -----------------------------------------------------------------


def _delta_stats(deltas: np.ndarray) -> dict:
    if deltas.size == 0:
        return {
            "n": 0,
            "frac_changed": 0.0,
            "mean_delta": 0.0,
            "median_abs_delta_changed": 0.0,
            "max_abs_delta": 0.0,
        }
    changed = np.abs(deltas[deltas != 0])
    return {
        "n": int(deltas.size),
        "frac_changed": float(np.mean(deltas != 0)),
        "mean_delta": float(np.mean(deltas)),
        "median_abs_delta_changed": (
            float(np.median(changed)) if changed.size else 0.0
        ),
        "max_abs_delta": float(np.max(np.abs(deltas))) if deltas.size else 0.0,
    }


def report(records: list[dict]) -> dict:
    """Aggregate perturbed-solve records against their per-star controls."""

    by_star: dict[str, list[dict]] = {}
    for record in records:
        by_star.setdefault(record["slug"], []).append(record)

    per_epsilon: dict[float, dict] = {}
    determinism_failures = []
    for slug, star_records in sorted(by_star.items()):
        controls = [r for r in star_records if r["epsilon"] == 0.0]
        if not controls:
            continue
        control_iters = {r["iterations_completed"] for r in controls}
        control_converged = {r["converged"] for r in controls}
        if len(control_iters) > 1 or len(control_converged) > 1:
            determinism_failures.append(slug)
        control = controls[0]
        n_control = control["iterations_completed"]
        for record in star_records:
            if record["epsilon"] == 0.0:
                continue
            bucket = per_epsilon.setdefault(
                record["epsilon"],
                {"deltas": [], "flips": 0, "errors": 0, "n": 0, "strata": {}},
            )
            bucket["n"] += 1
            if "error" in record:
                bucket["errors"] += 1
                continue
            if not control["converged"]:
                continue
            stratum = record.get("stratum", "?")
            if not record["converged"]:
                bucket["flips"] += 1
                bucket["strata"].setdefault(stratum, []).append(None)
            else:
                delta = record["iterations_completed"] - n_control
                bucket["deltas"].append(delta)
                bucket["strata"].setdefault(stratum, []).append(delta)

    summary = {"determinism_failures": determinism_failures, "per_epsilon": {}}
    for epsilon in sorted(per_epsilon):
        bucket = per_epsilon[epsilon]
        entry = _delta_stats(np.array(bucket["deltas"], dtype=float))
        entry["convergence_flips"] = bucket["flips"]
        entry["errors"] = bucket["errors"]
        entry["per_stratum"] = {
            name: _delta_stats(
                np.array([d for d in deltas if d is not None], dtype=float)
            )
            for name, deltas in sorted(bucket["strata"].items())
        }
        summary["per_epsilon"][f"{epsilon:.0e}"] = entry
    return summary


def _print_report(summary: dict) -> None:
    print("=" * 68)
    print("Deck-perturbation sensitivity: dN_iter vs perturbation level")
    print("=" * 68)
    failures = summary["determinism_failures"]
    print(
        f"control determinism: {'OK' if not failures else f'FAILED for {failures}'}"
    )
    header = f"{'epsilon':>8} {'n':>4} {'changed':>8} {'mean dN':>8} {'med|dN|':>8} {'max|dN|':>8} {'flips':>6}"
    print(header)
    for epsilon, entry in summary["per_epsilon"].items():
        print(
            f"{epsilon:>8} {entry['n']:>4} {entry['frac_changed']:>8.2f} "
            f"{entry['mean_delta']:>8.2f} {entry['median_abs_delta_changed']:>8.1f} "
            f"{entry['max_abs_delta']:>8.0f} {entry['convergence_flips']:>6}"
        )
        for name, stratum in entry["per_stratum"].items():
            print(
                f"  {name:>6} {stratum['n']:>4} {stratum['frac_changed']:>8.2f} "
                f"{stratum['mean_delta']:>8.2f} {stratum['median_abs_delta_changed']:>8.1f} "
                f"{stratum['max_abs_delta']:>8.0f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.perturb_deck", description=__doc__
    )
    parser.add_argument("records", type=Path, nargs="?", help="baseline records.jsonl")
    parser.add_argument("--out", type=Path, help="output directory for the run")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--stars-per-stratum", type=int, default=DEFAULT_STARS_PER_STRATUM)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=0, help="star selection seed")
    parser.add_argument("--report", type=Path, help="report on an existing records file")
    args = parser.parse_args(argv)

    if args.report is not None:
        lines = args.report.read_text().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        _print_report(report(records))
        return 0

    if args.records is None or args.out is None:
        parser.error("run mode requires RECORDS and --out")

    lines = args.records.read_text().splitlines()
    baseline = [json.loads(line) for line in lines if line.strip()]
    stars = select_stars(baseline, per_stratum=args.stars_per_stratum, seed=args.seed)
    jobs = build_jobs(stars, reps=args.reps)

    # Resume: skip (slug, epsilon, rep) combinations already recorded.
    out_path = args.out / "records.jsonl"
    if out_path.exists():
        done = {
            (r["slug"], r["epsilon"], r["rep"])
            for r in (json.loads(l) for l in out_path.read_text().splitlines() if l.strip())
        }
        jobs = [
            job for job in jobs
            if (StellarLabels(**{k: float(v) for k, v in job[1].items()}).slug, job[2], job[3])
            not in done
        ]
    print(f"{len(stars)} stars, {len(jobs)} jobs to run", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_config.json").write_text(
        json.dumps(
            {
                "stars": len(stars),
                "epsilons": list(EPSILONS),
                "reps": args.reps,
                "control_reps": CONTROL_REPS,
                "seed": args.seed,
                "perturbed_fields": list(PERTURBED_FIELDS),
                "iterations_per_trial": PRODUCTION_ITERATIONS_PER_TRIAL,
                "max_trials": 1,
            },
            indent=2,
        )
        + "\n"
    )
    run_jobs(jobs, workers=args.workers, out_path=out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
