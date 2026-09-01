"""Run the bounded real-solver funnel for the H2 analytic initializer.

Two arms share one star list and one solver policy so their rows can be paired:
``analytic`` seeds the solver from the no-emulator H2 formula, and
``production`` seeds it from the current emulator warm start.  Without the
second arm a funnel cannot tell an initializer defect from a solver hard
region -- ``runs_baseline.log`` shows the production path itself failing on
warm mid-gravity and hot low-gravity stars.

Each star solves in its own subprocess under a wall-clock timeout, and each
record streams to a JSON Lines file as it lands.  Both exist because the first
60-star funnel hung on one hard row and had to be killed, which discarded all
40 completed rows: they had only ever been printed, never written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import queue as queue_module
import time
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.compact_initializer import (
    COMPACT_CONFIGURATION,
    PARITY_CONFIGURATION,
    PHYSICAL_CONFIGURATION,
    fit_compact_profile_parameters,
    load_compact_profile_parameters,
    predict_compact_reduced_state,
)
from experiments.analytic_initializer.profile_initializer import (
    fit_analytic_profile_parameters,
    predict_analytic_reduced_state,
)

# Every arm but ``production`` seeds the solver from a formula and differs only
# in which constants it was fitted with, so they share the whole solve path and
# the same drawn stars.  That is what makes the comparison paired.
FORMULA_ARMS = {
    "analytic": None,
    "compact600": COMPACT_CONFIGURATION,
    "parity": PARITY_CONFIGURATION,
    "parity_polytrope": PARITY_CONFIGURATION,
    "physical": PHYSICAL_CONFIGURATION,
}
#: Training rows and grid used to fit the entropy closure's three constants.
ENTROPY_FIT_STARS = 600
ENTROPY_FIT_SEED = 20260817
PARITY_PARAMETER_ASSET = Path(
    "results/analytic_initializer/compact_profile_parameters_parity.npz"
)

ARM_CANDIDATES = {
    "analytic": "H2_standalone_low_rank_hopf_and_opacity_profile",
    "compact600": "compact_chebyshev_depth_within_600_stored_floats",
    "parity": "compact_chebyshev_depth_at_H2_accuracy",
    "parity_polytrope": "parity_plus_one_shot_eos_polytrope_projection",
    "physical": "compact_chebyshev_depth_with_saha_label_coordinates",
    "entropy": "dual_crossing_entropy_closure_three_global_constants",
    "cumtau": "four_window_cumulative_tau_physical_degree2",
    "textbook_v4r3": "grey_plus_adiabatic_v4r3_opacity_ode_mass",
    "textbook_v4r6": "grey_plus_adiabatic_v4r6_opacity_ode_mass",
    "textbook_v4r6_grey": "eddington_grey_v4r6_opacity_ode_mass",
    "textbook_v4r6_decoupled": "v4r6_decoupled_mgrey_tconv_v1",
    "production": "current_production_emulator_initializer",
}


MANIFESTS = (
    Path("results/reconstruction_metrics.json"),
    Path("results/sealed_solver_subset_20260808.json"),
    Path("results/sealed_audit_20260808.json"),
    Path("results/sealed_audit_20260811.json"),
    Path("results/sealed_initializer_holdout_20260812.json"),
    Path("results/initializer_calibration_20260812.json"),
    Path("results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json"),
)

# The 12 stars already spent on the H2 smoke test, held out of the funnel draw.
SMOKE_INDICES = (
    2891, 6896, 7811, 10082, 10313, 22134,
    25100, 25948, 27946, 35654, 38262, 41936,
)

ITERATIONS_PER_TRIAL = 15
LABEL_NAMES = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _choose_indices(
    labels: np.ndarray,
    available: np.ndarray,
    *,
    count: int,
    seed: int,
    exclude: np.ndarray = np.empty(0, dtype=np.int64),
) -> np.ndarray:
    if count < 3 or count % 3:
        raise ValueError("count must be a positive multiple of three")
    generator = np.random.default_rng(seed)
    remaining = np.setdiff1d(available, exclude)
    selected: list[int] = []
    per_band = count // 3
    for low, high in ((4000.0, 5500.0), (5500.0, 7500.0), (7500.0, 10500.0)):
        candidates = remaining[
            (labels[remaining, 0] >= low) & (labels[remaining, 0] < high)
        ]
        if candidates.size < per_band:
            raise RuntimeError(f"not enough validation stars in {low:g}--{high:g} K")
        selected.extend(generator.choice(candidates, per_band, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _label_fields(labels: np.ndarray) -> dict[str, float]:
    return {name: float(labels[position]) for position, name in enumerate(LABEL_NAMES)}


def _solver_slug(row: np.ndarray) -> str:
    """Public five-label solver slug, matching ``bench.labels`` (t-format) so
    products pair by stem on the spectral gate."""

    return (
        f"t{row[0]:07.1f}_g{row[1]:+05.2f}_m{row[2]:+05.2f}"
        f"_a{row[3]:+05.2f}_x{row[4]:04.2f}"
    )


def _stellar_labels(labels: np.ndarray):
    from bench.labels import StellarLabels

    return StellarLabels(**_label_fields(labels))


def _solve_analytic(
    labels: np.ndarray,
    mass: np.ndarray,
    temperature: np.ndarray,
    log_opacity: np.ndarray,
    tau: np.ndarray,
    *,
    eos_polytrope: bool = False,
    product_path: Path | None = None,
    iterations_per_trial: int = ITERATIONS_PER_TRIAL,
) -> dict[str, object]:
    """Seed the solver from the analytic formula and run one trial.

    ``product_path`` requests the release handoff (a
    ``save_product_structured_atmosphere`` NPZ named like the production baseline
    products) so a formula arm can be compared on the spectrum gate. Defaults to
    None, which keeps every existing funnel run unchanged.
    """

    from bench import environment as _environment  # noqa: F401
    from bench.run_reference import _atmosphere_is_finite, _solver_config
    from experiments.analytic_initializer.no_emulator_bridge import analytic_seed_model
    from payne_zero_atmosphere.runner import run_atmosphere_model

    seed = analytic_seed_model(labels, mass, temperature, log_opacity, tau)
    solver_config = _solver_config(
        seed,
        iterations_per_trial=int(iterations_per_trial),
        structured_atmosphere_path=product_path,
        debug_state_path=None,
    )
    if eos_polytrope:
        from experiments.analytic_initializer.eos_polytrope import (
            run_with_eos_polytrope,
        )

        result = run_with_eos_polytrope(solver_config)
    else:
        result = run_atmosphere_model(solver_config)
    finite = _atmosphere_is_finite(result.atmosphere)
    converged = bool(result.converged) and finite
    outcome = {
        "converged": converged,
        "solver_reported_converged": bool(result.converged),
        "finite_final_state": bool(finite),
        "iterations_completed": int(result.iterations_completed),
        "first_trial_converged": converged,
        "trials_used": 1,
        "deep_layer_relative_temperature_change": float(
            result.diagnostics["deep_layer_relative_temperature_change"]
        ),
    }
    if eos_polytrope:
        outcome["eos_polytrope"] = result.diagnostics.get(
            "after_iteration_hooks", {}
        ).get("1", {})
    return outcome


def _solve_production(
    labels: np.ndarray,
    *,
    iterations_per_trial: int = ITERATIONS_PER_TRIAL,
) -> dict[str, object]:
    """Seed the solver from the production emulator warm start.

    ``run_star`` keeps every trial, so one run answers both policies: the
    paired single-trial comparison against the analytic arm, and the two-trial
    production baseline the four-initializer benchmark is defined against.
    """

    from bench import environment as _environment  # noqa: F401
    from bench.run_reference import PRODUCTION_MAX_TRIALS, run_star

    record = run_star(
        _stellar_labels(labels),
        iterations_per_trial=int(iterations_per_trial),
        max_trials=PRODUCTION_MAX_TRIALS,
    )
    first = record.trials[0] if record.trials else None
    return {
        "converged": bool(record.converged),
        # ``run_star`` checks every field for finiteness and downgrades a
        # converged trial that fails, but it does not return the atmosphere, so
        # a non-converged star's finiteness is genuinely unknown here rather
        # than known-bad.  Recording None keeps it from being counted either way.
        "finite_final_state": True if record.converged else None,
        "iterations_completed": record.converging_trial_iterations,
        "first_trial_converged": bool(first.converged) if first is not None else False,
        "first_trial_iterations": int(first.iterations_completed) if first is not None else 0,
        "trials_used": len(record.trials),
        "deep_layer_relative_temperature_change": (
            first.diagnostics.get("deep_layer_relative_temperature_change")
            if first is not None
            else None
        ),
        "warnings": list(record.warnings),
    }


def _solve_payload(payload: dict) -> dict[str, object]:
    """Solve one star, turning any failure into a record rather than a crash."""

    try:
        # Dispatch on whether a reduced state was supplied, not on the arm
        # name.  Keying on the string silently routed three formula arms into
        # the emulator path when they were added, and the run looked healthy
        # because the emulator converges: the only symptom was ``trials_used``
        # coming back as 2 from an arm that is allocated exactly one trial.
        iterations_per_trial = int(
            payload.get("iterations_per_trial", ITERATIONS_PER_TRIAL)
        )
        if payload["mass"] is None:
            outcome = _solve_production(
                payload["labels"],
                iterations_per_trial=iterations_per_trial,
            )
        else:
            outcome = _solve_analytic(
                payload["labels"],
                payload["mass"],
                payload["temperature"],
                payload["log_opacity"],
                payload["tau"],
                eos_polytrope=bool(payload.get("eos_polytrope", False)),
                product_path=payload.get("product_path"),
                iterations_per_trial=iterations_per_trial,
            )
        outcome["solver_outcome"] = (
            "converged" if outcome["converged"] else "not_converged"
        )
    except Exception as exc:  # noqa: BLE001 - a failed star is a data point
        import traceback

        outcome = {
            "converged": False,
            "finite_final_state": False,
            "iterations_completed": 0,
            "solver_outcome": "error",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }
    return outcome


def _worker_loop(task_queue, result_queue) -> None:
    """Subprocess entry point: solve stars until handed the stop sentinel."""

    while True:
        payload = task_queue.get()
        if payload is None:
            return
        result_queue.put(_solve_payload(payload))


def _timed_out_record() -> dict[str, object]:
    """A star the solver never returned from, preserved rather than dropped."""

    return {
        "converged": False,
        "finite_final_state": False,
        "iterations_completed": None,
        "solver_outcome": "timeout",
    }


class _SolverWorker:
    """A spawned process that solves stars one at a time.

    Reusing one process amortizes Numba compilation and, on the production arm,
    the emulator checkpoint load.  Together those cost several minutes -- an
    order of magnitude more than a solve, so a process per star would dominate
    the run.  The process is still replaced whenever a star hangs, so one bad
    row cannot take the funnel down, which is what happened to the first
    60-star attempt.
    """

    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._tasks = None
        self._results = None
        self._process = None
        self._start()

    def _start(self) -> None:
        self._tasks = self._context.Queue()
        self._results = self._context.Queue()
        self._process = self._context.Process(
            target=_worker_loop, args=(self._tasks, self._results)
        )
        self._process.start()

    def _discard(self) -> None:
        if self._process is None:
            return
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()
        self._tasks.close()
        self._results.close()
        self._process = None

    def solve(self, payload: dict, *, timeout: float) -> dict[str, object]:
        start = time.perf_counter()
        if self._process is None or not self._process.is_alive():
            self._discard()
            self._start()
        self._tasks.put(payload)
        try:
            outcome = self._results.get(timeout=timeout)
        except queue_module.Empty:
            # The worker is wedged inside the solver and will never read the
            # next task, so it has to be replaced rather than reused.
            outcome = _timed_out_record()
            self._discard()
            self._start()
        outcome["seconds"] = float(time.perf_counter() - start)
        return outcome

    def close(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._tasks.put(None)
            self._process.join(timeout=30.0)
        self._discard()


def _streamed_records(jsonl_path: Path) -> list[dict[str, object]]:
    """Read a partial streamed run and reject duplicate corpus rows."""

    if not jsonl_path.exists():
        return []
    records: list[dict[str, object]] = []
    indices: set[int] = set()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            index = int(record["corpus_index"])
            if index in indices:
                raise ValueError(f"{jsonl_path} contains duplicate corpus index {index}")
            indices.add(index)
            records.append(record)
    return records


def _run_funnel(
    corpus,
    indices: np.ndarray,
    *,
    arm: str,
    reduced_state,
    timeout: float,
    jsonl_path: Path,
    resume: bool = False,
    product_dir: Path | None = None,
    iterations_per_trial: int = ITERATIONS_PER_TRIAL,
) -> list[dict[str, object]]:
    """Solve every star, appending each record to ``jsonl_path`` as it lands."""

    if reduced_state is None:
        mass = temperature = log_opacity = None
    else:
        mass, temperature, log_opacity = reduced_state

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    previous = _streamed_records(jsonl_path)
    by_index = {int(record["corpus_index"]): record for record in previous}
    for record in previous:
        index = int(record["corpus_index"])
        if record.get("arm") != arm:
            raise SystemExit(
                f"{jsonl_path} contains arm {record.get('arm')!r}, expected {arm!r}"
            )
        recorded_iterations = int(
            record.get("iterations_per_trial", ITERATIONS_PER_TRIAL)
        )
        if recorded_iterations != int(iterations_per_trial):
            raise SystemExit(
                f"{jsonl_path} was written with {recorded_iterations} iterations "
                f"per trial, expected {int(iterations_per_trial)}"
            )
        expected_labels = _label_fields(corpus.labels[index])
        if any(
            not np.isclose(float(record[name]), value, rtol=0.0, atol=1.0e-12)
            for name, value in expected_labels.items()
        ):
            raise SystemExit(
                f"{jsonl_path} labels disagree with corpus index {index}"
            )
    already = set(by_index)
    requested = {int(index) for index in indices}
    unexpected = sorted(already - requested)
    if unexpected:
        raise SystemExit(
            f"{jsonl_path} contains {len(unexpected)} rows outside this request "
            f"(first few: {unexpected[:5]})"
        )
    repeated = sorted(already.intersection(int(index) for index in indices))
    if repeated and not resume:
        # Appending is what lets an interrupted run keep its rows, but it also
        # means a careless rerun would leave two records per star in the file
        # and no way to tell which solve produced which.
        raise SystemExit(
            f"{jsonl_path} already holds records for {len(repeated)} of these "
            f"stars (first few: {repeated[:5]}). Move the file aside, or pass "
            "--out to a fresh path, before rerunning them."
        )

    missing = [int(index) for index in indices if int(index) not in already]
    if not missing:
        return [by_index[int(index)] for index in indices]

    worker = _SolverWorker()
    try:
        with jsonl_path.open("a", encoding="utf-8") as handle:
            for row, index in enumerate(indices):
                if int(index) in already:
                    continue
                payload = {
                    "arm": arm,
                    "labels": corpus.labels[index],
                    "mass": None if mass is None else mass[row],
                    "temperature": None if temperature is None else temperature[row],
                    "log_opacity": None if log_opacity is None else log_opacity[row],
                    "tau": corpus.tau,
                    "eos_polytrope": arm == "parity_polytrope",
                    "product_path": None if product_dir is None else str(product_dir / (_solver_slug(corpus.labels[index]) + ".npz")),
                    "iterations_per_trial": int(iterations_per_trial),
                }
                record: dict[str, object] = {
                    "corpus_index": int(index),
                    "arm": arm,
                    "iterations_per_trial": int(iterations_per_trial),
                    "slug": str(corpus.slugs[index]),
                    **_label_fields(corpus.labels[index]),
                }
                seed_is_finite = mass is None or (
                    np.all(np.isfinite(mass[row]))
                    and np.all(np.isfinite(temperature[row]))
                    and np.all(np.isfinite(log_opacity[row]))
                    and np.all(mass[row] > 0.0)
                    and np.all(temperature[row] > 0.0)
                )
                if not seed_is_finite:
                    record.update(
                        {
                            "converged": False,
                            "finite_final_state": False,
                            "iterations_completed": 0,
                            "solver_outcome": "error",
                            "error": "reduced-state seed is non-finite or non-positive",
                            "seconds": 0.0,
                        }
                    )
                else:
                    record.update(worker.solve(payload, timeout=timeout))
                # A formula arm is allocated exactly one trial, so more than
                # one means the star went down the emulator path.  That is a
                # wiring mistake rather than a solver outcome, and it is
                # invisible in the summary because the emulator converges --
                # so it has to fail loudly here, on the first star, instead of
                # forty minutes later.
                if mass is not None and int(record.get("trials_used", 1)) != 1:
                    raise SystemExit(
                        f"arm {arm!r} supplied a reduced state but star {index} "
                        f"used {record['trials_used']} trials, which only the "
                        "production path does. The solve dispatch is wired wrong."
                    )
                by_index[int(index)] = record
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                print(
                    f"[{len(by_index)}/{len(indices)}] {arm} index={index} "
                    f"outcome={record['solver_outcome']} "
                    f"iters={record['iterations_completed']} "
                    f"{record['seconds']:.1f}s",
                    flush=True,
                )
    finally:
        worker.close()
    return [by_index[int(index)] for index in indices]


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    outcomes = [str(item["solver_outcome"]) for item in records]
    converged = [bool(item["converged"]) for item in records]
    first_trial = [bool(item.get("first_trial_converged", item["converged"])) for item in records]
    return {
        "star_count": len(records),
        "converged_count": int(sum(converged)),
        "first_trial_converged_count": int(sum(first_trial)),
        # ``is True`` rather than ``bool``: the production arm reports None for
        # stars whose final state it never saw, and an unknown must not be
        # counted as either finite or non-finite.
        "finite_count": int(sum(item["finite_final_state"] is True for item in records)),
        "finite_unknown_count": int(
            sum(item["finite_final_state"] is None for item in records)
        ),
        "timeout_count": outcomes.count("timeout"),
        "error_count": outcomes.count("error"),
        "not_converged_indices": [
            int(item["corpus_index"]) for item, ok in zip(records, converged) if not ok
        ],
    }


def _teff_split_summary(
    records: list[dict[str, object]], *, split_K: float = 7500.0
) -> dict[str, object]:
    """Convergence counts on the frozen cool/hot funnel split."""

    rows: dict[str, dict[str, object]] = {}
    for name, predicate in (
        ("below_7500K", lambda teff: teff < split_K),
        ("at_least_7500K", lambda teff: teff >= split_K),
    ):
        selected = [
            item
            for item in records
            if predicate(float(item["effective_temperature"]))
        ]
        rows[name] = {
            "star_count": len(selected),
            "converged_count": int(sum(bool(item["converged"]) for item in selected)),
            "timeout_count": int(
                sum(str(item["solver_outcome"]) == "timeout" for item in selected)
            ),
            "error_count": int(
                sum(str(item["solver_outcome"]) == "error" for item in selected)
            ),
            "not_converged_count": int(
                sum(str(item["solver_outcome"]) == "not_converged" for item in selected)
            ),
        }
    return rows


def _gravity_split_summary(
    records: list[dict[str, object]], *, split_logg: float = 3.5
) -> dict[str, object]:
    """Convergence counts on the frozen dwarf/giant split."""

    rows: dict[str, dict[str, object]] = {}
    for name, predicate in (
        ("dwarf_logg_at_least_3.5", lambda logg: logg >= split_logg),
        ("giant_logg_below_3.5", lambda logg: logg < split_logg),
    ):
        selected = [
            item
            for item in records
            if predicate(float(item["log_surface_gravity"]))
        ]
        rows[name] = {
            "star_count": len(selected),
            "converged_count": int(sum(bool(item["converged"]) for item in selected)),
            "timeout_count": int(
                sum(str(item["solver_outcome"]) == "timeout" for item in selected)
            ),
            "error_count": int(
                sum(str(item["solver_outcome"]) == "error" for item in selected)
            ),
            "not_converged_count": int(
                sum(str(item["solver_outcome"]) == "not_converged" for item in selected)
            ),
        }
    return rows


def _masked_percentile_errors(
    prediction: np.ndarray,
    truth: np.ndarray,
    star_mask: np.ndarray,
    layer_mask: np.ndarray,
    *,
    kind: str,
) -> dict[str, float]:
    keep = star_mask[:, None] & layer_mask[None, :]
    if kind == "mass_dex":
        values = np.abs(np.log10(prediction) - np.log10(truth))[keep]
    elif kind == "temperature_relative":
        values = np.abs(prediction / truth - 1.0)[keep]
    else:
        raise ValueError(f"unknown error kind {kind!r}")
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot summarize an empty seed-error mask")
    return {
        "count": int(values.size),
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def _seed_error_splits(
    mass: np.ndarray,
    temperature: np.ndarray,
    *,
    truth_mass: np.ndarray,
    truth_temperature: np.ndarray,
    teff: np.ndarray,
    tau: np.ndarray,
    finite_stars: np.ndarray,
) -> dict[str, object]:
    """Seed-versus-truth errors that isolate convection from surface κ."""

    depth = np.asarray(tau, dtype=np.float64)
    cool = finite_stars & (teff < 7500.0)
    hot = finite_stars & (teff >= 7500.0)
    deep = depth > 10.0
    all_layers = np.ones(depth.shape, dtype=bool)
    rows: dict[str, object] = {}
    for band, star_mask in (("cool_below_7500K", cool), ("hot_at_least_7500K", hot)):
        if not np.any(star_mask):
            continue
        rows[band] = {
            "star_count": int(np.count_nonzero(star_mask)),
            "all_layers": {
                "log_mass_dex": _masked_percentile_errors(
                    mass, truth_mass, star_mask, all_layers, kind="mass_dex"
                ),
                "temperature_relative": _masked_percentile_errors(
                    temperature,
                    truth_temperature,
                    star_mask,
                    all_layers,
                    kind="temperature_relative",
                ),
            },
            "tau_gt_10": {
                "log_mass_dex": _masked_percentile_errors(
                    mass, truth_mass, star_mask, deep, kind="mass_dex"
                ),
                "temperature_relative": _masked_percentile_errors(
                    temperature,
                    truth_temperature,
                    star_mask,
                    deep,
                    kind="temperature_relative",
                ),
            },
        }
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=(
            "analytic",
            "compact600",
            "parity",
            "parity_polytrope",
            "physical",
            "entropy",
            "cumtau",
            "textbook_v4r3",
            "textbook_v4r6",
            "textbook_v4r6_grey",
            "textbook_v4r6_decoupled",
            "production",
        ),
        default="analytic",
    )
    parser.add_argument("--count", type=int, default=60)
    index_group = parser.add_mutually_exclusive_group()
    index_group.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="solve exactly these corpus indices instead of drawing the funnel",
    )
    index_group.add_argument(
        "--indices-from",
        type=Path,
        help="read exact corpus indices from a JSON file instead of drawing the funnel",
    )
    parser.add_argument(
        "--indices-key",
        default="star_indices",
        help="JSON key used with --indices-from (default: star_indices)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="solve only the first N drawn stars (the funnel is ordered)",
    )
    parser.add_argument("--per-star-timeout", type=float, default=900.0)
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS_PER_TRIAL,
        help="solver iterations per trial (default: 15). Historical funnels keep 15.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse unique rows already present in the output JSONL",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--product-dir",
        type=Path,
        help="write a release-handoff NPZ per converged star (t-format names)",
    )
    args = parser.parse_args(argv)
    if int(args.iterations) < 1:
        raise SystemExit("--iterations must be a positive integer")

    corpus = load_strict_truth(DEFAULT_CORPUS)
    excluded, used_manifests = collect_excluded_indices(
        MANIFESTS, corpus_size=corpus.size
    )
    split = make_split(corpus.size, excluded=excluded, seed=20260816)
    drawn = _choose_indices(
        corpus.labels,
        split.validation,
        count=args.count,
        seed=20260817,
        exclude=np.asarray(SMOKE_INDICES, dtype=np.int64),
    )
    if args.indices is not None:
        indices = np.asarray(sorted(set(args.indices)), dtype=np.int64)
    elif args.indices_from is not None:
        source = json.loads(args.indices_from.read_text(encoding="utf-8"))
        if args.indices_key not in source:
            raise SystemExit(
                f"{args.indices_from} has no key {args.indices_key!r}"
            )
        indices = np.asarray(
            sorted(set(int(index) for index in source[args.indices_key])),
            dtype=np.int64,
        )
    elif args.limit is not None:
        indices = drawn[: args.limit]
    else:
        indices = drawn

    if args.arm == "production":
        reduced_state, provenance = None, {"initializer": "production emulator"}
    elif args.arm == "analytic":
        parameters = fit_analytic_profile_parameters(corpus, split, degree=3, components=5)
        reduced_state = predict_analytic_reduced_state(
            corpus.labels[indices], corpus.tau, parameters
        )
        provenance = {
            "initializer": "tabulated H2",
            "degree": 3,
            "components": 5,
            "coefficient_count": parameters.coefficient_count,
            "basis_value_count": parameters.basis_value_count,
        }
    elif args.arm == "entropy":
        # The polytrope retest: parity's mass and radiative branch, with the
        # deep temperature replaced by the dual-crossing closure.  Pre-
        # registered in notes/entropy_closure_convergence_retest.md.
        from experiments.analytic_initializer.entropy_hybrid import (
            fit_entropy_hybrid_closure,
            predict_entropy_hybrid_state,
        )

        parameters = fit_compact_profile_parameters(
            corpus, split, configuration=PARITY_CONFIGURATION
        )
        generator = np.random.default_rng(ENTROPY_FIT_SEED)
        rows = generator.choice(split.train, ENTROPY_FIT_STARS, replace=False)
        closure, fit_report = fit_entropy_hybrid_closure(
            corpus.labels[rows], corpus.tau, corpus.temperature[rows], parameters
        )
        reduced_state = predict_entropy_hybrid_state(
            corpus.labels[indices], corpus.tau, parameters, closure
        )
        provenance = {
            "initializer": "parity mass and radiative branch + dual-crossing entropy closure",
            "gamma_ad": closure.gamma_ad,
            "a_enter": closure.a_enter,
            "a_exit": closure.a_exit,
            "closure_stored_floats": closure.stored_float_count,
            "compact_stored_floats": parameters.stored_float_count,
            "fit": fit_report,
        }
    elif args.arm == "cumtau":
        # The frozen four-window cumulative-tau family (physical degree-2,
        # width 0.35 in ln tau).  Same fit path as run_cumulative_tau_probe,
        # but on the full-corpus train split so the label support covers every
        # validation star that the solver funnel may draw.
        from experiments.analytic_initializer.cumulative_tau_initializer import (
            fit_cumulative_tau_parameters,
            fit_oracle_targets,
            predict_cumulative_tau_state,
        )

        oracle = fit_oracle_targets(
            corpus.tau,
            corpus.temperature,
            corpus.column_mass,
            corpus.labels[:, 0],
            width=0.35,
        )
        parameters = fit_cumulative_tau_parameters(
            corpus.labels,
            corpus.tau,
            oracle,
            split.train,
            degree=2,
            width=0.35,
            label_features_name="physical",
            support_indices=np.arange(corpus.size, dtype=np.int64),
        )
        prediction = predict_cumulative_tau_state(
            corpus.labels[indices],
            corpus.tau,
            parameters,
            check_support=False,
        )
        reduced_state = (
            prediction.column_mass,
            prediction.temperature,
            np.log10(prediction.opacity),
        )
        provenance = {
            "initializer": "four-window cumulative-tau physical degree-2, width 0.35",
            "degree": parameters.degree,
            "width_ln_tau": parameters.width,
            "label_features": parameters.label_features,
            "boundaries_tau": parameters.boundaries_tau.tolist(),
            "anchor_tau": parameters.anchor_tau,
            "slope_floor": parameters.slope_floor,
            "slope_ceiling": parameters.slope_ceiling,
            "fitted_parameter_count": parameters.fitted_parameter_count,
            "offline_gate": "diagnostic only; solver gates decide",
        }
    elif args.arm == "textbook_v4r6_decoupled":
        from experiments.analytic_initializer import textbook_opacity

        mass, temperature, log_opacity = (
            textbook_opacity.predict_textbook_reduced_state_v4r6_decoupled(
                corpus.labels[indices],
                corpus.tau,
            )
        )
        reduced_state = (mass, temperature, log_opacity)
        finite_seeds = np.all(
            np.isfinite(mass) & np.isfinite(temperature) & np.isfinite(log_opacity),
            axis=1,
        )
        seed_profile_errors = None
        if np.any(finite_seeds):
            truth_mass = corpus.column_mass[indices][finite_seeds]
            truth_temperature = corpus.temperature[indices][finite_seeds]
            temperature_relative = np.abs(
                temperature[finite_seeds] / truth_temperature - 1.0
            )
            mass_dex = np.abs(
                np.log10(mass[finite_seeds]) - np.log10(truth_mass)
            )
            seed_profile_errors = {
                "finite_seed_count": int(np.count_nonzero(finite_seeds)),
                "temperature_relative_p50": float(
                    np.percentile(temperature_relative, 50.0)
                ),
                "temperature_relative_p95": float(
                    np.percentile(temperature_relative, 95.0)
                ),
                "column_mass_dex_p50": float(np.percentile(mass_dex, 50.0)),
                "column_mass_dex_p95": float(np.percentile(mass_dex, 95.0)),
            }
        provenance = {
            "initializer": "v4r6_decoupled_mgrey_tconv_v1",
            "opacity": "textbook_rosseland_opacity_v4r6",
            "temperature": "current_saha_aware_convective_temperature",
            "mass": "v4r6_grey_integrated_mass",
            "mass_reintegrated_after_convection": False,
            "include_convection": True,
            "substeps_per_layer": 8,
            "formal_temperature_floor_K": textbook_opacity.V4R6_FORMAL_TEMPERATURE_FLOOR_K,
            "fitted_parameter_count": 0,
            "requires_neural_checkpoint_at_runtime": False,
            "offline_decision": "FAIL_STOP",
            "offline_result": (
                "results/analytic_initializer/"
                "textbook_opacity_v4r6_offline_validation_20260828.json"
            ),
            "finite_seed_count": int(np.count_nonzero(finite_seeds)),
            "seed_profile_errors_vs_stored_truth": seed_profile_errors,
            "seed_error_splits_vs_stored_truth": (
                None
                if not np.any(finite_seeds)
                else _seed_error_splits(
                    mass,
                    temperature,
                    truth_mass=corpus.column_mass[indices],
                    truth_temperature=corpus.temperature[indices],
                    teff=corpus.labels[indices, 0],
                    tau=corpus.tau,
                    finite_stars=finite_seeds,
                )
            ),
            "registered_v4r6_seed_unchanged": True,
            "registered_v4r6_grey_seed_unchanged": True,
        }
        print(
            f"built {args.arm} seeds for {int(indices.size)} stars "
            f"({int(np.count_nonzero(finite_seeds))} finite"
            ", mass=grey, T=convective, mass_reintegrated=false)",
            flush=True,
        )
    elif args.arm in ("textbook_v4r3", "textbook_v4r6", "textbook_v4r6_grey"):
        from experiments.analytic_initializer import textbook_opacity

        include_convection = args.arm != "textbook_v4r6_grey"
        if args.arm in ("textbook_v4r6", "textbook_v4r6_grey"):
            predict_reduced_state = textbook_opacity.predict_textbook_reduced_state_v4r6
            formal_floor = textbook_opacity.V4R6_FORMAL_TEMPERATURE_FLOOR_K
            offline_result = (
                "results/analytic_initializer/"
                "textbook_opacity_v4r6_offline_validation_20260828.json"
            )
        else:
            predict_reduced_state = textbook_opacity.predict_textbook_reduced_state_v4r3
            formal_floor = textbook_opacity.V4R3_FORMAL_TEMPERATURE_FLOOR_K
            offline_result = (
                "results/analytic_initializer/"
                "textbook_opacity_v4r3_offline_validation_20260827.json"
            )

        mass, temperature, log_opacity = predict_reduced_state(
            corpus.labels[indices],
            corpus.tau,
            include_convection=include_convection,
        )
        reduced_state = (mass, temperature, log_opacity)
        finite_seeds = np.all(
            np.isfinite(mass) & np.isfinite(temperature) & np.isfinite(log_opacity),
            axis=1,
        )
        seed_profile_errors = None
        if np.any(finite_seeds):
            truth_mass = corpus.column_mass[indices][finite_seeds]
            truth_temperature = corpus.temperature[indices][finite_seeds]
            temperature_relative = np.abs(
                temperature[finite_seeds] / truth_temperature - 1.0
            )
            mass_dex = np.abs(
                np.log10(mass[finite_seeds]) - np.log10(truth_mass)
            )
            seed_profile_errors = {
                "finite_seed_count": int(np.count_nonzero(finite_seeds)),
                "temperature_relative_p50": float(
                    np.percentile(temperature_relative, 50.0)
                ),
                "temperature_relative_p95": float(
                    np.percentile(temperature_relative, 95.0)
                ),
                "column_mass_dex_p50": float(np.percentile(mass_dex, 50.0)),
                "column_mass_dex_p95": float(np.percentile(mass_dex, 95.0)),
            }
        if include_convection:
            provenance = {
                "initializer": f"grey_plus_adiabatic_T_plus_{args.arm}_opacity_ODE_mass",
                "opacity": f"textbook_rosseland_opacity_{args.arm}",
                "temperature": "eddington_grey_then_saha_aware_adiabatic",
                "include_convection": True,
                "substeps_per_layer": 8,
                "formal_temperature_floor_K": formal_floor,
                "fitted_parameter_count": 0,
                "requires_neural_checkpoint_at_runtime": False,
                "offline_decision": "FAIL_STOP",
                "offline_result": offline_result,
                "finite_seed_count": int(np.count_nonzero(finite_seeds)),
                "seed_profile_errors_vs_stored_truth": seed_profile_errors,
            }
        else:
            provenance = {
                "initializer": "eddington_grey_T_plus_v4r6_opacity_ODE_mass",
                "opacity": "textbook_rosseland_opacity_v4r6",
                "temperature": "eddington_grey",
                "include_convection": False,
                "substeps_per_layer": 8,
                "formal_temperature_floor_K": formal_floor,
                "fitted_parameter_count": 0,
                "requires_neural_checkpoint_at_runtime": False,
                "offline_decision": "FAIL_STOP",
                "offline_result": offline_result,
                "finite_seed_count": int(np.count_nonzero(finite_seeds)),
                "seed_profile_errors_vs_stored_truth": seed_profile_errors,
                "seed_error_splits_vs_stored_truth": (
                    None
                    if not np.any(finite_seeds)
                    else _seed_error_splits(
                        mass,
                        temperature,
                        truth_mass=corpus.column_mass[indices],
                        truth_temperature=corpus.temperature[indices],
                        teff=corpus.labels[indices, 0],
                        tau=corpus.tau,
                        finite_stars=finite_seeds,
                    )
                ),
                "registered_v4r6_seed_unchanged": True,
            }
        print(
            f"built {args.arm} seeds for {int(indices.size)} stars "
            f"({int(np.count_nonzero(finite_seeds))} finite"
            f", convection={include_convection})",
            flush=True,
        )
    else:
        if args.arm in {"parity", "parity_polytrope"} and PARITY_PARAMETER_ASSET.is_file():
            parameters = load_compact_profile_parameters(PARITY_PARAMETER_ASSET)
            parameter_asset = {
                "path": str(PARITY_PARAMETER_ASSET),
                "sha256": _sha256(PARITY_PARAMETER_ASSET),
            }
        else:
            parameters = fit_compact_profile_parameters(
                corpus, split, configuration=FORMULA_ARMS[args.arm]
            )
            parameter_asset = None
        reduced_state = predict_compact_reduced_state(
            corpus.labels[indices], corpus.tau, parameters
        )
        provenance = {
            "initializer": (
                "parity compact Chebyshev depth + one-shot EOS polytrope"
                if args.arm == "parity_polytrope"
                else "compact Chebyshev depth"
            ),
            "configuration": FORMULA_ARMS[args.arm],
            "stored_float_count": parameters.stored_float_count,
            "label_features": parameters.temperature.label_features,
            "runtime_projection": (
                "solver EOS adiabatic gradient after iteration 1"
                if args.arm == "parity_polytrope"
                else None
            ),
            "parameter_asset": parameter_asset,
        }

    jsonl_path = args.out.with_suffix(".jsonl")
    records = _run_funnel(
        corpus,
        indices,
        arm=args.arm,
        reduced_state=reduced_state,
        timeout=args.per_star_timeout,
        jsonl_path=jsonl_path,
        resume=bool(args.resume),
        product_dir=args.product_dir,
        iterations_per_trial=int(args.iterations),
    )

    result = {
        "candidate": ARM_CANDIDATES[args.arm],
        "arm": args.arm,
        "status": "funnel_not_production",
        "corpus": str(corpus.path),
        "split_seed": split.seed,
        "seed": 20260817,
        "requested_count": int(args.count),
        "iterations_per_trial": int(args.iterations),
        "per_star_timeout_seconds": float(args.per_star_timeout),
        "excluded_count": int(excluded.size),
        "excluded_manifests": used_manifests,
        "initializer_provenance": provenance,
        "streamed_records_path": str(jsonl_path),
        "product_dir": None if args.product_dir is None else str(args.product_dir),
        "records": records,
        **_summarize(records),
    }
    if args.arm in ("textbook_v4r6_grey", "textbook_v4r6_decoupled"):
        result["teff_split"] = _teff_split_summary(records)
    if args.arm == "textbook_v4r6_decoupled":
        result["gravity_split"] = _gravity_split_summary(records)
        result["status"] = "development_only"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "arm",
                    "star_count",
                    "converged_count",
                    "first_trial_converged_count",
                    "finite_count",
                    "timeout_count",
                    "error_count",
                )
            },
            sort_keys=True,
        )
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
