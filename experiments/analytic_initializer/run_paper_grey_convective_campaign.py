"""Paper campaign for the hydrostatic grey--convective initializer.

This campaign is deliberately separate from the historical v4r6 funnels.  It
uses the frozen grey-mass/convective-temperature construction, but records the
full iteration diagnostics, final six-field atmosphere profiles, and spectrum
handoff products needed by the manuscript.

The existing 200-star sample is already open.  Its use here is therefore a
post-hoc generalization check for this initializer, not a new blind test.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import hashlib
import json
import multiprocessing
import os
import platform
import queue
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from experiments.analytic_initializer.discovery import DEFAULT_CORPUS, load_strict_truth
from experiments.analytic_initializer.no_emulator_bridge import analytic_seed_model
from experiments.analytic_initializer.textbook_opacity import (
    predict_textbook_reduced_state_v4r6_decoupled,
)
CAMPAIGN = "paper_grey_convective_20260829"
ITERATIONS = 60
PER_STAR_TIMEOUT_SECONDS = 900.0
SPECTRUM_WINDOW_NM = (400.0, 900.0)
SPECTRUM_RESOLUTION = 20_000.0
SPECTRUM_DTYPE = "float64"
DEFAULT_BAR = 5.0e-3
ARM = "hydrostatic_grey_convective"
DEVELOPMENT_MANIFEST = Path(
    "results/paper_physical_seed_20260820/learned/"
    "convergence_metrics_learned_monotone.json"
)
POSTHOC_MANIFEST = Path("results/sealed_audit_20260811.json")
HISTORICAL_DEVELOPMENT = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r6_decoupled_dev60_policy60_20260829.json"
)
HISTORICAL_RESIDUAL = Path(
    "results/analytic_initializer/"
    "textbook_opacity_v4r6_decoupled_dev60_iter100_residual_20260829.json"
)
REFERENCE_SPECTRA = {
    "development": Path(
        "runs/paper_physical_seed_20260820/learned/spectra/production_six_field"
    ),
    "posthoc200": Path(
        "runs/reduced_state_emulator/"
        "solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
        "blind200/spectra/production_six_field"
    ),
}
REFERENCE_PRODUCTS = {
    "posthoc200": Path(
        "runs/reduced_state_emulator/"
        "solver_in_loop_k1_qualified_tail3_profile_rescue_v4/"
        "blind200/products/production_six_field"
    ),
}
PROFILE_FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)
LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)
THREAD_ENV = {
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_THREADING_LAYER": "workqueue",
}
THREAD_LIMIT_KEYS = tuple(key for key in THREAD_ENV if key != "NUMBA_THREADING_LAYER")
HISTORICAL_REPLAY_ENV = {"NUMBA_THREADING_LAYER": "workqueue"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if dataclasses.is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _solver_slug(labels: np.ndarray) -> str:
    return (
        f"t{labels[0]:07.1f}_g{labels[1]:+05.2f}_m{labels[2]:+05.2f}"
        f"_a{labels[3]:+05.2f}_x{labels[4]:04.2f}"
    )


def _indices(sample: str) -> tuple[np.ndarray, Path]:
    manifest = DEVELOPMENT_MANIFEST if sample == "development" else POSTHOC_MANIFEST
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    values = np.asarray(payload["star_indices"], dtype=np.int64)
    expected = 60 if sample == "development" else 200
    if values.size != expected or np.unique(values).size != expected:
        raise SystemExit(
            f"{manifest} must contain {expected} unique star_indices; found "
            f"{values.size} rows and {np.unique(values).size} unique values"
        )
    return values, manifest


def _sample_paths(root: Path, sample: str) -> dict[str, Path]:
    result = root / "results" / CAMPAIGN / sample
    run = root / "runs" / CAMPAIGN / sample
    return {
        "result": result,
        "run": run,
        "seeds": result / "seeds.npz",
        "solver": result / "solver.json",
        "summary": result / "summary.json",
        "profile_metrics": result / "profile_metrics.npz",
        "profile_summary": result / "profile_metrics.json",
        "spectral_gate": result / "spectral_gate.json",
        "records": run / "records.jsonl",
        "shards": run / "record_shards",
        "profiles": run / "profiles" / ARM,
        "products": run / "products" / ARM,
        "spectra": run / "spectra",
    }


def _write_source_manifest(root: Path) -> Path:
    output = root / "results" / CAMPAIGN / "source_manifest.json"
    sources = (
        Path(__file__).resolve(),
        root / "experiments/analytic_initializer/textbook_opacity.py",
        root / "experiments/analytic_initializer/no_emulator_bridge.py",
        root / "bench/run_reference.py",
        root / "payne_zero_atmosphere/runner.py",
        root / "experiments/reduced_state_emulator/spectral_gate.py",
    )
    inputs = (
        root / DEFAULT_CORPUS,
        root / DEVELOPMENT_MANIFEST,
        root / POSTHOC_MANIFEST,
        root / HISTORICAL_DEVELOPMENT,
        root / HISTORICAL_RESIDUAL,
    )
    payload = {
        "campaign": CAMPAIGN,
        "initializer": "hydrostatic grey-convective",
        "policy": {
            "trials": 1,
            "iterations": ITERATIONS,
            "per_star_timeout_seconds": PER_STAR_TIMEOUT_SECONDS,
            "mass_reintegrated_after_convection": False,
            "requires_neural_checkpoint_at_runtime": False,
            "development_replay_uses_historical_thread_environment": True,
            "posthoc_thread_environment": THREAD_ENV,
        },
        "source_sha256": {
            str(path.relative_to(root)): sha256(path) for path in sources
        },
        "input_sha256": {
            str(path.relative_to(root)): sha256(path) for path in inputs
        },
    }
    if output.is_file():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous != payload:
            raise SystemExit(
                f"{output} disagrees with the current source or inputs; "
                "resume refused"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def _write_seeds(root: Path, sample: str) -> tuple[np.ndarray, dict[str, Path]]:
    paths = _sample_paths(root, sample)
    indices, manifest = _indices(sample)
    corpus = load_strict_truth(root / DEFAULT_CORPUS)
    labels = corpus.labels[indices]
    mass, temperature, log_opacity = predict_textbook_reduced_state_v4r6_decoupled(
        labels, corpus.tau
    )
    finite = np.all(
        np.isfinite(mass) & np.isfinite(temperature) & np.isfinite(log_opacity),
        axis=1,
    )
    positive = np.all(
        (mass > 0.0) & (temperature > 0.0) & np.isfinite(10.0**log_opacity),
        axis=1,
    )
    if paths["seeds"].is_file():
        with np.load(paths["seeds"], allow_pickle=False) as saved:
            checks = (
                np.array_equal(saved["corpus_indices"], indices),
                np.array_equal(saved["labels"], labels),
                np.array_equal(saved["tau"], corpus.tau),
                np.array_equal(saved["column_mass"], mass),
                np.array_equal(saved["temperature"], temperature),
                np.array_equal(saved["log_rosseland_opacity"], log_opacity),
            )
        if not all(checks):
            raise SystemExit(f"{paths['seeds']} disagrees with the frozen seed")
    else:
        paths["result"].mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            paths["seeds"],
            corpus_indices=indices,
            labels=labels,
            tau=corpus.tau,
            column_mass=mass,
            temperature=temperature,
            log_rosseland_opacity=log_opacity,
            truth_column_mass=corpus.column_mass[indices],
            truth_temperature=corpus.temperature[indices],
        )
    seed_record = {
        "sample": sample,
        "role": (
            "development replay"
            if sample == "development"
            else "post-hoc evaluation on a previously opened sample"
        ),
        "sample_manifest": str(manifest),
        "sample_manifest_sha256": sha256(root / manifest),
        "star_count": int(indices.size),
        "finite_seed_count": int(np.count_nonzero(finite)),
        "positive_seed_count": int(np.count_nonzero(positive)),
        "seed_sha256": sha256(paths["seeds"]),
    }
    seed_json = paths["result"] / "seed_summary.json"
    if seed_json.is_file():
        previous = json.loads(seed_json.read_text(encoding="utf-8"))
        if previous != seed_record:
            raise SystemExit(f"{seed_json} disagrees with the frozen seed summary")
    else:
        seed_json.write_text(json.dumps(seed_record, indent=2, sort_keys=True) + "\n")
    return indices, paths


def _profile_payload(atmosphere) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(getattr(atmosphere, field), dtype=np.float64)
        for field in PROFILE_FIELDS
    }


def _solve_one(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    profile_path = Path(payload["profile_path"])
    product_path = Path(payload["product_path"])
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    product_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.unlink(missing_ok=True)
    product_path.unlink(missing_ok=True)
    try:
        from bench.run_reference import _atmosphere_is_finite, _solver_config
        from payne_zero_atmosphere.runner import run_atmosphere_model

        labels = np.asarray(payload["labels"], dtype=np.float64)
        seed = analytic_seed_model(
            labels,
            np.asarray(payload["column_mass"], dtype=np.float64),
            np.asarray(payload["temperature"], dtype=np.float64),
            np.asarray(payload["log_rosseland_opacity"], dtype=np.float64),
            np.asarray(payload["tau"], dtype=np.float64),
        )
        result = run_atmosphere_model(
            _solver_config(
                seed,
                iterations_per_trial=ITERATIONS,
                structured_atmosphere_path=product_path,
                debug_state_path=None,
            )
        )
        finite = bool(_atmosphere_is_finite(result.atmosphere))
        converged = bool(result.converged) and finite
        if converged:
            temporary = profile_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **_profile_payload(result.atmosphere))
            temporary.replace(profile_path)
        if converged and (not profile_path.is_file() or not product_path.is_file()):
            converged = False
            error = "converged solve did not write both profile and spectrum product"
        else:
            error = None
        return {
            "converged": converged,
            "solver_reported_converged": bool(result.converged),
            "finite_final_state": finite,
            "iterations_completed": int(result.iterations_completed),
            "solver_outcome": "converged" if converged else "not_converged",
            "diagnostics": _plain(result.diagnostics),
            "profile_path": str(profile_path) if profile_path.is_file() else None,
            "product_path": str(product_path) if product_path.is_file() else None,
            "error": error,
            "seconds": float(time.perf_counter() - started),
        }
    except Exception as exc:  # noqa: BLE001 - one failed star is a result row
        profile_path.unlink(missing_ok=True)
        product_path.unlink(missing_ok=True)
        return {
            "converged": False,
            "solver_reported_converged": False,
            "finite_final_state": False,
            "iterations_completed": 0,
            "solver_outcome": "error",
            "diagnostics": None,
            "profile_path": None,
            "product_path": None,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "seconds": float(time.perf_counter() - started),
        }


def _worker_loop(tasks, results) -> None:
    while True:
        payload = tasks.get()
        if payload is None:
            return
        results.put(_solve_one(payload))


class _TimedWorker:
    def __init__(self) -> None:
        self.context = multiprocessing.get_context("spawn")
        self.tasks = None
        self.results = None
        self.process = None
        self._start()

    def _start(self) -> None:
        self.tasks = self.context.Queue()
        self.results = self.context.Queue()
        self.process = self.context.Process(
            target=_worker_loop, args=(self.tasks, self.results)
        )
        self.process.start()

    def _discard(self) -> None:
        if self.process is None:
            return
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10.0)
        if self.process.is_alive():
            self.process.kill()
            self.process.join()
        self.tasks.close()
        self.results.close()
        self.process = None

    def solve(self, payload: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        if self.process is None or not self.process.is_alive():
            self._discard()
            self._start()
        self.tasks.put(payload)
        try:
            return self.results.get(timeout=PER_STAR_TIMEOUT_SECONDS)
        except queue.Empty:
            self._discard()
            self._start()
            return {
                "converged": False,
                "solver_reported_converged": False,
                "finite_final_state": False,
                "iterations_completed": None,
                "solver_outcome": "timeout",
                "diagnostics": None,
                "profile_path": None,
                "product_path": None,
                "error": None,
                "seconds": float(time.perf_counter() - started),
            }

    def close(self) -> None:
        if self.process is not None and self.process.is_alive():
            self.tasks.put(None)
            self.process.join(timeout=30.0)
        self._discard()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run_shard(
    root: Path, sample: str, shard: int, workers: int, resume: bool
) -> int:
    indices, paths = _write_seeds(root, sample)
    with np.load(paths["seeds"], allow_pickle=False) as seeds:
        labels = seeds["labels"]
        tau = seeds["tau"]
        mass = seeds["column_mass"]
        temperature = seeds["temperature"]
        log_opacity = seeds["log_rosseland_opacity"]
    positions = np.arange(indices.size, dtype=int)[
        np.arange(indices.size, dtype=int) % workers == shard
    ]
    shard_path = paths["shards"] / f"shard_{shard:02d}.jsonl"
    paths["shards"].mkdir(parents=True, exist_ok=True)
    previous = _read_jsonl(shard_path)
    if previous and not resume:
        raise SystemExit(
            f"{shard_path} already contains {len(previous)} rows; "
            "pass --resume or use a fresh campaign namespace"
        )
    by_index = {int(row["corpus_index"]): row for row in previous}
    if len(by_index) != len(previous):
        raise SystemExit(f"{shard_path} contains duplicate corpus indices")
    worker = _TimedWorker()
    try:
        with shard_path.open("a", encoding="utf-8") as handle:
            for done, position in enumerate(positions, start=1):
                index = int(indices[position])
                if index in by_index:
                    continue
                slug = _solver_slug(labels[position])
                record = {
                    "sample": sample,
                    "corpus_index": index,
                    "position": int(position),
                    "slug": slug,
                    "arm": ARM,
                    "iterations_per_trial": ITERATIONS,
                    **{
                        name: float(value)
                        for name, value in zip(LABEL_FIELDS, labels[position])
                    },
                }
                finite_seed = bool(
                    np.all(
                        np.isfinite(mass[position])
                        & np.isfinite(temperature[position])
                        & np.isfinite(log_opacity[position])
                    )
                    and np.all(mass[position] > 0.0)
                    and np.all(temperature[position] > 0.0)
                )
                if finite_seed:
                    record.update(
                        worker.solve(
                            {
                                "labels": labels[position],
                                "tau": tau,
                                "column_mass": mass[position],
                                "temperature": temperature[position],
                                "log_rosseland_opacity": log_opacity[position],
                                "profile_path": str(
                                    paths["profiles"] / f"{slug}.npz"
                                ),
                                "product_path": str(
                                    paths["products"] / f"{slug}.npz"
                                ),
                            }
                        )
                    )
                else:
                    record.update(
                        {
                            "converged": False,
                            "solver_reported_converged": False,
                            "finite_final_state": False,
                            "iterations_completed": 0,
                            "solver_outcome": "error",
                            "diagnostics": None,
                            "profile_path": None,
                            "product_path": None,
                            "error": "non-finite or non-positive frozen seed",
                            "seconds": 0.0,
                        }
                    )
                by_index[index] = record
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                print(
                    f"[shard {shard + 1}/{workers} {done}/{len(positions)}] "
                    f"{sample} index={index} outcome={record['solver_outcome']} "
                    f"iters={record['iterations_completed']} "
                    f"{record['seconds']:.1f}s",
                    flush=True,
                )
    finally:
        worker.close()
    return 0


def _merge_records(root: Path, sample: str, workers: int) -> dict[str, Any]:
    indices, paths = _write_seeds(root, sample)
    records: list[dict[str, Any]] = []
    for shard in range(workers):
        shard_path = paths["shards"] / f"shard_{shard:02d}.jsonl"
        if not shard_path.is_file():
            raise SystemExit(f"missing completed shard {shard_path}")
        records.extend(_read_jsonl(shard_path))
    by_index = {int(row["corpus_index"]): row for row in records}
    if len(by_index) != len(records):
        raise SystemExit("merged shards contain duplicate corpus indices")
    if set(by_index) != set(int(index) for index in indices):
        missing = sorted(set(int(index) for index in indices) - set(by_index))
        extra = sorted(set(by_index) - set(int(index) for index in indices))
        raise SystemExit(f"shard coverage mismatch; missing={missing}, extra={extra}")
    ordered = [by_index[int(index)] for index in indices]
    paths["records"].parent.mkdir(parents=True, exist_ok=True)
    paths["records"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered),
        encoding="utf-8",
    )
    outcomes = [str(row["solver_outcome"]) for row in ordered]
    converged = [row for row in ordered if bool(row["converged"])]
    result = {
        "campaign": CAMPAIGN,
        "sample": sample,
        "role": (
            "development replay"
            if sample == "development"
            else "post-hoc evaluation on a previously opened sample"
        ),
        "arm": ARM,
        "star_count": len(ordered),
        "converged_count": len(converged),
        "not_converged_count": outcomes.count("not_converged"),
        "timeout_count": outcomes.count("timeout"),
        "error_count": outcomes.count("error"),
        "finite_final_count": sum(
            row.get("finite_final_state") is True for row in ordered
        ),
        "iterations_per_trial": ITERATIONS,
        "per_star_timeout_seconds": PER_STAR_TIMEOUT_SECONDS,
        "records_path": str(paths["records"]),
        "records_sha256": sha256(paths["records"]),
        "records": ordered,
    }
    paths["result"].mkdir(parents=True, exist_ok=True)
    paths["solver"].write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _development_replay(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    historical = json.loads((root / HISTORICAL_DEVELOPMENT).read_text())
    historical_by_index = {
        int(row["corpus_index"]): row for row in historical["records"]
    }
    current_by_index = {
        int(row["corpus_index"]): row for row in result["records"]
    }
    outcome_mismatches = []
    iteration_mismatches = []
    for index in sorted(historical_by_index):
        old = historical_by_index[index]
        new = current_by_index.get(index)
        if new is None or bool(old["converged"]) != bool(new["converged"]):
            outcome_mismatches.append(index)
            continue
        if old.get("iterations_completed") != new.get("iterations_completed"):
            iteration_mismatches.append(index)
    replay = {
        "historical_path": str(HISTORICAL_DEVELOPMENT),
        "historical_sha256": sha256(root / HISTORICAL_DEVELOPMENT),
        "expected_converged_count": 54,
        "observed_converged_count": int(result["converged_count"]),
        "outcome_mismatch_indices": outcome_mismatches,
        "iteration_mismatch_indices": iteration_mismatches,
    }
    replay["matches"] = (
        int(result["converged_count"]) == 54
        and not outcome_mismatches
        and not iteration_mismatches
    )
    replay_path = _sample_paths(root, "development")["result"] / "replay_check.json"
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")
    return replay


def _run_sample(
    root: Path,
    sample: str,
    workers: int,
    resume: bool,
    historical_thread_replay: bool = False,
) -> int:
    if historical_thread_replay and (sample != "development" or workers != 1):
        raise SystemExit(
            "the historical-thread replay is restricted to one development "
            "worker; posthoc200 always uses the single-thread worker environment"
        )
    _write_source_manifest(root)
    _write_seeds(root, sample)
    paths = _sample_paths(root, sample)
    environment = dict(os.environ)
    if historical_thread_replay:
        for key in THREAD_LIMIT_KEYS:
            environment.pop(key, None)
        environment.update(HISTORICAL_REPLAY_ENV)
    else:
        environment.update(THREAD_ENV)
    recorded_thread_environment = {
        key: environment.get(key)
        for key in (*THREAD_LIMIT_KEYS, "NUMBA_THREADING_LAYER")
    }
    runtime_guard = paths["result"] / "runtime.json"
    runtime = {
        "campaign": CAMPAIGN,
        "sample": sample,
        "workers": workers,
        "iterations": ITERATIONS,
        "per_star_timeout_seconds": PER_STAR_TIMEOUT_SECONDS,
        "execution_role": (
            "provenance replay in the frozen historical thread environment"
            if historical_thread_replay
            else "formal single-thread-per-worker campaign"
        ),
        "thread_environment": recorded_thread_environment,
    }
    if runtime_guard.is_file():
        previous = json.loads(runtime_guard.read_text(encoding="utf-8"))
        if previous != runtime:
            raise SystemExit(
                f"{runtime_guard} disagrees with this invocation; resume refused"
            )
    else:
        runtime_guard.write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n"
        )
    if sample == "posthoc200":
        replay_path = (
            _sample_paths(root, "development")["result"] / "replay_check.json"
        )
        if not replay_path.is_file() or not json.loads(
            replay_path.read_text()
        ).get("matches"):
            raise SystemExit(
                "posthoc200 is locked until the development replay matches "
                "the frozen 54/60 result and every per-star iteration count"
            )
    commands = []
    environment["PYTHONPATH"] = str(root)
    for shard in range(workers):
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--root",
            str(root),
            "--stage",
            "_worker",
            "--sample",
            sample,
            "--workers",
            str(workers),
            "--shard",
            str(shard),
        ]
        if resume:
            command.append("--resume")
        commands.append(
            subprocess.Popen(command, cwd=root, env=environment)
        )
    failures = [process.wait() for process in commands]
    if any(code != 0 for code in failures):
        raise SystemExit(f"{sample} shard failures: {failures}")
    result = _merge_records(root, sample, workers)
    if sample == "development":
        replay = _development_replay(root, result)
        if not replay["matches"]:
            raise SystemExit(
                "development replay did not match the frozen run; "
                "posthoc200 and manuscript integration remain locked"
            )
    return 0


def _spectrum_pair(
    *,
    slug: str,
    physical_product: Path,
    physical_spectrum: Path,
    reference_spectrum: Path,
    reference_product: Path | None,
) -> dict[str, Any]:
    from experiments.reduced_state_emulator.spectral_gate import (
        _absolute_stats,
        _continuum_scaled_stats,
        _load_spectrum_npz,
        _relative_stats,
        _synthesize_one,
    )

    if not physical_spectrum.is_file():
        _synthesize_one(
            physical_product,
            physical_spectrum,
            wavelength_start_nm=SPECTRUM_WINDOW_NM[0],
            wavelength_end_nm=SPECTRUM_WINDOW_NM[1],
            resolution=SPECTRUM_RESOLUTION,
            molecular_lines=True,
            device=None,
            dtype=SPECTRUM_DTYPE,
        )
    if not reference_spectrum.is_file():
        if reference_product is None or not reference_product.is_file():
            raise FileNotFoundError(f"missing reference spectrum/product for {slug}")
        _synthesize_one(
            reference_product,
            reference_spectrum,
            wavelength_start_nm=SPECTRUM_WINDOW_NM[0],
            wavelength_end_nm=SPECTRUM_WINDOW_NM[1],
            resolution=SPECTRUM_RESOLUTION,
            molecular_lines=True,
            device=None,
            dtype=SPECTRUM_DTYPE,
        )
    reference = _load_spectrum_npz(reference_spectrum)
    candidate = _load_spectrum_npz(physical_spectrum)
    return {
        "slug": slug,
        "normalized_flux": _absolute_stats(
            candidate["normalized_flux"], reference["normalized_flux"]
        ),
        "flux_total": _continuum_scaled_stats(
            candidate["flux_total"],
            reference["flux_total"],
            reference["flux_continuum"],
        ),
        "flux_continuum": _relative_stats(
            candidate["flux_continuum"], reference["flux_continuum"]
        ),
    }


def _spectrum_worker(payload: dict[str, Any]) -> dict[str, Any]:
    return _spectrum_pair(
        slug=payload["slug"],
        physical_product=Path(payload["physical_product"]),
        physical_spectrum=Path(payload["physical_spectrum"]),
        reference_spectrum=Path(payload["reference_spectrum"]),
        reference_product=(
            None
            if payload["reference_product"] is None
            else Path(payload["reference_product"])
        ),
    )


def _run_spectra(root: Path, sample: str, workers: int) -> int:
    paths = _sample_paths(root, sample)
    if not paths["solver"].is_file():
        raise SystemExit(f"missing solver result {paths['solver']}")
    solver = json.loads(paths["solver"].read_text())
    reference_spectra = root / REFERENCE_SPECTRA[sample]
    reference_products = (
        root / REFERENCE_PRODUCTS[sample]
        if sample in REFERENCE_PRODUCTS
        else None
    )
    physical_spectra = paths["spectra"] / ARM
    comparison_reference = paths["spectra"] / "six_field_reference"
    physical_spectra.mkdir(parents=True, exist_ok=True)
    comparison_reference.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for record in solver["records"]:
        slug = str(record["slug"])
        if not bool(record["converged"]):
            excluded.append({"slug": slug, "reason": "physical solver did not converge"})
            continue
        physical_product = paths["products"] / f"{slug}.npz"
        if not physical_product.is_file():
            excluded.append({"slug": slug, "reason": "physical spectrum product missing"})
            continue
        reference_spectrum = reference_spectra / f"{slug}.npz"
        reference_product = (
            reference_products / f"{slug}.npz"
            if reference_products is not None
            else None
        )
        if not reference_spectrum.is_file() and (
            reference_product is None or not reference_product.is_file()
        ):
            excluded.append({"slug": slug, "reason": "six-field reference product missing"})
            continue
        payloads.append(
            {
                "slug": slug,
                "physical_product": str(physical_product),
                "physical_spectrum": str(physical_spectra / f"{slug}.npz"),
                "reference_spectrum": str(
                    reference_spectrum
                    if reference_spectrum.is_file()
                    else comparison_reference / f"{slug}.npz"
                ),
                "reference_product": (
                    None if reference_product is None else str(reference_product)
                ),
            }
        )
    if workers <= 1:
        rows = [_spectrum_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_spectrum_worker, payloads))
    for position, row in enumerate(rows, start=1):
        print(
            f"[{position}/{len(rows)}] {sample} {row['slug']} "
            f"norm={row['normalized_flux']['max']:.3e}",
            flush=True,
        )
    fields = ("normalized_flux", "flux_total", "flux_continuum")
    result: dict[str, Any] = {
        "campaign": CAMPAIGN,
        "sample": sample,
        "role": (
            "development replay"
            if sample == "development"
            else "post-hoc evaluation on a previously opened sample"
        ),
        "bar": DEFAULT_BAR,
        "window_nm": list(SPECTRUM_WINDOW_NM),
        "resolution": SPECTRUM_RESOLUTION,
        "dtype": SPECTRUM_DTYPE,
        "gated_star_count": len(rows),
        "excluded_star_count": len(excluded),
        "excluded_stars": excluded,
        "per_star": rows,
    }
    for field in fields:
        maxima = np.asarray([row[field]["max"] for row in rows], dtype=float)
        result[field] = {
            "median_over_stars": float(np.median(maxima)) if maxima.size else None,
            "p95_over_stars": (
                float(np.percentile(maxima, 95.0)) if maxima.size else None
            ),
            "max_over_stars": float(np.max(maxima)) if maxima.size else None,
            "stars_over_bar": int(np.count_nonzero(maxima > DEFAULT_BAR)),
        }
    paths["spectral_gate"].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return 0


def _load_truth_profiles(root: Path, indices: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(root / DEFAULT_CORPUS, allow_pickle=False) as payload:
        names = tuple(str(value) for value in payload["target_fields"])
        profiles = np.asarray(payload["atmosphere_profiles"], dtype=np.float64)
    return {
        name: profiles[indices, :, position]
        for position, name in enumerate(names)
    }


def _profile_error(prediction: np.ndarray, truth: np.ndarray, field: str) -> np.ndarray:
    if field == "column_mass":
        return np.abs(
            np.log10(np.maximum(prediction, 1.0e-300))
            - np.log10(np.maximum(truth, 1.0e-300))
        )
    scale = np.maximum(np.abs(truth), 1.0e-300)
    return np.abs(prediction - truth) / scale


def _analyze_sample(root: Path, sample: str) -> int:
    indices, paths = _write_seeds(root, sample)
    if not paths["solver"].is_file():
        raise SystemExit(f"missing solver result {paths['solver']}")
    solver = json.loads(paths["solver"].read_text())
    by_slug = {str(row["slug"]): row for row in solver["records"]}
    corpus = load_strict_truth(root / DEFAULT_CORPUS)
    truth = _load_truth_profiles(root, indices)
    with np.load(paths["seeds"], allow_pickle=False) as seeds:
        seed_mass = seeds["column_mass"]
        seed_temperature = seeds["temperature"]
    seed_errors = {
        "column_mass": _profile_error(seed_mass, truth["column_mass"], "column_mass"),
        "temperature": _profile_error(
            seed_temperature, truth["temperature"], "temperature"
        ),
    }
    final_arrays: dict[str, list[np.ndarray]] = {field: [] for field in PROFILE_FIELDS}
    truth_arrays: dict[str, list[np.ndarray]] = {field: [] for field in PROFILE_FIELDS}
    positions: list[int] = []
    for position, index in enumerate(indices):
        labels = corpus.labels[index]
        slug = _solver_slug(labels)
        record = by_slug[slug]
        profile_path = paths["profiles"] / f"{slug}.npz"
        if not bool(record["converged"]) or not profile_path.is_file():
            continue
        with np.load(profile_path, allow_pickle=False) as profile:
            for field in PROFILE_FIELDS:
                final_arrays[field].append(np.asarray(profile[field], dtype=np.float64))
                truth_arrays[field].append(truth[field][position])
        positions.append(position)
    final_errors = {
        field: _profile_error(
            np.asarray(final_arrays[field]),
            np.asarray(truth_arrays[field]),
            field,
        )
        for field in PROFILE_FIELDS
    }
    paths["result"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["profile_metrics"],
        corpus_indices=indices,
        seed_column_mass_error=seed_errors["column_mass"],
        seed_temperature_error=seed_errors["temperature"],
        converged_positions=np.asarray(positions, dtype=np.int64),
        **{
            f"final_{field}_error": values
            for field, values in final_errors.items()
        },
    )

    def summarize(values: np.ndarray) -> dict[str, float]:
        flat = np.asarray(values, dtype=np.float64).ravel()
        return {
            "median": float(np.median(flat)),
            "p95": float(np.percentile(flat, 95.0)),
            "maximum": float(np.max(flat)),
        }

    profile_summary = {
        "campaign": CAMPAIGN,
        "sample": sample,
        "seed": {
            field: summarize(values) for field, values in seed_errors.items()
        },
        "final_converged_star_count": len(positions),
        "final": {
            field: summarize(values) for field, values in final_errors.items()
        },
    }
    paths["profile_summary"].write_text(
        json.dumps(profile_summary, indent=2, sort_keys=True) + "\n"
    )
    converged_iterations = np.asarray(
        [
            int(row["iterations_completed"])
            for row in solver["records"]
            if bool(row["converged"])
        ],
        dtype=int,
    )
    residual = json.loads(
        (root / HISTORICAL_RESIDUAL).read_text(encoding="utf-8")
    )
    spectral = (
        json.loads(paths["spectral_gate"].read_text())
        if paths["spectral_gate"].is_file()
        else None
    )
    summary = {
        "campaign": CAMPAIGN,
        "sample": sample,
        "role": solver["role"],
        "star_count": solver["star_count"],
        "converged_count": solver["converged_count"],
        "not_converged_count": solver["not_converged_count"],
        "timeout_count": solver["timeout_count"],
        "error_count": solver["error_count"],
        "iterations_among_converged": {
            "mean": float(np.mean(converged_iterations)),
            "median": float(np.median(converged_iterations)),
            "maximum": int(np.max(converged_iterations)),
            "more_than_15_count": int(np.count_nonzero(converged_iterations > 15)),
        },
        "profile_metrics": profile_summary,
        "spectra": spectral,
        "development_residual_iter100": (
            {
                "star_count": residual["star_count"],
                "converged_count": residual["converged_count"],
                "timeout_count": residual["timeout_count"],
                "iterations": residual["iterations_per_trial"],
                "per_star_timeout_seconds": residual[
                    "per_star_timeout_seconds"
                ],
            }
            if sample == "development"
            else None
        ),
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


def _write_campaign_manifest(root: Path) -> None:
    result_root = root / "results" / CAMPAIGN
    run_root = root / "runs" / CAMPAIGN
    source_manifest = result_root / "source_manifest.json"
    outputs = sorted(
        path
        for path in result_root.rglob("*")
        if path.is_file() and path.name != "campaign.json"
    )
    payload = {
        "campaign": CAMPAIGN,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "sample_runtime": {
            sample: json.loads(
                (result_root / sample / "runtime.json").read_text(
                    encoding="utf-8"
                )
            )
            for sample in ("development", "posthoc200")
        },
        "source_manifest": str(source_manifest.relative_to(root)),
        "source_manifest_sha256": sha256(source_manifest),
        "development_replay_matches": json.loads(
            (result_root / "development/replay_check.json").read_text()
        )["matches"],
        "posthoc200_is_new_blind_test": False,
        "result_sha256": {
            str(path.relative_to(root)): sha256(path) for path in outputs
        },
        "run_root": str(run_root.relative_to(root)),
    }
    (result_root / "campaign.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("development", "posthoc200", "spectra", "analyze", "all", "_worker"),
        required=True,
    )
    parser.add_argument(
        "--sample", choices=("development", "posthoc200"), default="development"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--historical-thread-replay",
        action="store_true",
        help=(
            "development only: reproduce the frozen run with one worker and "
            "the historical unconstrained Numba thread count"
        ),
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.stage == "_worker":
        if args.shard is None or not 0 <= args.shard < args.workers:
            raise SystemExit("_worker requires 0 <= --shard < --workers")
        return _run_shard(
            root, args.sample, args.shard, args.workers, args.resume
        )
    if args.stage in ("development", "posthoc200"):
        return _run_sample(
            root,
            args.stage,
            args.workers,
            args.resume,
            args.historical_thread_replay,
        )
    if args.stage == "spectra":
        return _run_spectra(root, args.sample, args.workers)
    if args.stage == "analyze":
        return _analyze_sample(root, args.sample)

    _run_sample(
        root,
        "development",
        1 if args.historical_thread_replay else args.workers,
        args.resume,
        args.historical_thread_replay,
    )
    _run_spectra(root, "development", args.workers)
    _analyze_sample(root, "development")
    _run_sample(root, "posthoc200", args.workers, args.resume)
    _run_spectra(root, "posthoc200", args.workers)
    _analyze_sample(root, "posthoc200")
    _write_campaign_manifest(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
