"""Run the clean-base calibration campaign for initializer improvement.

This driver is intentionally limited to the opened 400-star calibration set.
The new sealed holdout is read only as an exclusion during training and is not
predicted, solved, or synthesized here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from bench.labels import LABEL_FIELDS, StellarLabels


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = Path("results/initializer_calibration_20260812.json")
SEALED_HOLDOUT = Path("results/sealed_initializer_holdout_20260812.json")
RUN_ROOT = Path("runs/initializer_improvement_20260812/calibration400")
RESULT_ROOT = Path("results/initializer_improvement_20260812/calibration400")
MODEL_ROOT = Path(
    "artifacts/reduced_state_emulator/initializer_improvement_20260812/base_ensemble"
)
PREDICTION = RESULT_ROOT / "base_ensemble.npz"
SEEDS = "20260812,20260813,20260814"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _prepare_spectral_subset(
    calibration_path: Path, products_dir: Path, subset_dir: Path
) -> dict:
    manifest = json.loads(calibration_path.read_text())
    requested_indices = {
        int(value) for value in manifest["spectral_selection"]["star_indices"]
    }
    corpus_path = (
        REPO_ROOT
        / "source_data_files"
        / "atmosphere_emulator"
        / "five_label"
        / "strict_truth_52199.npz"
    )
    with np.load(corpus_path, allow_pickle=False) as corpus:
        labels_json = {
            index: json.loads(str(corpus["labels_json"][index]))
            for index in requested_indices
        }
    requested_slugs = [
        StellarLabels(
            **{field: float(labels_json[index][field]) for field in LABEL_FIELDS}
        ).slug
        for index in requested_indices
    ]
    result = {
        "selected_before_outcomes": True,
        "requested_star_indices": sorted(requested_indices),
        "requested_slugs": sorted(requested_slugs),
        "arms": {},
    }
    for arm in ("production_six_field", "learned_reduced_state"):
        destination = subset_dir / arm
        destination.mkdir(parents=True, exist_ok=True)
        linked = []
        missing = []
        for slug in requested_slugs:
            source = (products_dir / arm / f"{slug}.npz").resolve()
            target = destination / source.name
            if not source.is_file():
                missing.append(slug)
                continue
            if target.exists() or target.is_symlink():
                if target.resolve() != source:
                    raise RuntimeError(f"unexpected existing subset link: {target}")
            else:
                target.symlink_to(os.path.relpath(source, start=destination.resolve()))
            linked.append(slug)
        result["arms"][arm] = {"linked": linked, "missing": missing}
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-spectra", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 8:
        raise SystemExit("--workers must be between 1 and 8 on the 156 GB GPU node")

    os.chdir(REPO_ROOT)
    for path in (CALIBRATION, SEALED_HOLDOUT):
        if not path.is_file():
            raise SystemExit(f"missing required manifest: {path}")
    holdout = json.loads(SEALED_HOLDOUT.read_text())
    if not holdout.get("sealed") or holdout.get("opened"):
        raise SystemExit("sealed holdout contract is missing or already opened")
    calibration = json.loads(CALIBRATION.read_text())
    if len(calibration.get("star_indices", [])) != 400:
        raise SystemExit("calibration manifest must contain exactly 400 stars")

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "NUMBA_THREADING_LAYER": "workqueue",
            "NUMBA_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES", "0"),
        }
    )
    state_path = RESULT_ROOT / "long_run_state.json"
    state = {
        "format": "payne_zero_initializer_improvement_long_v1",
        "host": socket.gethostname(),
        "python": sys.executable,
        "started_unix": time.time(),
        "calibration_manifest": str(CALIBRATION),
        "calibration_sha256": _sha256(CALIBRATION),
        "sealed_holdout_manifest": str(SEALED_HOLDOUT),
        "sealed_holdout_sha256": _sha256(SEALED_HOLDOUT),
        "sealed_holdout_opened": False,
        "workers": args.workers,
        "stages": [],
    }

    python = sys.executable
    stages: list[tuple[str, list[str], Path]] = [
        (
            "train_clean_base_ensemble",
            [
                python,
                "-m",
                "experiments.reduced_state_emulator.train_physical",
                "--development-from",
                "results/sealed_solver_subset_20260808.json",
                "--audit-manifest",
                "results/sealed_audit_20260808.json",
                "--exclude-manifest",
                "results/reconstruction_metrics.json",
                "--exclude-manifest",
                "results/sealed_audit_20260811.json",
                "--exclude-manifest",
                str(CALIBRATION),
                "--exclude-manifest",
                str(SEALED_HOLDOUT),
                "--out",
                str(MODEL_ROOT),
                "--seeds",
                SEEDS,
                "--width",
                "512",
                "--depth",
                "4",
                "--batch-size",
                "1024",
                "--epochs",
                "300",
                "--patience",
                "60",
                "--dtype",
                "float64",
                "--device",
                "cuda",
                "--hard-weight",
                "3",
                "--hard-region",
                "solver_tail",
                "--tail-weight",
                "0.1",
                "--surface-weight",
                "0",
            ],
            MODEL_ROOT / "training_physical_summary.json",
        ),
        (
            "predict_calibration400",
            [
                python,
                "-m",
                "experiments.reduced_state_emulator.predict_physical",
                "--indices-from",
                str(CALIBRATION),
                "--checkpoint-dir",
                str(MODEL_ROOT),
                "--seeds",
                SEEDS,
                "--out",
                str(PREDICTION),
            ],
            PREDICTION.with_suffix(".json"),
        ),
        (
            "profile_calibration400",
            [
                python,
                "-m",
                "experiments.reduced_state_emulator.evaluate_physical",
                "--indices-from",
                str(CALIBRATION),
                "--prediction",
                str(PREDICTION),
                "--out",
                str(RESULT_ROOT / "profile_gate.json"),
                "--name",
                "initializer_calibration400_clean_base",
            ],
            RESULT_ROOT / "profile_gate.json",
        ),
        (
            "candidate_real_solver400",
            [
                python,
                "-m",
                "experiments.reduced_state_emulator.run_learned_restart",
                "--held-out-from",
                str(CALIBRATION),
                "--prediction",
                str(PREDICTION),
                "--arm",
                "physical_ensemble",
                "--workers",
                str(args.workers),
                "--skip-production-arm",
                "--products-dir",
                str(RUN_ROOT / "products"),
                "--records-dir",
                str(RUN_ROOT / "candidate_solver"),
                "--results-dir",
                str(RESULT_ROOT / "candidate_solver"),
            ],
            RESULT_ROOT
            / "candidate_solver"
            / "convergence_metrics_learned_physical_ensemble.json",
        ),
        (
            "production_real_solver400",
            [
                python,
                "-m",
                "experiments.reduced_state_emulator.run_learned_restart",
                "--held-out-from",
                str(CALIBRATION),
                "--workers",
                str(args.workers),
                "--production-only",
                "--products-dir",
                str(RUN_ROOT / "products"),
                "--records-dir",
                str(RUN_ROOT / "production_solver"),
                "--results-dir",
                str(RESULT_ROOT / "production_solver"),
            ],
            RESULT_ROOT
            / "production_solver"
            / "convergence_metrics_production_baseline.json",
        ),
    ]

    def run_stage(name: str, command: list[str], sentinel: Path) -> None:
        entry = {
            "name": name,
            "command": command,
            "sentinel": str(sentinel),
            "started_unix": time.time(),
        }
        state["stages"].append(entry)
        print(f"\n=== {name} ===", flush=True)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            entry["status"] = "dry_run"
            return
        if sentinel.is_file():
            entry.update(
                {
                    "status": "already_complete",
                    "sentinel_sha256": _sha256(sentinel),
                    "finished_unix": time.time(),
                }
            )
            print(f"already complete: {sentinel}", flush=True)
            _write_state(state_path, state)
            return
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
        except Exception as exc:
            entry.update(
                {
                    "status": "failed",
                    "error": repr(exc),
                    "finished_unix": time.time(),
                }
            )
            _write_state(state_path, state)
            raise
        if not sentinel.is_file():
            raise RuntimeError(f"stage {name} returned without writing {sentinel}")
        entry.update(
            {
                "status": "complete",
                "sentinel_sha256": _sha256(sentinel),
                "finished_unix": time.time(),
            }
        )
        _write_state(state_path, state)

    for stage in stages:
        run_stage(*stage)

    if not args.skip_spectra:
        subset_manifest = RESULT_ROOT / "spectral_subset60.json"
        print("\n=== prepare_preselected_spectral60 ===", flush=True)
        if args.dry_run:
            print(f"would write {subset_manifest}", flush=True)
        elif not subset_manifest.is_file():
            subset = _prepare_spectral_subset(
                CALIBRATION,
                RUN_ROOT / "products",
                RUN_ROOT / "spectral60_products",
            )
            subset_manifest.parent.mkdir(parents=True, exist_ok=True)
            subset_manifest.write_text(json.dumps(subset, indent=2) + "\n")
        spectral_command = [
            python,
            "-m",
            "experiments.reduced_state_emulator.spectral_gate",
            "--products-dir",
            str(RUN_ROOT / "spectral60_products"),
            "--spectra-dir",
            str(RUN_ROOT / "spectral60_spectra"),
            "--out",
            str(RESULT_ROOT / "spectral_gate60.json"),
            "--workers",
            "1",
            "--device",
            "cuda",
            "--dtype",
            "float64",
            "--wavelength-start-nm",
            "400",
            "--wavelength-end-nm",
            "900",
            "--resolution",
            "20000",
        ]
        run_stage(
            "preselected_spectral60",
            spectral_command,
            RESULT_ROOT / "spectral_gate60.json",
        )

    if not args.dry_run:
        state["status"] = "complete"
        state["finished_unix"] = time.time()
        _write_state(state_path, state)
    print(f"\nlong-run state: {state_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
