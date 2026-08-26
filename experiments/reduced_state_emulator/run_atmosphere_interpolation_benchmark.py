"""Run the complete-state atmosphere interpolation benchmark.

The existing grey/two-field/six-field results are reused through symlinks. Only
the new full six-field interpolation arm runs the physical solver here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGET_MANIFEST = Path(
    "results/grey_start_benchmark_20260812/calibration_spectral60_manifest.json"
)
OLD_RUN_ROOT = Path("runs/grey_start_benchmark_20260812/calibration_spectral60")
OLD_SPECTRA_ROOT = Path(
    "runs/initializer_improvement_20260812/calibration400/spectral60_spectra"
)
DEFAULT_RUN_ROOT = Path("runs/atmosphere_interpolation_benchmark_20260813")
DEFAULT_RESULT_ROOT = Path("results/atmosphere_interpolation_benchmark_20260813")
INTERPOLATION_ARM = "interpolated_full_state"
REUSED_ARMS = ("production_six_field", "learned_reduced_state", "grey15")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()), target_is_directory=source.is_dir())


def prepare_reused_products_and_spectra(run_root: Path) -> dict:
    """Expose the already completed three arms under the new benchmark root."""

    if not OLD_RUN_ROOT.is_dir():
        raise FileNotFoundError(f"existing solver products are missing: {OLD_RUN_ROOT}")
    reused = {"products": [], "spectra": []}
    for arm in REUSED_ARMS:
        source = OLD_RUN_ROOT / "products" / arm
        if source.is_dir():
            _link_or_copy(source, run_root / "products" / arm)
            reused["products"].append(arm)
        source = OLD_SPECTRA_ROOT / arm
        if source.is_dir():
            _link_or_copy(source, run_root / "spectra" / arm)
            reused["spectra"].append(arm)
    return reused


def _run(command: list[str], *, env: dict[str, str], allow_failure: bool = False) -> None:
    print("RUN", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"command failed with exit {completed.returncode}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--manifest", type=Path, default=TARGET_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--skip-spectra", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 6:
        raise SystemExit("--workers must be between 1 and 6")
    if not args.manifest.is_file():
        raise SystemExit(f"target manifest is missing: {args.manifest}")

    os.chdir(REPO_ROOT)
    run_root = args.run_root
    result_root = args.result_root
    run_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    reused = prepare_reused_products_and_spectra(run_root)

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "NUMBA_THREADING_LAYER": "workqueue",
            "NUMBA_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    python = sys.executable
    corpus = args.corpus
    if corpus is None:
        corpus = REPO_ROOT / "source_data_files" / "atmosphere_emulator" / "five_label" / "strict_truth_52199.npz"
    solve_result = result_root / f"convergence_{INTERPOLATION_ARM}.json"
    solve_command = [
        python,
        "-m",
        "experiments.reduced_state_emulator.grey_start_benchmark",
        "--arm",
        INTERPOLATION_ARM,
        "--manifest",
        str(args.manifest),
        "--corpus",
        str(corpus),
        "--workers",
        str(args.workers),
        "--run-root",
        str(run_root),
        "--result-root",
        str(result_root),
        "--interpolation-neighbours",
        "8",
        "--interpolation-power",
        "2.0",
    ]
    state = {
        "format": "payne_zero_atmosphere_interpolation_benchmark_v1",
        "target_manifest": str(args.manifest),
        "corpus": str(corpus),
        "workers": args.workers,
        "primary_iteration_cap": 15,
        "interpolation": {
            "arm": INTERPOLATION_ARM,
            "labels": ["5040/Teff", "logg", "metallicity", "alpha_enhancement"],
            "neighbours": 8,
            "power": 2.0,
            "microturbulence_in_distance": False,
            "full_state": True,
        },
        "provenance": {
            "target_manifest_sha256": _sha256(args.manifest),
            "corpus_sha256": _sha256(corpus),
            "interpolation_script_sha256": _sha256(
                REPO_ROOT / "experiments" / "reduced_state_emulator" / "grey_start_benchmark.py"
            ),
            "benchmark_driver_sha256": _sha256(Path(__file__).resolve()),
        },
        "reused_arms": reused,
        "solve_command": solve_command,
    }
    (result_root / "benchmark_state.json").write_text(json.dumps(state, indent=2) + "\n")

    if not args.dry_run and not solve_result.is_file():
        _run(solve_command, env=environment)
    elif args.dry_run:
        print("DRY RUN", " ".join(solve_command), flush=True)

    if not args.skip_spectra and not args.dry_run:
        products = run_root / "products"
        spectra = run_root / "spectra"
        for baseline, suffix in (
            ("production_six_field", "interpolation_vs_six"),
            ("learned_reduced_state", "interpolation_vs_two"),
            ("grey15", "interpolation_vs_grey15"),
        ):
            _run(
                [
                    python,
                    "-m",
                    "experiments.reduced_state_emulator.spectral_gate",
                    "--products-dir",
                    str(products),
                    "--spectra-dir",
                    str(spectra),
                    "--out",
                    str(result_root / f"spectral_{suffix}.json"),
                    "--baseline-arm",
                    baseline,
                    "--candidate-arm",
                    INTERPOLATION_ARM,
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
                ],
                env=environment,
                allow_failure=True,
            )
    state["status"] = "complete" if solve_result.is_file() else "dry_run"
    state["result_files"] = sorted(str(path) for path in result_root.glob("*.json"))
    (result_root / "benchmark_state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(f"benchmark state: {result_root / 'benchmark_state.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
