"""Finish the grey/two-field/six-field comparison after grey15 completes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("results/grey_start_benchmark_20260812/calibration_spectral60_manifest.json")
PREDICTION = Path("results/initializer_improvement_20260812/calibration400/base_ensemble.npz")
GREY_RUN = Path("runs/grey_start_benchmark_20260812/calibration_spectral60")
GREY_RESULTS = Path("results/grey_start_benchmark_20260812/calibration_spectral60")
PRODUCTS = GREY_RUN / "products"
SPECTRA = Path("runs/initializer_improvement_20260812/calibration400/spectral60_spectra")
RESULTS = Path("results/initializer_improvement_20260812/calibration400")


def run(command: list[str], *, allow_gate_failure: bool = False) -> None:
    print("RUN", " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode and not allow_gate_failure:
        raise RuntimeError(f"command failed with exit {result.returncode}: {command}")


def combine_grey_products() -> int:
    linked = {}
    for kind in ("products", "profiles"):
        destination = GREY_RUN / kind / "grey30_complete"
        destination.mkdir(parents=True, exist_ok=True)
        for old in destination.glob("*.npz"):
            old.unlink()
        for arm in ("grey15", "grey30"):
            for source in (GREY_RUN / kind / arm).glob("*.npz"):
                target = destination / source.name
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(os.path.relpath(source.resolve(), destination.resolve()))
                if kind == "products":
                    linked[source.stem] = arm
    mapping = GREY_RESULTS / "grey30_complete_products.json"
    mapping.write_text(json.dumps({"count": len(linked), "source_arm": linked}, indent=2) + "\n")
    print(f"linked {len(linked)} converged grey products", flush=True)
    return len(linked)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--wait-pid", type=int, default=None)
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    if args.wait_pid is not None:
        while Path(f"/proc/{args.wait_pid}").exists():
            print(f"waiting for grey15 pid {args.wait_pid}", flush=True)
            time.sleep(30)
    grey15 = GREY_RESULTS / "convergence_grey15.json"
    if not grey15.is_file():
        raise RuntimeError("grey15 did not produce its convergence result")

    python = sys.executable
    grey30 = GREY_RESULTS / "convergence_grey30.json"
    if not grey30.is_file():
        run(
            [
                python, "-m", "experiments.reduced_state_emulator.grey_start_benchmark",
                "--arm", "grey30", "--manifest", str(MANIFEST),
                "--prediction", str(PREDICTION), "--workers", str(args.workers),
                "--run-root", str(GREY_RUN), "--result-root", str(GREY_RESULTS),
            ]
        )
    if combine_grey_products() == 0:
        raise RuntimeError("no converged grey products are available for synthesis")

    for arm in ("learned_reduced_state", "production_six_field"):
        result_path = GREY_RESULTS / f"convergence_{arm}.json"
        if not result_path.is_file():
            run(
                [
                    python, "-m", "experiments.reduced_state_emulator.grey_start_benchmark",
                    "--arm", arm, "--manifest", str(MANIFEST),
                    "--prediction", str(PREDICTION), "--workers", str(args.workers),
                    "--run-root", str(GREY_RUN), "--result-root", str(GREY_RESULTS),
                ]
            )

    for baseline, candidate, suffix in (
        ("production_six_field", "learned_reduced_state", "two_vs_six_unified"),
        ("production_six_field", "grey15", "grey15_vs_six"),
        ("learned_reduced_state", "grey15", "grey15_vs_two"),
        ("production_six_field", "grey30_complete", "grey_vs_six"),
        ("learned_reduced_state", "grey30_complete", "grey_vs_two"),
    ):
        run(
            [
                python, "-m", "experiments.reduced_state_emulator.spectral_gate",
                "--products-dir", str(PRODUCTS), "--spectra-dir", str(SPECTRA),
                "--out", str(RESULTS / f"spectral_{suffix}.json"),
                "--baseline-arm", baseline, "--candidate-arm", candidate,
                "--workers", "1", "--device", "cuda", "--dtype", "float64",
                "--wavelength-start-nm", "400", "--wavelength-end-nm", "900",
                "--resolution", "20000",
            ],
            allow_gate_failure=True,
        )
    run(
        [
            python,
            "-m",
            "experiments.reduced_state_emulator.summarize_three_initializer_comparison",
            "--run-root",
            str(GREY_RUN),
            "--result-root",
            str(GREY_RESULTS),
        ]
    )
    print("grey/two-field/six-field spectral comparison complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
