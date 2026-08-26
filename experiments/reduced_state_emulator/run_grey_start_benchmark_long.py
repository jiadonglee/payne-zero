"""Resumable long driver for the grey-start benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = Path("runs/grey_start_benchmark_20260812")
RESULT_ROOT = Path("results/grey_start_benchmark_20260812")
PREREQUISITE = Path(
    "results/initializer_improvement_20260812/calibration400/long_run_state.json"
)


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _prerequisite_finished() -> bool:
    if not PREREQUISITE.is_file():
        return False
    try:
        state = json.loads(PREREQUISITE.read_text())
        return state.get("status") == "complete" or any(
            stage.get("status") == "failed" for stage in state.get("stages", [])
        )
    except (OSError, json.JSONDecodeError):
        return False


def _prepare_combined_grey30() -> None:
    for kind in ("products", "profiles"):
        destination = RUN_ROOT / kind / "grey30_complete"
        destination.mkdir(parents=True, exist_ok=True)
        for arm in ("grey15", "grey30"):
            for source in (RUN_ROOT / kind / arm).glob("*.npz"):
                target = destination / source.name
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(os.path.relpath(source.resolve(), destination.resolve()))


def _other_payne_processes() -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, check=True
    )
    own_pid = os.getpid()
    conflicts = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit():
            continue
        pid, command = int(fields[0]), fields[1]
        if pid == own_pid:
            continue
        if "experiments.reduced_state_emulator" in command and (
            "run_initializer_improvement_long" in command
            or "run_learned_restart" in command
            or "spectral_gate" in command
        ):
            conflicts.append(line.strip())
    return conflicts


def _all_long_run_stages_terminal() -> bool:
    if not PREREQUISITE.is_file():
        return False
    try:
        state = json.loads(PREREQUISITE.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return state.get("status") == "complete" or any(
        stage.get("status") == "failed" for stage in state.get("stages", [])
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--wait-for-prerequisite", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--skip-spectra", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.workers > 6:
        raise SystemExit("--workers must be between 1 and 6")

    os.chdir(REPO_ROOT)
    if args.wait_for_prerequisite and not args.dry_run:
        while not _all_long_run_stages_terminal():
            print(f"waiting for {PREREQUISITE} to finish", flush=True)
            time.sleep(max(10, args.poll_seconds))
    elif not args.dry_run and not _prerequisite_finished():
        raise SystemExit(
            f"prerequisite long run is not complete: {PREREQUISITE}; "
            "use --wait-for-prerequisite"
        )
    if not args.dry_run:
        while True:
            conflicts = _other_payne_processes()
            if not conflicts:
                break
            if not args.wait_for_prerequisite:
                raise RuntimeError(
                    "another payne-zero long process is still active:\n"
                    + "\n".join(conflicts)
                )
            print("waiting for active payne-zero workers to exit", flush=True)
            time.sleep(max(10, args.poll_seconds))

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
    python = sys.executable
    state_path = RESULT_ROOT / "long_run_state.json"
    state = {
        "format": "payne_zero_grey_start_benchmark_v1",
        "started_unix": time.time(),
        "prerequisite": str(PREREQUISITE),
        "workers": args.workers,
        "stages": [],
    }

    stages = []
    for arm in (
        "production_six_field",
        "learned_reduced_state",
        "truth_reduced_state",
        "grey15",
        "grey30",
        "grey_perturbed",
    ):
        stages.append(
            (
                f"solve_{arm}",
                [
                    python,
                    "-m",
                    "experiments.reduced_state_emulator.grey_start_benchmark",
                    "--arm",
                    arm,
                    "--workers",
                    str(args.workers),
                ],
                RESULT_ROOT / f"convergence_{arm}.json",
                True,
            )
        )

    def run_stage(name: str, command: list[str], sentinel: Path, check: bool) -> None:
        entry = {"name": name, "command": command, "started_unix": time.time()}
        state["stages"].append(entry)
        print(f"\n=== {name} ===\n{shlex.join(command)}", flush=True)
        if args.dry_run:
            entry["status"] = "dry_run"
            return
        if sentinel.is_file():
            entry["status"] = "already_complete"
            return
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
        if not sentinel.is_file() or (check and completed.returncode != 0):
            entry.update({"status": "failed", "returncode": completed.returncode})
            _write_state(state_path, state)
            raise RuntimeError(f"stage failed: {name}")
        entry.update(
            {
                "status": "complete" if completed.returncode == 0 else "gate_failed",
                "returncode": completed.returncode,
                "finished_unix": time.time(),
            }
        )
        _write_state(state_path, state)

    for stage in stages:
        run_stage(*stage)

    if not args.dry_run:
        _prepare_combined_grey30()

    if not args.skip_spectra:
        for candidate in (
            "learned_reduced_state",
            "truth_reduced_state",
            "grey15",
            "grey30_complete",
        ):
            output_name = "grey30" if candidate == "grey30_complete" else candidate
            run_stage(
                f"spectral_{output_name}",
                [
                    python,
                    "-m",
                    "experiments.reduced_state_emulator.spectral_gate",
                    "--products-dir",
                    str(RUN_ROOT / "products"),
                    "--spectra-dir",
                    str(RUN_ROOT / "spectra"),
                    "--out",
                    str(RESULT_ROOT / f"spectral_{output_name}.json"),
                    "--baseline-arm",
                    "production_six_field",
                    "--candidate-arm",
                    candidate,
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
                RESULT_ROOT / f"spectral_{output_name}.json",
                False,
            )

    run_stage(
        "summarize",
        [
            python,
            "-m",
            "experiments.reduced_state_emulator.summarize_grey_start_benchmark",
        ],
        RESULT_ROOT / "summary.json",
        True,
    )
    if not args.dry_run:
        state.update({"status": "complete", "finished_unix": time.time()})
        _write_state(state_path, state)
    print(f"long-run state: {state_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
