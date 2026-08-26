#!/usr/bin/env python3
"""Refresh every Paper II artifact affected by the physical reconstruction seed.

The frozen two-field predictor is unchanged.  This campaign only replaces the
historical six-field reconstruction seed with the current physical default.
Production-six-field products are reused as the unchanged comparison arm.

Long solver/synthesis stages write to a new namespace and receive completion
markers.  Re-running the driver skips completed stages and refuses to append to
an incomplete stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


CAMPAIGN = "paper_physical_seed_20260820"
RESOLUTIONS = (40, 80, 160, 320, 640)
THREAD_ENV = {
    "NUMBA_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_THREADING_LAYER": "workqueue",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_text(command: list[str]) -> str:
    return " ".join(command)


def stage_source_signature(root: Path, command: list[str]) -> dict[str, str]:
    paths = [
        root / "reduced_state" / "reconstruct.py",
        Path(__file__).resolve(),
    ]
    if len(command) > 2 and command[1] == "-m":
        paths.append(root / (command[2].replace(".", "/") + ".py"))
    elif len(command) > 1:
        entry = Path(command[1])
        paths.append(entry if entry.is_absolute() else root / entry)
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def run_stage(
    *,
    name: str,
    command: list[str],
    expected: list[Path],
    marker_dir: Path,
    log_dir: Path,
    env: dict[str, str],
    root: Path,
    dry_run: bool,
    acceptable_returncodes: tuple[int, ...] = (0,),
) -> None:
    marker = marker_dir / f"{name}.json"
    source_signature = stage_source_signature(root, command)
    if marker.is_file():
        record = json.loads(marker.read_text())
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise RuntimeError(
                f"{name}: completion marker exists but outputs are missing: {missing}"
            )
        if record.get("command") != command:
            raise RuntimeError(f"{name}: command changed since the completed stage")
        if record.get("source_sha256") != source_signature:
            raise RuntimeError(f"{name}: source changed since the completed stage")
        for path in expected:
            if path.is_file() and record["outputs"].get(str(path)) != sha256(path):
                raise RuntimeError(f"{name}: output changed since the completed stage: {path}")
        print(f"SKIP {name}: complete", flush=True)
        return

    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise RuntimeError(
            f"{name}: incomplete outputs already exist; move them aside before retrying: "
            + ", ".join(existing)
        )

    print(f"START {name}: {command_text(command)}", flush=True)
    if dry_run:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    started = datetime.now(timezone.utc)
    with log_path.open("w") as handle:
        completed = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode not in acceptable_returncodes:
        raise subprocess.CalledProcessError(completed.returncode, command)
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"{name}: command completed but outputs are missing: {missing}")

    marker_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "stage": name,
                "started_utc": started.isoformat(),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "command": command,
                "source_sha256": source_signature,
                "returncode": completed.returncode,
                "acceptable_returncodes": list(acceptable_returncodes),
                "log": str(log_path),
                "outputs": {
                    str(path): sha256(path) if path.is_file() else "directory"
                    for path in expected
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"DONE {name}", flush=True)


def aggregate_depth_results(result_root: Path, marker_dir: Path, dry_run: bool) -> None:
    out_dir = result_root / "depth_resolution"
    out = out_dir / "convergence_metrics_depth_resolution.json"
    marker = marker_dir / "depth_aggregate.json"
    inputs = [
        result_root
        / f"depth_resolution_n{resolution}"
        / "convergence_metrics_depth_resolution.json"
        for resolution in RESOLUTIONS
    ]
    if marker.is_file():
        if not out.is_file():
            raise RuntimeError("depth_aggregate: marker exists but output is missing")
        record = json.loads(marker.read_text())
        expected_inputs = {str(path): sha256(path) for path in inputs}
        if record.get("inputs") != expected_inputs:
            raise RuntimeError("depth_aggregate: input hashes changed")
        if record.get("sha256") != sha256(out):
            raise RuntimeError("depth_aggregate: output hash changed")
        print("SKIP depth_aggregate: complete", flush=True)
        return
    if out.exists():
        raise RuntimeError("depth_aggregate: output exists without a completion marker")
    if dry_run:
        print(f"START depth_aggregate: write {out}", flush=True)
        return

    blocks = [json.loads(path.read_text()) for path in inputs]
    reference = blocks[0]
    for block in blocks[1:]:
        if block["star_indices"] != reference["star_indices"]:
            raise RuntimeError("depth-resolution stages used different star samples")

    summary = {
        "star_count": reference["star_count"],
        "star_indices": reference["star_indices"],
        "seed": reference["seed"],
        "resolutions": list(RESOLUTIONS),
        "n_synchronizations": reference["n_synchronizations"],
        "reconstruction_seed": "physical",
        "by_resolution": {
            str(resolution): block["by_resolution"][str(resolution)]
            for resolution, block in zip(RESOLUTIONS, blocks)
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    marker.write_text(
        json.dumps(
            {
                "stage": "depth_aggregate",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "output": str(out),
                "sha256": sha256(out),
                "inputs": {str(path): sha256(path) for path in inputs},
            },
            indent=2,
        )
        + "\n"
    )
    print("DONE depth_aggregate", flush=True)


def ensure_production_link(run_root: Path, root: Path, dry_run: bool) -> None:
    target = root / "runs" / "reduced_state_emulator" / "products" / "production_six_field"
    link = run_root / "learned" / "products" / "production_six_field"
    if not target.is_dir():
        if dry_run:
            print(f"CHECK frozen production products: {target}", flush=True)
            return
        raise FileNotFoundError(f"missing frozen production products: {target}")
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"production link points to the wrong target: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing production path: {link}")
    print(f"LINK {link} -> {target}", flush=True)
    if not dry_run:
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)


def write_manifest(
    *,
    root: Path,
    result_root: Path,
    run_root: Path,
    marker_dir: Path,
    workers: int,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    source_paths = [
        Path(__file__).resolve(),
        root / "reduced_state" / "reconstruct.py",
        root / "experiments" / "reduced_state_parity" / "run_oracle_parity.py",
        root / "experiments" / "depth_resolution" / "run_depth_resolution.py",
        root
        / "experiments"
        / "reduced_state_emulator"
        / "run_learned_restart.py",
        root
        / "experiments"
        / "reduced_state_emulator"
        / "derived_field_accuracy.py",
        root / "experiments" / "reduced_state_emulator" / "spectral_gate.py",
        root / "artifacts" / "reduced_state_emulator" / "predicted_monotone.npz",
    ]
    output_files = sorted(
        path
        for path in result_root.rglob("*")
        if path.is_file() and path.name != "campaign.json"
    )
    manifest = {
        "campaign": CAMPAIGN,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "workers": workers,
        "thread_environment": THREAD_ENV,
        "reconstruction_seed": "physical",
        "two_field_predictor": "frozen predicted_monotone.npz; unchanged",
        "production_arm": (
            "reused frozen six-field products; not recomputed by this campaign"
        ),
        "blind200": (
            "reuse 2026-08-19 blind200_physical_seed; already-opened holdout, "
            "not rerun here"
        ),
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in source_paths},
        "stage_markers": sorted(str(path.relative_to(root)) for path in marker_dir.glob("*.json")),
        "output_sha256": {
            str(path.relative_to(root)): sha256(path) for path in output_files
        },
        "run_root": str(run_root.relative_to(root)),
        "result_root": str(result_root.relative_to(root)),
    }
    (result_root / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    python = root / ".venv-linux" / "bin" / "python"
    if not python.is_file():
        python = root / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"missing solver Python: {python}")

    result_root = root / "results" / CAMPAIGN
    run_root = root / "runs" / CAMPAIGN
    marker_dir = run_root / "markers"
    log_dir = run_root / "logs"
    env = os.environ.copy()
    env.update(THREAD_ENV)
    env["PYTHONPATH"] = str(root)

    parity_results = result_root / "parity"
    parity_run = run_root / "parity"
    run_stage(
        name="parity",
        command=[
            str(python),
            "experiments/reduced_state_parity/run_oracle_parity.py",
            "--count",
            "60",
            "--seed",
            "20260807",
            "--workers",
            str(args.workers),
            "--products-dir",
            str(parity_run / "products"),
            "--results-dir",
            str(parity_results),
            "--figures-dir",
            str(parity_run / "figures"),
            "--records-dir",
            str(parity_run / "records"),
        ],
        expected=[
            parity_results / "reconstruction_metrics.json",
            parity_results / "reconstruction_metrics.npz",
            parity_results / "convergence_metrics_reduced_state_parity.json",
        ],
        marker_dir=marker_dir,
        log_dir=log_dir,
        env=env,
        root=root,
        dry_run=args.dry_run,
    )

    for resolution in RESOLUTIONS:
        stage_results = result_root / f"depth_resolution_n{resolution}"
        stage_run = run_root / f"depth_resolution_n{resolution}"
        run_stage(
            name=f"depth_n{resolution}",
            command=[
                str(python),
                "experiments/depth_resolution/run_depth_resolution.py",
                "--workers",
                str(args.workers),
                "--results-dir",
                str(stage_results),
                "--figures-dir",
                str(stage_run / "figures"),
                "--records-dir",
                str(stage_run / "records"),
                "--reuse-indices-from",
                str(parity_results / "reconstruction_metrics.json"),
                "--resolutions",
                str(resolution),
            ],
            expected=[stage_results / "convergence_metrics_depth_resolution.json"],
            marker_dir=marker_dir,
            log_dir=log_dir,
            env=env,
            root=root,
            dry_run=args.dry_run,
        )
    aggregate_depth_results(result_root, marker_dir, args.dry_run)

    learned_results = result_root / "learned"
    learned_run = run_root / "learned"
    run_stage(
        name="learned_restart",
        command=[
            str(python),
            "-m",
            "experiments.reduced_state_emulator.run_learned_restart",
            "--arm",
            "monotone",
            "--held-out-from",
            str(parity_results / "reconstruction_metrics.json"),
            "--workers",
            str(args.workers),
            "--skip-production-arm",
            "--products-dir",
            str(learned_run / "products"),
            "--results-dir",
            str(learned_results),
            "--records-dir",
            str(learned_run / "records"),
        ],
        expected=[
            learned_results / "convergence_metrics_learned_monotone.json",
            learned_run / "records" / "learned_reduced_state" / "records.jsonl",
        ],
        marker_dir=marker_dir,
        log_dir=log_dir,
        env=env,
        root=root,
        dry_run=args.dry_run,
    )

    run_stage(
        name="derived_fields",
        command=[
            str(python),
            "-m",
            "experiments.reduced_state_emulator.derived_field_accuracy",
            "--workers",
            str(args.workers),
            "--out",
            str(learned_results / "learned_reduced_state_derived_errors.npz"),
        ],
        expected=[
            learned_results / "learned_reduced_state_derived_errors.npz",
            learned_results / "learned_reduced_state_derived_errors.json",
        ],
        marker_dir=marker_dir,
        log_dir=log_dir,
        env=env,
        root=root,
        dry_run=args.dry_run,
    )

    ensure_production_link(run_root, root, args.dry_run)
    run_stage(
        name="learned_spectral_gate",
        command=[
            str(python),
            "-m",
            "experiments.reduced_state_emulator.spectral_gate",
            "--products-dir",
            str(learned_run / "products"),
            "--spectra-dir",
            str(learned_run / "spectra"),
            "--out",
            str(learned_results / "spectral_gate.json"),
            "--workers",
            str(args.workers),
        ],
        expected=[learned_results / "spectral_gate.json"],
        marker_dir=marker_dir,
        log_dir=log_dir,
        env=env,
        root=root,
        dry_run=args.dry_run,
        acceptable_returncodes=(0, 1),
    )

    run_stage(
        name="parity_spectral_gate",
        command=[
            str(python),
            "-m",
            "experiments.reduced_state_emulator.spectral_gate",
            "--products-dir",
            str(parity_run / "products"),
            "--spectra-dir",
            str(parity_run / "spectra"),
            "--out",
            str(parity_results / "spectral_gate_truth_mT.json"),
            "--workers",
            str(args.workers),
            "--baseline-arm",
            "full_truth_oracle",
            "--candidate-arm",
            "reduced_state_reconstruction",
        ],
        expected=[parity_results / "spectral_gate_truth_mT.json"],
        marker_dir=marker_dir,
        log_dir=log_dir,
        env=env,
        root=root,
        dry_run=args.dry_run,
    )

    write_manifest(
        root=root,
        result_root=result_root,
        run_root=run_root,
        marker_dir=marker_dir,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    print(f"COMPLETE {CAMPAIGN}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
