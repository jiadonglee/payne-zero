"""Run the expanded four-initializer benchmark.

The open 400-star calibration already has production six-field and learned
two-field products.  This driver selects a fixed 200-star subset containing
the original 60-star benchmark, exposes those completed products under a new
run root, and runs only the new grey and complete-state interpolation arms.
All four arms then go through the same spectral gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from bench.labels import StellarLabels


REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
CALIBRATION_MANIFEST = REPO_ROOT / "results" / "initializer_calibration_20260812.json"
PREVIOUS_MANIFEST = (
    REPO_ROOT
    / "results"
    / "grey_start_benchmark_20260812"
    / "calibration_spectral60_manifest.json"
)
SEALED_HOLDOUT = REPO_ROOT / "results" / "sealed_initializer_holdout_20260812.json"
CALIBRATION_RUN_ROOT = REPO_ROOT / "runs" / "initializer_improvement_20260812" / "calibration400"
PREVIOUS_RUN_ROOT = REPO_ROOT / "runs" / "grey_start_benchmark_20260812" / "calibration_spectral60"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "four_initializer_benchmark_expanded_20260814"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "four_initializer_benchmark_expanded_20260814"
DEFAULT_MANIFEST = DEFAULT_RESULT_ROOT / "expanded200_manifest.json"
PREVIOUS_SUMMARY = REPO_ROOT / "results" / "atmosphere_interpolation_benchmark_20260813" / "four_initializer_comparison.json"
SEED = 20260814

ARMS = ("production_six_field", "learned_reduced_state", "grey15", "interpolated_full_state")
REUSED_ARMS = ("production_six_field", "learned_reduced_state")
ADDITIONAL_COUNTS = {"ordinary": 70, "hard": 35, "edge": 35}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(entry: dict[str, float]) -> str:
    return StellarLabels(
        float(entry["effective_temperature"]),
        float(entry["log_surface_gravity"]),
        float(entry["metallicity"]),
        float(entry["alpha_enhancement"]),
        float(entry["microturbulence_km_s"]),
    ).slug


def _load_corpus_labels(indices: list[int]) -> dict[int, tuple[str, dict]]:
    with np.load(CORPUS, allow_pickle=False) as data:
        result = {}
        for index in indices:
            entry = json.loads(str(data["labels_json"][int(index)]))
            result[int(index)] = (_slug(entry), entry)
    return result


def _record_rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows[row["slug"]] = row
    return rows


def _clear_file_or_link(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()


def _link_selected_products(source: Path, target: Path, slugs: list[str]) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    linked = []
    missing = []
    for slug in slugs:
        source_path = source / f"{slug}.npz"
        target_path = target / source_path.name
        _clear_file_or_link(target_path)
        if source_path.is_file():
            target_path.symlink_to(os.path.relpath(source_path, target))
            linked.append(slug)
        else:
            missing.append(slug)
    return {"linked": linked, "missing": missing, "source": str(source)}


def _write_selected_records(source: Path, target: Path, slugs: list[str]) -> dict:
    rows = _record_rows(source)
    missing = [slug for slug in slugs if slug not in rows]
    selected = [slug for slug in slugs if slug in rows]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(rows[slug]) + "\n" for slug in selected))
    return {
        "source": str(source),
        "star_count": len(selected),
        "missing": missing,
    }


def make_expanded_manifest(path: Path) -> dict:
    calibration = json.loads(CALIBRATION_MANIFEST.read_text())
    previous = json.loads(PREVIOUS_MANIFEST.read_text())
    sealed = json.loads(SEALED_HOLDOUT.read_text())
    calibration_indices = [int(value) for value in calibration["star_indices"]]
    previous_indices = [int(value) for value in previous["star_indices"]]
    if not set(previous_indices) <= set(calibration_indices):
        raise RuntimeError("the previous 60-star benchmark is not a subset of calibration")
    if sealed.get("opened") is not False:
        raise RuntimeError("sealed initializer holdout is not closed")
    labels = _load_corpus_labels(calibration_indices)

    source_record_paths = {
        "learned_reduced_state": CALIBRATION_RUN_ROOT / "candidate_solver" / "learned_reduced_state" / "records.jsonl",
        "production_six_field": CALIBRATION_RUN_ROOT / "production_solver" / "production_six_field" / "records.jsonl",
    }
    available = set(_record_rows(source_record_paths[REUSED_ARMS[0]])) & set(
        _record_rows(source_record_paths[REUSED_ARMS[1]])
    )
    previous_slugs = [labels[index][0] for index in previous_indices]

    category_indices = {
        category: [int(value) for value in payload["star_indices"]]
        for category, payload in calibration["categories"].items()
    }
    generator = np.random.default_rng(SEED)
    selected_additional: list[int] = []
    selected_set = set(previous_indices)
    for category in ("ordinary", "hard", "edge"):
        pool = [
            index
            for index in category_indices[category]
            if index not in selected_set and labels[index][0] in available
        ]
        count = ADDITIONAL_COUNTS[category]
        if len(pool) < count:
            raise RuntimeError(f"not enough reusable {category} stars: {len(pool)} < {count}")
        chosen = generator.choice(np.asarray(pool, dtype=np.int64), size=count, replace=False)
        chosen = [int(value) for value in chosen]
        selected_additional.extend(chosen)
        selected_set.update(chosen)

    selected_indices = previous_indices + selected_additional
    selected_slugs = [labels[index][0] for index in selected_indices]
    selected_categories = {
        category: [index for index in selected_indices if index in set(category_indices[category])]
        for category in ("ordinary", "hard", "edge")
    }
    manifest = {
        "format": "payne_zero_four_initializer_expanded_manifest_v1",
        "selection": "previous spectral60 plus balanced open calibration additions",
        "selection_seed": SEED,
        "star_count": len(selected_indices),
        "star_indices": selected_indices,
        "star_slugs": selected_slugs,
        "previous_60_indices": previous_indices,
        "previous_60_slugs": previous_slugs,
        "added_140_indices": selected_additional,
        "added_140_slugs": [labels[index][0] for index in selected_additional],
        "categories": {
            category: {
                "count": len(values),
                "star_indices": values,
            }
            for category, values in selected_categories.items()
        },
        "source_manifests": {
            "calibration": str(CALIBRATION_MANIFEST),
            "previous_60": str(PREVIOUS_MANIFEST),
            "sealed_holdout": str(SEALED_HOLDOUT),
        },
        "source_sha256": {
            "calibration": _sha256(CALIBRATION_MANIFEST),
            "previous_60": _sha256(PREVIOUS_MANIFEST),
            "sealed_holdout": _sha256(SEALED_HOLDOUT),
        },
        "sealed_holdout_opened": False,
        "reusable_record_intersection_count": len(available),
        "previous_missing_from_reusable_records": sorted(set(previous_slugs) - available),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("star_indices") != selected_indices:
            raise RuntimeError(f"existing manifest differs: {path}")
    else:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_reused_arms(run_root: Path, manifest: dict) -> dict:
    slugs = list(manifest["star_slugs"])
    records = {}
    products = {}
    profiles = {}
    sources = {
        "learned_reduced_state": {
            "records": CALIBRATION_RUN_ROOT / "candidate_solver" / "learned_reduced_state" / "records.jsonl",
            "products": CALIBRATION_RUN_ROOT / "products" / "learned_reduced_state",
        },
        "production_six_field": {
            "records": CALIBRATION_RUN_ROOT / "production_solver" / "production_six_field" / "records.jsonl",
            "products": CALIBRATION_RUN_ROOT / "products" / "production_six_field",
        },
    }
    for arm in REUSED_ARMS:
        records[arm] = _write_selected_records(
            sources[arm]["records"], run_root / "records" / arm / "records.jsonl", slugs
        )
        products[arm] = _link_selected_products(
            sources[arm]["products"], run_root / "products" / arm, slugs
        )
        profiles[arm] = _link_selected_products(
            PREVIOUS_RUN_ROOT / "profiles" / arm,
            run_root / "profiles" / arm,
            manifest["previous_60_slugs"],
        )

    # The original 60-star grey run is already a completed part of the
    # benchmark. Seed it into the expanded run root so only the new 140 stars
    # are solved again.
    records["grey15"] = _write_selected_records(
        PREVIOUS_RUN_ROOT / "records" / "grey15" / "records.jsonl",
        run_root / "records" / "grey15" / "records.jsonl",
        manifest["previous_60_slugs"],
    )
    products["grey15"] = _link_selected_products(
        PREVIOUS_RUN_ROOT / "products" / "grey15",
        run_root / "products" / "grey15",
        manifest["previous_60_slugs"],
    )
    profiles["grey15"] = _link_selected_products(
        PREVIOUS_RUN_ROOT / "profiles" / "grey15",
        run_root / "profiles" / "grey15",
        manifest["previous_60_slugs"],
    )
    return {"records": records, "products": products, "profiles": profiles}


def _run(command: list[str], environment: dict[str, str], *, allow_failure: bool = False) -> None:
    print("RUN", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    if completed.returncode and not allow_failure:
        raise RuntimeError(f"command failed with exit {completed.returncode}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-spectra", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 6:
        raise SystemExit("--workers must be between 1 and 6")

    manifest = make_expanded_manifest(args.manifest)
    run_root = args.run_root
    result_root = args.result_root
    run_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    reused = prepare_reused_arms(run_root, manifest)
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
    base = [
        python,
        "-m",
        "experiments.reduced_state_emulator.grey_start_benchmark",
        "--manifest",
        str(args.manifest),
        "--corpus",
        str(CORPUS),
        "--workers",
        str(args.workers),
        "--run-root",
        str(run_root),
        "--result-root",
        str(result_root),
    ]
    commands = {
        "grey15": base[:3] + ["--arm", "grey15"] + base[3:],
        "interpolated_full_state": base[:3] + ["--arm", "interpolated_full_state"] + base[3:] + [
            "--interpolation-neighbours",
            "8",
            "--interpolation-power",
            "2.0",
        ],
    }
    # The reusable 400-star campaign contains one failed two-field solve.  Keep
    # that star in the expanded sample and repair only the missing record with
    # the already-frozen 400-star prediction; this preserves the exact previous
    # 60-star comparison without rerunning the other 199 stars.
    missing_reused = {
        arm: reused["records"][arm]["missing"]
        for arm in REUSED_ARMS
        if reused["records"][arm]["missing"]
    }
    if missing_reused.get("learned_reduced_state"):
        commands["learned_reduced_state"] = base[:3] + [
            "--arm",
            "learned_reduced_state",
        ] + base[3:] + [
            "--prediction",
            str(
                REPO_ROOT
                / "results"
                / "initializer_improvement_20260812"
                / "calibration400"
                / "base_ensemble.npz"
            ),
        ]
    if missing_reused.get("production_six_field"):
        commands["production_six_field"] = base[:3] + [
            "--arm",
            "production_six_field",
        ] + base[3:]
    state = {
        "format": "payne_zero_expanded_four_initializer_benchmark_v1",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "run_root": str(run_root),
        "result_root": str(result_root),
        "star_count": manifest["star_count"],
        "previous_star_count": len(manifest["previous_60_indices"]),
        "added_star_count": len(manifest["added_140_indices"]),
        "workers": args.workers,
        "primary_iteration_cap": 15,
        "reused_arms": reused,
        "commands": commands,
        "status": "prepared",
    }
    (result_root / "benchmark_state.json").write_text(json.dumps(state, indent=2) + "\n")
    if args.dry_run:
        for command in commands.values():
            print("DRY RUN", " ".join(command), flush=True)
        return 0

    for arm, command in commands.items():
        result_path = result_root / f"convergence_{arm}.json"
        if not result_path.is_file():
            _run(command, environment)

    if not args.skip_spectra:
        for candidate in ("learned_reduced_state", "grey15", "interpolated_full_state"):
            output = result_root / f"spectral_{candidate}_vs_six.json"
            _run(
                [
                    python,
                    "-m",
                    "experiments.reduced_state_emulator.spectral_gate",
                    "--products-dir",
                    str(run_root / "products"),
                    "--spectra-dir",
                    str(run_root / "spectra"),
                    "--out",
                    str(output),
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
                environment,
                allow_failure=True,
            )

    state["status"] = "complete"
    state["result_files"] = sorted(str(path) for path in result_root.glob("*.json"))
    (result_root / "benchmark_state.json").write_text(json.dumps(state, indent=2) + "\n")
    print(f"benchmark state: {result_root / 'benchmark_state.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
