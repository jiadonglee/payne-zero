"""Combine grey-start convergence, final-atmosphere, and spectral results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.reduced_state_emulator.grey_start_benchmark import (
    DEFAULT_RESULT_ROOT,
    DEFAULT_RUN_ROOT,
)


FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)
G_RAD_SCALE = 2.0577175465785027


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _profile_metric(candidate: np.ndarray, reference: np.ndarray, field: str) -> dict:
    if field == "radiative_acceleration":
        error = np.abs(candidate - reference) / np.maximum(
            np.abs(reference), G_RAD_SCALE
        )
        name = "normalized_error"
    elif field in ("column_mass", "temperature", "gas_pressure", "electron_density", "rosseland_opacity"):
        error = np.abs(
            np.log10(np.maximum(candidate, 1.0e-300))
            - np.log10(np.maximum(reference, 1.0e-300))
        )
        name = "absolute_dex_error"
    else:
        raise ValueError(field)
    return {
        name: {
            "median": float(np.median(error)),
            "p95": float(np.percentile(error, 95)),
            "max": float(np.max(error)),
        }
    }


def compare_profiles(run_root: Path, candidate_arm: str) -> dict:
    reference_dir = run_root / "profiles" / "production_six_field"
    candidate_dir = run_root / "profiles" / candidate_arm
    slugs = sorted(
        {path.stem for path in reference_dir.glob("*.npz")}
        & {path.stem for path in candidate_dir.glob("*.npz")}
    )
    result = {"paired_star_count": len(slugs), "fields": {}}
    for field in FIELDS:
        reference = []
        candidate = []
        for slug in slugs:
            with np.load(reference_dir / f"{slug}.npz", allow_pickle=False) as data:
                reference.append(np.asarray(data[field], dtype=np.float64))
            with np.load(candidate_dir / f"{slug}.npz", allow_pickle=False) as data:
                candidate.append(np.asarray(data[field], dtype=np.float64))
        if reference:
            result["fields"][field] = _profile_metric(
                np.stack(candidate), np.stack(reference), field
            )
    return result


def _record_map(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {record["slug"]: record for record in records}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args(argv)

    convergence = {
        arm: _load_json(args.result_root / f"convergence_{arm}.json")
        for arm in (
            "production_six_field",
            "learned_reduced_state",
            "truth_reduced_state",
            "grey15",
            "grey30",
            "grey_perturbed",
        )
    }
    grey15 = _record_map(args.run_root / "records" / "grey15" / "records.jsonl")
    grey30 = _record_map(args.run_root / "records" / "grey30" / "records.jsonl")
    rescued = sorted(
        slug for slug, record in grey15.items()
        if not record["converged"] and grey30.get(slug, {}).get("converged", False)
    )
    still_failed = sorted(
        slug for slug, record in grey15.items()
        if not record["converged"] and not grey30.get(slug, {}).get("converged", False)
    )
    spectral = {
        arm: _load_json(args.result_root / f"spectral_{arm}.json")
        for arm in ("learned_reduced_state", "truth_reduced_state", "grey15", "grey30")
    }
    profile = {
        arm: compare_profiles(
            args.run_root, "grey30_complete" if arm == "grey30" else arm
        )
        for arm in ("learned_reduced_state", "truth_reduced_state", "grey15", "grey30")
    }
    out = {
        "convergence": convergence,
        "grey15_failed_but_grey30_converged": rescued,
        "grey30_still_failed": still_failed,
        "final_atmosphere_vs_production": profile,
        "spectra_vs_production": spectral,
        "interpretation": {
            "spectral_bar": 5.0e-3,
            "wide_basin": "grey mostly converges within 15 and spectra pass",
            "recoverable_but_initializer_matters": "grey needs 16-30 iterations",
            "basin_dependent": "grey often fails or converges to spectra above the bar",
        },
    }
    args.result_root.mkdir(parents=True, exist_ok=True)
    path = args.result_root / "summary.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
