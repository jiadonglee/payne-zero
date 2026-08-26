"""Unified Grey, two-field, and six-field convergence/structure/spectrum report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.report import load_records, summarize


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "runs" / "grey_start_benchmark_20260812" / "calibration_spectral60"
RESULT_ROOT = REPO_ROOT / "results" / "grey_start_benchmark_20260812" / "calibration_spectral60"
SPECTRAL_ROOT = REPO_ROOT / "results" / "initializer_improvement_20260812" / "calibration400"
FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)
POSITIVE_FIELDS = FIELDS[:-1]
G_RAD_FLOOR = 2.0577175465785027


def _records(path: Path) -> list[dict]:
    return load_records(path) if path.is_file() else []


def _converged(records: list[dict]) -> set[str]:
    return {row["slug"] for row in records if row["converged"]}


def _stats(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compare_profiles(profile_root: Path, reference: str, candidate: str, slugs: set[str]) -> dict:
    available = sorted(
        slugs
        & {path.stem for path in (profile_root / reference).glob("*.npz")}
        & {path.stem for path in (profile_root / candidate).glob("*.npz")}
    )
    result = {"reference": reference, "candidate": candidate, "star_count": len(available), "fields": {}}
    for field in FIELDS:
        ref, cand = [], []
        for slug in available:
            with np.load(profile_root / reference / f"{slug}.npz", allow_pickle=False) as data:
                ref.append(np.asarray(data[field], dtype=np.float64))
            with np.load(profile_root / candidate / f"{slug}.npz", allow_pickle=False) as data:
                cand.append(np.asarray(data[field], dtype=np.float64))
        if not ref:
            continue
        ref_array, cand_array = np.stack(ref), np.stack(cand)
        if field in POSITIVE_FIELDS:
            error = np.abs(
                np.log10(np.maximum(cand_array, 1.0e-300))
                - np.log10(np.maximum(ref_array, 1.0e-300))
            )
            result["fields"][field] = {"metric": "absolute_dex", **_stats(error)}
        else:
            error = np.abs(cand_array - ref_array) / np.maximum(np.abs(ref_array), G_RAD_FLOOR)
            result["fields"][field] = {"metric": "floored_normalized", **_stats(error)}
    return result


def spectral_common(path: Path, common: set[str]) -> dict | None:
    if not path.is_file():
        return None
    source = json.loads(path.read_text())
    rows = [row for row in source.get("per_star", []) if row["slug"] in common]
    result = {
        "reference": source.get("baseline_arm"),
        "candidate": source.get("candidate_arm"),
        "star_count": len(rows),
        "bar": source.get("bar", 5.0e-3),
    }
    for field in ("normalized_flux", "flux_total", "flux_continuum"):
        maxima = np.asarray([row[field]["max"] for row in rows], dtype=np.float64)
        result[field] = (
            {
                "median_max": float(np.median(maxima)),
                "max": float(np.max(maxima)),
                "stars_over_bar": int(np.sum(maxima > result["bar"])),
            }
            if maxima.size
            else None
        )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args(argv)

    record_paths = {
        "grey15": args.run_root / "records" / "grey15" / "records.jsonl",
        "grey30": args.run_root / "records" / "grey30" / "records.jsonl",
        "grey60": args.run_root / "records" / "grey60" / "records.jsonl",
        "interpolated": args.run_root / "records" / "interpolated_grid" / "records.jsonl",
        "two_field": args.run_root / "records" / "learned_reduced_state" / "records.jsonl",
        "six_field": args.run_root / "records" / "production_six_field" / "records.jsonl",
    }
    records = {name: _records(path) for name, path in record_paths.items()}
    grey15_converged = _converged(records["grey15"])
    grey30_converged = grey15_converged | _converged(records["grey30"])
    grey60_converged = grey30_converged | _converged(records["grey60"])
    converged = {
        "grey15": grey15_converged,
        "grey30_extended": grey30_converged,
        "grey60_extended": grey60_converged,
        "interpolated": _converged(records["interpolated"]),
        "two_field": _converged(records["two_field"]),
        "six_field": _converged(records["six_field"]),
    }
    common15 = converged["grey15"] & converged["two_field"] & converged["six_field"]
    common30 = converged["grey30_extended"] & converged["two_field"] & converged["six_field"]
    common60 = converged["grey60_extended"] & converged["two_field"] & converged["six_field"]
    # The interpolated-grid arm is intersected separately so that adding it does
    # not move any number the grey/two/six comparison already reported.
    common_interp = converged["interpolated"] & converged["two_field"] & converged["six_field"]

    convergence = {
        name: summarize(rows) if rows else None
        for name, rows in records.items()
    }
    convergence["grey30_extended"] = {
        "requested_star_count": len(records["grey15"]),
        "converged_count": len(grey30_converged),
        "converged_fraction": (
            len(grey30_converged) / len(records["grey15"]) if records["grey15"] else None
        ),
        "note": "grey15 successes plus grey30 reruns of grey15 failures",
    }
    convergence["grey60_extended"] = {
        "requested_star_count": len(records["grey15"]),
        "converged_count": len(grey60_converged),
        "converged_fraction": (
            len(grey60_converged) / len(records["grey15"]) if records["grey15"] else None
        ),
        "note": (
            "grey15 and grey30 successes plus grey60 reruns of finite grey30 failures; "
            "non-finite grey30 failures cannot recover without changing the solver"
        ),
    }

    profile_root = args.run_root / "profiles"
    profiles = {
        "common15": {
            "star_count": len(common15),
            "grey_vs_six": compare_profiles(profile_root, "production_six_field", "grey15", common15),
            "two_vs_six": compare_profiles(profile_root, "production_six_field", "learned_reduced_state", common15),
            "grey_vs_two": compare_profiles(profile_root, "learned_reduced_state", "grey15", common15),
        },
        "common30": {
            "star_count": len(common30),
            "grey_vs_six": compare_profiles(profile_root, "production_six_field", "grey30_complete", common30),
            "two_vs_six": compare_profiles(profile_root, "production_six_field", "learned_reduced_state", common30),
            "grey_vs_two": compare_profiles(profile_root, "learned_reduced_state", "grey30_complete", common30),
        },
        "common60": {
            "star_count": len(common60),
            "grey_vs_six": compare_profiles(profile_root, "production_six_field", "grey60_complete", common60),
            "two_vs_six": compare_profiles(profile_root, "production_six_field", "learned_reduced_state", common60),
            "grey_vs_two": compare_profiles(profile_root, "learned_reduced_state", "grey60_complete", common60),
        },
        "common_interpolated": {
            "star_count": len(common_interp),
            "interp_vs_six": compare_profiles(
                profile_root, "production_six_field", "interpolated_grid", common_interp
            ),
            "interp_vs_two": compare_profiles(
                profile_root, "learned_reduced_state", "interpolated_grid", common_interp
            ),
        },
    }
    spectra = {
        "common15": {
            "star_count": len(common15),
            "two_vs_six": spectral_common(SPECTRAL_ROOT / "spectral_two_vs_six_unified.json", common15),
            "grey_vs_six": spectral_common(SPECTRAL_ROOT / "spectral_grey15_vs_six.json", common15),
            "grey_vs_two": spectral_common(SPECTRAL_ROOT / "spectral_grey15_vs_two.json", common15),
        },
        "common30": {
            "star_count": len(common30),
            "two_vs_six": spectral_common(SPECTRAL_ROOT / "spectral_two_vs_six_unified.json", common30),
            "grey_vs_six": spectral_common(SPECTRAL_ROOT / "spectral_grey_vs_six.json", common30),
            "grey_vs_two": spectral_common(SPECTRAL_ROOT / "spectral_grey_vs_two.json", common30),
        },
        "common60": {
            "star_count": len(common60),
            "two_vs_six": spectral_common(SPECTRAL_ROOT / "spectral_two_vs_six_unified.json", common60),
            "grey_vs_six": spectral_common(SPECTRAL_ROOT / "spectral_grey60_vs_six.json", common60),
            "grey_vs_two": spectral_common(SPECTRAL_ROOT / "spectral_grey60_vs_two.json", common60),
        },
    }
    report = {
        "comparison_contract": {
            "same_star_manifest": True,
            "same_solver": True,
            "primary_iteration_cap": 15,
            "grey30_is_extended_diagnostic": True,
            "grey60_is_extended_finite_state_diagnostic": True,
            "spectrum_window_nm": [400.0, 900.0],
            "spectrum_resolution": 20000.0,
        },
        "convergence": convergence,
        "common_star_slugs": {
            "grey15": sorted(common15),
            "grey30_extended": sorted(common30),
            "grey60_extended": sorted(common60),
            "interpolated": sorted(common_interp),
        },
        "final_six_field_profiles": profiles,
        "spectra": spectra,
    }
    args.result_root.mkdir(parents=True, exist_ok=True)
    output = args.result_root / "three_initializer_comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
