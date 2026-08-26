"""Summarize the grey/two-field/six-field/full-interpolation benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.report import load_records, summarize


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
SIX = "production_six_field"
TWO = "learned_reduced_state"
GREY = "grey15"
INTERP = "interpolated_full_state"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _profile_path(profile_roots: dict[str, Path], arm: str, slug: str) -> Path:
    return profile_roots[arm] / f"{slug}.npz"


def compare_profiles(
    profile_roots: dict[str, Path],
    reference: str,
    candidate: str,
    slugs: set[str],
) -> dict:
    available = sorted(
        slugs
        & {
            path.stem for path in profile_roots[reference].glob("*.npz")
        }
        & {path.stem for path in profile_roots[candidate].glob("*.npz")}
    )
    result = {
        "reference": reference,
        "candidate": candidate,
        "star_count": len(available),
        "fields": {},
    }
    for field_index, field in enumerate(FIELDS):
        errors = []
        for slug in available:
            with np.load(_profile_path(profile_roots, reference, slug), allow_pickle=False) as ref:
                reference_values = np.asarray(ref[field], dtype=np.float64)
            with np.load(_profile_path(profile_roots, candidate, slug), allow_pickle=False) as cand:
                candidate_values = np.asarray(cand[field], dtype=np.float64)
            if field in POSITIVE_FIELDS:
                error = np.abs(
                    np.log10(np.maximum(candidate_values, 1.0e-300))
                    - np.log10(np.maximum(reference_values, 1.0e-300))
                )
                metric = "absolute_dex"
            else:
                error = np.abs(candidate_values - reference_values) / np.maximum(
                    np.abs(reference_values), G_RAD_FLOOR
                )
                metric = "floored_normalized"
            if not np.all(np.isfinite(error)):
                raise ValueError(f"non-finite {field} difference for {slug}")
            errors.append(error)
        if errors:
            result["fields"][field] = {"metric": metric, **_stats(np.concatenate(errors))}
    return result


def _spectral_summary(path: Path, slugs: set[str]) -> dict | None:
    if not path.is_file():
        return None
    source = json.loads(path.read_text())
    rows = [row for row in source.get("per_star", []) if row["slug"] in slugs]
    result = {
        "reference": source.get("baseline_arm"),
        "candidate": source.get("candidate_arm"),
        "star_count": len(rows),
        "bar": float(source.get("bar", 5.0e-3)),
    }
    for field in ("normalized_flux", "flux_total", "flux_continuum"):
        maxima = np.asarray([row[field]["max"] for row in rows], dtype=np.float64)
        result[field] = (
            {
                "median_max": float(np.median(maxima)),
                "p95_max": float(np.percentile(maxima, 95)),
                "max": float(np.max(maxima)),
                "stars_over_bar": int(np.sum(maxima > result["bar"])),
                "passes": bool(maxima.size and np.max(maxima) <= result["bar"]),
            }
            if maxima.size
            else None
        )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new-run-root",
        type=Path,
        default=Path("runs/atmosphere_interpolation_benchmark_20260813"),
    )
    parser.add_argument(
        "--new-result-root",
        type=Path,
        default=Path("results/atmosphere_interpolation_benchmark_20260813"),
    )
    parser.add_argument(
        "--old-run-root",
        type=Path,
        default=Path("runs/grey_start_benchmark_20260812/calibration_spectral60"),
    )
    parser.add_argument(
        "--old-result-root",
        type=Path,
        default=Path("results/initializer_improvement_20260812/calibration400"),
    )
    parser.add_argument(
        "--target-manifest",
        type=Path,
        default=Path("results/grey_start_benchmark_20260812/calibration_spectral60_manifest.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/atmosphere_interpolation_benchmark_20260813/four_initializer_comparison.json"),
    )
    args = parser.parse_args(argv)

    record_paths = {
        SIX: args.old_run_root / "records" / SIX / "records.jsonl",
        TWO: args.old_run_root / "records" / TWO / "records.jsonl",
        GREY: args.old_run_root / "records" / GREY / "records.jsonl",
        INTERP: args.new_run_root / "records" / INTERP / "records.jsonl",
    }
    records = {arm: _records(path) for arm, path in record_paths.items()}
    if any(not rows for rows in records.values()):
        missing = [arm for arm, rows in records.items() if not rows]
        raise SystemExit(f"missing benchmark records for: {missing}")
    converged = {arm: _converged(rows) for arm, rows in records.items()}
    common_four = set.intersection(*converged.values())
    grey30_records = _records(args.old_run_root / "records" / "grey30" / "records.jsonl")
    grey60_records = _records(args.old_run_root / "records" / "grey60" / "records.jsonl")
    grey15_converged = converged[GREY]
    grey30_converged = grey15_converged | _converged(grey30_records)
    grey60_converged = grey30_converged | _converged(grey60_records)

    profile_roots = {
        SIX: args.old_run_root / "profiles" / SIX,
        TWO: args.old_run_root / "profiles" / TWO,
        GREY: args.old_run_root / "profiles" / GREY,
        INTERP: args.new_run_root / "profiles" / INTERP,
    }
    pair_common = {
        arm: converged[SIX] & converged[arm]
        for arm in (TWO, GREY, INTERP)
    }
    profile_comparisons = {
        "all_four_vs_six": {
            arm: compare_profiles(profile_roots, SIX, arm, common_four)
            for arm in (TWO, GREY, INTERP)
        },
        "pair_vs_six": {
            arm: compare_profiles(profile_roots, SIX, arm, pair_common[arm])
            for arm in (TWO, GREY, INTERP)
        },
    }
    pair_common_slugs = {
        arm: sorted(values) for arm, values in pair_common.items()
    }

    spectral_paths = {
        "interpolation_vs_six": args.new_result_root / "spectral_interpolation_vs_six.json",
        "interpolation_vs_two": args.new_result_root / "spectral_interpolation_vs_two.json",
        "interpolation_vs_grey15": args.new_result_root / "spectral_interpolation_vs_grey15.json",
        "two_vs_six": args.old_result_root / "spectral_two_vs_six_unified.json",
        "grey15_vs_six": args.old_result_root / "spectral_grey15_vs_six.json",
    }
    spectra = {
        key: _spectral_summary(path, common_four)
        for key, path in spectral_paths.items()
    }
    pair_spectra = {
        "interpolation_vs_six": _spectral_summary(
            spectral_paths["interpolation_vs_six"], pair_common[INTERP]
        ),
        "interpolation_vs_two": _spectral_summary(
            spectral_paths["interpolation_vs_two"],
            converged[TWO] & converged[INTERP],
        ),
        "interpolation_vs_grey15": _spectral_summary(
            spectral_paths["interpolation_vs_grey15"],
            converged[GREY] & converged[INTERP],
        ),
        "two_vs_six": _spectral_summary(
            spectral_paths["two_vs_six"], converged[SIX] & converged[TWO]
        ),
        "grey15_vs_six": _spectral_summary(
            spectral_paths["grey15_vs_six"], converged[SIX] & converged[GREY]
        ),
    }

    manifest_info = None
    if args.target_manifest.is_file():
        target_manifest = json.loads(args.target_manifest.read_text())
        manifest_info = {
            "path": str(args.target_manifest),
            "sha256": _sha256(args.target_manifest),
            "star_count": len(target_manifest.get("star_indices", [])),
            "star_indices": [int(value) for value in target_manifest.get("star_indices", [])],
        }
    interpolation_result = args.new_result_root / "convergence_interpolated_full_state.json"
    state_result = args.new_result_root / "benchmark_state.json"
    provenance = None
    if interpolation_result.is_file():
        interpolation_json = json.loads(interpolation_result.read_text())
        provenance = {
            "donor_pool": interpolation_json.get("donor_pool"),
            "interpolation": interpolation_json.get("interpolation"),
            "source_sha256": _sha256(interpolation_result),
        }
    hashes = {
        "target_manifest": manifest_info["sha256"] if manifest_info else None,
        "interpolation_result": _sha256(interpolation_result) if interpolation_result.is_file() else None,
        "benchmark_state": _sha256(state_result) if state_result.is_file() else None,
    }

    report = {
        "format": "payne_zero_four_initializer_comparison_v1",
        "comparison_contract": {
            "same_star_manifest": True,
            "same_solver": True,
            "primary_iteration_cap": 15,
            "grey_extensions_are_diagnostic_only": True,
            "interpolation_arm": INTERP,
            "interpolation_mode": "complete_six_field_state",
            "interpolation_labels": ["5040/Teff", "logg", "metallicity", "alpha_enhancement"],
            "microturbulence_in_distance": False,
            "interpolation_neighbours": 8,
            "interpolation_power": 2.0,
            "spectrum_window_nm": [400.0, 900.0],
            "spectrum_resolution": 20000.0,
        },
        "target_manifest": manifest_info,
        "hashes": hashes,
        "convergence": {
            arm: {
                "star_count": len(rows),
                "converged_count": len(converged[arm]),
                "converged_fraction": len(converged[arm]) / len(rows),
                "summary": summarize(rows),
            }
            for arm, rows in records.items()
        },
        "grey_extensions": {
            "grey15": {"converged_count": len(grey15_converged), "star_count": len(records[GREY])},
            "grey30_cumulative": {
                "converged_count": len(grey30_converged),
                "star_count": len(records[GREY]),
                "note": "grey15 successes plus grey30 reruns of grey15 failures",
            },
            "grey60_cumulative": {
                "converged_count": len(grey60_converged),
                "star_count": len(records[GREY]),
                "note": "grey15 and grey30 successes plus finite grey60 reruns",
            },
        },
        "common_star_slugs": {
            "all_four": sorted(common_four),
            "pair_with_six": pair_common_slugs,
            "pair_spectra": {
                "interpolation_vs_two": sorted(converged[TWO] & converged[INTERP]),
                "interpolation_vs_grey15": sorted(converged[GREY] & converged[INTERP]),
                "two_vs_six": sorted(converged[SIX] & converged[TWO]),
                "grey15_vs_six": sorted(converged[SIX] & converged[GREY]),
            },
        },
        "final_six_field_profiles": profile_comparisons,
        "spectra": spectra,
        "spectral_comparisons": {
            "all_four_common": spectra,
            "pair_common": pair_spectra,
        },
        "interpolation_provenance": provenance,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(f"four-arm common converged stars: {len(common_four)}")
    for arm, entry in report["convergence"].items():
        print(f"{arm}: {entry['converged_count']}/{entry['star_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
