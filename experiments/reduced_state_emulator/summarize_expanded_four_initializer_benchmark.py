"""Summarize the expanded four-initializer benchmark and its 60-star subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.report import load_records, summarize


ARMS = ("production_six_field", "learned_reduced_state", "grey15", "interpolated_full_state")
SIX = "production_six_field"
BAR = 5.0e-3
FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)
POSITIVE_FIELDS = set(FIELDS[:-1])
G_RAD_FLOOR = 2.0577175465785027


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


def _metric(values: np.ndarray) -> dict | None:
    if values.size == 0:
        return None
    return {
        "median_max": float(np.median(values)),
        "p95_max": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "stars_over_bar": int(np.sum(values > BAR)),
        "passes": bool(np.max(values) <= BAR),
    }


def _spectral_summary(path: Path, slugs: set[str]) -> dict | None:
    if not path.is_file():
        return None
    source = json.loads(path.read_text())
    rows = [row for row in source.get("per_star", []) if row["slug"] in slugs]
    result = {
        "reference": source.get("baseline_arm"),
        "candidate": source.get("candidate_arm"),
        "star_count": len(rows),
        "bar": float(source.get("bar", BAR)),
    }
    for field in ("normalized_flux", "flux_total", "flux_continuum"):
        values = np.asarray([row[field]["max"] for row in rows], dtype=np.float64)
        result[field] = _metric(values)
    return result


def _arm_summary(rows: list[dict]) -> dict:
    return {
        "star_count": len(rows),
        "converged_count": sum(bool(row["converged"]) for row in rows),
        "converged_fraction": sum(bool(row["converged"]) for row in rows) / max(len(rows), 1),
        "summary": summarize(rows) if rows else None,
    }


def _structure_path(run_root: Path, arm: str, slug: str) -> Path | None:
    """Prefer full solver profiles; fall back to reusable product fields."""

    for folder in (run_root / "profiles" / arm, run_root / "products" / arm):
        path = folder / f"{slug}.npz"
        if path.is_file():
            return path
    return None


def _compare_structures(
    run_root: Path,
    reference: str,
    candidate: str,
    slugs: set[str],
) -> dict:
    available = []
    for slug in sorted(slugs):
        ref_path = _structure_path(run_root, reference, slug)
        cand_path = _structure_path(run_root, candidate, slug)
        if ref_path is None or cand_path is None:
            continue
        available.append((slug, ref_path, cand_path))
    result = {
        "reference": reference,
        "candidate": candidate,
        "star_count": len(available),
        "field_star_count": {},
        "fields": {},
        "note": (
            "The reusable two-field and six-field campaign saved structured product "
            "fields but not rosseland_opacity or radiative_acceleration; those two "
            "fields are reported only when both sides have full solver profiles."
        ),
    }
    for field in FIELDS:
        errors = []
        for _slug, ref_path, cand_path in available:
            with np.load(ref_path, allow_pickle=False) as ref, np.load(cand_path, allow_pickle=False) as cand:
                if field not in ref.files or field not in cand.files:
                    continue
                reference_values = np.asarray(ref[field], dtype=np.float64)
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
                raise ValueError(f"non-finite {field} difference for {_slug}")
            errors.append(error)
        result["field_star_count"][field] = len(errors)
        if errors:
            values = np.concatenate(errors)
            result["fields"][field] = {
                "metric": metric,
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }
    return result


def _subset_convergence(records: dict[str, list[dict]], slugs: set[str]) -> dict:
    return {
        arm: _arm_summary([row for row in records[arm] if row["slug"] in slugs])
        for arm in ARMS
    }


def _spectra_for_subset(spectral_paths: dict[str, Path], converged: dict[str, set[str]], slugs: set[str]) -> dict:
    six = converged[SIX] & slugs
    return {
        candidate: _spectral_summary(path, six & converged[candidate])
        for candidate, path in spectral_paths.items()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/four_initializer_benchmark_expanded_20260814"))
    parser.add_argument("--result-root", type=Path, default=Path("results/four_initializer_benchmark_expanded_20260814"))
    parser.add_argument("--manifest", type=Path, default=Path("results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json"))
    parser.add_argument("--previous-summary", type=Path, default=Path("results/atmosphere_interpolation_benchmark_20260813/four_initializer_comparison.json"))
    parser.add_argument("--out", type=Path, default=Path("results/four_initializer_benchmark_expanded_20260814/expanded_four_initializer_comparison.json"))
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text())
    records = {
        arm: _records(args.run_root / "records" / arm / "records.jsonl")
        for arm in ARMS
    }
    missing = [arm for arm, rows in records.items() if not rows]
    if missing:
        raise SystemExit(f"missing records: {missing}")
    converged = {arm: _converged(rows) for arm, rows in records.items()}
    all_four = set.intersection(*converged.values())
    previous_slugs = set(manifest["previous_60_slugs"])
    added_slugs = set(manifest["added_140_slugs"])
    if previous_slugs & added_slugs:
        raise SystemExit("previous and added subsets overlap")
    target_slugs = set(manifest["star_slugs"])
    if target_slugs != previous_slugs | added_slugs:
        raise SystemExit("manifest slug partition is inconsistent")

    spectral_paths = {
        "learned_reduced_state": args.result_root / "spectral_learned_reduced_state_vs_six.json",
        "grey15": args.result_root / "spectral_grey15_vs_six.json",
        "interpolated_full_state": args.result_root / "spectral_interpolated_full_state_vs_six.json",
    }
    spectral_named = {
        "expanded200": _spectra_for_subset(spectral_paths, converged, target_slugs),
        "previous60": _spectra_for_subset(spectral_paths, converged, previous_slugs),
        "added140": _spectra_for_subset(spectral_paths, converged, added_slugs),
        "all_four_common": _spectra_for_subset(spectral_paths, converged, all_four),
    }

    structure_named = {
        "expanded200": {
            arm: _compare_structures(args.run_root, SIX, arm, target_slugs)
            for arm in ARMS
            if arm != SIX
        },
        "previous60": {
            arm: _compare_structures(args.run_root, SIX, arm, previous_slugs)
            for arm in ARMS
            if arm != SIX
        },
        "added140": {
            arm: _compare_structures(args.run_root, SIX, arm, added_slugs)
            for arm in ARMS
            if arm != SIX
        },
        "all_four_common": {
            arm: _compare_structures(args.run_root, SIX, arm, all_four)
            for arm in ARMS
            if arm != SIX
        },
    }

    previous_report = None
    if args.previous_summary.is_file():
        old = json.loads(args.previous_summary.read_text())
        previous_report = {
            "path": str(args.previous_summary),
            "sha256": _sha256(args.previous_summary),
            "convergence": old.get("convergence"),
            "all_four_common_count": len(old.get("common_star_slugs", {}).get("all_four", [])),
            "spectra": old.get("spectra"),
        }

    report = {
        "format": "payne_zero_expanded_four_initializer_comparison_v1",
        "contract": {
            "same_four_arms": True,
            "same_solver": True,
            "primary_iteration_cap": 15,
            "spectrum_window_nm": [400.0, 900.0],
            "spectrum_resolution": 20000.0,
            "threshold": BAR,
            "sealed_test_opened": False,
        },
        "manifest": {
            "path": str(args.manifest),
            "sha256": _sha256(args.manifest),
            "star_count": len(target_slugs),
            "previous_count": len(previous_slugs),
            "added_count": len(added_slugs),
            "categories": manifest.get("categories"),
        },
        "convergence": {arm: _arm_summary(records[arm]) for arm in ARMS},
        "subsets": {
            "expanded200": {
                "star_count": len(target_slugs),
                "all_four_common_count": len(all_four),
                "convergence": _subset_convergence(records, target_slugs),
            },
            "previous60": {
                "star_count": len(previous_slugs),
                "all_four_common_count": len(all_four & previous_slugs),
                "convergence": _subset_convergence(records, previous_slugs),
            },
            "added140": {
                "star_count": len(added_slugs),
                "all_four_common_count": len(all_four & added_slugs),
                "convergence": _subset_convergence(records, added_slugs),
            },
        },
        "common_star_slugs": {
            "all_four": sorted(all_four),
            "previous60": sorted(previous_slugs),
            "added140": sorted(added_slugs),
        },
        "spectra": spectral_named,
        "final_structure": structure_named,
        "previous_report": previous_report,
        "record_counts": {arm: len(rows) for arm, rows in records.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}")
    for arm, entry in report["convergence"].items():
        print(f"{arm}: {entry['converged_count']}/{entry['star_count']}")
    print(f"all-four common: {len(all_four)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
