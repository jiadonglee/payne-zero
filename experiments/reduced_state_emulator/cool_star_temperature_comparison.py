"""Build a three-temperature continuation/MARCS comparison artifact.

Each case is supplied as ``temperature|two|six|marcs_reduced|marcs_full``.
The products are synthesized on one common spectrum grid so that the
two-field route, six-field continuation reference, and independent
MARCS-started endpoints can be compared directly.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.labels import StellarLabels

from .marcs_h5 import load_marcs_node
from .spectral_gate import (
    _absolute_stats,
    _continuum_scaled_stats,
    _load_spectrum_npz,
    _relative_stats,
    _synthesize_one,
)


STRUCTURE_FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "mass_density",
    "microturbulence",
)
SPECTRUM_FIELDS = ("normalized_flux", "flux_total", "flux_continuum")


def _load_product(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        missing = [field for field in STRUCTURE_FIELDS if field not in data.files]
        if missing:
            raise ValueError(f"{path} is missing structure fields: {missing}")
        result = {
            field: np.asarray(data[field], dtype=np.float64)
            for field in STRUCTURE_FIELDS
        }
    shape = result["temperature"].shape
    if shape != (80,):
        raise ValueError(f"{path} does not contain an 80-layer atmosphere: {shape}")
    if any(values.shape != shape for values in result.values()):
        raise ValueError(f"{path} has inconsistent structure-field shapes")
    if any(not np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError(f"{path} contains non-finite structure values")
    if np.any(result["column_mass"] <= 0.0) or np.any(
        np.diff(result["column_mass"]) <= 0.0
    ):
        raise ValueError(f"{path} has invalid column mass")
    return result


def _compare_spectra(
    baseline: dict[str, np.ndarray], candidate: dict[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    if not np.array_equal(baseline["wavelength_nm"], candidate["wavelength_nm"]):
        raise ValueError("comparison spectra do not share an identical wavelength grid")
    result: dict[str, dict[str, float]] = {}
    for field in SPECTRUM_FIELDS:
        result[field] = _absolute_stats(candidate[field], baseline[field])
    result["flux_total_continuum_scaled"] = _continuum_scaled_stats(
        candidate["flux_total"], baseline["flux_total"], baseline["flux_continuum"]
    )
    result["flux_continuum_relative"] = _relative_stats(
        candidate["flux_continuum"], baseline["flux_continuum"]
    )
    result["normalized_flux_relative"] = _relative_stats(
        candidate["normalized_flux"], baseline["normalized_flux"]
    )
    return result


def _parse_case(value: str) -> tuple[float, dict[str, Path]]:
    fields = value.split("|")
    if len(fields) != 5:
        raise argparse.ArgumentTypeError(
            "--case must be temperature|two|six|marcs_reduced|marcs_full"
        )
    try:
        temperature = float(fields[0])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case temperature must be numeric") from exc
    names = ("two", "six", "marcs_reduced", "marcs_full")
    paths = {name: Path(path) for name, path in zip(names, fields[1:])}
    return temperature, paths


def _write_structure_archive(
    out_root: Path,
    structures: dict[str, dict[str, dict[str, np.ndarray]]],
    raw_marcs: dict[str, dict[str, np.ndarray]],
) -> None:
    payload: dict[str, np.ndarray] = {}
    for temperature, products in structures.items():
        prefix = f"t{float(temperature):05.0f}"
        for arm, fields in products.items():
            for field, values in fields.items():
                payload[f"{prefix}__{arm}__{field}"] = values
    for temperature, fields in raw_marcs.items():
        prefix = f"t{float(temperature):05.0f}__marcs_raw"
        for field, values in fields.items():
            payload[f"{prefix}__{field}"] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(out_root / "structure_comparison.npz", **payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marcs-grid", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        required=True,
        help="temperature|two|six|marcs_reduced|marcs_full; repeat once per temperature",
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--wavelength-start-nm", type=float, default=400.0)
    parser.add_argument("--wavelength-end-nm", type=float, default=900.0)
    parser.add_argument("--resolution", type=float, default=20_000.0)
    args = parser.parse_args(argv)

    cases = dict(args.case)
    if len(cases) != len(args.case):
        raise SystemExit("duplicate temperatures in --case")
    out_root = args.out_root
    spectra_root = out_root / "spectra"
    out_root.mkdir(parents=True, exist_ok=True)

    structures: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    raw_marcs: dict[str, dict[str, np.ndarray]] = {}
    spectra: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    products_summary: dict[str, dict[str, str]] = {}
    synthesis_seconds: dict[str, float] = {}
    comparisons: dict[str, dict[str, dict[str, dict[str, float]]]] = {}

    for temperature in sorted(cases):
        paths = cases[temperature]
        label = StellarLabels(temperature, 5.0, 0.0, 0.0, 1.0)
        case_key = f"{temperature:.0f}"
        products = {arm: _load_product(path) for arm, path in paths.items()}
        structures[case_key] = products
        products_summary[case_key] = {arm: str(path) for arm, path in paths.items()}

        node = load_marcs_node(args.marcs_grid, label, verify_sha256=True)
        raw = {key: np.asarray(value, dtype=np.float64) for key, value in node.native_fields.items()}
        raw["column_mass"] = np.asarray(node.native_column_mass, dtype=np.float64)
        raw_marcs[case_key] = raw

        spectra[case_key] = {}
        for arm, product_path in paths.items():
            output = spectra_root / case_key / f"{arm}.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            _synthesize_one(
                product_path,
                output,
                wavelength_start_nm=args.wavelength_start_nm,
                wavelength_end_nm=args.wavelength_end_nm,
                resolution=args.resolution,
                molecular_lines=True,
                device=args.device,
                dtype=args.dtype,
            )
            synthesis_seconds[f"{case_key}/{arm}"] = float(time.perf_counter() - started)
            spectra[case_key][arm] = _load_spectrum_npz(output)

        baseline = spectra[case_key]["six"]
        case_comparisons: dict[str, dict[str, dict[str, float]]] = {}
        for arm in ("two", "marcs_reduced", "marcs_full"):
            case_comparisons[f"{arm}_vs_six"] = _compare_spectra(
                baseline, spectra[case_key][arm]
            )
        for arm in ("marcs_reduced", "marcs_full"):
            case_comparisons[f"two_vs_{arm}"] = _compare_spectra(
                spectra[case_key][arm], spectra[case_key]["two"]
            )
        comparisons[case_key] = case_comparisons

    _write_structure_archive(out_root, structures, raw_marcs)
    summary = {
        "format": "payne_zero_cool_star_temperature_comparison_v1",
        "temperatures": sorted(float(value) for value in cases),
        "track": {
            "log_surface_gravity": 5.0,
            "metallicity": 0.0,
            "alpha_enhancement": 0.0,
            "carbon_enhancement": 0.0,
            "microturbulence_km_s": 1.0,
        },
        "spectrum_contract": {
            "wavelength_nm": [args.wavelength_start_nm, args.wavelength_end_nm],
            "resolution": args.resolution,
            "dtype": args.dtype,
            "device": args.device,
            "molecular_lines": True,
        },
        "products": products_summary,
        "marcs_raw": {
            "path": str(args.marcs_grid),
            "native_layer_count": 56,
            "depth_coordinate": "native MARCS column mass",
            "sha256": node.source_sha256,
        },
        "synthesis_seconds": synthesis_seconds,
        "comparisons": comparisons,
        "notes": [
            "marcs_raw is a native 56-layer structural diagnostic and is not passed to the Payne-Zero spectrum synthesizer.",
            "marcs_reduced and marcs_full are complete Payne-Zero endpoints obtained from independent MARCS-started paths.",
            "The six-field continuation is the internal fixed-point reference for the two-field route at each temperature.",
        ],
    }
    (out_root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {out_root / 'comparison_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
