"""Build a 3500 K two-field/Payne/MARCS comparison artifact.

The comparison keeps two MARCS notions separate:

* ``marcs_raw`` is the native 56-layer MARCS node, retained as a structural
  diagnostic.  It is not passed to Payne-Zero's spectrum synthesizer.
* ``marcs_started_*`` are complete Payne-Zero products obtained after starting
  from a MARCS-derived state and continuing to 3500 K.  These can be
  synthesized on the same 400--900 nm, R=20,000 grid as the continuation
  products.

The script is intended to run in the Linux environment because the certified
line/spectrum path is not available in the local macOS environment.  It writes
portable NPZ/JSON artifacts that can then be plotted or inspected locally.
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


SLUG = "t03500.0_g+5.00_m+0.00_a+0.00_x1.00"
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
        result = {field: np.asarray(data[field], dtype=np.float64) for field in STRUCTURE_FIELDS}
    shape = result["temperature"].shape
    if shape != (80,):
        raise ValueError(f"{path} does not contain an 80-layer atmosphere: {shape}")
    if any(values.shape != shape for values in result.values()):
        raise ValueError(f"{path} has inconsistent structure-field shapes")
    if any(not np.all(np.isfinite(values)) for values in result.values()):
        raise ValueError(f"{path} contains non-finite structure values")
    if np.any(result["column_mass"] <= 0.0) or np.any(np.diff(result["column_mass"]) <= 0.0):
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


def _write_structure_archive(
    out_root: Path,
    products: dict[str, dict[str, np.ndarray]],
    raw_marcs: dict[str, np.ndarray],
) -> None:
    payload: dict[str, np.ndarray] = {}
    for arm, fields in products.items():
        for field, values in fields.items():
            payload[f"{arm}__{field}"] = values
    for field, values in raw_marcs.items():
        payload[f"marcs_raw__{field}"] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(out_root / "structure_comparison.npz", **payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--two-field-product", type=Path, required=True)
    parser.add_argument("--six-field-product", type=Path, required=True)
    parser.add_argument("--marcs-reduced-product", type=Path, required=True)
    parser.add_argument("--marcs-full-product", type=Path, required=True)
    parser.add_argument("--marcs-grid", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--wavelength-start-nm", type=float, default=400.0)
    parser.add_argument("--wavelength-end-nm", type=float, default=900.0)
    parser.add_argument("--resolution", type=float, default=20_000.0)
    args = parser.parse_args(argv)

    out_root = args.out_root
    spectra_root = out_root / "spectra"
    out_root.mkdir(parents=True, exist_ok=True)
    labels = StellarLabels(3500.0, 5.0, 0.0, 0.0, 1.0)
    product_paths = {
        "two_field_continuation": args.two_field_product,
        "six_field_continuation": args.six_field_product,
        "marcs_started_reduced": args.marcs_reduced_product,
        "marcs_started_full": args.marcs_full_product,
    }
    products = {name: _load_product(path) for name, path in product_paths.items()}

    node = load_marcs_node(args.marcs_grid, labels, verify_sha256=True)
    raw_marcs = node.native_fields
    raw_marcs["column_mass"] = node.native_column_mass

    spectra: dict[str, dict[str, np.ndarray]] = {}
    synthesis_seconds: dict[str, float] = {}
    for arm, product_path in product_paths.items():
        output = spectra_root / f"{arm}.npz"
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
        synthesis_seconds[arm] = float(time.perf_counter() - started)
        spectra[arm] = _load_spectrum_npz(output)

    comparisons = {}
    baseline = spectra["six_field_continuation"]
    for arm in ("two_field_continuation", "marcs_started_reduced", "marcs_started_full"):
        comparisons[f"{arm}_vs_six_field_continuation"] = _compare_spectra(
            baseline, spectra[arm]
        )
    for arm, benchmark in (
        ("marcs_started_reduced", "two_field_continuation"),
        ("marcs_started_full", "two_field_continuation"),
    ):
        comparisons[f"two_field_continuation_vs_{arm}"] = _compare_spectra(
            spectra[arm], spectra[benchmark]
        )

    summary = {
        "format": "payne_zero_cool_star_3500_comparison_v1",
        "labels": labels.as_kwargs(),
        "slug": SLUG,
        "spectrum_contract": {
            "wavelength_nm": [args.wavelength_start_nm, args.wavelength_end_nm],
            "resolution": args.resolution,
            "dtype": args.dtype,
            "device": args.device,
            "molecular_lines": True,
        },
        "products": {name: str(path) for name, path in product_paths.items()},
        "marcs_raw": {
            "path": str(args.marcs_grid),
            "native_layer_count": int(node.native_temperature.size),
            "sha256": node.source_sha256,
            "depth_coordinate": "native MARCS column mass",
        },
        "synthesis_seconds": synthesis_seconds,
        "comparisons": comparisons,
        "notes": [
            "marcs_raw is a native 56-layer structural diagnostic, not a Payne-Zero spectrum input.",
            "marcs_started_* are complete Payne-Zero products obtained from MARCS-derived starts.",
            "The six-field continuation is the internal fixed-point reference for the two-field route.",
        ],
    }
    _write_structure_archive(out_root, products, raw_marcs)
    (out_root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {out_root / 'comparison_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
