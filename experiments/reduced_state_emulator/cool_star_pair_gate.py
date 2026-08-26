"""Compare two explicitly supplied cool-star atmosphere products.

This is a small wrapper around the established three-metric spectral gate. It
is useful for follow-up products whose arm name is not part of the original
pilot manifest, such as an independent MARCS-started 3500 K solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .spectral_gate import gate_one


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-product", type=Path, required=True)
    parser.add_argument("--candidate-product", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--slug", default="t03500.0_g+5.00_m+0.00_a+0.00_x1.00"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--bar", type=float, default=5.0e-3)
    args = parser.parse_args()

    products_dir = args.out.parent / "pair_gate_products"
    baseline_arm = "baseline"
    candidate_arm = "candidate"
    for arm, source in (
        (baseline_arm, args.baseline_product),
        (candidate_arm, args.candidate_product),
    ):
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        arm_dir = products_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)
        target = arm_dir / f"{args.slug}.npz"
        target.unlink(missing_ok=True)
        target.symlink_to(source)

    spectra_dir = args.out.parent / "pair_gate_spectra"
    metrics = gate_one(
        args.slug,
        products_dir,
        spectra_dir,
        wavelength_start_nm=400.0,
        wavelength_end_nm=900.0,
        resolution=20000.0,
        molecular_lines=True,
        device=args.device,
        dtype=args.dtype,
        baseline_arm=baseline_arm,
        candidate_arm=candidate_arm,
    )
    metrics["bar"] = args.bar
    metrics["device"] = args.device
    metrics["dtype"] = args.dtype
    metrics["baseline_product"] = str(args.baseline_product)
    metrics["candidate_product"] = str(args.candidate_product)
    metrics["pass"] = all(
        metrics[field]["max"] <= args.bar
        for field in ("normalized_flux", "flux_total", "flux_continuum")
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"wrote {args.out}")
    return 0 if metrics["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
