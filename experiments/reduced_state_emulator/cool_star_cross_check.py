"""Cross-check cool-star products that the standard gate never compares.

The standard spectral gate only scores arms against the production
six-field reference, which does not exist below 3900 K.  This script
answers two validation questions the gate cannot:

1. Does the 250 K continuation land on the same solution as an
   independent MARCS-initialized run at 3750 K?
2. Do the full-carry and reduced-rematerialized 250 K continuations
   agree with each other at 3500 K, where no independent reference
   exists?

Example
-------
python -m experiments.reduced_state_emulator.cool_star_cross_check \
    --run-root runs/cool_star_step_test \
    --out results/cool_star_step_test/cross_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .spectral_gate import gate_one

BAR = 5.0e-3

SLUG_3750 = "t03750.0_g+5.00_m+0.00_a+0.00_x1.00"
SLUG_3500 = "t03500.0_g+5.00_m+0.00_a+0.00_x1.00"

COMPARISONS = (
    {
        "name": "marcs_vs_continuation_250_full_carry@3750",
        "slug": SLUG_3750,
        "baseline_arm": "marcs_target_reduced",
        "candidate_arm": "continuation_250_full_carry",
    },
    {
        "name": "marcs_vs_continuation_250_reduced_rematerialized@3750",
        "slug": SLUG_3750,
        "baseline_arm": "marcs_target_reduced",
        "candidate_arm": "continuation_250_reduced_rematerialized",
    },
    {
        "name": "marcs_vs_anchor_full_carry_direct@3750",
        "slug": SLUG_3750,
        "baseline_arm": "marcs_target_reduced",
        "candidate_arm": "anchor_full_carry",
    },
    {
        "name": "marcs_vs_anchor_reduced_rematerialized_direct@3750",
        "slug": SLUG_3750,
        "baseline_arm": "marcs_target_reduced",
        "candidate_arm": "anchor_reduced_rematerialized",
    },
    {
        "name": "full_carry_vs_reduced_rematerialized_continuation_250@3500",
        "slug": SLUG_3500,
        "baseline_arm": "continuation_250_full_carry",
        "candidate_arm": "continuation_250_reduced_rematerialized",
    },
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spectra-dir", type=Path, default=None)
    parser.add_argument("--bar", type=float, default=BAR)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    args = parser.parse_args()

    products_dir = args.run_root / "products"
    spectra_dir = args.spectra_dir or (args.run_root / "spectra_cross_check")
    results = {}
    for spec in COMPARISONS:
        missing = [
            arm
            for arm in (spec["baseline_arm"], spec["candidate_arm"])
            if not (products_dir / arm / f"{spec['slug']}.npz").is_file()
        ]
        if missing:
            results[spec["name"]] = {"pass": False, "error": f"missing products: {missing}"}
            continue
        check = gate_one(
            spec["slug"],
            products_dir,
            spectra_dir,
            wavelength_start_nm=400.0,
            wavelength_end_nm=900.0,
            resolution=20000.0,
            molecular_lines=True,
            device=args.device,
            dtype=args.dtype,
            baseline_arm=spec["baseline_arm"],
            candidate_arm=spec["candidate_arm"],
        )
        check["pass"] = all(
            check[field]["max"] <= args.bar
            for field in ("normalized_flux", "flux_total", "flux_continuum")
        )
        check["bar"] = args.bar
        results[spec["name"]] = check
        print(json.dumps({spec["name"]: check}, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
