"""Continue an independent MARCS-started 3750 K solution to 3500 K.

This is a targeted basin-independence check.  The source product is the
converged 3750 K MARCS ``(m,T)`` run; the target is solved twice, carrying all
six fields or rematerializing the four dependent fields from the carried
``(m,T)`` pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cool_star_step_test import (
    TrackSpec,
    _production_atmosphere,
    _reconstruct_from_mt,
    _retarget_full_state,
    _solve_attempt,
)
from .cool_star_targeted_3700 import _model_from_product


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-product", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--iteration-cap", type=int, default=30)
    parser.add_argument(
        "--mode",
        choices=("both", "full_carry", "reduced_rematerialized"),
        default="both",
    )
    args = parser.parse_args()

    track = TrackSpec(log_surface_gravity=5.0, metallicity=0.0)
    source_labels = track.labels(3750.0)
    target_labels = track.labels(3500.0)
    source = _model_from_product(args.source_product, source_labels)
    target_template = _production_atmosphere(target_labels)
    records = []

    modes = (
        ("full_carry", "reduced_rematerialized")
        if args.mode == "both"
        else (args.mode,)
    )
    for mode in modes:
        if mode == "full_carry":
            seed = _retarget_full_state(source, target_template)
        else:
            seed = _reconstruct_from_mt(
                target_labels, source.column_mass, source.temperature
            )
        record, _state = _solve_attempt(
            track=track,
            method=f"marcs_3750_to_3500_{mode}",
            schedule="marcs_250K",
            source_temperature=3750.0,
            target_labels=target_labels,
            initial_atmosphere=seed,
            product_dir=args.out_root / "products" / mode,
            iteration_cap=args.iteration_cap,
        )
        records.append(record)

    output = {
        "format": "payne_zero_marcs_to_3500_targeted_v1",
        "source_product": str(args.source_product),
        "source_temperature": 3750.0,
        "target_temperature": 3500.0,
        "records": records,
    }
    output_path = args.out_root / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
