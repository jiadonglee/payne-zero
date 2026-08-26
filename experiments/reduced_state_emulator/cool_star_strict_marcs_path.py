"""Re-converge a MARCS-started cool-star path with an all-layer stop.

The production solver historically stops on the deep-layer temperature change
only.  This diagnostic keeps the same physics and iteration map but also
requires the all-layer change to be below the supplied threshold.  It tests
whether the small spectral disagreement between independent MARCS and
continuation solutions is simply an early-stop effect.
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
    parser.add_argument(
        "--mode",
        choices=("both", "full_carry", "reduced_rematerialized"),
        default="both",
    )
    parser.add_argument("--source-temperature", type=float, default=3750.0)
    parser.add_argument("--target-temperature", type=float, default=3500.0)
    parser.add_argument("--logg", type=float, default=5.0)
    parser.add_argument("--metallicity", type=float, default=0.0)
    parser.add_argument("--all-layer-threshold", type=float, default=5.0e-4)
    parser.add_argument("--iteration-cap", type=int, default=30)
    args = parser.parse_args()
    if args.all_layer_threshold <= 0.0:
        raise SystemExit("--all-layer-threshold must be positive")

    track = TrackSpec(log_surface_gravity=args.logg, metallicity=args.metallicity)
    source_labels = track.labels(args.source_temperature)
    target_labels = track.labels(args.target_temperature)
    source = _model_from_product(args.source_product, source_labels)
    records = []
    strict_source_record, strict_source_state = _solve_attempt(
        track=track,
        method="strict_marcs_source_reconvergence",
        schedule="strict_all_layer",
        source_temperature=args.source_temperature,
        target_labels=source_labels,
        initial_atmosphere=source,
        product_dir=args.out_root / "products" / "strict_source",
        iteration_cap=args.iteration_cap,
        maximum_all_layer_relative_temperature_change=args.all_layer_threshold,
    )
    records.append(strict_source_record)

    modes = (
        ("full_carry", "reduced_rematerialized")
        if args.mode == "both"
        else (args.mode,)
    )
    target_template = _production_atmosphere(target_labels)
    for mode in modes:
        if strict_source_state is None:
            records.append(
                {
                    "method": f"strict_marcs_path_{mode}",
                    "schedule": "strict_all_layer",
                    "target_temperature": args.target_temperature,
                    "status": "blocked_by_source_reconvergence",
                    "converged": False,
                    "solver_converged": False,
                    "survives_solver": False,
                    "survives": False,
                }
            )
            continue
        if mode == "full_carry":
            seed = _retarget_full_state(strict_source_state, target_template)
        else:
            seed = _reconstruct_from_mt(
                target_labels,
                strict_source_state.column_mass,
                strict_source_state.temperature,
            )
        record, _state = _solve_attempt(
            track=track,
            method=f"strict_marcs_path_{mode}",
            schedule="strict_all_layer",
            source_temperature=args.source_temperature,
            target_labels=target_labels,
            initial_atmosphere=seed,
            product_dir=args.out_root / "products" / mode,
            iteration_cap=args.iteration_cap,
            maximum_all_layer_relative_temperature_change=args.all_layer_threshold,
        )
        records.append(record)

    output = {
        "format": "payne_zero_cool_star_strict_marcs_path_v1",
        "source_product": str(args.source_product),
        "source_temperature": args.source_temperature,
        "target_temperature": args.target_temperature,
        "all_layer_threshold": args.all_layer_threshold,
        "iteration_cap": args.iteration_cap,
        "records": records,
    }
    output_path = args.out_root / "summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output_path}")
    return 0 if all(record.get("survives_solver", False) for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
