"""Run an independent native-MARCS path to one cool-star target.

The source temperature and target temperature must be native nodes in the
MARCS HDF5 grid.  The source node is reconstructed into Payne-Zero's 80-layer
state and re-converged first.  The target is then reached twice: once by
carrying all six fields and once by carrying only ``(m,T)`` and rebuilding the
other four fields.
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
from .marcs_h5 import load_marcs_node


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marcs-grid", type=Path, required=True)
    parser.add_argument("--source-temperature", type=float, required=True)
    parser.add_argument("--target-temperature", type=float, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--logg", type=float, default=5.0)
    parser.add_argument("--metallicity", type=float, default=0.0)
    parser.add_argument("--iteration-cap", type=int, default=30)
    args = parser.parse_args(argv)

    track = TrackSpec(
        log_surface_gravity=args.logg,
        metallicity=args.metallicity,
    )
    source_labels = track.labels(args.source_temperature)
    target_labels = track.labels(args.target_temperature)
    args.out_root.mkdir(parents=True, exist_ok=True)

    node = load_marcs_node(args.marcs_grid, source_labels, verify_sha256=True)
    source_seed = _reconstruct_from_mt(
        source_labels,
        node.reduced_column_mass,
        node.reduced_temperature,
    )
    source_record, source_state = _solve_attempt(
        track=track,
        method="marcs_source_reduced",
        schedule="marcs_native_source",
        source_temperature=None,
        target_labels=source_labels,
        initial_atmosphere=source_seed,
        product_dir=args.out_root / "products" / "source_reduced",
        iteration_cap=args.iteration_cap,
    )

    records = [source_record]
    if source_state is not None:
        target_template = _production_atmosphere(target_labels)
        for mode in ("full_carry", "reduced_rematerialized"):
            if mode == "full_carry":
                seed = _retarget_full_state(source_state, target_template)
            else:
                seed = _reconstruct_from_mt(
                    target_labels,
                    source_state.column_mass,
                    source_state.temperature,
                )
            record, _state = _solve_attempt(
                track=track,
                method=(
                    f"marcs_{args.source_temperature:.0f}_to_"
                    f"{args.target_temperature:.0f}_{mode}"
                ),
                schedule=f"marcs_{abs(args.source_temperature - args.target_temperature):.0f}K",
                source_temperature=args.source_temperature,
                target_labels=target_labels,
                initial_atmosphere=seed,
                product_dir=args.out_root / "products" / mode,
                iteration_cap=args.iteration_cap,
            )
            records.append(record)

    output = {
        "format": "payne_zero_marcs_native_path_v1",
        "marcs_grid": str(args.marcs_grid),
        "source_temperature": args.source_temperature,
        "target_temperature": args.target_temperature,
        "track": track.as_json(),
        "records": records,
    }
    output_path = args.out_root / "summary.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output_path}", flush=True)
    return 0 if all(record.get("survives_solver", False) for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
