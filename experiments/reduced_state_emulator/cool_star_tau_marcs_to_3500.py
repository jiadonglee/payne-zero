"""Solve a 3750 -> 3500 K path from a depth-aware MARCS initializer.

The original cool-star pilot placed the native MARCS profile on the Payne grid
by uniform ``log m``.  This targeted runner uses the native MARCS ``tau5000``
coordinate instead, with the loader's controlled edge extrapolation, and then
compares full six-field carry with two-field rematerialization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cool_star_step_test import (
    TrackSpec,
    _production_atmosphere,
    _reconstruct_from_mt,
    _retarget_full_state,
    _solve_attempt,
)
from .marcs_h5 import (
    EXPECTED_MARCS_SHA256,
    inspect_marcs_grid,
    load_marcs_node,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marcs-grid", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--source-profile",
        type=Path,
        default=None,
        help="use a precomputed physical-Rosseland profile instead of raw MARCS",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "full_carry", "reduced_rematerialized"),
        default="both",
    )
    parser.add_argument("--iteration-cap", type=int, default=30)
    args = parser.parse_args()

    grid = Path(args.marcs_grid).expanduser().resolve()
    digest = sha256_file(grid)
    if digest != EXPECTED_MARCS_SHA256:
        raise ValueError(f"MARCS SHA-256 mismatch: {digest} != {EXPECTED_MARCS_SHA256}")
    schema = inspect_marcs_grid(
        grid, verify_sha256=False, expected_sha256=None
    )
    track = TrackSpec(log_surface_gravity=5.0, metallicity=0.0)
    source_labels = track.labels(3750.0)
    target_labels = track.labels(3500.0)
    if args.source_profile is None:
        source_node = load_marcs_node(
            grid,
            source_labels,
            schema=schema,
            verify_sha256=False,
            expected_sha256=None,
            depth_coordinate="tau5000",
        )
        source_column_mass = source_node.reduced_column_mass
        source_temperature = source_node.reduced_temperature
        source_kind = "tau5000"
    else:
        with np.load(args.source_profile, allow_pickle=False) as profile:
            source_column_mass = np.asarray(profile["column_mass"], dtype=np.float64)
            source_temperature = np.asarray(profile["temperature"], dtype=np.float64)
        source_kind = "physical_rosseland_log_tau"
    output_root = Path(args.out_root)
    output_root.mkdir(parents=True, exist_ok=True)
    anchor_record, anchor_state = _solve_attempt(
        track=track,
        method="marcs_tau5000_anchor_3750",
        schedule="marcs_tau5000_anchor",
        source_temperature=None,
        target_labels=source_labels,
        initial_atmosphere=_reconstruct_from_mt(
            source_labels,
            source_column_mass,
            source_temperature,
        ),
        product_dir=output_root / "products" / "anchor_3750",
        iteration_cap=args.iteration_cap,
    )
    records = [anchor_record]
    if anchor_state is not None:
        target_template = _production_atmosphere(target_labels)
        modes = (
            ("full_carry", "reduced_rematerialized")
            if args.mode == "both"
            else (args.mode,)
        )
        for mode in modes:
            if mode == "full_carry":
                seed = _retarget_full_state(anchor_state, target_template)
            else:
                seed = _reconstruct_from_mt(
                    target_labels,
                    anchor_state.column_mass,
                    anchor_state.temperature,
                )
            record, _state = _solve_attempt(
                track=track,
                method=f"marcs_tau5000_3750_to_3500_{mode}",
                schedule="marcs_tau5000_250K",
                source_temperature=3750.0,
                target_labels=target_labels,
                initial_atmosphere=seed,
                product_dir=output_root / "products" / mode,
                iteration_cap=args.iteration_cap,
            )
            records.append(record)

    summary = {
        "format": "payne_zero_tau5000_marcs_to_3500_v1",
        "marcs_grid": str(grid),
        "marcs_sha256": digest,
        "marcs_depth_coordinate": (
            "tau5000" if args.source_profile is None else "payne_rosseland_log_tau"
        ),
        "source_kind": source_kind,
        "source_profile": (
            None if args.source_profile is None else str(args.source_profile)
        ),
        "track_id": track.track_id,
        "source_temperature": 3750.0,
        "target_temperature": 3500.0,
        "records": records,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
