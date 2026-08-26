"""Export native MARCS structural nodes for a small auditable comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bench.labels import StellarLabels

from .marcs_h5 import load_marcs_node


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marcs-grid", type=Path, required=True)
    parser.add_argument("--temperatures", nargs="+", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload: dict[str, np.ndarray] = {}
    sha256 = None
    for temperature in args.temperatures:
        labels = StellarLabels(float(temperature), 5.0, 0.0, 0.0, 1.0)
        node = load_marcs_node(args.marcs_grid, labels, verify_sha256=True)
        sha256 = node.source_sha256
        prefix = f"t{float(temperature):05.0f}__marcs_raw"
        for field, values in node.native_fields.items():
            payload[f"{prefix}__{field}"] = np.asarray(values, dtype=np.float64)
        payload[f"{prefix}__column_mass"] = np.asarray(
            node.native_column_mass, dtype=np.float64
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"wrote {args.out}")
    print(f"sha256={sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
