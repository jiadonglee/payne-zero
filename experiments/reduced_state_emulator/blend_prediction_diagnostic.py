"""Create a diagnostic log-space blend of two materialized (m,T) predictions.

This is for basin diagnosis only.  The resulting artifact must not be used as
a production prediction because it deliberately uses the shipped teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prediction", type=Path, required=True)
    parser.add_argument("--teacher-prediction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--logg-threshold", type=float, default=3.2)
    args = parser.parse_args(argv)
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")

    with np.load(args.base_prediction, allow_pickle=False) as base:
        result = {name: np.asarray(base[name]) for name in base.files}
    with np.load(args.teacher_prediction, allow_pickle=False) as teacher:
        if not np.array_equal(result["star_indices"], teacher["star_indices"]):
            raise SystemExit("base and teacher predictions cover different stars")
        labels = np.asarray(result["labels"], dtype=np.float64)
        teacher_mass = np.asarray(teacher["column_mass"], dtype=np.float64)
        teacher_temperature = np.asarray(teacher["temperature"], dtype=np.float64)

    mask = labels[:, 1] < float(args.logg_threshold)
    mass = np.asarray(result["column_mass"], dtype=np.float64).copy()
    temperature = np.asarray(result["temperature"], dtype=np.float64).copy()
    mass[mask] = np.exp(
        (1.0 - args.alpha) * np.log(mass[mask])
        + args.alpha * np.log(teacher_mass[mask])
    )
    temperature[mask] = np.exp(
        (1.0 - args.alpha) * np.log(temperature[mask])
        + args.alpha * np.log(teacher_temperature[mask])
    )
    result["column_mass"] = mass
    result["temperature"] = temperature
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **result)
    provenance = {
        "base_prediction": str(args.base_prediction),
        "base_prediction_sha256": _sha256(args.base_prediction),
        "teacher_prediction": str(args.teacher_prediction),
        "teacher_prediction_sha256": _sha256(args.teacher_prediction),
        "out": str(args.out),
        "alpha": float(args.alpha),
        "logg_threshold": float(args.logg_threshold),
        "masked_star_count": int(mask.sum()),
        "diagnostic_only": True,
    }
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.out} ({int(mask.sum())} stars blended)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
