"""Blend two physical ``(m,T)`` prediction artifacts in label space.

This is a small, auditable correction probe: the base ensemble remains the
production candidate, while a correction ensemble contributes a configurable
fraction only in a label region selected from the development set. Blending
``log(T)`` and ``log(m)`` preserves positive temperatures and strict mass
monotonicity when both inputs have those properties.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--correction", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--logg-threshold", type=float, default=3.0)
    args = parser.parse_args(argv)
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    with np.load(args.base, allow_pickle=False) as base, np.load(
        args.correction, allow_pickle=False
    ) as correction:
        base_indices = np.asarray(base["star_indices"], dtype=np.int64)
        correction_indices = np.asarray(correction["star_indices"], dtype=np.int64)
        if not np.array_equal(base_indices, correction_indices):
            raise ValueError("base and correction prediction manifests differ")
        labels = np.asarray(base["labels"], dtype=np.float64)
        if labels.shape[0] != len(base_indices):
            raise ValueError("base labels and star_indices differ in length")
        base_mass = np.asarray(base["column_mass"], dtype=np.float64)
        base_temperature = np.asarray(base["temperature"], dtype=np.float64)
        correction_mass = np.asarray(correction["column_mass"], dtype=np.float64)
        correction_temperature = np.asarray(correction["temperature"], dtype=np.float64)

        low_gravity = labels[:, 1] < float(args.logg_threshold)
        mass = base_mass.copy()
        temperature = base_temperature.copy()
        mass[low_gravity] = np.exp(
            (1.0 - args.alpha) * np.log(base_mass[low_gravity])
            + args.alpha * np.log(correction_mass[low_gravity])
        )
        temperature[low_gravity] = np.exp(
            (1.0 - args.alpha) * np.log(base_temperature[low_gravity])
            + args.alpha * np.log(correction_temperature[low_gravity])
        )
        if np.any(np.diff(mass, axis=1) <= 0.0):
            raise ValueError("blended column-mass profiles are not strictly monotone")

        payload = {
            "star_indices": base_indices,
            "labels": labels,
            "column_mass": mass,
            "temperature": temperature,
        }
        for key in ("truth_column_mass", "truth_temperature"):
            if key in base:
                payload[key] = np.asarray(base[key])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **payload)

    provenance = {
        "base_prediction": str(args.base),
        "base_prediction_sha256": _sha256(args.base),
        "correction_prediction": str(args.correction),
        "correction_prediction_sha256": _sha256(args.correction),
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "out": str(args.out),
        "out_sha256": _sha256(args.out),
        "alpha": float(args.alpha),
        "logg_threshold": float(args.logg_threshold),
        "corrected_star_count": int(np.sum(low_gravity)),
        "star_count": int(len(base_indices)),
    }
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"wrote {args.out} ({len(base_indices)} stars, corrected="
        f"{int(np.sum(low_gravity))}, alpha={args.alpha}, logg<{args.logg_threshold})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
