"""Materialize the shipped six-field initializer's ``(m,T)`` for comparison.

This is a diagnostic artifact, not the reduced-state production model.  Keeping
the prediction on disk makes the default-trajectory correction probe
reproducible without calling the torch warm-start model from a solver worker.
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
DEFAULT_MANIFEST = REPO_ROOT / "results" / "sealed_solver_subset_20260808.json"
DEFAULT_OUT = REPO_ROOT / "results" / "production_six_field_mT.npz"
LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    indices = np.asarray(
        json.loads(args.manifest.read_text())["star_indices"], dtype=np.int64
    )
    with np.load(args.corpus, allow_pickle=False) as data:
        labels_json = [json.loads(str(value)) for value in data["labels_json"][indices]]
        labels = np.asarray(
            [[row[field] for field in LABEL_FIELDS] for row in labels_json],
            dtype=np.float64,
        )
        truth_column_mass = np.asarray(data["atmosphere_profiles"][indices, :, 0])
        truth_temperature = np.asarray(data["atmosphere_profiles"][indices, :, 1])

    from payne_zero_atmosphere.warm_start import emulator_warm_start_model

    atmospheres = [
        emulator_warm_start_model(
            device="cpu",
            **{field: float(row[field]) for field in LABEL_FIELDS},
        )[0]
        for row in labels_json
    ]
    column_mass = np.asarray([atmosphere.column_mass for atmosphere in atmospheres])
    temperature = np.asarray([atmosphere.temperature for atmosphere in atmospheres])
    if np.any(np.diff(column_mass, axis=1) <= 0.0):
        raise SystemExit("the shipped initializer produced a non-monotonic profile")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        star_indices=indices,
        labels=labels,
        column_mass=column_mass,
        temperature=temperature,
        truth_column_mass=truth_column_mass,
        truth_temperature=truth_temperature,
    )
    provenance = {
        "prediction": "production_six_field_mT",
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "output": str(args.out),
        "output_sha256": _sha256(args.out),
        "star_count": int(len(indices)),
    }
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.out} ({len(indices)} stars)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
