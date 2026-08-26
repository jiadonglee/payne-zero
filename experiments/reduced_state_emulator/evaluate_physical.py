"""Evaluate physical ``(m,T)`` predictions against a frozen manifest.

The acceptance checks use both kinds of threshold in the repair plan:
point-wise profile p95 values and the worst layer of every star. The script
does not read or expose a sealed audit result until it is explicitly called
with that manifest after a checkpoint has been frozen.
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
DEFAULT_MANIFEST = REPO_ROOT / "results" / "reconstruction_metrics.json"
DEFAULT_PREDICTION = (
    REPO_ROOT
    / "artifacts"
    / "reduced_state_emulator"
    / "physical"
    / "predicted_physical_ensemble.npz"
)
DEFAULT_OUT = REPO_ROOT / "results" / "physical_profile_accuracy.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
        "max": float(np.max(values)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--indices-from", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--name", default="evaluation")
    args = parser.parse_args(argv)

    manifest = json.loads(args.indices_from.read_text())
    indices = np.asarray(manifest["star_indices"], dtype=np.int64)
    with np.load(args.prediction, allow_pickle=False) as prediction:
        prediction_indices = np.asarray(prediction["star_indices"], dtype=np.int64)
        if not np.array_equal(prediction_indices, indices):
            raise ValueError("prediction and evaluation index manifests differ")
        predicted_mass = np.asarray(prediction["column_mass"], dtype=np.float64)
        predicted_temperature = np.asarray(
            prediction["temperature"], dtype=np.float64
        )
    with np.load(args.corpus, allow_pickle=False) as corpus:
        profiles = np.asarray(corpus["atmosphere_profiles"][indices], dtype=np.float64)
        truth_mass = profiles[:, :, 0]
        truth_temperature = profiles[:, :, 1]

    temperature_error = np.abs(predicted_temperature - truth_temperature) / truth_temperature
    mass_error = np.abs(np.log10(predicted_mass) - np.log10(truth_mass))
    temperature_star_max = np.max(temperature_error, axis=1)
    mass_star_max = np.max(mass_error, axis=1)
    per_star = [
        {
            "star_index": int(index),
            "temperature_max_relative": float(temperature_star_max[row]),
            "mass_max_dex": float(mass_star_max[row]),
            "temperature_pass": bool(temperature_star_max[row] < 0.10),
            "mass_pass": bool(mass_star_max[row] < 0.10),
        }
        for row, index in enumerate(indices)
    ]
    prediction_provenance_path = args.prediction.with_suffix(".json")
    prediction_provenance = {}
    if prediction_provenance_path.exists():
        prediction_provenance = json.loads(prediction_provenance_path.read_text())

    result = {
        "name": args.name,
        "star_count": int(len(indices)),
        "star_indices": [int(index) for index in indices],
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "indices_manifest": str(args.indices_from),
        "indices_manifest_sha256": _sha256(args.indices_from),
        "prediction": str(args.prediction),
        "prediction_sha256": _sha256(args.prediction),
        "prediction_provenance": str(prediction_provenance_path),
        "prediction_provenance_sha256": (
            _sha256(prediction_provenance_path)
            if prediction_provenance_path.exists()
            else None
        ),
        "model_checkpoints": prediction_provenance.get("checkpoints", []),
        "pointwise": {
            "temperature_relative": _percentiles(temperature_error.ravel()),
            "mass_dex": _percentiles(mass_error.ravel()),
        },
        "per_star_maximum": {
            "temperature_relative": _percentiles(temperature_star_max),
            "mass_dex": _percentiles(mass_star_max),
        },
        "per_star": per_star,
        "failure_stars": {
            "temperature": [
                row["star_index"] for row in per_star if not row["temperature_pass"]
            ],
            "mass": [
                row["star_index"] for row in per_star if not row["mass_pass"]
            ],
        },
        "thresholds": {
            "temperature_star_max_lt": 0.10,
            "mass_star_max_lt_dex": 0.10,
            "temperature_pointwise_p95_lte": 3.74e-3,
            "mass_pointwise_p95_lte_dex": 1.52e-2,
        },
        "gate": {
            "temperature_no_blowout": bool(np.all(temperature_star_max < 0.10)),
            "mass_no_blowout": bool(np.all(mass_star_max < 0.10)),
            "temperature_pointwise_p95": bool(
                np.percentile(temperature_error, 95.0) <= 3.74e-3
            ),
            "mass_pointwise_p95": bool(
                np.percentile(mass_error, 95.0) <= 1.52e-2
            ),
        },
    }
    result["gate"]["profile_gate_passed"] = all(result["gate"].values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{args.name}: T p95={result['pointwise']['temperature_relative']['p95']:.3e}, "
        f"m p95={result['pointwise']['mass_dex']['p95']:.3e} dex, "
        f"bad stars={int((temperature_star_max >= 0.10).sum())}/"
        f"{int((mass_star_max >= 0.10).sum())}, "
        f"gate={result['gate']['profile_gate_passed']}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
