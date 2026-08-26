"""Compare direct six-field output and all reduced-state reconstructions.

The four arms are evaluated on the same stars:

* ``direct_six_field`` — the shipped checkpoint's four direct outputs;
* ``reconstruct_six_mT`` — physics rebuilt from that checkpoint's own ``m,T``;
* ``reconstruct_truth_mT`` — the oracle physics rebuild from corpus truth;
* ``reconstruct_reduced_mT`` — physics rebuilt from the new reduced model.

The script keeps the raw profiles as well as robust metrics. In particular,
``g_rad`` uses the same 2.0577 cgs floor as the release checkpoint rather than
dividing by values close to zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench import environment as _environment  # noqa: F401,E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
DEFAULT_INDICES = REPO_ROOT / "results" / "reconstruction_metrics.json"
DEFAULT_PREDICTION = (
    REPO_ROOT
    / "artifacts"
    / "reduced_state_emulator"
    / "physical"
    / "predicted_physical_ensemble.npz"
)
DEFAULT_OUT = REPO_ROOT / "results" / "field_consistency"
LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)
FIELDS = (
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)
G_RAD_SCALE = 2.0577175465785027


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker(payload):
    from payne_zero_atmosphere.warm_start import emulator_warm_start_model
    from reduced_state.reconstruct import (
        ReducedAtmosphere,
        ReconstructionConvergenceError,
        reconstruct_full_atmosphere,
    )

    index, labels, truth_profile, reduced_mass, reduced_temperature = payload
    label_dict = {field: float(labels[field]) for field in LABEL_FIELDS}
    truth = np.asarray(truth_profile, dtype=np.float64)
    six, _deck = emulator_warm_start_model(device="cpu", **label_dict)

    def reconstruct(column_mass, temperature):
        try:
            result = reconstruct_full_atmosphere(
                ReducedAtmosphere(
                    column_mass=np.asarray(column_mass, dtype=np.float64),
                    temperature=np.asarray(temperature, dtype=np.float64),
                    labels=label_dict,
                ),
                n_synchronizations=None,
                max_synchronizations=8,
                pressure_tolerance_dex=1.0e-3,
            )
        except ReconstructionConvergenceError as error:
            return {"error": str(error)}
        return {
            "fields": {
                field: np.asarray(getattr(result.atmosphere, field), dtype=np.float64)
                for field in FIELDS
            },
            "pressure_change_dex_by_pass": result.pressure_change_dex_by_pass,
            "n_synchronizations": result.n_synchronizations,
            "n_evaluations": result.n_evaluations,
            "n_pressure_updates": result.n_pressure_updates,
            "synchronized": result.synchronized,
        }

    return {
        "index": int(index),
        "truth": {field: truth[:, i + 2] for i, field in enumerate(FIELDS)},
        "direct_six_field": {
            field: np.asarray(getattr(six, field), dtype=np.float64) for field in FIELDS
        },
        "reconstruct_six_mT": reconstruct(six.column_mass, six.temperature),
        "reconstruct_truth_mT": reconstruct(truth[:, 0], truth[:, 1]),
        "reconstruct_reduced_mT": reconstruct(reduced_mass, reduced_temperature),
    }


def _metric(predicted: np.ndarray, truth: np.ndarray, field: str) -> dict:
    difference = np.abs(np.asarray(predicted) - np.asarray(truth))
    if field == "radiative_acceleration":
        error = difference / np.maximum(np.abs(truth), G_RAD_SCALE)
        name = "normalized_error"
    else:
        error = difference / np.maximum(np.abs(truth), 1.0e-300)
        name = "relative_error"
    log_error = None
    if field != "radiative_acceleration":
        log_error = np.abs(
            np.log10(np.maximum(np.asarray(predicted), 1.0e-300))
            - np.log10(np.maximum(np.asarray(truth), 1.0e-300))
        )
    summary = {
        name: {
            "median": float(np.median(error)),
            "p90": float(np.percentile(error, 90.0)),
            "p95": float(np.percentile(error, 95.0)),
            "max": float(np.max(error)),
        }
    }
    if log_error is not None:
        summary["log10_error"] = {
            "median": float(np.median(log_error)),
            "p90": float(np.percentile(log_error, 90.0)),
            "p95": float(np.percentile(log_error, 95.0)),
            "max": float(np.max(log_error)),
        }
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--indices-from", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="only evaluate the first N stars from the sealed/development manifest",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    manifest_indices = np.asarray(
        json.loads(args.indices_from.read_text())["star_indices"], dtype=np.int64
    )
    if args.count is not None and args.count < 1:
        raise ValueError("--count must be positive")
    indices = manifest_indices if args.count is None else manifest_indices[: args.count]
    with np.load(args.corpus, allow_pickle=False) as data:
        labels_json = [json.loads(str(value)) for value in data["labels_json"][indices]]
        profiles = np.asarray(data["atmosphere_profiles"][indices], dtype=np.float64)
    with np.load(args.prediction, allow_pickle=False) as data:
        prediction_indices = np.asarray(data["star_indices"], dtype=np.int64)
        prediction_rows = {int(index): row for row, index in enumerate(prediction_indices)}
        missing = [int(index) for index in indices if int(index) not in prediction_rows]
        if missing:
            raise ValueError(
                f"prediction does not contain {len(missing)} requested evaluation stars"
            )
        selected_rows = [prediction_rows[int(index)] for index in indices]
        reduced_mass = np.asarray(data["column_mass"], dtype=np.float64)[selected_rows]
        reduced_temperature = np.asarray(data["temperature"], dtype=np.float64)[
            selected_rows
        ]

    payloads = [
        (int(index), labels_json[row], profiles[row], reduced_mass[row], reduced_temperature[row])
        for row, index in enumerate(indices)
    ]
    if args.workers <= 1:
        rows = [_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_worker, payloads))

    arms = (
        "direct_six_field",
        "reconstruct_six_mT",
        "reconstruct_truth_mT",
        "reconstruct_reduced_mT",
    )
    raw: dict[str, np.ndarray] = {}
    metrics: dict[str, dict] = {}
    failures: dict[str, list[dict]] = {arm: [] for arm in arms if arm != "direct_six_field"}
    for arm in arms:
        metrics[arm] = {}
        for row in rows:
            if arm != "direct_six_field" and "error" in row[arm]:
                failures[arm].append(
                    {
                        "star_index": row["index"],
                        "error": row[arm]["error"],
                    }
                )
        for field in FIELDS:
            values = np.full((len(rows), 80), np.nan, dtype=np.float64)
            truth = np.stack([row["truth"][field] for row in rows])
            for row_index, row in enumerate(rows):
                if arm == "direct_six_field":
                    values[row_index] = row[arm][field]
                elif "error" not in row[arm]:
                    values[row_index] = row[arm]["fields"][field]
            raw[f"{arm}__{field}"] = values
            valid = np.all(np.isfinite(values), axis=1)
            if not np.any(valid):
                metrics[arm][field] = {"valid_star_count": 0}
            else:
                metrics[arm][field] = {
                    "valid_star_count": int(valid.sum()),
                    **_metric(values[valid], truth[valid], field),
                }
    metrics["failures"] = failures
    metrics["star_count"] = len(rows)
    metrics["g_rad_scale"] = G_RAD_SCALE
    metrics["synchronization"] = {
        arm: [
            {
                "star_index": row["index"],
                "n_synchronizations": row[arm].get("n_synchronizations"),
                "n_evaluations": row[arm].get("n_evaluations"),
                "n_pressure_updates": row[arm].get("n_pressure_updates"),
                "pressure_change_dex_by_pass": row[arm].get(
                    "pressure_change_dex_by_pass"
                ),
                "synchronized": row[arm].get("synchronized", False),
            }
            for row in rows
        ]
        for arm in arms
        if arm != "direct_six_field"
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), star_indices=indices, **raw)
    prediction_provenance_path = args.prediction.with_suffix(".json")
    prediction_provenance = {}
    if prediction_provenance_path.exists():
        prediction_provenance = json.loads(prediction_provenance_path.read_text())
    provenance = {
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
        "direct_six_field_checkpoint": str(
            REPO_ROOT
            / "source_data_files"
            / "atmosphere_emulator"
            / "five_label"
            / "checkpoint.pt"
        ),
        "direct_six_field_checkpoint_sha256": _sha256(
            REPO_ROOT
            / "source_data_files"
            / "atmosphere_emulator"
            / "five_label"
            / "checkpoint.pt"
        ),
        "evaluation_count": int(len(indices)),
        "metrics": metrics,
    }
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.out.with_suffix('.json')}", flush=True)
    print(f"wrote {args.out.with_suffix('.npz')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
