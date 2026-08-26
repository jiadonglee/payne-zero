"""Record per-star pressure-synchronization diagnostics for a prediction."""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from reduced_state.emulator import load_corpus
from reduced_state.reconstruct import (
    ReducedAtmosphere,
    ReconstructionConvergenceError,
    reconstruct_full_atmosphere,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
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


def _label_subset(row: np.ndarray) -> dict[str, float]:
    return {field: float(value) for field, value in zip(LABEL_FIELDS, row)}


def _diagnose_worker(payload: tuple[int, np.ndarray, np.ndarray, dict, int]) -> dict:
    position, column_mass, temperature, labels, max_synchronizations = payload
    try:
        result = reconstruct_full_atmosphere(
            ReducedAtmosphere(
                column_mass=column_mass,
                temperature=temperature,
                labels=labels,
            ),
            n_synchronizations=None,
            max_synchronizations=max_synchronizations,
            pressure_tolerance_dex=1.0e-3,
        )
    except ReconstructionConvergenceError as exc:
        return {
            "position": int(position),
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pressure_change_dex_by_pass": list(
                exc.pressure_change_dex_by_pass
            ),
        }
    except Exception as exc:  # record unexpected per-star failures too
        return {
            "position": int(position),
            "ok": False,
            "error_type": type(exc).__name__,
            "error": repr(exc),
            "pressure_change_dex_by_pass": [],
        }
    return {
        "position": int(position),
        "ok": bool(result.synchronized),
        "error_type": None,
        "error": None,
        "n_evaluations": int(result.n_evaluations),
        "pressure_change_dex_by_pass": [
            float(value) for value in result.pressure_change_dex_by_pass
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--held-out-from", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-synchronizations", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.max_synchronizations < 1:
        raise SystemExit("--workers and --max-synchronizations must be positive")

    held_out = np.asarray(
        json.loads(args.held_out_from.read_text())["star_indices"], dtype=np.int64
    )
    with np.load(args.prediction, allow_pickle=False) as prediction:
        if not np.array_equal(prediction["star_indices"], held_out):
            raise SystemExit("prediction artifact and held-out manifest differ")
        column_mass = np.asarray(prediction["column_mass"], dtype=np.float64)
        temperature = np.asarray(prediction["temperature"], dtype=np.float64)
        labels = np.asarray(prediction["labels"], dtype=np.float64)
    corpus_labels = np.asarray(load_corpus(args.corpus)["labels"][held_out], dtype=np.float64)
    if not np.array_equal(labels, corpus_labels):
        raise SystemExit("prediction labels and corpus labels differ")

    payloads = [
        (
            position,
            column_mass[position],
            temperature[position],
            _label_subset(labels[position]),
            int(args.max_synchronizations),
        )
        for position in range(len(held_out))
    ]
    if args.workers == 1:
        rows = [_diagnose_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_diagnose_worker, payloads))
    for row, star_index in zip(rows, held_out):
        row["star_index"] = int(star_index)
    rows.sort(key=lambda row: row["position"])
    payload = {
        "prediction": str(args.prediction),
        "prediction_sha256": _sha256(args.prediction),
        "held_out_manifest": str(args.held_out_from),
        "held_out_manifest_sha256": _sha256(args.held_out_from),
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "max_synchronizations": int(args.max_synchronizations),
        "pressure_tolerance_dex": 1.0e-3,
        "star_count": int(len(rows)),
        "synchronized_count": int(sum(row["ok"] for row in rows)),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {args.out} ({payload['synchronized_count']}/{payload['star_count']} "
        "synchronized)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
