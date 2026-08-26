"""Create the opened calibration set and the next sealed initializer holdout.

Both manifests are selected before any new model is trained.  The calibration
set contains disjoint ordinary, solver-tail, and label-edge slices.  The sealed
holdout uses the same proportions and must not be predicted until the complete
initializer policy has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reduced_state.emulator import load_corpus


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
DEFAULT_EXCLUSIONS = (
    REPO_ROOT / "results" / "reconstruction_metrics.json",
    REPO_ROOT / "results" / "sealed_solver_subset_20260808.json",
    REPO_ROOT / "results" / "sealed_audit_20260808.json",
    REPO_ROOT / "results" / "sealed_audit_20260811.json",
)
DEFAULT_CALIBRATION = (
    REPO_ROOT / "results" / "initializer_calibration_20260812.json"
)
DEFAULT_HOLDOUT = (
    REPO_ROOT / "results" / "sealed_initializer_holdout_20260812.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_indices(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    values = payload.get("star_indices")
    if not isinstance(values, list):
        raise ValueError(f"{path} does not contain a star_indices list")
    return np.asarray(values, dtype=np.int64)


def _pick(
    rng: np.random.Generator, pool: np.ndarray, count: int, name: str
) -> np.ndarray:
    if count < 0:
        raise ValueError(f"{name} count must be non-negative")
    if pool.size < count:
        raise ValueError(f"{name} pool has {pool.size} rows, fewer than {count}")
    return np.asarray(rng.choice(pool, size=count, replace=False), dtype=np.int64)


def _stratified_pick(
    rng: np.random.Generator,
    category_pools: dict[str, np.ndarray],
    counts: dict[str, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    selected = {
        name: _pick(rng, category_pools[name], counts[name], name)
        for name in ("ordinary", "hard", "edge")
    }
    combined = np.concatenate(list(selected.values()))
    rng.shuffle(combined)
    return combined, selected


def _category_masks(
    labels: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, np.ndarray]:
    edge = np.any((labels <= lower) | (labels >= upper), axis=1)
    solver_tail = (
        (labels[:, 1] < 3.2)
        | (labels[:, 4] > 3.0)
        | (labels[:, 3] > 0.4)
    )
    return {
        "ordinary": ~edge & ~solver_tail,
        "hard": ~edge & solver_tail,
        "edge": edge,
    }


def _category_payload(selected: dict[str, np.ndarray]) -> dict:
    return {
        name: {
            "count": int(values.size),
            "star_indices": [int(value) for value in values],
        }
        for name, values in selected.items()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=None,
        help="manifest to keep out of both new sets; repeat as needed",
    )
    parser.add_argument("--calibration-out", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--holdout-out", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--edge-quantile", type=float, default=0.02)
    parser.add_argument("--calibration-seed", type=int, default=20260812)
    parser.add_argument("--holdout-seed", type=int, default=20260813)
    parser.add_argument("--split-seed", type=int, default=20260814)
    args = parser.parse_args(argv)

    if args.calibration_out.exists() or args.holdout_out.exists():
        existing = [
            str(path)
            for path in (args.calibration_out, args.holdout_out)
            if path.exists()
        ]
        raise SystemExit(f"refusing to overwrite existing manifests: {existing}")
    if not 0.0 < args.edge_quantile < 0.5:
        raise SystemExit("--edge-quantile must be between 0 and 0.5")

    exclusion_paths = tuple(args.exclude_manifest or DEFAULT_EXCLUSIONS)
    excluded: set[int] = set()
    exclusion_payload = []
    for path in exclusion_paths:
        values = _load_indices(path)
        excluded.update(int(value) for value in values)
        exclusion_payload.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "star_count": int(values.size),
            }
        )

    corpus = load_corpus(args.corpus)
    labels = np.asarray(corpus["labels"], dtype=np.float64)
    roles = np.asarray(corpus["roles"])
    train_labels = labels[roles == "train"]
    lower = np.quantile(train_labels, args.edge_quantile, axis=0)
    upper = np.quantile(train_labels, 1.0 - args.edge_quantile, axis=0)

    candidates = np.asarray(
        [
            index
            for index, role in enumerate(roles)
            if role == "validation" and index not in excluded
        ],
        dtype=np.int64,
    )
    masks = _category_masks(labels[candidates], lower, upper)
    pools = {name: candidates[mask] for name, mask in masks.items()}

    calibration_counts = {"ordinary": 200, "hard": 100, "edge": 100}
    calibration, calibration_categories = _stratified_pick(
        np.random.default_rng(args.calibration_seed), pools, calibration_counts
    )

    calibration_set = set(int(value) for value in calibration)
    remaining_pools = {
        name: np.asarray(
            [value for value in pool if int(value) not in calibration_set],
            dtype=np.int64,
        )
        for name, pool in pools.items()
    }
    holdout_counts = {"ordinary": 100, "hard": 50, "edge": 50}
    holdout, holdout_categories = _stratified_pick(
        np.random.default_rng(args.holdout_seed), remaining_pools, holdout_counts
    )

    if set(map(int, calibration)) & set(map(int, holdout)):
        raise RuntimeError("calibration and holdout selections overlap")
    if (set(map(int, calibration)) | set(map(int, holdout))) & excluded:
        raise RuntimeError("a new selection overlaps an excluded manifest")

    split_rng = np.random.default_rng(args.split_seed)
    gate_train_categories: dict[str, np.ndarray] = {}
    gate_validation_categories: dict[str, np.ndarray] = {}
    spectral_categories: dict[str, np.ndarray] = {}
    validation_counts = {"ordinary": 50, "hard": 25, "edge": 25}
    spectral_counts = {"ordinary": 30, "hard": 15, "edge": 15}
    for name, values in calibration_categories.items():
        shuffled = values.copy()
        split_rng.shuffle(shuffled)
        validation_count = validation_counts[name]
        gate_validation_categories[name] = shuffled[:validation_count]
        gate_train_categories[name] = shuffled[validation_count:]
        spectral_categories[name] = shuffled[: spectral_counts[name]]
    gate_train = np.concatenate(list(gate_train_categories.values()))
    gate_validation = np.concatenate(list(gate_validation_categories.values()))
    spectral = np.concatenate(list(spectral_categories.values()))
    split_rng.shuffle(gate_train)
    split_rng.shuffle(gate_validation)
    split_rng.shuffle(spectral)

    solver_rng = np.random.default_rng(args.split_seed + 1)
    solver_categories = {}
    solver_counts = {"ordinary": 30, "hard": 15, "edge": 15}
    for name, values in holdout_categories.items():
        solver_categories[name] = _pick(
            solver_rng, values, solver_counts[name], f"holdout solver {name}"
        )
    solver = np.concatenate(list(solver_categories.values()))
    solver_rng.shuffle(solver)

    common = {
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "excluded_manifests": exclusion_payload,
        "selection_role": "validation",
        "edge_definition": {
            "training_label_quantile": float(args.edge_quantile),
            "lower": lower.tolist(),
            "upper": upper.tolist(),
            "rule": "any label at or outside the train quantile bounds",
        },
        "hard_definition": (
            "not edge and (logg < 3.2 or vmic > 3.0 or alpha > 0.4)"
        ),
        "candidate_counts": {
            name: int(pool.size) for name, pool in pools.items()
        },
    }
    calibration_payload = {
        "format": "payne_zero_initializer_calibration_v1",
        "sealed": False,
        **common,
        "selection_seed": int(args.calibration_seed),
        "star_indices": [int(value) for value in calibration],
        "categories": _category_payload(calibration_categories),
        "gate_split": {
            "seed": int(args.split_seed),
            "train_star_indices": [int(value) for value in gate_train],
            "validation_star_indices": [int(value) for value in gate_validation],
            "train_categories": _category_payload(gate_train_categories),
            "validation_categories": _category_payload(gate_validation_categories),
        },
        "spectral_selection": {
            "selected_before_outcomes": True,
            "star_indices": [int(value) for value in spectral],
            "categories": _category_payload(spectral_categories),
        },
    }
    holdout_payload = {
        "format": "payne_zero_initializer_sealed_holdout_v1",
        "sealed": True,
        "opened": False,
        **common,
        "selection_seed": int(args.holdout_seed),
        "star_indices": [int(value) for value in holdout],
        "categories": _category_payload(holdout_categories),
        "solver_selection": {
            "selected_before_outcomes": True,
            "star_indices": [int(value) for value in solver],
            "categories": _category_payload(solver_categories),
        },
        "contract": (
            "Do not predict, inspect, solve, or synthesize these stars until the "
            "initializer and fallback gate are frozen."
        ),
    }

    args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
    args.holdout_out.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_out.write_text(json.dumps(calibration_payload, indent=2) + "\n")
    args.holdout_out.write_text(json.dumps(holdout_payload, indent=2) + "\n")
    print(
        f"wrote calibration {len(calibration)} and sealed holdout {len(holdout)}; "
        f"spectral/solver subsets={len(spectral)}/{len(solver)}",
        flush=True,
    )
    print(f"calibration: {args.calibration_out}", flush=True)
    print(f"holdout:     {args.holdout_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
