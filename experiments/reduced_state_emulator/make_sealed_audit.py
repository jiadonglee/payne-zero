"""Create the untouched 200-star audit manifest for the physical emulator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reduced_state.emulator import load_corpus
from reduced_state.sampling import sample_star_indices

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
DEFAULT_DEVELOPMENT = REPO_ROOT / "results" / "reconstruction_metrics.json"
DEFAULT_OUT = REPO_ROOT / "results" / "sealed_audit_20260808.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--development-from", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="additional manifests whose star indices must stay out of the new audit",
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--solver-count", type=int, default=60)
    parser.add_argument("--solver-seed", type=int, default=20260809)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite sealed manifest: {args.out}")

    development_payload = json.loads(args.development_from.read_text())
    development = set(int(value) for value in development_payload["star_indices"])
    excluded_manifests = []
    excluded = set(development)
    for manifest in args.exclude_manifest:
        payload = json.loads(manifest.read_text())
        values = payload.get("star_indices")
        if not isinstance(values, list):
            raise ValueError(f"{manifest} has no star_indices list")
        excluded.update(int(value) for value in values)
        excluded_manifests.append(
            {
                "path": str(manifest),
                "sha256": _sha256(manifest),
                "star_count": len(values),
            }
        )
    corpus = load_corpus(args.corpus)
    candidates = np.array(
        [
            index
            for index, role in enumerate(corpus["roles"])
            if role == "validation" and index not in excluded
        ],
        dtype=np.int64,
    )
    candidate_labels = [corpus["labels_json"][index] for index in candidates]
    local_audit = sample_star_indices(
        candidate_labels, args.count, args.seed, hard_fraction=1.0 / 3.0
    )
    audit_indices = candidates[local_audit]
    local_solver = sample_star_indices(
        [corpus["labels_json"][index] for index in audit_indices],
        args.solver_count,
        args.solver_seed,
        hard_fraction=1.0 / 3.0,
    )
    solver_indices = audit_indices[local_solver]
    payload = {
        "format": "payne_zero_reduced_state_sealed_audit_v1",
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "development_manifest": str(args.development_from),
        "development_indices": sorted(development),
        "excluded_manifests": excluded_manifests,
        "selection": {
            "role": "validation",
            "candidate_count": int(len(candidates)),
            "count": int(args.count),
            "seed": int(args.seed),
            "hard_fraction": 1.0 / 3.0,
        },
        "star_indices": [int(index) for index in audit_indices],
        "solver_selection": {
            "count": int(args.solver_count),
            "seed": int(args.solver_seed),
            "hard_fraction": 1.0 / 3.0,
            "star_indices": [int(index) for index in solver_indices],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"sealed {args.count} audit stars and {args.solver_count} solver stars in {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
