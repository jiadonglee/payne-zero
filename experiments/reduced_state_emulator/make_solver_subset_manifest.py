"""Materialize the preselected real-solver subset from a sealed audit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    source = json.loads(args.source.read_text())
    selection = source.get("solver_selection")
    if not isinstance(selection, dict) or "star_indices" not in selection:
        raise ValueError("source manifest has no solver_selection.star_indices")
    indices = [int(value) for value in selection["star_indices"]]
    if len(indices) != len(set(indices)):
        raise ValueError("solver subset contains duplicate star indices")
    payload = {
        "format": "sealed_solver_subset_v1",
        "source_manifest": str(args.source),
        "source_manifest_sha256": _sha256(args.source),
        "source_corpus": source.get("corpus"),
        "source_corpus_sha256": source.get("corpus_sha256"),
        "selection": {
            key: value for key, value in selection.items() if key != "star_indices"
        },
        "star_indices": indices,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out} ({len(indices)} stars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
