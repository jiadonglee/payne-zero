"""Recover validated development rows after an interrupted parallel replay.

The recovery is deliberately conservative.  A row is imported only when its
convergence outcome and completed iteration count exactly match the frozen
historical development result.  Timeout, error, duplicate, or incomplete rows
remain in the diagnostic namespace and are recomputed in the formal run.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from experiments.analytic_initializer import (
    run_paper_grey_convective_campaign as campaign,
)


def _read_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("shard_*.jsonl")):
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def recover(
    root: Path,
    diagnostic_result: Path,
    diagnostic_run: Path,
    isolated_results: tuple[tuple[int, Path], ...] = (),
) -> dict[str, Any]:
    root = root.resolve()
    diagnostic_result = diagnostic_result.resolve()
    diagnostic_run = diagnostic_run.resolve()
    indices, paths = campaign._write_seeds(root, "development")
    destination = paths["shards"] / "shard_00.jsonl"
    if destination.exists() or paths["profiles"].exists() or paths["products"].exists():
        raise SystemExit(
            "formal development rows or products already exist; recovery refused"
        )

    historical = json.loads(
        (root / campaign.HISTORICAL_DEVELOPMENT).read_text(encoding="utf-8")
    )
    expected = {
        int(row["corpus_index"]): (
            bool(row["converged"]),
            row.get("iterations_completed"),
        )
        for row in historical["records"]
    }
    rows = _read_rows(diagnostic_run / "record_shards")
    with np.load(paths["seeds"], allow_pickle=False) as seeds:
        labels = np.asarray(seeds["labels"], dtype=np.float64)
    position_by_index = {
        int(index): position for position, index in enumerate(indices)
    }
    isolated_sources: dict[int, str] = {}
    for index, log_path in isolated_results:
        index = int(index)
        if index not in position_by_index:
            raise SystemExit(f"isolated result index {index} is not in development")
        lines = [
            line
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.startswith("{") and line.endswith("}")
        ]
        if not lines:
            raise SystemExit(f"isolated result log has no JSON row: {log_path}")
        result = json.loads(lines[-1])
        position = position_by_index[index]
        label_values = labels[position]
        rows.append(
            {
                "sample": "development",
                "corpus_index": index,
                "position": position,
                "slug": campaign._solver_slug(label_values),
                "arm": campaign.ARM,
                "iterations_per_trial": campaign.ITERATIONS,
                **{
                    name: float(value)
                    for name, value in zip(campaign.LABEL_FIELDS, label_values)
                },
                **result,
            }
        )
        isolated_sources[index] = str(log_path)

    attempts_by_index: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        index = int(row["corpus_index"])
        attempts_by_index.setdefault(index, []).append(row)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for position, index_value in enumerate(indices):
        index = int(index_value)
        attempts = attempts_by_index.get(index, [])
        if not attempts:
            continue
        matches = [
            row
            for row in attempts
            if (bool(row["converged"]), row.get("iterations_completed"))
            == expected[index]
        ]
        if len(matches) > 1:
            raise SystemExit(f"multiple exact diagnostic rows for index {index}")
        if not matches:
            rejected.append(
                {
                    "corpus_index": index,
                    "expected": list(expected[index]),
                    "attempts": [
                        {
                            "observed": [
                                bool(row["converged"]),
                                row.get("iterations_completed"),
                            ],
                            "solver_outcome": row.get("solver_outcome"),
                        }
                        for row in attempts
                    ],
                }
            )
            continue
        row = matches[0]
        slug = str(row["slug"])
        imported = dict(row)
        imported["position"] = position
        if bool(row["converged"]):
            source_profile = (
                diagnostic_run / "profiles" / campaign.ARM / f"{slug}.npz"
            )
            source_product = (
                diagnostic_run / "products" / campaign.ARM / f"{slug}.npz"
            )
            if not source_profile.is_file() or not source_product.is_file():
                raise SystemExit(
                    f"validated row {index} lacks its profile or product"
                )
            paths["profiles"].mkdir(parents=True, exist_ok=True)
            paths["products"].mkdir(parents=True, exist_ok=True)
            target_profile = paths["profiles"] / source_profile.name
            target_product = paths["products"] / source_product.name
            shutil.copy2(source_profile, target_profile)
            shutil.copy2(source_product, target_product)
            imported["profile_path"] = str(target_profile)
            imported["product_path"] = str(target_product)
        else:
            imported["profile_path"] = None
            imported["product_path"] = None
        accepted.append(imported)

    paths["shards"].mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted),
        encoding="utf-8",
    )
    diagnostic_shards = sorted(
        (diagnostic_run / "record_shards").glob("shard_*.jsonl")
    )
    record = {
        "campaign": campaign.CAMPAIGN,
        "sample": "development",
        "decision": "IMPORT_EXACT_MATCHING_PREFIX_ONLY",
        "diagnostic_result": str(diagnostic_result),
        "diagnostic_run": str(diagnostic_run),
        "requested_star_count": int(indices.size),
        "diagnostic_row_count": len(rows),
        "imported_row_count": len(accepted),
        "rejected_rows": rejected,
        "remaining_row_count": int(indices.size - len(accepted)),
        "destination": str(destination),
        "destination_sha256": campaign.sha256(destination),
        "diagnostic_shard_sha256": {
            str(path): campaign.sha256(path) for path in diagnostic_shards
        },
        "isolated_result_logs": isolated_sources,
        "isolated_result_sha256": {
            str(path): campaign.sha256(path)
            for _, path in isolated_results
        },
        "recovery_source_sha256": campaign.sha256(Path(__file__).resolve()),
    }
    output = paths["result"] / "recovery_import.json"
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--diagnostic-result", type=Path, required=True)
    parser.add_argument("--diagnostic-run", type=Path, required=True)
    parser.add_argument(
        "--isolated-result",
        action="append",
        default=[],
        metavar="INDEX=LOG",
        help="validated isolated result to consider in addition to shard rows",
    )
    args = parser.parse_args()
    isolated_results = []
    for value in args.isolated_result:
        index, separator, path = value.partition("=")
        if not separator:
            raise SystemExit("--isolated-result must have the form INDEX=LOG")
        isolated_results.append((int(index), Path(path).resolve()))
    result = recover(
        args.root,
        args.diagnostic_result,
        args.diagnostic_run,
        tuple(isolated_results),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
