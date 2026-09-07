"""Assemble the safe-zone cool-star truth corpus from the certified inventory.

Train rows are the in-boundary admitted products of the certified cold-star
inventory, minus six fixed nodes carved out as cool validation.  The opened
validation rows of the v1r1 policy60 corpus are imported unchanged, matching
the v1r2 corpus convention.  Neither stored corpus is modified.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .m_star_bootstrap_v1 import FLUX_METRICS, _load_mt, _sha256, _write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_cool_corpus_safezone_v1"
DEFAULT_INVENTORY = (
    REPO_ROOT / "results" / "m_star_cold_library_inventory_v1" / "inventory.json"
)
DEFAULT_VALIDATION_CORPUS = (
    REPO_ROOT
    / "results"
    / "m_star_emulator_v1r1_policy60_garching_20260831"
    / "policy60"
    / "cool_truth_corpus.npz"
)
DEFAULT_FLUX_GATE = (
    REPO_ROOT / "results" / "m_star_iteration_tomography_v1" / "flux_gate.json"
)
DEFAULT_OUT = REPO_ROOT / "results" / CAMPAIGN

# Cool-edge validation carve-out, fixed before training: one node per
# (stellar class, metallicity) boundary corner of the safe zone.
CARVED_VALIDATION_NODES = (
    (4.5, -1.0, 3800.0),
    (4.5, 0.0, 3400.0),
    (4.5, 0.5, 3600.0),
    (1.5, -1.0, 3500.0),
    (2.5, 0.0, 3500.0),
    (1.5, 0.5, 3500.0),
)


def _hash_payload(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument(
        "--validation-corpus", type=Path, default=DEFAULT_VALIDATION_CORPUS
    )
    parser.add_argument("--flux-gate", type=Path, default=DEFAULT_FLUX_GATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    inventory = json.loads(args.inventory.read_text())
    rows = inventory["in_boundary_rows"]
    by_node = {
        (float(row["logg"]), float(row["metallicity"]), float(row["teff_K"])): row
        for row in rows
    }
    missing = [node for node in CARVED_VALIDATION_NODES if node not in by_node]
    if missing:
        raise SystemExit(f"FAIL_STOP: carved validation nodes absent: {missing}")

    carved = set(CARVED_VALIDATION_NODES)
    train_rows = [row for node, row in by_node.items() if node not in carved]
    validation_rows = [by_node[node] for node in CARVED_VALIDATION_NODES]

    def row_values(row: dict[str, Any]) -> tuple[list[float], np.ndarray, np.ndarray]:
        mass, temperature = _load_mt(REPO_ROOT / row["canonical_product"])
        labels = [
            float(row["teff_K"]),
            float(row["logg"]),
            float(row["metallicity"]),
            0.0,
            float(row["vmic_km_s"]),
        ]
        return labels, mass, temperature

    def _flux_row(row: dict[str, Any]) -> list[float]:
        metrics = row.get("primary_flux_metrics") or {}
        values = [metrics.get(name) for name in FLUX_METRICS]
        if any(value is None for value in values):
            raise SystemExit(
                f"FAIL_STOP: missing flux metrics for {row['canonical_product']}"
            )
        return values

    train_labels = []
    train_mass = []
    train_temperature = []
    train_flux = []
    carved_labels = []
    carved_mass = []
    carved_temperature = []
    carved_flux = []
    for row in train_rows:
        labels, mass, temperature = row_values(row)
        train_labels.append(labels)
        train_mass.append(mass)
        train_temperature.append(temperature)
        train_flux.append(_flux_row(row))
    for row in validation_rows:
        labels, mass, temperature = row_values(row)
        carved_labels.append(labels)
        carved_mass.append(mass)
        carved_temperature.append(temperature)
        carved_flux.append(_flux_row(row))

    with np.load(args.validation_corpus, allow_pickle=False) as parent:
        parent_roles = np.asarray(parent["roles"]).astype(str)
        if "sealed" in set(parent_roles):
            raise ValueError("validation parent corpus contains sealed rows")
        validation_index = np.flatnonzero(parent_roles == "validation")
        if not len(validation_index):
            raise ValueError("validation parent corpus has no validation rows")
        imported_labels = np.asarray(
            parent["labels"][validation_index], dtype=np.float64
        )
        imported_mass = np.asarray(
            parent["column_mass"][validation_index], dtype=np.float64
        )
        imported_temperature = np.asarray(
            parent["temperature"][validation_index], dtype=np.float64
        )
        imported_track_ids = (
            np.asarray(parent["track_ids"][validation_index]).astype(str).tolist()
        )
        imported_node_ids = (
            np.asarray(parent["node_ids"][validation_index]).astype(str).tolist()
        )
        imported_products = (
            np.asarray(parent["source_product_paths"][validation_index])
            .astype(str)
            .tolist()
        )
        imported_flux = np.asarray(
            parent["flux_metrics"][validation_index], dtype=np.float64
        )

    train_keys = {
        (float(row["teff_K"]), float(row["logg"]), float(row["metallicity"]))
        for row in train_rows
    }
    overlap = [
        (float(row[0]), float(row[1]), float(row[2]))
        for row in imported_labels
        if (float(row[0]), float(row[1]), float(row[2])) in train_keys
    ]
    if overlap:
        raise SystemExit(f"FAIL_STOP: validation nodes overlap train rows: {overlap}")

    labels = np.vstack(
        [
            np.asarray(train_labels + carved_labels, dtype=np.float64),
            imported_labels,
        ]
    )
    column_mass = np.vstack(
        [
            np.asarray(train_mass + carved_mass, dtype=np.float64),
            imported_mass,
        ]
    )
    temperature = np.vstack(
        [
            np.asarray(train_temperature + carved_temperature, dtype=np.float64),
            imported_temperature,
        ]
    )
    flux_metrics = np.vstack(
        [
            np.asarray(train_flux + carved_flux, dtype=np.float64),
            imported_flux,
        ]
    )
    def _track_id(row: dict[str, Any]) -> str:
        return (
            f"g{row['logg']:+.2f}_m{row['metallicity']:+.2f}"
            f"_a+0.00_c+0.00_x{row['vmic_km_s']:.2f}"
        )

    pool_rows = train_rows + validation_rows
    pool_track_ids = [_track_id(row) for row in pool_rows]
    pool_node_ids = [
        f"{track}_t{int(row['teff_K']):04d}"
        for track, row in zip(pool_track_ids, pool_rows)
    ]
    roles = np.asarray(
        ["train"] * len(train_rows)
        + ["validation"] * (len(validation_rows) + len(imported_node_ids)),
        dtype="U16",
    )
    track_ids = np.asarray(
        pool_track_ids + imported_track_ids,
        dtype="U96",
    )
    node_ids = np.asarray(
        pool_node_ids + imported_node_ids,
        dtype="U128",
    )
    source_products = np.asarray(
        [row["canonical_product"] for row in pool_rows] + imported_products,
        dtype="U512",
    )
    source_campaigns = np.asarray(
        [row["canonical_campaign"] for row in pool_rows]
        + ["m_star_emulator_v1r1_policy60"] * len(imported_node_ids),
        dtype="U64",
    )

    flux_gate = json.loads(args.flux_gate.read_text())
    payload = {
        "inventory_sha256": _sha256(args.inventory),
        "validation_import_sha256": _sha256(args.validation_corpus),
        "carved_validation_nodes": [
            {"logg": logg, "metallicity": metallicity, "teff_K": teff}
            for logg, metallicity, teff in CARVED_VALIDATION_NODES
        ],
    }
    output_path = args.out / "cool_truth_corpus.npz"
    args.out.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            labels=labels,
            label_fields=np.asarray(
                [
                    "effective_temperature",
                    "log_surface_gravity",
                    "metallicity",
                    "alpha_enhancement",
                    "microturbulence_km_s",
                ],
                dtype="U40",
            ),
            column_mass=column_mass,
            temperature=temperature,
            roles=roles,
            track_ids=track_ids,
            node_ids=node_ids,
            source_product_paths=source_products,
            source_campaigns=source_campaigns,
            flux_metric_fields=np.asarray(FLUX_METRICS, dtype="U64"),
            flux_metrics=flux_metrics,
            protocol_hash=np.asarray([_hash_payload(payload)], dtype="U64"),
            flux_gate_hash=np.asarray([flux_gate["gate_hash"]], dtype="U64"),
        )
    temporary.replace(output_path)

    train_labels_array = labels[roles == "train"]
    validation_labels_array = labels[roles == "validation"]
    summary = {
        "campaign": CAMPAIGN,
        "status": "complete",
        "path": str(output_path),
        "sha256": _sha256(output_path),
        "row_count": int(len(labels)),
        "role_counts": {
            "train": int(np.sum(roles == "train")),
            "validation": int(np.sum(roles == "validation")),
        },
        "role_class_counts": {
            "train": {
                "giant": int(np.sum(train_labels_array[:, 1] < 3.5)),
                "dwarf": int(np.sum(train_labels_array[:, 1] >= 3.5)),
            },
            "validation": {
                "giant": int(np.sum(validation_labels_array[:, 1] < 3.5)),
                "dwarf": int(np.sum(validation_labels_array[:, 1] >= 3.5)),
            },
        },
        "carved_validation_node_ids": pool_node_ids[
            len(train_rows) : len(train_rows) + len(validation_rows)
        ],
        "validation_import": str(args.validation_corpus),
        "validation_import_sha256": _sha256(args.validation_corpus),
        "inventory": str(args.inventory),
        "inventory_sha256": _sha256(args.inventory),
        "flux_gate_hash": flux_gate["gate_hash"],
        "sealed_rows_included": False,
        "marcs_is_training_target": False,
    }
    _write_json(args.out / "cool_truth_corpus.json", summary)
    (args.out / "CORPUS_READY").touch()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
