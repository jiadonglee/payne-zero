"""Identify the checkpoint that produced each stored M-giant emulator seed.

The dual-path products record the reconstructed start atmosphere as
``iter_0000.npz`` but not the checkpoint that predicted it.  Physical
reconstruction holds ``(m, T)`` fixed, so the stored column mass and
temperature equal the coordinate-wise median of the three seed predictions.
This script repeats that prediction with the v3 and v4 M-giant checkpoints and
reports the maximum difference from every stored seed.  No solver is run.

    PYTHONPATH=. .venv/bin/python -m \\
        experiments.reduced_state_emulator.m_star_dual_path_checkpoint_provenance
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from reduced_state.emulator import load_physical_checkpoint, predict_physical_state

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260831, 20260901, 20260902)
VERSIONS = ("v3", "v4")
OUTPUT = (
    REPO_ROOT / "results" / "m_star_dual_path_checkpoint_provenance_20260923"
    / "provenance.json"
)
# (sample, product root, emulator arm directory, table listing its node ids)
SAMPLES = (
    (
        "stop_rule_twelve",
        "results/m_star_stop_rule_v1/validate",
        "emulator",
        "results/m_star_stop_rule_v1/validation_table.json",
    ),
    (
        "resolve_undetermined",
        "results/m_star_resolve_undetermined_v1/points",
        "emulator60",
        "results/m_star_stop_rule_v1/validation_table.json",
    ),
    (
        "sixty_budget_six",
        "results/m_star_sixty_budget_v1/points",
        "emulator",
        "results/m_star_sixty_budget_v1/merged_table.json",
    ),
)


def _median_prediction(models, labels_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    masses, temperatures = [], []
    for model, standardization, _meta in models:
        mass, temperature = predict_physical_state(
            model, standardization, labels_row.reshape(1, -1)
        )
        masses.append(np.asarray(mass)[0])
        temperatures.append(np.asarray(temperature)[0])
    return (
        np.median(np.stack(masses), axis=0),
        np.median(np.stack(temperatures), axis=0),
    )


def main() -> None:
    models = {
        version: [
            load_physical_checkpoint(
                REPO_ROOT / "artifacts" / f"m_star_emulator_mgiant_{version}"
                / f"checkpoint_mstar_seed{seed}.pt"
            )
            for seed in SEEDS
        ]
        for version in VERSIONS
    }
    records = []
    for sample, root, arm, table in SAMPLES:
        rows = json.loads((REPO_ROOT / table).read_text())["rows"]
        for row in rows:
            path = REPO_ROOT / root / row["node_id"] / arm / "iter_0000.npz"
            if not path.is_file():
                continue
            with np.load(path) as stored:
                stored_mass = np.asarray(stored["column_mass"], dtype=np.float64)
                stored_temperature = np.asarray(stored["temperature"], dtype=np.float64)
            labels_row = np.asarray(
                [row["teff_K"], row["logg"], row["metallicity"], 0.0, 2.0],
                dtype=np.float64,
            )
            differences = {}
            for version in VERSIONS:
                mass, temperature = _median_prediction(models[version], labels_row)
                differences[version] = {
                    "max_relative_temperature": float(
                        np.max(np.abs(temperature - stored_temperature) / stored_temperature)
                    ),
                    "max_log10_column_mass_dex": float(
                        np.max(np.abs(np.log10(mass) - np.log10(stored_mass)))
                    ),
                }
            best = min(
                VERSIONS,
                key=lambda v: differences[v]["max_relative_temperature"]
                + differences[v]["max_log10_column_mass_dex"],
            )
            records.append({
                "sample": sample,
                "node_id": row["node_id"],
                "product": str(path.relative_to(REPO_ROOT)),
                "differences": differences,
                "matching_checkpoint": best,
            })
    matches = sorted({record["matching_checkpoint"] for record in records})
    payload = {
        "campaign": "m_star_dual_path_checkpoint_provenance_20260923",
        "method": (
            "coordinate-wise median of three seed predictions compared with the "
            "stored reconstructed start; no solver run"
        ),
        "seed_count": len(records),
        "unique_targets": len({record["node_id"] for record in records}),
        "matching_checkpoints": matches,
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}: {len(records)} seeds, "
          f"{payload['unique_targets']} targets, matching {matches}")


if __name__ == "__main__":
    main()
