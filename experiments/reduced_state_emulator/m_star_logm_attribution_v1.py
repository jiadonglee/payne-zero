"""Attribute the v3 cool-validation column-mass error.

Loads the three M-giant checkpoints, predicts every cool validation row with
each seed and with the coordinate-wise median ensemble, and decomposes the
log-column-mass error by layer, by stellar parameters, and into ensemble bias
versus seed scatter.  Read-only analysis; no solver involvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from reduced_state.emulator import (
    load_physical_checkpoint,
    predict_physical_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "artifacts" / "m_star_emulator_mgiant_v3"
DEFAULT_CORPUS = (
    REPO_ROOT / "results" / "m_star_cool_corpus_mgiant_v1" / "cool_truth_corpus.npz"
)
DEFAULT_OUT = REPO_ROOT / "results" / "m_star_logm_attribution_v1"
SEEDS = (20260831, 20260901, 20260902)
TAU_GRID = 10.0 ** (-6.875 + 0.125 * np.arange(80))


def _predict_all(checkpoint_dir: Path, labels: np.ndarray):
    per_seed = []
    for seed in SEEDS:
        model, standardization, _meta = load_physical_checkpoint(
            checkpoint_dir / f"checkpoint_mstar_seed{seed}.pt"
        )
        mass, temperature = predict_physical_state(
            model, standardization, labels
        )
        per_seed.append((np.asarray(mass), np.asarray(temperature)))
    mass_stack = np.stack([m for m, _ in per_seed], axis=0)
    temperature_stack = np.stack([t for _, t in per_seed], axis=0)
    median_mass = np.median(mass_stack, axis=0)
    median_temperature = np.median(temperature_stack, axis=0)
    return per_seed, (median_mass, median_temperature)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    with np.load(args.corpus, allow_pickle=False) as data:
        roles = np.asarray(data["roles"]).astype(str)
        index = np.flatnonzero(roles == "validation")
        labels = np.asarray(data["labels"], dtype=np.float64)[index]
        truth_mass = np.asarray(data["column_mass"], dtype=np.float64)[index]
        node_ids = np.asarray(data["node_ids"]).astype(str)[index]

    per_seed, (median_mass, _median_temperature) = _predict_all(
        args.checkpoint_dir, labels
    )
    log_truth = np.log10(truth_mass)
    log_median = np.log10(median_mass)
    ensemble_error = log_median - log_truth
    seed_errors = np.stack(
        [np.log10(m) - log_truth for m, _t in per_seed], axis=0
    )
    seed_scatter = seed_errors.std(axis=0)
    seed_mean_error = seed_errors.mean(axis=0)
    bias_component = np.abs(seed_mean_error)
    scatter_component = 1.2533 * seed_scatter

    deep = TAU_GRID > 1.0
    star_rows = []
    for i, node in enumerate(node_ids):
        p95 = float(
            np.percentile(np.abs(ensemble_error[i]), 95.0)
        )
        star_rows.append(
            {
                "node_id": str(node),
                "teff_K": float(labels[i, 0]),
                "logg": float(labels[i, 1]),
                "metallicity": float(labels[i, 2]),
                "logm_p95_dex": p95,
                "logm_deep_p95_dex": float(
                    np.percentile(np.abs(ensemble_error[i, deep]), 95.0)
                ),
                "logm_surface_p95_dex": float(
                    np.percentile(np.abs(ensemble_error[i, ~deep]), 95.0)
                ),
                "seed_scatter_p95_dex": float(
                    np.percentile(seed_scatter[i], 95.0)
                ),
            }
        )
    star_rows.sort(key=lambda row: -row["logm_p95_dex"])

    layer_report = {
        "tau": TAU_GRID.tolist(),
        "ensemble_abs_error_p95": np.percentile(
            np.abs(ensemble_error), 95.0, axis=0
        ).tolist(),
        "ensemble_abs_error_median": np.percentile(
            np.abs(ensemble_error), 50.0, axis=0
        ).tolist(),
        "bias_component_median": np.percentile(
            bias_component, 50.0, axis=0
        ).tolist(),
        "scatter_component_median": np.percentile(
            scatter_component, 50.0, axis=0
        ).tolist(),
    }
    overall = {
        "logm_p95_dex": float(np.percentile(np.abs(ensemble_error), 95.0)),
        "logm_deep_p95_dex": float(
            np.percentile(np.abs(ensemble_error[:, deep]), 95.0)
        ),
        "logm_surface_p95_dex": float(
            np.percentile(np.abs(ensemble_error[:, ~deep]), 95.0)
        ),
        "bias_share_median_over_layers": float(
            np.median(
                bias_component
                / np.maximum(bias_component + scatter_component, 1e-12)
            )
        ),
        "temperature_p95": None,
    }

    temperature_truth = None
    for seed_index, (_m, t) in enumerate(per_seed):
        pass
    payload: dict[str, Any] = {
        "campaign": "m_star_logm_attribution_v1",
        "overall": overall,
        "layer_report": layer_report,
        "stars": star_rows,
        "seed_errors_layer_median": np.percentile(
            np.abs(seed_errors), 50.0, axis=0
        ).tolist(),
    }
    (args.out / "attribution.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(overall, indent=2))
    print("\nworst stars by logm p95:")
    for row in star_rows[:5]:
        print(
            f"  {row['node_id']:<36} T={row['teff_K']:.0f} logg={row['logg']:.2f} "
            f"m={row['metallicity']:+.1f} p95={row['logm_p95_dex']:.4f} dex "
            f"(deep {row['logm_deep_p95_dex']:.4f}, surface {row['logm_surface_p95_dex']:.4f}, "
            f"scatter {row['seed_scatter_p95_dex']:.4f})"
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.semilogx(
        layer_report["tau"],
        layer_report["ensemble_abs_error_p95"],
        label="ensemble error p95",
    )
    ax.semilogx(
        layer_report["tau"],
        layer_report["bias_component_median"],
        label="bias component (median)",
    )
    ax.semilogx(
        layer_report["tau"],
        layer_report["scatter_component_median"],
        label="seed scatter (median)",
    )
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Rosseland optical depth")
    ax.set_ylabel("log column-mass error (dex)")
    ax.set_title("error by layer")
    ax.legend(fontsize=8)

    ax = axes[1]
    for row in star_rows:
        color = "#d62728" if row["logg"] >= 3.5 else "#1f77b4"
        ax.scatter(row["teff_K"], row["logm_p95_dex"], color=color, s=45)
        ax.annotate(
            f"g{row['logg']:.1f} m{row['metallicity']:+.1f}",
            (row["teff_K"], row["logm_p95_dex"]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=6,
        )
    ax.axhline(7.7e-3, color="crimson", linewidth=1, linestyle=":")
    ax.text(3960, 7.9e-3, "gate 7.7e-3", fontsize=8, color="crimson", ha="right")
    ax.set_xlabel("Teff (K)")
    ax.set_ylabel("log column-mass error p95 (dex)")
    ax.set_title("per validation star")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "figures" / "m_star_logm_attribution_v1.png", dpi=160)
    print("saved figures/m_star_logm_attribution_v1.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
