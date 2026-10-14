"""Test-set (m,T) prediction errors of the matched normal-star networks."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("results/coauthor_validation_20261005/five_label")
    corpus = Path("source_data_files/atmosphere_emulator/five_label/strict_truth_52199.npz")
    with np.load(corpus, allow_pickle=False) as d:
        truth = d["atmosphere_profiles"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    report = {}
    for fields, color, style in ((2, "#1565c0", "-"), (6, "#d55e00", "--")):
        with np.load(root / f"predicted_{fields}profile_test.npz", allow_pickle=False) as d:
            index, predicted = d["star_indices"], d["profiles"]
        reference = truth[index]
        errors = (np.max(np.abs(predicted[:, :, 1] / reference[:, :, 1] - 1), axis=1),
                  np.max(np.abs(np.log10(predicted[:, :, 0] / reference[:, :, 0])), axis=1))
        for ax, error in zip(axes, errors):
            ax.plot(np.sort(error), np.arange(1, len(error) + 1) / len(error),
                    color=color, linestyle=style, label=f"{fields}-profile neural network")
        report[str(fields)] = dict(test_count=len(index),
            temperature_max_relative_median=float(np.median(errors[0])),
            column_mass_max_dex_median=float(np.median(errors[1])))
    for ax, label in zip(axes, (r"Max depth error in $T$ (relative)", r"Max depth error in $\log_{10}m$ (dex)")):
        ax.set(xscale="log", xlabel=label, ylabel="Fraction of test stars", ylim=(0, 1.01))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.savefig(root / "profile_accuracy.png", dpi=180)
    fig.savefig(root / "profile_accuracy.pdf")
    plt.close(fig)
    (root / "profile_accuracy.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
