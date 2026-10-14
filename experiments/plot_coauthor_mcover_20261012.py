"""Coverage of the bootstrapped M-giant corpus and the admitted fraction per round."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10})
ROOT = Path(__file__).resolve().parents[1] / "results/coauthor_validation_20261005"
OUT = Path(__file__).resolve().parents[1] / "paper/figs"


def main():
    corpus = np.load(ROOT / "mgiant_round3/corpus/corpus.npz")
    round3 = np.load(ROOT / "mgiant_cno_round3/cno_development_corpus.npz")
    labels = np.concatenate((corpus["labels"], round3["labels"]))
    setup = json.loads((ROOT / "mgiant_cno_multiparent/setup.json").read_text())
    summary = json.loads((ROOT / "mgiant_bootstrap_summary.json").read_text())
    box = summary["holdout_box"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=(1.5, 1)))
    ax = axes[0]
    points = ax.scatter(labels[:, 0], labels[:, 1], c=labels[:, 2], s=9, cmap="viridis", vmin=-1.0, vmax=0.5,
                        linewidths=0)
    for parent in setup["parents"]:
        t, g = parent["labels"]["effective_temperature"], parent["labels"]["log_surface_gravity"]
        if parent["role"] == "holdout":
            ax.add_patch(Rectangle((t - box["effective_temperature"], g - box["log_surface_gravity"]),
                                   2 * box["effective_temperature"], 2 * box["log_surface_gravity"],
                                   fill=False, ec="#d55e00", lw=1.0))
        else:
            ax.scatter(t, g, marker="s", s=28, facecolors="none", edgecolors="k", linewidths=0.8)
    ax.set(xlabel=r"$T_{\rm eff}$ (K)", ylabel=r"$\log g$", xlim=(4080, 2920), ylim=(2.65, 0.35))
    fig.colorbar(points, ax=ax, label=r"$[\mathrm{M}/\mathrm{H}]$", pad=0.01)
    ax.grid(alpha=.2)
    ax = axes[1]
    rounds = summary["rounds"]
    x = np.arange(1, len(rounds) + 1)
    all_labels = [rounds[str(k)]["admitted_fraction"] for k in x]
    with_reference = [rounds[str(k)]["admitted_fraction_with_reference"] for k in x]
    ax.bar(x - 0.18, all_labels, width=0.34, color="#1565c0", label="all labels")
    ax.bar(x + 0.18, with_reference, width=0.34, color="#90caf9", label="with a reference")
    ax.set_xticks(x, ["1\nnear parents", "2\nfull range", "3\ncool corner"])
    ax.set(xlabel="Bootstrapping round", ylabel="Admitted fraction", ylim=(0, 1.05))
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=.2, axis="y")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_mcover.pdf")
    fig.savefig(ROOT / "fig_mcover.png", dpi=180)


if __name__ == "__main__":
    main()
