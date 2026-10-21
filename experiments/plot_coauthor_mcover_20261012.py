"""Coverage of the bootstrapped M-giant corpus and the admitted fraction per round."""
import json
from pathlib import Path
import sys

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import paper_style as ps  # noqa: E402

METALLICITY = LinearSegmentedColormap.from_list("metallicity", ["#cfe3fb", "#5ea1ee", ps.BLUE, "#003a75"])
ROOT = Path(__file__).resolve().parents[1] / "results/coauthor_validation_20261005"
OUT = Path(__file__).resolve().parents[1] / "paper/figs"


def main():
    corpus = np.load(ROOT / "mgiant_round3/corpus/corpus.npz")
    round3 = np.load(ROOT / "mgiant_cno_round3/cno_development_corpus.npz")
    labels = np.concatenate((corpus["labels"], round3["labels"]))
    setup = json.loads((ROOT / "mgiant_cno_multiparent/setup.json").read_text())
    summary = json.loads((ROOT / "mgiant_bootstrap_summary.json").read_text())
    box = summary["holdout_box"]
    fig, axes = plt.subplots(1, 2, figsize=(ps.PAGE, 2.8), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=(1.6, 1)))
    ax = axes[0]
    points = ax.scatter(labels[:, 0], labels[:, 1], c=labels[:, 2], s=7, cmap=METALLICITY, vmin=-1.0, vmax=0.5,
                        linewidths=0, alpha=0.9)
    for parent in setup["parents"]:
        t, g = parent["labels"]["effective_temperature"], parent["labels"]["log_surface_gravity"]
        if parent["role"] == "holdout":
            ax.add_patch(Rectangle((t - box["effective_temperature"], g - box["log_surface_gravity"]),
                                   2 * box["effective_temperature"], 2 * box["log_surface_gravity"],
                                   fill=False, ec=ps.PRIMARY, lw=1.0))
        else:
            ax.scatter(t, g, marker="s", s=20, facecolors="none", edgecolors=ps.INK, linewidths=0.7)
    ax.set(xlabel=r"$T_{\rm eff}$ (K)", ylabel=r"$\log g$", xlim=(4080, 2920), ylim=(2.65, 0.35))
    handles = [plt.Line2D([], [], marker="s", ls="", mfc="none", mec=ps.INK, ms=4.5, mew=0.7),
               Rectangle((0, 0), 1, 1, fill=False, ec=ps.PRIMARY, lw=1.0)]
    ax.legend(handles, ["Development parent", "Holdout box"], loc="lower right", bbox_to_anchor=(1.0, 1.0),
              ncol=2, borderaxespad=0.2, handletextpad=0.4, columnspacing=1.2)
    bar = fig.colorbar(points, ax=ax, pad=0.01, aspect=30)
    bar.set_label(r"$[\mathrm{M}/\mathrm{H}]$")
    bar.outline.set_visible(False)
    ps.panel(ax, "a")
    ax = axes[1]
    rounds = summary["rounds"]
    x = np.arange(1, len(rounds) + 1)
    all_labels = [rounds[str(k)]["admitted_fraction"] for k in x]
    with_reference = [rounds[str(k)]["admitted_fraction_with_reference"] for k in x]
    for shift, values, color, label in ((-0.19, all_labels, ps.BLUE, "All labels"),
                                        (0.19, with_reference, "#8fbdf3", "Labels with a reference")):
        bars = ax.bar(x + shift, values, width=0.36, color=color, label=label)
        for rect, value in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, value + 0.015, f"{value:.0%}", ha="center", va="bottom",
                    fontsize=8, color=ps.SECONDARY)
    ax.set_xticks(x, ["1\nnear parents", "2\nfull range", "3\ncool corner"])
    ax.tick_params(axis="x", which="both", length=0)
    ax.set(xlabel="Bootstrapping round", ylabel="Admitted fraction", ylim=(0, 1.12), yticks=[0, 0.25, 0.5, 0.75, 1])
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.2,
              handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
    ps.panel(ax, "b")
    OUT.mkdir(parents=True, exist_ok=True)
    ps.save(fig, OUT / "fig_mcover.pdf")


if __name__ == "__main__":
    main()
