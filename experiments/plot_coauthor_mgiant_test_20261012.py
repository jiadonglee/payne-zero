"""Success against iteration and start-spectrum accuracy for the final M-giant test."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10})
REPO = Path(__file__).resolve().parents[1]
STYLE = dict(two_profile=("two-profile network", "-", "#1565c0"),
             six_profile=("six-profile network", "--", "#d55e00"),
             marcs_node=("MARCS grid node", ":", "#2e7d32"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=REPO / "results/coauthor_validation_20261005/mgiant_holdout_test/summary.json")
    parser.add_argument("--out", type=Path, default=REPO / "paper/figs")
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text())["rows"]
    with_reference = [r for r in rows if r["reference_stable_iteration"] is not None]
    n = len(with_reference)
    args.out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    iterations = np.arange(0, 91)
    for arm, (label, style, color) in STYLE.items():
        stable = np.asarray([r["arms"][arm]["stable_iteration"] for r in with_reference if r["arms"][arm]["success"]])
        ax.step(iterations, [(stable <= k).sum() / n for k in iterations], where="post",
                ls=style, color=color, label=label)
    ax.set(xlabel="Solver iteration", ylabel="Fraction of test labels with a success", xlim=(0, 90), ylim=(0, 1.02))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=.2)
    fig.savefig(args.out / "fig_mconv.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    for arm, (label, style, color) in STYLE.items():
        values = np.sort([r["arms"][arm]["start_difference"]["normalized_flux"] for r in with_reference
                          if "start_difference" in r["arms"][arm]])
        # Starts without a spectrum stay in the denominator.
        ax.step(values, np.arange(1, len(values) + 1) / n, where="post", ls=style, color=color, label=label)
    ax.set(xscale="log", xlabel="Largest normalized-flux difference before solving",
           ylabel="Fraction of test labels below", ylim=(0, 1.02))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=.2)
    fig.savefig(args.out / "fig_mspectral.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
