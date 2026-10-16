"""Success against iteration and start-spectrum accuracy for the final M-giant test."""
import argparse
import json
from pathlib import Path

import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
import matplotlib.pyplot as plt  # noqa: E402
import paper_style as ps  # noqa: E402

ARMS = ("two_profile", "six_profile", "marcs_node")


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

    fig, ax = plt.subplots(figsize=(ps.COLUMN, 3.0), constrained_layout=True)
    iterations = np.arange(0, 91)
    for arm in ARMS:
        role = ps.ROLES[arm]
        stable = np.asarray([r["arms"][arm]["stable_iteration"] for r in with_reference if r["arms"][arm]["success"]])
        fraction = [(stable <= k).sum() / n for k in iterations]
        ax.step(iterations, fraction, where="post", color=role["color"], ls=role["ls"],
                label=f"{role['label']}, {fraction[-1]:.0%}")
    ax.set(xlabel="Solver iteration", ylabel="Fraction of test labels converged", xlim=(0, 90), ylim=(0, 1.02))
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1)
    ps.save(fig, args.out / "fig_mconv.pdf")

    fig, ax = plt.subplots(figsize=(ps.COLUMN, 2.5), constrained_layout=True)
    for arm in ARMS:
        role = ps.ROLES[arm]
        values = np.sort([r["arms"][arm]["start_difference"]["normalized_flux"] for r in with_reference
                          if "start_difference" in r["arms"][arm]])
        # Starts without a spectrum stay in the denominator.
        ax.step(values, np.arange(1, len(values) + 1) / n, where="post", color=role["color"], ls=role["ls"],
                label=role["label"])
    ax.axvline(5e-3, color=ps.MUTED, lw=0.7, ls=(0, (1, 2)))
    ax.set(xscale="log", xlabel="Largest normalized-flux difference before solving",
           ylabel="Fraction of test labels", ylim=(0, 1.02))
    ax.legend(loc="upper left")
    ps.save(fig, args.out / "fig_mspectral.pdf")


if __name__ == "__main__":
    main()
