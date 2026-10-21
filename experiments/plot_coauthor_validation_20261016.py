"""Figures of the normal-star validation (Sect. 3): reconstruction, convergence, mismatch, spectra.

``export`` runs where the restart records are (Garching) and writes the plotted
arrays with the selections of ``plot_coauthor_restarts_20261005.py`` and the
spectral CDF of ``coauthor_collect_20261005.py``; ``plot`` draws them in the
paper style.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ARMS = ("two_profile", "six_profile", "nearest_0", "nearest_1", "grey_convective")
SPECTRAL_ARMS = ("exact_rebuild", "two_profile", "six_profile", "nearest_0", "nearest_1", "grey_convective")
PROFILES = (r"$P_{\rm gas}$", r"$n_{\rm e}$", r"$\kappa_{\rm R}$", r"$g_{\rm rad}$")
KEY = "deep_layer_relative_temperature_change"


def export(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "reduced_state_emulator"))
    from coauthor_benchmark_20261005 import load_data

    records = [json.loads(p.read_text()) for p in sorted(args.runs.glob("star_*/record.json"))]
    data = load_data(args.corpus_root, "five_label", Path("results/coauthor_validation_20261005"))
    truth, tau = data["profiles"], data["tau"]
    out = dict(tau=tau)
    errors = []
    for record in records:
        path = args.runs / f"star_{record['index']:06d}/exact_rebuild.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as d:
                reference = truth[record["index"], :, 2:6]
                errors.append(np.abs(d["start"][:, 2:6] - reference) / np.maximum(np.abs(reference), 1e-30))
    errors = np.asarray(errors)
    out["reconstruction_percentiles"] = np.stack([np.percentile(errors[:, :, j], (25, 50, 90), axis=0) for j in range(4)])
    out["reconstruction_count"] = len(errors)
    for arm in ARMS:
        entries = [r["arms"][arm] for r in records if arm in r["arms"]]
        trajectories = [e.get("trajectory", []) for e in entries]
        nmax = max(len(t) for t in trajectories)
        out[f"{arm}_median"] = np.asarray([np.median([t[n][KEY] for t in trajectories if len(t) > n]) for n in range(nmax)])
        out[f"{arm}_quartiles"] = np.asarray([np.percentile([t[n][KEY] for t in trajectories if len(t) > n], (25, 75))
                                             for n in range(nmax)])
        counts = np.asarray([e["iterations"] if e["converged"] else np.inf for e in entries])
        out[f"{arm}_converged_fraction"] = np.asarray([(counts <= k).mean() for k in range(1, 16)])
    candidates = [r for r in records if all(r["arms"].get(a, {}).get("trajectory") for a in ("two_profile", "six_profile"))]
    candidates.sort(key=lambda r: r["arms"]["two_profile"]["trajectory"][0][KEY])
    typical = candidates[len(candidates) // 2]
    out["typical_index"] = typical["index"]
    for arm in ("two_profile", "six_profile"):
        trajectories = [r["arms"][arm]["trajectory"] for r in candidates]
        nmax = max(map(len, trajectories))
        out[f"mismatch_{arm}_median"] = np.asarray([np.median([t[n][KEY] for t in trajectories if len(t) > n]) for n in range(nmax)])
        out[f"mismatch_{arm}_quartiles"] = np.asarray([np.percentile([t[n][KEY] for t in trajectories if len(t) > n], (25, 75))
                                                      for n in range(nmax)])
        with np.load(args.runs / f"star_{typical['index']:06d}/{arm}.npz", allow_pickle=False) as d:
            out[f"mismatch_{arm}_corrections"] = np.asarray(d["relative_corrections"][:3])
    spectra = [json.loads(p.read_text()) for p in sorted(args.spectra.glob("star_*.json"))]
    out["spectral_expected"] = len(spectra)
    for arm in SPECTRAL_ARMS:
        values = np.asarray([r.get("arms", {}).get(arm, {}).get("maximum_normalized_flux_difference", np.inf) for r in spectra])
        out[f"spectral_{arm}"] = np.sort(values[np.isfinite(values)])
    np.savez(args.out, **out)


def plot(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import matplotlib.pyplot as plt
    import paper_style as ps

    d = np.load(args.data)
    tau = d["tau"]
    style = lambda arm: {k: v for k, v in ps.ROLES[arm].items() if k != "label"}

    fig, axes = plt.subplots(2, 2, figsize=(ps.COLUMN, 2.9), sharex=True, constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        q25, median, q90 = np.maximum(d["reconstruction_percentiles"][j], 1e-12)
        ax.fill_between(tau, q25, q90, color=ps.PRIMARY, alpha=0.18, lw=0)
        ax.plot(tau, median, color=ps.PRIMARY, lw=1.3)
        ax.set(xscale="log", yscale="log", xticks=[1e-6, 1e-3, 1, 1e3])
        ax.text(0.04, 0.92, PROFILES[j], transform=ax.transAxes, va="top", fontsize=9)
        if j >= 2:
            ax.set_xlabel(r"$\tau_{\rm R}$")
        if j % 2 == 0:
            ax.set_ylabel("Relative error")
    ps.save(fig, args.figs / "fig_reconstruction.pdf")

    fig, (left, right) = plt.subplots(1, 2, figsize=(ps.PAGE, 2.5), constrained_layout=True)
    n = np.arange(1, 16)
    for arm in ARMS:
        median = d[f"{arm}_median"][:15]
        x = np.arange(1, len(median) + 1)
        left.plot(x, median, **style(arm))
        right.plot(n, d[f"{arm}_converged_fraction"], **style(arm))
    left.axhline(5e-4, color=ps.MUTED, lw=0.7, ls=(0, (1, 2)))
    ps.end_label(left, 15, 5e-4, "stop limit", dx=-3, dy=-2, ha="right", va="top", color=ps.MUTED)
    left.set(yscale="log", xlim=(1, 15), ylim=(2.5e-4, None), xlabel="Completed iteration $n$", ylabel=r"Deep-layer temperature change $\epsilon_T$")
    right.set(xlim=(1, 15), ylim=(0, 1.02), xlabel="Completed iteration $n$", ylabel="Fraction converged")
    handles = [plt.Line2D([], [], **style(arm)) for arm in ARMS]
    left.legend(handles, [ps.ROLES[arm]["label"] for arm in ARMS], loc="upper right")
    for ax, letter in ((left, "a"), (right, "b")):
        ps.panel(ax, letter)
    ps.save(fig, args.figs / "fig_convergence.pdf")

    fig, (left, right) = plt.subplots(1, 2, figsize=(ps.PAGE, 2.5), constrained_layout=True)
    for arm in ("two_profile", "six_profile"):
        median = d[f"mismatch_{arm}_median"][:15]
        q = d[f"mismatch_{arm}_quartiles"][:15]
        x = np.arange(1, len(median) + 1)
        left.fill_between(x, q[:, 0], q[:, 1], color=ps.ROLES[arm]["color"], alpha=0.12, lw=0)
        left.plot(x, median, **style(arm), label=ps.ROLES[arm]["label"])
        for k, correction in enumerate(d[f"mismatch_{arm}_corrections"]):
            right.plot(np.arange(1, 81), correction, color=ps.ROLES[arm]["color"], ls=ps.ROLES[arm]["ls"],
                       lw=1.4 - 0.4 * k, alpha=1.0 - 0.3 * k)
    left.set(yscale="log", xlim=(1, 15), xlabel="Completed iteration $n$", ylabel=r"Deep-layer temperature change $\epsilon_T$")
    left.legend(loc="upper right")
    right.set(yscale="log", xlim=(1, 80), xlabel="Layer $j$", ylabel=r"Temperature correction $|\Delta T_j/T_j|$")
    right.legend([plt.Line2D([], [], color=ps.MUTED, lw=1.4 - 0.4 * k, alpha=1.0 - 0.3 * k) for k in range(3)],
                 [f"$n$ = {k + 1}" for k in range(3)], loc="lower left", ncol=3, handlelength=1.6)
    for ax, letter in ((left, "a"), (right, "b")):
        ps.panel(ax, letter)
    ps.save(fig, args.figs / "fig_mismatch.pdf")

    fig, ax = plt.subplots(figsize=(ps.COLUMN, 2.9), constrained_layout=True)
    expected = int(d["spectral_expected"])
    for arm in SPECTRAL_ARMS:
        values = np.maximum(d[f"spectral_{arm}"], 1e-12)
        ax.plot(values, np.arange(1, len(values) + 1) / expected, **style(arm), lw=1.3, label=ps.ROLES[arm]["label"])
    ax.axvline(5e-3, color=ps.MUTED, lw=0.7, ls=(0, (1, 2)))
    ax.set(xscale="log", ylim=(0, 1.02), xlabel="Maximum normalized-flux difference",
           ylabel="Fraction of test stars")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, handlelength=2.0)
    ps.save(fig, args.figs / "fig_spectral.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("export", "plot"))
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--spectra", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--figs", type=Path, default=Path("paper/figs"))
    args = parser.parse_args()
    export(args) if args.stage == "export" else plot(args)


if __name__ == "__main__":
    main()
