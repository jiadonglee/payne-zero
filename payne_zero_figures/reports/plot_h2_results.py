"""Paper figures for the H2 paired funnel and gate diagnostics."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.analytic_initializer.discovery import DEFAULT_CORPUS, load_strict_truth
from experiments.analytic_initializer.deep_diagnostics import deep_window
from experiments.analytic_initializer.profile_initializer import (
    load_analytic_profile_parameters, predict_analytic_reduced_state,
)

OUT = ROOT / "figures" / "analytic_initializer"
OUT.mkdir(parents=True, exist_ok=True)


def fig1_error_vs_teff():
    corpus = load_strict_truth(DEFAULT_CORPUS)
    params = load_analytic_profile_parameters(ROOT / "results/analytic_initializer/h2_profile_parameters_v1.npz")
    _, baseline, _ = predict_analytic_reduced_state(corpus.labels, corpus.tau, params)
    start, stop = deep_window(corpus.layers)
    logT = np.log10(corpus.temperature)
    err = np.abs(logT - np.log10(baseline))[:, start:stop].max(axis=1)
    teff = corpus.labels[:, 0]
    plt.figure(figsize=(7, 4.5))
    plt.scatter(teff, err, s=4, alpha=0.18, rasterized=True, label="per-star deep max |ΔlogT| (dex)")
    # band medians
    xs, ys, ylo, yhi = [], [], [], []
    for lo, hi in ((4000,5000),(5000,6000),(6000,7000),(7000,7500),(7500,8000),(8000,9000),(9000,10500)):
        m = (teff>=lo) & (teff<hi)
        if m.sum():
            xs.append(0.5*(lo+hi)); ys.append(np.median(err[m]))
            ylo.append(np.percentile(err[m], 5)); yhi.append(np.percentile(err[m], 95))
    plt.errorbar(xs, ys, yerr=[np.subtract(ys,ylo), np.subtract(yhi,ys)], fmt="o-", ms=5, lw=1.5, color="C1", capsize=3, label="band median with 5-95%")
    plt.axhline(0.02, color="k", ls="--", lw=1, label="Gate-1 deep p95 target")
    plt.axvspan(7000, 8000, color="C3", alpha=0.08)
    plt.xlabel("$T_{\\rm eff}$ (K)")
    plt.ylabel("deep-window max |$\\Delta\\log T$| (dex)")
    plt.ylim(0, 0.16)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "error_vs_teff.png", dpi=200)
    plt.close()
    print("fig1 error_vs_teff.png")


def fig2_profile(t_idx, outname):
    corpus = load_strict_truth(DEFAULT_CORPUS)
    params = load_analytic_profile_parameters(ROOT / "results/analytic_initializer/h2_profile_parameters_v1.npz")
    _, baseline, _ = predict_analytic_reduced_state(corpus.labels, corpus.tau, params)
    tau = corpus.tau
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(np.log10(tau), corpus.temperature[t_idx] / 1e3, "k-.", lw=1.5, label="truth")
    ax[0].plot(np.log10(tau), baseline[t_idx] / 1e3, "C0-", lw=1.5, label="H2 analytic")
    ax[0].set_xlabel("$\\log\\tau$")
    ax[0].set_ylabel("$T$ (kK)")
    ax[0].legend(frameon=False, fontsize=8)
    lo = int(max(np.log10(tau[0]), -1.5))
    ax[0].set_xlim(lo, np.log10(tau[-1]))
    # deep window zoom
    start, stop = deep_window(corpus.layers)
    t = np.log10(tau[start:stop])
    ax[1].plot(t, corpus.temperature[t_idx, start:stop] / 1e3, "k-.", lw=1.5, label="truth")
    ax[1].plot(t, baseline[t_idx, start:stop] / 1e3, "C0-", lw=1.5, label="H2 analytic")
    ax[1].set_xlabel("$\\log\\tau$ (deep window)")
    ax[1].set_ylabel("$T$ (kK)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"corpus index {corpus.slugs[t_idx] if hasattr(corpus,'slugs') else t_idx}", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / outname, dpi=200)
    plt.close()
    print("fig2", outname)


def fig3_iterations():
    anal = [json.loads(l) for l in open(ROOT/"results/analytic_initializer/h2_solver_funnel60.jsonl")]
    anal += json.load(open(ROOT/"results/analytic_initializer/h2_solver_funnel60_analytic_rest.json"))["records"]
    prod = json.load(open(ROOT/"results/analytic_initializer/h2_solver_funnel60_production.json"))["records"]
    bya = {a["corpus_index"]: a for a in anal}; byp = {p["corpus_index"]: p for p in prod}
    both = [i for i in bya if bya[i]["converged"] and byp[i]["converged"] and byp[i].get("iterations_completed") is not None]
    pa = [bya[i]["iterations_completed"] for i in both]
    pp = [byp[i]["iterations_completed"] for i in both]
    plt.figure(figsize=(6.5, 5))
    plt.scatter(pp, pa, s=30, alpha=0.7, edgecolor="none")
    lim = (0, max(max(pa), max(pp)) + 1)
    plt.plot(lim, lim, "k:", lw=1)
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("production iterations (emulator warm start)")
    plt.ylabel("H2 analytic iterations")
    below = sum(x < y for x, y in zip(pa, pp))
    equal = sum(x == y for x, y in zip(pa, pp))
    plt.title(f"paired converged stars n={len(both)}\nanalytic faster: {below}, equal: {equal}", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT / "paired_iterations.png", dpi=200)
    plt.close()
    print("fig3 paired_iterations.png")


if __name__ == "__main__":
    with open(ROOT / "results/analytic_initializer/entropy_closure_oracle.json") as f:
        oracle = json.load(f)
    caps = ["fi", "very"][0]
    fig1_error_vs_teff()
    fig2_profile(25948, "profile_transition_cool.png")  # 7246 K
    fig2_profile(35654, "profile_transition_warm.png")  # 7397 K
    fig3_iterations()
    print("all figs written to", OUT)
