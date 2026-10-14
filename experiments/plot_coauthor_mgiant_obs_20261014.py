"""Benchmark-giant and APOGEE M-giant fits of Sect. 5 (fig_mgiant_obs, fig_mgiant_apogee)."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments/reduced_state_emulator")]
import coauthor_mgiant_apogee_fit_20261013 as apogee_fit  # noqa: E402
from experiments import mgiant_apogee_pzfit_v1 as apogee  # noqa: E402

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10})
ROOT = REPO / "results/coauthor_validation_20261005"
OUT = REPO / "paper/figs"
BENCH_STAR, BENCH_WINDOW_A = "gamSge", (6645.0, 6690.0)
APOGEE_STAR, APOGEE_WINDOW_NM = 86947577, (1555.0, 1570.0)
NAMES = {"alfCet": r"$\alpha$ Cet", "alfTau": r"$\alpha$ Tau", "gamSge": r"$\gamma$ Sge"}
COLOURS = dict(default="0.55", calibrated="#d55e00")


def benchmark_figure():
    summary = json.loads((ROOT / "mgiant_benchmark_fit/summary.json").read_text())
    fig = plt.figure(figsize=(9.0, 3.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.7, 1), height_ratios=(3, 1))
    top, bottom, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
    wave, flux, _, mask = np.load(ROOT / f"mgiant_benchmark_fit/calibrated/{BENCH_STAR}/observed.npy")
    inside = (wave >= BENCH_WINDOW_A[0]) & (wave <= BENCH_WINDOW_A[1])
    shown = np.where(mask.astype(bool), flux, np.nan)
    top.plot(wave[inside], shown[inside], color="k", lw=0.7, label=NAMES[BENCH_STAR])
    for arm in ("default", "calibrated"):
        model = np.load(ROOT / f"mgiant_benchmark_fit/{arm}/{BENCH_STAR}/physical_model_flux.npy")
        top.plot(wave[inside], model[inside], color=COLOURS[arm], lw=0.8, label=f"{arm} list")
        bottom.plot(wave[inside], (shown - model)[inside], color=COLOURS[arm], lw=0.6)
    bottom.axhline(0.0, color="k", lw=0.4)
    top.set(ylabel="Normalized flux")
    top.tick_params(labelbottom=False)
    top.legend(fontsize=8, loc="lower left", ncol=3)
    top.text(0.01, 0.95, "(a)", transform=top.transAxes, va="top", fontsize=10)
    right.text(0.03, 0.97, "(b)", transform=right.transAxes, va="top", ha="left", fontsize=10)
    bottom.set(xlabel=r"Air wavelength ($\mathrm{\AA}$)", ylabel="Residual")
    for arm, marker in (("default", "o"), ("calibrated", "s")):
        rows = [r for r in summary["rows"] if r["arm"] == arm]
        right.scatter([r["benchmark"]["teff"] for r in rows], [r["parameters"]["teff"] for r in rows],
                      marker=marker, s=36, facecolors="none" if arm == "default" else COLOURS[arm],
                      edgecolors=COLOURS[arm], label=f"{arm} list")
    for r in summary["rows"]:
        if r["arm"] == "calibrated":
            right.annotate(NAMES[r["star"]], (r["benchmark"]["teff"], r["parameters"]["teff"]), fontsize=8,
                           xytext=(4, 4), textcoords="offset points")
    lims = (3450, 4050)
    right.plot(lims, lims, color="k", lw=0.6)
    right.set(xlim=lims, ylim=lims, xlabel=r"Benchmark $T_{\rm eff}$ (K)", ylabel=r"Fitted $T_{\rm eff}$ (K)")
    right.legend(fontsize=8, loc="lower right")
    fig.savefig(OUT / "fig_mgiant_obs.pdf")
    plt.close(fig)


def apogee_figure():
    summary = json.loads((ROOT / "mgiant_apogee_fit/summary.json").read_text())
    rows = [r for r in summary["arms"]["offsets_aspcap"] if r["role"] != "calibration"]
    fig = plt.figure(figsize=(9.0, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=(1.6, 0.6, 2.0))
    top, bottom = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, :], sharex=None)
    obs = apogee_fit.Observation(APOGEE_STAR)
    astra = apogee.read_astra_aspcap(APOGEE_STAR)
    pixels = obs.mask & np.isfinite(astra["model_flux"]) & (astra["model_flux"] > 0)
    model = np.load(ROOT / f"mgiant_apogee_fit/offsets_aspcap/{APOGEE_STAR}/physical_model_flux.npy")
    design = astra["model_flux"][pixels, None] * obs.basis[pixels]
    w = np.sqrt(obs.weight[pixels])
    coefficients = np.linalg.lstsq(design * w[:, None], obs.flux[pixels] * w, rcond=None)[0]
    aspcap = np.full(obs.flux.size, np.nan)
    aspcap[pixels] = design @ coefficients
    inside = (obs.wave_nm >= APOGEE_WINDOW_NM[0]) & (obs.wave_nm <= APOGEE_WINDOW_NM[1])
    data = np.where(pixels, obs.flux, np.nan)
    top.plot(obs.wave_nm[inside], data[inside], color="k", lw=0.8, label="APOGEE")
    top.plot(obs.wave_nm[inside], model[inside], color="#d55e00", lw=0.8, label="Payne Zero")
    top.plot(obs.wave_nm[inside], aspcap[inside], color="#1565c0", lw=0.8, ls="--", label="ASPCAP model")
    bottom.plot(obs.wave_nm[inside], (data - model)[inside], color="#d55e00", lw=0.7)
    bottom.plot(obs.wave_nm[inside], (data - aspcap)[inside], color="#1565c0", lw=0.7, ls="--")
    bottom.axhline(0.0, color="k", lw=0.4)
    top.set(ylabel="Normalized flux", xlim=APOGEE_WINDOW_NM)
    top.tick_params(labelbottom=False)
    top.legend(fontsize=8, loc="lower right", ncol=3)
    top.text(0.005, 0.05, "(a)", transform=top.transAxes, va="bottom", fontsize=10)
    bottom.set(xlabel="Vacuum wavelength (nm)", ylabel="Residual", xlim=APOGEE_WINDOW_NM)
    sub = grid[2, :].subgridspec(1, 5)
    panels = (("teff", "teff", r"$T_{\rm eff}$", (3000, 4000)), ("logg", "logg", r"$\log g$", (0.4, 2.0)),
              ("m_h", "m_h_atm", r"[M/H]", (-1.0, 0.5)), ("c_m", "c_m_atm", r"[C/M]", (-0.3, 0.5)),
              ("n_m", "n_m_atm", r"[N/M]", (-0.1, 0.6)))
    for i, (key, raw, label, lims) in enumerate(panels):
        ax = fig.add_subplot(sub[0, i])
        x = np.array([r["raw"][raw] for r in rows])
        y = np.array([r["parameters"][key] for r in rows])
        ax.scatter(x, y, s=5, color="#1565c0", alpha=0.6, linewidths=0)
        ax.plot(lims, lims, color="k", lw=0.6)
        d = y - x
        unit = " K" if key == "teff" else ""
        fmt = "{:+.0f}" if key == "teff" else "{:+.2f}"
        ax.text(0.04, 0.96, rf"$\Delta$ = {fmt.format(d.mean())}{unit}" + "\n" + rf"$\sigma$ = {fmt.replace('+', '').format(d.std())}{unit}",
                transform=ax.transAxes, va="top", fontsize=8)
        ax.set(xlim=lims, ylim=lims, xlabel=f"ASPCAP {label}")
        ax.text(0.96, 0.04, f"({'bcdef'[i]})", transform=ax.transAxes, ha="right", fontsize=10)
        ax.set_ylabel(f"Payne Zero {label}" if i == 0 else "")
        ax.tick_params(labelsize=8)
        ax.locator_params(nbins=3)
    fig.savefig(OUT / "fig_mgiant_apogee.pdf")
    star = next(s for s in json.loads((ROOT / "mgiant_apogee_fit/sample.json").read_text())["stars"]
                if s["sdss_id"] == APOGEE_STAR)
    (ROOT / "mgiant_apogee_fit/figure_example.json").write_text(json.dumps(dict(
        sdss_id=APOGEE_STAR, role=star["role"], raw_teff=star["raw"]["teff"], window_nm=APOGEE_WINDOW_NM), indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    benchmark_figure()
    apogee_figure()
