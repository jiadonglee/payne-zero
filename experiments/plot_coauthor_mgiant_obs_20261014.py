"""Benchmark-giant and APOGEE M-giant fits of Sect. 5 (fig_mgiant_obs, fig_mgiant_apogee)."""
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments/reduced_state_emulator"), str(REPO / "experiments")]
import coauthor_mgiant_apogee_fit_20261013 as apogee_fit  # noqa: E402
from experiments import mgiant_apogee_pzfit_v1 as apogee  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import paper_style as ps  # noqa: E402

ROOT = REPO / "results/coauthor_validation_20261005"
OUT = REPO / "paper/figs"
BENCH_STAR, BENCH_WINDOW_A = "gamSge", (6645.0, 6690.0)
APOGEE_STAR, APOGEE_WINDOW_NM = 86947577, (1555.0, 1570.0)
NAMES = {"alfCet": r"$\alpha$ Cet", "alfTau": r"$\alpha$ Tau", "gamSge": r"$\gamma$ Sge"}
COLOURS = dict(default=ps.LIGHT, calibrated=ps.PRIMARY)
LABELS = dict(default="Default line list", calibrated="Calibrated TiO lines")


def benchmark_figure():
    summary = json.loads((ROOT / "mgiant_benchmark_fit/summary.json").read_text())
    fig = plt.figure(figsize=(ps.PAGE, 2.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.75, 1), height_ratios=(2.4, 1))
    top, bottom, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[:, 1])
    bottom.sharex(top)
    wave, flux, _, mask = np.load(ROOT / f"mgiant_benchmark_fit/calibrated/{BENCH_STAR}/observed.npy")
    inside = (wave >= BENCH_WINDOW_A[0]) & (wave <= BENCH_WINDOW_A[1])
    shown = np.where(mask.astype(bool), flux, np.nan)
    top.plot(wave[inside], shown[inside], color=ps.INK, lw=0.9, label=f"{NAMES[BENCH_STAR]} (NARVAL)")
    for arm in ("default", "calibrated"):
        model = np.load(ROOT / f"mgiant_benchmark_fit/{arm}/{BENCH_STAR}/physical_model_flux.npy")
        width = 1.8 if arm == "default" else 1.0
        colour = COLOURS[arm]
        top.plot(wave[inside], model[inside], color=colour, lw=width, label=LABELS[arm])
        bottom.plot(wave[inside], (shown - model)[inside], color=colour, lw=width * 0.8)
    bottom.axhline(0.0, color=ps.MUTED, lw=0.6)
    top.set(ylabel="Normalized flux", xlim=BENCH_WINDOW_A)
    top.tick_params(labelbottom=False)
    top.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
    bottom.set(xlabel=r"Air wavelength ($\mathrm{\AA}$)", ylabel="Residual")
    ps.panel(top, "a", y=1.14)
    rows = {(r["arm"], r["star"]): r for r in summary["rows"]}
    stars = ("alfCet", "gamSge", "alfTau")
    for y, star in enumerate(stars):
        before = rows["default", star]["parameters"]["teff"] - rows["default", star]["benchmark"]["teff"]
        after = rows["calibrated", star]["parameters"]["teff"] - rows["calibrated", star]["benchmark"]["teff"]
        right.annotate("", (after, y), (before, y),
                       arrowprops=dict(arrowstyle="-|>", color=ps.MUTED, lw=0.9, shrinkA=4, shrinkB=4))
        right.scatter(before, y, s=34, facecolors="white", edgecolors=ps.MUTED, linewidths=1.2, zorder=3)
        right.scatter(after, y, s=34, color=ps.PRIMARY, zorder=3)
        ps.end_label(right, after, y, f"{after:+.0f} K".replace("-", "\u2212"), dx=6, ha="left")
    right.axvline(0, color=ps.INK, lw=0.6)
    right.set_yticks(range(len(stars)), [NAMES[s] for s in stars])
    right.tick_params(axis="y", which="both", length=0)
    right.tick_params(axis="y", which="both", right=False)
    right.set(xlim=(-320, 60), ylim=(-0.6, len(stars) - 0.4),
              xlabel=r"Fitted $-$ benchmark $T_{\rm eff}$ (K)")
    right.text(0.02, 0.98, "γ Sge not used in the calibration", transform=right.transAxes,
               color=ps.SECONDARY, va="top")
    handles = [plt.Line2D([], [], ls="", marker="o", mfc="white", mec=ps.MUTED, mew=1.2, ms=5.5),
               plt.Line2D([], [], ls="", marker="o", color=ps.PRIMARY, ms=5.5)]
    right.legend(handles, ["Default", "Calibrated"], loc="lower left", bbox_to_anchor=(0.1, 1.0), ncol=2,
                 borderaxespad=0.2, handletextpad=0.3)
    ps.panel(right, "b", y=1.06)
    ps.save(fig, OUT / "fig_mgiant_obs.pdf")


def apogee_figure():
    summary = json.loads((ROOT / "mgiant_apogee_fit/summary.json").read_text())
    rows = [r for r in summary["arms"]["linebyline_aspcap"] if r["role"] != "calibration"]
    fig = plt.figure(figsize=(ps.PAGE, 4.6), constrained_layout=True)
    grid = fig.add_gridspec(3, 5, height_ratios=(1.6, 0.6, 1.9))
    top = fig.add_subplot(grid[0, :])
    bottom = fig.add_subplot(grid[1, :], sharex=top)
    obs = apogee_fit.Observation(APOGEE_STAR)
    astra = apogee.read_astra_aspcap(APOGEE_STAR)
    pixels = obs.mask & np.isfinite(astra["model_flux"]) & (astra["model_flux"] > 0)
    model = np.load(ROOT / f"mgiant_apogee_fit/linebyline_aspcap/{APOGEE_STAR}/physical_model_flux.npy")
    design = astra["model_flux"][pixels, None] * obs.basis[pixels]
    w = np.sqrt(obs.weight[pixels])
    coefficients = np.linalg.lstsq(design * w[:, None], obs.flux[pixels] * w, rcond=None)[0]
    aspcap = np.full(obs.flux.size, np.nan)
    aspcap[pixels] = design @ coefficients
    inside = (obs.wave_nm >= APOGEE_WINDOW_NM[0]) & (obs.wave_nm <= APOGEE_WINDOW_NM[1])
    data = np.where(pixels, obs.flux, np.nan)
    top.plot(obs.wave_nm[inside], data[inside], color=ps.INK, lw=0.9, label="APOGEE")
    top.plot(obs.wave_nm[inside], aspcap[inside], color=ps.ALTERNATIVE, lw=1.0, ls=(0, (3, 1.5)), label="ASPCAP model")
    top.plot(obs.wave_nm[inside], model[inside], color=ps.PRIMARY, lw=1.0, label="Payne Zero")
    bottom.plot(obs.wave_nm[inside], (data - aspcap)[inside], color=ps.ALTERNATIVE, lw=0.8, ls=(0, (3, 1.5)))
    bottom.plot(obs.wave_nm[inside], (data - model)[inside], color=ps.PRIMARY, lw=0.8)
    bottom.axhline(0.0, color=ps.MUTED, lw=0.6)
    top.set(ylabel="Normalized flux", xlim=APOGEE_WINDOW_NM)
    top.tick_params(labelbottom=False)
    top.legend(loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2)
    ps.panel(top, "a", y=1.13)
    bottom.set(xlabel="Vacuum wavelength (nm)", ylabel="Residual")
    panels = (("teff", "teff", r"$T_{\rm eff}$ (K)", (3000, 4000)), ("logg", "logg", r"$\log g$", (0.4, 2.0)),
              ("m_h", "m_h_atm", r"[M/H]", (-1.0, 0.5)), ("c_m", "c_m_atm", r"[C/M]", (-0.3, 0.5)),
              ("n_m", "n_m_atm", r"[N/M]", (-0.1, 0.6)))
    for i, (key, raw, label, lims) in enumerate(panels):
        ax = fig.add_subplot(grid[2, i])
        x = np.array([r["raw"][raw] for r in rows])
        y = np.array([r["parameters"][key] for r in rows])
        ax.plot(lims, lims, color=ps.MUTED, lw=0.6, zorder=1)
        ax.scatter(x, y, s=6, color=ps.PRIMARY, alpha=0.5, linewidths=0, zorder=2, rasterized=True)
        d = y - x
        unit = " K" if key == "teff" else ""
        fmt = "{:+.0f}" if key == "teff" else "{:+.2f}"
        ax.text(0.05, 0.95, rf"$\Delta$ {fmt.format(d.mean())}{unit}" + "\n" + rf"$\sigma$ {fmt.replace('+', '').format(d.std())}{unit}",
                transform=ax.transAxes, va="top", color=ps.SECONDARY)
        ax.set(xlim=lims, ylim=lims, aspect="equal")
        ax.set_xlabel(f"ASPCAP {label}")
        ax.set_ylabel(f"Payne Zero {label}" if i == 0 else "")
        ax.locator_params(nbins=3)
        ps.panel(ax, "bcdef"[i])
    ps.save(fig, OUT / "fig_mgiant_apogee.pdf")
    star = next(s for s in json.loads((ROOT / "mgiant_apogee_fit/sample.json").read_text())["stars"]
                if s["sdss_id"] == APOGEE_STAR)
    (ROOT / "mgiant_apogee_fit/figure_example.json").write_text(json.dumps(dict(
        sdss_id=APOGEE_STAR, role=star["role"], raw_teff=star["raw"]["teff"], window_nm=APOGEE_WINDOW_NM), indent=2) + "\n")


if __name__ == "__main__":
    benchmark_figure()
    apogee_figure()
