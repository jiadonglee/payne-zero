"""Figure: why the one-shot adiabatic (EOS-polytrope) closure fails.

Two panels for one cool giant from the strict-truth corpus:

(a) exact converged temperature profile against the adiabatic projection the
    vetoed hook applied from layer 52 down;
(b) the exact logarithmic temperature gradient nabla_T against the EOS
    adiabatic band 0.14--0.39 observed for these stars.

The point is structural, not a fit residual: the real deep gradient is several
times the adiabatic value, so any closure that imposes the adiabatic slope
moves the deep temperature in the wrong direction.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from payne_zero_figures import style

RADIATION_CONSTANT = 7.5657e-15  # a_R, cgs
TRUTH_CORPUS = (
    "source_data_files/atmosphere_emulator/five_label/strict_truth_52199.npz"
)
STAR_SLUG = "five_label_bootstrap_005366"
ANCHOR_LAYER = 52
ADIABATIC_MID = 0.30
ADIABATIC_LOW = 0.14
ADIABATIC_HIGH = 0.39


def _load_star(slug):
    data = np.load(TRUTH_CORPUS)
    fields = list(data["target_fields"])
    slugs = np.asarray(data["slugs"])
    profiles = data["atmosphere_profiles"]
    index = int(np.flatnonzero(slugs == slug)[0])
    labels = json.loads(str(data["labels_json"][index]))
    profile = profiles[index]
    temperature = profile[:, fields.index("temperature")]
    gas_pressure = profile[:, fields.index("gas_pressure")]
    total_pressure = gas_pressure + RADIATION_CONSTANT / 3.0 * temperature**4
    return labels, temperature, total_pressure


def _adiabatic_projection(temperature, total_pressure, anchor, grad_ad):
    log_pressure = np.log(total_pressure)
    log_temperature = np.log(temperature).copy()
    for layer in range(anchor, temperature.size - 1):
        step = log_pressure[layer + 1] - log_pressure[layer]
        log_temperature[layer + 1] = log_temperature[layer] + grad_ad * step
    return np.exp(log_temperature)


def _population_peak_gradient():
    data = np.load(TRUTH_CORPUS)
    fields = list(data["target_fields"])
    profiles = data["atmosphere_profiles"]
    labels = [json.loads(str(x)) for x in data["labels_json"]]
    effective_temperature = np.array(
        [entry["effective_temperature"] for entry in labels]
    )
    temperature = profiles[:, :, fields.index("temperature")]
    gas_pressure = profiles[:, :, fields.index("gas_pressure")]
    total_pressure = gas_pressure + RADIATION_CONSTANT / 3.0 * temperature**4
    peak = np.empty(profiles.shape[0])
    for index in range(profiles.shape[0]):
        gradient = np.gradient(
            np.log(temperature[index]), np.log(total_pressure[index])
        )
        peak[index] = np.max(gradient[50:79])
    bins = np.array([4000.0, 5500.0, 7000.0, 8000.0, 10500.0])
    medians = []
    fraction_above = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (effective_temperature >= low) & (effective_temperature < high)
        medians.append(float(np.median(peak[mask])))
        fraction_above.append(float(np.mean(peak[mask] > ADIABATIC_HIGH)))
    return bins, np.array(medians), np.array(fraction_above)


def build_figure():
    style.configure("PAPER")
    palette = style.PaperPalette

    labels, temperature, total_pressure = _load_star(STAR_SLUG)
    layers = np.arange(temperature.size)
    nabla = np.gradient(np.log(temperature), np.log(total_pressure))
    projected = _adiabatic_projection(
        temperature, total_pressure, ANCHOR_LAYER, ADIABATIC_MID
    )

    fig = plt.figure(figsize=(style.DOUBLE, style.SINGLE * 1.2))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.28)
    ax_temp = fig.add_subplot(grid[0])
    ax_grad = fig.add_subplot(grid[1])
    ax_pop = fig.add_subplot(grid[2])

    ax_temp.plot(layers, temperature, color=palette.INK, lw=1.4,
                 label="exact solution")
    ax_temp.plot(layers[ANCHOR_LAYER:], projected[ANCHOR_LAYER:],
                 color=palette.CRITICAL, lw=1.4, ls="--",
                 label="adiabatic projection (vetoed)")
    ax_temp.axvline(ANCHOR_LAYER, color=palette.INK_MUTED, lw=0.7, ls=":")
    ax_temp.annotate(
        "projection starts", xy=(ANCHOR_LAYER, temperature[ANCHOR_LAYER]),
        xytext=(30, 7000), fontsize=7, color=palette.INK_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=palette.INK_MUTED, lw=0.6),
    )
    ax_temp.set_xlabel("layer index (deep toward right)")
    ax_temp.set_ylabel("temperature [K]")
    ax_temp.set_xlim(28, 79)
    ax_temp.set_ylim(3000, 9500)
    ax_temp.legend(loc="upper left")
    ax_temp.set_title("(a) temperature profile")

    ax_grad.plot(layers, nabla, color=palette.ORACLE, lw=1.4,
                 label="exact gradient")
    ax_grad.axhspan(ADIABATIC_LOW, ADIABATIC_HIGH, color=palette.PRODUCTION,
                    alpha=0.18, lw=0)
    ax_grad.axhline(ADIABATIC_MID, color=palette.PRODUCTION, lw=0.9, ls="--")
    ax_grad.annotate(
        "EOS adiabatic band", xy=(34, ADIABATIC_MID),
        xytext=(29, 1.6), fontsize=7, color=palette.INK_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=palette.INK_MUTED, lw=0.6),
    )
    ax_grad.axvline(ANCHOR_LAYER, color=palette.INK_MUTED, lw=0.7, ls=":")
    ax_grad.annotate(
        "real gradient is several times adiabatic",
        xy=(58, nabla[58]), xytext=(40, 3.55), fontsize=7, color=palette.CRITICAL,
        arrowprops=dict(arrowstyle="-", color=palette.INK_MUTED, lw=0.6),
    )
    ax_grad.set_xlabel("layer index (deep toward right)")
    ax_grad.set_ylabel("d ln T / d ln P")
    ax_grad.set_xlim(28, 79)
    ax_grad.set_ylim(-0.05, 4.0)
    ax_grad.set_title("(b) gradient vs. adiabatic")

    # (c) population evidence: the deep superadiabatic peak is the norm, not
    # an oddball.  Fraction of corpus stars whose layers 50--79 peak gradient
    # exceeds the EOS adiabatic ceiling, per effective-temperature bin.
    bins, medians, frac_above = _population_peak_gradient()
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax_pop.bar(range(len(centers)), frac_above, color=palette.ORACLE,
               alpha=0.85, width=0.62)
    for pos, (med, frac) in enumerate(zip(medians, frac_above)):
        ax_pop.text(pos, frac + 0.03, f"{med:.1f}", ha="center", va="bottom",
                    fontsize=6.5, color=palette.INK_SECONDARY)
    ax_pop.axhline(1.0, color=palette.INK_MUTED, lw=0.6, ls=":")
    ax_pop.set_xticks(range(len(centers)))
    ax_pop.set_xticklabels(
        [f"{int(lo)/1000:.0f}\u2013{int(hi)/1000:.0f}"
         for lo, hi in zip(bins[:-1], bins[1:])],
        fontsize=6.5,
    )
    ax_pop.set_xlabel("effective-temperature bin [$10^3$ K]")
    ax_pop.set_ylabel("fraction with deep peak above adiabatic")
    ax_pop.set_ylim(0.0, 1.2)
    ax_pop.set_title("(c) it is the whole population")
    ax_pop.annotate(
        "bar labels: median\npeak gradient",
        xy=(0.02, 0.97), xycoords="axes fraction", fontsize=6.2,
        va="top", color=palette.INK_MUTED,
    )

    fig.suptitle(
        "Why the adiabatic closure fails: one cool giant "
        "(Teff 4398 K, log g 1.53, [M/H] -2.03)",
        fontsize=8.5, y=1.02,
    )
    fig.tight_layout()
    return fig


def main():
    figure = build_figure()
    out = Path("payne_zero_figures/paper/adiabatic_closure_failure")
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(str(out) + ".png", dpi=300)
    figure.savefig(str(out) + ".pdf")
    print("wrote", str(out) + ".png")


if __name__ == "__main__":
    main()
