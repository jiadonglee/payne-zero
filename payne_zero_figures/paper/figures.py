#!/usr/bin/env python3
"""Publication figures for the A&A manuscript, at journal column widths.

Nothing here recomputes physics -- every panel re-plots arrays already written
to ``results/`` and ``runs/``, so the figures cannot disagree with the tables
they sit next to.  Palettes, rcParams and artifact loading come from
:mod:`payne_zero_figures.style` and :mod:`payne_zero_figures.data`; this module
holds only panel composition.

The visual style follows Paper I (arXiv:2607.24141): serif text (TeX Gyre
Termes, matching the txfonts body), STIX math, a restrained
black/orange/steel-blue palette, no gridlines, inward ticks on all four sides,
and one horizontal legend row above the figure rather than legends inside the
panels.

Run from the repository root, with an interpreter that has matplotlib -- which
``.venv`` (the solver environment) does not::

    MPLCONFIGDIR=/tmp/mpl python3 -m payne_zero_figures.paper.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colorbar  # noqa: E402
import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from payne_zero_figures import data, style  # noqa: E402
from payne_zero_figures.data import (  # noqa: E402
    binned_max as _binned_max,
    load_records as _load_records,
)
from payne_zero_figures.style import (  # noqa: E402
    BAR,
    DOUBLE,
    SINGLE,
    inward as _inward,
    top_legend as _top_legend,
)

REPO = data.REPO
RESULTS = data.RESULTS
RUNS = data.EMULATOR_RUNS
BLIND = RESULTS / "solver_in_loop_k1_qualified_tail3_profile_rescue_v4"
PHYSICAL_RESULTS = RESULTS / "paper_physical_seed_20260820"
PHYSICAL_RUNS = REPO / "runs" / "paper_physical_seed_20260820"
GREY_CONVECTIVE_RESULTS = RESULTS / "paper_grey_convective_20260829"
GREY_CONVECTIVE_RUNS = REPO / "runs" / "paper_grey_convective_20260829"
GREY_CONVECTIVE_ARM = "hydrostatic_grey_convective"
MSTAR_RESULTS = RESULTS / "m_star_science_case_v1"
FIGS = REPO / "paper" / "figs"

# Paper I semantic roles, spelled out locally so the panel code below reads the
# way it did before the palette moved into style.py.
_P = style.PaperPalette
INK = _P.INK
INK_SECONDARY = _P.INK_SECONDARY
INK_MUTED = _P.INK_MUTED
LEARNED = _P.LEARNED
PRODUCTION = _P.PRODUCTION
ORACLE = _P.ORACLE
CRITICAL = _P.CRITICAL
ANALYTIC = _P.ORACLE

# Neutral hues for series that carry no arm semantics (the four fields,
# closure quantiles).
FIELD_COLORS = [INK, ORACLE, LEARNED, INK_MUTED]
QUANTILE_COLORS = {"median": INK, "$p_{99}$": ORACLE, "max": INK_MUTED}

style.configure("PAPER")

FIELDS = [
    ("gas_pressure", r"$P_{\rm gas}$"),
    ("electron_density", r"$n_{\rm e}$"),
    ("rosseland_opacity", r"$\kappa_{\rm R}$"),
    ("radiative_acceleration", r"$g_{\rm rad}$"),
]


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


# --------------------------------------------------------------------------
# Fig. 1 -- what the reduced state is
# --------------------------------------------------------------------------


def figure_state() -> None:
    """Three initialization paths entering the same atmosphere solver."""

    fig, ax = plt.subplots(figsize=(DOUBLE, 3.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 39)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, fontsize=6.2, weight="normal", tc=INK):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.22,rounding_size=0.65",
            facecolor=face, edgecolor=edge, linewidth=0.8, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=tc, zorder=4, weight=weight,
                linespacing=1.25)

    def arrow(x0, y0, x1, y1, color=INK_MUTED, lw=0.9, connectionstyle=None):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
            color=color, linewidth=lw, zorder=2, shrinkA=0, shrinkB=0,
            connectionstyle=connectionstyle))

    # Shared inputs and output.
    box(1.5, 34.2, 16.5, 3.2, "stellar labels  $\\theta$",
        "#f2f2f0", INK_MUTED, weight="bold")
    box(20.0, 34.2, 18.5, 3.2,
        r"fixed grid  $\tau_{\rm std}$", "#f2f2f0", INK_MUTED)
    ax.text(
        41.0, 35.8,
        r"same labels, depth grid, solver, and convergence criterion",
        ha="left", va="center", fontsize=6.2, color=INK_SECONDARY,
        style="italic",
    )
    box(
        79.0, 8.2, 19.3, 20.7,
        "same atmosphere solver\n\n"
        "radiative transfer\n"
        "equation of state\n"
        "hydrostatic closure\n"
        "energy correction\n\n"
        r"stop when"
        "\n"
        r"$\max_{\rm deep}|\Delta T/T|$"
        "\n"
        r"$<5\times10^{-4}$",
        "#e2ebf2", ORACLE, fontsize=6.0, weight="bold",
    )

    # Six-field reference path.
    ax.text(1.5, 31.7, "(a) six-field neural initializer",
            ha="left", va="center",
            fontsize=7.0, color=INK, weight="bold")
    box(2.0, 25.6, 16.0, 4.7, "six-field\nnetwork",
        "#efefed", PRODUCTION, weight="bold")
    box(
        21.5, 25.6, 44.0, 4.7,
        r"predicts $m,T,P_{\rm gas},n_{\rm e},$"
        r"$\kappa_{\rm R},g_{\rm rad}$ independently",
        "#efefed", PRODUCTION,
    )
    arrow(18.4, 27.95, 21.1, 27.95, color=PRODUCTION, lw=1.1)
    arrow(65.9, 27.95, 78.6, 25.1, color=PRODUCTION, lw=1.1)

    # Learned two-field path.
    ax.text(1.5, 23.6, "(b) learned two-field initializer", ha="left",
            va="center", fontsize=7.0, color=INK, weight="bold")
    box(2.0, 17.4, 16.0, 4.7, "two-field\nnetwork",
        "#f6e9db", LEARNED, weight="bold")
    box(21.5, 17.4, 11.0, 4.7, r"predicts $m,T$",
        "#f6e9db", LEARNED, weight="bold")
    box(
        36.0, 17.4, 29.5, 4.7,
        r"physical synchronization at fixed $(m,T)$"
        "\n"
        r"EOS, opacity, transfer, hydrostatic closure",
        "#f7f2f8", "#a77aae", fontsize=5.7,
    )
    arrow(18.4, 19.75, 21.1, 19.75, color=LEARNED, lw=1.1)
    arrow(32.9, 19.75, 35.6, 19.75, color=LEARNED, lw=1.1)
    arrow(65.9, 19.75, 78.6, 19.0, color=LEARNED, lw=1.1)

    # Emulator-independent physical path.
    ax.text(
        1.5, 15.5,
        "(c) hydrostatic grey–convective initializer",
        ha="left", va="center", fontsize=7.0, color=INK, weight="bold",
    )
    box(
        2.0, 5.3, 16.0, 7.8,
        "grey atmosphere\n"
        "hydrostatic balance\n"
        "convective adjustment",
        "#e9f1ec", "#2f7d63", fontsize=5.9, weight="bold",
    )
    box(
        21.5, 5.3, 21.0, 7.8,
        r"$m_{\rm seed}=m_{\rm grey}$"
        "\n"
        r"$T_{\rm seed}=T_{\rm conv}$"
        "\n"
        r"$P_{\rm seed}=g\,m_{\rm grey}$",
        "#e9f1ec", "#2f7d63", fontsize=6.1,
    )
    box(
        46.0, 5.3, 19.5, 7.8,
        r"$\kappa_{\rm seed}=$"
        "\n"
        r"$\kappa(T_{\rm conv},P_{\rm seed})$"
        "\n"
        "no atmosphere emulator",
        "#e9f1ec", "#2f7d63", fontsize=5.8, weight="bold",
    )
    arrow(18.4, 9.2, 21.1, 9.2, color="#2f7d63", lw=1.1)
    arrow(42.9, 9.2, 45.6, 9.2, color="#2f7d63", lw=1.1)
    arrow(65.9, 9.2, 78.6, 12.4, color="#2f7d63", lw=1.1)
    ax.text(
        2.0, 2.4,
        r"The convective temperature correction does not trigger a second "
        r"integration of the column-mass coordinate.",
        ha="left", va="center", fontsize=5.8, color=INK_SECONDARY,
        style="italic",
    )

    _save(fig, "fig_state.pdf")


# --------------------------------------------------------------------------
# Fig. 2 -- the atmospheres themselves, the way Paper I Fig. 4 shows them
# --------------------------------------------------------------------------

# The four rows of the main-text representation-spectrum figure. They span
# the label support and include the exact-(m,T) parity star with the largest
# normalized-flux difference. Even that deliberate worst case passes the
# 5e-3 acceptance bar.
SPECTRA_ROWS = [
    ("Cool giant", "t04096.6_g+0.72_m-1.91_a-0.10_x3.87", "Na D",          (588.4, 590.2)),
    ("Solar type", "t05098.3_g+4.44_m-0.18_a+0.03_x3.07", "Mg b",          (516.4, 518.6)),
    ("Hot, metal rich", "t10469.8_g+2.80_m+0.28_a+0.35_x3.31",
     r"H$\beta$", (485.0, 487.5)),
    ("Largest difference", "t07769.7_g+2.77_m-0.56_a-0.03_x3.39",
     "Ca triplet", (849.0, 867.5)),
]

# Which label each column of Fig. 2 colours by, with the colormap Paper I uses.
PROFILE_COLUMNS = [
    (0, r"$T_{\rm eff}$ (K)", "{:.0f}"),
    (1, r"$\log g$", "{:.2f}"),
    (2, r"$[\mathrm{M}/\mathrm{H}]$", "{:.2f}"),
]


def _label_colors(values):
    """Paper I Fig. 4 colour scale: plasma, spanning the sampled range."""

    lo, hi = float(np.min(values)), float(np.max(values))
    norm = matplotlib.colors.Normalize(vmin=lo, vmax=hi)
    return plt.get_cmap("plasma"), norm, lo, hi


def figure_profiles() -> None:
    """Reference, learned, and hydrostatic grey--convective seed profiles."""

    with np.load(
        PHYSICAL_RESULTS / "parity" / "reconstruction_metrics.npz",
        allow_pickle=False,
    ) as data:
        tau = data["tau_std"]
    with np.load(REPO / "artifacts" / "reduced_state_emulator"
                 / "predicted_monotone.npz", allow_pickle=False) as data:
        labels = data["labels"]
        pred = {"m": data["column_mass"], "T": data["temperature"]}
        truth = {"m": data["truth_column_mass"], "T": data["truth_temperature"]}
    with np.load(
        GREY_CONVECTIVE_RESULTS / "development" / "seeds.npz",
        allow_pickle=False,
    ) as data:
        if not np.array_equal(labels, data["labels"]):
            raise ValueError("learned and physical profile samples are not aligned")
        if not np.array_equal(tau, data["tau"]):
            raise ValueError("learned and physical depth grids are not aligned")
        physical = {"m": data["column_mass"], "T": data["temperature"]}

    # Thin the sample: 60 pairs of curves per panel is a solid block.  Twelve
    # stars evenly spaced in the colouring label read as a family.
    rows = [
        ("m", r"$\log_{10}\, m$ (g cm$^{-2}$)",
         lambda p: np.log10(np.maximum(p, 1e-12)), (-6.6, 3.7), False),
        ("T", r"$T$ (kK)", lambda p: p / 1000.0, (1.8, 180.0), True),
    ]

    fig, axes = plt.subplots(len(rows), len(PROFILE_COLUMNS),
                             figsize=(DOUBLE, 3.6), sharex=True)
    for col, (index, name, fmt) in enumerate(PROFILE_COLUMNS):
        values = labels[:, index]
        cmap, norm, lo, hi = _label_colors(values)
        order = np.argsort(values)
        pick = order[np.linspace(0, len(order) - 1, 12).astype(int)]

        cax = fig.add_axes([0.085 + col * 0.313, 0.888, 0.205, 0.022])
        bar = matplotlib.colorbar.ColorbarBase(
            cax, cmap=cmap, norm=norm, orientation="horizontal")
        bar.outline.set_linewidth(0.6)
        bar.set_ticks([lo, hi])
        bar.ax.set_xticklabels([fmt.format(lo), fmt.format(hi)], fontsize=6.2)
        bar.ax.tick_params(length=0, pad=1.5)
        cax.set_title(name, fontsize=7, pad=3.0, color=INK)

        for row, (key, ylabel, transform, ylim, logarithmic) in enumerate(rows):
            ax = axes[row, col]
            for star in pick:
                color = cmap(norm(values[star]))
                ax.plot(tau, transform(truth[key][star]),
                        color=color, linewidth=2.2,
                        linestyle=(0, (3.0, 1.5)), alpha=0.55, zorder=2)
                ax.plot(tau, transform(pred[key][star]),
                        color=color, linewidth=0.6, zorder=3)
                ax.plot(
                    tau, transform(physical[key][star]),
                    color=color, linewidth=0.8, linestyle=(0, (1.0, 1.25)),
                    alpha=0.9, zorder=4,
                )
            ax.set_xscale("log")
            if logarithmic:
                ax.set_yscale("log")
            ax.set_ylim(*ylim)
            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
            _inward(ax)

    for ax in axes[-1]:
        ax.set_xlabel(r"$\tau_{\rm std}$")

    handles = [
        plt.Line2D([], [], color=INK_SECONDARY, linewidth=2.2, alpha=0.55,
                   linestyle=(0, (3.0, 1.5))),
        plt.Line2D([], [], color=INK_SECONDARY, linewidth=0.9),
        plt.Line2D([], [], color=INK_SECONDARY, linewidth=0.9,
                   linestyle=(0, (1.0, 1.25))),
    ]
    _top_legend(fig, handles,
                ["reference atmosphere", r"learned $(m,T)$ initializer",
                 "hydrostatic grey–convective initializer"],
                ncol=3, y=1.005)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.845, bottom=0.105,
                        hspace=0.10, wspace=0.07)
    _save(fig, "fig_profiles.pdf")


# --------------------------------------------------------------------------
# Fig. 2b -- how well the four fields come back from exact (m,T)
# --------------------------------------------------------------------------


def figure_reconstruction() -> None:
    with np.load(
        PHYSICAL_RESULTS / "parity" / "reconstruction_metrics.npz",
        allow_pickle=False,
    ) as data:
        tau = data["tau_std"]
        errors = {key: data[f"{key}_relative_error"] for key, _ in FIELDS}

    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    for (key, label), color in zip(FIELDS, FIELD_COLORS):
        err = errors[key]
        ax.plot(tau, np.median(err, axis=0), color=color, label=label, zorder=3)
        ax.fill_between(tau, np.percentile(err, 25, axis=0),
                        np.percentile(err, 90, axis=0),
                        color=color, alpha=0.14, linewidth=0, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Rosseland optical depth $\tau_{\rm std}$")
    ax.set_ylabel("relative error vs. reference")
    ax.set_ylim(1e-5, 2e-1)
    _inward(ax)
    _top_legend(fig, *ax.get_legend_handles_labels(), ncol=4, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, "fig_reconstruction.pdf")


# --------------------------------------------------------------------------
# Fig. 3 -- what the acceptance bar looks like on real spectra
# --------------------------------------------------------------------------


_arm_spectrum = data.arm_spectrum


def figure_spectra() -> None:
    """Converged spectra for the exact-(m,T) representation test.

    The manuscript gates on a per-star maximum normalized-flux difference and
    then argues about its cumulative distribution.  A reader has no way to
    judge whether 5e-3 is a tight bar or a loose one without seeing it against
    a real line profile. Rows span the label support; the bottom row is the
    exact-(m,T) parity star with the largest normalized-flux difference.
    """

    gate = json.loads(
        (PHYSICAL_RESULTS / "parity" / "spectral_gate_truth_mT.json").read_text()
    )
    per_star = {row["slug"]: row for row in gate["per_star"]}
    records = _load_records(
        PHYSICAL_RUNS / "parity" / "records" / "reduced_state_reconstruction"
        / "records.jsonl"
    )

    fig, axes = plt.subplots(len(SPECTRA_ROWS), 3, figsize=(DOUBLE, 6.2),
                             gridspec_kw={"width_ratios": [1.55, 1.0, 1.30]})

    for row, (kind, slug, feature, window) in enumerate(SPECTRA_ROWS):
        full_truth = _arm_spectrum(
            "full_truth_oracle",
            slug,
            root=PHYSICAL_RUNS / "parity" / "spectra",
        )
        reduced = _arm_spectrum(
            "reduced_state_reconstruction",
            slug,
            root=PHYSICAL_RUNS / "parity" / "spectra",
        )
        wavelength = full_truth["wavelength_nm"]
        delta = reduced["normalized_flux"] - full_truth["normalized_flux"]
        labels = records[slug]["labels"]

        # (1) the whole gated window
        ax = axes[row, 0]
        ax.plot(wavelength, full_truth["normalized_flux"], color=INK,
                linewidth=0.30, zorder=2)
        ax.plot(wavelength, reduced["normalized_flux"], color=LEARNED,
                linewidth=0.30, linestyle=(0, (2.2, 1.1)), zorder=3)
        ax.set_xlim(wavelength[0], wavelength[-1])
        ax.set_ylim(-0.03, 1.09)
        ax.set_ylabel(f"{kind}\n" + r"$f_{\rm norm}$", linespacing=1.6)
        ax.set_title(
            r"$T_{\rm eff}=%.0f$ K,  $\log g=%.2f$,  $[\mathrm{M}/\mathrm{H}]=%+.2f$"
            % (labels["effective_temperature"],
               labels["log_surface_gravity"], labels["metallicity"]),
            loc="left", fontsize=6.6, pad=3, color=INK_SECONDARY)
        _inward(ax)
        # Mark where the zoom is taken from.
        ax.axvspan(window[0], window[1], color=ORACLE, alpha=0.16,
                   linewidth=0, zorder=1)

        # (2) one diagnostic feature, resolved
        ax = axes[row, 1]
        cut = (wavelength >= window[0]) & (wavelength <= window[1])
        ax.plot(wavelength[cut], full_truth["normalized_flux"][cut], color=INK,
                linewidth=0.75, zorder=2)
        ax.plot(wavelength[cut], reduced["normalized_flux"][cut],
                color=LEARNED, linewidth=0.75, linestyle=(0, (2.4, 1.2)),
                zorder=3)
        ax.set_xlim(*window)
        ax.set_title(feature, fontsize=7.5, pad=3)
        _inward(ax)

        # (3) the difference the gate actually measures
        ax = axes[row, 2]
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.6, zorder=1)
        ax.plot(wavelength, 1e3 * delta, color=LEARNED, linewidth=0.30,
                zorder=3)
        for sign in (-1.0, 1.0):
            ax.axhline(sign * 1e3 * BAR, color=CRITICAL, linewidth=0.9,
                       linestyle=(0, (3.5, 2.0)), zorder=4)
        ax.set_xlim(wavelength[0], wavelength[-1])
        ax.set_ylim(-9.5, 9.5)
        ax.set_ylabel(r"$10^3\,\Delta f_{\rm norm}$")
        peak = per_star[slug]["normalized_flux"]["max"]
        ax.text(0.985, 0.93,
                r"$\max|\Delta f|=%.2f\times10^{-3}$" % (1e3 * peak),
                transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
                color=CRITICAL if peak > BAR else INK_SECONDARY)
        _inward(ax)

        if row < len(SPECTRA_ROWS) - 1:
            for ax in axes[row]:
                ax.tick_params(labelbottom=False)

    for ax in axes[-1]:
        ax.set_xlabel("wavelength (nm)")

    handles = [
        plt.Line2D([], [], color=INK, linewidth=1.0),
        plt.Line2D([], [], color=LEARNED, linewidth=1.0,
                   linestyle=(0, (2.4, 1.2))),
        plt.Line2D([], [], color=CRITICAL, linewidth=1.0,
                   linestyle=(0, (3.5, 2.0))),
    ]
    _top_legend(fig, handles,
                ["reference atmosphere", r"exact $(m,T)$ + physics",
                 r"$5\times10^{-3}$ reference"], ncol=3, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    _save(fig, "fig_spectra.pdf")


# --------------------------------------------------------------------------
# Fig. 4 -- neither the representation nor the grid is the bottleneck
# --------------------------------------------------------------------------


def _grid_ticks(ax, values) -> None:
    """Label the five sampled grids, and nothing else -- the decade minor ticks
    that a log axis draws by default collide at this width."""

    ax.set_xticks(values)
    ax.set_xticklabels([str(int(v)) for v in values])
    ax.set_xticks([], minor=True)
    ax.set_xlim(values.min() * 0.78, values.max() * 1.28)


def figure_resolution() -> None:
    resolution = json.loads(
        (
            PHYSICAL_RESULTS
            / "depth_resolution"
            / "convergence_metrics_depth_resolution.json"
        ).read_text())
    continuity = json.loads((REPO / "runs" / "continuity" / "summary.json").read_text())

    keys = sorted(resolution["by_resolution"], key=int)
    n_points = np.array([int(k) for k in keys], float)
    repr_err = np.array([
        resolution["by_resolution"][k]["representation_error"]
        ["column_mass_relative_error"]["median"] for k in keys])
    iters = np.array([
        resolution["by_resolution"][k]["convergence"]
        ["converging_trial_iterations"]["mean"] for k in keys])
    nonmono = np.array([
        resolution["by_resolution"][k]["convergence"]["contraction"]
        ["non_monotonic_fraction"] for k in keys])

    scan = continuity["refinement_scan"]["by_refinement"]
    ref_keys = sorted(scan, key=int)
    refine = np.array([int(k) for k in ref_keys], float)
    closure = {
        "median": np.array([scan[k]["top_layer_residual_dex"]["median"] for k in ref_keys]),
        "$p_{99}$": np.array([scan[k]["top_layer_residual_dex"]["p99"] for k in ref_keys]),
        "max": np.array([scan[k]["top_layer_residual_dex"]["max"] for k in ref_keys]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.4))

    # N = 80 is the production grid, so the round trip is the identity and its
    # error is at float64 round-off.  Plotting it on the same line would put a
    # spike at 1e-16 in the middle of a panel whose point is the trend either
    # side of it; it is drawn as a clipped open marker instead.
    resampled = n_points != 80
    floor = 4e-10

    ax = axes[0, 0]
    ax.plot(n_points[resampled], repr_err[resampled], "o-", color=INK,
            markersize=3.5, zorder=3)
    ax.plot([80], [floor], "v", markerfacecolor="white", markeredgecolor=INK,
            markersize=4.5, zorder=3)
    ax.annotate("identity", (80, floor), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=6.0, color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(floor / 2.2, 3e-4)
    _grid_ticks(ax, n_points)
    ax.set_xlabel("intermediate grid points $N$")
    ax.set_ylabel("representation error")
    ax.set_title("(a) what the round trip costs", loc="left")
    _inward(ax)

    ax = axes[0, 1]
    ax.plot(n_points, iters, "o-", color=LEARNED, markersize=3.5,
            label="mean iterations", zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(2.6, 4.2)
    _grid_ticks(ax, n_points)
    ax.set_xlabel("intermediate grid points $N$")
    ax.set_ylabel("mean iterations", color=LEARNED)
    twin = ax.twinx()
    twin.plot(n_points, 100.0 * nonmono, "s--", color=PRODUCTION, markersize=3.2,
              zorder=3)
    twin.set_ylabel("non-monotonic (%)", color=PRODUCTION)
    twin.set_ylim(0, 40)
    twin.tick_params(colors=PRODUCTION, direction="in")
    ax.set_title("(b) what the solver notices", loc="left")
    _inward(ax)
    ax.tick_params(axis="y", colors=LEARNED)
    twin.grid(False)

    ax = axes[1, 0]
    for (label, values), marker in zip(closure.items(), ("o", "s", "^")):
        ax.plot(refine, values, marker=marker, linestyle="-", markersize=3.2,
                color=QUANTILE_COLORS[label], label=label, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(refine)
    ax.set_xticklabels([rf"$\times{int(r)}$" for r in refine])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("quadrature refinement")
    ax.set_ylabel("closure residual (dex)")
    ax.set_title("(c) and what refining it does not", loc="left")
    _inward(ax)
    ax.legend(loc="upper right", fontsize=6.4, frameon=False)

    # (d) the whole-corpus seed survey: median iterations per residual bin.
    # The bins are the harness's own, from the 52,199-star survey in the same
    # summary file; this is the binned view behind the tail numbers quoted in
    # the text, not a new experiment.
    bins = continuity["seed_survey"]["iterations_by_residual_bin"]
    # The last bin is open-ended ("> 0.1 dex"); place its marker at 0.316 dex,
    # the geometric midpoint between the 0.1 dex edge and the corpus maximum.
    edges = np.array([b["range_dex"][0] for b in bins], float)
    upper = np.array([b["range_dex"][1] if b["range_dex"][1] is not None
                      else 0.316 for b in bins], float)
    centres = np.sqrt(np.maximum(edges, 1e-12) * upper)
    centres[0] = upper[0] / 4.0  # the first bin starts at zero; park it
    med = np.array([b["iterations_median"] for b in bins], float)
    counts = np.array([b["count"] for b in bins], float)

    ax = axes[1, 1]
    ax.plot(centres, med, "o-", color=INK, markersize=3.6, zorder=3)
    for x, y, c in zip(centres, med, counts):
        ax.annotate(f"{int(c):,}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=5.8,
                    color=INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlabel("top-boundary seed residual (dex)")
    ax.set_ylabel("median iterations")
    ax.set_ylim(0, max(med) * 1.45)
    ax.set_title("(d) the seed tail is where the cost is", loc="left")
    _inward(ax)

    fig.tight_layout(h_pad=1.5)
    _save(fig, "fig_resolution.pdf")


# --------------------------------------------------------------------------
# Fig. 4 -- the learned two-field start on the real solver
# --------------------------------------------------------------------------


_iteration_counts = data.iteration_counts
_residual_traces = data.residual_traces


def _violin(ax, position, values, color, *, width=0.72):
    """Paper I Fig. 6 stopping distribution: a violin, a median bar, nothing
    else.  Iteration counts are small integers, so the kernel is widened to
    keep the shape readable rather than spiky."""

    body = ax.violinplot([values], positions=[position], widths=width,
                         showextrema=False, showmedians=False,
                         bw_method=0.45)
    for part in body["bodies"]:
        part.set_facecolor(color)
        part.set_alpha(0.30)
        part.set_edgecolor(color)
        part.set_linewidth(0.7)
    median = float(np.median(values))
    ax.hlines(median, position - 0.30, position + 0.30, color=color,
              linewidth=2.4, zorder=4)
    return median


def figure_convergence() -> None:
    learned_path = (
        PHYSICAL_RUNS / "learned" / "records" / "learned_reduced_state"
        / "records.jsonl"
    )
    production_path = RUNS / "production_six_field" / "records.jsonl"
    learned = _iteration_counts(learned_path)
    production = _iteration_counts(production_path)
    physical_rows = [
        json.loads(line)
        for line in (
            GREY_CONVECTIVE_RUNS / "development" / "records.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.6),
                             gridspec_kw={"width_ratios": [1.12, 1.0, 1.14]})

    # (a) contraction, as the stopping criterion measures it
    ax = axes[0]
    physical_traces = []
    physical_nonfinite_onsets = []
    for row in physical_rows:
        diagnostics = row.get("diagnostics") or {}
        timings = diagnostics.get("iteration_timings") or []
        trace = np.asarray(
            [
                timing["deep_layer_relative_temperature_change"]
                for timing in timings
            ],
            dtype=float,
        )
        if trace.size:
            physical_traces.append(trace)
            nonfinite = np.flatnonzero(~np.isfinite(trace))
            if nonfinite.size:
                physical_nonfinite_onsets.append(int(nonfinite[0] + 1))
            ax.plot(
                np.arange(1, trace.size + 1),
                np.where(np.isfinite(trace), np.maximum(trace, 1.0e-12), np.nan),
                color="#2f7d63", linewidth=0.32, alpha=0.18, zorder=1,
            )
    for traces, color in (
        (_residual_traces(production_path), PRODUCTION),
        (_residual_traces(learned_path), LEARNED),
        (physical_traces, "#2f7d63"),
    ):
        for trace in traces:
            if color != "#2f7d63":
                ax.plot(np.arange(1, len(trace) + 1), trace, color=color,
                        linewidth=0.35, alpha=0.12, zorder=2)
        longest = max(len(t) for t in traces)
        median, support = [], 0
        for k in range(min(longest, 30)):
            if color == "#2f7d63":
                values = []
                for trace in traces:
                    finite = np.flatnonzero(np.isfinite(trace))
                    if finite.size == 0:
                        continue
                    available = finite[finite <= k]
                    index = int(available[-1]) if available.size else int(finite[0])
                    values.append(trace[index])
                finite_alive = np.asarray(values, dtype=float)
            else:
                alive = np.asarray(
                    [t[k] for t in traces if len(t) > k], dtype=float
                )
                finite_alive = alive[np.isfinite(alive)]
                if finite_alive.size < 8:
                    break
            median.append(np.median(finite_alive))
            support = k + 1
        ax.plot(np.arange(1, support + 1), median, color=color, linewidth=1.8,
                zorder=4)
    if physical_nonfinite_onsets:
        ax.scatter(
            physical_nonfinite_onsets,
            np.full(len(physical_nonfinite_onsets), 0.94),
            marker="x", s=18, linewidths=0.85, color=CRITICAL, zorder=6,
            transform=ax.get_xaxis_transform(),
            label="non-finite physical trace",
        )
    ax.axhline(5e-4, color=CRITICAL, linewidth=0.9, linestyle=(0, (3.5, 2.0)),
               zorder=5)
    ax.text(0.025, 5.8e-4, r"stop at $5\times10^{-4}$",
            transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6.2, color=CRITICAL)
    ax.set_yscale("log")
    ax.set_xlim(0.6, 30.4)
    ax.set_ylim(3e-5, 2.0)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"deep-layer residual $\epsilon_T$")
    ax.set_title("(a) complete contraction histories", loc="left")
    _inward(ax)
    if physical_nonfinite_onsets:
        ax.legend(
            loc="upper right", frameon=False, fontsize=5.8,
            handletextpad=0.35, borderaxespad=0.35,
        )

    # (b) Cumulative stopping distribution, retaining failures in the denominator.
    ax = axes[1]
    arms = [
        (
            "six-field initializer",
            list(production.values()),
            len(_load_records(production_path)),
            PRODUCTION,
            "-",
        ),
        (
            r"learned $(m,T)$",
            list(learned.values()),
            len(_load_records(learned_path)),
            LEARNED,
            "--",
        ),
        (
            "hydrostatic grey–convective",
            [
                int(row["iterations_completed"])
                for row in physical_rows
                if bool(row["converged"])
            ],
            len(physical_rows),
            "#2f7d63",
            (0, (1.0, 1.25)),
        ),
    ]
    grid = np.arange(1, 31)
    for name, values, denominator, color, linestyle in arms:
        values = np.asarray(values, dtype=int)
        fraction = np.asarray(
            [np.count_nonzero(values <= limit) / denominator for limit in grid]
        )
        ax.step(
            grid, fraction, where="post", color=color, linestyle=linestyle,
            linewidth=1.45, label=name,
        )
    ax.axvline(15, color=INK_MUTED, linewidth=0.8, linestyle=":", zorder=1)
    ax.set_xlim(1, 30)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_xlabel("solver iteration")
    ax.set_ylabel("fraction of all stars converged")
    ax.set_title("(b) stopping distribution", loc="left")
    _inward(ax)

    # (c) Physical-initializer behavior across label space.
    ax = axes[2]
    converged_rows = [row for row in physical_rows if bool(row["converged"])]
    failed_rows = [row for row in physical_rows if not bool(row["converged"])]
    scatter = ax.scatter(
        [row["effective_temperature"] for row in converged_rows],
        [row["log_surface_gravity"] for row in converged_rows],
        c=[row["iterations_completed"] for row in converged_rows],
        cmap="viridis", vmin=3, vmax=30, s=28, linewidth=0.4,
        edgecolor=INK, zorder=3,
    )
    if failed_rows:
        ax.scatter(
            [row["effective_temperature"] for row in failed_rows],
            [row["log_surface_gravity"] for row in failed_rows],
            marker="x", s=34, color=CRITICAL, linewidth=1.0, zorder=4,
            label=f"not converged ({len(failed_rows)})",
        )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"$T_{\rm eff}$ (K)")
    ax.set_ylabel(r"$\log g$")
    ax.set_title("(c) physical initializer", loc="left")
    _inward(ax)
    bar = fig.colorbar(scatter, ax=ax, pad=0.025, fraction=0.055)
    bar.set_label("iterations to convergence", fontsize=6.8)
    bar.ax.tick_params(labelsize=6.2)
    bar.outline.set_linewidth(0.6)
    if failed_rows:
        ax.legend(loc="lower left", fontsize=6.2, frameon=False)

    handles = [
        plt.Line2D([], [], color=PRODUCTION, linewidth=1.6),
        plt.Line2D([], [], color=LEARNED, linewidth=1.6),
        plt.Line2D([], [], color="#2f7d63", linewidth=1.6),
    ]
    _top_legend(fig, handles,
                ["six-field initializer", r"learned $(m,T)$",
                 "hydrostatic grey–convective"],
                ncol=3, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, "fig_convergence.pdf")


# --------------------------------------------------------------------------
# Fig. 5 -- learned emulator against the non-neural analytic formula
# --------------------------------------------------------------------------


def figure_analytic_comparison() -> None:
    """Same-star comparison of learned, fitted, and zero-fit starts."""

    path = PHYSICAL_RESULTS / "analytic" / "paper_dev60_comparison.npz"
    with np.load(path, allow_pickle=False) as payload:
        learned_t = payload["learned_temperature_relative_error"].ravel()
        analytic_t = payload["analytic_temperature_relative_error"].ravel()
        learned_m = payload["learned_column_mass_dex_error"].ravel()
        analytic_m = payload["analytic_column_mass_dex_error"].ravel()
        learned_ok = payload["learned_converged"].astype(bool)
        analytic_ok = payload["analytic_converged"].astype(bool)
        learned_iterations = payload["learned_iterations"]
        analytic_iterations = payload["analytic_iterations"]
    with np.load(
        GREY_CONVECTIVE_RESULTS / "development" / "profile_metrics.npz",
        allow_pickle=False,
    ) as payload:
        physical_t = payload["seed_temperature_error"].ravel()
        physical_m = payload["seed_column_mass_error"].ravel()
    physical_solver = json.loads(
        (GREY_CONVECTIVE_RESULTS / "development" / "solver.json").read_text()
    )
    physical_ok = np.asarray(
        [bool(row["converged"]) for row in physical_solver["records"]]
    )
    physical_iterations = np.asarray(
        [
            np.nan
            if row["iterations_completed"] is None
            else float(row["iterations_completed"])
            for row in physical_solver["records"]
        ]
    )

    fig, axes = plt.subplots(
        1, 3, figsize=(DOUBLE, 2.35),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.05]},
    )

    def cumulative(
        ax, learned_values, analytic_values, physical_values, xlabel, title
    ):
        for values, color, linestyle, label in (
            (learned_values, LEARNED, "-", r"learned $(m,T)$"),
            (analytic_values, ANALYTIC, "--", "fitted analytic"),
            (
                physical_values,
                "#2f7d63",
                (0, (1.0, 1.25)),
                "zero-fit physical",
            ),
        ):
            ordered = np.sort(np.maximum(np.asarray(values, dtype=float), 1.0e-12))
            fraction = np.arange(1, ordered.size + 1) / ordered.size
            ax.step(
                ordered, fraction, where="post", color=color,
                linestyle=linestyle, linewidth=1.45, label=label,
            )
            p95 = float(np.percentile(values, 95.0))
            ax.plot(
                p95, 0.95, marker="o", markersize=3.6,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=0.9, zorder=4,
            )
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("fraction of layer values below")
        ax.set_title(title, loc="left")
        _inward(ax)

    cumulative(
        axes[0], learned_t, analytic_t, physical_t,
        r"$|\Delta T|/T$", "(a) temperature profile",
    )
    cumulative(
        axes[1], learned_m, analytic_m, physical_m,
        r"$|\Delta\log_{10}m|$ (dex)", "(b) column-mass profile",
    )

    ax = axes[2]
    iteration_grid = np.arange(1, 61)
    for values, converged, color, linestyle, label in (
        (
            learned_iterations,
            learned_ok,
            LEARNED,
            "-",
            r"learned $(m,T)$",
        ),
        (
            analytic_iterations,
            analytic_ok,
            ANALYTIC,
            "--",
            "fitted analytic",
        ),
        (
            physical_iterations,
            physical_ok,
            "#2f7d63",
            (0, (1.0, 1.25)),
            "zero-fit physical",
        ),
    ):
        fraction = np.asarray(
            [np.count_nonzero(converged & (values <= limit)) / len(values)
             for limit in iteration_grid]
        )
        ax.step(
            iteration_grid, fraction, where="post", color=color,
            linestyle=linestyle, linewidth=1.45, label=label,
        )
    ax.axvline(15, color=INK_MUTED, linestyle=":", linewidth=0.8)
    ax.set_xlim(1, 60)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("solver iteration")
    ax.set_ylabel("fraction of all stars converged")
    ax.set_title("(c) convergence by iteration", loc="left")
    _inward(ax)

    handles = [
        plt.Line2D([], [], color=LEARNED, linestyle="-", linewidth=1.5),
        plt.Line2D([], [], color=ANALYTIC, linestyle="--", linewidth=1.5),
        plt.Line2D([], [], color="#2f7d63",
                   linestyle=(0, (1.0, 1.25)), linewidth=1.5),
    ]
    _top_legend(
        fig, handles,
        [r"learned $(m,T)$", "fitted analytic", "zero-fit physical"],
        ncol=3, y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90], w_pad=1.3)
    _save(fig, "fig_analytic.pdf")


# --------------------------------------------------------------------------
# Fig. 6 -- zero-fit first-principles v4r6 development diagnostic
# --------------------------------------------------------------------------


def figure_v4r6() -> None:
    """Show the v4r6 comparison under a common 60-iteration setting."""

    root = RESULTS / "analytic_initializer"
    paths = {
        "decoupled": (
            root / "textbook_opacity_v4r6_decoupled_dev60_policy60_20260829.json"
        ),
        "grey": root / "textbook_opacity_v4r6_grey_dev60_policy60_20260829.json",
        "coupled convective": (
            root / "textbook_opacity_v4r6_convective_dev60_policy60_20260829.json"
        ),
    }
    payloads = {
        name: json.loads(path.read_text())
        for name, path in paths.items()
    }
    score = json.loads(
        (
            root
            / "textbook_opacity_v4r6_policy60_matched_dev60_20260829.json"
        ).read_text()
    )

    def split_counts(payload):
        records = payload["records"]
        groups = [
            records,
            [
                row for row in records
                if float(row["effective_temperature"]) < 7500.0
            ],
            [
                row for row in records
                if float(row["effective_temperature"]) >= 7500.0
            ],
        ]
        return [
            (
                sum(bool(row.get("converged")) for row in group),
                len(group),
            )
            for group in groups
        ]

    colors = {
        "decoupled": "#2f7d63",
        "grey": "#aaa9a2",
        "coupled convective": "#9a745d",
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(DOUBLE, 2.55),
        gridspec_kw={"width_ratios": [1.2, 0.92, 0.82]},
    )

    ax = axes[0]
    groups = ["all", r"cool, $T_{\rm eff}<7500$ K", "hot"]
    centres = np.arange(len(groups), dtype=float)
    width = 0.24
    offsets = np.linspace(-width, width, len(paths))
    for offset, (name, payload) in zip(offsets, payloads.items()):
        counts = split_counts(payload)
        fraction = np.array([100.0 * n / d for n, d in counts])
        bars = ax.bar(
            centres + offset, fraction, width=width * 0.92,
            color=colors[name], edgecolor="white", linewidth=0.45,
            label=name, zorder=3,
        )
        for bar, (number, denominator) in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.4,
                f"{number}/{denominator}",
                ha="center", va="bottom", fontsize=5.35,
                color=INK_SECONDARY, rotation=90,
            )
    ax.set_xticks(centres)
    ax.set_xticklabels(groups)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(0, 112)
    ax.set_ylabel("solver convergence (%)")
    ax.set_title("(a) development-sample convergence", loc="left")
    _inward(ax)
    ax.tick_params(axis="x", which="both", top=False)

    ax = axes[1]
    iteration_sets = [
        np.array(
            [
                int(row["iterations_completed"])
                for row in payload["records"]
                if bool(row.get("converged"))
                and row.get("iterations_completed") is not None
            ],
            dtype=int,
        )
        for payload in payloads.values()
    ]
    boxes = ax.boxplot(
        iteration_sets,
        positions=np.arange(len(iteration_sets)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 1.1},
        whiskerprops={"color": INK_SECONDARY, "linewidth": 0.7},
        capprops={"color": INK_SECONDARY, "linewidth": 0.7},
    )
    for patch, name in zip(boxes["boxes"], payloads):
        patch.set_facecolor(colors[name])
        patch.set_edgecolor("white")
        patch.set_linewidth(0.55)
    rng = np.random.default_rng(446)
    for position, (name, values) in enumerate(zip(payloads, iteration_sets)):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(
            position + jitter, values, s=5.0, color=colors[name],
            edgecolor="white", linewidth=0.2, alpha=0.65, zorder=2,
        )
        ax.text(
            position, 56.5, f"mean\n{values.mean():.2f}",
            ha="center", va="top", fontsize=5.8, color=INK_SECONDARY,
        )
    ax.axhline(
        15.5, color=CRITICAL, linewidth=1.0,
        linestyle=(0, (3.5, 2.0)), zorder=4,
    )
    ax.text(
        2.35, 16.5, "15 iterations",
        ha="right", va="bottom", fontsize=5.8, color=CRITICAL,
    )
    ax.set_xticks(np.arange(len(iteration_sets)))
    ax.set_xticklabels(
        ["grey–convective", "grey", "coupled convective"], rotation=20
    )
    ax.set_ylim(0, 60)
    ax.set_ylabel("iterations to convergence")
    ax.set_title("(b) matched iteration cost", loc="left")
    _inward(ax)
    ax.tick_params(axis="x", which="both", top=False)

    ax = axes[2]
    paired = score["gate"]["paired_vs_grey"]
    observed = np.array(
        [
            paired["all"]["net_gain"],
            paired["cool"]["net_gain"],
            paired["hot"]["net_gain"],
        ],
        dtype=float,
    )
    y = np.arange(3)
    ax.axvline(0, color=INK_MUTED, linewidth=0.8, zorder=1)
    for idx, value in enumerate(observed):
        point_color = colors["decoupled"] if value >= 0 else INK_MUTED
        ax.plot(
            [0, value], [idx, idx], color=point_color,
            linewidth=1.4, zorder=2,
        )
        ax.scatter(
            value, idx, s=28, color=point_color, edgecolor="white",
            linewidth=0.55, zorder=4,
        )
        ax.text(
            value + (0.25 if value >= 0 else -0.25), idx,
            f"{value:+.0f}", ha="left" if value >= 0 else "right",
            va="center", fontsize=6.2, color=point_color,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(["all", "cool", "hot"])
    ax.set_ylim(2.35, -0.35)
    ax.set_xlim(-2.8, 5.2)
    ax.set_xticks([-2, 0, 2, 4])
    ax.set_xlabel("paired net gain vs grey")
    ax.set_title("(c) paired gain relative to grey", loc="left")
    _inward(ax)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[name], edgecolor="none")
        for name in paths
    ]
    _top_legend(
        fig, handles,
        ["grey–convective", "grey", "coupled convective"],
        ncol=3, y=1.04, fontsize=6.4,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88], w_pad=1.25)
    _save(fig, "fig_v4r6.pdf")


# --------------------------------------------------------------------------
# Fig. 7 -- two predicted fields beat six, on the four they both supply
# --------------------------------------------------------------------------


ALL_FIELDS = [
    ("column_mass", "$m$"),
    ("temperature", "$T$"),
] + FIELDS


def figure_fields() -> None:
    """Separate seed accuracy from converged six-field atmosphere accuracy."""

    def star_medians(values: np.ndarray) -> np.ndarray:
        return np.median(np.asarray(values, dtype=float), axis=1)

    def learned_seed(path: Path) -> dict[str, np.ndarray]:
        with np.load(path, allow_pickle=False) as payload:
            mass = np.abs(
                np.log10(np.maximum(payload["column_mass"], 1.0e-300))
                - np.log10(np.maximum(payload["truth_column_mass"], 1.0e-300))
            )
            temperature = np.abs(
                payload["temperature"] - payload["truth_temperature"]
            ) / np.maximum(np.abs(payload["truth_temperature"]), 1.0e-300)
        return {
            "column_mass": star_medians(mass),
            "temperature": star_medians(temperature),
        }

    def physical_metrics(sample: str) -> tuple[dict, dict]:
        with np.load(
            GREY_CONVECTIVE_RESULTS / sample / "profile_metrics.npz",
            allow_pickle=False,
        ) as payload:
            seed = {
                "column_mass": star_medians(payload["seed_column_mass_error"]),
                "temperature": star_medians(payload["seed_temperature_error"]),
            }
            final = {
                field: star_medians(payload[f"final_{field}_error"])
                for field, _ in ALL_FIELDS
            }
        return seed, final

    learned_dev = learned_seed(
        REPO / "artifacts" / "reduced_state_emulator" / "predicted_monotone.npz"
    )
    learned_test = learned_seed(
        RESULTS
        / "solver_in_loop_k1_qualified_tail3_profile_rescue_v4"
        / "blind200"
        / "final.npz"
    )
    physical_dev, final_dev = physical_metrics("development")
    physical_test, final_test = physical_metrics("posthoc200")

    fig, axes = plt.subplots(
        1, 2, figsize=(DOUBLE, 2.7),
        gridspec_kw={"width_ratios": [0.72, 1.28]},
    )

    def draw_points(ax, series, fields):
        handles = []
        for label, values, color, marker, offset, face in series:
            x, median, low, high = [], [], [], []
            for position, (key, _) in enumerate(fields):
                sample = np.asarray(values[key], dtype=float)
                x.append(position + offset)
                median.append(np.median(sample))
                low.append(np.percentile(sample, 16))
                high.append(np.percentile(sample, 84))
            ax.vlines(x, low, high, color=color, linewidth=0.8, zorder=2)
            handles.append(
                ax.scatter(
                    x, median, s=22, marker=marker, facecolor=face,
                    edgecolor=color, linewidth=0.9, zorder=3, label=label,
                )
            )
        return handles

    seed_series = [
        ("learned, development", learned_dev, LEARNED, "o", -0.24, LEARNED),
        (
            "grey–convective, development",
            physical_dev,
            "#2f7d63",
            "s",
            -0.08,
            "#2f7d63",
        ),
        ("learned, independent test", learned_test, LEARNED, "o", 0.08, "white"),
        (
            "grey–convective, post-hoc sample",
            physical_test,
            "#2f7d63",
            "s",
            0.24,
            "white",
        ),
    ]
    handles = draw_points(axes[0], seed_series, ALL_FIELDS[:2])
    axes[0].set_yscale("log")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels([label for _, label in ALL_FIELDS[:2]])
    axes[0].set_xlim(-0.48, 1.48)
    axes[0].set_ylabel("profile error vs. reference")
    axes[0].set_title("(a) before the atmosphere solve", loc="left")
    _inward(axes[0])

    final_series = [
        (
            "grey–convective, development",
            final_dev,
            "#2f7d63",
            "s",
            -0.10,
            "#2f7d63",
        ),
        (
            "grey–convective, post-hoc sample",
            final_test,
            "#2f7d63",
            "s",
            0.10,
            "white",
        ),
    ]
    draw_points(axes[1], final_series, ALL_FIELDS)
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(len(ALL_FIELDS)))
    axes[1].set_xticklabels([label for _, label in ALL_FIELDS])
    axes[1].set_xlim(-0.45, len(ALL_FIELDS) - 0.55)
    axes[1].set_ylabel("profile error vs. reference")
    axes[1].set_title("(b) after convergence", loc="left")
    _inward(axes[1])

    _top_legend(
        fig, handles,
        [label for label, *_ in seed_series],
        ncol=2, y=1.05, fontsize=6.3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84], w_pad=1.6)
    _save(fig, "fig_fields.pdf")


# --------------------------------------------------------------------------
# Fig. 6 -- the spectral gate, and what the bar means
# --------------------------------------------------------------------------


def figure_spectral() -> None:
    sources = [
        (
            r"exact $(m,T)$ vs. reference atmosphere",
            PHYSICAL_RESULTS / "parity" / "spectral_gate_truth_mT.json",
            ORACLE,
            ":",
        ),
        (
            r"learned $(m,T)$ vs. six-field initializer",
            PHYSICAL_RESULTS / "learned" / "spectral_gate.json",
            LEARNED,
            "-",
        ),
        (
            "grey–convective, development",
            GREY_CONVECTIVE_RESULTS / "development" / "spectral_gate.json",
            "#2f7d63",
            "--",
        ),
        (
            "grey–convective, post-hoc sample",
            GREY_CONVECTIVE_RESULTS / "posthoc200" / "spectral_gate.json",
            "#2f7d63",
            (0, (1.0, 1.25)),
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))

    ax = axes[0]
    for label, path, color, linestyle in sources:
        gate = json.loads(path.read_text())
        values = np.sort(np.array(
            [row["normalized_flux"]["max"] for row in gate["per_star"]]))
        fraction = np.arange(1, values.size + 1) / values.size
        ax.step(
            values, fraction, where="post", color=color,
            linestyle=linestyle, label=label, zorder=3,
        )
    ax.axvline(BAR, color=CRITICAL, linestyle="--", linewidth=1.1, zorder=4)
    ax.text(BAR * 1.10, 0.42, r"$5\times10^{-3}$ reference", color=CRITICAL,
            fontsize=6.5, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel(r"per-star max $|\Delta|$, normalized flux")
    ax.set_ylabel("fraction of stars below")
    ax.set_ylim(0, 1.02)
    ax.set_title("(a) cumulative spectral discrepancies", loc="left")
    _inward(ax)
    _top_legend(fig, *ax.get_legend_handles_labels(), ncol=2, y=1.08, fontsize=6.8)

    ax = axes[1]
    comparisons = [
        (
            "development worst",
            "development",
            PHYSICAL_RUNS / "learned" / "spectra" / "production_six_field",
            ORACLE,
        ),
        (
            "post-hoc worst",
            "posthoc200",
            (
                RUNS
                / "solver_in_loop_k1_qualified_tail3_profile_rescue_v4"
                / "blind200"
                / "spectra"
                / "production_six_field"
            ),
            CRITICAL,
        ),
    ]
    for label, sample, reference_root, color in comparisons:
        gate = json.loads(
            (GREY_CONVECTIVE_RESULTS / sample / "spectral_gate.json").read_text()
        )
        row = max(
            gate["per_star"], key=lambda item: item["normalized_flux"]["max"]
        )
        slug = row["slug"]
        with np.load(
            GREY_CONVECTIVE_RUNS
            / sample
            / "spectra"
            / GREY_CONVECTIVE_ARM
            / f"{slug}.npz",
            allow_pickle=False,
        ) as physical, np.load(
            reference_root / f"{slug}.npz", allow_pickle=False
        ) as reference:
            wavelength = physical["wavelength_nm"]
            delta = np.abs(
                physical["normalized_flux"] - reference["normalized_flux"]
            )
        centres, peak = _binned_max(wavelength, delta)
        ax.plot(centres, peak, color=color, linewidth=0.9, zorder=3)
        ax.text(
            0.98,
            0.95 if sample == "development" else 0.86,
            f"{label}: {row['normalized_flux']['max']:.2e}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.3, color=color,
        )
    ax.axhline(BAR, color=CRITICAL, linestyle="--", linewidth=1.1, zorder=4)
    ax.set_yscale("log")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel(r"max $|\Delta|$ per bin")
    ax.set_ylim(1e-6, 4e-2)
    ax.set_xlim(395, 905)
    ax.set_title("(b) where the difference sits", loc="left")
    _inward(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.84])
    _save(fig, "fig_spectral.pdf")


# --------------------------------------------------------------------------
# Fig. 7 -- independent test
# --------------------------------------------------------------------------


def _figure_blind_legacy() -> None:
    cand = json.loads(
        (BLIND / "blind200_physical_seed"
         / "convergence_metrics_learned_monotone.json").read_text()
    )["learned_reduced_state"]
    physical_summary = json.loads(
        (BLIND / "blind200_physical_seed" / "summary.json").read_text()
    )
    prod = json.loads(
        (BLIND / "blind200" / "production_solver"
         / "convergence_metrics_production_baseline.json").read_text()
    )["production_six_field"]
    gate_blind = json.loads(
        (BLIND / "blind200_physical_seed" / "spectral_gate.json").read_text())
    gate_dev = json.loads((BLIND / "dev60_solver" / "spectral_gate.json").read_text())
    assert gate_blind["bar"] == gate_dev["bar"] == BAR
    audit = np.load(BLIND / "blind200" / "final.npz")
    dev = np.load(BLIND / "dev60.npz")
    cand_records = _load_records(
        REPO / "runs" / "reduced_state_emulator"
        / "solver_in_loop_k1_qualified_tail3_profile_rescue_v4"
        / "blind200_physical_seed" / "records"
        / "learned_reduced_state" / "records.jsonl")

    def histogram(arm):
        h = arm["converging_trial_iterations"]["histogram"]
        ks = sorted(int(k) for k in h)
        return np.array(ks, float), np.array([h[str(k)] for k in ks], float)

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.6))

    ax = axes[0, 0]
    kp, vp = histogram(prod)
    kc, vc = histogram(cand)
    ax.bar(kp - 0.19, vp, width=0.38, color=PRODUCTION, alpha=0.75,
           label="six-field initializer", zorder=3)
    ax.bar(kc + 0.19, vc, width=0.38, color=LEARNED, alpha=0.85,
           label="two-field initializer", zorder=3)
    lo = min(kp.min(), kc.min()) - 0.7
    hi = max(kp.max(), kc.max()) + 0.7
    ax.set_xlim(lo, hi)
    ax.set_xticks(range(int(np.ceil(lo)), int(hi) + 1, 2))
    ax.set_xlabel("iterations to convergence, 200-star test sample")
    ax.set_ylabel("stars")
    ax.axvline(3, color=INK_MUTED, linestyle=":", linewidth=0.9, zorder=2)
    ax.text(0.97, 0.74,
            f"two-field: {cand['converged_count']}/{cand['star_count']} "
            f"converged; "
            f"{physical_summary['reconstruction']['failure_count']} "
            f"reconstruction failures\n"
            f"six-field: {prod['converged_count']}/{prod['star_count']} "
            f"converged",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
            color=INK_SECONDARY, linespacing=1.5)
    ax.set_title("(a) iteration distributions", loc="left")
    ax.legend(loc="upper left", fontsize=6.4, frameon=False)
    _inward(ax)

    ax = axes[0, 1]
    for gate, color, linestyle, label in (
            (gate_dev, ORACLE, "--", "development sample (60 stars)"),
            (gate_blind, LEARNED, "-",
             f"independent test "
             f"({gate_blind['gated_star_count']} stars)")):
        values = np.sort(np.array(
            [row["normalized_flux"]["max"] for row in gate["per_star"]]))
        fraction = np.arange(1, values.size + 1) / values.size
        ax.step(values, fraction, where="post", color=color, linestyle=linestyle,
                label=label, zorder=3)
    ax.axvline(BAR, color=CRITICAL, linestyle="--", linewidth=1.1, zorder=4)
    ax.text(BAR * 1.10, 0.42, r"$5\times10^{-3}$ reference", color=CRITICAL,
            fontsize=6.5, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel(r"per-star max $|\Delta|$, normalized flux")
    ax.set_ylabel("fraction of stars below")
    ax.set_ylim(0, 1.02)
    ax.set_title("(b) normalized-flux discrepancies", loc="left")
    ax.legend(loc="lower right", fontsize=6.4, frameon=False)
    _inward(ax)

    # (c) the two samples in the (T_eff, log g) plane.
    ax = axes[1, 0]
    ax.scatter(dev["labels"][:, 0], dev["labels"][:, 1], s=14, color=ORACLE,
               alpha=0.9, linewidth=0, zorder=3, label="development 60")
    ax.scatter(audit["labels"][:, 0], audit["labels"][:, 1], s=14, color=INK,
               alpha=0.55, linewidth=0, zorder=2, label="independent test 200")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"$T_{\rm eff}$ (K)")
    ax.set_ylabel(r"$\log g$")
    ax.set_title("(c) development and test samples", loc="left")
    ax.legend(loc="lower left", fontsize=6.4, frameon=False)
    _inward(ax)

    # (d) which gate each sealed star fails.  Profile blow-outs are recomputed
    # from the prediction npz with the gate's own thresholds; spectral
    # exceedances come from the spectral gate's per-star table; solver failures
    # come from the candidate's own records.  The npz carries no slugs, so they
    # are rebuilt from the labels with the solver's public slug format
    # (bench/labels.py); the rebuild is checked against the gate table before
    # anything is drawn.  A star failing more than one gate takes the
    # highest-severity marker it qualifies for, in the order profile, spectral,
    # solver.
    def _slug(row):
        return (f"t{row[0]:07.1f}_g{row[1]:+05.2f}_m{row[2]:+05.2f}"
                f"_a{row[3]:+05.2f}_x{row[4]:04.2f}")

    labels = audit["labels"]
    audit_slugs = np.array([_slug(row) for row in labels])
    truth_t = audit["truth_temperature"]
    pred_t = audit["temperature"]
    truth_m = audit["truth_column_mass"]
    pred_m = audit["column_mass"]
    t_err = np.abs(pred_t / truth_t - 1.0).max(axis=1)
    m_dex = np.abs(np.log10(np.maximum(pred_m, 1e-300))
                   - np.log10(np.maximum(truth_m, 1e-300))).max(axis=1)
    prof_fail = (t_err > 0.10) | (m_dex > 0.10)

    spec_over = {row["slug"] for row in gate_blind["per_star"]
                 if row["normalized_flux"]["max"] > BAR}
    solver_fail_slugs = {s for s, row in cand_records.items()
                         if not row["converged"]}
    reconstruction_fail_slugs = set(audit_slugs) - set(cand_records)

    flags = []
    for i, slug in enumerate(audit_slugs):
        if prof_fail[i]:
            flags.append("profile")
        elif slug in spec_over:
            flags.append("spectral")
        elif slug in solver_fail_slugs or slug in reconstruction_fail_slugs:
            flags.append("solver")
        else:
            flags.append("pass")
    flags = np.array(flags)

    gate_slugs = {row["slug"] for row in gate_blind["per_star"]}
    matched = sum(s in gate_slugs or s in cand_records for s in audit_slugs)
    # Two of the 200 sealed stars (reconstruction ids 19807/49580) failed to
    # synchronize in the physical-seed rerun and carry no records or gate row,
    # so the rebuild is checked against that tolerance rather than a fixed 190.
    if matched < len(audit_slugs) - 2:
        raise SystemExit(
            f"slug rebuild matched only {matched}/{len(audit_slugs)} sealed stars")

    ax = axes[1, 1]
    style_map = [("pass", "within all criteria", INK, 8, 0.35, "o"),
                 ("spectral", "spectral discrepancy", LEARNED, 20, 0.9, "s"),
                 ("solver", "solver/reconstruction failure",
                  INK_MUTED, 22, 0.95, "x"),
                 ("profile", "profile discrepancy", CRITICAL, 24, 0.95, "^")]
    for key, label, color, size, alpha, marker in style_map:
        mask = flags == key
        if mask.any():
            ax.scatter(labels[mask, 2], labels[mask, 1], s=size, color=color,
                       alpha=alpha, marker=marker, linewidth=0.5, zorder=3,
                       label=f"{label} ({int(mask.sum())})")
    ax.invert_yaxis()
    ax.set_xlabel(r"$[\mathrm{M}/\mathrm{H}]$")
    ax.set_ylabel(r"$\log g$")
    ax.set_title("(d) discrepancies across the test sample", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.0,
              frameon=False, borderaxespad=0.0)
    _inward(ax)

    fig.tight_layout(h_pad=1.4)
    _save(fig, "fig_blind.pdf")


def figure_blind() -> None:
    """Independent learned test and post-hoc physical check on the same stars."""

    learned_gates = {
        "development": json.loads(
            (BLIND / "dev60_solver" / "spectral_gate.json").read_text()
        ),
        "independent": json.loads(
            (BLIND / "blind200" / "spectral_gate.json").read_text()
        ),
    }
    physical_gates = {
        sample: json.loads(
            (GREY_CONVECTIVE_RESULTS / sample / "spectral_gate.json").read_text()
        )
        for sample in ("development", "posthoc200")
    }
    if any(
        gate["bar"] != BAR
        for gate in (*learned_gates.values(), *physical_gates.values())
    ):
        raise ValueError("spectral thresholds differ across comparison arms")

    learned_records = _load_records(
        RUNS
        / "solver_in_loop_k1_qualified_tail3_profile_rescue_v4"
        / "blind200"
        / "records"
        / "learned_reduced_state"
        / "records.jsonl"
    )
    production_records = _load_records(
        RUNS
        / "solver_in_loop_k1_qualified_tail3_profile_rescue_v4"
        / "blind200"
        / "production_records"
        / "production_six_field"
        / "records.jsonl"
    )
    physical_solver = json.loads(
        (GREY_CONVECTIVE_RESULTS / "posthoc200" / "solver.json").read_text()
    )
    with np.load(BLIND / "blind200" / "final.npz", allow_pickle=False) as payload:
        audit = {key: payload[key] for key in payload.files}
    with np.load(
        GREY_CONVECTIVE_RESULTS / "posthoc200" / "profile_metrics.npz",
        allow_pickle=False,
    ) as payload:
        if not np.array_equal(payload["corpus_indices"], audit["star_indices"]):
            raise ValueError("learned and physical 200-star samples are not aligned")
        physical_seed_m = payload["seed_column_mass_error"]
        physical_seed_t = payload["seed_temperature_error"]

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.6))

    # All 200 stars remain in the denominator, including solver failures.
    ax = axes[0, 0]
    physical_iterations = np.asarray(
        [
            row["iterations_completed"]
            for row in physical_solver["records"]
            if bool(row["converged"])
        ],
        dtype=int,
    )
    arms = [
        (
            "six-field, independent test",
            np.asarray(
                [
                    row["converging_trial_iterations"]
                    for row in production_records.values()
                    if bool(row["converged"])
                ]
            ),
            PRODUCTION,
            "-",
        ),
        (
            r"learned $(m,T)$, independent test",
            np.asarray(
                [
                    row["converging_trial_iterations"]
                    for row in learned_records.values()
                    if bool(row["converged"])
                ]
            ),
            LEARNED,
            "--",
        ),
        (
            "grey–convective, post-hoc sample",
            physical_iterations,
            "#2f7d63",
            (0, (1.0, 1.25)),
        ),
    ]
    iteration_grid = np.arange(1, 61)
    for label, values, color, linestyle in arms:
        fraction = np.asarray(
            [np.count_nonzero(values <= limit) / 200 for limit in iteration_grid]
        )
        ax.step(
            iteration_grid, fraction, where="post", color=color,
            linestyle=linestyle, linewidth=1.4, label=label,
        )
    ax.set_xlim(1, 60)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 15, 30, 45, 60])
    ax.set_xlabel("solver iteration")
    ax.set_ylabel("fraction of all 200 stars converged")
    ax.set_title("(a) convergence and iteration cost", loc="left")
    ax.legend(loc="lower right", fontsize=5.9, frameon=False)
    _inward(ax)

    ax = axes[0, 1]
    for gate, color, linestyle, label in (
        (learned_gates["development"], LEARNED, "--", "learned, development"),
        (
            learned_gates["independent"],
            LEARNED,
            "-",
            "learned, independent test",
        ),
        (
            physical_gates["development"],
            "#2f7d63",
            "--",
            "grey–convective, development",
        ),
        (
            physical_gates["posthoc200"],
            "#2f7d63",
            "-",
            "grey–convective, post-hoc sample",
        ),
    ):
        values = np.sort(
            np.asarray(
                [row["normalized_flux"]["max"] for row in gate["per_star"]]
            )
        )
        fraction = np.arange(1, values.size + 1) / values.size
        ax.step(
            values, fraction, where="post", color=color,
            linestyle=linestyle, linewidth=1.25, label=label,
        )
    ax.axvline(BAR, color=CRITICAL, linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(r"per-star max $|\Delta|$, normalized flux")
    ax.set_ylabel("fraction of paired spectra below")
    ax.set_title("(b) spectral differences", loc="left")
    ax.legend(loc="lower right", fontsize=5.9, frameon=False)
    _inward(ax)

    ax = axes[1, 0]
    physical_ok = [
        row for row in physical_solver["records"] if bool(row["converged"])
    ]
    physical_failed = [
        row for row in physical_solver["records"] if not bool(row["converged"])
    ]
    scatter = ax.scatter(
        [row["effective_temperature"] for row in physical_ok],
        [row["log_surface_gravity"] for row in physical_ok],
        c=[row["iterations_completed"] for row in physical_ok],
        cmap="viridis", vmin=3, vmax=60, s=18, linewidth=0.25,
        edgecolor=INK, zorder=3,
    )
    if physical_failed:
        ax.scatter(
            [row["effective_temperature"] for row in physical_failed],
            [row["log_surface_gravity"] for row in physical_failed],
            marker="x", s=28, color=CRITICAL, linewidth=0.8, zorder=4,
            label=f"not converged ({len(physical_failed)})",
        )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"$T_{\rm eff}$ (K)")
    ax.set_ylabel(r"$\log g$")
    ax.set_title("(c) physical post-hoc result", loc="left")
    bar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.05)
    bar.set_label("iterations", fontsize=6.5)
    bar.ax.tick_params(labelsize=6.0)
    if physical_failed:
        ax.legend(loc="lower left", fontsize=5.9, frameon=False)
    _inward(ax)

    ax = axes[1, 1]
    learned_m = np.abs(
        np.log10(np.maximum(audit["column_mass"], 1.0e-300))
        - np.log10(np.maximum(audit["truth_column_mass"], 1.0e-300))
    )
    learned_t = np.abs(
        audit["temperature"] - audit["truth_temperature"]
    ) / np.maximum(np.abs(audit["truth_temperature"]), 1.0e-300)
    for values, color, linestyle, label in (
        (learned_m.max(axis=1), LEARNED, "-", r"learned $m$"),
        (learned_t.max(axis=1), LEARNED, "--", r"learned $T$"),
        (
            physical_seed_m.max(axis=1),
            "#2f7d63",
            "-",
            "grey–convective $m$",
        ),
        (
            physical_seed_t.max(axis=1),
            "#2f7d63",
            "--",
            "grey–convective $T$",
        ),
    ):
        ordered = np.sort(np.maximum(values, 1.0e-12))
        fraction = np.arange(1, ordered.size + 1) / ordered.size
        ax.step(
            ordered, fraction, where="post", color=color,
            linestyle=linestyle, linewidth=1.25, label=label,
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("maximum seed-profile error")
    ax.set_ylabel("fraction of stars below")
    ax.set_title("(d) seed profiles", loc="left")
    ax.legend(loc="lower right", fontsize=5.9, frameon=False)
    _inward(ax)

    fig.tight_layout(h_pad=1.4)
    _save(fig, "fig_blind.pdf")


def _load_mstar_cases() -> list[dict]:
    path = MSTAR_RESULTS / "cases.json"
    if not path.is_file():
        return []
    with path.open() as handle:
        return json.load(handle).get("cases", [])


def _mstar_selected(case: dict) -> dict | None:
    route = case.get("selected_route")
    if route == "continuation_primary":
        return case.get("continuation_primary")
    if route == "direct":
        return case.get("direct")
    return None


def figure_mstar_solver() -> None:
    """Temperature and gravity dependence for the eight M-star atmospheres."""

    cases = _load_mstar_cases()
    fig, axes = plt.subplots(
        2, 2, figsize=(DOUBLE, 4.65), sharex="col", sharey="row"
    )
    if not cases:
        for ax in axes.flat:
            ax.axis("off")
        axes[0, 0].text(
            0.5, 0.5, "M-star results unavailable", ha="center", va="center"
        )
        _save(fig, "fig_mstar_solver.pdf")
        return

    groups = (
        ("M-dwarf", r"M dwarfs ($\log g=5.0$)", "M-dwarf"),
        ("M-giant", r"M giants ($\log g=1.5$)", "M-giant"),
    )
    series = (
        ("direct", ORACLE, "o", r"direct MARCS $(m,T)$"),
        ("continuation_primary", LEARNED, "s", "temperature sequence"),
    )

    for column, (star_class, class_label, adjective) in enumerate(groups):
        group = sorted(
            (case for case in cases if case["class"] == star_class),
            key=lambda case: float(case["labels"]["effective_temperature"]),
            reverse=True,
        )
        temperature = np.array([
            float(case["labels"]["effective_temperature"]) for case in group
        ])

        ax = axes[0, column]
        for key, color, marker, name in series:
            records = [case[key] for case in group]
            iterations = np.array([
                float(record["iterations"])
                if record.get("iterations") is not None
                else np.nan
                for record in records
            ])
            converged = np.array([
                bool(record.get("survives_solver")) and np.isfinite(value)
                for record, value in zip(records, iterations)
            ])
            failed = np.isfinite(iterations) & ~converged
            not_reached = ~np.isfinite(iterations)
            ax.scatter(
                temperature[converged],
                iterations[converged],
                color=color,
                marker=marker,
                s=25,
                label=name,
                zorder=3,
            )
            ax.scatter(
                temperature[failed],
                iterations[failed],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=29,
                linewidth=0.9,
                zorder=3,
            )
            ax.scatter(
                temperature[not_reached],
                np.full(np.count_nonzero(not_reached), 1.2),
                color=color,
                marker="x",
                s=27,
                linewidth=0.9,
                zorder=3,
            )
        ax.axhline(30, color=CRITICAL, linestyle="--", linewidth=0.75)
        ax.set_ylim(0, 32)
        ax.set_title(f"({chr(97 + column)}) {class_label}", loc="left")
        ax.set_xticks([3750, 3500, 3300, 3000])
        ax.set_xlim(3825, 2925)
        _inward(ax)

        ax = axes[1, column]
        for key, color, marker, _ in series:
            selected_temperature = []
            selected_flux = []
            for case in group:
                if case.get("selected_route") != key:
                    continue
                record = _mstar_selected(case)
                value = (
                    record.get("flux_imbalance", {}).get("max_percent")
                    if record is not None
                    else None
                )
                if value is None:
                    continue
                selected_temperature.append(
                    float(case["labels"]["effective_temperature"])
                )
                selected_flux.append(float(value))
            ax.scatter(
                selected_temperature,
                selected_flux,
                color=color,
                marker=marker,
                s=27,
                zorder=3,
            )
        ax.set_yscale("log")
        ax.set_ylim(0.7, 600)
        ax.set_xlim(3825, 2925)
        ax.set_xticks([3750, 3500, 3300, 3000])
        ax.set_xlabel(r"$T_{\rm eff}$ (K)")
        ax.set_title(
            f"({chr(99 + column)}) retained {adjective} atmospheres",
            loc="left",
        )
        _inward(ax)

    axes[0, 0].set_ylabel("iterations")
    axes[1, 0].set_ylabel(r"max $|\Delta F/F|$ (\%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    _top_legend(fig, handles, labels, ncol=2, y=1.015, fontsize=6.8)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=1.4, w_pad=1.4)
    _save(fig, "fig_mstar_solver.pdf")


def _load_h5_spectrum(path: Path) -> dict[str, np.ndarray]:
    import h5py
    with h5py.File(path, "r") as handle:
        return {key: np.asarray(handle[key], dtype=float) for key in handle.keys()}


def figure_mstar_spectra() -> None:
    """Representative same-atmosphere and independent-MARCS spectra."""

    cases = _load_mstar_cases()
    cases = [
        case for case in cases
        if case.get("korg_comparison", {}).get("status") == "ok"
    ]
    if not cases:
        fig, ax = plt.subplots(figsize=(DOUBLE, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "M-star Korg spectra unavailable", ha="center", va="center")
        _save(fig, "fig_mstar_spectra.pdf")
        return
    chosen = []
    for class_name in ("M-dwarf", "M-giant"):
        matching = [case for case in cases if case["class"] == class_name]
        if matching:
            chosen.append(next(
                (case for case in matching
                 if float(case["labels"]["effective_temperature"]) == 3300.0),
                matching[0],
            ))
    if len(chosen) == 1:
        chosen.append(chosen[0])

    fig, axes = plt.subplots(len(chosen), 2, figsize=(DOUBLE, 2.0 * len(chosen)), squeeze=False)
    for row_index, case in enumerate(chosen):
        pz = case.get("payne_zero_spectra", {})
        axis_data = {}
        for arm in ("molecular", "atomic_only"):
            pz_path = pz.get(arm, {}).get("path")
            if pz_path and Path(pz_path).is_file():
                with np.load(pz_path, allow_pickle=False) as data_npz:
                    axis_data["PZ " + arm] = (
                        np.asarray(data_npz["wavelength_nm"]),
                        np.asarray(data_npz["normalized_flux"]),
                    )
            korg = case.get("korg_comparison", {}).get("arms", {}).get(arm, {})
            for key, label in (
                ("same_atmosphere_path", "Korg same " + arm),
                ("independent_marcs_path", "Korg MARCS " + arm),
            ):
                if korg.get(key) and Path(korg[key]).is_file():
                    data_h5 = _load_h5_spectrum(Path(korg[key]))
                    axis_data[label] = (
                        data_h5["wavelength_A"] / 10.0,
                        data_h5["normalized_flux"],
                    )
        for col_index, (lo, hi) in enumerate(((500.0, 502.0), (820.0, 822.0))):
            ax = axes[row_index, col_index]
            for label, (wavelength, flux) in axis_data.items():
                mask = (wavelength >= lo) & (wavelength <= hi)
                if not np.any(mask):
                    continue
                if "MARCS" in label:
                    linestyle = "--"
                    color = ORACLE
                elif "same" in label:
                    linestyle = "-"
                    color = LEARNED
                else:
                    linestyle = ":"
                    color = INK_MUTED
                ax.plot(wavelength[mask], flux[mask], color=color, linestyle=linestyle,
                        linewidth=0.62, label=label)
            ax.set_xlim(lo, hi)
            ax.set_ylim(0.0, 1.08)
            ax.set_xlabel("wavelength (nm)")
            if col_index == 0:
                ax.set_ylabel("normalized flux")
                ax.set_title(
                    f"{case['class']} {int(float(case['labels']['effective_temperature']))} K",
                    loc="left",
                )
            if row_index == 0 and col_index == 0:
                ax.legend(loc="lower left", fontsize=4.8, frameon=False, ncol=2)
            _inward(ax)
    fig.tight_layout(h_pad=1.3, w_pad=1.4)
    _save(fig, "fig_mstar_spectra.pdf")


def main() -> None:
    figure_state()
    figure_profiles()
    figure_reconstruction()
    figure_spectra()
    figure_resolution()
    figure_convergence()
    figure_analytic_comparison()
    figure_v4r6()
    figure_fields()
    figure_spectral()
    figure_blind()
    figure_mstar_solver()


if __name__ == "__main__":
    main()
