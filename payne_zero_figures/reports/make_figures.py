"""Evidence figures for the learned reduced-state initializer (Parts 4-5).

Three figures, each answering one question a reader would actually ask:

``learned_vs_production.png``  Is the iteration improvement real, or an artifact
                              of averaging? Paired per-star, so it cannot be.
``residual_scale.png``         Why the two oracle arms are excluded from the
                              contraction comparison at all.
``spectral_gate.png``          Does the faster initializer land on the same
                              physical answer? Where, and by how much, it does not.

Colors are the first three slots of the validated categorical palette, which is
the set that clears the all-pairs CVD and normal-vision floors (worst pair
ΔE 9.2 CVD / 24.0 normal). Aqua sits below 3:1 on the light surface, so every
series is directly labelled -- identity is never carried by hue alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from payne_zero_figures import data, style  # noqa: E402
from payne_zero_figures.data import (  # noqa: E402
    binned_max as _binned_max,
    load_records as _load_records,
    metric_trace as _metric_trace,
)
from payne_zero_figures.style import BAR, recess as _recess  # noqa: E402

REPO_ROOT = data.REPO
RESULTS = data.RESULTS
FIGURES = data.FIGURES
RUNS = data.EMULATOR_RUNS

# Screen palette -- note that LEARNED/PRODUCTION are NOT the manuscript's
# colours; see the warning in payne_zero_figures.style.
_E = style.EvidencePalette
LEARNED = _E.LEARNED
PRODUCTION = _E.PRODUCTION
ORACLE = _E.ORACLE
CRITICAL = _E.CRITICAL
INK = _E.INK
INK_SECONDARY = _E.INK_SECONDARY
INK_MUTED = _E.INK_MUTED
SURFACE = _E.SURFACE

style.configure("EVIDENCE")


def figure_learned_vs_production() -> None:
    learned = _load_records(RUNS / "learned_reduced_state" / "records.jsonl")
    production = _load_records(RUNS / "production_six_field" / "records.jsonl")
    shared = sorted(set(learned) & set(production))

    both = [s for s in shared if learned[s]["converged"] and production[s]["converged"]]
    a = np.array([learned[s]["converging_trial_iterations"] for s in both], float)
    b = np.array([production[s]["converging_trial_iterations"] for s in both], float)

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))

    # (a) paired iterations. Jitter only to separate coincident integer pairs;
    # the y=x line is what the eye should read against.
    ax = axes[0]
    rng = np.random.default_rng(0)
    jitter = lambda v: v + rng.uniform(-0.16, 0.16, size=v.shape)
    limit = max(a.max(), b.max()) + 1.2
    ax.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1.0, zorder=1)
    ax.scatter(
        jitter(b), jitter(a), s=26, color=LEARNED, alpha=0.85,
        edgecolor=SURFACE, linewidth=0.8, zorder=3,
    )
    ax.set_xlim(2, limit)
    ax.set_ylim(2, limit)
    ax.set_xlabel("production six-field — iterations")
    ax.set_ylabel("learned reduced state — iterations")
    ax.set_title("(a) paired, same star", loc="left")
    fewer = int((a < b).sum())
    more = int((a > b).sum())
    ax.text(
        0.97, 0.05,
        f"below the line  {fewer}\nabove  {more}\ntied  {len(a) - fewer - more}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
        color=INK_SECONDARY, linespacing=1.5,
        bbox=dict(facecolor=SURFACE, edgecolor="none", alpha=0.85, pad=2.5),
    )
    ax.text(
        0.06, 0.94, "learned is faster\nbelow the diagonal",
        transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK_MUTED,
        style="italic", linespacing=1.4,
    )
    _recess(ax, grid_axis="both")

    # (b) iteration distribution, direct-labelled rather than legended.
    ax = axes[1]
    bins = np.arange(2.5, max(a.max(), b.max()) + 1.5)
    for values, color, label, offset in (
        (b, PRODUCTION, "production", -0.19),
        (a, LEARNED, "learned", 0.19),
    ):
        counts, _ = np.histogram(values, bins=bins)
        centers = 0.5 * (bins[:-1] + bins[1:]) + offset
        ax.bar(centers, counts, width=0.36, color=color, label=label, zorder=3)
    ax.set_xlabel("iterations to convergence")
    ax.set_ylabel("stars")
    ax.set_title("(b) distribution", loc="left")
    ax.text(0.97, 0.90, f"production  mean {b.mean():.2f}", transform=ax.transAxes,
            ha="right", color=PRODUCTION, fontsize=8, fontweight="bold")
    ax.text(0.97, 0.81, f"learned  mean {a.mean():.2f}", transform=ax.transAxes,
            ha="right", color=LEARNED, fontsize=8, fontweight="bold")
    _recess(ax)

    # (c) the cost side, stated as plainly as the win.
    ax = axes[2]
    learned_fail = sum(1 for s in shared if not learned[s]["converged"])
    production_fail = sum(1 for s in shared if not production[s]["converged"])
    ax.bar([0], [production_fail], width=0.5, color=PRODUCTION, zorder=3)
    ax.bar([1], [learned_fail], width=0.5, color=LEARNED, zorder=3)
    for x, value in ((0, production_fail), (1, learned_fail)):
        ax.text(x, value + 0.08, str(value), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["production", "learned"])
    ax.set_ylabel(f"failures (of {len(shared)})")
    ax.set_ylim(0, max(learned_fail, production_fail) + 1.6)
    # A count axis must not offer half a failure.
    ax.set_yticks(np.arange(0, max(learned_fail, production_fail) + 2))
    ax.set_title("(c) the cost", loc="left")
    ax.text(
        0.02, 0.97,
        "1 of the 3 is shared — production fails it too.\nNot resolved at n=60: Fisher p = 0.62",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        color=INK_SECONDARY, linespacing=1.6,
    )
    _recess(ax)

    fig.suptitle(
        "Learned two-field initializer vs the shipped six-field one, 60 held-out stars",
        fontsize=11, fontweight="bold", color=INK, x=0.007, ha="left", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGURES / "learned_vs_production.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def figure_residual_scale() -> None:
    """Why the oracle arms are not in panel (b) of the convergence comparison."""

    parity = json.loads(
        (RESULTS / "convergence_metrics_reduced_state_parity.json").read_text()
    )
    production = json.loads(
        (RESULTS / "convergence_metrics_production_baseline.json").read_text()
    )["production_six_field"]
    learned = json.loads(
        (RESULTS / "convergence_metrics_learned_monotone.json").read_text()
    )["learned_reduced_state"]

    arms = [
        ("full six-field truth", parity["full_truth_oracle"], ORACLE),
        ("truth (m,T) oracle", parity["reduced_state_reconstruction"], ORACLE),
        ("learned reduced state", learned, LEARNED),
        ("production six-field", production, PRODUCTION),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    ax = axes[0]
    positions = np.arange(len(arms))
    values = [arm["contraction"]["first_iteration_residual"]["p50"] for _, arm, _ in arms]
    colors = [color for _, _, color in arms]
    ax.barh(positions, values, height=0.55, color=colors, zorder=3)
    ax.axvline(5.0e-4, color=CRITICAL, linewidth=1.4, linestyle="--", zorder=4)
    ax.text(
        5.0e-4, len(arms) - 0.42, "convergence threshold 5e-4",
        color=CRITICAL, fontsize=8, va="bottom", ha="center",
    )
    ax.set_ylim(-0.6, len(arms) - 0.15)
    for y, value in zip(positions, values):
        ax.text(value * 1.12, y, f"{value:.1e}", va="center", fontsize=8, color=INK)
    ax.set_yticks(positions)
    ax.set_yticklabels([name for name, _, _ in arms])
    ax.set_xscale("log")
    ax.set_xlim(1.5e-4, 1.1e-2)
    ax.set_xlabel("median first-iteration residual, deep-layer |ΔT/T|")
    ax.set_title("(a) where each arm starts", loc="left")
    _recess(ax, grid_axis="x")

    ax = axes[1]
    q = [arm["contraction"]["q_ratio"]["geometric_mean"] for _, arm, _ in arms]
    nonmono = [arm["contraction"]["non_monotonic_fraction"] for _, arm, _ in arms]
    # Hollow markers for the two arms whose statistics are not comparable, so
    # the distinction is carried by shape as well as by the caption.
    offsets = {"full six-field truth": (0, 15), "truth (m,T) oracle": (0, -22),
               "learned reduced state": (0, 15), "production six-field": (0, -22)}
    for (name, _, color), qi, ni in zip(arms, q, nonmono):
        comparable = name in ("learned reduced state", "production six-field")
        ax.scatter(
            [qi], [ni], s=150,
            color=color if comparable else SURFACE,
            edgecolor=color, linewidth=2.0, zorder=3,
        )
        ax.annotate(
            name, (qi, ni), textcoords="offset points", xytext=offsets[name],
            ha="center", fontsize=8, color=INK if comparable else INK_SECONDARY,
        )
    ax.annotate(
        "", xy=(q[2], nonmono[2]), xytext=(q[3], nonmono[3]),
        arrowprops=dict(arrowstyle="->", color=LEARNED, linewidth=1.6,
                        shrinkA=11, shrinkB=11), zorder=2,
    )
    ax.set_xlabel("geometric-mean contraction q")
    ax.set_ylabel("non-monotonic trajectories")
    ax.set_ylim(0.06, 0.66)
    ax.set_xlim(0.515, 0.70)
    ax.set_title("(b) contraction", loc="left")
    ax.text(
        0.02, 0.97,
        "filled = comparable (matched starting residual)\n"
        "hollow = starts inside the threshold, so its q and\n"
        "oscillation measure noise rather than contraction",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
        color=INK_SECONDARY, style="italic", linespacing=1.6,
    )
    _recess(ax, grid_axis="both")

    fig.suptitle(
        "Why the oracle arms cannot be compared on contraction",
        fontsize=11, fontweight="bold", color=INK, x=0.007, ha="left", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = FIGURES / "residual_scale.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def figure_spectral_gate(
    gate_path: Path = RESULTS / "spectral_gate.json",
    spectra_root: Path = RUNS / "spectra",
    out_path: Path = FIGURES / "spectral_gate.png",
) -> None:
    gate = json.loads(gate_path.read_text())
    rows = gate["per_star"]
    fields = (
        ("normalized_flux", "normalized flux", "max |\u0394|"),
        ("flux_total", "full flux", "max |\u0394| / continuum"),
        ("flux_continuum", "continuum", "max relative \u0394"),
    )

    fig = plt.figure(figsize=(11.8, 6.9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.30)

    for column, (field, name, unit) in enumerate(fields):
        ax = fig.add_subplot(grid[0, column])
        values = np.sort(np.array([row[field]["max"] for row in rows]))
        over = values > BAR
        ax.scatter(np.arange(len(values))[~over], values[~over], s=15,
                   color=LEARNED, edgecolor=SURFACE, linewidth=0.5, zorder=3)
        if over.any():
            ax.scatter(np.arange(len(values))[over], values[over], s=46,
                       color=CRITICAL, edgecolor=SURFACE, linewidth=0.9,
                       zorder=4, marker="D")
        ax.axhline(BAR, color=CRITICAL, linewidth=1.4, linestyle="--", zorder=2)
        median = float(np.median(values))
        ax.axhline(median, color=INK_MUTED, linewidth=1.0, linestyle=":", zorder=2)
        ax.set_yscale("log")
        ax.set_ylim(5e-5, 2.2e-2)
        ax.set_xlim(-2, len(values) + 1)
        ax.text(0.02, 0.955, "gate 5e-3", transform=ax.transAxes, color=CRITICAL,
                fontsize=8, va="top")
        ax.text(0.02, 0.06, f"median {median:.1e}", transform=ax.transAxes,
                color=INK_SECONDARY, fontsize=8, va="bottom")
        ax.set_xlabel("stars, sorted")
        if column == 0:
            ax.set_ylabel("metric value")
        ax.set_title(f"{name}  ·  {unit}", loc="left", fontsize=9)
        ax.text(0.98, 0.955, f"{int(over.sum())} over", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontweight="bold",
                color=CRITICAL if over.any() else "#0ca30c")
        _recess(ax, grid_axis="y")

    # Bottom: the two exceedances, each in the metric it actually failed, plus a
    # typical star for scale. Envelope, not raw trace -- see _binned_max.
    over_normalized = [r for r in rows if r["normalized_flux"]["max"] > BAR]
    over_total = [r for r in rows if r["flux_total"]["max"] > BAR]
    typical = sorted(rows, key=lambda r: abs(r["normalized_flux"]["max"] - 1.6e-3))[0]
    panels = [
        (
            over_total[0] if over_total else max(rows, key=lambda r: r["flux_total"]["max"]),
            "flux_total",
            "full flux |\u0394| / continuum",
            bool(over_total),
        ),
        (
            over_normalized[0] if over_normalized else max(
                rows, key=lambda r: r["normalized_flux"]["max"]
            ),
            "normalized_flux",
            "normalized flux |\u0394|",
            bool(over_normalized),
        ),
        (typical, "normalized_flux", "normalized flux |\u0394|", False),
    ]

    for column, (row, field, ylabel, is_over) in enumerate(panels):
        ax = fig.add_subplot(grid[1, column])
        slug = row["slug"]
        wavelength, delta = _metric_trace(slug, field, spectra_root=spectra_root)
        centers, peak = _binned_max(wavelength, delta)
        ax.plot(centers, peak, color=LEARNED, linewidth=1.0, zorder=3)
        ax.axhline(BAR, color=CRITICAL, linewidth=1.3, linestyle="--", zorder=4)
        worst = int(np.argmax(delta))
        ax.scatter([wavelength[worst]], [delta[worst]], s=52,
                   color=CRITICAL if is_over else INK_MUTED,
                   edgecolor=SURFACE, linewidth=1.0, zorder=5,
                   marker="D" if is_over else "o")
        if is_over:
            ax.annotate(
                f"{delta[worst]:.2e}\n@ {wavelength[worst]:.0f} nm",
                (wavelength[worst], delta[worst]), textcoords="offset points",
                xytext=(0, 13), ha="right" if wavelength[worst] > 780 else "center",
                fontsize=7.5, color=CRITICAL, linespacing=1.4,
            )
        else:
            # The peak sits inside the trace here; a leader line into dense ink
            # is unreadable, so the value goes in clear space instead.
            ax.text(
                0.03, 0.90,
                f"peak {delta[worst]:.2e}\n@ {wavelength[worst]:.0f} nm",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
                color=INK_SECONDARY, linespacing=1.4,
            )
        exceeding = int((delta > BAR).sum())
        ax.set_yscale("log")
        ax.set_ylim(2e-6, 4.5e-2)
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel(ylabel, fontsize=8)
        temperature, gravity = slug.split("_")[0], slug.split("_")[1]
        ax.set_title(
            f"{temperature} {gravity}  ·  {'OVER' if is_over else 'typical'}",
            loc="left", fontsize=9, color=CRITICAL if is_over else INK,
        )
        ax.text(0.98, 0.05,
                f"{exceeding} of {len(delta)} samples over ({100*exceeding/len(delta):.3f}%)",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                color=INK_SECONDARY)
        _recess(ax, grid_axis="y")

    handles = [
        Line2D([], [], color=LEARNED, marker="o", linestyle="none", markersize=5,
               label="star within the gate"),
        Line2D([], [], color=CRITICAL, marker="D", linestyle="none", markersize=6,
               label="star over the gate"),
        Line2D([], [], color=LEARNED, linewidth=1.2,
               label="max difference per wavelength bin"),
        Line2D([], [], color=CRITICAL, linestyle="--", label="5e-3 acceptance bar"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.012))

    excluded = gate["excluded_stars"]
    fig.suptitle(
        "Spectral gate — learned vs production converged products  ·  400\u2013900 nm  ·  R=20000  ·  float64",
        fontsize=11, fontweight="bold", color=INK, x=0.006, ha="left", y=0.995,
    )
    pass_all_three = sum(
        all(row[field]["max"] <= BAR for field, _, _ in fields)
        for row in rows
    )
    fig.text(
        0.006, 0.952,
        f"{gate['gated_star_count']} pairs gated  ·  {pass_all_three} pass all three  ·  "
        f"{len(excluded)} excluded: the candidate arm produced no converged atmosphere "
        f"({', '.join(s.split('_')[0] for s in excluded)})",
        fontsize=8.5, color=INK_SECONDARY, ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.935))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    print(f"wrote {out_path}")


def figure_reduced_state_sufficiency(production_errors: Path) -> None:
    """The six-parameter half of the (m,T) sufficiency claim.

    Two ways to obtain P, n_e, kappa_R and g_rad for a star: predict all six
    fields with the shipped network, or predict only (m,T) and derive the other
    four through the certified physics. This compares them against the same
    truth atmospheres, per field and per depth. Truth (m,T) is used as the input
    to the reconstruction, so this isolates the *representation* question --
    whether (m,T) carries the information -- from the separate question of
    whether a network can predict (m,T), which Part 4 answers.
    """

    def load_error(data, field):
        if field == "radiative_acceleration":
            key = (
                f"{field}_normalized_error"
                if f"{field}_normalized_error" in data.files
                else f"{field}_relative_error"
            )
            return data[key]
        return data[f"{field}_relative_error"]

    with np.load(RESULTS / "reconstruction_metrics.npz", allow_pickle=False) as data:
        tau = data["tau_std"]
        reconstructed = {
            field: load_error(data, field)
            for field in ("gas_pressure", "electron_density",
                          "rosseland_opacity", "radiative_acceleration")
        }
    with np.load(production_errors, allow_pickle=False) as data:
        predicted = {
            field: load_error(data, field) for field in reconstructed
        }
    with np.load(
        RESULTS / "learned_reduced_state_derived_errors.npz", allow_pickle=False
    ) as data:
        learned = {field: load_error(data, field) for field in reconstructed}

    titles = {
        "gas_pressure": "gas pressure  P",
        "electron_density": "electron density  n$_e$",
        "rosseland_opacity": "Rosseland opacity  $\\kappa_R$",
        "radiative_acceleration": "radiative acceleration  g$_{rad}$",
    }

    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.6), sharey=True)
    for ax, field in zip(axes, reconstructed):
        for values, color, style in (
            (predicted[field], PRODUCTION, "-"),
            (learned[field], LEARNED, "-"),
            (reconstructed[field], ORACLE, "--"),
        ):
            median = np.median(values, axis=0)
            ax.plot(tau, median, color=color, linewidth=1.8, linestyle=style, zorder=3)
            if style == "-":
                ax.fill_between(
                    tau, np.percentile(values, 25, axis=0),
                    np.percentile(values, 75, axis=0),
                    color=color, alpha=0.14, linewidth=0, zorder=2,
                )
        ratio = np.median(predicted[field]) / np.median(learned[field])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(4e-5, 3e-1)
        ax.set_xlabel(r"$\tau_{\rm Ross}$")
        ax.set_title(titles[field], loc="left", fontsize=9.5)
        ax.text(
            0.97, 0.95, f"{ratio:.1f}× better", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, fontweight="bold", color=LEARNED,
        )
        _recess(ax, grid_axis="y")
    axes[0].set_ylabel("relative error vs truth")

    handles = [
        Line2D([], [], color=PRODUCTION, linewidth=2,
               label="shipped six-field network predicts the field directly"),
        Line2D([], [], color=LEARNED, linewidth=2,
               label="derived from the trained network's (m,T)  — the deployable path"),
        Line2D([], [], color=ORACLE, linewidth=2, linestyle="--",
               label="derived from truth (m,T)  — the oracle, i.e. the headroom left"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "(m,T) sufficiency, six-parameter dimension — median over 60 held-out stars, band = interquartile",
        fontsize=11, fontweight="bold", color=INK, x=0.006, ha="left", y=0.99,
    )
    fig.text(
        0.006, 0.925,
        "Two fields plus physics beats six predicted fields on every one of the four, "
        "and the oracle shows an order of magnitude still on the table.",
        fontsize=8.5, color=INK_SECONDARY, ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    out = FIGURES / "reduced_state_sufficiency.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")




def figure_spectral_sufficiency() -> None:
    """The spectral half of the (m,T) sufficiency claim, with its own yardstick.

    Three comparisons, all in the same three metrics, all on the same stars:

    truth (m,T) vs full six-field truth   does discarding four fields and
                                          rebuilding them change the observable?
    learned vs production                 does the deployable initializer?
    production vs its own jitter retry    how much does the solver move on its
                                          own, between two starts production is
                                          equally willing to ship from?

    The third is the calibration the first two need. Ting's 5e-3 bar was set for
    a candidate-vs-reference-implementation comparison, not for two runs of the
    same solver from different starts; without measuring the solver's own width,
    a number near the bar cannot be read either way.
    """

    sources = [
        ("truth (m,T)\nvs full six-field truth", "spectral_gate_truth_mT.json", ORACLE),
        ("learned reduced state\nvs production", "spectral_gate.json", LEARNED),
        ("production\nvs its own jitter start", "spectral_gate_jitter_control.json", PRODUCTION),
    ]
    fields = (
        ("normalized_flux", "normalized flux"),
        ("flux_total", "full flux / continuum"),
        ("flux_continuum", "continuum"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3), sharey=True)
    for ax, (field, title) in zip(axes, fields):
        for position, (name, filename, color) in enumerate(sources):
            gate = json.loads((RESULTS / filename).read_text())
            values = np.array([row[field]["max"] for row in gate["per_star"]])
            jitter = np.random.default_rng(position).uniform(-0.13, 0.13, values.size)
            over = values > BAR
            ax.scatter(position + jitter[~over], values[~over], s=17, color=color,
                       alpha=0.8, edgecolor=SURFACE, linewidth=0.4, zorder=3)
            ax.scatter(position + jitter[over], values[over], s=40, color=CRITICAL,
                       marker="D", edgecolor=SURFACE, linewidth=0.7, zorder=4)
            median = float(np.median(values))
            ax.plot([position - 0.30, position + 0.30], [median, median],
                    color=INK, linewidth=2.2, zorder=5, solid_capstyle="butt")
            ax.text(position, 2.6e-2, f"{int(over.sum())}/{values.size}",
                    ha="center", fontsize=8.5, fontweight="bold",
                    color=CRITICAL if over.any() else "#0ca30c")
        ax.axhline(BAR, color=CRITICAL, linewidth=1.4, linestyle="--", zorder=2)
        ax.set_yscale("log")
        ax.set_ylim(4e-6, 5.5e-2)
        ax.set_xlim(-0.6, len(sources) - 0.4)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels([name for name, _, _ in sources], fontsize=8)
        ax.set_title(title, loc="left", fontsize=9.5)
        _recess(ax, grid_axis="y")
    axes[0].set_ylabel("max difference over the 400\u2013900 nm window")
    axes[0].text(-0.45, BAR * 1.25, "5e-3 gate", color=CRITICAL, fontsize=8, va="bottom")
    axes[0].text(-0.45, 3.4e-2, "stars over:", ha="left", fontsize=8, color=INK_SECONDARY)

    handles = [
        Line2D([], [], color=INK, linewidth=2.2, label="median over stars"),
        Line2D([], [], color=CRITICAL, marker="D", linestyle="none", markersize=6,
               label="star over the gate"),
        Line2D([], [], color=CRITICAL, linestyle="--", label="5e-3 acceptance bar"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle(
        "(m,T) sufficiency, spectral dimension \u2014 converged products compared in Ting's three metrics",
        fontsize=11, fontweight="bold", color=INK, x=0.006, ha="left", y=0.985,
    )
    fig.text(
        0.006, 0.925,
        "Discarding four fields and rebuilding them from (m,T) changes the spectrum "
        "less than the solver moves between two starts production treats as equivalent.",
        fontsize=8.5, color=INK_SECONDARY, ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    out = FIGURES / "spectral_sufficiency.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def figure_field_consistency() -> None:
    """Plot the unified four-arm field comparison when its result exists."""

    path = RESULTS / "field_consistency_dev.json"
    if not path.is_file():
        print(f"skip {path}: run field_consistency first")
        return
    payload = json.loads(path.read_text())
    arms = (
        ("direct_six_field", "six-field direct", PRODUCTION),
        ("reconstruct_six_mT", "six-field m,T + physics", "#b56b2a"),
        ("reconstruct_truth_mT", "truth m,T + physics", ORACLE),
        ("reconstruct_reduced_mT", "new m,T + physics", LEARNED),
    )
    fields = (
        ("gas_pressure", "P"),
        ("electron_density", "n$_e$"),
        ("rosseland_opacity", "$\\kappa_R$"),
        ("radiative_acceleration", "$g_{rad}$"),
    )
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.7), sharey=True)
    positions = np.arange(len(arms))
    for ax, (field, title) in zip(axes, fields):
        values = []
        for key, _label, _color in arms:
            metric = payload["metrics"][key][field]
            metric_name = (
                "normalized_error"
                if field == "radiative_acceleration"
                else "relative_error"
            )
            values.append(metric[metric_name]["median"])
        ax.bar(
            positions,
            values,
            color=[color for _, _, color in arms],
            width=0.72,
            zorder=3,
        )
        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels([label for _, label, _ in arms], rotation=42, ha="right")
        ax.set_title(title, loc="left", fontsize=9.5)
        for position, value in zip(positions, values):
            ax.text(position, value * 1.14, f"{value:.1e}", ha="center", va="bottom", fontsize=7)
        _recess(ax)
    axes[0].set_ylabel("median error vs truth")
    fig.suptitle(
        "Four-field comparison after the (m,T) reconstruction repair",
        fontsize=11,
        fontweight="bold",
        color=INK,
        x=0.006,
        ha="left",
        y=0.99,
    )
    fig.text(
        0.006,
        0.925,
        f"{payload['metrics']['star_count']} development stars · "
        "g_rad uses the 2.0577 cm s⁻² floor · all m,T arms use adaptive pressure synchronization",
        fontsize=8.5,
        color=INK_SECONDARY,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    out = FIGURES / "field_consistency_comparison.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spectral-gate",
        type=Path,
        default=RESULTS / "spectral_gate.json",
        help="gate JSON used for the spectral-gate figure",
    )
    parser.add_argument(
        "--spectra-root",
        type=Path,
        default=RUNS / "spectra",
        help="directory containing learned_reduced_state/ and production_six_field/",
    )
    parser.add_argument(
        "--spectral-out",
        type=Path,
        default=FIGURES / "spectral_gate.png",
        help="output path for the selected spectral-gate figure",
    )
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_learned_vs_production()
    figure_residual_scale()
    figure_spectral_gate(args.spectral_gate, args.spectra_root, args.spectral_out)
    production_errors = RESULTS / "production_sixfield_errors.npz"
    if production_errors.is_file():
        figure_reduced_state_sufficiency(production_errors)
    figure_spectral_sufficiency()
    figure_field_consistency()
