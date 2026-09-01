#!/usr/bin/env python3
"""Build the English PDF for the four-initializer benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from payne_zero_figures import style  # noqa: E402
from payne_zero_figures.data import (  # noqa: E402
    load_json as _load_json,
    load_npz as _load_npz,
    sha256 as _sha256,
)

_OI = style.OkabeIto
BLUE = _OI.BLUE
ORANGE = _OI.ORANGE
GREEN = _OI.GREEN
PURPLE = _OI.PURPLE
SKY = _OI.SKY
RED = _OI.RED
BLACK = _OI.BLACK
GREY_COLOR = _OI.GREY
GREY = _OI.GREY
LIGHT_GREY = _OI.LIGHT_GREY


def _configure_style() -> None:
    style.configure(
        "REPORT",
    )



DEFAULT_SUMMARY = Path(
    "results/atmosphere_interpolation_benchmark_20260813/"
    "four_initializer_comparison.json"
)
DEFAULT_RUN_ROOT = Path("runs/atmosphere_interpolation_benchmark_20260813")
DEFAULT_OLD_RUN_ROOT = Path("runs/grey_start_benchmark_20260812/calibration_spectral60")
DEFAULT_OUT = Path("results/four_initializer_comparison_20260813_en.pdf")

SIX = "production_six_field"
TWO = "learned_reduced_state"
GREY = "grey15"
INTERP = "interpolated_full_state"
BAR = 5.0e-3
FLOOR = 2.0577175465785027



def _panel(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=11, weight="bold")


def _header(fig: mpl.figure.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.94, title, fontsize=20, weight="bold", color=BLACK)
    fig.text(0.055, 0.905, subtitle, fontsize=9.5, color=GREY_COLOR)


def _footer(fig: mpl.figure.Figure, page: int) -> None:
    fig.text(
        0.99,
        0.012,
        f"Four-initializer Payne-Zero benchmark | 2026-08-13 | Page {page}",
        ha="right",
        va="bottom",
        fontsize=7,
        color=GREY_COLOR,
    )


def _summary_page(pdf: PdfPages, report: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _header(
        fig,
        "Four atmosphere initializers",
        "Complete six-field interpolation, two-field network, six-field network, and grey atmosphere",
    )
    convergence = report["convergence"]
    rows = [
        ["Grey atmosphere", f"{convergence[GREY]['converged_count']} / {convergence[GREY]['star_count']}", f"{convergence[GREY]['summary']['converging_trial_iterations']['mean']:.2f}", f"{convergence[GREY]['converged_fraction']:.1%}"],
        ["Two-field (m, T) + physics", f"{convergence[TWO]['converged_count']} / {convergence[TWO]['star_count']}", f"{convergence[TWO]['summary']['converging_trial_iterations']['mean']:.2f}", f"{convergence[TWO]['converged_fraction']:.1%}"],
        ["Production six-field", f"{convergence[SIX]['converged_count']} / {convergence[SIX]['star_count']}", f"{convergence[SIX]['summary']['converging_trial_iterations']['mean']:.2f}", f"{convergence[SIX]['converged_fraction']:.1%}"],
        ["Interpolated full six-field", f"{convergence[INTERP]['converged_count']} / {convergence[INTERP]['star_count']}", f"{convergence[INTERP]['summary']['converging_trial_iterations']['mean']:.2f}", f"{convergence[INTERP]['converged_fraction']:.1%}"],
    ]
    ax = fig.add_axes([0.055, 0.56, 0.89, 0.25])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Initializer", "Converged", "Mean iterations*", "Rate"],
        cellLoc="center",
        colWidths=[0.43, 0.19, 0.23, 0.15],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.75)
    row_colors = ["#EEEEEE", "#FFF1E8", "#E9EEF2", "#F4EAF2"]
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(row_colors[row - 1])
    fig.text(0.055, 0.54, "*Mean among converged stars; failures remain in the denominator of the convergence rate.", fontsize=8, color=GREY_COLOR)

    fig.text(0.055, 0.465, "What this test asks", fontsize=14, weight="bold")
    fig.text(
        0.055,
        0.415,
        "Does a conventional interpolated atmosphere provide a useful starting point, and how does it compare "
        "with the learned two-field and six-field initializers? All four arms use the same 60 stars and the "
        "same unchanged physical solver.",
        fontsize=11,
        linespacing=1.45,
        wrap=True,
    )
    common = len(report["common_star_slugs"]["all_four"])
    extensions = report.get("grey_extensions", {})
    grey30 = extensions.get("grey30_cumulative", {}).get("converged_count")
    grey60 = extensions.get("grey60_cumulative", {}).get("converged_count")
    fig.text(0.055, 0.285, "Interpretation", fontsize=14, weight="bold")
    fig.text(
        0.055,
        0.235,
        f"Final-atmosphere and spectrum comparisons on the strict common sample use {common} stars for which all four "
        "arms converged within the primary 15-iteration cap. The grey 30/60-iteration extensions remain diagnostics.",
        fontsize=10.5,
        color=GREY_COLOR,
        wrap=True,
    )
    fig.text(0.055, 0.145, "Interpolation coordinates: 5040/Teff, logg, metallicity, and alpha enhancement; 8 donors, inverse-distance-squared weights.", fontsize=9.5)
    if grey30 is not None and grey60 is not None:
        fig.text(0.055, 0.105, f"Grey extension diagnostic: {grey30}/60 by 30 iterations and {grey60}/60 by 60 iterations.", fontsize=9.5, color=GREY_COLOR)
    _footer(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)


def _convergence_page(pdf: PdfPages, report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.82, wspace=0.30, hspace=0.42)
    _header(fig, "Convergence comparison", "All four arms use the same 15-iteration primary cap")
    arms = [GREY, TWO, SIX, INTERP]
    labels = ["Grey", "Two-field", "Six-field", "Interpolated"]
    colors = [GREY_COLOR, ORANGE, BLUE, PURPLE]
    conv = report["convergence"]

    ax = axes[0, 0]
    fractions = [conv[arm]["converged_fraction"] for arm in arms]
    bars = ax.bar(labels, fractions, color=colors, width=0.62)
    ax.axhline(59 / 60, color=BLACK, ls=":", lw=1, label="59/60 reference")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Converged fraction")
    ax.set_title("Convergence within 15 iterations")
    for bar, value, arm in zip(bars, fractions, arms):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{conv[arm]['converged_count']}/60", ha="center", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    _panel(ax, "A")

    ax = axes[0, 1]
    for arm, label, color in zip(arms, labels, colors):
        hist = conv[arm]["summary"]["converging_trial_iterations"]["histogram"]
        if not hist:
            continue
        x = np.array([int(key) for key in hist])
        y = np.array([hist[str(value)] for value in x])
        order = np.argsort(x)
        ax.step(x[order], y[order], where="mid", lw=1.8, color=color, label=label)
        ax.scatter(x[order], y[order], s=20, color=color)
    ax.set_xlim(2.5, 15.5)
    ax.set_xlabel("Iteration at convergence")
    ax.set_ylabel("Number of stars")
    ax.set_title("Iteration distribution among successes")
    ax.legend(frameon=False)
    _panel(ax, "B")

    ax = axes[1, 0]
    means = [conv[arm]["summary"]["converging_trial_iterations"]["mean"] for arm in arms]
    p90 = [conv[arm]["summary"]["converging_trial_iterations"]["p90"] for arm in arms]
    y = np.arange(len(arms))
    for yi, mean, upper, color in zip(y, means, p90, colors):
        ax.hlines(yi, mean, upper, color=color, lw=3)
        ax.scatter(mean, yi, color=color, s=65, zorder=3)
        ax.scatter(upper, yi, facecolor="white", edgecolor=color, linewidth=2, s=50, zorder=3)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 16)
    ax.set_xlabel("Iterations among converged stars")
    ax.set_title("Mean and p90 iteration count")
    ax.scatter([], [], color=BLACK, s=55, label="mean")
    ax.scatter([], [], facecolor="white", edgecolor=BLACK, linewidth=2, s=45, label="p90")
    ax.legend(frameon=False, loc="lower right")
    _panel(ax, "C")

    ax = axes[1, 1]
    nonmono = [conv[arm]["summary"]["contraction"]["non_monotonic_fraction"] for arm in arms]
    bars = ax.bar(labels, nonmono, color=colors, width=0.62)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Non-monotonic trajectory fraction")
    ax.set_title("Solver trajectory stability")
    for bar, value in zip(bars, nonmono):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.0%}", ha="center")
    _panel(ax, "D")
    _footer(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


def _profile_page(pdf: PdfPages, report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.085, right=0.96, bottom=0.10, top=0.82, wspace=0.30, hspace=0.43)
    common_count = len(report["common_star_slugs"]["all_four"])
    _header(fig, "Final six-field atmosphere comparison", f"All four arms converged on the same {common_count}-star common sample")
    profile = report["final_six_field_profiles"]["all_four_vs_six"]
    arms = [TWO, GREY, INTERP]
    labels = ["Two-field", "Grey", "Interpolated"]
    colors = [ORANGE, GREY_COLOR, PURPLE]
    fields = ["column_mass", "temperature", "gas_pressure", "electron_density", "rosseland_opacity", "radiative_acceleration"]
    field_labels = ["m", "T", "P", r"$n_e$", r"$\kappa_R$", r"$g_{rad}$"]
    x = np.arange(len(fields))
    width = 0.25
    for ax, statistic, panel, title in (
        (axes[0, 0], "median", "A", "Median layer-wise difference"),
        (axes[0, 1], "p95", "B", "95th-percentile layer-wise difference"),
    ):
        for offset, arm, label, color in zip((-width, 0, width), arms, labels, colors):
            values = [profile[arm]["fields"][field][statistic] for field in fields]
            ax.bar(x + offset, values, width, color=color, label=f"{label} vs six-field")
        ax.set_yscale("log")
        ax.set_xticks(x, field_labels)
        ax.set_ylabel("Absolute difference")
        ax.set_title(title)
        ax.grid(axis="y", which="both", alpha=0.18)
        ax.legend(frameon=False)
        _panel(ax, panel)
    axes[0, 0].text(0.01, -0.30, "Positive fields use dex; radiative acceleration uses the 2.0577 cm s^-2 floor.", transform=axes[0, 0].transAxes, fontsize=8, color=GREY_COLOR)

    ax = axes[1, 0]
    mean_values = [profile[arm]["fields"]["temperature"]["median"] for arm in arms]
    p95_values = [profile[arm]["fields"]["temperature"]["p95"] for arm in arms]
    ax.bar(np.arange(3) - 0.18, mean_values, 0.36, color=colors, label="Median")
    ax.bar(np.arange(3) + 0.18, p95_values, 0.36, color=colors, alpha=0.38, label="p95")
    ax.set_xticks(np.arange(3), labels)
    ax.set_yscale("log")
    ax.set_ylabel("Temperature difference (dex)")
    ax.set_title("Temperature agreement with six-field solution")
    ax.legend(frameon=False)
    _panel(ax, "C")

    ax = axes[1, 1]
    gmean = [profile[arm]["fields"]["radiative_acceleration"]["median"] for arm in arms]
    gp95 = [profile[arm]["fields"]["radiative_acceleration"]["p95"] for arm in arms]
    ax.bar(np.arange(3) - 0.18, gmean, 0.36, color=colors, label="Median")
    ax.bar(np.arange(3) + 0.18, gp95, 0.36, color=colors, alpha=0.38, label="p95")
    ax.set_xticks(np.arange(3), labels)
    ax.set_yscale("log")
    ax.set_ylabel("Floored normalized difference")
    ax.set_title("Radiative acceleration agreement")
    ax.legend(frameon=False)
    _panel(ax, "D")
    _footer(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


def _spectral_page(pdf: PdfPages, report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.085, right=0.96, bottom=0.10, top=0.82, wspace=0.30, hspace=0.43)
    common_count = len(report["common_star_slugs"]["all_four"])
    _header(fig, "Synthetic spectrum comparison", f"400-900 nm at R=20,000; common sample n={common_count}; threshold = 0.005")
    entries = [("two_vs_six", "Two-field", ORANGE), ("grey15_vs_six", "Grey", GREY_COLOR), ("interpolation_vs_six", "Interpolated", PURPLE)]
    metrics = [("normalized_flux", "Maximum normalized-flux difference"), ("flux_total", "Maximum total-flux difference")]
    x = np.arange(3)
    for ax, (metric, title), panel in zip((axes[0, 0], axes[0, 1]), metrics, ("A", "B")):
        med = [report["spectra"][key][metric]["median_max"] for key, _, _ in entries]
        max_values = [report["spectra"][key][metric]["max"] for key, _, _ in entries]
        ax.bar(x - 0.18, med, 0.36, color=[color for _, _, color in entries], label="Median over stars")
        ax.bar(x + 0.18, max_values, 0.36, color=[color for _, _, color in entries], alpha=0.38, label="Largest star")
        ax.axhline(BAR, color=RED, ls="--", lw=1.1, label="0.005 threshold")
        ax.set_xticks(x, [label for _, label, _ in entries])
        ax.set_yscale("log")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(frameon=False)
        _panel(ax, panel)

    ax = axes[1, 0]
    values = [report["spectra"][key]["normalized_flux"]["stars_over_bar"] for key, _, _ in entries]
    bars = ax.bar([label for _, label, _ in entries], values, color=[color for _, _, color in entries])
    ax.set_ylim(0, max(3, common_count))
    ax.set_ylabel("Stars above 0.005")
    ax.set_title("Normalized-flux threshold failures")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15, str(value), ha="center", weight="bold")
    _panel(ax, "C")

    ax = axes[1, 1]
    ax.axis("off")
    rows = []
    for key, label, _ in entries:
        entry = report["spectra"][key]["normalized_flux"]
        rows.append([label, f"{entry['median_max']:.3e}", f"{entry['max']:.3e}", f"{entry['stars_over_bar']} / {common_count}"])
    table = ax.table(cellText=rows, colLabels=["Candidate", "Median max", "Largest", "> 0.005"], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(["#FFF1E8", "#EEEEEE", "#F4EAF2"][row - 1])
    ax.set_title("Normalized-flux summary", pad=18)
    _panel(ax, "D")
    _footer(fig, 4)
    pdf.savefig(fig)
    plt.close(fig)


def _choose_representatives(report: dict) -> list[str]:
    common = set(report["common_star_slugs"]["all_four"])
    path = Path("results/atmosphere_interpolation_benchmark_20260813/spectral_interpolation_vs_six.json")
    if path.is_file():
        rows = [row for row in _load_json(path).get("per_star", []) if row["slug"] in common]
        rows.sort(key=lambda row: row["normalized_flux"]["max"])
        if rows:
            return [rows[len(rows) // 2]["slug"], rows[-1]["slug"]]
    return sorted(common)[:2]


def _representative_spectra_page(pdf: PdfPages, report: dict, run_root: Path, old_run_root: Path) -> None:
    reps = _choose_representatives(report)
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.82, wspace=0.28, hspace=0.42)
    _header(fig, "Representative narrow-band spectra", "5 nm windows centered on the largest grey-vs-six-field residual for each selected star")
    profile_roots = {SIX: old_run_root / "profiles" / SIX}
    spectrum_roots = {
        SIX: run_root / "spectra" / SIX,
        TWO: run_root / "spectra" / TWO,
        GREY: run_root / "spectra" / GREY,
        INTERP: run_root / "spectra" / INTERP,
    }
    colors = {SIX: BLUE, TWO: ORANGE, GREY: GREY_COLOR, INTERP: PURPLE}
    styles = {SIX: "-", TWO: "--", GREY: ":", INTERP: "-."}
    labels = {SIX: "Six-field", TWO: "Two-field", GREY: "Grey", INTERP: "Interpolated"}
    for row, slug in enumerate(reps):
        spectra = {arm: _load_npz(spectrum_roots[arm] / f"{slug}.npz") for arm in (SIX, TWO, GREY, INTERP)}
        wavelength = spectra[SIX]["wavelength_nm"]
        differences = np.abs(spectra[GREY]["normalized_flux"] - spectra[SIX]["normalized_flux"])
        centre = float(wavelength[int(np.argmax(differences))])
        window = np.abs(wavelength - centre) <= 2.5
        ax = axes[row, 0]
        for arm in (SIX, TWO, GREY, INTERP):
            ax.plot(wavelength[window], spectra[arm]["normalized_flux"][window], color=colors[arm], ls=styles[arm], lw=1.5, label=labels[arm])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized flux")
        ax.set_title(f"{slug[:7]}: spectrum near {centre:.1f} nm")
        ax.legend(frameon=False, ncol=2)
        _panel(ax, "A" if row == 0 else "C")

        ax = axes[row, 1]
        reference = spectra[SIX]["normalized_flux"]
        for arm in (TWO, GREY, INTERP):
            ax.plot(wavelength[window], (spectra[arm]["normalized_flux"] - reference)[window], color=colors[arm], ls=styles[arm], lw=1.5, label=f"{labels[arm]} - six-field")
        ax.axhline(0.0, color=BLACK, lw=0.8)
        ax.axhline(BAR, color=RED, ls="--", lw=0.9)
        ax.axhline(-BAR, color=RED, ls="--", lw=0.9, label="+/- 0.005")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized-flux residual")
        ax.set_title("Residual to six-field solution")
        ax.legend(frameon=False, ncol=2)
        _panel(ax, "B" if row == 0 else "D")
    fig.text(0.08, 0.045, "The report selects one middle and one worst interpolation case from the four-arm common sample.", fontsize=8, color=GREY_COLOR)
    _footer(fig, 5)
    pdf.savefig(fig)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--old-run-root", type=Path, default=DEFAULT_OLD_RUN_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not args.summary.is_file():
        raise SystemExit(f"missing summary: {args.summary}")
    report = _load_json(args.summary)
    common = report["common_star_slugs"]["all_four"]
    if len(common) == 0:
        raise SystemExit("no four-arm common stars available for the PDF")
    reps = _choose_representatives(report)
    required = [
        args.run_root / "spectra" / arm / f"{slug}.npz"
        for arm in (SIX, TWO, GREY, INTERP)
        for slug in reps
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing representative spectra: {missing[:5]}")
    _configure_style()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Four atmosphere initializer comparison",
        "Author": "Payne-Zero reduced-state validation",
        "Subject": "Convergence, final atmospheres, and synthetic spectra",
        "Keywords": "stellar atmosphere, interpolation, reduced state, Payne-Zero",
    }
    with PdfPages(args.out, metadata=metadata) as pdf:
        _summary_page(pdf, report)
        _convergence_page(pdf, report)
        _profile_page(pdf, report)
        _spectral_page(pdf, report)
        _representative_spectra_page(pdf, report, args.run_root, args.old_run_root)
    manifest = args.out.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "format": "payne_zero_four_initializer_report_v1",
                "pdf": str(args.out),
                "pdf_sha256": _sha256(args.out),
                "summary": str(args.summary),
                "summary_sha256": _sha256(args.summary),
                "four_arm_common_star_count": len(common),
                "representative_slugs": reps,
                "spectrum_window_nm": [400.0, 900.0],
                "spectrum_resolution": 20000.0,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.out}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
