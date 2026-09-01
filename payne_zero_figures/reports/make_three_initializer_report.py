#!/usr/bin/env python3
"""Create an English PDF for the grey/two-field/six-field comparison."""

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
        **{"savefig.facecolor": "white"},
    )



SUMMARY = Path(
    "results/grey_start_benchmark_20260812/calibration_spectral60/"
    "three_initializer_comparison.json"
)
SPECTRAL_ROOT = Path("results/grey_start_benchmark_20260812/spectral_jsons")
REPORT_DATA = Path("results/grey_start_benchmark_20260812/report_data")
DEFAULT_OUT = Path("results/grey_two_six_initializer_comparison_20260813_en.pdf")

SIX = "production_six_field"
TWO = "learned_reduced_state"
GREY = "grey15"
BAR = 5.0e-3


REPRESENTATIVE_STARS = (
    "t04561.6_g+4.56_m+0.40_a+0.38_x1.57",
    "t09120.4_g+2.05_m-0.19_a+0.26_x2.74",
)


def _panel(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=11, weight="bold")


def _footer(fig: mpl.figure.Figure, page: int) -> None:
    fig.text(
        0.99,
        0.012,
        f"Grey, two-field, and six-field initialization | 2026-08-13 | Page {page}",
        ha="right",
        va="bottom",
        fontsize=7,
        color=GREY_COLOR,
    )


def _header(fig: mpl.figure.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.94, title, fontsize=20, weight="bold", color=BLACK)
    fig.text(0.055, 0.905, subtitle, fontsize=9.5, color=GREY_COLOR)


def _summary_page(pdf: PdfPages, report: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _header(
        fig,
        "Initializer comparison: grey atmosphere, two fields, and six fields",
        "Same 60 stars, unchanged Payne-Zero solver, 15-iteration primary cap",
    )

    ax = fig.add_axes([0.055, 0.54, 0.89, 0.28])
    ax.axis("off")
    rows = [
        ["Grey atmosphere", "12 / 60", "13.08", "20%"],
        ["Two-field (m, T) + physics", "57 / 60", "3.51", "95%"],
        ["Production six-field", "58 / 60", "6.12", "96.7%"],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Initializer", "Converged", "Mean iterations*", "Rate"],
        cellLoc="center",
        colWidths=[0.43, 0.19, 0.23, 0.15],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row == 2:
            cell.set_facecolor("#FFF1E8")
        else:
            cell.set_facecolor("#F3F5F7" if row % 2 else "#E9EEF2")
    fig.text(0.055, 0.525, "*Among converged stars; failures remain counted in the convergence rate.", fontsize=8, color=GREY_COLOR)

    fig.text(0.055, 0.47, "Main result", fontsize=15, weight="bold", color=BLACK)
    conclusion = (
        "The two-field initializer reaches essentially the same final atmosphere and spectrum as the six-field\n"
        "initializer, while using about 2.6 fewer iterations on average. Grey starts usually do not converge\n"
        "within 15 iterations; even formally converged grey cases often retain large spectral differences."
    )
    fig.text(0.055, 0.425, conclusion, fontsize=12, linespacing=1.45, va="top", color=BLACK)

    fig.text(0.055, 0.285, "Grey extension diagnostic", fontsize=13, weight="bold")
    fig.text(
        0.055,
        0.245,
        "Recovered grey starts: 12/60 by iteration 15, 22/60 by iteration 30, and 28/60 by iteration 60.",
        fontsize=11,
    )
    fig.text(
        0.055,
        0.19,
        "Fair profile and spectrum comparisons use the 11 stars for which all three primary arms converged.\n"
        "The relaxed grey runs are shown separately and do not change the 15-iteration result.",
        fontsize=10,
        color=GREY_COLOR,
        wrap=True,
    )

    _footer(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)


def _histogram_from_summary(ax: mpl.axes.Axes, summary: dict, label: str, color: str) -> None:
    histogram = summary["converging_trial_iterations"]["histogram"]
    x = np.array([int(value) for value in histogram], dtype=int)
    y = np.array([histogram[str(value)] for value in x], dtype=int)
    order = np.argsort(x)
    ax.step(x[order], y[order], where="mid", lw=2.0, color=color, label=label)
    ax.scatter(x[order], y[order], s=24, color=color, zorder=3)


def _convergence_page(pdf: PdfPages, report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.09, top=0.82, wspace=0.30, hspace=0.42)
    _header(fig, "Convergence behaviour", "The primary comparison uses the same 15-iteration cap for all three initializers")

    conv = report["convergence"]
    ax = axes[0, 0]
    names = ["Grey", "Two-field", "Six-field"]
    values = [conv["grey15"]["converged_fraction"], conv["two_field"]["converged_fraction"], conv["six_field"]["converged_fraction"]]
    colors = [GREY_COLOR, ORANGE, BLUE]
    bars = ax.bar(names, values, color=colors, width=0.62)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Converged fraction")
    ax.set_title("Convergence within 15 iterations")
    ax.axhline(59 / 60, color=BLACK, ls=":", lw=1, label="59/60 target")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value*60:.0f}/60", ha="center", weight="bold")
    ax.legend(frameon=False, loc="lower right")
    _panel(ax, "A")

    ax = axes[0, 1]
    _histogram_from_summary(ax, conv["grey15"], "Grey", GREY_COLOR)
    _histogram_from_summary(ax, conv["two_field"], "Two-field", ORANGE)
    _histogram_from_summary(ax, conv["six_field"], "Six-field", BLUE)
    ax.set_xlim(2.5, 15.5)
    ax.set_xlabel("Iteration at convergence")
    ax.set_ylabel("Number of stars")
    ax.set_title("Iteration distribution among successes")
    ax.legend(frameon=False)
    _panel(ax, "B")

    ax = axes[1, 0]
    caps = np.array([15, 30, 60])
    recovered = np.array([12, 22, 28])
    ax.plot(caps, recovered, marker="o", ms=7, lw=2.2, color=GREY_COLOR)
    ax.fill_between(caps, 0, recovered, color=GREY_COLOR, alpha=0.12)
    for x, y in zip(caps, recovered):
        ax.text(x, y + 1.5, f"{y}/60", ha="center", weight="bold")
    ax.set_xlim(10, 65)
    ax.set_ylim(0, 60)
    ax.set_xticks(caps)
    ax.set_xlabel("Maximum permitted iterations")
    ax.set_ylabel("Cumulative grey successes")
    ax.set_title("Relaxing the grey-start cap helps only partly")
    _panel(ax, "C")

    ax = axes[1, 1]
    mean_iters = [13.08, 3.51, 6.12]
    p90 = [14.9, 4.0, 8.0]
    y = np.arange(3)
    ax.hlines(y, mean_iters, p90, color=colors, lw=3, alpha=0.7)
    ax.scatter(mean_iters, y, s=70, color=colors, label="Mean", zorder=3)
    ax.scatter(p90, y, s=55, facecolor="white", edgecolor=colors, linewidth=2, label="p90", zorder=3)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 16)
    ax.set_xlabel("Iterations among converged stars")
    ax.set_title("Two-field initialization is fastest")
    ax.legend(frameon=False, loc="lower right")
    _panel(ax, "D")

    _footer(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


def _profile_page(pdf: PdfPages, report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.085, right=0.96, bottom=0.10, top=0.82, wspace=0.30, hspace=0.43)
    _header(fig, "Final six-field atmosphere comparison", "Common primary sample: 11 stars for which grey, two-field, and six-field starts all converged")

    common = report["final_six_field_profiles"]["common15"]
    fields = ["column_mass", "temperature", "gas_pressure", "electron_density", "rosseland_opacity", "radiative_acceleration"]
    labels = ["m", "T", "P", r"$n_e$", r"$\kappa_R$", r"$g_{rad}$"]
    x = np.arange(len(fields))
    width = 0.37
    for ax, statistic, panel, title in (
        (axes[0, 0], "median", "A", "Median layer-wise difference"),
        (axes[0, 1], "p95", "B", "95th-percentile layer-wise difference"),
    ):
        two = [common["two_vs_six"]["fields"][name][statistic] for name in fields]
        grey = [common["grey_vs_six"]["fields"][name][statistic] for name in fields]
        ax.bar(x - width / 2, two, width, color=ORANGE, label="Two-field vs six-field")
        ax.bar(x + width / 2, grey, width, color=GREY_COLOR, label="Grey vs six-field")
        ax.set_yscale("log")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Absolute difference")
        ax.set_title(title)
        ax.grid(axis="y", which="both", alpha=0.18)
        ax.legend(frameon=False)
        _panel(ax, panel)
    axes[0, 0].text(0.01, -0.30, "Positive fields use dex; radiative acceleration uses the floored normalized error.", transform=axes[0, 0].transAxes, fontsize=8, color=GREY_COLOR)

    slug = REPRESENTATIVE_STARS[1]
    profiles = {
        "Grey": _load_npz(REPORT_DATA / "profiles" / GREY / f"{slug}.npz"),
        "Two-field": _load_npz(REPORT_DATA / "profiles" / TWO / f"{slug}.npz"),
        "Six-field": _load_npz(REPORT_DATA / "profiles" / SIX / f"{slug}.npz"),
    }
    colors = {"Grey": GREY_COLOR, "Two-field": ORANGE, "Six-field": BLUE}
    styles = {"Grey": ":", "Two-field": "--", "Six-field": "-"}
    ax = axes[1, 0]
    for name, profile in profiles.items():
        ax.plot(np.log10(profile["column_mass"]), profile["temperature"], color=colors[name], ls=styles[name], lw=1.8, label=name)
    ax.set_xlabel(r"$\log_{10}(m / \mathrm{g\,cm^{-2}})$")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Representative final temperature profile")
    ax.legend(frameon=False)
    _panel(ax, "C")

    ax = axes[1, 1]
    reference = profiles["Six-field"]
    for name in ("Grey", "Two-field"):
        delta_t = (profiles[name]["temperature"] - reference["temperature"]) / reference["temperature"]
        ax.plot(np.log10(reference["column_mass"]), 100 * delta_t, color=colors[name], ls=styles[name], lw=1.8, label=f"{name} - six-field")
    ax.axhline(0, color=BLACK, lw=0.8)
    ax.set_xlabel(r"$\log_{10}(m / \mathrm{g\,cm^{-2}})$")
    ax.set_ylabel("Temperature difference (%)")
    ax.set_title("Grey can stop far from the six-field solution")
    ax.legend(frameon=False)
    _panel(ax, "D")

    fig.text(0.085, 0.045, "Representative star: Teff=9120 K, log g=2.05, [M/H]=-0.19. The aggregate bars remain the primary result.", fontsize=8, color=GREY_COLOR)
    _footer(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


def _spectral_rows(path: Path, common: set[str]) -> dict[str, dict]:
    source = _load_json(path)
    return {row["slug"]: row for row in source["per_star"] if row["slug"] in common}


def _spectrum_population_page(pdf: PdfPages, report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.085, right=0.96, bottom=0.10, top=0.82, wspace=0.30, hspace=0.43)
    _header(fig, "Spectral comparison", "400-900 nm at R=20,000; primary common sample n=11; threshold = 0.005")

    common = set(report["common_star_slugs"]["grey15"])
    two = _spectral_rows(SPECTRAL_ROOT / "spectral_two_vs_six_unified.json", common)
    grey = _spectral_rows(SPECTRAL_ROOT / "spectral_grey15_vs_six.json", common)
    ordered = sorted(common, key=lambda slug: grey[slug]["normalized_flux"]["max"])
    x = np.arange(1, len(ordered) + 1)

    for ax, metric, ylabel, panel in (
        (axes[0, 0], "normalized_flux", "Maximum normalized-flux difference", "A"),
        (axes[0, 1], "flux_total", "Maximum total-flux difference", "B"),
    ):
        ax.plot(x, [grey[s][metric]["max"] for s in ordered], marker="o", color=GREY_COLOR, lw=1.8, label="Grey vs six-field")
        ax.plot(x, [two[s][metric]["max"] for s in ordered], marker="s", color=ORANGE, lw=1.8, label="Two-field vs six-field")
        ax.axhline(BAR, color=RED, ls="--", lw=1.2, label="0.005 threshold")
        ax.set_yscale("log")
        ax.set_xlabel("Stars ordered by grey-start difference")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        _panel(ax, panel)

    ax = axes[1, 0]
    aggregate = report["spectra"]["common15"]
    categories = ["Normalized", "Total", "Continuum"]
    grey_counts = [aggregate["grey_vs_six"][key]["stars_over_bar"] for key in ("normalized_flux", "flux_total", "flux_continuum")]
    two_counts = [aggregate["two_vs_six"][key]["stars_over_bar"] for key in ("normalized_flux", "flux_total", "flux_continuum")]
    idx = np.arange(3)
    ax.bar(idx - 0.19, two_counts, 0.38, color=ORANGE, label="Two-field vs six-field")
    ax.bar(idx + 0.19, grey_counts, 0.38, color=GREY_COLOR, label="Grey vs six-field")
    ax.set_xticks(idx, categories)
    ax.set_ylabel("Stars above 0.005")
    ax.set_ylim(0, 11)
    ax.set_title("Threshold failures")
    ax.legend(frameon=False)
    _panel(ax, "C")

    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ["Two-field vs six-field", "0.00118", "0.00523", "1 / 11"],
        ["Grey vs six-field", "0.01989", "0.11585", "9 / 11"],
    ]
    table = ax.table(cellText=rows, colLabels=["Comparison", "Median max", "Largest", "> 0.005"], cellLoc="center", loc="center")
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
            cell.set_facecolor("#FFF1E8" if row == 1 else "#EEEEEE")
    ax.set_title("Normalized-flux summary", pad=18)
    _panel(ax, "D")

    _footer(fig, 4)
    pdf.savefig(fig)
    plt.close(fig)


def _narrow_window(spectra: dict[str, dict[str, np.ndarray]], width_nm: float = 5.0) -> tuple[np.ndarray, float]:
    wavelength = spectra["Six-field"]["wavelength_nm"]
    difference = np.abs(spectra["Grey"]["normalized_flux"] - spectra["Six-field"]["normalized_flux"])
    centre = float(wavelength[int(np.argmax(difference))])
    window = np.abs(wavelength - centre) <= width_nm / 2
    return window, centre


def _representative_spectra_page(pdf: PdfPages) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.82, wspace=0.28, hspace=0.42)
    _header(fig, "Representative narrow-band spectra", "Each panel shows the 5 nm window containing the largest grey-vs-six-field difference")

    for row, slug in enumerate(REPRESENTATIVE_STARS):
        spectra = {
            "Grey": _load_npz(REPORT_DATA / "spectra" / GREY / f"{slug}.npz"),
            "Two-field": _load_npz(REPORT_DATA / "spectra" / TWO / f"{slug}.npz"),
            "Six-field": _load_npz(REPORT_DATA / "spectra" / SIX / f"{slug}.npz"),
        }
        window, centre = _narrow_window(spectra)
        wavelength = spectra["Six-field"]["wavelength_nm"]
        ax = axes[row, 0]
        ax.plot(wavelength[window], spectra["Six-field"]["normalized_flux"][window], color=BLUE, lw=1.8, label="Six-field")
        ax.plot(wavelength[window], spectra["Two-field"]["normalized_flux"][window], color=ORANGE, ls="--", lw=1.5, label="Two-field")
        ax.plot(wavelength[window], spectra["Grey"]["normalized_flux"][window], color=GREY_COLOR, ls=":", lw=1.5, label="Grey")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized flux")
        ax.set_title(f"{slug[:7]}: spectrum near {centre:.1f} nm")
        ax.legend(frameon=False, ncol=3)
        _panel(ax, "A" if row == 0 else "C")

        ax = axes[row, 1]
        reference = spectra["Six-field"]["normalized_flux"]
        ax.plot(wavelength[window], (spectra["Two-field"]["normalized_flux"] - reference)[window], color=ORANGE, lw=1.7, label="Two-field - six-field")
        ax.plot(wavelength[window], (spectra["Grey"]["normalized_flux"] - reference)[window], color=GREY_COLOR, lw=1.5, label="Grey - six-field")
        ax.axhline(0, color=BLACK, lw=0.8)
        ax.axhline(BAR, color=RED, ls="--", lw=0.9)
        ax.axhline(-BAR, color=RED, ls="--", lw=0.9, label="+/- 0.005")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized-flux residual")
        ax.set_title("Residual to the six-field solution")
        ax.legend(frameon=False, ncol=2)
        _panel(ax, "B" if row == 0 else "D")

    fig.text(
        0.08,
        0.045,
        "Top: a median grey-start spectral case (Teff=4562 K, log g=4.56). Bottom: the largest grey-start difference (Teff=9120 K, log g=2.05).",
        fontsize=8,
        color=GREY_COLOR,
    )
    _footer(fig, 5)
    pdf.savefig(fig)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    required = [
        args.summary,
        SPECTRAL_ROOT / "spectral_two_vs_six_unified.json",
        SPECTRAL_ROOT / "spectral_grey15_vs_six.json",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing report inputs: {missing}")

    _configure_style()
    report = _load_json(args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Grey, two-field, and six-field Payne-Zero initializer comparison",
        "Author": "Payne-Zero reduced-state validation",
        "Subject": "Convergence, final six-field atmospheres, and synthetic spectra",
        "Keywords": "stellar atmosphere, grey atmosphere, reduced state, spectrum, Payne-Zero",
    }
    with PdfPages(args.out, metadata=metadata) as pdf:
        _summary_page(pdf, report)
        _convergence_page(pdf, report)
        _profile_page(pdf, report)
        _spectrum_population_page(pdf, report)
        _representative_spectra_page(pdf)

    manifest = args.out.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "format": "payne_zero_three_initializer_report_v1",
                "pdf": str(args.out),
                "pdf_sha256": _sha256(args.out),
                "summary": str(args.summary),
                "summary_sha256": _sha256(args.summary),
                "primary_star_count": 60,
                "common_primary_star_count": report["final_six_field_profiles"]["common15"]["star_count"],
                "spectral_window_nm": report["comparison_contract"]["spectrum_window_nm"],
                "spectral_resolution": report["comparison_contract"]["spectrum_resolution"],
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
