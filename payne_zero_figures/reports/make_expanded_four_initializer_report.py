#!/usr/bin/env python3
"""Build an English PDF for the expanded four-initializer benchmark."""

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



SIX = "production_six_field"
TWO = "learned_reduced_state"
GREY = "grey15"
INTERP = "interpolated_full_state"
ARMS = (SIX, TWO, GREY, INTERP)
CANDIDATES = (TWO, GREY, INTERP)
BAR = 5.0e-3

COLORS = {SIX: BLUE, TWO: ORANGE, GREY: GREY_COLOR, INTERP: PURPLE}
LABELS = {
    SIX: "Six-field network",
    TWO: "Two-field (m,T)",
    GREY: "Grey atmosphere",
    INTERP: "Full-state interpolation",
}
SHORT_LABELS = {SIX: "Six-field", TWO: "Two-field", GREY: "Grey", INTERP: "Interpolated"}
LINESTYLES = {SIX: "-", TWO: "--", GREY: ":", INTERP: "-."}

DEFAULT_SUMMARY = Path(
    "results/four_initializer_benchmark_expanded_20260814/"
    "expanded_four_initializer_comparison.json"
)
DEFAULT_RUN_ROOT = Path("runs/four_initializer_benchmark_expanded_20260814")
DEFAULT_RESULT_ROOT = Path("results/four_initializer_benchmark_expanded_20260814")
DEFAULT_OUT = Path("results/four_initializer_benchmark_expanded_20260814_en.pdf")


def _header(fig: mpl.figure.Figure, title: str, subtitle: str) -> None:
    fig.text(0.055, 0.94, title, fontsize=20, weight="bold", color=BLACK)
    fig.text(0.055, 0.905, subtitle, fontsize=9.5, color=GREY_COLOR)


def _footer(fig: mpl.figure.Figure, page: int) -> None:
    fig.text(
        0.99,
        0.012,
        f"Expanded Payne-Zero initializer benchmark | 2026-08-14 | Page {page}",
        ha="right",
        va="bottom",
        fontsize=7,
        color=GREY_COLOR,
    )


def _panel(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=11, weight="bold")


def _stats(entry: dict, field: str = "converging_trial_iterations") -> tuple[float, float]:
    summary = entry.get("summary") or {}
    values = summary.get(field) or {}
    return float(values.get("mean", np.nan)), float(values.get("p90", np.nan))


def _summary_page(pdf: PdfPages, report: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _header(
        fig,
        "Expanded four-initializer benchmark",
        "200 open stars: the previous 60 plus 140 new stars; same solver and 15-iteration primary cap",
    )
    subset_names = ("expanded200", "previous60", "added140")
    subset_labels = ("All 200", "Previous 60", "New 140")
    rows = []
    for arm in (GREY, TWO, SIX, INTERP):
        values = []
        for subset in subset_names:
            entry = report["subsets"][subset]["convergence"][arm]
            mean, p90 = _stats(entry)
            values.append(
                f"{entry['converged_count']}/{entry['star_count']}\n"
                f"{mean:.2f} / {p90:.0f} it"
            )
        rows.append([LABELS[arm], *values])
    ax = fig.add_axes([0.055, 0.58, 0.89, 0.25])
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Initializer", *subset_labels],
        cellLoc="center",
        colWidths=[0.31, 0.23, 0.23, 0.23],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
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
    fig.text(
        0.055,
        0.55,
        "Each cell is converged stars / total, followed by mean / p90 iterations among converged stars.",
        fontsize=8,
        color=GREY_COLOR,
    )
    fig.text(0.055, 0.47, "Test design", fontsize=14, weight="bold")
    fig.text(
        0.055,
        0.415,
        "The 140 additions are sampled from the already-open calibration set with 70 ordinary, 35 hard, "
        "and 35 edge stars. The original 60-star comparison is retained exactly. The sealed 200-star holdout "
        "was not opened.",
        fontsize=10.5,
        linespacing=1.45,
        wrap=True,
    )
    fig.text(0.055, 0.285, "Interpretation", fontsize=14, weight="bold")
    fig.text(
        0.055,
        0.23,
        "Spectra and final structures are compared only on stars for which the relevant arms produced a converged "
        "product. The six-field network is the reference initializer, not a physical truth model.",
        fontsize=10.5,
        color=GREY_COLOR,
        wrap=True,
    )
    fig.text(
        0.055,
        0.13,
        "Spectrum gate: 400-900 nm, R=20,000; normalized-flux threshold = 0.005. Grey results are limited to 15 iterations here.",
        fontsize=9.5,
    )
    _footer(fig, 1)
    pdf.savefig(fig)
    plt.close(fig)


def _convergence_page(pdf: PdfPages, report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.82, wspace=0.30, hspace=0.42)
    _header(fig, "Convergence across the expanded sample", "The new 140-star slice is shown separately from the previous 60-star result")
    arms = (GREY, TWO, SIX, INTERP)
    colors = [COLORS[arm] for arm in arms]
    labels = [SHORT_LABELS[arm] for arm in arms]
    subsets = ("expanded200", "previous60", "added140")

    ax = axes[0, 0]
    x = np.arange(len(arms))
    width = 0.24
    for offset, subset in zip((-width, 0, width), subsets):
        values = [report["subsets"][subset]["convergence"][arm]["converged_fraction"] for arm in arms]
        bars = ax.bar(x + offset, values, width, label={"expanded200": "All 200", "previous60": "Previous 60", "added140": "New 140"}[subset], alpha=0.95)
        if subset == "expanded200":
            for bar, arm in zip(bars, arms):
                bar.set_color(COLORS[arm])
        else:
            for bar, arm in zip(bars, arms):
                bar.set_color(COLORS[arm])
                bar.set_alpha(0.42 if subset == "previous60" else 0.68)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Converged fraction")
    ax.set_title("Convergence within 15 iterations")
    ax.legend(frameon=False, fontsize=7)
    _panel(ax, "A")

    ax = axes[0, 1]
    for subset, marker, linestyle in zip(subsets, ("o", "s", "^"), ("-", "--", ":")):
        means = [_stats(report["subsets"][subset]["convergence"][arm])[0] for arm in arms]
        ax.plot(labels, means, marker=marker, ls=linestyle, lw=1.6, color=BLACK, label={"expanded200": "All 200", "previous60": "Previous 60", "added140": "New 140"}[subset])
    for x_pos, arm in enumerate(arms):
        ax.scatter(x_pos, _stats(report["subsets"]["expanded200"]["convergence"][arm])[0], color=COLORS[arm], s=48, zorder=3)
    ax.set_ylabel("Mean iterations among successes")
    ax.set_title("Iteration cost")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=7)
    _panel(ax, "B")

    ax = axes[1, 0]
    values = [
        len(report["common_star_slugs"]["all_four"]),
        len(set(report["common_star_slugs"]["all_four"]) & set(report["common_star_slugs"]["previous60"])),
        len(set(report["common_star_slugs"]["all_four"]) & set(report["common_star_slugs"]["added140"])),
    ]
    bars = ax.bar(("All 200", "Previous 60", "New 140"), values, color=(BLUE, ORANGE, PURPLE))
    ax.set_ylabel("Stars converged in all four arms")
    ax.set_title("Strict common samples")
    ax.set_ylim(0, max(values + [1]) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, str(value), ha="center", weight="bold")
    _panel(ax, "C")

    ax = axes[1, 1]
    for arm in arms:
        fractions = [report["subsets"][subset]["convergence"][arm]["converged_fraction"] for subset in subsets]
        ax.plot(("All 200", "Previous 60", "New 140"), fractions, marker="o", lw=1.8, color=COLORS[arm], label=SHORT_LABELS[arm])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Converged fraction")
    ax.set_title("Old versus new stars")
    ax.legend(frameon=False, fontsize=7)
    _panel(ax, "D")
    _footer(fig, 2)
    pdf.savefig(fig)
    plt.close(fig)


def _spectral_entry(report: dict, subset: str, candidate: str, metric: str) -> dict | None:
    entry = report.get("spectra", {}).get(subset, {}).get(candidate)
    return None if entry is None else entry.get(metric)


def _spectral_page(pdf: PdfPages, report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.82, wspace=0.30, hspace=0.42)
    _header(fig, "Synthetic spectrum statistics", "400-900 nm at R=20,000; each number uses the common gateable stars for that subset")
    subsets = ("expanded200", "previous60", "added140")
    subset_labels = ("All 200", "Previous 60", "New 140")
    candidates = CANDIDATES
    x = np.arange(len(subsets))
    width = 0.24

    ax = axes[0, 0]
    for offset, candidate in zip((-width, 0, width), candidates):
        values = [
            (_spectral_entry(report, subset, candidate, "normalized_flux") or {}).get("median_max", np.nan)
            for subset in subsets
        ]
        bars = ax.bar(x + offset, values, width, color=COLORS[candidate], label=SHORT_LABELS[candidate])
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value * 1.2, f"{value:.1e}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.axhline(BAR, color=RED, ls="--", lw=1.0, label="0.005 threshold")
    ax.set_yscale("log")
    ax.set_xticks(x, subset_labels)
    ax.set_ylabel("Median max normalized-flux difference")
    ax.set_title("Median spectral difference")
    ax.legend(frameon=False, fontsize=7)
    _panel(ax, "A")

    ax = axes[0, 1]
    for offset, candidate in zip((-width, 0, width), candidates):
        values = [
            (_spectral_entry(report, subset, candidate, "normalized_flux") or {}).get("stars_over_bar", np.nan)
            for subset in subsets
        ]
        ax.bar(x + offset, values, width, color=COLORS[candidate], label=SHORT_LABELS[candidate])
    ax.set_xticks(x, subset_labels)
    ax.set_ylabel("Stars above 0.005")
    ax.set_title("Normalized-flux threshold counts")
    ax.legend(frameon=False, fontsize=7)
    _panel(ax, "B")

    ax = axes[1, 0]
    for offset, candidate in zip((-width, 0, width), candidates):
        values = [
            (_spectral_entry(report, subset, candidate, "flux_total") or {}).get("median_max", np.nan)
            for subset in subsets
        ]
        ax.bar(x + offset, values, width, color=COLORS[candidate], label=SHORT_LABELS[candidate])
    ax.axhline(BAR, color=RED, ls="--", lw=1.0)
    ax.set_yscale("log")
    ax.set_xticks(x, subset_labels)
    ax.set_ylabel("Median max total-flux difference")
    ax.set_title("Absolute-flux comparison")
    ax.legend(frameon=False, fontsize=7)
    _panel(ax, "C")

    ax = axes[1, 1]
    ax.axis("off")
    rows = []
    for candidate in candidates:
        pair = report.get("spectra", {}).get("expanded200", {}).get(candidate)
        entry = None if pair is None else pair.get("normalized_flux")
        if entry is None:
            rows.append([SHORT_LABELS[candidate], "n/a", "n/a", "n/a"])
        else:
            rows.append([
                SHORT_LABELS[candidate],
                f"{entry['median_max']:.3e}",
                f"{entry['max']:.3e}",
                f"{entry['stars_over_bar']} / {pair.get('star_count', 'n/a')}",
            ])
    table = ax.table(cellText=rows, colLabels=["Candidate", "Median", "Largest", "> 0.005"], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(("#FFF1E8", "#EEEEEE", "#F4EAF2")[row - 1])
    ax.set_title("All-200 normalized-flux gate", pad=18)
    _panel(ax, "D")
    _footer(fig, 3)
    pdf.savefig(fig)
    plt.close(fig)


def _profile_page(pdf: PdfPages, report: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.13, top=0.82, wspace=0.30, hspace=0.44)
    _header(fig, "Final atmosphere structure", "Comparison to the six-field production solution on the expanded common sample")
    structure = report.get("final_structure", {}).get("expanded200", {})
    fields = ("column_mass", "temperature", "gas_pressure", "electron_density", "rosseland_opacity", "radiative_acceleration")
    field_labels = ("m", "T", "P", "n_e", "kappa_R", "g_rad")
    available = [field for field in fields if any(field in structure.get(arm, {}).get("fields", {}) for arm in CANDIDATES)]
    plot_fields = available[:4]
    x = np.arange(len(plot_fields))
    width = 0.25
    ax = axes[0, 0]
    for offset, arm in zip((-width, 0, width), CANDIDATES):
        values = [structure.get(arm, {}).get("fields", {}).get(field, {}).get("median", np.nan) for field in plot_fields]
        ax.bar(x + offset, values, width, color=COLORS[arm], label=SHORT_LABELS[arm])
    ax.set_xticks(x, [field_labels[fields.index(field)] for field in plot_fields])
    ax.set_yscale("log")
    ax.set_ylabel("Median absolute difference")
    ax.set_title("Stored final fields: median")
    ax.legend(frameon=False, fontsize=7)
    _panel(ax, "A")

    ax = axes[0, 1]
    for offset, arm in zip((-width, 0, width), CANDIDATES):
        values = [structure.get(arm, {}).get("fields", {}).get(field, {}).get("p95", np.nan) for field in plot_fields]
        ax.bar(x + offset, values, width, color=COLORS[arm], label=SHORT_LABELS[arm])
    ax.set_xticks(x, [field_labels[fields.index(field)] for field in plot_fields])
    ax.set_yscale("log")
    ax.set_ylabel("95th-percentile absolute difference")
    ax.set_title("Stored final fields: p95")
    ax.legend(frameon=False, fontsize=7)
    _panel(ax, "B")

    ax = axes[1, 0]
    ax.axis("off")
    rows = []
    for field, label in zip(fields, field_labels):
        counts = [structure.get(arm, {}).get("field_star_count", {}).get(field, 0) for arm in CANDIDATES]
        rows.append([label, *[str(value) for value in counts]])
    table = ax.table(cellText=rows, colLabels=["Field", "Two-field", "Grey", "Interpolated"], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F3F3F3" if row % 2 else "#FFFFFF")
    ax.set_title("Stars with a finite comparison for each field", pad=18)
    _panel(ax, "C")

    ax = axes[1, 1]
    ax.axis("off")
    fig.text(
        0.53,
        0.28,
        "The reusable 400-star two-field and six-field runs saved the structured product fields m, T, P, and n_e. "
        "They did not save the solver's final kappa_R and g_rad arrays. Those two fields are therefore not fabricated: "
        "the table reports their availability, and the plots use only fields present on both sides.",
        fontsize=10,
        color=GREY_COLOR,
        wrap=True,
    )
    _panel(ax, "D")
    _footer(fig, 4)
    pdf.savefig(fig)
    plt.close(fig)


def _spectral_paths(run_root: Path, slug: str) -> dict[str, Path]:
    return {arm: run_root / "spectra" / arm / f"{slug}.npz" for arm in ARMS}


def _representatives(report: dict, result_root: Path, count: int = 6) -> list[str]:
    common = set(report.get("common_star_slugs", {}).get("all_four", []))
    scores: dict[str, float] = {slug: 0.0 for slug in common}
    for candidate in CANDIDATES:
        path = result_root / f"spectral_{candidate}_vs_six.json"
        if not path.is_file():
            continue
        for row in _load_json(path).get("per_star", []):
            if row["slug"] in scores:
                scores[row["slug"]] = max(scores[row["slug"]], float(row["normalized_flux"]["max"]))
    ordered = sorted(scores, key=lambda slug: (scores[slug], slug))
    if not ordered:
        return []
    selected_positions = np.unique(np.linspace(0, len(ordered) - 1, min(count, len(ordered)), dtype=int))
    return [ordered[int(position)] for position in selected_positions]


def _representative_spectra_pages(
    pdf: PdfPages,
    report: dict,
    run_root: Path,
    result_root: Path,
    start_page: int,
) -> tuple[list[str], int]:
    reps = _representatives(report, result_root, count=6)
    page = start_page
    for group_start in range(0, len(reps), 2):
        group = reps[group_start : group_start + 2]
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.82, wspace=0.28, hspace=0.42)
        _header(
            fig,
            "Representative narrow-band spectra",
            f"Stars {group_start + 1}-{group_start + len(group)} of {len(reps)}; 5 nm windows; curves compared to the six-field solution",
        )
        axes = np.asarray(axes).reshape(2, 2)
        for row, slug in enumerate(group):
            paths = _spectral_paths(run_root, slug)
            spectra = {arm: _load_npz(paths[arm]) for arm in ARMS}
            wavelength = spectra[SIX]["wavelength_nm"]
            reference = spectra[SIX]["normalized_flux"]
            residuals = [np.abs(spectra[arm]["normalized_flux"] - reference) for arm in CANDIDATES]
            all_residual = np.maximum.reduce(residuals)
            centre = float(wavelength[int(np.argmax(all_residual))])
            window = np.abs(wavelength - centre) <= 2.5

            ax = axes[row, 0]
            for arm in ARMS:
                ax.plot(
                    wavelength[window],
                    spectra[arm]["normalized_flux"][window],
                    color=COLORS[arm],
                    ls=LINESTYLES[arm],
                    lw=1.5,
                    label=SHORT_LABELS[arm],
                )
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Normalized flux")
            ax.set_title(f"{slug} near {centre:.1f} nm")
            ax.legend(frameon=False, ncol=2, fontsize=7)

            ax = axes[row, 1]
            for arm in CANDIDATES:
                ax.plot(
                    wavelength[window],
                    (spectra[arm]["normalized_flux"] - reference)[window],
                    color=COLORS[arm],
                    ls=LINESTYLES[arm],
                    lw=1.5,
                    label=f"{SHORT_LABELS[arm]} - six",
                )
            ax.axhline(0.0, color=BLACK, lw=0.8)
            ax.axhline(BAR, color=RED, ls="--", lw=0.9)
            ax.axhline(-BAR, color=RED, ls="--", lw=0.9, label="+/- 0.005")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Normalized-flux residual")
            ax.set_title("Residual to six-field solution")
            ax.legend(frameon=False, ncol=2, fontsize=7)
        if len(group) == 1:
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")
        fig.text(0.08, 0.045, "The window is centered on the largest residual among the three candidate initializers for that star.", fontsize=8, color=GREY_COLOR)
        _footer(fig, page)
        pdf.savefig(fig)
        plt.close(fig)
        page += 1
    return reps, page


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    if not args.summary.is_file():
        raise SystemExit(f"missing summary: {args.summary}")
    report = _load_json(args.summary)
    common = report.get("common_star_slugs", {}).get("all_four", [])
    if not common:
        raise SystemExit("no four-arm common stars available for the PDF")
    reps = _representatives(report, args.result_root, count=6)
    missing = [path for slug in reps for path in _spectral_paths(args.run_root, slug).values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing representative spectra: {missing[:5]}")
    _configure_style()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "Expanded four atmosphere initializer comparison",
        "Author": "Payne-Zero reduced-state validation",
        "Subject": "Convergence, final structures, and representative synthetic spectra",
        "Keywords": "stellar atmosphere, interpolation, reduced state, Payne-Zero",
    }
    with PdfPages(args.out, metadata=metadata) as pdf:
        _summary_page(pdf, report)
        _convergence_page(pdf, report)
        _spectral_page(pdf, report)
        _profile_page(pdf, report)
        reps, next_page = _representative_spectra_pages(pdf, report, args.run_root, args.result_root, 5)
    manifest = args.out.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "format": "payne_zero_expanded_four_initializer_report_v1",
                "pdf": str(args.out),
                "pdf_sha256": _sha256(args.out),
                "summary": str(args.summary),
                "summary_sha256": _sha256(args.summary),
                "four_arm_common_star_count": len(common),
                "representative_slugs": reps,
                "representative_count": len(reps),
                "representative_window_nm": 5.0,
                "spectrum_window_nm": [400.0, 900.0],
                "spectrum_resolution": 20000.0,
                "page_count": next_page - 1,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.out}")
    print(f"wrote {manifest}")
    print(f"representative spectra: {len(reps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
