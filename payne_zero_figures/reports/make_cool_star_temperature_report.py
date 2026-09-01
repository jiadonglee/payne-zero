"""Create the final three-temperature cool-star comparison PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from payne_zero_figures import style  # noqa: E402
from payne_zero_figures.data import load_npz as _load_npz  # noqa: E402


def _configure_style() -> None:
    style.configure("COOL_STAR")



ARM_ORDER = ("six", "two", "marcs_reduced", "marcs_full")
PLOT_ARMS = ("two", "marcs_reduced", "marcs_full")
COLORS = {
    "six": "#222222",
    "two": "#0072B2",
    "marcs_reduced": "#D55E00",
    "marcs_full": "#009E73",
    "raw": "#777777",
}
LABELS = {
    "six": "Six-field continuation",
    "two": "Two-field continuation",
    "marcs_reduced": "MARCS-started, m,T rematerialized",
    "marcs_full": "MARCS-started, full carry",
    "raw": "Raw MARCS (56 layers)",
}
LINESTYLES = {"six": "-", "two": "--", "marcs_reduced": "-.", "marcs_full": ":"}
FIELD_LABELS = {
    "temperature": ("Temperature (K)", False),
    "gas_pressure": ("Gas pressure (dyn cm$^{-2}$)", True),
    "electron_density": ("Electron density (cm$^{-3}$)", True),
    "mass_density": ("Mass density (g cm$^{-3}$)", True),
}


def _case_key(temperature: float) -> str:
    return f"{float(temperature):.0f}"


def _load_inputs(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    summary = json.loads((root / "comparison_summary.json").read_text(encoding="utf-8"))
    structures = _load_npz(root / "structure_comparison.npz")
    return summary, structures


def _prefix(temperature: float) -> str:
    return f"t{float(temperature):05.0f}"


def _structure(structures: dict[str, np.ndarray], temperature: float, arm: str, field: str) -> np.ndarray:
    return structures[f"{_prefix(temperature)}__{arm}__{field}"]


def _raw_structure(structures: dict[str, np.ndarray], temperature: float, field: str) -> np.ndarray | None:
    key = f"{_prefix(temperature)}__marcs_raw__{field}"
    return structures.get(key)


def _spectrum(root: Path, temperature: float, arm: str) -> dict[str, np.ndarray]:
    return _load_npz(root / "spectra" / _case_key(temperature) / f"{arm}.npz")


def _profile_metrics(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "median": float(np.median(absolute)),
        "p95": float(np.percentile(absolute, 95.0)),
        "max": float(np.max(absolute)),
    }


def _structure_metrics(
    temperatures: list[float], structures: dict[str, np.ndarray]
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    result: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for temperature in temperatures:
        case = _case_key(temperature)
        case_result: dict[str, dict[str, dict[str, float]]] = {}
        base_m = _structure(structures, temperature, "six", "column_mass")
        base_log_m = np.log10(base_m)
        for arm in ("two", "marcs_reduced", "marcs_full"):
            arm_result: dict[str, dict[str, float]] = {}
            for field in ("column_mass", "temperature", "gas_pressure", "electron_density", "mass_density"):
                candidate = _structure(structures, temperature, arm, field)
                baseline = _structure(structures, temperature, "six", field)
                if field == "column_mass":
                    delta = np.log10(candidate) - np.log10(baseline)
                else:
                    delta = candidate / np.maximum(np.abs(baseline), 1.0e-300) - 1.0
                arm_result[field] = _profile_metrics(delta)
            case_result[arm] = arm_result

        raw_result: dict[str, dict[str, float]] = {}
        raw_m = _raw_structure(structures, temperature, "column_mass")
        if raw_m is not None:
            raw_log_m = np.log10(raw_m)
            overlap = (raw_log_m >= base_log_m[0]) & (raw_log_m <= base_log_m[-1])
            for field in ("temperature", "gas_pressure", "electron_density"):
                raw_values = _raw_structure(structures, temperature, field)
                if raw_values is None:
                    continue
                base = np.interp(
                    raw_log_m[overlap],
                    base_log_m,
                    _structure(structures, temperature, "six", field),
                )
                raw_result[field] = _profile_metrics(
                    raw_values[overlap] / np.maximum(np.abs(base), 1.0e-300) - 1.0
                )
            raw_result["overlap_layers"] = {"max": float(np.count_nonzero(overlap))}
        case_result["raw"] = raw_result
        result[case] = case_result
    return result


def _spectrum_metric_table(summary: dict, temperatures: list[float]) -> list[list[str]]:
    rows = []
    for temperature in temperatures:
        case = _case_key(temperature)
        comparisons = summary["comparisons"][case]
        two = comparisons["two_vs_six"]
        marcs_red = comparisons["marcs_reduced_vs_six"]
        marcs_full = comparisons["marcs_full_vs_six"]
        rows.append(
            [
                f"{temperature:.0f}",
                f"{two['normalized_flux']['max']:.2e}",
                f"{two['flux_total_continuum_scaled']['max']:.2e}",
                f"{marcs_red['normalized_flux']['max']:.2e}",
                f"{marcs_full['normalized_flux']['max']:.2e}",
            ]
        )
    return rows


def _header(fig: mpl.figure.Figure, title: str, subtitle: str, page: int) -> None:
    fig.text(0.055, 0.945, title, fontsize=18, weight="bold", color="#222222")
    fig.text(0.055, 0.912, subtitle, fontsize=9, color="#666666")
    fig.text(
        0.99,
        0.015,
        f"Payne-Zero cool-star comparison | 2026-08-17 | Page {page}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#777777",
    )


def _panel(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=10)


def _plot_lines(
    ax: mpl.axes.Axes,
    x: np.ndarray,
    structures: dict[str, np.ndarray],
    temperature: float,
    field: str,
    *,
    raw: bool = True,
    log_y: bool = False,
    legend: bool = False,
) -> None:
    for arm in ARM_ORDER:
        ax.plot(
            x,
            _structure(structures, temperature, arm, field),
            color=COLORS[arm],
            lw=1.0 if arm == "six" else 0.85,
            ls=LINESTYLES[arm],
            label=LABELS[arm],
        )
    raw_x = _raw_structure(structures, temperature, "column_mass")
    raw_y = _raw_structure(structures, temperature, field)
    if raw and raw_x is not None and raw_y is not None:
        ax.plot(
            np.log10(raw_x),
            raw_y,
            color=COLORS["raw"],
            lw=0.75,
            ls="--",
            marker=".",
            ms=2.3,
            label=LABELS["raw"],
        )
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10} m$ (g cm$^{-2}$)")
    if legend:
        ax.legend(frameon=False, loc="best", ncol=2, fontsize=6.7)


def _page_summary(
    pdf: PdfPages,
    summary: dict,
    metrics: dict,
    temperatures: list[float],
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _header(
        fig,
        "Cool-star three-temperature comparison",
        "Solar-composition pilot track: log g = 5.0, [M/H] = 0.0, [alpha/M] = [C/M] = 0.0, vmic = 1 km/s",
        1,
    )
    ax = fig.add_axes([0.055, 0.57, 0.89, 0.25])
    ax.axis("off")
    rows = []
    for temperature in temperatures:
        case = _case_key(temperature)
        m = metrics[case]
        rows.append(
            [
                f"{temperature:.0f} K",
                f"{m['two']['temperature']['max'] * 100:.3f}%",
                f"{m['two']['electron_density']['max'] * 100:.3f}%",
                f"{m['marcs_reduced']['temperature']['max'] * 100:.2f}%",
                f"{summary['comparisons'][case]['two_vs_six']['normalized_flux']['max']:.2e}",
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["Target", "Two-field max |dT/T|", "Two-field max |dne/ne|", "MARCS red. max |dT/T|", "Two vs six max |dF|"],
        colWidths=[0.12, 0.22, 0.22, 0.22, 0.22],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#222222")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F2F4F5" if row % 2 else "#E7EEF2")

    fig.text(0.055, 0.51, "What is being compared", fontsize=13, weight="bold")
    fig.text(
        0.055,
        0.465,
        "The six-field continuation is the internal fixed-point reference. The two-field curve carries only column mass and temperature between solver steps, then reconstructs pressure, electron density, opacity, and radiative acceleration. The two paths are 250 K from 4000 to 3500 K, 100 K from 4000 to 3800 K, and 250 K from 3500 to 3000 K.",
        fontsize=9.7,
        linespacing=1.45,
        wrap=True,
    )
    fig.text(0.055, 0.345, "Independent MARCS check", fontsize=13, weight="bold")
    fig.text(
        0.055,
        0.30,
        "The MARCS-started endpoints use native-grid paths 3900 -> 3800 K and 3750 -> 3500 K. The native 3200 K MARCS start did not converge in the 30-iteration check, so the 3000 K MARCS comparison continues the independently MARCS-derived 3500 K endpoint with 100 K steps. Raw MARCS remains a native 56-layer structural diagnostic; it is not used as a Payne-Zero spectrum-synthesis input.",
        fontsize=9.7,
        linespacing=1.45,
        wrap=True,
    )
    fig.text(0.055, 0.18, "Bottom line", fontsize=13, weight="bold")
    two_values = [summary["comparisons"][_case_key(t)]["two_vs_six"]["normalized_flux"]["max"] for t in temperatures]
    fig.text(
        0.055,
        0.135,
        "The requested three solutions are finite, monotonic 80-layer Payne-Zero products and are synthesized on the same 400-900 nm, R=20,000 grid. The two-field agreement with the six-field continuation is reported as a measured result for this one pilot track, not as proof that the emulator is generally valid below 4000 K.",
        fontsize=9.7,
        color="#444444",
        linespacing=1.45,
        wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _page_structure_temperature(
    pdf: PdfPages,
    structures: dict[str, np.ndarray],
    temperatures: list[float],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27), sharex=False)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.82, wspace=0.26, hspace=0.42)
    _header(fig, "Atmosphere structure: temperature and electron density", "All Payne-Zero products have 80 layers; grey points are the native 56-layer MARCS node", 2)
    for index, temperature in enumerate(temperatures):
        x = np.log10(_structure(structures, temperature, "six", "column_mass"))
        ax = axes[index, 0]
        _plot_lines(ax, x, structures, temperature, "temperature", legend=index == 0)
        ax.set_ylabel("Temperature (K)")
        ax.set_title(f"{temperature:.0f} K")
        _panel(ax, "ABCDEF"[2 * index])
        ax = axes[index, 1]
        _plot_lines(ax, x, structures, temperature, "electron_density", log_y=True, legend=False)
        ax.set_ylabel("Electron density (cm$^{-3}$)")
        ax.set_title(f"{temperature:.0f} K")
        _panel(ax, "ABCDEF"[2 * index + 1])
    pdf.savefig(fig)
    plt.close(fig)


def _page_structure_pressure_density(
    pdf: PdfPages,
    structures: dict[str, np.ndarray],
    temperatures: list[float],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27), sharex=False)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.82, wspace=0.26, hspace=0.42)
    _header(fig, "Atmosphere structure: pressure and mass density", "The four dependent fields are recomputed by the physical reconstruction path before the solver is run", 3)
    for index, temperature in enumerate(temperatures):
        x = np.log10(_structure(structures, temperature, "six", "column_mass"))
        ax = axes[index, 0]
        _plot_lines(ax, x, structures, temperature, "gas_pressure", log_y=True, legend=index == 0)
        ax.set_ylabel("Gas pressure (dyn cm$^{-2}$)")
        ax.set_title(f"{temperature:.0f} K")
        _panel(ax, "ABCDEF"[2 * index])
        ax = axes[index, 1]
        _plot_lines(ax, x, structures, temperature, "mass_density", log_y=True, legend=False)
        ax.set_ylabel("Mass density (g cm$^{-3}$)")
        ax.set_title(f"{temperature:.0f} K")
        _panel(ax, "ABCDEF"[2 * index + 1])
    pdf.savefig(fig)
    plt.close(fig)


def _page_structure_residuals(
    pdf: PdfPages,
    structures: dict[str, np.ndarray],
    temperatures: list[float],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27), sharex=False)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.82, wspace=0.26, hspace=0.42)
    _header(fig, "Structure residuals relative to the six-field continuation", "Residuals are candidate/reference - 1; column mass uses delta log10(m)", 4)
    for index, temperature in enumerate(temperatures):
        x = np.log10(_structure(structures, temperature, "six", "column_mass"))
        for column, field in enumerate(("temperature", "electron_density")):
            ax = axes[index, column]
            reference = _structure(structures, temperature, "six", field)
            for arm in PLOT_ARMS:
                candidate = _structure(structures, temperature, arm, field)
                delta = candidate / np.maximum(np.abs(reference), 1.0e-300) - 1.0
                ax.plot(x, delta, color=COLORS[arm], lw=0.9, ls=LINESTYLES[arm], label=LABELS[arm])
            raw_x = _raw_structure(structures, temperature, "column_mass")
            raw_y = _raw_structure(structures, temperature, field)
            if raw_x is not None and raw_y is not None:
                raw_base = np.interp(x=np.log10(raw_x), xp=x, fp=reference)
                ax.plot(np.log10(raw_x), raw_y / np.maximum(np.abs(raw_base), 1.0e-300) - 1.0, color=COLORS["raw"], lw=0.75, ls="--", marker=".", ms=2.0, label=LABELS["raw"])
            ax.axhline(0.0, color="#666666", lw=0.5)
            ax.set_xlabel(r"$\log_{10} m$ (g cm$^{-2}$)")
            ax.set_ylabel("Relative residual")
            ax.set_title(f"{temperature:.0f} K: {field.replace('_', ' ')}")
            ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1.0, decimals=1))
            if index == 0 and column == 0:
                ax.legend(frameon=False, fontsize=6.5, ncol=2)
            _panel(ax, "ABCDEF"[2 * index + column])
    pdf.savefig(fig)
    plt.close(fig)


def _page_spectra(
    pdf: PdfPages,
    root: Path,
    summary: dict,
    temperatures: list[float],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.69, 8.27), sharex=False)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.82, wspace=0.26, hspace=0.42)
    _header(fig, "Synthetic spectra on a common high-resolution grid", "400-900 nm, R = 20,000, float64, molecular lines enabled; six-field continuation is the line-by-line reference", 5)
    for index, temperature in enumerate(temperatures):
        spectra = {arm: _spectrum(root, temperature, arm) for arm in ARM_ORDER}
        wavelength = spectra["six"]["wavelength_nm"]
        reference = spectra["six"]["normalized_flux"]
        ax = axes[index, 0]
        for arm in ARM_ORDER:
            ax.plot(wavelength, spectra[arm]["normalized_flux"], color=COLORS[arm], lw=1.0 if arm == "six" else 0.75, ls=LINESTYLES[arm], label=LABELS[arm])
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized flux")
        ax.set_title(f"{temperature:.0f} K")
        if index == 0:
            ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="lower right")
        _panel(ax, "ABCDEF"[2 * index])
        ax = axes[index, 1]
        for arm in PLOT_ARMS:
            delta = spectra[arm]["normalized_flux"] - reference
            ax.plot(wavelength, delta, color=COLORS[arm], lw=0.8, ls=LINESTYLES[arm], label=LABELS[arm])
        ax.axhline(0.0, color="#555555", lw=0.5)
        ax.axhline(5.0e-3, color="#CC3311", lw=0.7, ls=":")
        ax.axhline(-5.0e-3, color="#CC3311", lw=0.7, ls=":")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Delta normalized flux")
        ax.set_title(f"Residual from six-field: {temperature:.0f} K")
        _panel(ax, "ABCDEF"[2 * index + 1])
    pdf.savefig(fig)
    plt.close(fig)


def _page_metrics(
    pdf: PdfPages,
    summary: dict,
    metrics: dict,
    temperatures: list[float],
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _header(fig, "Numerical comparison and interpretation", "Maximum absolute values are reported over all wavelength pixels or all 80 layers", 6)
    spectrum_rows = _spectrum_metric_table(summary, temperatures)
    ax = fig.add_axes([0.055, 0.60, 0.89, 0.22])
    ax.axis("off")
    table = ax.table(
        cellText=spectrum_rows,
        colLabels=["Target", "Two vs six |dF|", "Two vs six total/cont.", "MARCS red. vs six |dF|", "MARCS full vs six |dF|"],
        colWidths=[0.12, 0.21, 0.21, 0.23, 0.23],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#222222")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F2F4F5" if row % 2 else "#E7EEF2")
    fig.text(0.055, 0.55, "Structure maximum residuals", fontsize=12, weight="bold")
    structure_rows = []
    for temperature in temperatures:
        case = _case_key(temperature)
        structure_rows.append(
            [
                f"{temperature:.0f}",
                f"{metrics[case]['two']['temperature']['max'] * 100:.3f}%",
                f"{metrics[case]['two']['gas_pressure']['max'] * 100:.3f}%",
                f"{metrics[case]['two']['electron_density']['max'] * 100:.3f}%",
                f"{metrics[case]['marcs_reduced']['temperature']['max'] * 100:.2f}%",
            ]
        )
    ax = fig.add_axes([0.055, 0.36, 0.89, 0.14])
    ax.axis("off")
    table = ax.table(
        cellText=structure_rows,
        colLabels=["Target", "Two |dT/T|", "Two |dP/P|", "Two |dne/ne|", "MARCS red. |dT/T|"],
        colWidths=[0.12, 0.22, 0.22, 0.22, 0.22],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#222222")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F2F4F5" if row % 2 else "#E7EEF2")
    fig.text(0.055, 0.27, "Interpretation limits", fontsize=12, weight="bold")
    fig.text(
        0.055,
        0.215,
        "These are three solar-composition pilot-track solutions. The six-field continuation is a numerical reference, not an independent physical truth. Raw MARCS is shown only in structure panels because it has a different native depth grid and was not passed to the Payne-Zero synthesizer. Agreement here supports the cool-star continuation test; it does not establish full emulator coverage for every M dwarf below 4000 K.",
        fontsize=9.4,
        color="#444444",
        linespacing=1.45,
        wrap=True,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _page_provenance(pdf: PdfPages, summary: dict, temperatures: list[float]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _header(fig, "Provenance and reproducibility", "Files and conventions used for the delivered comparison", 7)
    fig.text(0.055, 0.82, "MARCS source", fontsize=12, weight="bold")
    fig.text(
        0.055,
        0.775,
        f"SDSS_MARCS_atmospheres.h5\nSHA-256: {summary['marcs_raw']['sha256']}\nNative grid: 56 layers; displayed native nodes: 3000, 3500, 3800 K at log g = 5.0 and [M/H] = 0.0.",
        fontsize=9.5,
        linespacing=1.5,
    )
    fig.text(0.055, 0.63, "Spectrum contract", fontsize=12, weight="bold")
    contract = summary["spectrum_contract"]
    fig.text(
        0.055,
        0.585,
        f"Wavelength: {contract['wavelength_nm'][0]:.0f}-{contract['wavelength_nm'][1]:.0f} nm\nResolution: R = {contract['resolution']:,.0f}\nArithmetic: {contract['dtype']}\nMolecular lines: {contract['molecular_lines']}\nSpectrum device: {contract['device']}",
        fontsize=9.5,
        linespacing=1.5,
    )
    fig.text(0.055, 0.40, "Temperature paths", fontsize=12, weight="bold")
    rows = [
        ["3800 K", "4000 -> 3900 -> 3800 K", "MARCS 3900 -> 3800 K"],
        ["3500 K", "4000 -> 3750 -> 3500 K", "MARCS 3750 -> 3500 K"],
        ["3000 K", "3500 -> 3400 -> ... -> 3000 K", "MARCS-derived 3500 -> ... -> 3000 K"],
    ]
    ax = fig.add_axes([0.055, 0.19, 0.89, 0.17])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Target", "Continuation", "Independent MARCS path"], colWidths=[0.15, 0.38, 0.38], cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#222222")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F2F4F5" if row % 2 else "#E7EEF2")
    fig.text(0.055, 0.10, "Delivered artifact contains the comparison summary, structure archive, and four synthesized spectra per temperature under the same result root.", fontsize=8.8, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    _configure_style()
    summary, structures = _load_inputs(args.root)
    temperatures = [float(value) for value in summary["temperatures"]]
    metrics = _structure_metrics(temperatures, structures)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        _page_summary(pdf, summary, metrics, temperatures)
        _page_structure_temperature(pdf, structures, temperatures)
        _page_structure_pressure_density(pdf, structures, temperatures)
        _page_structure_residuals(pdf, structures, temperatures)
        _page_spectra(pdf, args.root, summary, temperatures)
        _page_metrics(pdf, summary, metrics, temperatures)
        _page_provenance(pdf, summary, temperatures)
    metrics_path = args.out.with_name("temperature_structure_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
