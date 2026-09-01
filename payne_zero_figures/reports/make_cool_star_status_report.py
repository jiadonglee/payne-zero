"""Create a failure-aware 3000/3500/3800 K comparison PDF."""

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


def _style() -> None:
    style.configure("COOL_STAR", **{"legend.fontsize": 7})



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
SUCCESS_TEMPERATURES = (3500.0, 3800.0)
ALL_TEMPERATURES = (3000.0, 3500.0, 3800.0)
ARMS = ("six", "two", "marcs_reduced", "marcs_full")


def _prefix(temperature: float) -> str:
    return f"t{float(temperature):05.0f}"


def _structure(data: dict[str, np.ndarray], temperature: float, arm: str, field: str) -> np.ndarray:
    return data[f"{_prefix(temperature)}__{arm}__{field}"]


def _raw(data: dict[str, np.ndarray], temperature: float, field: str) -> np.ndarray:
    return data[f"{_prefix(temperature)}__marcs_raw__{field}"]


def _spectrum(root: Path, temperature: float, arm: str) -> dict[str, np.ndarray]:
    return _load_npz(root / "spectra" / f"{temperature:.0f}" / f"{arm}.npz")


def _load_failures(path: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in sorted(path.glob("*.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        records = payload.get("attempts") or payload.get("records") or [payload]
        for record in records:
            diagnostics = record.get("solver_diagnostics") or {}
            first = diagnostics.get("first_iteration") or {}
            final = diagnostics.get("final_iteration") or {}
            rows.append(
                {
                    "source": summary_path.stem,
                    "method": record.get("method", summary_path.stem),
                    "target": record.get("target_temperature"),
                    "step": record.get("requested_step"),
                    "status": record.get("status"),
                    "iterations": record.get("iterations"),
                    "seconds": record.get("seconds", record.get("wall_seconds", record.get("wall_seconds_including_setup"))),
                    "first_deep": first.get("deep_layer_relative_temperature_change"),
                    "final_deep": final.get("deep_layer_relative_temperature_change"),
                    "first_flux": first.get("maximum_absolute_flux_error_percent"),
                    "final_flux": final.get("maximum_absolute_flux_error_percent"),
                }
            )
    return rows


def _metric(values: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values, dtype=np.float64))))


def _structure_metrics(data: dict[str, np.ndarray], temperature: float) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for arm in ("two", "marcs_reduced", "marcs_full"):
        result[arm] = {}
        for field in ("temperature", "gas_pressure", "electron_density", "mass_density"):
            base = _structure(data, temperature, "six", field)
            candidate = _structure(data, temperature, arm, field)
            result[arm][field] = _metric(candidate / np.maximum(np.abs(base), 1.0e-300) - 1.0)
    return result


def _header(fig: mpl.figure.Figure, title: str, subtitle: str, page: int) -> None:
    fig.text(0.055, 0.945, title, fontsize=18, weight="bold", color="#222222")
    fig.text(0.055, 0.912, subtitle, fontsize=9, color="#666666")
    fig.text(0.99, 0.015, f"Payne-Zero cool-star status report | 2026-08-17 | Page {page}", ha="right", va="bottom", fontsize=7, color="#777777")


def _panel(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=10)


def _table_style(table) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#222222")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#F2F4F5" if row % 2 else "#E7EEF2")


def _plot_structure(ax: mpl.axes.Axes, data: dict[str, np.ndarray], temperature: float, field: str, *, legend: bool = False) -> None:
    x = np.log10(_structure(data, temperature, "six", "column_mass"))
    for arm in ARMS:
        ax.plot(x, _structure(data, temperature, arm, field), color=COLORS[arm], lw=1.0 if arm == "six" else 0.85, ls=LINESTYLES[arm], label=LABELS[arm])
    raw_x = np.log10(_raw(data, temperature, "column_mass"))
    ax.plot(raw_x, _raw(data, temperature, field), color=COLORS["raw"], lw=0.75, ls="--", marker=".", ms=2.2, label=LABELS["raw"])
    ax.set_xlabel(r"$\log_{10} m$ (g cm$^{-2}$)")
    if legend:
        ax.legend(frameon=False, ncol=2, fontsize=6.5, loc="best")


def _page_summary(pdf: PdfPages, summary: dict, failures: list[dict]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _header(fig, "Cool-star 3000/3500/3800 K status and comparison", "Solar-composition pilot track: log g = 5.0, [M/H] = 0.0, vmic = 1 km/s", 1)
    ax = fig.add_axes([0.055, 0.62, 0.89, 0.20])
    ax.axis("off")
    rows = [
        ["3800 K", "PASS", "PASS", "PASS", "400-900 nm spectrum"],
        ["3500 K", "PASS", "PASS", "PASS", "400-900 nm spectrum"],
        ["3000 K", "NO", "NO", "NO formal endpoint", "raw MARCS structure only"],
    ]
    table = ax.table(cellText=rows, colLabels=["Target", "Two-field", "Six-field", "MARCS-started", "Deliverable"], colWidths=[0.14, 0.16, 0.16, 0.25, 0.25], cellLoc="center", loc="center")
    _table_style(table)
    fig.text(0.055, 0.55, "Result", fontsize=13, weight="bold")
    fig.text(0.055, 0.505, "The 3500 K and 3800 K products are formal, finite, monotonic 80-layer Payne-Zero endpoints with common-grid spectra. A formal 3000 K Payne-Zero endpoint was not obtained under the stated 30-iteration solver cap: 250, 100, 50, and 25 K first steps all failed for both carried-state representations. Independent H2 analytic exact-seed checks also failed.", fontsize=9.8, linespacing=1.45, wrap=True, va="top")
    fig.text(0.055, 0.375, "How to read the 3000 K panels", fontsize=13, weight="bold")
    fig.text(0.055, 0.33, "The 3000 K plots show the native 56-layer MARCS benchmark and the measured solver failure boundary. They are not presented as a converged Payne-Zero solution or spectrum. This is the scientifically relevant result for the current model: the cool-star test reaches 3500 K, but the present evidence does not support claiming 3000 K coverage.", fontsize=9.8, color="#444444", linespacing=1.45, wrap=True, va="top")
    fig.text(0.055, 0.18, "Reference convention", fontsize=13, weight="bold")
    fig.text(0.055, 0.135, "The six-field continuation is an internal numerical reference, not physical truth. Raw MARCS is used only for structure because it is native 56-layer data; MARCS-started products are complete Payne-Zero endpoints only where the solver formally converged.", fontsize=9.8, color="#444444", linespacing=1.45, wrap=True, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _page_success_spectra(pdf: PdfPages, root: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.82, wspace=0.26, hspace=0.40)
    _header(fig, "Formal solutions: synthetic spectra", "400-900 nm, R = 20,000, float64, molecular lines enabled; residuals use the six-field continuation", 2)
    for index, temperature in enumerate((3800.0, 3500.0)):
        spectra = {arm: _spectrum(root, temperature, arm) for arm in ARMS}
        wave = spectra["six"]["wavelength_nm"]
        reference = spectra["six"]["normalized_flux"]
        ax = axes[index, 0]
        for arm in ARMS:
            ax.plot(wave, spectra[arm]["normalized_flux"], color=COLORS[arm], lw=1.0 if arm == "six" else 0.75, ls=LINESTYLES[arm], label=LABELS[arm])
        ax.set(xlabel="Wavelength (nm)", ylabel="Normalized flux", title=f"{temperature:.0f} K")
        if index == 0:
            ax.legend(frameon=False, ncol=2, fontsize=6.3, loc="lower right")
        _panel(ax, "ABCD"[2 * index])
        ax = axes[index, 1]
        for arm in ("two", "marcs_reduced", "marcs_full"):
            ax.plot(wave, spectra[arm]["normalized_flux"] - reference, color=COLORS[arm], lw=0.8, ls=LINESTYLES[arm], label=LABELS[arm])
        ax.axhline(0.0, color="#555555", lw=0.5)
        ax.axhline(5.0e-3, color="#CC3311", lw=0.7, ls=":")
        ax.axhline(-5.0e-3, color="#CC3311", lw=0.7, ls=":")
        ax.set(xlabel="Wavelength (nm)", ylabel="Delta normalized flux", title=f"Residual from six-field: {temperature:.0f} K")
        _panel(ax, "ABCD"[2 * index + 1])
    pdf.savefig(fig)
    plt.close(fig)


def _page_success_structure(pdf: PdfPages, data: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.82, wspace=0.25, hspace=0.42)
    _header(fig, "Formal solutions: atmosphere structure", "Solid/dashed/colored curves are complete Payne-Zero products; grey points are native MARCS layers", 3)
    fields = ("temperature", "gas_pressure", "electron_density")
    labels = ("Temperature (K)", "Gas pressure (dyn cm$^{-2}$)", "Electron density (cm$^{-3}$)")
    for row, temperature in enumerate((3800.0, 3500.0)):
        for col, (field, ylabel) in enumerate(zip(fields, labels)):
            ax = axes[row, col]
            _plot_structure(ax, data, temperature, field, legend=row == 0 and col == 0)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{temperature:.0f} K")
            if field != "temperature":
                ax.set_yscale("log")
            _panel(ax, "ABCDEF"[row * 3 + col])
    pdf.savefig(fig)
    plt.close(fig)


def _page_raw_marcs(pdf: PdfPages, raw: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11.69, 8.27))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.82, wspace=0.25, hspace=0.42)
    _header(fig, "Native MARCS structural benchmark", "56 native layers at the same solar-composition log g = 5.0 track; this page is not a Payne-Zero convergence result", 4)
    fields = ("temperature", "gas_pressure", "electron_density")
    labels = ("Temperature (K)", "Gas pressure (dyn cm$^{-2}$)", "Electron density (cm$^{-3}$)")
    for row, temperature in enumerate((3000.0, 3500.0, 3800.0)):
        x = np.log10(raw[f"{_prefix(temperature)}__marcs_raw__column_mass"])
        for col, (field, ylabel) in enumerate(zip(fields, labels)):
            ax = axes[row, col]
            ax.plot(x, raw[f"{_prefix(temperature)}__marcs_raw__{field}"], color=COLORS["raw"], lw=1.0, ls="--", marker=".", ms=2.3)
            ax.set_xlabel(r"$\log_{10} m$ (g cm$^{-2}$)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{temperature:.0f} K")
            if field != "temperature":
                ax.set_yscale("log")
            _panel(ax, "ABCDEFGHI"[row * 3 + col])
    pdf.savefig(fig)
    plt.close(fig)


def _page_failure(pdf: PdfPages, failures: list[dict]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _header(fig, "3000 K solver boundary", "Each row is a real solver attempt; no row produced a structured converged endpoint", 5)
    rows = []
    for item in failures:
        target = "-" if item["target"] is None else f"{float(item['target']):.0f}"
        step = "-" if item["step"] is None else f"{float(item['step']):.0f}"
        first = "-" if item["first_deep"] is None else f"{float(item['first_deep']):.2e}"
        final = "-" if item["final_deep"] is None else f"{float(item['final_deep']):.2e}"
        flux = "-" if item["final_flux"] is None else f"{float(item['final_flux']):.2e}"
        rows.append([item["source"], target, step, str(item["iterations"] or "-"), first, final, flux])
    ax = fig.add_axes([0.055, 0.49, 0.89, 0.31])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Attempt", "Target K", "Step K", "Iter", "First dT", "Final dT", "Final flux err %"], colWidths=[0.24, 0.10, 0.10, 0.08, 0.14, 0.14, 0.18], cellLoc="center", loc="center")
    _table_style(table)
    fig.text(0.055, 0.42, "Interpretation", fontsize=13, weight="bold")
    fig.text(0.055, 0.375, "The first temperature step can look numerically small in the first iteration, but the iteration map later diverges: final deep-layer changes grow to percent or larger and the flux-error diagnostic becomes enormous. Because no attempt wrote a formal converged structured product, there is no valid 3000 K Payne-Zero spectrum to compare against the 3500/3800 K spectra.", fontsize=9.8, color="#444444", linespacing=1.45, wrap=True, va="top")
    fig.text(0.055, 0.205, "This is a boundary result, not a claim that no 3000 K atmosphere exists physically. It says the current initialization plus current solver settings do not reach one under the tested 30-iteration criterion.", fontsize=9.8, color="#444444", linespacing=1.45, wrap=True, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def _page_metrics(pdf: PdfPages, summary: dict, data: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    _header(fig, "Formal-solution metrics and provenance", "Maximum differences over the common spectrum or the 80-layer product grid", 6)
    rows = []
    for temperature in (3800.0, 3500.0):
        case = f"{temperature:.0f}"
        comp = summary["comparisons"][case]
        rows.append([
            case,
            f"{comp['two_vs_six']['normalized_flux']['max']:.2e}",
            f"{comp['two_vs_six']['flux_total_continuum_scaled']['max']:.2e}",
            f"{comp['marcs_reduced_vs_six']['normalized_flux']['max']:.2e}",
            f"{comp['marcs_full_vs_six']['normalized_flux']['max']:.2e}",
        ])
    ax = fig.add_axes([0.055, 0.64, 0.89, 0.18])
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Target", "Two vs six max |dF|", "Two vs six total/cont.", "MARCS m,T vs six", "MARCS full vs six"], colWidths=[0.12, 0.22, 0.22, 0.22, 0.22], cellLoc="center", loc="center")
    _table_style(table)
    structure_rows = []
    for temperature in (3800.0, 3500.0):
        metrics = _structure_metrics(data, temperature)
        structure_rows.append([
            f"{temperature:.0f}",
            f"{metrics['two']['temperature'] * 100:.3f}%",
            f"{metrics['two']['gas_pressure'] * 100:.3f}%",
            f"{metrics['two']['electron_density'] * 100:.3f}%",
        ])
    ax = fig.add_axes([0.055, 0.39, 0.89, 0.14])
    ax.axis("off")
    table = ax.table(cellText=structure_rows, colLabels=["Target", "Two |dT/T|", "Two |dP/P|", "Two |dne/ne|"], colWidths=[0.12, 0.29, 0.29, 0.29], cellLoc="center", loc="center")
    _table_style(table)
    fig.text(0.055, 0.33, "Contract", fontsize=12, weight="bold")
    fig.text(0.055, 0.285, f"MARCS SHA-256: {summary['marcs_raw']['sha256']}\nSpectrum grid: 400-900 nm, R = 20,000, float64, molecular lines enabled, synthesized on Garching A100.\nRaw MARCS: native 56 layers; not passed to Payne-Zero synthesis.\nProducts: 3500 K uses 250 K continuation; 3800 K uses 100 K continuation; 3000 K has no formal product under the tested cap.", fontsize=9.3, linespacing=1.45, color="#444444", va="top")
    fig.text(0.055, 0.10, "The delivered PDF intentionally distinguishes a measured convergence boundary from a physical non-existence claim.", fontsize=9.0, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    _style()
    summary = json.loads((args.root / "comparison_summary.json").read_text(encoding="utf-8"))
    data = _load_npz(args.root / "structure_comparison.npz")
    raw = _load_npz(args.raw)
    failures = _load_failures(args.failures)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        _page_summary(pdf, summary, failures)
        _page_success_spectra(pdf, args.root)
        _page_success_structure(pdf, data)
        _page_raw_marcs(pdf, raw)
        _page_failure(pdf, failures)
        _page_metrics(pdf, summary, data)
    metrics_path = args.out.with_name("cool_star_status_metrics.json")
    metrics_path.write_text(json.dumps({"failures": failures, "success_temperatures": list(SUCCESS_TEMPERATURES)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
