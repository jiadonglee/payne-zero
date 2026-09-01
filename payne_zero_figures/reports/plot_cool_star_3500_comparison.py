"""Plot the 3500 K two-field continuation versus six-field and MARCS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "two_field_continuation": "#0072B2",  # Okabe-Ito blue
    "six_field_continuation": "#000000",
    "marcs_started_reduced": "#D55E00",  # vermillion
    "marcs_started_full": "#009E73",  # green
    "marcs_raw": "#6E6E6E",
}
LABELS = {
    "two_field_continuation": "Two-field continuation",
    "six_field_continuation": "Six-field continuation",
    "marcs_started_reduced": "MARCS-started, rematerialized",
    "marcs_started_full": "MARCS-started, full carry",
}
PRODUCT_ARMS = tuple(LABELS)
STRUCTURE_FIELDS = ("temperature", "gas_pressure", "electron_density", "mass_density")
FIELD_LABELS = {
    "temperature": ("Temperature (K)", False),
    "gas_pressure": ("Gas pressure (dyn cm$^{-2}$)", True),
    "electron_density": ("Electron density (cm$^{-3}$)", True),
    "mass_density": ("Mass density (g cm$^{-3}$)", True),
}


def _load_spectrum(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=np.float64) for key in data.files}


def _load_structures(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=np.float64) for key in data.files}


def _percentile_max(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return {
        "median": float(np.median(absolute)),
        "p95": float(np.percentile(absolute, 95.0)),
        "max": float(np.max(absolute)),
    }


def _structure_metrics(structures: dict[str, np.ndarray]) -> dict[str, dict[str, dict[str, float]]]:
    reference = "six_field_continuation"
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for arm in PRODUCT_ARMS:
        if arm == reference:
            continue
        metrics[arm] = {}
        for field in ("column_mass", *STRUCTURE_FIELDS):
            candidate = structures[f"{arm}__{field}"]
            base = structures[f"{reference}__{field}"]
            if field == "column_mass":
                values = np.log10(candidate) - np.log10(base)
            else:
                values = candidate / np.maximum(np.abs(base), 1.0e-300) - 1.0
            metrics[arm][field] = _percentile_max(values)

    # Raw MARCS is on its native 56-layer grid. Compare only over the overlap
    # in log m, interpolating the six-field reference in that coordinate.
    raw_m = structures["marcs_raw__column_mass"]
    raw_log_m = np.log10(raw_m)
    base_m = structures["six_field_continuation__column_mass"]
    base_log_m = np.log10(base_m)
    overlap = (raw_log_m >= base_log_m[0]) & (raw_log_m <= base_log_m[-1])
    raw_metrics: dict[str, dict[str, float]] = {}
    for field in ("temperature", "gas_pressure", "electron_density"):
        raw = structures[f"marcs_raw__{field}"][overlap]
        base = np.interp(
            raw_log_m[overlap],
            base_log_m,
            structures[f"six_field_continuation__{field}"],
        )
        raw_metrics[field] = _percentile_max(raw / np.maximum(np.abs(base), 1.0e-300) - 1.0)
    raw_metrics["overlap_log_m"] = {
        "min": float(max(raw_log_m[0], base_log_m[0])),
        "max": float(min(raw_log_m[-1], base_log_m[-1])),
        "layers": float(np.count_nonzero(overlap)),
    }
    metrics["marcs_raw_vs_six_field_overlap"] = raw_metrics
    return metrics


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def _save(fig: plt.Figure, root: Path, stem: str) -> None:
    fig.savefig(root / f"{stem}.pdf")
    fig.savefig(root / f"{stem}.png", dpi=300)
    plt.close(fig)


def _plot_spectra(root: Path, out_root: Path) -> None:
    spectra = {arm: _load_spectrum(root / "spectra" / f"{arm}.npz") for arm in PRODUCT_ARMS}
    wavelength = spectra["six_field_continuation"]["wavelength_nm"]
    reference = spectra["six_field_continuation"]["normalized_flux"]
    differences = {
        arm: spectra[arm]["normalized_flux"] - reference
        for arm in PRODUCT_ARMS
        if arm != "six_field_continuation"
    }
    peak_arm = "two_field_continuation"
    peak_index = int(np.argmax(np.abs(differences[peak_arm])))
    centre = float(wavelength[peak_index])
    zoom = (wavelength >= centre - 2.0) & (wavelength <= centre + 2.0)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    ax = axes[0, 0]
    for arm in PRODUCT_ARMS:
        ax.plot(wavelength, spectra[arm]["normalized_flux"], lw=0.8 if arm != "six_field_continuation" else 1.0, color=COLORS[arm], label=LABELS[arm])
    ax.set(xlabel="Wavelength (nm)", ylabel="Normalized flux", title="3500 K spectrum")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    for arm, values in differences.items():
        ax.plot(wavelength, values, lw=0.8, color=COLORS[arm], label=f"{LABELS[arm]} − six-field")
    ax.axhline(0.0, color="#555555", lw=0.5)
    ax.axhline(5.0e-3, color="#CC3311", lw=0.7, ls=":")
    ax.axhline(-5.0e-3, color="#CC3311", lw=0.7, ls=":")
    ax.set(xlabel="Wavelength (nm)", ylabel="$\\Delta$ normalized flux", title="Difference from six-field continuation")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1, 0]
    for arm in PRODUCT_ARMS:
        ax.plot(wavelength[zoom], spectra[arm]["normalized_flux"][zoom], lw=1.0 if arm == "six_field_continuation" else 0.8, color=COLORS[arm], label=LABELS[arm])
    ax.set(xlabel="Wavelength (nm)", ylabel="Normalized flux", title=f"Zoom near largest two-field residual ({centre:.2f} nm)")

    ax = axes[1, 1]
    metric_names = ("normalized_flux", "flux_total_continuum_scaled", "flux_continuum_relative")
    metric_labels = ("Normalized", "Total / continuum", "Continuum")
    arms = ("two_field_continuation", "marcs_started_reduced", "marcs_started_full")
    x = np.arange(len(arms))
    width = 0.24
    summary = json.loads((root / "comparison_summary.json").read_text(encoding="utf-8"))
    for index, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        values = [summary["comparisons"][f"{arm}_vs_six_field_continuation"][metric]["max"] for arm in arms]
        ax.bar(x + (index - 1) * width, values, width=width, label=label, color=("#0072B2", "#D55E00", "#009E73")[index])
    ax.axhline(5.0e-3, color="#CC3311", ls=":", lw=0.8, label="5×10$^{-3}$ bar")
    ax.set_xticks(x, ("Two-field", "MARCS red.", "MARCS full"))
    ax.set_yscale("log")
    ax.set_ylabel("Maximum difference")
    ax.set_title("Spectral comparison metrics")
    ax.legend(frameon=False, fontsize=6)

    for index, axis in enumerate(axes.flat):
        axis.text(-0.12, 1.05, "ABCD"[index], transform=axis.transAxes, fontweight="bold", va="top")
    _save(fig, out_root, "cool_star_3500_spectrum_comparison")


def _plot_structure(root: Path, out_root: Path) -> dict:
    structures = _load_structures(root / "structure_comparison.npz")
    metrics = _structure_metrics(structures)
    base_m = structures["six_field_continuation__column_mass"]
    raw_m = structures["marcs_raw__column_mass"]
    base_x = np.log10(base_m)
    raw_x = np.log10(raw_m)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2), constrained_layout=True)
    for axis, field in zip(axes.ravel()[:4], STRUCTURE_FIELDS):
        ylabel, log_y = FIELD_LABELS[field]
        for arm in PRODUCT_ARMS:
            values = structures[f"{arm}__{field}"]
            axis.plot(base_x if values.size == 80 else raw_x, values, color=COLORS[arm], lw=0.9, label=LABELS[arm])
        if field in ("temperature", "gas_pressure", "electron_density"):
            raw_values = structures[f"marcs_raw__{field}"]
            axis.plot(raw_x, raw_values, color=COLORS["marcs_raw"], lw=0.8, ls="--", marker=".", ms=2.0, label="Raw MARCS (56 layers)")
        if log_y:
            axis.set_yscale("log")
        axis.set(xlabel="$\\log_{10} m$ (g cm$^{-2}$)", ylabel=ylabel, title=field.replace("_", " ").title())

    axis = axes[1, 1]
    reference = structures["six_field_continuation__temperature"]
    for arm in ("two_field_continuation", "marcs_started_reduced", "marcs_started_full"):
        delta = structures[f"{arm}__temperature"] / reference - 1.0
        axis.plot(base_x, delta, color=COLORS[arm], lw=0.9, label=LABELS[arm])
    axis.axhline(0.0, color="#555555", lw=0.5)
    axis.set(xlabel="$\\log_{10} m$ (g cm$^{-2}$)", ylabel="$\\Delta T/T$", title="Temperature difference from six-field")
    axis.legend(frameon=False, fontsize=6)

    axis = axes[1, 2]
    for arm in PRODUCT_ARMS:
        axis.plot(np.arange(80), np.log10(structures[f"{arm}__column_mass"]), color=COLORS[arm], lw=0.9, label=LABELS[arm])
    axis.plot(np.linspace(0, 79, raw_m.size), raw_x, color=COLORS["marcs_raw"], ls="--", marker=".", ms=2.0, lw=0.8, label="Raw MARCS")
    axis.set(xlabel="Layer index", ylabel="$\\log_{10} m$ (g cm$^{-2}$)", title="Column-mass coordinate")

    for index, axis in enumerate(axes.ravel()):
        axis.text(-0.12, 1.05, "ABCDEF"[index], transform=axis.transAxes, fontweight="bold", va="top")
    _save(fig, out_root, "cool_star_3500_structure_comparison")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    _style()
    metrics = _plot_structure(args.root, args.root)
    _plot_spectra(args.root, args.root)
    (args.root / "structure_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote figures and {args.root / 'structure_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
