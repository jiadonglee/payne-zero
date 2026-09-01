#!/usr/bin/env python3
"""Build a PDF comparing converged two-field and six-field products.

The report uses the frozen 2026-08-11 blind run.  It compares the 189 stars
for which both initializers produced a usable converged atmosphere and a
synthetic spectrum.  The six-field arm is the shipped production product;
the two-field arm predicts (m, T), reconstructs the other fields, and then
runs the same reference solver and synthesis code.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
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
        **{"axes.titlesize": 10},
    )

from matplotlib.colors import LogNorm


RUN_ROOT = Path(
    "runs/reduced_state_emulator/"
    "solver_in_loop_k1_qualified_tail3_profile_rescue_v4/blind200"
)
RESULT_ROOT = Path(
    "results/solver_in_loop_k1_qualified_tail3_profile_rescue_v4/blind200"
)
DEFAULT_OUT = Path("results/two_vs_six_field_comparison_20260811_en.pdf")

BASELINE = "production_six_field"
CANDIDATE = "learned_reduced_state"
BAR = 5.0e-3
SPECTRAL_WINDOW_WIDTH_NM = 5.0

BLACK = "#111111"  # deliberately not style.OkabeIto.BLACK

SLUG_RE = re.compile(
    r"^t(?P<teff>\d+(?:\.\d+)?)_g(?P<logg>[+-]\d+(?:\.\d+)?)_"
    r"m(?P<metal>[+-]\d+(?:\.\d+)?)_a(?P<alpha>[+-]\d+(?:\.\d+)?)_"
    r"x(?P<vmic>\d+(?:\.\d+)?)$"
)


@dataclass(frozen=True)
class StarMetrics:
    slug: str
    teff: float
    logg: float
    metallicity: float
    alpha: float
    vmic: float
    temperature_max_relative: float
    mass_max_dex: float
    pressure_max_dex: float
    electron_max_dex: float
    density_max_dex: float
    normalized_flux_max: float
    total_flux_max: float
    continuum_flux_max: float


def _labels_from_slug(slug: str) -> tuple[float, float, float, float, float]:
    match = SLUG_RE.match(slug)
    if match is None:
        raise ValueError(f"cannot parse stellar labels from {slug!r}")
    return tuple(float(match.group(key)) for key in ("teff", "logg", "metal", "alpha", "vmic"))


def _signed_relative(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    return (candidate - baseline) / np.maximum(np.abs(baseline), np.finfo(float).tiny)


def _signed_dex(candidate: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    if np.any(candidate <= 0.0) or np.any(baseline <= 0.0):
        raise ValueError("dex comparison requires positive values")
    return np.log10(candidate) - np.log10(baseline)


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.06,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=11,
        va="top",
    )


def _footer(fig: mpl.figure.Figure, page: int) -> None:
    fig.text(
        0.99,
        0.012,
        f"Two-field vs six-field comparison | 2026-08-11 blind test | Page {page}",
        ha="right",
        va="bottom",
        fontsize=7,
        color=GREY,
    )


def _collect_metrics(
    atmosphere_root: Path,
    spectral_gate: dict,
) -> tuple[list[StarMetrics], dict[str, np.ndarray]]:
    gate_rows = {row["slug"]: row for row in spectral_gate["per_star"]}
    candidate_root = atmosphere_root / "products" / CANDIDATE
    baseline_root = atmosphere_root / "products" / BASELINE
    common = sorted(
        set(path.stem for path in candidate_root.glob("*.npz"))
        & set(path.stem for path in baseline_root.glob("*.npz"))
        & set(gate_rows)
    )
    if len(common) != int(spectral_gate["gated_star_count"]):
        raise ValueError(
            f"paired product count {len(common)} does not match spectral gate "
            f"{spectral_gate['gated_star_count']}"
        )

    pointwise: dict[str, list[np.ndarray]] = {
        "temperature_relative": [],
        "mass_dex": [],
        "pressure_dex": [],
        "electron_dex": [],
        "density_dex": [],
    }
    stars: list[StarMetrics] = []
    for slug in common:
        candidate = _load_npz(candidate_root / f"{slug}.npz")
        baseline = _load_npz(baseline_root / f"{slug}.npz")
        errors = {
            "temperature_relative": np.abs(
                _signed_relative(candidate["temperature"], baseline["temperature"])
            ),
            "mass_dex": np.abs(
                _signed_dex(candidate["column_mass"], baseline["column_mass"])
            ),
            "pressure_dex": np.abs(
                _signed_dex(candidate["gas_pressure"], baseline["gas_pressure"])
            ),
            "electron_dex": np.abs(
                _signed_dex(candidate["electron_density"], baseline["electron_density"])
            ),
            "density_dex": np.abs(
                _signed_dex(candidate["mass_density"], baseline["mass_density"])
            ),
        }
        for key, values in errors.items():
            if not np.all(np.isfinite(values)):
                raise ValueError(f"non-finite {key} difference for {slug}")
            pointwise[key].append(values)
        labels = _labels_from_slug(slug)
        gate = gate_rows[slug]
        stars.append(
            StarMetrics(
                slug=slug,
                teff=labels[0],
                logg=labels[1],
                metallicity=labels[2],
                alpha=labels[3],
                vmic=labels[4],
                temperature_max_relative=float(np.max(errors["temperature_relative"])),
                mass_max_dex=float(np.max(errors["mass_dex"])),
                pressure_max_dex=float(np.max(errors["pressure_dex"])),
                electron_max_dex=float(np.max(errors["electron_dex"])),
                density_max_dex=float(np.max(errors["density_dex"])),
                normalized_flux_max=float(gate["normalized_flux"]["max"]),
                total_flux_max=float(gate["flux_total"]["max"]),
                continuum_flux_max=float(gate["flux_continuum"]["max"]),
            )
        )
    return stars, {key: np.concatenate(values) for key, values in pointwise.items()}


def _rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def _pick_typical(
    stars: list[StarMetrics],
    predicate,
    used: set[str],
) -> StarMetrics:
    subset = [star for star in stars if predicate(star) and star.slug not in used]
    if not subset:
        subset = [star for star in stars if star.slug not in used]
    columns = np.array(
        [
            [
                star.temperature_max_relative,
                star.mass_max_dex,
                star.pressure_max_dex,
                star.electron_max_dex,
                star.density_max_dex,
                star.normalized_flux_max,
            ]
            for star in subset
        ]
    )
    ranks = np.column_stack([_rank_fraction(columns[:, index]) for index in range(columns.shape[1])])
    score = np.mean(np.abs(ranks - 0.5), axis=1)
    chosen = subset[int(np.argmin(score))]
    used.add(chosen.slug)
    return chosen


def _select_examples(stars: list[StarMetrics]) -> list[tuple[str, StarMetrics]]:
    used: set[str] = set()
    selected: list[tuple[str, StarMetrics]] = []
    categories = [
        ("Typical cool giant", lambda s: s.teff < 5500.0 and s.logg < 2.5),
        ("Typical cool dwarf", lambda s: s.teff < 5500.0 and s.logg >= 4.0),
        ("Typical hot giant", lambda s: s.teff >= 7500.0 and s.logg < 3.0),
        ("Typical hot dwarf", lambda s: s.teff >= 7500.0 and s.logg >= 4.0),
    ]
    for label, predicate in categories:
        selected.append((label, _pick_typical(stars, predicate, used)))

    scales = {
        "temperature_max_relative": np.quantile(
            [s.temperature_max_relative for s in stars], 0.95
        ),
        "mass_max_dex": np.quantile([s.mass_max_dex for s in stars], 0.95),
        "pressure_max_dex": np.quantile([s.pressure_max_dex for s in stars], 0.95),
        "electron_max_dex": np.quantile([s.electron_max_dex for s in stars], 0.95),
        "density_max_dex": np.quantile([s.density_max_dex for s in stars], 0.95),
    }

    def structure_score(star: StarMetrics) -> float:
        return max(
            getattr(star, key) / max(scale, np.finfo(float).tiny)
            for key, scale in scales.items()
        )

    worst_structure = max(
        (star for star in stars if star.slug not in used), key=structure_score
    )
    used.add(worst_structure.slug)
    selected.append(("Largest final-atmosphere difference", worst_structure))
    worst_spectrum = max(
        (star for star in stars if star.slug not in used),
        key=lambda star: star.normalized_flux_max,
    )
    selected.append(("Largest normalized-spectrum difference", worst_spectrum))
    return selected


def _structure_summary(pointwise: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values)),
        }
        for key, values in pointwise.items()
    }


def _make_title_page(
    pdf: PdfPages,
    summary: dict,
    spectral_gate: dict,
    structure: dict[str, dict[str, float]],
    selected: list[tuple[str, StarMetrics]],
    page: int,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.91, "Two-field vs six-field models: final atmospheres and spectra", fontsize=21, weight="bold")
    fig.text(
        0.06,
        0.865,
        "Frozen 2026-08-11 blind test; only stars with usable converged products from both models",
        fontsize=11,
        color=GREY,
    )

    solver = summary["solver"]
    spectra = spectral_gate
    p95_rows = [
        ("Temperature", f"{100.0 * structure['temperature_relative']['p95']:.3f}%"),
        ("Column mass", f"{structure['mass_dex']['p95']:.4f} dex"),
        ("Gas pressure", f"{structure['pressure_dex']['p95']:.4f} dex"),
        ("Electron density", f"{structure['electron_dex']['p95']:.4f} dex"),
        ("Mass density", f"{structure['density_dex']['p95']:.4f} dex"),
    ]
    ax_table = fig.add_axes([0.06, 0.47, 0.38, 0.32])
    ax_table.axis("off")
    table = ax_table.table(
        cellText=p95_rows,
        colLabels=["Final atmosphere quantity", "p95 difference (all points)"],
        loc="center",
        cellLoc="left",
        colWidths=[0.42, 0.58],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.4)
    table.scale(1.0, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#EAF2F8")
            cell.set_text_props(weight="bold")

    ax_text = fig.add_axes([0.50, 0.46, 0.44, 0.34])
    ax_text.axis("off")
    candidate = solver["candidate"]
    production = solver["production"]
    text = (
        "Main result\n\n"
        f"• Usable converged atmospheres: two-field {candidate['usable_products']}/200; "
        f"six-field {production['usable_products']}/200.\n"
        f"• For 189 paired usable stars, mean iterations: two-field "
        f"{solver['paired_usable_products']['candidate_mean_iterations']:.2f}; "
        f"six-field {solver['paired_usable_products']['production_mean_iterations']:.2f}.\n"
        f"• Normalized spectra: median stellar maximum difference "
        f"{100.0 * spectra['normalized_flux']['median_over_stars']:.2f}%;\n"
        f"  {spectra['normalized_flux']['stars_over_bar']}/189 exceed 0.5%.\n"
        f"• Total flux: median stellar maximum difference "
        f"{100.0 * spectra['flux_total']['median_over_stars']:.2f}%;\n"
        f"  {spectra['flux_total']['stars_over_bar']}/189 exceed 0.5%.\n\n"
        "Interpretation: final atmospheres and spectra are very close for most stars, and the\n"
        "two-field model usually converges faster. Edge cases retain a substantial long tail."
    )
    ax_text.text(0.0, 1.0, text, va="top", linespacing=1.45, fontsize=10.2)

    fig.text(0.06, 0.39, "Stellar examples in this report", fontsize=12, weight="bold")
    y = 0.35
    for index, (label, star) in enumerate(selected, start=1):
        short_label = {
            "Largest final-atmosphere difference": "Final-atmosphere outlier",
            "Largest normalized-spectrum difference": "Normalized-spectrum outlier",
        }.get(label, label)
        fig.text(
            0.07 + 0.46 * ((index - 1) // 3),
            y - 0.073 * ((index - 1) % 3),
            f"{index}. {short_label}: T={star.teff:.0f} K, log g={star.logg:.2f}, "
            f"[M/H]={star.metallicity:+.2f}\n"
            f"   [alpha/M]={star.alpha:+.2f}, xi={star.vmic:.2f} km/s",
            fontsize=8.8,
            linespacing=1.25,
        )
    fig.text(
        0.06,
        0.12,
        "Scope and limitations",
        fontsize=11,
        weight="bold",
    )
    fig.text(
        0.06,
        0.065,
        "Final products store T, m, P, n_e, and mass density; final kappa_R and g_rad were not saved and are therefore omitted.\n"
        "The six-field network overlaps the corpus and is a product baseline, not an independent generalization control.\n"
        "Full-range metrics use 400–900 nm at R=20,000; each example plots a 5 nm window centered on its largest normalized-flux difference.",
        fontsize=8.3,
        color=GREY,
        linespacing=1.4,
    )
    _footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    return x, np.arange(1, len(x) + 1, dtype=np.float64) / len(x)


def _make_population_page(
    pdf: PdfPages,
    stars: list[StarMetrics],
    page: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
    fig.suptitle("Population comparison: 189 jointly converged stars", fontsize=16, weight="bold", y=0.97)
    teff = np.array([s.teff for s in stars])
    logg = np.array([s.logg for s in stars])
    norm = np.array([s.normalized_flux_max for s in stars])

    ax = axes[0, 0]
    scatter = ax.scatter(
        teff,
        logg,
        c=norm,
        s=29,
        cmap="cividis",
        norm=LogNorm(vmin=max(float(np.min(norm)), 1.0e-4), vmax=float(np.max(norm))),
        edgecolor="none",
    )
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Effective temperature (K)")
    ax.set_ylabel("log g")
    ax.set_title("Color: maximum normalized-spectrum difference")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Maximum |delta normalized flux|")
    _panel_label(ax, "A")

    ax = axes[0, 1]
    temp = 100.0 * np.array([s.temperature_max_relative for s in stars])
    x, y = _ecdf(temp)
    ax.plot(x, y, color=ORANGE, lw=2.0, label="Maximum temperature difference per star")
    ax.axvline(np.median(temp), color=ORANGE, ls="--", lw=1.2, label=f"Median {np.median(temp):.2f}%")
    ax.set_xscale("log")
    ax.set_xlabel("Maximum final-temperature difference (%)")
    ax.set_ylabel("Cumulative fraction of stars")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#E5E5E5", lw=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Long tail in temperature differences")
    _panel_label(ax, "B")

    ax = axes[1, 0]
    structure_series = [
        ("Column mass", [s.mass_max_dex for s in stars], BLUE, "-"),
        ("Gas pressure", [s.pressure_max_dex for s in stars], ORANGE, "--"),
        ("Electron density", [s.electron_max_dex for s in stars], GREEN, "-."),
        ("Mass density", [s.density_max_dex for s in stars], PURPLE, ":"),
    ]
    for label, values, color, style in structure_series:
        x, y = _ecdf(np.asarray(values))
        ax.plot(x, y, label=label, color=color, ls=style, lw=1.9)
    ax.set_xscale("log")
    ax.set_xlabel("Maximum layer-by-layer difference per star (dex)")
    ax.set_ylabel("Cumulative fraction of stars")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#E5E5E5", lw=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Final-atmosphere structure differences")
    _panel_label(ax, "C")

    ax = axes[1, 1]
    spectral_series = [
        ("Normalized spectrum", [s.normalized_flux_max for s in stars], BLUE, "-"),
        ("Total flux", [s.total_flux_max for s in stars], ORANGE, "--"),
        ("Continuum", [s.continuum_flux_max for s in stars], GREEN, "-."),
    ]
    for label, values, color, style in spectral_series:
        x, y = _ecdf(np.asarray(values))
        ax.plot(x, y, label=label, color=color, ls=style, lw=1.9)
    ax.axvline(BAR, color=BLACK, ls=":", lw=1.5, label="0.5% threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Maximum spectral difference per star")
    ax.set_ylabel("Cumulative fraction of stars")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", color="#E5E5E5", lw=0.7)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Synthetic-spectrum differences")
    _panel_label(ax, "D")

    fig.subplots_adjust(left=0.075, right=0.96, bottom=0.08, top=0.90, hspace=0.34, wspace=0.28)
    _footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def _make_landscape_page(
    pdf: PdfPages,
    stars: list[StarMetrics],
    selected: list[tuple[str, StarMetrics]],
    page: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    fig.suptitle("Where the differences lie in stellar-parameter space", fontsize=16, weight="bold", y=0.97)
    teff = np.array([s.teff for s in stars])
    logg = np.array([s.logg for s in stars])
    panels = [
        ("Maximum temperature difference", 100.0 * np.array([s.temperature_max_relative for s in stars]), "%"),
        ("Maximum column-mass difference", np.array([s.mass_max_dex for s in stars]), "dex"),
        ("Maximum gas-pressure difference", np.array([s.pressure_max_dex for s in stars]), "dex"),
        ("Maximum electron-density difference", np.array([s.electron_max_dex for s in stars]), "dex"),
        ("Maximum mass-density difference", np.array([s.density_max_dex for s in stars]), "dex"),
        ("Maximum normalized-spectrum difference", np.array([s.normalized_flux_max for s in stars]), "absolute difference"),
    ]
    for index, (ax, (title, values, unit)) in enumerate(zip(axes.flat, panels)):
        positive = values[values > 0.0]
        vmin = max(float(np.quantile(positive, 0.05)), np.finfo(float).tiny)
        vmax = float(np.max(positive))
        scatter = ax.scatter(
            teff,
            logg,
            c=values,
            s=23,
            cmap="cividis",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            edgecolor="none",
        )
        for label, star in selected:
            ax.scatter(
                [star.teff],
                [star.logg],
                marker="o",
                s=55,
                facecolors="none",
                edgecolors="white",
                linewidths=1.1,
            )
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_xlabel("T_eff (K)")
        ax.set_ylabel("log g")
        ax.set_title(title)
        cbar = fig.colorbar(scatter, ax=ax, pad=0.015)
        cbar.set_label(unit)
        _panel_label(ax, chr(ord("A") + index))
    fig.text(
        0.075,
        0.025,
        "White open circles mark the typical and outlier stars shown on later pages; colors use logarithmic scaling.",
        fontsize=8,
        color=GREY,
    )
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.90, hspace=0.34, wspace=0.30)
    _footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def _star_title(label: str, star: StarMetrics) -> str:
    return (
        f"{label}: T_eff={star.teff:.1f} K, log g={star.logg:.2f}, "
        f"[M/H]={star.metallicity:+.2f}, [alpha/M]={star.alpha:+.2f}, "
        f"xi={star.vmic:.2f} km/s"
    )


def _make_star_page(
    pdf: PdfPages,
    run_root: Path,
    label: str,
    star: StarMetrics,
    page: int,
) -> None:
    candidate = _load_npz(run_root / "products" / CANDIDATE / f"{star.slug}.npz")
    baseline = _load_npz(run_root / "products" / BASELINE / f"{star.slug}.npz")
    candidate_spectrum = _load_npz(run_root / "spectra" / CANDIDATE / f"{star.slug}.npz")
    baseline_spectrum = _load_npz(run_root / "spectra" / BASELINE / f"{star.slug}.npz")
    layer = np.arange(candidate["temperature"].size)
    wavelength = baseline_spectrum["wavelength_nm"]
    norm_delta = np.abs(candidate_spectrum["normalized_flux"] - baseline_spectrum["normalized_flux"])
    total_delta = np.abs(candidate_spectrum["flux_total"] - baseline_spectrum["flux_total"]) / np.maximum(
        np.abs(baseline_spectrum["flux_continuum"]), np.finfo(float).tiny
    )
    continuum_delta = np.abs(
        candidate_spectrum["flux_continuum"] - baseline_spectrum["flux_continuum"]
    ) / np.maximum(np.abs(baseline_spectrum["flux_continuum"]), np.finfo(float).tiny)
    peak_wavelength = float(wavelength[int(np.argmax(norm_delta))])
    window_min = float(
        np.clip(
            peak_wavelength - 0.5 * SPECTRAL_WINDOW_WIDTH_NM,
            float(wavelength[0]),
            float(wavelength[-1]) - SPECTRAL_WINDOW_WIDTH_NM,
        )
    )
    window_max = window_min + SPECTRAL_WINDOW_WIDTH_NM
    window = (wavelength >= window_min) & (wavelength <= window_max)
    if int(np.sum(window)) < 2:
        raise ValueError(f"narrow spectral window has fewer than two samples for {star.slug}")

    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    fig.suptitle(_star_title(label, star), fontsize=13.5, weight="bold", y=0.97)
    fig.text(
        0.5,
        0.925,
        f"Max temperature difference {100.0 * star.temperature_max_relative:.2f}% | "
        f"column mass {star.mass_max_dex:.3f} dex | pressure {star.pressure_max_dex:.3f} dex | "
        f"normalized spectrum {100.0 * star.normalized_flux_max:.2f}%",
        ha="center",
        fontsize=9.5,
        color=GREY,
    )

    ax = axes[0, 0]
    ax.plot(layer, baseline["temperature"], color=BLUE, lw=2.0, label="Six-field default")
    ax.plot(layer, candidate["temperature"], color=ORANGE, lw=1.7, ls="--", label="Two-field + physics")
    ax.set_xlabel("Atmosphere layer")
    ax.set_ylabel("Temperature (K)")
    ax.set_title("Final temperature profile")
    ax.legend(frameon=False)
    _panel_label(ax, "A")

    ax = axes[0, 1]
    ax.plot(layer, np.log10(baseline["column_mass"]), color=BLUE, lw=2.0, label="Six-field default")
    ax.plot(layer, np.log10(candidate["column_mass"]), color=ORANGE, lw=1.7, ls="--", label="Two-field + physics")
    ax.set_xlabel("Atmosphere layer")
    ax.set_ylabel("log10 m (g cm$^{-2}$)")
    ax.set_title("Final column-mass profile")
    ax.legend(frameon=False)
    _panel_label(ax, "B")

    ax = axes[0, 2]
    temperature_delta = 100.0 * _signed_relative(candidate["temperature"], baseline["temperature"])
    ax.axhline(0.0, color="#BBBBBB", lw=0.8)
    ax.plot(layer, temperature_delta, color=ORANGE, lw=1.8)
    ax.set_xlabel("Atmosphere layer")
    ax.set_ylabel("(two-field - six-field) / six-field (%)")
    ax.set_title("Layer-by-layer temperature difference")
    ax.set_yscale("symlog", linthresh=0.05)
    _panel_label(ax, "C")

    ax = axes[1, 0]
    dex_series = [
        ("Column mass", "column_mass", BLUE, "-"),
        ("Gas pressure", "gas_pressure", ORANGE, "--"),
        ("Electron density", "electron_density", GREEN, "-."),
        ("Mass density", "mass_density", PURPLE, ":"),
    ]
    ax.axhline(0.0, color="#BBBBBB", lw=0.8)
    for field_label, field, color, style in dex_series:
        ax.plot(
            layer,
            _signed_dex(candidate[field], baseline[field]),
            label=field_label,
            color=color,
            ls=style,
            lw=1.7,
        )
    ax.set_xlabel("Atmosphere layer")
    ax.set_ylabel("log10(two-field / six-field) (dex)")
    ax.set_title("Other final-structure differences")
    ax.set_yscale("symlog", linthresh=1.0e-3)
    ax.legend(frameon=False, ncol=2)
    _panel_label(ax, "D")

    ax = axes[1, 1]
    ax.plot(wavelength[window], baseline_spectrum["normalized_flux"][window], color=BLUE, lw=1.1, label="Six-field default")
    ax.plot(wavelength[window], candidate_spectrum["normalized_flux"][window], color=ORANGE, lw=1.0, ls="--", label="Two-field + physics")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized flux")
    ax.set_title(f"Narrow spectral window: {window_min:.1f}-{window_max:.1f} nm")
    ax.legend(frameon=False)
    _panel_label(ax, "E")

    ax = axes[1, 2]
    ax.plot(wavelength[window], norm_delta[window], color=BLUE, lw=1.0, label="Normalized spectrum")
    ax.plot(wavelength[window], total_delta[window], color=ORANGE, lw=0.9, ls="--", label="Total flux / continuum")
    ax.plot(wavelength[window], continuum_delta[window], color=GREEN, lw=0.9, ls="-.", label="Relative continuum")
    ax.axhline(BAR, color=BLACK, lw=1.1, ls=":", label="0.5% threshold")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absolute or scaled difference")
    ax.set_title("Spectral differences in the same 5 nm window")
    ax.set_yscale("log")
    positive = np.concatenate(
        [
            norm_delta[window][norm_delta[window] > 0],
            total_delta[window][total_delta[window] > 0],
            continuum_delta[window][continuum_delta[window] > 0],
        ]
    )
    ax.set_ylim(max(float(np.quantile(positive, 0.01)) * 0.5, 1.0e-8), max(float(np.max(positive)) * 1.4, BAR * 1.4))
    ax.legend(frameon=True, framealpha=0.85, ncol=2, loc="best", fontsize=7.2)
    _panel_label(ax, "F")

    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.08, top=0.88, hspace=0.34, wspace=0.30)
    _footer(fig, page)
    pdf.savefig(fig)
    plt.close(fig)


def build_report(run_root: Path, result_root: Path, out: Path) -> Path:
    summary_path = result_root / "summary.json"
    gate_path = result_root / "spectral_gate.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    spectral_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    stars, pointwise = _collect_metrics(run_root, spectral_gate)
    structure = _structure_summary(pointwise)
    selected = _select_examples(stars)
    out.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "Title": "Two-field versus six-field atmosphere and spectrum comparison",
        "Author": "Payne-zero reduced-state validation",
        "Subject": "Frozen 2026-08-11 blind comparison",
        "Keywords": "stellar atmosphere, spectrum, reduced state, Payne-zero",
    }
    with PdfPages(out, metadata=metadata) as pdf:
        page = 1
        _make_title_page(pdf, summary, spectral_gate, structure, selected, page)
        page += 1
        _make_population_page(pdf, stars, page)
        page += 1
        _make_landscape_page(pdf, stars, selected, page)
        for label, star in selected:
            page += 1
            _make_star_page(pdf, run_root, label, star, page)

    provenance = {
        "report": str(out),
        "report_sha256": _sha256(out),
        "run_root": str(run_root),
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "spectral_gate": str(gate_path),
        "spectral_gate_sha256": _sha256(gate_path),
        "paired_stars": len(stars),
        "structure_pointwise": structure,
        "spectral_plot_window": {
            "width_nm": SPECTRAL_WINDOW_WIDTH_NM,
            "selection": "centered on each star's largest full-range normalized-flux difference",
            "full_range_metrics_nm": [400.0, 900.0],
        },
        "selected_examples": [
            {"category": label, **asdict(star)} for label, star in selected
        ],
        "limitations": [
            "Final kappa_R and g_rad were not saved in the structured products.",
            "The six-field checkpoint is a product baseline, not an independent generalization control.",
        ],
    }
    provenance_path = out.with_suffix(".json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    _configure_style()
    report = build_report(args.run_root, args.result_root, args.out)
    print(f"wrote {report}")
    print(f"wrote {report.with_suffix('.json')}")


if __name__ == "__main__":
    main()
