#!/usr/bin/env python3
"""Paper-style cool-grid comparison: native PZ, Korg native, Korg+Kurucz."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-mstar-grid-v6")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import h5py
import matplotlib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from payne_zero_figures import style
from payne_zero_synthesis.api import _surface_flux_per_wavelength_nm
from payne_zero_synthesis.pipeline import window_invariants_for
from payne_zero_synthesis.synthesis import synthesize_structured_atmosphere

OUT = ROOT / "results/m_star_cool_grid_v6_three_spectra"
KORG_ROOT = Path("/Users/jdli/Project/jorg/Korg.jl-1.0.1")
KORG_HELPER = Path(__file__).with_name("korg_three_linelist_v6.jl")
KURUCZ_LINES = ROOT / "results/m_star_eso_highres_comparison_v5_same_linelist/inputs/pz_atomic_molecular_transitions_tio.tsv"
R_SYNTH = 600_000.0
R_FINAL = 80_000.0
AIR_WINDOW = (6650.0, 6670.0)
PAD_A = 3.0

NODES = [
    ("poor giant", 3200, 2.5, -1.0, 2.0),
    ("poor giant", 3500, 2.5, -1.0, 2.0),
    ("poor giant", 4000, 2.5, -1.0, 2.0),
    ("rich giant", 3200, 1.5, +0.5, 2.0),
    ("rich giant", 3500, 1.5, +0.5, 2.0),
    ("rich giant", 4000, 1.5, +0.5, 2.0),
    ("poor dwarf", 3200, 4.5, -1.0, 1.0),
    ("poor dwarf", 3500, 4.5, -1.0, 1.0),
    ("poor dwarf", 4000, 4.5, -1.0, 1.0),
    ("rich dwarf", 3200, 5.0, +0.5, 1.0),
    ("rich dwarf", 3500, 5.0, +0.5, 1.0),
    ("rich dwarf", 4000, 5.0, +0.5, 1.0),
]


def air_to_vacuum_A(wave_air_A):
    s2 = (1e4 / np.asarray(wave_air_A, dtype=float)) ** 2
    n = 1 + 8.34254e-5 + 2.406147e-2 / (130 - s2) + 1.5998e-4 / (38.9 - s2)
    return np.asarray(wave_air_A) * n


def vacuum_to_air_A(wave_vac_A):
    wave_air = np.asarray(wave_vac_A, dtype=float).copy()
    for _ in range(4):
        wave_air *= np.asarray(wave_vac_A) / air_to_vacuum_A(wave_air)
    return wave_air


def atmosphere_path(kind, teff, logg, mh, vmic):
    cls = kind.split()[1]
    track = f"g{logg:+.2f}_m{mh:+.2f}_a+0.00_c+0.00_x{vmic:.2f}"
    directory = ROOT / f"results/m_star_emulator_v1r2_marcs100/cases/{cls}/{track}/t{teff}/products/primary"
    paths = sorted(directory.glob("*.npz"))
    return paths[0] if paths else None


def export_atmosphere(path, stem):
    atm_path = OUT / "inputs" / f"{stem}_atmosphere.tsv"
    abund_path = OUT / "inputs" / f"{stem}_abundances.tsv"
    atm_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(path, allow_pickle=False) as data:
        matrix = np.column_stack([data[k] for k in ("temperature", "electron_density", "gas_pressure", "mass_density", "column_mass")])
        abund = np.asarray(data["elemental_abundances"][:92], dtype=float)
    np.savetxt(atm_path, matrix, delimiter="\t", header="temperature_K\telectron_density_cm-3\tgas_pressure_dyn_cm-2\tmass_density_g_cm-3\tcolumn_mass_g_cm-2", comments="")
    np.savetxt(abund_path, abund[None, :], delimiter="\t")
    return atm_path, abund_path


def common_spectrum(wave_A, total, continuum, common_vac_A):
    total_i = np.interp(common_vac_A, wave_A, total)
    cont_i = np.interp(common_vac_A, wave_A, continuum)
    step = float(np.median(np.diff(common_vac_A)))
    sigma_pix = float(np.mean(common_vac_A) / R_FINAL / 2.354820045 / step)
    return gaussian_filter1d(total_i, sigma_pix, mode="nearest") / gaussian_filter1d(cont_i, sigma_pix, mode="nearest")


def run_korg(mode, atm_tsv, abund_tsv, output, start_A, end_A, vmic):
    subprocess.run([
        "julia", "--project=" + str(KORG_ROOT), str(KORG_HELPER), mode,
        str(KURUCZ_LINES), str(atm_tsv), str(abund_tsv), str(output),
        f"{start_A:.8f}", f"{end_A:.8f}", "0.01", f"{vmic:.2f}",
    ], check=True)


def replot_existing():
    rows = json.loads((OUT / "metrics.json").read_text())["rows"]
    style.configure("PAPER")
    fig, axes = plt.subplots(4, 3, figsize=(style.DOUBLE, 6.8), sharex=True, sharey=True)
    colors = {"payne_zero": style.PaperPalette.INK, "korg_native": style.PaperPalette.ORACLE, "korg_kurucz": style.PaperPalette.LEARNED}
    lines = {"payne_zero": "-", "korg_native": "--", "korg_kurucz": ":"}
    labels = {"payne_zero": "Payne-Zero", "korg_native": "Korg + GALAH DR3", "korg_kurucz": "Korg + PZ/Kurucz"}
    row_names = ["poor giant", "rich giant", "poor dwarf", "rich dwarf"]
    for ax, row in zip(axes.flat, rows):
        style.inward(ax)
        if row["status"] == "COMPLETE":
            with np.load(row["spectrum_path"]) as d:
                for key in ("payne_zero", "korg_native", "korg_kurucz"):
                    ax.plot(d["wavelength_air_A"], d[key], color=colors[key], ls=lines[key], lw=0.85, label=labels[key])
        else:
            ax.text(0.5, 0.5, "MISSING PZ atmosphere", ha="center", va="center", transform=ax.transAxes, color=style.PaperPalette.INK_MUTED, fontsize=7)
        ax.text(0.02, 0.05, rf"$\log g={row['logg']:.1f},\ [{{\rm M/H}}]={row['metallicity']:+.1f}$", transform=ax.transAxes, fontsize=6.5)
    for i, name in enumerate(row_names):
        axes[i, 0].set_ylabel(name + "\nNormalized flux")
    for ax in axes[-1]:
        ax.set_xlabel(r"Air wavelength ($\AA$)")
    for j, t in enumerate((3200, 3500, 4000)):
        axes[0, j].set_title(f"{t} K")
    handles, leglabels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leglabels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.997))
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.07, top=0.94, hspace=0.08, wspace=0.08)
    fig_path = OUT / "figures/cool_grid_tio_three_spectra"
    fig.savefig(fig_path.with_suffix(".pdf"))
    fig.savefig(fig_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    start_A, end_A = air_to_vacuum_A(np.array([AIR_WINDOW[0] - PAD_A, AIR_WINDOW[1] + PAD_A]))
    common_vac_A = np.arange(start_A, end_A + 0.005, 0.01)
    bundle = window_invariants_for(
        wl_start_nm=start_A / 10, wl_end_nm=end_A / 10, resolution=R_SYNTH,
        molecular_lines=True, runtime_device=torch.device("cpu"), work_dtype=torch.float64,
    )
    rows = []
    for kind, teff, logg, mh, vmic in NODES:
        stem = f"t{teff}_g{logg:.1f}_m{mh:+.1f}".replace("+", "p").replace("-", "m")
        source = atmosphere_path(kind, teff, logg, mh, vmic)
        row = {"kind": kind, "teff_K": teff, "logg": logg, "metallicity": mh,
               "alpha": 0.0, "vmic_km_s": vmic, "atmosphere_path": str(source.resolve()) if source else None}
        if source is None:
            row["status"] = "MISSING_LOCAL_ELIGIBLE_PZ_ATMOSPHERE"
            rows.append(row)
            continue
        atm_tsv, abund_tsv = export_atmosphere(source, stem)
        pz, seconds = synthesize_structured_atmosphere(
            source, wavelength_start_nm=start_A / 10, wavelength_end_nm=end_A / 10,
            resolution=R_SYNTH, molecular_lines=True, device="cpu", dtype=torch.float64,
            window_invariants=bundle,
        )
        pz_wave_A = np.asarray(pz.wavelength_nm) * 10
        pz_total = _surface_flux_per_wavelength_nm(np.asarray(pz.wavelength_nm), pz.eddington_flux_total_per_frequency)
        pz_cont = _surface_flux_per_wavelength_nm(np.asarray(pz.wavelength_nm), pz.eddington_flux_continuum_per_frequency)
        spectra = {"pz": common_spectrum(pz_wave_A, pz_total, pz_cont, common_vac_A)}
        inventories = {}
        for mode in ("native", "kurucz"):
            path = OUT / "raw" / f"{stem}_korg_{mode}.h5"
            path.parent.mkdir(parents=True, exist_ok=True)
            run_korg(mode, atm_tsv, abund_tsv, path, start_A, end_A, vmic)
            with h5py.File(path, "r") as h:
                spectra[mode] = common_spectrum(np.asarray(h["wavelength_vacuum_A"]), np.asarray(h["flux_total"]), np.asarray(h["flux_continuum"]), common_vac_A)
                inventories[mode] = {k: int(h.attrs[k]) for k in ("source_line_count", "window_line_count", "atomic_line_count", "molecular_line_count", "tio_line_count")}
        keep = (vacuum_to_air_A(common_vac_A) >= AIR_WINDOW[0]) & (vacuum_to_air_A(common_vac_A) <= AIR_WINDOW[1])
        out_npz = OUT / "spectra" / f"{stem}_three_spectra.npz"
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_npz, wavelength_air_A=vacuum_to_air_A(common_vac_A[keep]),
                            payne_zero=spectra["pz"][keep], korg_native=spectra["native"][keep],
                            korg_kurucz=spectra["kurucz"][keep])
        diffs = {
            "rms_pz_minus_korg_native": float(np.sqrt(np.mean((spectra["pz"][keep] - spectra["native"][keep]) ** 2))),
            "rms_pz_minus_korg_kurucz": float(np.sqrt(np.mean((spectra["pz"][keep] - spectra["kurucz"][keep]) ** 2))),
            "rms_korg_native_minus_korg_kurucz": float(np.sqrt(np.mean((spectra["native"][keep] - spectra["kurucz"][keep]) ** 2))),
        }
        row.update(status="COMPLETE", spectrum_path=str(out_npz.resolve()), pz_seconds=float(seconds), inventories=inventories, metrics=diffs)
        rows.append(row)
        print("completed", stem, diffs, flush=True)

    style.configure("PAPER")
    fig, axes = plt.subplots(4, 3, figsize=(style.DOUBLE, 6.8), sharex=True, sharey=True)
    colors = {"payne_zero": style.PaperPalette.INK, "korg_native": style.PaperPalette.ORACLE, "korg_kurucz": style.PaperPalette.LEARNED}
    lines = {"payne_zero": "-", "korg_native": "--", "korg_kurucz": ":"}
    labels = {"payne_zero": "Payne-Zero", "korg_native": "Korg + GALAH DR3", "korg_kurucz": "Korg + PZ/Kurucz"}
    row_names = ["poor giant", "rich giant", "poor dwarf", "rich dwarf"]
    for ax, row in zip(axes.flat, rows):
        style.inward(ax)
        if row["status"] == "COMPLETE":
            with np.load(row["spectrum_path"]) as d:
                for key in ("payne_zero", "korg_native", "korg_kurucz"):
                    ax.plot(d["wavelength_air_A"], d[key], color=colors[key], ls=lines[key], lw=0.85, label=labels[key])
        else:
            ax.text(0.5, 0.5, "MISSING PZ atmosphere", ha="center", va="center", transform=ax.transAxes, color=style.PaperPalette.INK_MUTED, fontsize=7)
        ax.text(0.02, 0.05, rf"$\log g={row['logg']:.1f},\ [{{\rm M/H}}]={row['metallicity']:+.1f}$", transform=ax.transAxes, fontsize=6.5)
    for i, name in enumerate(row_names):
        axes[i, 0].set_ylabel(name + "\nNormalized flux")
    for ax in axes[-1]: ax.set_xlabel(r"Air wavelength ($\AA$)")
    for j, t in enumerate((3200, 3500, 4000)): axes[0, j].set_title(f"{t} K")
    handles, leglabels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leglabels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.997))
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.07, top=0.94, hspace=0.08, wspace=0.08)
    fig_path = OUT / "figures/cool_grid_tio_three_spectra"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path.with_suffix(".pdf"))
    fig.savefig(fig_path.with_suffix(".png"), dpi=300)
    plt.close(fig)
    payload = {
        "definition": {"metal_poor": -1.0, "metal_rich": 0.5, "alpha": 0.0,
                       "giant_logg": {"poor": 2.5, "rich": 1.5}, "dwarf_logg": {"poor": 4.5, "rich": 5.0}},
        "window_air_A": list(AIR_WINDOW), "resolution": R_FINAL, "rv_km_s": 0.0,
        "rotation_km_s": 0.0, "continuum": "each total spectrum divided by its own continuum after the same R=80000 convolution",
        "wavelength_convention": "synthesis in vacuum; plotted after deterministic vacuum-to-air conversion",
        "kurucz_line_provenance": str(KURUCZ_LINES.resolve()), "rows": rows,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    replot_existing() if "--plot-only" in sys.argv else main()
