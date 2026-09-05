#!/usr/bin/env python3
"""One atomic-window companion to the v6 TiO grid figure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-mstar-ca-v6")
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

import compare_mstar_grid_three_spectra_v6 as grid

OUT = grid.OUT
SOURCE = ROOT / "results/m_star_emulator_v1r2_marcs100/cases/dwarf/g+4.50_m-1.00_a+0.00_c+0.00_x1.00/t4000/products/primary/t04000.0_g+4.50_m-1.00_a+0.00_x1.00.npz"
LINES = ROOT / "results/m_star_eso_highres_comparison_v5_same_linelist/inputs/pz_atomic_transitions_ca_i.tsv"
AIR_WINDOW = (6155.0, 6170.0)


def common_spectrum(wave_A, total, continuum, common_A):
    total_i = np.interp(common_A, wave_A, total)
    cont_i = np.interp(common_A, wave_A, continuum)
    sigma = np.mean(common_A) / grid.R_FINAL / 2.354820045 / np.median(np.diff(common_A))
    return gaussian_filter1d(total_i, sigma, mode="nearest") / gaussian_filter1d(cont_i, sigma, mode="nearest")


def main():
    start_A, end_A = grid.air_to_vacuum_A(np.array([AIR_WINDOW[0] - grid.PAD_A, AIR_WINDOW[1] + grid.PAD_A]))
    common_A = np.arange(start_A, end_A + 0.005, 0.01)
    bundle = window_invariants_for(wl_start_nm=start_A / 10, wl_end_nm=end_A / 10,
        resolution=grid.R_SYNTH, molecular_lines=True, runtime_device=torch.device("cpu"), work_dtype=torch.float64)
    pz, seconds = synthesize_structured_atmosphere(SOURCE, wavelength_start_nm=start_A / 10,
        wavelength_end_nm=end_A / 10, resolution=grid.R_SYNTH, molecular_lines=True,
        device="cpu", dtype=torch.float64, window_invariants=bundle)
    wave = np.asarray(pz.wavelength_nm) * 10
    spectra = {"payne_zero": common_spectrum(wave,
        _surface_flux_per_wavelength_nm(np.asarray(pz.wavelength_nm), pz.eddington_flux_total_per_frequency),
        _surface_flux_per_wavelength_nm(np.asarray(pz.wavelength_nm), pz.eddington_flux_continuum_per_frequency), common_A)}
    atm, abund = grid.export_atmosphere(SOURCE, "t4000_g4.5_mm1.0_ca")
    inventory = {}
    for key, mode in (("korg_native", "native_atomic"), ("korg_kurucz", "kurucz_atomic")):
        path = OUT / "raw" / f"t4000_g4.5_mm1.0_ca_{mode}.h5"
        subprocess.run(["julia", "--project=" + str(grid.KORG_ROOT), str(grid.KORG_HELPER), mode,
            str(LINES), str(atm), str(abund), str(path), f"{start_A:.8f}", f"{end_A:.8f}", "0.01", "1.0"], check=True)
        with h5py.File(path, "r") as h:
            spectra[key] = common_spectrum(np.asarray(h["wavelength_vacuum_A"]), np.asarray(h["flux_total"]), np.asarray(h["flux_continuum"]), common_A)
            inventory[key] = {name: int(h.attrs[name]) for name in ("source_line_count", "window_line_count", "atomic_line_count", "molecular_line_count", "tio_line_count")}
    air = grid.vacuum_to_air_A(common_A)
    keep = (air >= AIR_WINDOW[0]) & (air <= AIR_WINDOW[1])
    spec_path = OUT / "spectra/t4000_g4.5_mm1.0_ca_three_spectra.npz"
    np.savez_compressed(spec_path, wavelength_air_A=air[keep], **{k: v[keep] for k, v in spectra.items()})
    metrics = {"rms_pz_minus_korg_native": float(np.sqrt(np.mean((spectra["payne_zero"][keep]-spectra["korg_native"][keep])**2))),
               "rms_pz_minus_korg_kurucz": float(np.sqrt(np.mean((spectra["payne_zero"][keep]-spectra["korg_kurucz"][keep])**2))),
               "rms_korg_native_minus_korg_kurucz": float(np.sqrt(np.mean((spectra["korg_native"][keep]-spectra["korg_kurucz"][keep])**2)))}
    style.configure("PAPER")
    fig, ax = plt.subplots(figsize=(style.SINGLE, 2.35))
    style.inward(ax)
    ax.plot(air[keep], spectra["payne_zero"][keep], color=style.PaperPalette.INK, ls="-", label="Payne-Zero")
    ax.plot(air[keep], spectra["korg_native"][keep], color=style.PaperPalette.ORACLE, ls="--", label="Korg/VALD")
    ax.plot(air[keep], spectra["korg_kurucz"][keep], color=style.PaperPalette.LEARNED, ls=":", label="Korg/PZ-Kurucz")
    ax.set(xlabel=r"Air wavelength ($\AA$)", ylabel="Normalized flux", title=r"4000 K dwarf, $\log g=4.5$, $[{\rm M/H}]=-1.0$")
    ax.legend(ncol=1, loc="lower left", fontsize=6.5, labelspacing=0.55)
    fig.tight_layout()
    base = OUT / "figures/cool_grid_ca_three_spectra_t4000_poor_dwarf"
    fig.savefig(base.with_suffix(".pdf")); fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)
    (OUT / "ca_metrics.json").write_text(json.dumps({"node": {"teff_K": 4000, "logg": 4.5, "metallicity": -1.0, "alpha": 0.0, "vmic_km_s": 1.0},
        "window_air_A": list(AIR_WINDOW), "resolution": grid.R_FINAL, "pz_seconds": float(seconds),
        "inventories": inventory, "metrics": metrics, "spectrum_path": str(spec_path.resolve())}, indent=2) + "\n")


if __name__ == "__main__": main()
