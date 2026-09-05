#!/usr/bin/env python3
"""Ca-window matched-transition comparison of native Payne-Zero and Korg."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-mstar-eso-v5")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault(
    "PAYNE_ZERO_SYNTHESIS_CACHE_DIR",
    str(PROJECT_ROOT / ".cache/payne-zero/synthesis"),
)
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import compare_mstar_eso_highres_v1 as v1
import compare_mstar_eso_highres_v3_shared_nuisance as v3
import compare_mstar_eso_native_paynezero_v4 as v4
from payne_zero_synthesis import atomic_lines, paths as synthesis_paths
from payne_zero_synthesis.api import _surface_flux_per_wavelength_nm
from payne_zero_synthesis import line_opacity as line_opacity_engine
from payne_zero_synthesis.pipeline import SynthesisPipeline, window_invariants_for
from payne_zero_synthesis.synthesis import synthesize_structured_atmosphere


RESULT_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v5_same_linelist"
V1_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v1"
ATMOSPHERE_PATH = v4.ATMOSPHERE_PATH
KORG_ROOT = Path("/Users/jdli/Project/jorg/Korg.jl-1.0.1")
KORG_HELPER = Path(__file__).resolve().parent / "korg_synthesize_pz_matched_lines_v5.jl"
KORG_MIXED_HELPER = Path(__file__).resolve().parent / "korg_synthesize_pz_matched_mixed_lines_v5.jl"
TARGET = "IC2391-0096"
R_GRID = 600_000.0
KORG_STEP_A = 0.002
WINDOW = {"name": "ca_i_6162", "label": "Ca I", "lo": 6155.0, "hi": 6170.0}
TIO_WINDOW = {"name": "tio_6650", "label": "TiO-rich", "lo": 6650.0, "hi": 6670.0}
PADDING_A = 3.0
CM_TO_EV = 1.2398419843320026e-4
LIGHT_SPEED_NM_S = 2.99792458e17


def export_atmosphere_and_abundances() -> tuple[Path, Path]:
    atmosphere_tsv = RESULT_ROOT / "inputs" / "payne_zero_atmosphere.tsv"
    abundance_tsv = RESULT_ROOT / "inputs" / "elemental_number_fractions_Z1_Z92.tsv"
    atmosphere_tsv.parent.mkdir(parents=True, exist_ok=True)
    with np.load(ATMOSPHERE_PATH, allow_pickle=False) as data:
        matrix = np.column_stack(
            [data[key] for key in ("temperature", "electron_density", "gas_pressure", "mass_density", "column_mass")]
        )
        abundances = np.asarray(data["elemental_abundances"][:92], dtype=np.float64)
    np.savetxt(
        atmosphere_tsv,
        matrix,
        delimiter="\t",
        header="temperature_K\telectron_density_cm-3\tgas_pressure_dyn_cm-2\tmass_density_g_cm-3\tcolumn_mass_g_cm-2",
        comments="",
    )
    np.savetxt(abundance_tsv, abundances[None, :], delimiter="\t")
    return atmosphere_tsv, abundance_tsv


def build_common_atomic_bundle(start_nm: float, end_nm: float):
    bundle = window_invariants_for(
        wl_start_nm=start_nm,
        wl_end_nm=end_nm,
        resolution=R_GRID,
        molecular_lines=False,
        runtime_device=torch.device("cpu"),
        work_dtype=torch.float64,
    )
    line_type = np.asarray(bundle.atomic_kernel_catalog["line_type"])
    ion_stage = np.asarray(bundle.atomic_kernel_catalog["ion_stage"])
    metal = ((line_type == 0) | (line_type == 1) | (line_type == 3)) & (ion_stage <= 3)
    common_indices = np.flatnonzero(metal)
    common_catalog = SynthesisPipeline._slice_atomic_catalog(bundle.atomic_kernel_catalog, common_indices)
    common_invariants = line_opacity_engine.precompute_invariants(
        common_catalog,
        bundle.synthesis_wavelength_nm,
        runtime_device=torch.device("cpu"),
    )
    return dataclasses.replace(
        bundle,
        n_atomic=int(metal.sum()),
        has_helium=False,
        has_hydrogen=False,
        helium_invariants=None,
        hydrogen_invariants_template=None,
        metal_invariant_chunks=[common_invariants],
    ), metal


def export_atomic_transitions(bundle, metal: np.ndarray) -> tuple[Path, dict[str, np.ndarray]]:
    catalog = bundle.atomic_kernel_catalog
    frequency_hz = LIGHT_SPEED_NM_S / np.asarray(catalog["wavelength_nm"])[metal]
    damping_factor = 12.5664 * frequency_hz
    arrays = {
        "wavelength_nm": np.asarray(catalog["wavelength_nm"])[metal].astype(np.float64),
        "loggf": np.asarray(catalog["log_oscillator_strength"])[metal].astype(np.float64),
        "atomic_number": np.asarray(catalog["atomic_number"])[metal].astype(np.int64),
        "ion_stage": np.asarray(catalog["ion_stage"])[metal].astype(np.int64),
        "E_lower_eV": np.asarray(catalog["lower_excitation_cm"])[metal].astype(np.float64) * CM_TO_EV,
        "gamma_rad_s": np.asarray(catalog["radiative_damping"])[metal].astype(np.float64) * damping_factor,
        "gamma_stark_s": np.asarray(catalog["stark_damping"])[metal].astype(np.float64) * damping_factor,
        "gamma_vdw_s": np.asarray(catalog["van_der_waals_damping"])[metal].astype(np.float64) * damping_factor,
    }
    arrays["log_gamma_vdw"] = np.log10(arrays["gamma_vdw_s"])
    order = np.argsort(arrays["wavelength_nm"], kind="stable")
    arrays = {key: value[order] for key, value in arrays.items()}
    path = RESULT_ROOT / "inputs" / "pz_atomic_transitions_ca_i.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.column_stack(
        [arrays[key] for key in ("wavelength_nm", "loggf", "atomic_number", "ion_stage", "E_lower_eV", "gamma_rad_s", "gamma_stark_s", "log_gamma_vdw")]
    )
    np.savetxt(
        path,
        values,
        delimiter="\t",
        header="wavelength_nm\tloggf\tatomic_number\tion_stage\tE_lower_eV\tgamma_rad_s\tgamma_stark_s\tlog_gamma_vdw",
        comments="",
        fmt=["%.16e", "%.16e", "%d", "%d", "%.16e", "%.16e", "%.16e", "%.16e"],
    )
    return path, arrays


def run_native(bundle, start_nm: float, end_nm: float, *, molecular_lines: bool = False):
    result, seconds = synthesize_structured_atmosphere(
        ATMOSPHERE_PATH,
        wavelength_start_nm=start_nm,
        wavelength_end_nm=end_nm,
        resolution=R_GRID,
        molecular_lines=molecular_lines,
        device="cpu",
        dtype=torch.float64,
        window_invariants=bundle,
    )
    wavelength_nm = np.asarray(result.wavelength_nm, dtype=np.float64)
    return {
        "wavelength_nm": wavelength_nm,
        "flux_total": _surface_flux_per_wavelength_nm(wavelength_nm, result.eddington_flux_total_per_frequency),
        "flux_continuum": _surface_flux_per_wavelength_nm(wavelength_nm, result.eddington_flux_continuum_per_frequency),
        "seconds": float(seconds),
    }


def run_korg(line_path: Path, atmosphere_tsv: Path, abundance_tsv: Path, start_nm: float, end_nm: float) -> Path:
    output_path = RESULT_ROOT / "spectra" / "IC2391-0096_ca_i_6162_korg_matched_atomic.h5"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "julia",
            "--project=" + str(KORG_ROOT),
            str(KORG_HELPER),
            str(line_path),
            str(atmosphere_tsv),
            str(abundance_tsv),
            str(output_path),
            f"{start_nm * 10.0:.12f}",
            f"{end_nm * 10.0:.12f}",
            f"{KORG_STEP_A:.12f}",
        ],
        check=True,
    )
    return output_path


def build_common_mixed_bundle(start_nm: float, end_nm: float):
    bundle = window_invariants_for(
        wl_start_nm=start_nm,
        wl_end_nm=end_nm,
        resolution=R_GRID,
        molecular_lines=True,
        runtime_device=torch.device("cpu"),
        work_dtype=torch.float64,
    )
    line_type = np.asarray(bundle.atomic_kernel_catalog["line_type"])
    ion_stage = np.asarray(bundle.atomic_kernel_catalog["ion_stage"])
    atomic_common = ((line_type == 0) | (line_type == 1) | (line_type == 3)) & (ion_stage <= 3)
    atomic_indices = np.flatnonzero(atomic_common)
    atomic_catalog = SynthesisPipeline._slice_atomic_catalog(bundle.atomic_kernel_catalog, atomic_indices)
    atomic_invariants = line_opacity_engine.precompute_invariants(
        atomic_catalog, bundle.synthesis_wavelength_nm, runtime_device=torch.device("cpu")
    )
    return dataclasses.replace(
        bundle,
        n_atomic=int(atomic_common.sum()),
        has_helium=False,
        has_hydrogen=False,
        helium_invariants=None,
        hydrogen_invariants_template=None,
        metal_invariant_chunks=[atomic_invariants],
    ), atomic_common


def export_mixed_transitions(bundle, atomic_common: np.ndarray) -> tuple[Path, dict[str, np.ndarray], dict]:
    atomic = bundle.atomic_kernel_catalog
    atomic_frequency = LIGHT_SPEED_NM_S / np.asarray(atomic["wavelength_nm"])[atomic_common]
    atomic_factor = 12.5664 * atomic_frequency
    atomic_arrays = {
        "wavelength_nm": np.asarray(atomic["wavelength_nm"])[atomic_common].astype(np.float64),
        "loggf": np.asarray(atomic["log_oscillator_strength"])[atomic_common].astype(np.float64),
        "kind": np.zeros(int(atomic_common.sum()), dtype=np.int64),
        "source_species_code": np.asarray(atomic["atomic_number"])[atomic_common].astype(np.int64),
        "ion_stage": np.asarray(atomic["ion_stage"])[atomic_common].astype(np.int64),
        "E_lower_eV": np.asarray(atomic["lower_excitation_cm"])[atomic_common].astype(np.float64) * CM_TO_EV,
        "gamma_rad_s": np.asarray(atomic["radiative_damping"])[atomic_common].astype(np.float64) * atomic_factor,
        "gamma_stark_s": np.asarray(atomic["stark_damping"])[atomic_common].astype(np.float64) * atomic_factor,
        "gamma_vdw_s": np.asarray(atomic["van_der_waals_damping"])[atomic_common].astype(np.float64) * atomic_factor,
    }
    molecular = bundle.molecular_invariants
    line_species_index = molecular.line_species_index.detach().cpu().numpy().astype(np.int64)
    molecular_codes = molecular.species_code.detach().cpu().numpy().astype(np.int64)[line_species_index]
    molecular_wavelength = molecular.line_wavelength_nm.detach().cpu().numpy().astype(np.float64)
    molecular_frequency = LIGHT_SPEED_NM_S / molecular_wavelength
    molecular_factor = 12.5664 * molecular_frequency
    molecular_gf = (
        molecular.classical_line_strength.detach().cpu().numpy().astype(np.float64)
        * molecular_frequency / 0.01502
    )
    molecular_arrays = {
        "wavelength_nm": molecular_wavelength,
        "loggf": np.log10(molecular_gf),
        "kind": np.ones(molecular_wavelength.size, dtype=np.int64),
        "source_species_code": molecular_codes,
        "ion_stage": np.zeros(molecular_wavelength.size, dtype=np.int64),
        "E_lower_eV": molecular.lower_excitation_cm.detach().cpu().numpy().astype(np.float64) * CM_TO_EV,
        "gamma_rad_s": molecular.radiative_damping.detach().cpu().numpy().astype(np.float64) * molecular_factor,
        "gamma_stark_s": molecular.stark_damping.detach().cpu().numpy().astype(np.float64) * molecular_factor,
        "gamma_vdw_s": molecular.van_der_waals_damping.detach().cpu().numpy().astype(np.float64) * molecular_factor,
    }
    keys = tuple(atomic_arrays)
    arrays = {key: np.concatenate((atomic_arrays[key], molecular_arrays[key])) for key in keys}
    order = np.argsort(arrays["wavelength_nm"], kind="stable")
    arrays = {key: value[order] for key, value in arrays.items()}
    arrays["log_gamma_vdw"] = np.log10(arrays["gamma_vdw_s"])
    path = RESULT_ROOT / "inputs" / "pz_atomic_molecular_transitions_tio.tsv"
    columns = (
        "wavelength_nm", "loggf", "kind", "source_species_code", "ion_stage",
        "E_lower_eV", "gamma_rad_s", "gamma_stark_s", "log_gamma_vdw",
    )
    np.savetxt(
        path,
        np.column_stack([arrays[key] for key in columns]),
        delimiter="\t",
        header="\t".join(columns),
        comments="",
        fmt=["%.16e", "%.16e", "%d", "%d", "%d", "%.16e", "%.16e", "%.16e", "%.16e"],
    )
    codes, counts = np.unique(molecular_codes, return_counts=True)
    inventory = {
        "atomic_count": int(atomic_common.sum()),
        "molecular_count": int(molecular_wavelength.size),
        "total_count": int(arrays["wavelength_nm"].size),
        "molecular_species_counts": {str(int(code)): int(count) for code, count in zip(codes, counts)},
        "isotope_identity_available": False,
        "isotope_note": "compiled PZ records retain isotope-adjusted gf but aggregate isotopologues into one molecular species code",
    }
    return path, arrays, inventory


def run_korg_mixed(line_path: Path, atmosphere_tsv: Path, abundance_tsv: Path, start_nm: float, end_nm: float) -> Path:
    output_path = RESULT_ROOT / "spectra" / "IC2391-0096_tio_6650_korg_matched_transitions.h5"
    subprocess.run(
        [
            "julia", "--project=" + str(KORG_ROOT), str(KORG_MIXED_HELPER),
            str(line_path), str(atmosphere_tsv), str(abundance_tsv), str(output_path),
            f"{start_nm * 10.0:.12f}", f"{end_nm * 10.0:.12f}", f"{KORG_STEP_A:.12f}",
        ],
        check=True,
    )
    return output_path


def compare_mixed_identity(expected: dict[str, np.ndarray], korg_path: Path) -> dict:
    dataset_names = {
        "wavelength_nm": "line_wavelength_nm",
        "loggf": "line_loggf",
        "kind": "line_kind",
        "source_species_code": "line_source_species_code",
        "ion_stage": "line_ion_stage",
        "E_lower_eV": "line_E_lower_eV",
        "gamma_rad_s": "line_gamma_rad_s",
        "gamma_stark_s": "line_gamma_stark_s",
        "gamma_vdw_s": "line_gamma_vdw_s",
    }
    with h5py.File(korg_path, "r") as handle:
        actual = {key: np.asarray(handle[name]) for key, name in dataset_names.items()}
        attrs = {key: handle.attrs[key].item() if isinstance(handle.attrs[key], np.generic) else handle.attrs[key] for key in handle.attrs}
    exact_keys = {"kind", "source_species_code", "ion_stage"}
    tolerances = {
        "wavelength_nm": (0.0, 2e-13), "loggf": (0.0, 2e-14),
        "E_lower_eV": (2e-15, 1e-14), "gamma_rad_s": (2e-15, 0.0),
        "gamma_stark_s": (2e-15, 0.0), "gamma_vdw_s": (2e-15, 0.0),
    }
    fields = {}
    for key, values in actual.items():
        expected_values = expected[key]
        delta = values.astype(float) - expected_values.astype(float)
        if key in exact_keys:
            matched = np.array_equal(values, expected_values)
            tolerance = {"exact": True}
        else:
            rtol, atol = tolerances[key]
            matched = np.allclose(values, expected_values, rtol=rtol, atol=atol)
            tolerance = {"rtol": rtol, "atol": atol}
        fields[key] = {
            "within_tolerance": bool(matched),
            "maximum_absolute_difference": float(np.max(np.abs(delta), initial=0.0)),
            "tolerance": tolerance,
        }
    return {
        "korg_input_line_count": int(attrs["input_line_count"]),
        "korg_used_line_count": int(attrs["used_line_count"]),
        "korg_atomic_line_count": int(attrs["atomic_line_count"]),
        "korg_molecular_line_count": int(attrs["molecular_line_count"]),
        "fields": fields,
        "matched_transition_fields": bool(
            int(attrs["used_line_count"]) == expected["wavelength_nm"].size
            and all(item["within_tolerance"] for item in fields.values())
        ),
        "identity_label": "matched-transition list, not identical line list, because isotope identity is absent from PZ compiled records",
    }


def compare_line_identity(expected: dict[str, np.ndarray], korg_path: Path) -> dict:
    with h5py.File(korg_path, "r") as handle:
        actual = {
            "wavelength_nm": np.asarray(handle["line_wavelength_nm"]),
            "loggf": np.asarray(handle["line_loggf"]),
            "atomic_number": np.asarray(handle["line_atomic_number"]),
            "ion_stage": np.asarray(handle["line_ion_stage"]),
            "E_lower_eV": np.asarray(handle["line_E_lower_eV"]),
            "gamma_rad_s": np.asarray(handle["line_gamma_rad_s"]),
            "gamma_stark_s": np.asarray(handle["line_gamma_stark_s"]),
            "gamma_vdw_s": np.asarray(handle["line_gamma_vdw_s"]),
        }
        attrs = {key: handle.attrs[key].item() if isinstance(handle.attrs[key], np.generic) else handle.attrs[key] for key in handle.attrs}
    tolerances = {
        "wavelength_nm": {"rtol": 0.0, "atol": 2e-13},
        "loggf": {"rtol": 0.0, "atol": 1e-14},
        "atomic_number": {"rtol": 0.0, "atol": 0.0},
        "ion_stage": {"rtol": 0.0, "atol": 0.0},
        "E_lower_eV": {"rtol": 2e-15, "atol": 1e-14},
        "gamma_rad_s": {"rtol": 2e-15, "atol": 0.0},
        "gamma_stark_s": {"rtol": 2e-15, "atol": 0.0},
        "gamma_vdw_s": {"rtol": 2e-15, "atol": 0.0},
    }
    fields = {}
    for key in actual:
        delta = np.asarray(actual[key], dtype=np.float64) - np.asarray(expected[key], dtype=np.float64)
        scale = np.maximum(np.abs(np.asarray(expected[key], dtype=np.float64)), np.finfo(float).tiny)
        fields[key] = {
            "exact_array_equal": bool(np.array_equal(actual[key], expected[key])),
            "maximum_absolute_difference": float(np.max(np.abs(delta), initial=0.0)),
            "maximum_relative_difference": float(np.max(np.abs(delta) / scale, initial=0.0)),
            "tolerance": tolerances[key],
            "within_tolerance": bool(np.allclose(actual[key], expected[key], **tolerances[key])),
        }
    return {
        "pz_metal_transition_count": int(expected["wavelength_nm"].size),
        "korg_input_line_count": int(attrs["input_line_count"]),
        "korg_used_line_count": int(attrs["used_line_count"]),
        "fields": fields,
        "transition_identity": bool(
            int(attrs["used_line_count"]) == expected["wavelength_nm"].size
            and all(field["within_tolerance"] for field in fields.values())
        ),
        "damping_interpretation": (
            "PZ normalized gamma fields were inverted with the same 12.5664*nu factor used by PZ; "
            "Korg stores the recovered gamma values, but applies its own density and temperature scaling."
        ),
    }


def project(native_wavelength_nm: np.ndarray, total: np.ndarray, continuum: np.ndarray, observed_air_A: np.ndarray):
    operator = v4.RotationThenInstrument(native_wavelength_nm, v4.air_to_vacuum_nm(observed_air_A / 10.0))
    total_tensor = torch.as_tensor(total, dtype=torch.float64)
    continuum_tensor = torch.as_tensor(continuum, dtype=torch.float64)
    projected = operator.convolve_fluxes(total_tensor, continuum_tensor)
    return projected[2].detach().cpu().numpy(), operator.metadata()


def make_figure(wave, obs, error, native_prediction, korg_prediction, metrics, title, path):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.0]})
    axes[0].fill_between(wave, obs - error, obs + error, color="0.75", alpha=0.25, lw=0)
    axes[0].plot(wave, obs, color="black", lw=0.8, label="UVES-POP")
    axes[0].plot(wave, native_prediction, color="#2878B5", lw=0.95, label="native Payne-Zero")
    axes[0].plot(wave, korg_prediction, color="#E07B39", lw=0.95, label="Korg")
    axes[1].plot(wave, obs - native_prediction, color="#2878B5", lw=0.85, label="obs - native PZ")
    axes[1].plot(wave, obs - korg_prediction, color="#E07B39", lw=0.85, label="obs - Korg")
    axes[1].axhline(0, color="0.35", lw=0.6)
    axes[0].set_ylabel("locally normalized flux")
    axes[1].set_ylabel("obs - model")
    axes[1].set_xlabel("catalogue-rest air wavelength (A)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    axes[0].set_title(title)
    axes[1].text(
        0.01, 0.96,
        f"native RMS {metrics['native_payne_zero']['rms']:.5f}; Korg RMS {metrics['korg']['rms']:.5f}\n"
        f"native-Korg {metrics['prediction_rms_difference']:.5f}",
        transform=axes[1].transAxes, va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    config = v1.TARGETS[TARGET]
    wave, flux, error, good, inventory = v1.load_observation(
        V1_ROOT / "observations" / config["fits"], config["rv_km_s"]
    )
    selected = good & (wave >= WINDOW["lo"]) & (wave <= WINDOW["hi"])
    w, f, e = wave[selected], flux[selected], error[selected]
    scale = float(np.percentile(f, 95.0))
    obs, obs_error = f / scale, e / scale
    start_nm = float(v4.air_to_vacuum_nm((WINDOW["lo"] - PADDING_A) / 10.0))
    end_nm = float(v4.air_to_vacuum_nm((WINDOW["hi"] + PADDING_A) / 10.0))

    bundle, metal = build_common_atomic_bundle(start_nm, end_nm)
    line_path, expected_lines = export_atomic_transitions(bundle, metal)
    atmosphere_tsv, abundance_tsv = export_atmosphere_and_abundances()
    native = run_native(bundle, start_nm, end_nm)
    korg_path = run_korg(line_path, atmosphere_tsv, abundance_tsv, start_nm, end_nm)
    line_identity = compare_line_identity(expected_lines, korg_path)
    catalog_line_type = np.asarray(bundle.atomic_kernel_catalog["line_type"])
    all_metal = (catalog_line_type == 0) | (catalog_line_type == 1) | (catalog_line_type == 3)
    line_identity["pz_source_metal_count_before_common_species_support"] = int(all_metal.sum())
    line_identity["excluded_higher_ion_stage_count"] = int(all_metal.sum() - metal.sum())
    line_identity["common_species_support"] = "atomic ion stages I-III, supported by both synthesizers"
    line_identity["korg_internal_reference_lines"] = (
        "enabled only for the anchored tau5000 reference opacity; they do not enter target-window line opacity"
    )
    if not line_identity["transition_identity"]:
        raise RuntimeError("Korg did not use the exported PZ transition list exactly")

    native_raw, operator_metadata = project(
        native["wavelength_nm"], native["flux_total"], native["flux_continuum"], w
    )
    with h5py.File(korg_path, "r") as handle:
        korg_wave_nm = np.asarray(handle["wavelength_vacuum_A"], dtype=np.float64) / 10.0
        korg_total = np.asarray(handle["flux_total"], dtype=np.float64)
        korg_continuum = np.asarray(handle["flux_continuum"], dtype=np.float64)
    korg_total_on_native = np.interp(native["wavelength_nm"], korg_wave_nm, korg_total)
    korg_continuum_on_native = np.interp(native["wavelength_nm"], korg_wave_nm, korg_continuum)
    korg_raw, _ = project(native["wavelength_nm"], korg_total_on_native, korg_continuum_on_native, w)
    native_prediction, native_coeff = v1.continuum_fit(obs, native_raw, w)
    korg_prediction, korg_coeff = v1.continuum_fit(obs, korg_raw, w)
    native_residual = obs - native_prediction
    korg_residual = obs - korg_prediction

    window_metrics = {
        "native_payne_zero": {
            **v1.residual_metrics(native_residual),
            "continuum_coefficients_a_b": [float(x) for x in native_coeff],
            "ca_i_line_diagnostic": v3.line_center_and_fwhm(w, native_prediction),
        },
        "korg": {
            **v1.residual_metrics(korg_residual),
            "continuum_coefficients_a_b": [float(x) for x in korg_coeff],
            "ca_i_line_diagnostic": v3.line_center_and_fwhm(w, korg_prediction),
        },
        "observation": {"ca_i_line_diagnostic": v3.line_center_and_fwhm(w, obs)},
        "noise_rms": float(np.sqrt(np.mean(obs_error**2))),
        "prediction_rms_difference": float(np.sqrt(np.mean((native_prediction - korg_prediction) ** 2))),
        "residual_correlation": float(np.corrcoef(native_residual, korg_residual)[0, 1]),
    }
    output = {
        "experiment": "same atomic transitions in native Payne-Zero and Korg",
        "target": TARGET,
        "window": WINDOW,
        "atmosphere_path": str(ATMOSPHERE_PATH.resolve()),
        "observation": inventory,
        "line_identity": line_identity,
        "fixed_nuisance": {
            "residual_rv_km_s": v4.RESIDUAL_RV_KM_S,
            "vsini_km_s": v4.VSINI_KM_S,
            "instrument_resolving_power": v4.INSTRUMENT_R,
            "fit_status": "frozen from v3",
        },
        "sampling": {
            "native_r_grid": R_GRID,
            "korg_raw_step_A": KORG_STEP_A,
            "korg_interpolated_to_native_log_grid_before_common_broadening": True,
            "wavelength_projection": "observed air pixels converted to vacuum",
        },
        "common_projection": operator_metadata,
        "continuum_nuisance": "each synthesizer independently gets model*(a+b*x)",
        "metrics": window_metrics,
        "remaining_model_differences": [
            "equation of state and electron-density treatment",
            "partition functions and ionization equilibrium",
            "molecular equilibrium",
            "continuum opacity",
            "intrinsic thermal and pressure broadening, line opacity, and radiative transfer",
            "Korg planar remapping of the PZ column-mass structure",
        ],
        "solver_campaign_success_rate_used": False,
        "giant_used": False,
        "tio_status": "not_attempted_yet",
    }
    native_ca_path = RESULT_ROOT / "spectra" / "IC2391-0096_ca_i_6162_native_matched_atomic.npz"
    np.savez_compressed(native_ca_path, **native)

    tio_selected = good & (wave >= TIO_WINDOW["lo"]) & (wave <= TIO_WINDOW["hi"])
    tio_w, tio_f, tio_e = wave[tio_selected], flux[tio_selected], error[tio_selected]
    tio_scale = float(np.percentile(tio_f, 95.0))
    tio_obs, tio_obs_error = tio_f / tio_scale, tio_e / tio_scale
    tio_start_nm = float(v4.air_to_vacuum_nm((TIO_WINDOW["lo"] - PADDING_A) / 10.0))
    tio_end_nm = float(v4.air_to_vacuum_nm((TIO_WINDOW["hi"] + PADDING_A) / 10.0))
    mixed_bundle, mixed_atomic_common = build_common_mixed_bundle(tio_start_nm, tio_end_nm)
    mixed_line_path, mixed_expected, mixed_inventory = export_mixed_transitions(
        mixed_bundle, mixed_atomic_common
    )
    native_tio = run_native(mixed_bundle, tio_start_nm, tio_end_nm, molecular_lines=True)
    native_tio_path = RESULT_ROOT / "spectra" / "IC2391-0096_tio_6650_native_matched_transitions.npz"
    np.savez_compressed(native_tio_path, **native_tio)
    korg_tio_path = run_korg_mixed(
        mixed_line_path, atmosphere_tsv, abundance_tsv, tio_start_nm, tio_end_nm
    )
    mixed_identity = compare_mixed_identity(mixed_expected, korg_tio_path)
    if not mixed_identity["matched_transition_fields"]:
        raise RuntimeError("Korg did not retain every exported PZ matched transition")
    native_tio_raw, tio_operator_metadata = project(
        native_tio["wavelength_nm"], native_tio["flux_total"], native_tio["flux_continuum"], tio_w
    )
    with h5py.File(korg_tio_path, "r") as handle:
        korg_tio_wave_nm = np.asarray(handle["wavelength_vacuum_A"], dtype=np.float64) / 10.0
        korg_tio_total = np.asarray(handle["flux_total"], dtype=np.float64)
        korg_tio_continuum = np.asarray(handle["flux_continuum"], dtype=np.float64)
    korg_tio_total_native = np.interp(native_tio["wavelength_nm"], korg_tio_wave_nm, korg_tio_total)
    korg_tio_continuum_native = np.interp(native_tio["wavelength_nm"], korg_tio_wave_nm, korg_tio_continuum)
    korg_tio_raw, _ = project(
        native_tio["wavelength_nm"], korg_tio_total_native, korg_tio_continuum_native, tio_w
    )
    native_tio_prediction, native_tio_coeff = v1.continuum_fit(tio_obs, native_tio_raw, tio_w)
    korg_tio_prediction, korg_tio_coeff = v1.continuum_fit(tio_obs, korg_tio_raw, tio_w)
    native_tio_residual = tio_obs - native_tio_prediction
    korg_tio_residual = tio_obs - korg_tio_prediction
    tio_metrics = {
        "native_payne_zero": {
            **v1.residual_metrics(native_tio_residual),
            "continuum_coefficients_a_b": [float(x) for x in native_tio_coeff],
            "synthesis_seconds": native_tio["seconds"],
        },
        "korg": {
            **v1.residual_metrics(korg_tio_residual),
            "continuum_coefficients_a_b": [float(x) for x in korg_tio_coeff],
        },
        "noise_rms": float(np.sqrt(np.mean(tio_obs_error**2))),
        "prediction_rms_difference": float(
            np.sqrt(np.mean((native_tio_prediction - korg_tio_prediction) ** 2))
        ),
        "residual_correlation": float(np.corrcoef(native_tio_residual, korg_tio_residual)[0, 1]),
    }
    output["tio_status"] = "completed_matched_transition_list"
    output["tio"] = {
        "window": TIO_WINDOW,
        "transition_inventory": mixed_inventory,
        "line_identity": mixed_identity,
        "common_projection": tio_operator_metadata,
        "metrics": tio_metrics,
        "scope": (
            "same compiled transition centers, aggregate species codes, isotope-adjusted loggf, lower energies, "
            "and damping parameters; isotope labels cannot be reconstructed from the compiled PZ records"
        ),
    }
    tio_processed_path = RESULT_ROOT / "processed" / f"{TARGET}_tio_6650.npz"
    tio_processed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        tio_processed_path,
        wavelength_air_catalogue_rest_A=tio_w,
        observed_normalized=tio_obs,
        observed_error_normalized=tio_obs_error,
        native_prediction=native_tio_prediction,
        native_residual=native_tio_residual,
        korg_prediction=korg_tio_prediction,
        korg_residual=korg_tio_residual,
    )
    tio_figure_path = RESULT_ROOT / "figures" / f"{TARGET}_tio_6650_matched_transitions.png"
    tio_figure_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(
        tio_w, tio_obs, tio_obs_error, native_tio_prediction, korg_tio_prediction,
        tio_metrics,
        f"IC2391-0096 TiO-rich 6650-6670 A: {mixed_inventory['total_count']} matched transitions",
        tio_figure_path,
    )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "metrics.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    processed_path = RESULT_ROOT / "processed" / f"{TARGET}_ca_i_6162.npz"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_path,
        wavelength_air_catalogue_rest_A=w,
        observed_normalized=obs,
        observed_error_normalized=obs_error,
        native_prediction=native_prediction,
        native_residual=native_residual,
        korg_prediction=korg_prediction,
        korg_residual=korg_residual,
    )
    figure_path = RESULT_ROOT / "figures" / f"{TARGET}_ca_i_6162_same_atomic_transitions.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(
        w, obs, obs_error, native_prediction, korg_prediction, window_metrics,
        f"IC2391-0096 Ca I 6155-6170 A: {line_identity['pz_metal_transition_count']} identical atomic transitions",
        figure_path,
    )


if __name__ == "__main__":
    main()
