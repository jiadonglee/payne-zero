#!/usr/bin/env python3
"""UVES-POP Payne-Zero/MARCS comparison in one TiO-rich GALAH window."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-mstar-eso-v2")

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_mstar_eso_highres_v1 as v1


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v1"
RESULT_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v2"
KORG_ROOT = Path("/Users/jdli/Project/jorg/Korg.jl-1.0.1")
GALAH_H5 = KORG_ROOT / "data" / "linelists" / "GALAH_DR3" / "galah_dr3_linelist.h5"
WINDOW = {"name": "tio_6650", "lo": 6650.0, "hi": 6670.0, "padded_lo": 6645.0, "padded_hi": 6675.0}


def decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_synthesis(target, atmosphere, line_mode, expected):
    path = RESULT_ROOT / "models" / "spectra" / (
        "%s_%s_%s_%s.h5" % (target, WINDOW["name"], atmosphere, line_mode)
    )
    with h5py.File(path, "r") as handle:
        wave = np.asarray(handle["wavelength_air_A"], dtype=float)
        flux = np.asarray(handle["normalized_flux"], dtype=float)
        attrs = {key: decode(value) for key, value in handle.attrs.items()}
    for label, key in (("teff_K", "teff_K"), ("logg", "logg"), ("metallicity", "metallicity"),
                       ("alpha", "alpha_enhancement"), ("vmic_km_s", "microturbulence_km_s")):
        if not np.isclose(float(attrs[key]), float(expected[label]), rtol=0.0, atol=1e-10):
            raise ValueError("label mismatch in %s: %s" % (path, label))
    if attrs["linelist"] != "Korg GALAH DR3 HDF5" or attrs["line_mode"] != line_mode:
        raise ValueError("line-list provenance mismatch in %s" % path)
    if attrs["wavelength_system"] != "air" or not np.isclose(attrs["resolution"], 80000.0):
        raise ValueError("wavelength/resolution mismatch in %s" % path)
    return wave, flux, attrs


def make_figure(target, config, data, path):
    colors = {"payne_zero": "#2878B5", "marcs": "#E07B39"}
    fig, axes = plt.subplots(3, 1, figsize=(12.0, 8.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.3, 1.15, 1.0]})
    wave = data["wavelength_air_rest_A"]
    obs = data["observed_normalized"]
    error = data["observed_error_normalized"]
    axes[0].fill_between(wave, obs - error, obs + error, color="0.75", alpha=0.25, linewidth=0)
    axes[0].plot(wave, obs, color="black", lw=0.8, label="UVES-POP")
    for atmosphere, label in (("payne_zero", "Payne-Zero"), ("marcs", "MARCS")):
        model = data[atmosphere]
        axes[0].plot(wave, model["full"]["prediction"], color=colors[atmosphere], lw=1.0,
                     label=label + " + full GALAH")
        axes[0].plot(wave, model["no_tio"]["prediction"], color=colors[atmosphere], lw=0.75,
                     ls="--", alpha=0.55, label=label + " no-TiO")
        axes[1].plot(wave, model["full"]["residual"], color=colors[atmosphere], lw=0.85,
                     label="obs − " + label)
        axes[2].plot(wave, model["sensitivity_full_minus_no_tio"], color=colors[atmosphere], lw=0.9,
                     label=label + " TiO: full−noTiO")
        axes[2].plot(wave, model["sensitivity_no_tio_minus_atomic"], color=colors[atmosphere], lw=0.75,
                     ls="--", alpha=0.65, label=label + " other molecules")
    axes[0].set_ylabel("locally normalized flux")
    axes[1].axhline(0.0, color="0.35", lw=0.6)
    axes[1].set_ylabel("obs − full model")
    axes[2].axhline(0.0, color="0.35", lw=0.6)
    axes[2].set_ylabel("line-list ablation")
    axes[2].set_xlabel("rest-frame air wavelength (Å)")
    for axis in axes:
        axis.legend(fontsize=8, loc="best", ncol=2 if axis is axes[0] else 1)
    delta = v1.label_delta(config)
    metrics = data["summary"]
    axes[1].text(
        0.01, 0.96,
        "PZ RMS %.5f; MARCS RMS %.5f; Δ(PZ−MARCS) %+.6f"
        % (metrics["payne_zero_full_rms"], metrics["marcs_full_rms"], metrics["delta_rms_pz_minus_marcs"]),
        transform=axes[1].transAxes, va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    axes[2].text(
        0.01, 0.96,
        "RMS(full−noTiO): PZ %.4f; MARCS %.4f"
        % (metrics["payne_zero_tio_sensitivity_rms"], metrics["marcs_tio_sensitivity_rms"]),
        transform=axes[2].transAxes, va="top", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )
    fig.suptitle(
        "%s (%s): 6650–6670 Å molecular/TiO-rich window\n"
        "node ΔTeff=%+.0f K, Δlogg=%+.3f, Δ[M/H]=%+.3f, Δα=%+.3f"
        % (target, config["role"], delta["teff_K"], delta["logg"], delta["metallicity"], delta["alpha"]),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    figures = RESULT_ROOT / "figures"
    processed = RESULT_ROOT / "processed"
    figures.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    rows = []
    result = {
        "comparison_scope": "one 6650-6670 A molecular/TiO-rich GALAH DR3 window",
        "primary_comparison": "full GALAH line list; no-TiO and atomic-only are sensitivity controls",
        "targets": {},
    }

    audit_attrs = None
    for target, config in v1.TARGETS.items():
        wave, flux, error, good, inventory = v1.load_observation(
            V1_ROOT / "observations" / config["fits"], config["rv_km_s"]
        )
        select = good & (wave >= WINDOW["lo"]) & (wave <= WINDOW["hi"])
        w = wave[select]
        f = flux[select]
        e = error[select]
        scale = float(np.percentile(f, 95.0))
        obs = f / scale
        obs_error = e / scale
        plot_data = {
            "wavelength_air_rest_A": w, "observed_normalized": obs,
            "observed_error_normalized": obs_error,
        }
        target_result = {
            "class": config["class"], "spectral_type": config["spectral_type"], "role": config["role"],
            "source_labels": config["source_labels"], "source_errors": config["source_errors"],
            "adopted_labels": config["adopted_labels"],
            "label_delta_adopted_minus_source": v1.label_delta(config),
            "n_pixels": int(w.size), "median_snr": float(np.median(f / e)),
            "normalization": "observed local 95th percentile",
            "continuum_nuisance": "model * (a + b*x), equal-weight least squares",
            "atmospheres": {},
        }
        npz = {"wavelength_air_rest_A": w, "observed_normalized": obs, "observed_error_normalized": obs_error}
        full_predictions = {}
        summary = {}
        for atmosphere in ("payne_zero", "marcs"):
            atmosphere_result = {"line_modes": {}}
            model_interpolated = {}
            plot_data[atmosphere] = {}
            for line_mode in ("full", "no_tio", "atomic_only"):
                model_wave, model_flux, attrs = load_synthesis(target, atmosphere, line_mode, config["adopted_labels"])
                model = np.interp(w, model_wave, model_flux)
                prediction, coefficients = v1.continuum_fit(obs, model, w)
                residual = obs - prediction
                metrics = v1.residual_metrics(residual)
                atmosphere_result["line_modes"][line_mode] = {
                    "metrics": metrics,
                    "continuum_coefficients_a_b": [float(x) for x in coefficients],
                    "synthesis_metadata": attrs,
                }
                rows.append({
                    "target": target, "class": config["class"], "role": config["role"],
                    "atmosphere": atmosphere, "line_mode": line_mode, "n_pixels": int(w.size),
                    "median_snr": float(np.median(f / e)), **metrics,
                })
                model_interpolated[line_mode] = model
                plot_data[atmosphere][line_mode] = {"prediction": prediction, "residual": residual, "metrics": metrics}
                npz[atmosphere + "_" + line_mode + "_prediction"] = prediction
                npz[atmosphere + "_" + line_mode + "_residual"] = residual
                if line_mode == "full":
                    full_predictions[atmosphere] = prediction
                    summary[atmosphere + "_full_rms"] = metrics["rms"]
                if audit_attrs is None:
                    audit_attrs = attrs
            sensitivity = model_interpolated["full"] - model_interpolated["atomic_only"]
            tio_sensitivity = model_interpolated["full"] - model_interpolated["no_tio"]
            other_molecular_sensitivity = model_interpolated["no_tio"] - model_interpolated["atomic_only"]
            sensitivity_metrics = {
                "rms_full_minus_atomic": float(np.sqrt(np.mean(sensitivity ** 2))),
                "median_full_minus_atomic": float(np.median(sensitivity)),
                "p95_absolute_full_minus_atomic": float(np.percentile(np.abs(sensitivity), 95.0)),
                "rms_full_minus_no_tio": float(np.sqrt(np.mean(tio_sensitivity ** 2))),
                "median_full_minus_no_tio": float(np.median(tio_sensitivity)),
                "p95_absolute_full_minus_no_tio": float(np.percentile(np.abs(tio_sensitivity), 95.0)),
                "rms_no_tio_minus_atomic": float(np.sqrt(np.mean(other_molecular_sensitivity ** 2))),
            }
            atmosphere_result["molecular_sensitivity"] = sensitivity_metrics
            target_result["atmospheres"][atmosphere] = atmosphere_result
            plot_data[atmosphere]["sensitivity_full_minus_atomic"] = sensitivity
            plot_data[atmosphere]["sensitivity_full_minus_no_tio"] = tio_sensitivity
            plot_data[atmosphere]["sensitivity_no_tio_minus_atomic"] = other_molecular_sensitivity
            npz[atmosphere + "_full_minus_atomic"] = sensitivity
            npz[atmosphere + "_full_minus_no_tio"] = tio_sensitivity
            npz[atmosphere + "_no_tio_minus_atomic"] = other_molecular_sensitivity
            summary[atmosphere + "_sensitivity_rms"] = sensitivity_metrics["rms_full_minus_atomic"]
            summary[atmosphere + "_tio_sensitivity_rms"] = sensitivity_metrics["rms_full_minus_no_tio"]
            summary[atmosphere + "_other_molecular_sensitivity_rms"] = sensitivity_metrics["rms_no_tio_minus_atomic"]
        summary["delta_rms_pz_minus_marcs"] = summary["payne_zero_full_rms"] - summary["marcs_full_rms"]
        summary["continuum_adjusted_full_model_rms_difference"] = float(
            np.sqrt(np.mean((full_predictions["payne_zero"] - full_predictions["marcs"]) ** 2))
        )
        target_result["summary"] = summary
        target_result["scientific_verdict"] = "cannot_determine"
        result["targets"][target] = target_result
        plot_data["summary"] = summary
        np.savez_compressed(processed / (target + "_tio_6650.npz"), **npz)
        make_figure(target, config, plot_data, figures / (target + "_tio_6650_observed_model_residuals.png"))

    fields = ["target", "class", "role", "atmosphere", "line_mode", "n_pixels", "median_snr",
              "rms", "mad", "p95_absolute", "median_signed"]
    with (RESULT_ROOT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RESULT_ROOT / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    audit = {
        "local_source": str(GALAH_H5),
        "is_real_hdf5": bool(h5py.is_hdf5(GALAH_H5)),
        "file_size_bytes": GALAH_H5.stat().st_size,
        "reader": "Korg.get_GALAH_DR3_linelist() in Korg 1.0.1",
        "species_name_note": "Korg canonicalizes TiO as OTi",
        "counts_from_successful_reader": {
            "total": int(audit_attrs["global_total_line_count"]),
            "atomic": int(audit_attrs["global_atomic_line_count"]),
            "molecular": int(audit_attrs["global_molecular_line_count"]),
            "tio": int(audit_attrs["global_tio_line_count"]),
        },
        "padded_synthesis_window_counts": {
            "range_air_A": [WINDOW["padded_lo"], WINDOW["padded_hi"]],
            "total": int(audit_attrs["padded_window_total_line_count"]),
            "molecular": int(audit_attrs["padded_window_molecular_line_count"]),
            "tio": int(audit_attrs["padded_window_tio_line_count"]),
        },
        "derived_no_tio_counts": {
            "total": int(audit_attrs["global_total_line_count"] - audit_attrs["global_tio_line_count"]),
            "molecular": int(audit_attrs["global_molecular_line_count"] - audit_attrs["global_tio_line_count"]),
            "padded_window_total": int(
                audit_attrs["padded_window_total_line_count"] - audit_attrs["padded_window_tio_line_count"]
            ),
            "padded_window_molecular": int(
                audit_attrs["padded_window_molecular_line_count"] - audit_attrs["padded_window_tio_line_count"]
            ),
        },
        "excluded_large_pointer_route": {
            "schwenke_bin": {"working_tree_bytes": 134, "lfs_declared_bytes": 603911984},
            "eschwenke_bin": {"working_tree_bytes": 133, "lfs_declared_bytes": 22621208},
            "reason": "Git LFS pointers only; not needed because the bundled GALAH HDF5 is executable now",
        },
        "validation_boundary": "Line-format/readability, no-TiO ablation, and measured spectral sensitivity verified; TiO completeness and oscillator-strength accuracy not audited",
    }
    (RESULT_ROOT / "linelist_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
