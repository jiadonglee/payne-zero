#!/usr/bin/env python3
"""Fit one shared RV offset and Korg rotational broadening per UVES-POP star."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-mstar-eso-v3")

import h5py
import matplotlib
import numpy as np
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_mstar_eso_highres_v1 as v1


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V1_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v1"
V2_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v2"
RESULT_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v3_shared_nuisance"
KORG_ROOT = Path("/Users/jdli/Project/jorg/Korg.jl-1.0.1")
C_KM_S = 299792.458
CA_REST_A = 6162.173
WINDOWS = {
    "ca_i_6162": {"label": "Ca I", "lo": 6155.0, "hi": 6170.0},
    "tio_6650": {"label": "TiO-rich", "lo": 6650.0, "hi": 6670.0},
}


def decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def doppler_factor(rv_km_s):
    beta = rv_km_s / C_KM_S
    return np.sqrt((1.0 + beta) / (1.0 - beta))


def synthesize_models():
    script = RESULT_ROOT / "models" / "korg_synthesize_unbroadened_grid.jl"
    manifest = V2_ROOT / "models" / "model_manifest.tsv"
    out = RESULT_ROOT / "models" / "spectra"
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["julia", "--project=" + str(KORG_ROOT), str(script), str(manifest), str(out)],
        check=True,
    )


def load_model(target, window, atmosphere):
    path = RESULT_ROOT / "models" / "spectra" / f"{target}_{window}_{atmosphere}.h5"
    with h5py.File(path, "r") as handle:
        wave = np.asarray(handle["wavelength_air_A"], dtype=float)
        raw = np.asarray(handle["raw_normalized_flux"], dtype=float)
        vsini = np.asarray(handle["vsini_grid_km_s"], dtype=float)
        grid = np.asarray(handle["normalized_flux_R80000"], dtype=float)
        attrs = {key: decode(value) for key, value in handle.attrs.items()}
    if grid.shape == (vsini.size, wave.size):
        grid = grid.T
    if grid.shape != (wave.size, vsini.size):
        raise ValueError(f"unexpected broadened-grid shape {grid.shape} in {path}")
    if attrs["raw_rotation_applied"] or attrs["raw_lsf_applied"]:
        raise ValueError(f"raw model was already broadened in {path}")
    if not np.isclose(float(attrs["instrument_resolution"]), 80000.0):
        raise ValueError(f"wrong instrumental resolution in {path}")
    if window == "tio_6650" and attrs["linelist"] != "Korg GALAH DR3 HDF5 full molecular":
        raise ValueError(f"TiO window does not use the full GALAH list in {path}")
    return {"path": str(path), "wave": wave, "raw": raw, "vsini": vsini, "grid": grid, "attrs": attrs}


def flux_at_vsini(model, vsini_km_s):
    grid_v = model["vsini"]
    value = float(np.clip(vsini_km_s, grid_v[0], grid_v[-1]))
    upper = int(np.searchsorted(grid_v, value, side="right"))
    if upper == 0:
        return model["grid"][:, 0]
    if upper == grid_v.size:
        return model["grid"][:, -1]
    lower = upper - 1
    fraction = (value - grid_v[lower]) / (grid_v[upper] - grid_v[lower])
    return (1.0 - fraction) * model["grid"][:, lower] + fraction * model["grid"][:, upper]


def predict(observed, wavelength, model, rv_offset_km_s, vsini_km_s):
    broadened = flux_at_vsini(model, vsini_km_s)
    shifted = np.interp(wavelength / doppler_factor(rv_offset_km_s), model["wave"], broadened)
    prediction, coefficients = v1.continuum_fit(observed, shifted, wavelength)
    return prediction, coefficients


def fit_shared_nuisance(observed, wavelength, models):
    def objective(parameters):
        rv_offset, vsini = parameters
        scores = []
        for model in models.values():
            prediction, _ = predict(observed, wavelength, model, rv_offset, vsini)
            scores.append(np.mean((observed - prediction) ** 2))
        return float(np.mean(scores))

    best = None
    for rv in np.arange(-8.0, 8.0001, 0.25):
        for vsini in np.arange(0.0, 40.0001, 0.5):
            score = objective((rv, vsini))
            if best is None or score < best[0]:
                best = (score, rv, vsini)
    fit = minimize(
        objective, np.array(best[1:]), method="Powell",
        bounds=((-8.0, 8.0), (0.0, 40.0)),
        options={"xtol": 1e-5, "ftol": 1e-12, "maxiter": 250},
    )
    return {
        "rv_offset_km_s": float(fit.x[0]),
        "vsini_km_s": float(fit.x[1]),
        "joint_equal_model_mse": float(fit.fun),
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "bounds": {"rv_offset_km_s": [-8.0, 8.0], "vsini_km_s": [0.0, 40.0]},
    }


def line_center_and_fwhm(wavelength, flux):
    region = (wavelength >= CA_REST_A - 0.8) & (wavelength <= CA_REST_A + 0.8)
    w = wavelength[region]
    f = flux[region]
    core = np.abs(w - CA_REST_A) <= 0.35
    center_index = np.flatnonzero(core)[np.argmin(f[core])]
    center = float(w[center_index])
    if 0 < center_index < w.size - 1:
        local_w = w[center_index - 1:center_index + 2]
        local_f = f[center_index - 1:center_index + 2]
        quadratic = np.polyfit(local_w - center, local_f, 2)
        if quadratic[0] > 0:
            center += float(-quadratic[1] / (2.0 * quadratic[0]))
    continuum = float(np.percentile(f, 95.0))
    half_level = 0.5 * (continuum + float(f[center_index]))
    left_candidates = np.flatnonzero((np.arange(w.size) < center_index) & (f >= half_level))
    right_candidates = np.flatnonzero((np.arange(w.size) > center_index) & (f >= half_level))
    if left_candidates.size == 0 or right_candidates.size == 0:
        return {"center_air_A": center, "fwhm_km_s": None, "half_depth_flux": half_level}
    li = int(left_candidates[-1])
    ri = int(right_candidates[0])
    left = float(np.interp(half_level, [f[li + 1], f[li]], [w[li + 1], w[li]]))
    right = float(np.interp(half_level, [f[ri - 1], f[ri]], [w[ri - 1], w[ri]]))
    return {
        "center_air_A": center,
        "center_velocity_relative_6162p173_km_s": float(C_KM_S * (center / CA_REST_A - 1.0)),
        "fwhm_km_s": float(C_KM_S * (right - left) / CA_REST_A),
        "half_depth_flux": half_level,
    }


def make_figure(target_result, processed, path):
    colors = {"payne_zero": "#2878B5", "marcs": "#E07B39"}
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.8), sharex="col", gridspec_kw={"height_ratios": [2.3, 1.0]})
    for column, (window_name, window) in enumerate(WINDOWS.items()):
        data = processed[window_name]
        wave = data["wavelength_air_catalogue_rest_A"]
        obs = data["observed_normalized"]
        error = data["observed_error_normalized"]
        axes[0, column].fill_between(wave, obs - error, obs + error, color="0.75", alpha=0.25, lw=0)
        axes[0, column].plot(wave, obs, color="black", lw=0.8, label="UVES-POP")
        for atmosphere, label in (("payne_zero", "Payne-Zero"), ("marcs", "MARCS")):
            axes[0, column].plot(wave, data[atmosphere]["prediction"], color=colors[atmosphere], lw=1.0, label=label)
            axes[1, column].plot(wave, data[atmosphere]["residual"], color=colors[atmosphere], lw=0.85, label="obs − " + label)
        axes[0, column].set_title(f"{window['label']}: {window['lo']:.0f}–{window['hi']:.0f} Å")
        axes[0, column].set_ylabel("locally normalized flux")
        axes[1, column].axhline(0.0, color="0.35", lw=0.6)
        axes[1, column].set_ylabel("obs − model")
        axes[1, column].set_xlabel("catalogue-rest air wavelength (Å)")
        axes[0, column].legend(fontsize=8, loc="best")
        axes[1, column].legend(fontsize=8, loc="lower right")
        metrics = target_result["windows"][window_name]
        axes[1, column].text(
            0.01, 0.96,
            "PZ RMS %.5f; MARCS RMS %.5f\nΔ(PZ−MARCS) %+.6f"
            % (metrics["payne_zero"]["rms"], metrics["marcs"]["rms"], metrics["delta_rms_pz_minus_marcs"]),
            transform=axes[1, column].transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    nuisance = target_result["shared_nuisance_from_ca_i"]
    fig.suptitle(
        "%s: shared Ca I nuisance, ΔRV=%+.3f km/s, v sin i=%.3f km/s, R=80,000"
        % (target_result["target"], nuisance["rv_offset_km_s"], nuisance["vsini_km_s"]),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesize", action="store_true", help="regenerate unbroadened models and Korg broadening grids")
    args = parser.parse_args()
    if args.synthesize:
        synthesize_models()

    (RESULT_ROOT / "processed").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    output = {
        "experiment": "shared RV offset and shared Korg rotational broadening fitted jointly to PZ and MARCS in Ca I",
        "instrument_resolution": 80000.0,
        "delta_rms_definition": "RMS(observation-PayneZero) minus RMS(observation-MARCS); negative favors PZ",
        "fit_window": "Ca I 6155-6170 A",
        "frozen_evaluation_windows": ["Ca I 6155-6170 A", "TiO-rich 6650-6670 A"],
        "targets": {},
    }
    csv_rows = []
    for target, config in v1.TARGETS.items():
        wave, flux, error, good, inventory = v1.load_observation(V1_ROOT / "observations" / config["fits"], config["rv_km_s"])
        models = {
            window_name: {atmosphere: load_model(target, window_name, atmosphere) for atmosphere in ("payne_zero", "marcs")}
            for window_name in WINDOWS
        }
        ca_select = good & (wave >= WINDOWS["ca_i_6162"]["lo"]) & (wave <= WINDOWS["ca_i_6162"]["hi"])
        ca_wave = wave[ca_select]
        ca_scale = float(np.percentile(flux[ca_select], 95.0))
        ca_obs = flux[ca_select] / ca_scale
        shared = fit_shared_nuisance(ca_obs, ca_wave, models["ca_i_6162"])
        target_result = {
            "target": target, "class": config["class"], "role": config["role"],
            "catalogue_rv_km_s": config["rv_km_s"],
            "catalogue_vsini_km_s": config["vsini_km_s"],
            "source_labels": config["source_labels"], "adopted_labels": config["adopted_labels"],
            "label_delta_adopted_minus_source": v1.label_delta(config),
            "shared_nuisance_from_ca_i": shared, "windows": {},
        }
        processed = {}
        for window_name, window in WINDOWS.items():
            select = good & (wave >= window["lo"]) & (wave <= window["hi"])
            w = wave[select]
            f = flux[select]
            e = error[select]
            scale = float(np.percentile(f, 95.0))
            obs = f / scale
            obs_error = e / scale
            window_result = {
                "n_pixels": int(w.size), "median_snr": float(np.median(f / e)),
                "normalization": "observed local 95th percentile",
                "continuum_nuisance": "each atmosphere gets model*(a+b*x), same rule and order",
            }
            processed_window = {
                "wavelength_air_catalogue_rest_A": w, "observed_normalized": obs,
                "observed_error_normalized": obs_error,
            }
            npz = {"wavelength_air_catalogue_rest_A": w, "observed_normalized": obs, "observed_error_normalized": obs_error}
            for atmosphere in ("payne_zero", "marcs"):
                prediction, coefficients = predict(obs, w, models[window_name][atmosphere], shared["rv_offset_km_s"], shared["vsini_km_s"])
                residual = obs - prediction
                metrics = v1.residual_metrics(residual)
                window_result[atmosphere] = {
                    **metrics, "continuum_coefficients_a_b": [float(x) for x in coefficients],
                    "model_path": models[window_name][atmosphere]["path"],
                }
                processed_window[atmosphere] = {"prediction": prediction, "residual": residual}
                npz[atmosphere + "_prediction"] = prediction
                npz[atmosphere + "_residual"] = residual
                csv_rows.append({
                    "target": target, "role": config["role"], "window": window_name,
                    "atmosphere": atmosphere, "rv_offset_km_s": shared["rv_offset_km_s"],
                    "vsini_km_s": shared["vsini_km_s"], **metrics,
                })
            window_result["delta_rms_pz_minus_marcs"] = window_result["payne_zero"]["rms"] - window_result["marcs"]["rms"]
            if window_name == "ca_i_6162":
                widths = {
                    "definition": "half depth around Ca I 6162.173 A using a 1.6 A local interval; center from a three-pixel parabola",
                    "observation": line_center_and_fwhm(w, obs),
                    "fixed_catalogue_nuisance_reference": {},
                }
                for atmosphere in ("payne_zero", "marcs"):
                    widths[atmosphere] = line_center_and_fwhm(w, processed_window[atmosphere]["prediction"])
                    widths[atmosphere]["center_offset_obs_minus_model_km_s"] = (
                        widths["observation"].get("center_velocity_relative_6162p173_km_s", np.nan)
                        - widths[atmosphere].get("center_velocity_relative_6162p173_km_s", np.nan)
                    )
                    fixed_prediction, _ = predict(obs, w, models[window_name][atmosphere], 0.0, config["vsini_km_s"])
                    widths["fixed_catalogue_nuisance_reference"][atmosphere] = line_center_and_fwhm(w, fixed_prediction)
                window_result["ca_i_6162_line_diagnostic"] = widths
            target_result["windows"][window_name] = window_result
            processed[window_name] = processed_window
            np.savez_compressed(RESULT_ROOT / "processed" / f"{target}_{window_name}.npz", **npz)
        output["targets"][target] = target_result
        if target == "IC2391-0096":
            make_figure(target_result, processed, RESULT_ROOT / "figures" / "IC2391-0096_shared_nuisance_observed_models_residuals.png")

    fields = ["target", "role", "window", "atmosphere", "rv_offset_km_s", "vsini_km_s", "rms", "mad", "p95_absolute", "median_signed"]
    with (RESULT_ROOT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    (RESULT_ROOT / "metrics.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
