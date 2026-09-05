#!/usr/bin/env python3
"""Minimal UVES-POP comparison of Payne-Zero and MARCS M-star spectra.

The comparison is deliberately limited to two atomic-dominated windows because
the shared Korg line list used here contains no TiO.  Both atmosphere families
use the same labels, abundances, line list, synthesis backend, resolution,
rotation, fixed catalogue RV, observed pixels, and continuum nuisance model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v1"
CAMPAIGN_ROOT = PROJECT_ROOT / "results" / "m_star_emulator_v1r2_marcs100"
C_KM_S = 299792.458

WINDOWS = {
    "mg_b": {"label": "Mg b", "lo": 5160.0, "hi": 5190.0},
    "ca_i_6162": {"label": "Ca I 6162", "lo": 6155.0, "hi": 6170.0},
}

# These masks do not intersect the selected windows, but are applied so that a
# changed window cannot silently admit known UVES-POP gaps or strong tellurics.
ESO_BAD_RANGES_A = (
    (3245.700, 3250.620), (4901.813, 4923.912), (4932.325, 4955.838),
    (6055.687, 6075.892), (6374.181, 6402.670), (6729.249, 6749.815),
    (6758.079, 6779.335), (6817.200, 6861.866), (6921.556, 6938.529),
    (6978.586, 7017.304), (9076.721, 9118.055),
)
TELLURIC_RANGES_A = (
    (6270.0, 6330.0), (6860.0, 6960.0), (7160.0, 7340.0),
    (7580.0, 7700.0), (8100.0, 8400.0), (8900.0, 9900.0),
)

TARGETS = {
    "IC2391-0096": {
        "class": "M dwarf",
        "spectral_type": "M0-5? (ESO UVES-POP table)",
        "role": "primary_nearest_node_diagnostic",
        "rv_km_s": 12.60,
        "rv_error_km_s": 0.38,
        "vsini_km_s": 21.07,
        "source_labels": {"teff_K": 3728.0, "logg": 4.9498, "metallicity": 0.0391, "alpha": 0.0995},
        "source_errors": {"teff_K": 37.0, "logg": 0.030, "metallicity": 0.030, "alpha": 0.030},
        "adopted_labels": {"teff_K": 3750.0, "logg": 5.0, "metallicity": 0.0, "alpha": 0.0, "carbon": 0.0, "vmic_km_s": 1.0},
        "case": "cases/dwarf/g+5.00_m+0.00_a+0.00_c+0.00_x1.00/t3750/case.json",
        "fits": "IC2391-0096_R80k.fits",
        "product_url": "https://data.voxastro.org/uves-pop/model_spec/merged_221115/IC2391-0096_R80k.fits.gz",
    },
    "HD219215": {
        "class": "M giant",
        "spectral_type": "M1.5III",
        "role": "label_mismatch_diagnostic",
        "rv_km_s": 1.90,
        "rv_error_km_s": 1.02,
        "vsini_km_s": 10.40,
        "source_labels": {"teff_K": 3871.0, "logg": 0.826, "metallicity": -0.097, "alpha": 0.529},
        "source_errors": {"teff_K": 39.0, "logg": 0.176, "metallicity": 0.110, "alpha": 0.084},
        "adopted_labels": {"teff_K": 3900.0, "logg": 1.5, "metallicity": 0.0, "alpha": 0.0, "carbon": 0.0, "vmic_km_s": 2.0},
        "case": "cases/giant/g+1.50_m+0.00_a+0.00_c+0.00_x2.00/t3900/case.json",
        "fits": "HD219215_R80k.fits",
        "product_url": "https://data.voxastro.org/uves-pop/model_spec/merged_221115/HD219215_R80k.fits.gz",
    },
}

CANDIDATES = (
    ("IC2391-0096", "M dwarf", "selected", "nearest eligible dwarf node"),
    ("HD219215", "M1.5III", "selected_diagnostic", "closest non-variable M giant, but large alpha and logg mismatch"),
    ("HD119149", "M1+III", "not_selected", "larger eligible-node distance"),
    ("HD120052", "M2III", "not_selected", "larger eligible-node distance"),
    ("HD102212", "M1III", "rejected", "known SRB variable and large alpha mismatch"),
    ("HD092305", "K4III", "rejected", "not M giant in the uniform Borisov table"),
    ("HD156274", "G9V", "rejected", "not M dwarf in the uniform Borisov table"),
)


def _decode(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _doppler_factor(rv_km_s: float) -> float:
    beta = rv_km_s / C_KM_S
    return float(np.sqrt((1.0 + beta) / (1.0 - beta)))


def _outside_ranges(wavelength: np.ndarray, ranges: Iterable[Tuple[float, float]]) -> np.ndarray:
    keep = np.ones(wavelength.shape, dtype=bool)
    for lo, hi in ranges:
        keep &= ~((wavelength >= lo) & (wavelength <= hi))
    return keep


def _flatten_column(table, name: str) -> np.ndarray:
    return np.asarray(table[name]).reshape(-1)


def load_observation(path: Path, rv_km_s: float):
    with fits.open(path, memmap=True) as hdul:
        header = hdul[0].header.copy()
        table_hdu = hdul["SPECTRUM"]
        table = table_hdu.data
        wave_observed = _flatten_column(table, "WAVE").astype(float)
        flux = _flatten_column(table, "FLUX").astype(float)
        error = _flatten_column(table, "ERROR").astype(float)
        pixmask = _flatten_column(table, "PIXMASK")
        quality = _flatten_column(table, "QUALITY")
        column_units = {name: table_hdu.columns[name].unit for name in ("WAVE", "FLUX", "ERROR", "PIXMASK", "QUALITY")}

    wave_rest = wave_observed / _doppler_factor(rv_km_s)
    good = (
        np.isfinite(wave_rest) & np.isfinite(flux) & np.isfinite(error)
        & (flux > 0.0) & (error > 0.0) & (pixmask == 1) & (quality == 0)
    )
    good &= _outside_ranges(wave_rest, ESO_BAD_RANGES_A)
    good &= _outside_ranges(wave_rest, TELLURIC_RANGES_A)
    inventory = {
        "path": str(path),
        "object": header.get("OBJECT"),
        "ra_deg": float(header.get("RA")),
        "dec_deg": float(header.get("DEC")),
        "date_obs": header.get("DATE-OBS"),
        "program_id": header.get("ESO OBS PROG ID"),
        "observation_block_id": int(header.get("ESO OBS ID")),
        "instrument": "VLT/UVES",
        "spectral_resolution": float(header.get("SPEC_RP")),
        "wavelength_axis_header": header.get("CTYPE1"),
        "wavelength_system": "air (CTYPE1=AWAV)",
        "primary_flux_unit": header.get("BUNIT"),
        "flux_calibration": header.get("FLUX_CAL"),
        "table_column_units": column_units,
        "error_column_present": True,
        "pixel_mask_used": "PIXMASK == 1 and QUALITY == 0",
        "wavelength_min_A": float(np.nanmin(wave_observed)),
        "wavelength_max_A": float(np.nanmax(wave_observed)),
        "n_pixels": int(wave_observed.size),
        "n_good_pixels_full_spectrum": int(good.sum()),
        "catalogue_rv_km_s_applied_as_common_fixed_shift": rv_km_s,
    }
    return wave_rest, flux, error, good, inventory


def load_model(path: Path, expected: Mapping[str, float]):
    with h5py.File(path, "r") as handle:
        wave = np.asarray(handle["wavelength_air_A"], dtype=float)
        flux = np.asarray(handle["normalized_flux"], dtype=float)
        attrs = {key: _decode(value) for key, value in handle.attrs.items()}
    checks = {
        "teff_K": "teff_K", "logg": "logg", "metallicity": "metallicity",
        "alpha": "alpha_enhancement", "vmic_km_s": "microturbulence_km_s",
    }
    for label, attr in checks.items():
        if not np.isclose(float(attrs[attr]), float(expected[label]), rtol=0.0, atol=1e-10):
            raise ValueError("model metadata mismatch for %s: %s" % (path, label))
    if attrs["wavelength_system"] != "air" or not np.isclose(float(attrs["resolution"]), 80000.0):
        raise ValueError("model wavelength system or resolution mismatch for %s" % path)
    if int(attrs["molecular_line_count"]) != 0:
        raise ValueError("this diagnostic requires the frozen atomic-only synthesis")
    return wave, flux, attrs


def continuum_fit(observed: np.ndarray, model: np.ndarray, wavelength: np.ndarray):
    x = 2.0 * (wavelength - wavelength.min()) / (wavelength.max() - wavelength.min()) - 1.0
    design = np.column_stack((model, model * x))
    coefficients, _, _, _ = np.linalg.lstsq(design, observed, rcond=None)
    prediction = design @ coefficients
    return prediction, coefficients


def residual_metrics(residual: np.ndarray) -> Dict[str, float]:
    absolute = np.abs(residual)
    median = np.median(residual)
    return {
        "rms": float(np.sqrt(np.mean(residual ** 2))),
        "mad": float(np.median(np.abs(residual - median))),
        "p95_absolute": float(np.percentile(absolute, 95.0)),
        "median_signed": float(median),
    }


def physical_diagnostics(case_path: Path) -> Dict[str, object]:
    case = json.loads(case_path.read_text())
    final = case["primary"]["solver_diagnostics"]["final_diagnostics"]
    return {
        "case_path": str(case_path),
        "status": case["status"],
        "training_eligible": bool(case["training_eligible"]),
        "primary_flux_error_percent": {
            "median": float(final["median_absolute_flux_error_percent"]),
            "p95": float(final["p95_absolute_flux_error_percent"]),
            "maximum": float(final["maximum_absolute_flux_error_percent"]),
        },
        "primary_path_consistency": case["path_consistency"],
    }


def label_delta(target: Mapping[str, object]) -> Dict[str, float]:
    source = target["source_labels"]
    adopted = target["adopted_labels"]
    return {
        "teff_K": float(adopted["teff_K"] - source["teff_K"]),
        "logg": float(adopted["logg"] - source["logg"]),
        "metallicity": float(adopted["metallicity"] - source["metallicity"]),
        "alpha": float(adopted["alpha"] - source["alpha"]),
    }


def normalized_label_distance(delta: Mapping[str, float]) -> float:
    return float(np.sqrt((delta["teff_K"] / 100.0) ** 2 + (delta["logg"] / 0.5) ** 2
                         + (delta["metallicity"] / 0.5) ** 2 + (delta["alpha"] / 0.2) ** 2))


def write_candidates(path: Path):
    fields = ["target", "spectral_type", "decision", "reason", "normalized_distance_to_nearest_eligible_node"]
    known_distances = {
        "IC2391-0096": 0.5586605856868732, "HD219215": 2.989124453748957,
        "HD119149": 4.201555545271297, "HD120052": 5.382406153385306,
        "HD102212": 4.625607095290303, "HD092305": 4.977978003165543,
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target, spectral_type, decision, reason in CANDIDATES:
            writer.writerow({
                "target": target, "spectral_type": spectral_type, "decision": decision,
                "reason": reason, "normalized_distance_to_nearest_eligible_node": known_distances.get(target, ""),
            })


def make_figure(target_name: str, target: Mapping[str, object], processed: Mapping[str, object], path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.8), sharex="col", gridspec_kw={"height_ratios": [2.25, 1.0]})
    colors = {"payne_zero": "#2878B5", "marcs": "#E07B39"}
    for column, (window_name, window) in enumerate(WINDOWS.items()):
        data = processed[window_name]
        wave = data["wavelength_air_rest_A"]
        obs = data["observed_normalized"]
        err = data["observed_error_normalized"]
        upper, lower = axes[0, column], axes[1, column]
        upper.fill_between(wave, obs - err, obs + err, color="0.75", alpha=0.25, linewidth=0)
        upper.plot(wave, obs, color="black", lw=0.75, label="UVES-POP")
        for model_name, label in (("payne_zero", "Payne-Zero atmosphere"), ("marcs", "MARCS atmosphere")):
            upper.plot(wave, data[model_name]["prediction"], color=colors[model_name], lw=0.9, label=label)
            lower.plot(wave, data[model_name]["residual"], color=colors[model_name], lw=0.75,
                       label="obs - %s" % ("PZ" if model_name == "payne_zero" else "MARCS"))
        upper.set_title("%s: %.0f–%.0f Å" % (window["label"], window["lo"], window["hi"]))
        upper.set_ylabel("locally normalized flux")
        lower.axhline(0.0, color="0.35", lw=0.6)
        lower.set_ylabel("obs − model")
        lower.set_xlabel("rest-frame air wavelength (Å)")
        lim = max(0.05, float(np.percentile(np.abs(np.r_[data["payne_zero"]["residual"], data["marcs"]["residual"]]), 99.0)))
        lower.set_ylim(-1.05 * lim, 1.05 * lim)
        upper.legend(fontsize=8, loc="lower right")
        lower.legend(fontsize=8, loc="lower right")
        text_lines = []
        for model_name, short in (("payne_zero", "PZ"), ("marcs", "MARCS")):
            text_lines.append("%s RMS %.4f; MAD %.4f" % (short, data[model_name]["metrics"]["rms"], data[model_name]["metrics"]["mad"]))
        lower.text(0.012, 0.96, "\n".join(text_lines), transform=lower.transAxes, va="top", fontsize=8,
                   bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78})
    delta = label_delta(target)
    fig.suptitle(
        "%s (%s; %s)\nnode ΔTeff=%+.0f K, Δlogg=%+.3f, Δ[M/H]=%+.3f, Δα=%+.3f"
        % (target_name, target["spectral_type"], target["role"], delta["teff_K"], delta["logg"], delta["metallicity"], delta["alpha"]),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    figures_dir = RESULT_ROOT / "figures"
    processed_dir = RESULT_ROOT / "processed"
    figures_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    inventories = {}
    output = {"comparison_scope": "two atomic-dominated windows; no TiO; no broad-band ranking", "targets": {}}

    for target_name, target in TARGETS.items():
        wave, flux, error, good, inventory = load_observation(RESULT_ROOT / "observations" / target["fits"], target["rv_km_s"])
        inventories[target_name] = inventory
        target_output = {
            "class": target["class"], "spectral_type": target["spectral_type"], "role": target["role"],
            "source_labels": target["source_labels"], "source_errors": target["source_errors"],
            "adopted_labels": target["adopted_labels"], "label_delta_adopted_minus_source": label_delta(target),
            "normalized_label_distance": normalized_label_distance(label_delta(target)),
            "rv_km_s": target["rv_km_s"], "rv_error_km_s": target["rv_error_km_s"],
            "vsini_km_s": target["vsini_km_s"], "product_url": target["product_url"],
            "payne_zero_physical_diagnostics": physical_diagnostics(CAMPAIGN_ROOT / target["case"]),
            "windows": {},
        }
        plot_data = {}
        for window_name, window in WINDOWS.items():
            selected = good & (wave >= window["lo"]) & (wave <= window["hi"])
            w = wave[selected]
            f = flux[selected]
            e = error[selected]
            if w.size < 100:
                raise RuntimeError("too few good pixels for %s %s" % (target_name, window_name))
            scale = float(np.percentile(f, 95.0))
            obs = f / scale
            obs_err = e / scale
            window_result = {
                "n_pixels": int(w.size), "normalization": "observed local 95th percentile",
                "continuum_nuisance": "model * (a + b*x), equal-weight least squares",
                "median_snr": float(np.median(f / e)), "models": {},
            }
            npz_data = {"wavelength_air_rest_A": w, "observed_normalized": obs, "observed_error_normalized": obs_err}
            plot_window = {"wavelength_air_rest_A": w, "observed_normalized": obs, "observed_error_normalized": obs_err}
            predictions = {}
            for model_name in ("payne_zero", "marcs"):
                model_path = RESULT_ROOT / "models" / "spectra" / (target_name + "_" + window_name + "_" + model_name + ".h5")
                model_wave, model_flux, attrs = load_model(model_path, target["adopted_labels"])
                interpolated = np.interp(w, model_wave, model_flux)
                prediction, coefficients = continuum_fit(obs, interpolated, w)
                residual = obs - prediction
                metrics = residual_metrics(residual)
                predictions[model_name] = prediction
                window_result["models"][model_name] = {
                    "metrics": metrics, "continuum_coefficients_a_b": [float(value) for value in coefficients],
                    "synthesis_metadata": attrs,
                }
                row = {"target": target_name, "class": target["class"], "role": target["role"], "window": window_name,
                       "model": model_name, "n_pixels": int(w.size), "median_snr": window_result["median_snr"], **metrics}
                rows.append(row)
                npz_data[model_name + "_prediction"] = prediction
                npz_data[model_name + "_residual"] = residual
                plot_window[model_name] = {"prediction": prediction, "residual": residual, "metrics": metrics}
            separation = predictions["payne_zero"] - predictions["marcs"]
            window_result["continuum_adjusted_model_rms_difference"] = float(np.sqrt(np.mean(separation ** 2)))
            window_result["delta_rms_payne_zero_minus_marcs"] = float(
                window_result["models"]["payne_zero"]["metrics"]["rms"] - window_result["models"]["marcs"]["metrics"]["rms"]
            )
            target_output["windows"][window_name] = window_result
            plot_data[window_name] = plot_window
            np.savez_compressed(processed_dir / (target_name + "_" + window_name + ".npz"), **npz_data)
        target_output["scientific_verdict"] = "cannot_determine"
        output["targets"][target_name] = target_output
        make_figure(target_name, target, plot_data, figures_dir / (target_name + "_observed_model_residuals.png"))

    fields = ["target", "class", "role", "window", "model", "n_pixels", "median_snr", "rms", "mad", "p95_absolute", "median_signed"]
    with (RESULT_ROOT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (RESULT_ROOT / "metrics.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    (RESULT_ROOT / "fits_inventory.json").write_text(json.dumps(inventories, indent=2, sort_keys=True) + "\n")
    write_candidates(RESULT_ROOT / "target_candidates.csv")

    source_manifest = {
        "paper_lookup": {
            "database": "arXiv export API",
            "endpoint": "https://export.arxiv.org/api/query?search_query=all:%22UVES-POP%22&start=0&max_results=20&sortBy=relevance&sortOrder=descending",
            "raw_response": str(RESULT_ROOT / "sources" / "arxiv_uves_pop_atom.xml"),
        },
        "papers": [
            {"title": "The recalibration of the UVES-POP stellar spectral library", "arxiv": "1802.03570", "url": "https://arxiv.org/abs/1802.03570"},
            {"title": "New Generation Stellar Spectral Libraries in the Optical and Near-Infrared I", "arxiv": "2211.09130", "doi": "10.3847/1538-4365/acc321", "url": "https://arxiv.org/abs/2211.09130"},
        ],
        "catalogue": {"name": "Borisov et al. 2023 Table 6", "cds_id": "J/ApJS/266/11", "local_table": str(RESULT_ROOT / "sources" / "borisov2023_table6.dat")},
        "eso_program": "266.D-5655(A)",
        "official_eso_pages_saved": [
            str(RESULT_ROOT / "sources" / "eso_uvespop_field_stars.html"),
            str(RESULT_ROOT / "sources" / "eso_uvespop_ic2391.html"),
            str(RESULT_ROOT / "sources" / "eso_uvespop_legend.html"),
            str(RESULT_ROOT / "sources" / "eso_uves_flag.txt"),
        ],
        "selected_products": {name: target["product_url"] for name, target in TARGETS.items()},
    }
    (RESULT_ROOT / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
