#!/usr/bin/env python3
"""Compare native Payne-Zero synthesis with UVES-POP for IC2391-0096."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-mstar-eso-native-pz-v4")
os.environ.setdefault(
    "PAYNE_ZERO_SYNTHESIS_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / ".cache/payne-zero/synthesis"),
)

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from fitter import ObservedSpectrumOperator, RotationalBroadening
from payne_zero_synthesis import synthesize, validate_atmosphere_npz

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_mstar_eso_highres_v1 as v1


RESULT_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v4_native_paynezero"
V1_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v1"
V3_ROOT = PROJECT_ROOT / "results" / "m_star_eso_highres_comparison_v3_shared_nuisance"
ATMOSPHERE_PATH = (
    PROJECT_ROOT
    / "results/m_star_emulator_v1r2_marcs100/cases/dwarf/"
    "g+5.00_m+0.00_a+0.00_c+0.00_x1.00/t3750/products/primary/"
    "t03750.0_g+5.00_m+0.00_a+0.00_x1.00.npz"
)
CASE_PATH = ATMOSPHERE_PATH.parents[2] / "case.json"
TARGET = "IC2391-0096"
R_GRID = 600_000.0
INSTRUMENT_R = 80_000.0
VSINI_KM_S = 7.1124722692
RESIDUAL_RV_KM_S = -0.6918215629
PADDING_A = 3.0
WINDOWS = {
    "ca_i_6162": {"label": "Ca I", "lo": 6155.0, "hi": 6170.0},
    "tio_6650": {"label": "TiO-rich", "lo": 6650.0, "hi": 6670.0},
}


def vacuum_to_air_nm(wavelength_vacuum_nm: np.ndarray | float) -> np.ndarray:
    """Convert optical vacuum wavelengths to air using the native catalog law."""

    vacuum = np.asarray(wavelength_vacuum_nm, dtype=np.float64)
    wavenumber_cm = 1.0e7 / vacuum
    refractive_index = (
        1.0000834213
        + 2406030.0 / (1.30e10 - wavenumber_cm**2)
        + 15997.0 / (3.89e9 - wavenumber_cm**2)
    )
    return vacuum / refractive_index


def air_to_vacuum_nm(wavelength_air_nm: np.ndarray | float) -> np.ndarray:
    """Invert ``vacuum_to_air_nm`` to sub-microangstrom precision."""

    air = np.asarray(wavelength_air_nm, dtype=np.float64)
    vacuum = air * 1.00028
    for _ in range(5):
        vacuum *= air / vacuum_to_air_nm(vacuum)
    return vacuum


class RotationThenInstrument:
    """Apply the fixed Gray rotation before the one instrumental LSF."""

    name = "rotation_then_constant_R_instrument"

    def __init__(self, native_wavelength_nm: np.ndarray, observed_wavelength_nm: np.ndarray):
        self.output_wavelength_nm = np.asarray(observed_wavelength_nm, dtype=np.float64)
        self.rotation = RotationalBroadening(
            native_wavelength_nm,
            maximum_vsini_km_s=VSINI_KM_S,
            limb_darkening=0.6,
            device="cpu",
            dtype=torch.float64,
        )
        self.instrument = ObservedSpectrumOperator(
            native_wavelength_nm,
            self.output_wavelength_nm,
            resolving_power=INSTRUMENT_R,
            device="cpu",
            dtype=torch.float64,
        )
        self.instrument.set_parameters(
            radial_velocity_km_s=RESIDUAL_RV_KM_S,
            broadening_sigma_km_s=0.0,
        )
        self.last_seconds = 0.0

    def convolve_fluxes(self, total_flux: torch.Tensor, continuum_flux: torch.Tensor):
        total_rotated = self.rotation(total_flux, vsini_km_s=VSINI_KM_S)
        continuum_rotated = self.rotation(continuum_flux, vsini_km_s=VSINI_KM_S)
        result = self.instrument.convolve_fluxes(total_rotated, continuum_rotated)
        self.last_seconds = self.instrument.last_seconds
        return result

    def metadata(self) -> dict[str, object]:
        return {
            "order": "native total/continuum -> Gray rotation -> constant-R Gaussian LSF -> observed pixels",
            "rotation": self.rotation.metadata(),
            "instrument": self.instrument.metadata(),
        }


def _metrics(residual: np.ndarray) -> dict[str, float]:
    return v1.residual_metrics(np.asarray(residual, dtype=np.float64))


def _make_figure(processed: dict[str, dict[str, np.ndarray]], metrics: dict, path: Path) -> None:
    colors = {"native": "#2878B5", "korg": "#E07B39"}
    fig, axes = plt.subplots(
        2, 2, figsize=(12.0, 6.8), sharex="col",
        gridspec_kw={"height_ratios": [2.3, 1.0]},
    )
    for column, (window_name, window) in enumerate(WINDOWS.items()):
        data = processed[window_name]
        wave = data["wavelength_air_catalogue_rest_A"]
        obs = data["observed_normalized"]
        err = data["observed_error_normalized"]
        axes[0, column].fill_between(wave, obs - err, obs + err, color="0.75", alpha=0.25, lw=0)
        axes[0, column].plot(wave, obs, color="black", lw=0.75, label="UVES-POP")
        axes[0, column].plot(wave, data["native_prediction"], color=colors["native"], lw=0.9, label="native Payne-Zero")
        axes[0, column].plot(wave, data["korg_prediction"], color=colors["korg"], lw=0.9, label="Korg + PZ atmosphere")
        axes[1, column].plot(wave, data["native_residual"], color=colors["native"], lw=0.8, label="obs - native PZ")
        axes[1, column].plot(wave, data["korg_residual"], color=colors["korg"], lw=0.8, label="obs - Korg")
        axes[0, column].set_title(f"{window['label']}: {window['lo']:.0f}-{window['hi']:.0f} A")
        axes[0, column].set_ylabel("locally normalized flux")
        axes[1, column].set_ylabel("obs - model")
        axes[1, column].set_xlabel("catalogue-rest air wavelength (A)")
        axes[1, column].axhline(0.0, color="0.35", lw=0.6)
        axes[0, column].legend(fontsize=8, loc="best")
        axes[1, column].legend(fontsize=8, loc="lower right")
        wm = metrics["windows"][window_name]
        axes[1, column].text(
            0.01, 0.96,
            f"native RMS {wm['native_payne_zero']['rms']:.5f}; Korg RMS {wm['korg_v3']['rms']:.5f}\n"
            f"native-Korg {wm['native_vs_korg_prediction_rms']:.5f}",
            transform=axes[1, column].transAxes, va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    fig.suptitle(
        f"{TARGET}: fixed dRV={RESIDUAL_RV_KM_S:+.3f} km/s, v sin i={VSINI_KM_S:.3f} km/s, "
        f"R_grid={metrics['synthesis']['r_grid']:,.0f}, instrument R={INSTRUMENT_R:,.0f}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r-grid", type=float, default=R_GRID)
    args = parser.parse_args()
    r_grid = float(args.r_grid)

    schema_names = validate_atmosphere_npz(ATMOSPHERE_PATH)
    case = json.loads(CASE_PATH.read_text())
    if case.get("status") != "training_eligible" or not case.get("training_eligible"):
        raise RuntimeError("selected atmosphere case is not training_eligible")
    if not case["primary"].get("solver_converged"):
        raise RuntimeError("selected primary atmosphere did not converge")

    config = v1.TARGETS[TARGET]
    wave, flux, error, good, inventory = v1.load_observation(
        V1_ROOT / "observations" / config["fits"], config["rv_km_s"]
    )
    (RESULT_ROOT / "spectra").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "processed").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "figures").mkdir(parents=True, exist_ok=True)

    output = {
        "experiment": "native Payne-Zero forward synthesis on the converged PZ atmosphere",
        "target": TARGET,
        "atmosphere": {
            "path": str(ATMOSPHERE_PATH.resolve()),
            "case_path": str(CASE_PATH.resolve()),
            "case_status": case["status"],
            "training_eligible": bool(case["training_eligible"]),
            "primary_solver_converged": bool(case["primary"]["solver_converged"]),
            "schema_field_count": len(schema_names),
        },
        "observation": inventory,
        "fixed_nuisance": {
            "catalogue_rv_km_s_applied_by_v1": config["rv_km_s"],
            "residual_rv_km_s": RESIDUAL_RV_KM_S,
            "vsini_km_s": VSINI_KM_S,
            "limb_darkening": 0.6,
            "instrument_resolving_power": INSTRUMENT_R,
            "fit_status": "frozen from v3; not refitted",
        },
        "synthesis": {
            "backend": "payne_zero_synthesis.synthesize",
            "molecular_lines": True,
            "r_grid": r_grid,
            "r_grid_meaning": "intrinsic logarithmic wavelength sampling, not instrumental resolving power",
            "padding_A": PADDING_A,
            "device": "cpu",
            "dtype": "float64",
            "atomic_source": str((PROJECT_ROOT / "source_data_files/source_catalogs/lines/atomic_source_lines_parsed.npz").resolve()),
            "molecular_source": str((PROJECT_ROOT / "source_data_files/source_catalogs/molecules/molecular_band_lines.npz").resolve()),
            "molecular_manifest": str((PROJECT_ROOT / "source_data_files/source_catalogs/molecules/manifest.json").resolve()),
            "tio_source": str((PROJECT_ROOT / "source_data_files/source_catalogs/molecules/titanium_oxide_lines.npy").resolve()),
            "tio_provenance": "tio/schwenke.bin (Schwenke TiO), per molecular manifest",
            "native_wavelength_system": "vacuum",
            "observation_wavelength_system": "air",
            "wavelength_conversion": "air observation pixels converted to vacuum before native projection; air retained for plotting",
        },
        "comparison_limit": (
            "This compares two complete forward pipelines, not radiation-transfer code alone: native Payne-Zero uses its "
            "native atomic/molecular catalogs; Korg v3 used the GALAH molecular list for TiO and an atomic-only list for Ca I."
        ),
        "solver_campaign_success_rate_used_in_spectral_score": False,
        "windows": {},
    }
    processed: dict[str, dict[str, np.ndarray]] = {}

    for window_name, window in WINDOWS.items():
        selected = good & (wave >= window["lo"]) & (wave <= window["hi"])
        w = wave[selected]
        f = flux[selected]
        e = error[selected]
        scale = float(np.percentile(f, 95.0))
        obs = f / scale
        obs_error = e / scale

        output_wavelength_vacuum_nm = air_to_vacuum_nm(w / 10.0)
        start_nm = float(air_to_vacuum_nm((window["lo"] - PADDING_A) / 10.0))
        end_nm = float(air_to_vacuum_nm((window["hi"] + PADDING_A) / 10.0))
        native = synthesize(
            ATMOSPHERE_PATH,
            wavelength_start_nm=start_nm,
            wavelength_end_nm=end_nm,
            resolution=r_grid,
            molecular_lines=True,
            device="cpu",
            dtype="float64",
        )
        operator = RotationThenInstrument(native.wavelength_nm, output_wavelength_vacuum_nm)
        total = torch.as_tensor(native.flux_total, dtype=torch.float64)
        continuum = torch.as_tensor(native.flux_continuum, dtype=torch.float64)
        projected_total, projected_continuum, projected_normalized = operator.convolve_fluxes(total, continuum)
        raw_native = projected_normalized.detach().cpu().numpy()
        native_prediction, native_coefficients = v1.continuum_fit(obs, raw_native, w)
        native_residual = obs - native_prediction

        v3_path = V3_ROOT / "processed" / f"{TARGET}_{window_name}.npz"
        with np.load(v3_path, allow_pickle=False) as v3:
            if not np.allclose(w, v3["wavelength_air_catalogue_rest_A"], rtol=0.0, atol=1e-10):
                raise ValueError(f"v3 observed pixels differ for {window_name}")
            korg_prediction = np.asarray(v3["payne_zero_prediction"], dtype=np.float64)
        korg_residual = obs - korg_prediction

        spectrum_path = RESULT_ROOT / "spectra" / f"{TARGET}_{window_name}_native_r{int(r_grid)}.npz"
        native.save_npz(spectrum_path)
        processed_path = RESULT_ROOT / "processed" / f"{TARGET}_{window_name}.npz"
        np.savez_compressed(
            processed_path,
            wavelength_air_catalogue_rest_A=w,
            wavelength_vacuum_catalogue_rest_nm=output_wavelength_vacuum_nm,
            observed_normalized=obs,
            observed_error_normalized=obs_error,
            native_projected_precontinuum=raw_native,
            native_prediction=native_prediction,
            native_residual=native_residual,
            korg_prediction=korg_prediction,
            korg_residual=korg_residual,
        )
        output["windows"][window_name] = {
            "label": window["label"],
            "wavelength_air_A": [window["lo"], window["hi"]],
            "n_pixels": int(w.size),
            "normalization": "observation divided by local 95th percentile",
            "continuum_nuisance": "each model multiplied by independently fitted (a+b*x)",
            "noise_rms": float(np.sqrt(np.mean(obs_error ** 2))),
            "native_payne_zero": {
                **_metrics(native_residual),
                "continuum_coefficients_a_b": [float(value) for value in native_coefficients],
                "synthesis_seconds": float(native.seconds),
                "native_pixels": int(native.wavelength_nm.size),
                "spectrum_path": str(spectrum_path.resolve()),
                "processed_path": str(processed_path.resolve()),
                "operator": operator.metadata(),
            },
            "korg_v3": {
                **_metrics(korg_residual),
                "processed_source": str(v3_path.resolve()),
                "atmosphere": "same Payne-Zero converged atmosphere",
            },
            "native_vs_korg_prediction_rms": float(np.sqrt(np.mean((native_prediction - korg_prediction) ** 2))),
            "residual_correlation_native_vs_korg": float(np.corrcoef(native_residual, korg_residual)[0, 1]),
        }
        processed[window_name] = {
            "wavelength_air_catalogue_rest_A": w,
            "observed_normalized": obs,
            "observed_error_normalized": obs_error,
            "native_prediction": native_prediction,
            "native_residual": native_residual,
            "korg_prediction": korg_prediction,
            "korg_residual": korg_residual,
        }

    metrics_path = RESULT_ROOT / "metrics.json"
    metrics_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    _make_figure(processed, output, RESULT_ROOT / "figures" / f"{TARGET}_native_paynezero_vs_korg.png")


if __name__ == "__main__":
    main()
