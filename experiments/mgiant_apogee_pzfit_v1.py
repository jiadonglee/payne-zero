#!/usr/bin/env python3
"""Payne Zero's M-giant fitter on DR20 APOGEE M giants (H band).

The fit route is the one used on the ESO UVES M giants
(``experiments/mgiant_eso_bestfit_v1.py pz-fit``): labels -> v4 M-giant (m,T)
three-seed median -> physical reconstruction -> native Payne Zero synthesis
-> ``ObservedSpectrumOperator`` -> ``fitter.fit_normalized_spectrum``, then
optionally ``refine_with_physical_atmosphere`` with the unchanged solver.
The production ``fitter.apogee`` path is not used: its label initializers are
trained above 4000 K.

APOGEE-specific pieces:

- Data: the eight ``mgiant_aspcap_fit_v1`` stars, mwmStar-0.6.0 coadds
  (vacuum wavelengths), flux divided by the file continuum.
- Synthesis 1510-1700 nm at R_grid=120,000; Gaussian LSF R=22,500; extra
  Gaussian broadening fitted (start 3.5 km/s, the v1 macroturbulence).
- Normalization as in v1: the same 339-pixel running median on data and on
  every projected model, then a per-chip linear multiplicative continuum
  profiled by the fitter. Relative to v1 only the node grid is replaced by
  the continuous v4 fast model.
- Start at the ASPCAP labels clipped into the v4 box; RV start from a coarse
  scan of the start spectrum, bounded to +-5 km/s around it.

``--atomic-calibration PATH`` runs a second arm: the same fit with a schema-4
atomic overlay (``fitter.apogee.calibrated_window_invariants``) applied to the
metal-line tensors of the H-band window. Every other input is unchanged; its
results go to ``stars_<calibration stem>/``.

Stages::

    fast     --star SDSS_ID   continuous fast fit (``--physical`` adds refinement)
    collect  [--stars-dir D]  per-star table in ``summary.json`` and a label figure
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments import mgiant_eso_bestfit_v1 as eso  # noqa: E402

RESULT_ROOT = PROJECT_ROOT / "results" / "mgiant_apogee_pzfit_v1"
V1_REPORT = PROJECT_ROOT / "results" / "mgiant_aspcap_fit_v1" / "fit_report.json"
SPECTRA_DIR = PROJECT_ROOT / "data" / "dr20" / "spectra" / "star"

SYNTH_VAC_NM = (1510.0, 1700.0)
R_GRID = 120_000.0
INSTRUMENT_R = 22_500.0
SIGMA_START_KM_S = 3.5
PSEUDO_CONTINUUM_PIXELS = 339
CHIPS_NM = ((1510.0, 1581.0), (1589.0, 1635.0), (1648.0, 1695.0))
C_KM_S = 299_792.458
NAMES = eso.PZ_FIT_NAMES
BOX = eso.V4_BOX


def pseudo_continuum_normalize(flux: np.ndarray) -> np.ndarray:
    from scipy.ndimage import median_filter

    return flux / median_filter(flux, size=PSEUDO_CONTINUUM_PIXELS, mode="nearest")


class ApogeeObservation:
    """One mwmStar coadd on its native pixel grid, pseudo-continuum normalized."""

    def __init__(self, sdss_id: int):
        from astropy.io import fits

        path = SPECTRA_DIR / f"{(sdss_id // 100) % 100:02d}" / f"{sdss_id % 100:02d}" / f"mwmStar-0.6.0-{sdss_id}.fits"
        with fits.open(path) as hdu_list:
            hdu = hdu_list[3] if hdu_list[3].data.size else hdu_list[4]
            row = hdu.data[0]
            wave_A = np.asarray(row["wavelength"], np.float64)
            continuum = np.asarray(row["continuum"], np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                flux = np.asarray(row["flux"], np.float64) / continuum
                ivar = np.asarray(row["ivar"], np.float64) * continuum**2
        good = (ivar > 0) & np.isfinite(flux) & np.isfinite(ivar)
        wave_nm = wave_A / 10.0
        in_chip = np.zeros(wave_nm.size, bool)
        for lo, hi in CHIPS_NM:
            in_chip |= (wave_nm >= lo) & (wave_nm <= hi)
        filled = np.interp(wave_nm, wave_nm[good], flux[good])
        pseudo = filled / pseudo_continuum_normalize(filled)
        self.path = path
        self.wave_vac = wave_A
        self.wave_nm = wave_nm
        self.flux = filled / pseudo
        self.weight = np.where(good, ivar * pseudo**2, 0.0)
        self.mask = good & in_chip
        columns = []
        for lo, hi in CHIPS_NM:
            chip = ((wave_nm >= lo) & (wave_nm <= hi)).astype(np.float64)
            x = (wave_nm - 0.5 * (lo + hi)) / (hi - lo)
            columns += [chip, chip * x]
        self.basis = np.stack(columns, axis=1)


def calibrated_bundle(calibration: Path, *, device: str, dtype: str):
    """H-band window invariants with an atomic overlay on the metal-line tensors."""

    import torch

    from fitter.apogee.atomic_calibration import calibrated_window_invariants
    from payne_zero_synthesis.pipeline import window_invariants_for

    base = window_invariants_for(wl_start_nm=SYNTH_VAC_NM[0], wl_end_nm=SYNTH_VAC_NM[1], resolution=R_GRID,
                                 molecular_lines=True, runtime_device=torch.device(device),
                                 work_dtype=torch.float64 if dtype == "float64" else torch.float32)
    return calibrated_window_invariants(base, calibration)


class ApogeeFastModel(eso.V4FastModel):
    """``V4FastModel`` over the APOGEE H band, R=22,500, v1 normalization.

    With ``bundle``, synthesis follows ``payne_zero_synthesis.synthesize`` with
    those window invariants in place of the default ones.
    """

    def _synthesize_path(self, path):
        import torch

        from payne_zero_synthesis import synthesize
        from payne_zero_synthesis.api import _surface_flux_per_wavelength_nm
        from payne_zero_synthesis.atmosphere import load_atmosphere_npz
        from payne_zero_synthesis.synthesis import synthesize_structured_atmosphere

        if self.bundle is None:
            spectrum = synthesize(path, wavelength_start_nm=SYNTH_VAC_NM[0], wavelength_end_nm=SYNTH_VAC_NM[1],
                                  resolution=R_GRID, molecular_lines=True, device=self.device, dtype=self.dtype)
            return (np.asarray(spectrum.wavelength_nm, np.float64), np.asarray(spectrum.flux_total, np.float64),
                    np.asarray(spectrum.flux_continuum, np.float64))
        result, _ = synthesize_structured_atmosphere(
            load_atmosphere_npz(path), wavelength_start_nm=SYNTH_VAC_NM[0], wavelength_end_nm=SYNTH_VAC_NM[1],
            resolution=R_GRID, molecular_lines=True, device=torch.device(self.device),
            dtype=torch.float64 if self.dtype == "float64" else torch.float32, window_invariants=self.bundle)
        wavelength = np.asarray(result.wavelength_nm, np.float64)
        return (wavelength,
                _surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_total_per_frequency),
                _surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_continuum_per_frequency))

    def project(self, native, rv_km_s, sigma_km_s) -> np.ndarray:
        import torch

        from fitter import ObservedSpectrumOperator

        wavelength, total, continuum = native
        if self._operator is None or not np.array_equal(self._operator.input_wavelength_nm, wavelength):
            self._operator = ObservedSpectrumOperator(wavelength, self.obs.wave_nm, resolving_power=INSTRUMENT_R,
                                                      device="cpu", dtype=torch.float64)
        self._operator.set_parameters(radial_velocity_km_s=float(rv_km_s),
                                      broadening_sigma_km_s=float(abs(sigma_km_s)))
        _, _, normalized = self._operator.convolve_fluxes(torch.as_tensor(total), torch.as_tensor(continuum))
        return pseudo_continuum_normalize(normalized.detach().cpu().numpy())


def _profiled_chi2(obs: ApogeeObservation, model: np.ndarray) -> float:
    m = obs.mask
    design = model[m, None] * obs.basis[m]
    w = np.sqrt(obs.weight[m])
    coef = np.linalg.lstsq(design * w[:, None], obs.flux[m] * w, rcond=None)[0]
    return float(np.sum(obs.weight[m] * (obs.flux[m] - design @ coef) ** 2))


def _rv_start(obs, fast, start) -> float:
    native = fast.native(*start[:3])
    coarse = np.arange(-30.0, 30.0 + 1e-9, 1.0)
    chi = [_profiled_chi2(obs, fast.project(native, v, SIGMA_START_KM_S)) for v in coarse]
    v0 = coarse[int(np.argmin(chi))]
    fine = np.arange(v0 - 1.0, v0 + 1.0 + 1e-9, 0.1)
    chi = [_profiled_chi2(obs, fast.project(native, v, SIGMA_START_KM_S)) for v in fine]
    return float(fine[int(np.argmin(chi))])


def _stars() -> dict[int, dict]:
    return {int(s["sdss_id"]): s for s in json.loads(V1_REPORT.read_text())["stars"]}


def _faces(parameters: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[str]:
    faces = []
    for name, value, lo, hi in zip(NAMES, parameters, lower, upper):
        if value <= lo + 1e-6:
            faces.append(f"{name}_lower")
        elif value >= hi - 1e-6:
            faces.append(f"{name}_upper")
    return faces


def _code_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()

    dirty = [p for p in ("payne_zero_synthesis", "payne_zero_atmosphere", "fitter",
                         "experiments/mgiant_eso_bestfit_v1.py", "experiments/mgiant_experimental_interface_v1.py")
             if git("status", "--porcelain", "--", p)]
    return {"commit": git("rev-parse", "HEAD"), "dirty_paths": dirty}


def stage_fast(args) -> None:
    from fitter import (FitConfiguration, NormalizedSpectrum, PhysicalAtmosphereConfiguration,
                        fit_normalized_spectrum, refine_with_physical_atmosphere)

    star = _stars()[args.star]
    calibration = None if args.atomic_calibration is None else Path(args.atomic_calibration).resolve()
    arm = "stars" if calibration is None else f"stars_{calibration.stem}"
    out = RESULT_ROOT / arm / str(args.star)
    out.mkdir(parents=True, exist_ok=True)
    bundle, calibration_metadata = (None, None) if calibration is None else calibrated_bundle(
        calibration, device=args.device, dtype=args.dtype)
    obs = ApogeeObservation(args.star)
    fast = ApogeeFastModel(obs, device=args.device, dtype=args.dtype, cache_dir=out / "native_cache", bundle=bundle)
    start = np.array([np.clip(star["teff"], *BOX["teff"]), np.clip(star["logg"], *BOX["logg"]),
                      np.clip(star["mh"], *BOX["m_h"])])
    started = time.time()
    rv0 = _rv_start(obs, fast, start)
    print(f"{args.star}: rv0={rv0:+.2f} km/s ({time.time() - started:.0f} s)", flush=True)

    spectrum = NormalizedSpectrum(wavelength=obs.wave_nm, flux=obs.flux, inverse_variance=obs.weight, mask=obs.mask)
    configuration = FitConfiguration(
        names=NAMES,
        initial=np.array([*start, rv0, SIGMA_START_KM_S]),
        lower=np.array([BOX["teff"][0], BOX["logg"][0], BOX["m_h"][0], rv0 - 5.0, 0.0]),
        upper=np.array([BOX["teff"][1], BOX["logg"][1], BOX["m_h"][1], rv0 + 5.0, 10.0]),
        derivative_steps=np.array([20.0, 0.05, 0.05, 0.1, 0.1]),
        trust_half_width=np.array([150.0, 0.4, 0.3, 1.0, 1.5]),
        maximum_iterations=args.max_iterations,
    )
    fit_started = time.time()
    fast_result = fit_normalized_spectrum(spectrum, configuration, fast, continuum_basis=obs.basis)
    fast_result.save(out / "fast")
    np.save(out / "fast_model_flux.npy", fast_result.model_flux)
    n_good = int(obs.mask.sum())
    summary = {
        "sdss_id": args.star,
        "spectrum": str(obs.path.relative_to(PROJECT_ROOT)),
        "aspcap": {k: star[k] for k in ("teff", "logg", "fe_h", "alpha_m", "mh", "v_rad", "file_snr")},
        "v1_grid_best": star["best_labels"][:3],
        "v1_grid_chi2_reduced": star["chi2_reduced"],
        "start": dict(zip(NAMES, map(float, configuration.initial))),
        "fast": {"parameters": dict(zip(NAMES, map(float, fast_result.parameters))),
                 "faces": _faces(fast_result.parameters, configuration.lower, configuration.upper),
                 "chi2_per_pixel": float(fast_result.mean_weighted_squared_residual),
                 "start_chi2_per_pixel": float(fast_result.objective_path[0]),
                 "converged": fast_result.converged, "stop_reason": fast_result.stop_reason,
                 "evaluations": fast.evaluations, "syntheses": fast.syntheses,
                 "seconds": time.time() - fit_started},
        "n_pixels": n_good,
        "config": {"device": args.device, "dtype": args.dtype, "max_iterations": args.max_iterations,
                   "linelist": "native Payne Zero" if calibration is None else
                   f"native Payne Zero + atomic overlay {calibration.relative_to(PROJECT_ROOT)}",
                   "atomic_calibration": calibration_metadata, "r_grid": R_GRID, "instrument_r": INSTRUMENT_R,
                   "pseudo_continuum_pixels": PSEUDO_CONTINUUM_PIXELS, "continuum": "per-chip linear",
                   "initializer": "m_star_emulator_mgiant_v4 (three-seed median) + physical reconstruction",
                   "code_state": _code_state()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"{args.star} fast: {json.dumps(summary['fast'])}", flush=True)
    if not args.physical:
        return

    converged = eso.ConvergedModel(fast, out / "solves")
    physical_configuration = PhysicalAtmosphereConfiguration(
        maximum_discrepancy_rms=2.0e-3,
        maximum_objective_degradation=0.1,
        minimum_predicted_objective_improvement=1.0e-3,
        maximum_physical_evaluations=args.max_physical,
    )
    started = time.time()
    try:
        physical = refine_with_physical_atmosphere(spectrum, configuration, fast_result, fast, converged,
                                                   physical_configuration, continuum_basis=obs.basis)
    except RuntimeError as error:
        summary["physical"] = {"error": str(error), "solves": converged.records, "seconds": time.time() - started}
    else:
        physical.save(out / "physical")
        np.save(out / "physical_model_flux.npy", physical.model_flux)
        summary["physical"] = {
            "parameters": dict(zip(NAMES, map(float, physical.parameters))),
            "chi2_per_pixel": float(physical.mean_weighted_squared_residual),
            "successful": physical.successful, "physical_fit_stationary": physical.physical_fit_stationary,
            "fast_physical_gates_passed": physical.fast_physical_gates_passed,
            "stop_reason": physical.stop_reason,
            "checks": [{"discrepancy_rms": c.discrepancy_rms, "objective_degradation": c.objective_degradation,
                        "accepted": c.accepted} for c in physical.physical_checks],
            "solves": converged.records, "seconds": time.time() - started,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"{args.star} physical: {json.dumps(summary['physical'], default=str)}", flush=True)


def stage_collect(args) -> None:
    rows = []
    for path in sorted((RESULT_ROOT / args.stars_dir).glob("*/summary.json")):
        s = json.loads(path.read_text())
        fast = s["fast"]["parameters"]
        rows.append({
            "sdss_id": s["sdss_id"],
            "aspcap": [s["aspcap"]["teff"], s["aspcap"]["logg"], s["aspcap"]["mh"]],
            "fast": [fast["teff"], fast["logg"], fast["m_h"]],
            "rv_sigma": [fast["rv_km_s"], fast["sigma_extra_km_s"]],
            "faces": s["fast"]["faces"],
            "chi2_per_pixel": s["fast"]["chi2_per_pixel"],
            "v1_grid_best": s["v1_grid_best"], "v1_grid_chi2_reduced": s["v1_grid_chi2_reduced"],
            "stop_reason": s["fast"]["stop_reason"], "syntheses": s["fast"]["syntheses"],
            "seconds": s["fast"]["seconds"],
            "physical": s.get("physical"),
        })
    rows.sort(key=lambda r: r["aspcap"][0])
    (RESULT_ROOT / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    for r in rows:
        print(f"{r['sdss_id']:>10}  ASPCAP {r['aspcap'][0]:6.0f}/{r['aspcap'][1]:5.2f}/{r['aspcap'][2]:+5.2f}  "
              f"PZ {r['fast'][0]:6.0f}/{r['fast'][1]:5.2f}/{r['fast'][2]:+5.2f}  chi2/px {r['chi2_per_pixel']:8.1f}  "
              f"{r['stop_reason']:>18}  faces {','.join(r['faces']) or '-'}")
    label_figure(rows, RESULT_ROOT / "labels_vs_aspcap.png")


def label_figure(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), layout="constrained")
    for column, (ax, label) in enumerate(zip(axes, ("Teff (K)", "log g", "[M/H]"))):
        reference = np.array([r["aspcap"][column] for r in rows])
        fitted = np.array([r["fast"][column] for r in rows])
        railed = np.array([any(f.startswith(NAMES[column]) for f in r["faces"]) for r in rows])
        ax.scatter(reference[~railed], fitted[~railed], color="C0", label="interior")
        ax.scatter(reference[railed], fitted[railed], facecolor="none", edgecolor="C3", label="box face")
        lims = (min(reference.min(), fitted.min()), max(reference.max(), fitted.max()))
        pad = 0.05 * (lims[1] - lims[0])
        lims = (lims[0] - pad, lims[1] + pad)
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set(xlim=lims, ylim=lims, xlabel=f"ASPCAP {label}", ylabel=f"Payne Zero {label}")
        delta = fitted - reference
        ax.set_title(f"mean {delta.mean():+.3g}, rms {np.sqrt(np.mean(delta**2)):.3g}", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle("DR20 APOGEE M giants: Payne Zero M-giant fitter vs ASPCAP", fontsize=11)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    fast = sub.add_parser("fast")
    fast.add_argument("--star", type=int, required=True)
    fast.add_argument("--device", default="mps")
    fast.add_argument("--dtype", default="float32")
    fast.add_argument("--max-iterations", type=int, default=8)
    fast.add_argument("--physical", action="store_true")
    fast.add_argument("--max-physical", type=int, default=4)
    fast.add_argument("--atomic-calibration", default=None)
    collect = sub.add_parser("collect")
    collect.add_argument("--stars-dir", default="stars")
    args = parser.parse_args()
    {"fast": stage_fast, "collect": stage_collect}[args.stage](args)


if __name__ == "__main__":
    main()
