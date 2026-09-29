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
    free     --star SDSS_ID   ten-parameter fit (adds [alpha/M], [C/M], [N/M], vmic, vsini)
    astra    [--arms A ...]   Astra ASPCAP best-fit model vs the Payne Zero fits

The ten-parameter fit keeps the atmosphere structure on the scaled-solar slice
that the v4 initializer and the solver support (alpha=C=N=0, vmic=2 km/s): at
fixed temperature and gas pressure, [alpha/M] (O, Ne, Mg, Si, S, Ca, Ti), [C/M]
and [N/M] rescale the abundance vector, the EOS and molecular equilibrium are
recomputed, and vmic sets the microturbulence of the synthesis. vsini is a
linear-limb-darkened rotation kernel on the native grid; the extra Gaussian
broadening stays free. It starts from the converged result of a five-parameter
arm with alpha=C=N=0, vmic=2 km/s and vsini=3 km/s.
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
ASTRA_DIR = PROJECT_ROOT / "data" / "dr20" / "results" / "star"
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


FREE_NAMES = ("teff", "logg", "m_h", "alpha_m", "c_m", "n_m", "vmic_km_s", "rv_km_s", "sigma_extra_km_s",
              "vsini_km_s")
FREE_BOUNDS = {"alpha_m": (-0.3, 0.5), "c_m": (-0.8, 0.5), "n_m": (-0.5, 1.2), "vmic_km_s": (0.5, 4.0),
               "vsini_km_s": (0.0, 15.0)}
ALPHA_Z = (8, 10, 12, 14, 16, 20, 22)
HELIUM_FRACTION_INDEX = 1


def perturbed_abundances(abundances: np.ndarray, alpha_m: float, c_m: float, n_m: float) -> np.ndarray:
    """Scale alpha elements, C and N of a linear number-fraction vector (Z=1..99); H absorbs the change."""

    out = np.asarray(abundances, np.float64).copy()
    for z in ALPHA_Z:
        out[z - 1] *= 10.0 ** alpha_m
    out[5] *= 10.0 ** c_m
    out[6] *= 10.0 ** n_m
    out[0] = abundances[0] + float(np.sum(abundances[2:]) - np.sum(out[2:]))
    return out


class FreeAbundanceFastModel(ApogeeFastModel):
    """Ten-parameter fast model: v4 structure at (Teff, logg, [M/H]), perturbed synthesis."""

    def __init__(self, obs, **kwargs):
        super().__init__(obs, **kwargs)
        self.atmosphere_dir = self.cache_dir.parent / "atmosphere_cache"
        self.atmosphere_dir.mkdir(parents=True, exist_ok=True)
        self._rotation = None

    @staticmethod
    def free_key(teff, logg, m_h, alpha_m, c_m, n_m, vmic):
        return (*eso.V4FastModel.deck_labels(teff, logg, m_h), round(float(alpha_m), 4), round(float(c_m), 4),
                round(float(n_m), 4), round(float(vmic), 3))

    def base_atmosphere(self, teff, logg, m_h) -> Path:
        from experiments import mgiant_experimental_interface_v1 as interface
        from payne_zero_atmosphere.synthesis_bridge import save_product_structured_atmosphere

        key = self.deck_labels(teff, logg, m_h)
        path = self.atmosphere_dir / f"t{key[0]:.0f}_g{key[1]:+.4f}_m{key[2]:+.4f}.npz"
        if not path.is_file():
            atmosphere, _ = interface.build_v4_initializer(self.labels(*key), interface.DEFAULT_CHECKPOINT_DIR)
            save_product_structured_atmosphere(atmosphere, path, device="cpu", dtype="float64")
        return path

    def synthesize_perturbed(self, path, alpha_m, c_m, n_m, vmic):
        """Synthesis of a saved atmosphere with rescaled abundances and microturbulence."""

        import torch

        from payne_zero_synthesis.api import _surface_flux_per_wavelength_nm
        from payne_zero_synthesis.synthesis import (build_structured_atmosphere_from_columns,
                                                    synthesize_structured_atmosphere)

        dtype = torch.float64 if self.dtype == "float64" else torch.float32
        with np.load(path, allow_pickle=False) as data:
            columns = {key: np.asarray(data[key], np.float64) for key in
                       ("temperature", "column_mass", "gas_pressure", "electron_density", "elemental_abundances")}
        atmosphere = build_structured_atmosphere_from_columns(
            temperature=columns["temperature"], column_mass=columns["column_mass"],
            gas_pressure=columns["gas_pressure"], electron_density=columns["electron_density"],
            elemental_abundances=perturbed_abundances(columns["elemental_abundances"], alpha_m, c_m, n_m),
            microturbulence=np.full(columns["temperature"].size, float(vmic) * 1.0e5), mass_density=None,
            device=self.device, dtype=dtype, molecular_lines=True)
        result, _ = synthesize_structured_atmosphere(
            atmosphere, wavelength_start_nm=SYNTH_VAC_NM[0], wavelength_end_nm=SYNTH_VAC_NM[1], resolution=R_GRID,
            molecular_lines=True, device=torch.device(self.device), dtype=dtype, window_invariants=self.bundle)
        wavelength = np.asarray(result.wavelength_nm, np.float64)
        return (wavelength,
                _surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_total_per_frequency),
                _surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_continuum_per_frequency))

    def native_free(self, teff, logg, m_h, alpha_m, c_m, n_m, vmic):
        key = self.free_key(teff, logg, m_h, alpha_m, c_m, n_m, vmic)
        if key in self._native:
            return self._native[key]
        cache = self.cache_dir / ("t{:.0f}_g{:+.4f}_m{:+.4f}_a{:+.4f}_c{:+.4f}_n{:+.4f}_v{:.3f}.npz".format(*key))
        if cache.is_file():
            with np.load(cache) as data:
                self._native[key] = (data["wavelength_nm"], data["flux_total"], data["flux_continuum"])
            return self._native[key]
        native = self.synthesize_perturbed(self.base_atmosphere(*key[:3]), *key[3:])
        np.savez(cache, wavelength_nm=native[0], flux_total=native[1], flux_continuum=native[2])
        self._native[key] = native
        self.syntheses += 1
        return native

    def project_free(self, native, rv_km_s, sigma_km_s, vsini_km_s) -> np.ndarray:
        import torch

        from fitter import RotationalBroadening

        wavelength, total, continuum = native
        if self._rotation is None or not np.array_equal(self._rotation.wavelength_nm, wavelength):
            self._rotation = RotationalBroadening(wavelength, maximum_vsini_km_s=20.0, device="cpu",
                                                  dtype=torch.float64)
        vsini = max(float(vsini_km_s), 0.0)
        total = self._rotation(torch.as_tensor(total), vsini_km_s=vsini).numpy()
        continuum = self._rotation(torch.as_tensor(continuum), vsini_km_s=vsini).numpy()
        return self.project((wavelength, total, continuum), rv_km_s, sigma_km_s)

    def __call__(self, parameters) -> np.ndarray:
        self.evaluations += 1
        teff, logg, m_h, alpha_m, c_m, n_m, vmic, rv, sigma, vsini = parameters
        return self.project_free(self.native_free(teff, logg, m_h, alpha_m, c_m, n_m, vmic), rv, sigma, vsini)


class FreeConvergedModel:
    """Converged atmosphere on the scaled-solar slice, synthesized with the perturbed abundances."""

    def __init__(self, fast: FreeAbundanceFastModel, solve_dir: Path):
        self.fast = fast
        self.solve_dir = solve_dir
        self.records: list[dict] = []

    def __call__(self, parameters) -> np.ndarray:
        from experiments import mgiant_experimental_interface_v1 as interface

        teff, logg, m_h, alpha_m, c_m, n_m, vmic, rv, sigma, vsini = parameters
        started = time.time()
        summary = interface.solve_experimental_point(self.fast.labels(teff, logg, m_h), output_dir=self.solve_dir)
        record = {"parameters": dict(zip(FREE_NAMES, map(float, parameters))),
                  "solver": summary["solver"], "quality_flag": summary["quality_flag"],
                  "final_product": summary["final_product"], "seconds": time.time() - started}
        self.records.append(record)
        if not summary["quality_flag"]["valid"] or summary["final_product"] is None:
            raise RuntimeError(f"converged atmosphere failed its quality flag at {record['parameters']}")
        key = self.fast.free_key(teff, logg, m_h, alpha_m, c_m, n_m, vmic)
        native = self.fast.synthesize_perturbed(summary["final_product"], *key[3:])
        return self.fast.project_free(native, rv, sigma, vsini)


def stage_free(args) -> None:
    from fitter import (FitConfiguration, NormalizedSpectrum, PhysicalAtmosphereConfiguration,
                        fit_normalized_spectrum, refine_with_physical_atmosphere)

    star = _stars()[args.star]
    calibration = None if args.atomic_calibration is None else Path(args.atomic_calibration).resolve()
    base_arm = "stars" if calibration is None else f"stars_{calibration.stem}"
    if args.linelist == "native":
        out = RESULT_ROOT / f"free_{base_arm}" / str(args.star)
        base = json.loads((RESULT_ROOT / base_arm / str(args.star) / "summary.json").read_text())
        start5 = (base.get("physical") or {}).get("parameters") or base["fast"]["parameters"]
        start = [start5["teff"], start5["logg"], start5["m_h"], 0.0, 0.0, 0.0, 2.0, start5["rv_km_s"],
                 start5["sigma_extra_km_s"], 3.0]
        bundle, calibration_metadata = (None, None) if calibration is None else calibrated_bundle(
            calibration, device=args.device, dtype=args.dtype)
    else:
        # APOGEE DR17 line-list arm, started from the ten-parameter Sun/Arcturus solution.
        from experiments import mgiant_apogee_linelist_v1 as linelist

        out = RESULT_ROOT / f"free_{args.linelist}" / str(args.star)
        base_arm = "free_stars_sun_arcturus_fts_hband_shared"
        base = json.loads((RESULT_ROOT / base_arm / str(args.star) / "summary.json").read_text())
        p = dict((base.get("physical") or {}).get("parameters") or base["fast"]["parameters"])
        astra = next((r["astra"] for r in json.loads((RESULT_ROOT / "astra_comparison.json").read_text())
                      if r["sdss_id"] == args.star), None)
        if args.start == "astra" and astra is not None:
            p.update(teff=astra["teff"], logg=astra["logg"], m_h=astra["m_h_atm"], alpha_m=astra["alpha_m_atm"],
                     c_m=astra["c_m_atm"], n_m=astra["n_m_atm"], vmic_km_s=astra["v_micro"])
        start = [p[name] for name in FREE_NAMES]
        start[FREE_NAMES.index("vsini_km_s")] = max(start[FREE_NAMES.index("vsini_km_s")], 3.0)
        bundles, calibration_metadata = linelist.build_bundles(args.device, args.dtype, out / "linelist_scratch")
        bundle = bundles[args.linelist]
    out.mkdir(parents=True, exist_ok=True)
    obs = ApogeeObservation(args.star)
    fast = FreeAbundanceFastModel(obs, device=args.device, dtype=args.dtype, cache_dir=out / "native_cache",
                                  bundle=bundle)
    rv0 = start[FREE_NAMES.index("rv_km_s")]
    initial = np.array(start, dtype=np.float64)
    lower = np.array([BOX["teff"][0], BOX["logg"][0], BOX["m_h"][0], FREE_BOUNDS["alpha_m"][0],
                      FREE_BOUNDS["c_m"][0], FREE_BOUNDS["n_m"][0], FREE_BOUNDS["vmic_km_s"][0], rv0 - 5.0, 0.0,
                      FREE_BOUNDS["vsini_km_s"][0]])
    upper = np.array([BOX["teff"][1], BOX["logg"][1], BOX["m_h"][1], FREE_BOUNDS["alpha_m"][1],
                      FREE_BOUNDS["c_m"][1], FREE_BOUNDS["n_m"][1], FREE_BOUNDS["vmic_km_s"][1], rv0 + 5.0, 10.0,
                      FREE_BOUNDS["vsini_km_s"][1]])
    initial = np.clip(initial, lower, upper)
    spectrum = NormalizedSpectrum(wavelength=obs.wave_nm, flux=obs.flux, inverse_variance=obs.weight, mask=obs.mask)
    configuration = FitConfiguration(
        names=FREE_NAMES, initial=initial, lower=lower, upper=upper,
        derivative_steps=np.array([20.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.5]),
        trust_half_width=np.array([150.0, 0.4, 0.3, 0.2, 0.2, 0.3, 0.5, 1.0, 1.5, 3.0]),
        maximum_iterations=args.max_iterations,
    )
    started = time.time()
    fast_result = fit_normalized_spectrum(spectrum, configuration, fast, continuum_basis=obs.basis)
    fast_result.save(out / "fast")
    np.save(out / "fast_model_flux.npy", fast_result.model_flux)
    summary = {
        "sdss_id": args.star,
        "spectrum": str(obs.path.relative_to(PROJECT_ROOT)),
        "aspcap": {k: star[k] for k in ("teff", "logg", "fe_h", "alpha_m", "mh", "v_rad", "file_snr")},
        "start_arm": base_arm if args.linelist == "native" else f"{args.start} ({base_arm} for RV and broadening)",
        "start": dict(zip(FREE_NAMES, map(float, initial))),
        "fast": {"parameters": dict(zip(FREE_NAMES, map(float, fast_result.parameters))),
                 "faces": [f for f in _faces_free(fast_result.parameters, lower, upper)],
                 "chi2_per_pixel": float(fast_result.mean_weighted_squared_residual),
                 "start_chi2_per_pixel": float(fast_result.objective_path[0]),
                 "converged": fast_result.converged, "stop_reason": fast_result.stop_reason,
                 "evaluations": fast.evaluations, "syntheses": fast.syntheses, "seconds": time.time() - started},
        "n_pixels": int(obs.mask.sum()),
        "config": {"device": args.device, "dtype": args.dtype, "max_iterations": args.max_iterations,
                   "linelist": (args.linelist if args.linelist != "native" else
                                "native Payne Zero" if calibration is None else
                                f"native Payne Zero + atomic overlay {calibration.relative_to(PROJECT_ROOT)}"),
                   "atomic_calibration": calibration_metadata, "r_grid": R_GRID, "instrument_r": INSTRUMENT_R,
                   "structure": "v4 / converged atmosphere at (Teff, logg, [M/H], alpha=0, vmic=2); alpha, C, N and "
                                "vmic applied in the synthesis at fixed temperature and gas pressure",
                   "rotation": "linear limb darkening 0.6 on the native grid",
                   "code_state": _code_state()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"{args.star} free fast: {json.dumps(summary['fast'])}", flush=True)

    converged = FreeConvergedModel(fast, out / "solves")
    physical_configuration = PhysicalAtmosphereConfiguration(
        maximum_discrepancy_rms=2.0e-3, maximum_objective_degradation=0.1,
        minimum_predicted_objective_improvement=1.0e-3, maximum_physical_evaluations=args.max_physical)
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
            "parameters": dict(zip(FREE_NAMES, map(float, physical.parameters))),
            "chi2_per_pixel": float(physical.mean_weighted_squared_residual),
            "successful": physical.successful, "physical_fit_stationary": physical.physical_fit_stationary,
            "fast_physical_gates_passed": physical.fast_physical_gates_passed, "stop_reason": physical.stop_reason,
            "checks": [{"discrepancy_rms": c.discrepancy_rms, "objective_degradation": c.objective_degradation,
                        "accepted": c.accepted} for c in physical.physical_checks],
            "solves": converged.records, "seconds": time.time() - started,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"{args.star} free physical: {json.dumps(summary['physical'], default=str)}", flush=True)


def _faces_free(parameters, lower, upper) -> list[str]:
    return [f"{name}_{'lower' if value <= lo + 1e-6 else 'upper'}"
            for name, value, lo, hi in zip(FREE_NAMES, parameters, lower, upper)
            if value <= lo + 1e-6 or value >= hi - 1e-6]


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
    out = RESULT_ROOT / (args.output or "summary.json")
    rows = []
    for path in sorted((RESULT_ROOT / args.stars_dir).glob("*/summary.json")):
        s = json.loads(path.read_text())
        fast = s["fast"]["parameters"]
        rows.append({
            "sdss_id": s["sdss_id"],
            "file_snr": s["aspcap"]["file_snr"],
            "rv_start_km_s": s["start"]["rv_km_s"],
            "aspcap": [s["aspcap"]["teff"], s["aspcap"]["logg"], s["aspcap"]["mh"]],
            "fast": [fast["teff"], fast["logg"], fast["m_h"]],
            "rv_sigma": [fast["rv_km_s"], fast["sigma_extra_km_s"]],
            "faces": s["fast"]["faces"],
            "chi2_per_pixel": s["fast"]["chi2_per_pixel"],
            "v1_grid_best": s.get("v1_grid_best"), "v1_grid_chi2_reduced": s.get("v1_grid_chi2_reduced"),
            "stop_reason": s["fast"]["stop_reason"], "syntheses": s["fast"]["syntheses"],
            "seconds": s["fast"]["seconds"],
            "physical": s.get("physical"),
        })
    rows.sort(key=lambda r: r["aspcap"][0])
    out.write_text(json.dumps(rows, indent=2, default=str))
    for r in rows:
        print(f"{r['sdss_id']:>10}  ASPCAP {r['aspcap'][0]:6.0f}/{r['aspcap'][1]:5.2f}/{r['aspcap'][2]:+5.2f}  "
              f"PZ {r['fast'][0]:6.0f}/{r['fast'][1]:5.2f}/{r['fast'][2]:+5.2f}  chi2/px {r['chi2_per_pixel']:8.1f}  "
              f"{r['stop_reason']:>18}  faces {','.join(r['faces']) or '-'}")
    label_figure(rows, out.with_name(out.name.replace("summary", "labels_vs_aspcap")).with_suffix(".png"))


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


def _profiled_fit(obs: ApogeeObservation, model: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Model times the per-chip linear continuum fitted on ``pixels``."""

    design = model[:, None] * obs.basis
    w = np.sqrt(obs.weight[pixels])
    coef = np.linalg.lstsq(design[pixels] * w[:, None], obs.flux[pixels] * w, rcond=None)[0]
    return design @ coef


def read_astra_aspcap(sdss_id: int) -> dict | None:
    """First APOGEE row of the star's astraStarASPCAP-0.6.0 file (raw FERRE labels)."""

    from astropy.io import fits

    path = ASTRA_DIR / f"{(sdss_id // 100) % 100:02d}" / f"{sdss_id % 100:02d}" / f"astraStarASPCAP-0.6.0-{sdss_id}.fits"
    if not path.is_file():
        return None
    with fits.open(path) as hdu_list:
        hdu = hdu_list[3] if len(hdu_list[3].data) else hdu_list[4]
        row = hdu.data[0]
        record = {"path": str(path.relative_to(PROJECT_ROOT)), "hdu": hdu.name,
                  "model_flux": np.asarray(row["model_flux"], np.float64),
                  "spectrum_pk_id": int(row["spectrum_pk_id"]), "short_grid_name": str(row["short_grid_name"]),
                  "rchi2": float(row["rchi2"])}
        for key in ("teff", "logg", "m_h_atm", "alpha_m_atm", "c_m_atm", "n_m_atm", "v_micro", "v_sini"):
            record[key] = float(row[key])
    return record


def stage_astra(args) -> None:
    """Astra ASPCAP (FERRE) best-fit model and Payne Zero fits on one pixel set and normalization.

    Both models go through the same 339-pixel running median and per-chip
    linear continuum against the same mwmStar coadd. Scored pixels are Payne
    Zero fit pixels at least half a median window inside the retained apStar
    segments where the Astra model is defined (one mask for every star), so the
    running median never reaches an undefined model pixel. The Astra model is at
    ASPCAP's raw FERRE labels with its own [alpha/M], [C/M], [N/M], vmic and
    vsini; it was fitted to an earlier coadd of the same star.
    """

    from scipy.ndimage import binary_erosion

    windows = ((1518.0, 1528.0), (1553.0, 1563.0), (1600.0, 1610.0), (1670.0, 1680.0))
    stars = _stars()
    astras = {sdss_id: read_astra_aspcap(sdss_id) for sdss_id in stars}
    reference = next(a["model_flux"] for a in astras.values() if a is not None)
    defined = np.isfinite(reference) & (reference > 0)
    interior = binary_erosion(defined, np.ones(PSEUDO_CONTINUUM_PIXELS, bool))
    records = []
    arrays = {}
    for sdss_id in sorted(stars, key=lambda i: stars[i]["teff"]):
        astra = astras[sdss_id]
        obs = ApogeeObservation(sdss_id)
        row = {"sdss_id": sdss_id, "astra": None if astra is None else
               {k: v for k, v in astra.items() if k != "model_flux"}}
        models = {}
        if astra is not None:
            own = np.isfinite(astra["model_flux"]) & (astra["model_flux"] > 0)
            if not np.array_equal(own, defined):
                raise ValueError(f"{sdss_id}: Astra model defined on a different pixel set")
            filled = np.interp(obs.wave_nm, obs.wave_nm[own], astra["model_flux"][own])
            models["astra"] = pseudo_continuum_normalize(filled)
        pixels = obs.mask & interior
        for arm in args.arms:
            summary = RESULT_ROOT / "garching" / arm / str(sdss_id) / "summary.json"
            flux = summary.parent / "physical_model_flux.npy"
            if not flux.is_file():
                continue
            fitted = json.loads(summary.read_text())
            labels = (fitted.get("physical") or {}).get("parameters") or fitted["fast"]["parameters"]
            # The saved flux already carries the fit's continuum; refit it on the common pixels.
            models[arm] = np.load(flux)
            row[arm] = {"labels": [labels["teff"], labels["logg"], labels["m_h"]],
                        "sigma_extra_km_s": labels["sigma_extra_km_s"],
                        "extra_labels": {k: labels[k] for k in ("alpha_m", "c_m", "n_m", "vmic_km_s", "vsini_km_s")
                                         if k in labels}}
        predictions = {name: _profiled_fit(obs, model, pixels) for name, model in models.items()}
        row["n_pixels"] = int(pixels.sum())
        row["metrics"] = {}
        for name, prediction in predictions.items():
            residual = (obs.flux - prediction)[pixels]
            chips = {}
            for lo, hi in CHIPS_NM:
                chip = pixels & (obs.wave_nm >= lo) & (obs.wave_nm <= hi)
                r = (obs.flux - prediction)[chip]
                chips[f"{lo:.0f}-{hi:.0f}"] = {"chi2_per_pixel": float(np.mean(obs.weight[chip] * r**2)),
                                              "residual_rms": float(np.sqrt(np.mean(r**2)))}
            row["metrics"][name] = {"chi2_per_pixel": float(np.mean(obs.weight[pixels] * residual**2)),
                                    "residual_rms": float(np.sqrt(np.mean(residual**2))), "chips": chips}
        names = list(predictions)
        row["model_difference_rms"] = {f"{a}-{b}": float(np.sqrt(np.mean((predictions[a] - predictions[b])[pixels] ** 2)))
                                       for i, a in enumerate(names) for b in names[i + 1:]}
        records.append(row)
        arrays[f"{sdss_id}_wave_nm"] = obs.wave_nm
        arrays[f"{sdss_id}_flux"] = obs.flux.astype(np.float32)
        arrays[f"{sdss_id}_pixels"] = pixels
        for name, prediction in predictions.items():
            arrays[f"{sdss_id}_model_{name}"] = prediction.astype(np.float32)
        text = "  ".join(f"{n} chi2/px {m['chi2_per_pixel']:7.1f} rms {m['residual_rms']:.4f}"
                         for n, m in row["metrics"].items())
        print(f"{sdss_id:>10} npix {row['n_pixels']}  {text}", flush=True)
        if astra is not None and args.figure_star == sdss_id:
            _astra_figure(obs, predictions, pixels, windows, sdss_id, row)
    (RESULT_ROOT / "astra_comparison.json").write_text(json.dumps(records, indent=2))
    np.savez_compressed(RESULT_ROOT / "astra_predictions.npz", **arrays)


ARM_LABELS = {"astra": "Astra ASPCAP", "stars": "PZ native", "stars_sun_arcturus_fts_hband_shared": "PZ Sun+Arcturus",
              "free_stars_sun_arcturus_fts_hband_shared": "PZ Sun+Arcturus, 10 par."}


def _astra_figure(obs, predictions, pixels, windows, sdss_id, row) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"astra": "C1", "stars": "C0", "stars_sun_arcturus_fts_hband_shared": "C2",
              "free_stars_sun_arcturus_fts_hband_shared": "C3"}
    fig, axes = plt.subplots(2, len(windows), figsize=(15.0, 6.0), sharex="col", layout="constrained",
                             gridspec_kw={"height_ratios": (3, 1)})
    for column, (lo, hi) in enumerate(windows):
        inside = (obs.wave_nm >= lo) & (obs.wave_nm <= hi)
        top, bottom = axes[0, column], axes[1, column]
        top.plot(obs.wave_nm[inside], np.where(pixels, obs.flux, np.nan)[inside], color="k", lw=0.8, label="APOGEE")
        for name, prediction in predictions.items():
            color = colors.get(name, "C4")
            label = ARM_LABELS.get(name, name)
            top.plot(obs.wave_nm[inside], prediction[inside], color=color, lw=0.8, label=label)
            bottom.plot(obs.wave_nm[inside], np.where(pixels, obs.flux - prediction, np.nan)[inside], color=color, lw=0.7)
        bottom.axhline(0.0, color="0.5", lw=0.5)
        bottom.set_xlabel("vacuum wavelength (nm)")
    axes[0, 0].set_ylabel("normalized flux")
    axes[1, 0].set_ylabel("data - model")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    astra = row["astra"]
    fig.suptitle(f"sdss_id {sdss_id}: Astra ASPCAP {astra['teff']:.0f}/{astra['logg']:.2f}/{astra['m_h_atm']:+.2f} (raw)"
                 + "".join(f"; {ARM_LABELS.get(name, name)} {v['labels'][0]:.0f}/{v['labels'][1]:.2f}/{v['labels'][2]:+.2f}"
                           for name, v in row.items() if isinstance(v, dict) and "labels" in v), fontsize=10)
    fig.savefig(RESULT_ROOT / f"astra_vs_pz_{sdss_id}.png", dpi=150)
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
    collect.add_argument("--output", default=None, help="file name under the result root (default summary.json)")
    free = sub.add_parser("free")
    free.add_argument("--star", type=int, required=True)
    free.add_argument("--device", default="mps")
    free.add_argument("--dtype", default="float32")
    free.add_argument("--max-iterations", type=int, default=12)
    free.add_argument("--max-physical", type=int, default=4)
    free.add_argument("--atomic-calibration", default=None)
    free.add_argument("--linelist", choices=("native", "apogee_atoms", "apogee_molec", "apogee_all"),
                      default="native")
    free.add_argument("--start", choices=("astra", "free"), default="astra",
                      help="line-list arms: start at the Astra raw FERRE labels where available, else the "
                           "ten-parameter Sun/Arcturus solution")
    astra = sub.add_parser("astra")
    astra.add_argument("--arms", nargs="*", default=["stars", "stars_sun_arcturus_fts_hband_shared",
                                                     "free_stars_sun_arcturus_fts_hband_shared"])
    astra.add_argument("--figure-star", type=int, default=96257659)
    args = parser.parse_args()
    {"fast": stage_fast, "collect": stage_collect, "free": stage_free, "astra": stage_astra}[args.stage](args)


if __name__ == "__main__":
    main()
