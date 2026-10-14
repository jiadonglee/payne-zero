"""Optical fits of the Gaia benchmark giants with the final networks (Sect. 5).

alpha Cet and alpha Tau (UVES-POP) calibrate the TiO lines; gamma Sge (NARVAL,
Gaia FGK benchmark library) is kept out of the calibration as the test. The
benchmark labels are those of the third version (Soubiran et al. 2024). The
window is 6480-6735 A (air) with H-alpha and Li I masked, and a cubic B-spline
continuum with 25 A knots is profiled at every model evaluation. Fits free
Teff, logg, [M/H], RV and the extra broadening, with [alpha/M]=[C/M]=[N/M]=
[O/M]=0 and the benchmark microturbulence, starting from 3700 K, 1.0, -0.3.

Stages::

    reference --star KEY       solved atmosphere at the benchmark labels, RV and broadening fitted
    calibrate                  TiO log gf offsets in five wavelength bins on the calibration stars
    fit --star KEY [--offsets] fast fit, then refinement with solved atmospheres
    collect                    fitted against benchmark labels for both lists
"""
from bench import environment as _environment

import argparse
import csv
import dataclasses
import json
from pathlib import Path
import time

import numpy as np

from coauthor_mgiant_apogee_fit_20261013 import ROOT, GATE, NetworkModel, SolvedModel, solve_stable
from experiments import mgiant_eso_bestfit_v1 as eso
from experiments import mgiant_apogee_linelist_v1 as linelist

OUT = ROOT / "results/coauthor_validation_20261005/mgiant_benchmark_fit"
GBS = ROOT / "data/gbs"
STARS = dict(alfCet="calibration", alfTau="calibration", gamSge="test")
NARVAL_R = 70_000.0
TIO_SPECIES = 366
TIO_BINS_A = np.linspace(eso.WINDOW_AIR_A[0], eso.WINDOW_AIR_A[1], 6)
START = dict(teff=3700.0, logg=1.0, m_h=-0.3)
PARAMETERS = ("teff", "logg", "m_h", "rv_km_s", "sigma_extra_km_s")
BOUNDS = dict(teff=(3000.0, 4000.0), logg=(0.5, 2.5), m_h=(-1.0, 0.5))


def gbs_labels(key):
    """Third-version benchmark labels (Soubiran et al. 2024), microturbulence from the library catalogue."""
    v3 = json.loads((GBS / "soubiran2024_cool_giants.json").read_text())["stars"][key]
    with (GBS / "metadata/catalog_with_local_paths.csv").open() as handle:
        header = handle.readline().lstrip("#").strip().split(",")
        row = next(r for r in csv.DictReader(handle, fieldnames=header) if r["star"] == key)
    return dict(teff=v3["teff"], logg=v3["logg"], m_h=v3["m_h"], vmic_km_s=float(row["vmic"]))


class Observation:
    """Fit pixels in the TiO window: UVES-POP for the ESO targets, the GBS library spectrum otherwise."""

    def __init__(self, key):
        targets = {t["key"]: t for t in json.loads((ROOT / "results/mgiant_eso_bestfit_v1/targets.json").read_text())}
        if key in targets:
            base = eso.Observation(targets[key])
            self.instrument_r = eso.INSTRUMENT_R
            wave_air, flux, weight, basis = base.wave_air, base.flux, base.weight, base.basis
        else:
            from astropy.io import fits
            from scipy.interpolate import BSpline

            with fits.open(GBS / f"spectra_fits/NARVAL_{key}.fits") as hdu:
                flux, sigma = np.asarray(hdu[0].data, np.float64), np.asarray(hdu[1].data, np.float64)
                header = hdu[0].header
            wave_air = (header["CRVAL1"] + header["CDELT1"] * np.arange(flux.size)) * 10.0
            good = ((wave_air >= eso.WINDOW_AIR_A[0]) & (wave_air <= eso.WINDOW_AIR_A[1]) & np.isfinite(flux)
                    & np.isfinite(sigma) & (sigma > 0) & (flux > 0))
            wave_air, flux, weight = wave_air[good], flux[good], 1.0 / sigma[good] ** 2
            lo, hi = wave_air[0], wave_air[-1]
            interior = np.linspace(lo, hi, max(1, int(round((hi - lo) / eso.KNOT_SPACING_A))) + 1)
            basis = BSpline.design_matrix(np.clip(wave_air, lo, hi),
                                          np.concatenate([[lo] * 3, interior, [hi] * 3]), 3).toarray()
            self.instrument_r = NARVAL_R
        self.key, self.wave_air, self.flux, self.weight, self.basis = key, wave_air, flux, weight, basis
        self.wave_nm = eso.air_to_vacuum_A(wave_air) / 10.0
        self.mask = np.ones(wave_air.size, bool)

    def set_rest_frame_mask(self, rv_km_s):
        rest = self.wave_air / (1.0 + rv_km_s / eso.C_KM_S)
        self.mask = np.ones(rest.size, bool)
        for lo, hi in eso.MASKS_AIR_A:
            self.mask &= ~((rest >= lo) & (rest <= hi))


class OpticalModel(NetworkModel):
    """Five-parameter fast model over the TiO window; the other labels are fixed."""

    window_nm, r_grid = eso.SYNTH_VAC_NM, eso.PZ_RESOLUTION

    def __init__(self, obs, bundle, device, dtype, directory, vmic):
        super().__init__(obs, bundle, device, dtype, directory)
        self.instrument_r, self.vmic = obs.instrument_r, vmic

    def full(self, parameters):
        teff, logg, m_h, rv, sigma = parameters
        return [teff, logg, m_h, 0.0, 0.0, 0.0, self.vmic, rv, sigma, 0.0]

    def __call__(self, parameters):
        return super().__call__(self.full(parameters))


class OpticalSolved(SolvedModel):
    def __call__(self, parameters):
        return super().__call__(self.fast.full(parameters))


def bundle_for(offsets, device, dtype):
    """Default optical invariants with the TiO lines shifted in log gf by wavelength bin."""
    import torch
    from payne_zero_synthesis import pipeline as pz

    base = pz.window_invariants_for(wl_start_nm=eso.SYNTH_VAC_NM[0], wl_end_nm=eso.SYNTH_VAC_NM[1],
                                    resolution=eso.PZ_RESOLUTION, molecular_lines=True,
                                    runtime_device=torch.device(device), work_dtype=getattr(torch, dtype))
    if not any(offsets):
        return base
    compiled = dict(pz.SynthesisPipeline._compile_molecular(eso.SYNTH_VAC_NM[0], eso.SYNTH_VAC_NM[1],
                                                             eso.PZ_RESOLUTION))
    ratio, origin = float(compiled["log_grid_ratio"]), int(compiled["grid_origin_index"])
    wave_air = eso.vacuum_to_air_A(np.exp((compiled["center_index_1based"] - 1 + origin) * ratio) * 10.0)
    factor = np.ones(wave_air.size)
    tio = compiled["species_code"] == TIO_SPECIES
    for i, offset in enumerate(offsets):
        inside = tio & (wave_air >= TIO_BINS_A[i]) & (wave_air < TIO_BINS_A[i + 1])
        factor[inside] = 10.0 ** offset
    compiled["classical_line_strength"] = (compiled["classical_line_strength"] * factor).astype(np.float32)
    invariants, count = linelist._molecular_invariants(base, compiled, device)
    return dataclasses.replace(base, molecular_invariants=invariants, n_molecular=count)


def rv_start(obs, model, labels):
    native = model.native(model.key(model.full([*labels, 0.0, 0.0])))
    chi = lambda v: _chi2(obs, model.project(native, v, 3.0, 0.0))
    coarse = np.arange(-150.0, 150.01, 1.0)
    v0 = coarse[int(np.argmin([chi(v) for v in coarse]))]
    fine = np.arange(v0 - 2.0, v0 + 2.01, 0.05)
    return float(fine[int(np.argmin([chi(v) for v in fine]))])


def _chi2(obs, flux):
    m = obs.mask
    design = flux[m, None] * obs.basis[m]
    w = np.sqrt(obs.weight[m])
    coefficients = np.linalg.lstsq(design * w[:, None], obs.flux[m] * w, rcond=None)[0]
    return float(np.sum(obs.weight[m] * (obs.flux[m] - design @ coefficients) ** 2))


def stage_reference(args):
    from fitter import FitConfiguration, NormalizedSpectrum, fit_normalized_spectrum

    out = OUT / "calibration" / args.star
    out.mkdir(parents=True, exist_ok=True)
    labels = gbs_labels(args.star)
    obs = Observation(args.star)
    model = OpticalModel(obs, bundle_for([], args.device, args.dtype), args.device, args.dtype, out, labels["vmic_km_s"])
    key = model.key(model.full([labels["teff"], labels["logg"], labels["m_h"], 0.0, 0.0]))
    started = time.time()
    profile, row = model.start(key)
    path, iteration = solve_stable(out / "solve", profile, row, json.loads(GATE.read_text())["thresholds"])
    report = dict(star=args.star, labels=labels, stable_iteration=iteration, solve_seconds=time.time() - started)
    if path is not None:
        native = model.synthesize(path)
        rv0 = rv_start(obs, model, [labels["teff"], labels["logg"], labels["m_h"]])
        obs.set_rest_frame_mask(rv0)
        spectrum = NormalizedSpectrum(wavelength=obs.wave_nm, flux=obs.flux, inverse_variance=obs.weight,
                                      mask=obs.mask)
        nuisance = fit_normalized_spectrum(spectrum, FitConfiguration(names=PARAMETERS[3:],
            initial=np.array([rv0, 3.0]), lower=np.array([rv0 - 5.0, 0.0]), upper=np.array([rv0 + 5.0, 10.0]),
            derivative_steps=np.array([0.1, 0.1]), trust_half_width=np.array([1.0, 1.5]), maximum_iterations=8),
            lambda p: model.project(native, p[0], p[1], 0.0), continuum_basis=obs.basis)
        report.update(endpoint=str(path.relative_to(ROOT)),
                      nuisance=dict(zip(PARAMETERS[3:], map(float, nuisance.parameters))),
                      chi2_per_pixel=float(nuisance.mean_weighted_squared_residual))
    (out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def stage_calibrate(args):
    references = [json.loads((OUT / "calibration" / k / "reference.json").read_text())
                  for k, role in STARS.items() if role == "calibration"]
    stars = []
    for r in references:
        obs = Observation(r["star"])
        obs.set_rest_frame_mask(r["nuisance"]["rv_km_s"])
        stars.append((r, obs))
    step, width, bound, groups = 0.05, 0.3, 1.0, len(TIO_BINS_A) - 1
    history = []

    def residuals(offsets):
        bundle = bundle_for(list(offsets), args.device, args.dtype)
        parts, total, pixels = [], 0.0, 0
        for r, obs in stars:
            model = OpticalModel(obs, bundle, args.device, args.dtype, OUT / "calibration/scratch", 2.0)
            flux = model.project(model.synthesize(ROOT / r["endpoint"]), r["nuisance"]["rv_km_s"],
                                 r["nuisance"]["sigma_extra_km_s"], 0.0)
            m = obs.mask
            design = flux[m, None] * obs.basis[m]
            w = np.sqrt(obs.weight[m])
            coefficients = np.linalg.lstsq(design * w[:, None], (obs.flux[m] - flux[m]) * w, rcond=None)[0]
            residual = (flux[m] + design @ coefficients - obs.flux[m]) * w
            parts.append(residual)
            total, pixels = total + float(residual @ residual), pixels + m.sum()
        return np.concatenate(parts), total / pixels

    offsets = np.zeros(groups)
    current, chi2 = residuals(offsets)
    history.append(dict(offsets=offsets.tolist(), chi2_per_pixel=chi2))
    for _ in range(args.max_iterations):
        jacobian = np.empty((current.size, groups))
        for j in range(groups):
            trial = offsets.copy()
            trial[j] += step
            jacobian[:, j] = (residuals(trial)[0] - current) / step
        delta = np.clip(np.linalg.lstsq(jacobian, -current, rcond=None)[0], -width, width)
        for scale in (1.0, 0.5, 0.25):
            trial = np.clip(offsets + scale * delta, -bound, bound)
            candidate, candidate_chi2 = residuals(trial)
            if candidate_chi2 < chi2:
                break
        else:
            break
        improvement = chi2 - candidate_chi2
        offsets, current, chi2 = trial, candidate, candidate_chi2
        history.append(dict(offsets=offsets.tolist(), chi2_per_pixel=chi2))
        print(json.dumps(history[-1]), flush=True)
        if improvement < 1e-3 * chi2:
            break
    (OUT / "calibration/offsets.json").write_text(json.dumps(dict(offsets=offsets.tolist(),
        bins_air_A=TIO_BINS_A.tolist(), stars=[r["star"] for r, _ in stars], history=history,
        derivative_step_dex=step, trust_half_width_dex=width, bound_dex=bound), indent=2) + "\n")


def stage_fit(args):
    from fitter import (FitConfiguration, NormalizedSpectrum, PhysicalAtmosphereConfiguration,
                        fit_normalized_spectrum, refine_with_physical_atmosphere)

    offsets = [] if args.offsets is None else json.loads(Path(args.offsets).read_text())["offsets"]
    out = OUT / ("default" if args.offsets is None else "calibrated") / args.star
    out.mkdir(parents=True, exist_ok=True)
    labels = gbs_labels(args.star)
    obs = Observation(args.star)
    model = OpticalModel(obs, bundle_for(offsets, args.device, args.dtype), args.device, args.dtype, out,
                         labels["vmic_km_s"])
    start = [START["teff"], START["logg"], START["m_h"]]
    started = time.time()
    rv0 = rv_start(obs, model, start)
    obs.set_rest_frame_mask(rv0)
    spectrum = NormalizedSpectrum(wavelength=obs.wave_nm, flux=obs.flux, inverse_variance=obs.weight, mask=obs.mask)
    lower = np.array([BOUNDS["teff"][0], BOUNDS["logg"][0], BOUNDS["m_h"][0], rv0 - 5.0, 0.0])
    upper = np.array([BOUNDS["teff"][1], BOUNDS["logg"][1], BOUNDS["m_h"][1], rv0 + 5.0, 10.0])
    configuration = FitConfiguration(names=PARAMETERS, initial=np.array([*start, rv0, 3.0]), lower=lower,
        upper=upper, derivative_steps=np.array([20.0, 0.05, 0.05, 0.1, 0.1]),
        trust_half_width=np.array([150.0, 0.4, 0.3, 1.0, 1.5]), maximum_iterations=args.max_iterations)
    fast = fit_normalized_spectrum(spectrum, configuration, model, continuum_basis=obs.basis)
    fast.save(out / "fast")
    summary = dict(star=args.star, role=STARS[args.star], benchmark=labels, offsets=offsets,
        fast=dict(parameters=dict(zip(PARAMETERS, map(float, fast.parameters))),
                  chi2_per_pixel=float(fast.mean_weighted_squared_residual), stop_reason=fast.stop_reason,
                  syntheses=model.syntheses, seconds=time.time() - started))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary["fast"], default=str), flush=True)
    solved = OpticalSolved(model, out / "solves")
    started = time.time()
    try:
        physical = refine_with_physical_atmosphere(spectrum, configuration, fast, model, solved,
            PhysicalAtmosphereConfiguration(maximum_discrepancy_rms=2.0e-3, maximum_objective_degradation=0.1,
                minimum_predicted_objective_improvement=1.0e-3, maximum_physical_evaluations=args.max_physical),
            continuum_basis=obs.basis)
    except RuntimeError as error:
        summary["physical"] = dict(error=str(error), solves=solved.records, seconds=time.time() - started)
    else:
        physical.save(out / "physical")
        np.save(out / "physical_model_flux.npy", physical.model_flux)
        np.save(out / "observed.npy", np.stack([obs.wave_air, obs.flux, obs.weight, obs.mask]))
        summary["physical"] = dict(parameters=dict(zip(PARAMETERS, map(float, physical.parameters))),
            chi2_per_pixel=float(physical.mean_weighted_squared_residual), successful=physical.successful,
            stationary=physical.physical_fit_stationary, stop_reason=physical.stop_reason,
            solves=solved.records, seconds=time.time() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary["physical"], default=str), flush=True)


def stage_collect(args):
    """Fitted labels against the benchmark values for both line lists, and the TiO offsets."""
    rows = []
    for arm in ("default", "calibrated"):
        for star in STARS:
            fit = json.loads((OUT / arm / star / "summary.json").read_text())
            physical = fit["physical"]
            refined = "parameters" in physical
            p = physical["parameters"] if refined else fit["fast"]["parameters"]
            benchmark = gbs_labels(star)
            rows.append(dict(arm=arm, star=star, role=STARS[star], benchmark=benchmark, parameters=p,
                refined=refined, ends_at_cap=any(s["stable_iteration"] is None for s in physical["solves"]),
                chi2_per_pixel=physical.get("chi2_per_pixel", fit["fast"]["chi2_per_pixel"]),
                difference={k: p[k] - benchmark[k] for k in ("teff", "logg", "m_h")},
                on_bound=[k for k in ("teff", "logg", "m_h") if p[k] <= BOUNDS[k][0] + 1e-6 or p[k] >= BOUNDS[k][1] - 1e-6]))
    calibration = json.loads((OUT / "calibration/offsets.json").read_text())
    (OUT / "summary.json").write_text(json.dumps(dict(rows=rows, tio_offsets=calibration["offsets"],
        tio_bins_air_A=calibration["bins_air_A"], calibration_chi2=[h["chi2_per_pixel"] for h in calibration["history"]]),
        indent=2) + "\n")
    for r in rows:
        print(r["arm"], r["star"], {k: round(v, 2) for k, v in r["difference"].items()}, round(r["chi2_per_pixel"]),
              r["ends_at_cap"], r["on_bound"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("reference", "calibrate", "fit", "collect"))
    parser.add_argument("--star", choices=tuple(STARS))
    parser.add_argument("--offsets")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-physical", type=int, default=4)
    args = parser.parse_args()
    dict(reference=stage_reference, calibrate=stage_calibrate, fit=stage_fit, collect=stage_collect)[args.stage](args)


if __name__ == "__main__":
    main()
