"""Ten-parameter fits of APOGEE M giants with the final eight-label network (Sect. 5).

At each trial set of labels (Teff, logg, [M/H], [alpha/M], vmic, [C/M],
[N/M], [O/M]=[alpha/M]) the final six-profile network gives the fast-model
atmosphere, synthesized over 1510-1700 nm with the Sun/Arcturus-calibrated
atomic lines and the APOGEE DR17 molecular lines; the calibrated arm adds
per-group log gf offsets to the molecular lines, and the line-by-line arm per-line
corrections of the CN, CO and OH lines. Refinement solves the
atmosphere from the final two-profile network, rebuilt at the composition,
until it meets the stability rule of the M-giant test (cap 90). The data are divided
by the file continuum only; a per-chip cubic B-spline continuum is profiled
at every model evaluation.

Stages::

    sample                       draw the sample and its calibration/test split
    fit --star ID [--offsets F]  fast fit, then refinement with solved atmospheres
    reference --star ID          solved atmosphere at the raw ASPCAP labels of a calibration star
    calibrate                    molecular log gf offsets by group on the calibration stars
    collect                      per-arm table against ASPCAP, with the ASPCAP model chi2 on common pixels
"""
from bench import environment as _environment

import argparse
import dataclasses
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments import mgiant_apogee_pzfit_v1 as apogee  # noqa: E402
from experiments import mgiant_apogee_linelist_v1 as linelist  # noqa: E402
from bench.run_reference import _solver_config  # noqa: E402
from payne_zero_atmosphere.runner import run_atmosphere_model  # noqa: E402
from payne_zero_atmosphere.synthesis_bridge import save_product_structured_atmosphere  # noqa: E402
from coauthor_benchmark_20261005 import atmosphere  # noqa: E402
from coauthor_mgiant_holdout_test_20261012 import rebuilt, difference, NAMES_FLUX  # noqa: E402
from coauthor_mgiant_stability_20261007 import RULE  # noqa: E402
from coauthor_mgiant_round2_corpus_20261009 import NAMES  # noqa: E402
from coauthor_mgiant_train_20261009 import predict  # noqa: E402

OUT = ROOT / "results/coauthor_validation_20261005/mgiant_apogee_fit"
NETWORKS = ROOT / "results/coauthor_validation_20261005/mgiant_final"
GATE = ROOT / "results/m_star_emulator_v1r2_marcs100/flux_gate.json"
DR20 = ROOT / "data/dr20"
SEED = 20261014
PER_BIN, CALIBRATION_PER_BIN, CONTROL_PER_BIN = 100, (17, 17, 16), (7, 7, 6)
TEMPERATURE_BINS = ((3000, 3300), (3300, 3600), (3600, 4001))
SELECTION = dict(teff=(3000.0, 4000.0), logg=(0.5, 2.5), m_h=(-1.0, 0.5), good_pixel_fraction=0.9,
                 telescope="apo25m")
PARAMETERS = ("teff", "logg", "m_h", "alpha_m", "c_m", "n_m", "vmic_km_s", "rv_km_s", "sigma_extra_km_s",
              "vsini_km_s")
BOUNDS = dict(teff=(3000.0, 4000.0), logg=(0.5, 2.5), m_h=(-1.0, 0.5), alpha_m=(-0.2, 0.4), c_m=(-0.5, 0.3),
              n_m=(-0.3, 0.8), vmic_km_s=(1.0, 3.0), sigma_extra_km_s=(0.0, 10.0), vsini_km_s=(0.0, 15.0))
FIXED_START = dict(teff=3600.0, logg=1.0, m_h=-0.3, alpha_m=0.0, c_m=0.0, n_m=0.2, vmic_km_s=2.0)
KNOT_SPACING_NM = 10.0
SOLVE_CAP = 90
# Offset groups of the calibration: every line of a molecule, and the 13C isotopologues of CN and CO.
OFFSET_GROUPS = ("CN", "CO", "OH", "13C")


def stratum(teff):
    return next(i for i, (lo, hi) in enumerate(TEMPERATURE_BINS) if lo <= teff < hi)


def stage_sample(args):
    from astropy.io import fits

    pool = fits.open(DR20 / "mgiant_quality_pool.fits")[1].data
    qc = fits.open(DR20 / "mgiant_snr500_files_qc.fits")[1].data
    good = set(qc["sdss_id"][qc["good_pixel_frac"] >= SELECTION["good_pixel_fraction"]])
    ids = set(pool["sdss_id"][(pool["telescope"] == SELECTION["telescope"])]) & good
    table = fits.open(DR20 / "astraAllStarASPCAP-0.6.0.fits", memmap=True)[2].data
    keep = np.isin(table["sdss_id"], list(ids)) & (table["telescope"] == SELECTION["telescope"])
    rows = table[keep]
    raw = {k: rows[f"raw_{k}"] for k in ("teff", "logg", "m_h_atm", "alpha_m_atm", "c_m_atm", "n_m_atm", "v_micro")}
    finite = np.all([np.isfinite(v) for v in raw.values()], axis=0) & (rows["flag_bad"] == 0)
    inside = ((raw["teff"] >= SELECTION["teff"][0]) & (raw["teff"] < SELECTION["teff"][1])
              & (raw["logg"] >= SELECTION["logg"][0]) & (raw["logg"] <= SELECTION["logg"][1])
              & (raw["m_h_atm"] >= SELECTION["m_h"][0]) & (raw["m_h_atm"] <= SELECTION["m_h"][1]))
    rows = rows[finite & inside]
    _, first = np.unique(rows["sdss_id"], return_index=True)
    rows = rows[np.sort(first)]
    snr = dict(zip(pool["sdss_id"], pool["snr"]))
    rng = np.random.default_rng(SEED)
    sample = []
    for b, (lo, hi) in enumerate(TEMPERATURE_BINS):
        members = np.flatnonzero((rows["raw_teff"] >= lo) & (rows["raw_teff"] < hi))
        chosen = rng.permutation(members)[:PER_BIN]
        for j, i in enumerate(chosen):
            r = rows[i]
            role = ("calibration" if j < CALIBRATION_PER_BIN[b] else
                    "control" if j < CALIBRATION_PER_BIN[b] + CONTROL_PER_BIN[b] else "test")
            sample.append(dict(sdss_id=int(r["sdss_id"]), role=role, temperature_bin=f"{lo}-{min(hi, 4000)}",
                snr=float(snr[r["sdss_id"]]), v_rad=float(r["v_rad"]),
                raw={k: float(r[f"raw_{k}"]) for k in ("teff", "logg", "m_h_atm", "alpha_m_atm", "c_m_atm",
                                                       "n_m_atm", "v_micro", "c_12_13")},
                calibrated={k: float(r[k]) for k in ("teff", "logg", "m_h_atm", "alpha_m_atm", "c_m_atm",
                                                      "n_m_atm", "v_micro", "c_12_13")}))
        print(f"bin {lo}-{hi}: {members.size} eligible, {chosen.size} drawn", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sample.json").write_text(json.dumps(dict(seed=SEED, selection=SELECTION, eligible=int(len(rows)),
        per_bin=PER_BIN, calibration_per_bin=CALIBRATION_PER_BIN, control_per_bin=CONTROL_PER_BIN,
        note="control stars are test stars that are also fitted from the fixed start",
        stars=sample), indent=2) + "\n")


def star_record(sdss_id):
    return next(s for s in json.loads((OUT / "sample.json").read_text())["stars"] if s["sdss_id"] == sdss_id)


class Observation(apogee.ApogeeObservation):
    """mwmStar coadd divided by its file continuum, with a per-chip cubic B-spline basis."""

    def __init__(self, sdss_id):
        from astropy.io import fits
        from scipy.interpolate import BSpline

        self.path = (apogee.SPECTRA_DIR / f"{(sdss_id // 100) % 100:02d}" / f"{sdss_id % 100:02d}"
                     / f"mwmStar-0.6.0-{sdss_id}.fits")
        with fits.open(self.path) as hdu_list:
            hdu = hdu_list[3] if hdu_list[3].data.size else hdu_list[4]
            row = hdu.data[0]
            wave_A = np.asarray(row["wavelength"], np.float64)
            continuum = np.asarray(row["continuum"], np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                flux = np.asarray(row["flux"], np.float64) / continuum
                ivar = np.asarray(row["ivar"], np.float64) * continuum ** 2
        good = (ivar > 0) & np.isfinite(flux) & np.isfinite(ivar)
        self.wave_vac, self.wave_nm = wave_A, wave_A / 10.0
        self.flux = np.where(good, flux, 1.0)
        self.weight = np.where(good, ivar, 0.0)
        columns, in_chip = [], np.zeros(self.wave_nm.size, bool)
        for lo, hi in apogee.CHIPS_NM:
            chip = (self.wave_nm >= lo) & (self.wave_nm <= hi)
            in_chip |= chip
            interior = np.linspace(lo, hi, max(1, int(round((hi - lo) / KNOT_SPACING_NM))) + 1)
            knots = np.concatenate([[lo] * 3, interior, [hi] * 3])
            design = BSpline.design_matrix(np.clip(self.wave_nm, lo, hi), knots, 3).toarray()
            columns.append(design * chip[:, None])
        self.basis = np.concatenate(columns, axis=1)
        self.mask = good & in_chip


# APOGEE CN, CO and OH lines, the lines of the line-by-line calibration.
LINE_BY_LINE_SPECIES = (270, 276, 258)


def molecular_table(offsets, line_offsets=None):
    """Native and APOGEE molecular arrays with the per-group offsets and optional per-line corrections.

    Returns the merged arrays and the indices of the APOGEE CN, CO and OH lines in them.
    """
    from payne_zero_synthesis import pipeline as pz

    molecules = {code: (name, species, fraction + offsets.get(name, 0.0)
                        + (offsets.get("13C", 0.0) if ".013" in code else 0.0))
                 for code, (name, species, fraction) in linelist.MOLECULES.items()}
    native = pz.SynthesisPipeline._compile_molecular(apogee.SYNTH_VAC_NM[0], apogee.SYNTH_VAC_NM[1], apogee.R_GRID)
    merged, inventory = linelist.apogee_molecular_compiled(native, molecules)
    start = inventory["native_lines_kept"]
    selected = start + np.flatnonzero(np.isin(merged["species_code"][start:], LINE_BY_LINE_SPECIES))
    if line_offsets is not None:
        with np.load(ROOT / line_offsets) as d:
            for key in ("center_index_1based", "species_code", "lower_excitation_cm"):
                if not np.array_equal(d[key], merged[key][selected]):
                    raise ValueError(f"{line_offsets} does not match the compiled lines in {key}")
            strength = merged["classical_line_strength"].astype(np.float64)
            strength[selected] *= 10.0 ** d["corrections_dex"]
            merged["classical_line_strength"] = strength.astype(merged["classical_line_strength"].dtype)
    return merged, selected


def bundle_for(offsets, device, dtype, scratch, line_offsets=None):
    """Sun/Arcturus atomic lines with the APOGEE DR17 molecular lines, shifted in log gf by group and by line."""
    base, _ = apogee.calibrated_bundle(linelist.CALIBRATION, device=device, dtype=dtype)
    merged, _ = molecular_table(offsets, line_offsets)
    invariants, count = linelist._molecular_invariants(base, merged, device)
    return dataclasses.replace(base, molecular_invariants=invariants, n_molecular=count)


def label_row(teff, logg, m_h, alpha_m, c_m, n_m, vmic):
    row = dict(effective_temperature=float(teff), log_surface_gravity=float(logg), metallicity=float(m_h),
               alpha_enhancement=float(alpha_m), microturbulence_km_s=float(vmic), carbon_enhancement=float(c_m),
               nitrogen_enhancement=float(n_m), oxygen_enhancement=float(alpha_m))
    row["absolute_abundance_offsets"] = {z: row["metallicity"] + row[f] for z, f in zip((6, 7, 8), NAMES[-3:])}
    return row


class NetworkModel:
    """Six-profile network atmosphere, synthesized and projected onto the APOGEE pixels."""

    window_nm, r_grid, instrument_r = apogee.SYNTH_VAC_NM, apogee.R_GRID, apogee.INSTRUMENT_R

    def __init__(self, obs, bundle, device, dtype, directory):
        self.obs, self.bundle, self.device, self.dtype = obs, bundle, device, dtype
        self.networks = {n: torch.load(NETWORKS / f"checkpoint_{n}profile.pt", map_location="cpu", weights_only=False)
                         for n in (2, 6)}
        self.atmospheres = directory / "atmospheres"
        self.atmospheres.mkdir(parents=True, exist_ok=True)
        self._native, self._operator, self._rotation = {}, None, None
        self.evaluations = self.syntheses = 0

    @staticmethod
    def key(parameters):
        teff, logg, m_h, alpha_m, c_m, n_m, vmic = parameters[:7]
        return (round(float(teff)), *(round(float(x), 4) for x in (logg, m_h, alpha_m, c_m, n_m)),
                round(float(vmic), 3))

    def profiles(self, key, n):
        row = label_row(*key)
        return predict(self.networks[n], np.array([[row[f] for f in NAMES]]))[0], row

    def start(self, key):
        """Solver start: the two-profile network rebuilt at the composition, and its label row."""
        profile, row = self.profiles(key, 2)
        return rebuilt(profile, row)[0], row

    def synthesize(self, path):
        from payne_zero_synthesis.api import _surface_flux_per_wavelength_nm
        from payne_zero_synthesis.atmosphere import load_atmosphere_npz
        from payne_zero_synthesis.synthesis import synthesize_structured_atmosphere

        result, _ = synthesize_structured_atmosphere(
            load_atmosphere_npz(path), wavelength_start_nm=self.window_nm[0],
            wavelength_end_nm=self.window_nm[1], resolution=self.r_grid, molecular_lines=True,
            device=torch.device(self.device), dtype=getattr(torch, self.dtype), window_invariants=self.bundle)
        wavelength = np.asarray(result.wavelength_nm, np.float64)
        self.syntheses += 1
        return (wavelength, _surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_total_per_frequency),
                _surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_continuum_per_frequency))

    def native(self, key):
        if key not in self._native:
            path = self.atmospheres / ("t{}_g{:+.4f}_m{:+.4f}_a{:+.4f}_c{:+.4f}_n{:+.4f}_v{:.3f}.npz".format(*key))
            if not path.is_file():
                profile, row = self.profiles(key, 6)
                save_product_structured_atmosphere(atmosphere(profile, row), path, device="cpu", dtype="float64")
            self._native[key] = self.synthesize(path)
        return self._native[key]

    def project(self, native, rv, sigma, vsini):
        from fitter import ObservedSpectrumOperator, RotationalBroadening

        wavelength, total, continuum = native
        if self._rotation is None or not np.array_equal(self._rotation.wavelength_nm, wavelength):
            self._rotation = RotationalBroadening(wavelength, maximum_vsini_km_s=20.0, device="cpu",
                                                  dtype=torch.float64)
            self._operator = ObservedSpectrumOperator(wavelength, self.obs.wave_nm, resolving_power=self.instrument_r,
                                                      device="cpu", dtype=torch.float64)
        vsini = max(float(vsini), 0.0)
        total = self._rotation(torch.as_tensor(total), vsini_km_s=vsini)
        continuum = self._rotation(torch.as_tensor(continuum), vsini_km_s=vsini)
        self._operator.set_parameters(radial_velocity_km_s=float(rv), broadening_sigma_km_s=float(abs(sigma)))
        _, _, normalized = self._operator.convolve_fluxes(total, continuum)
        return normalized.detach().cpu().numpy()

    def __call__(self, parameters):
        self.evaluations += 1
        return self.project(self.native(self.key(parameters)), *parameters[7:])


class _Stable(Exception):
    pass


def tio(path):
    from payne_zero_synthesis import build_structured_atmosphere, synthesize

    with np.load(path) as data:
        columns = build_structured_atmosphere(temperature=data["temperature"], column_mass=data["column_mass"],
            gas_pressure=data["gas_pressure"], electron_density=data["electron_density"],
            elemental_abundances=data["elemental_abundances"], microturbulence=data["microturbulence"],
            molecular_lines=True, device="cpu", dtype="float64")
        mass, temperature = data["column_mass"].copy(), data["temperature"].copy()
    spectrum = synthesize(columns, wavelength_start_nm=RULE["wavelength_nm"][0], wavelength_end_nm=RULE["wavelength_nm"][1],
                          resolution=RULE["resolution"], molecular_lines=True, device="cpu", dtype="float64")
    return tuple(np.asarray(a) for a in (spectrum.normalized_flux, spectrum.flux_total, spectrum.flux_continuum)), mass, temperature


def solve_stable(directory, start, row, gate, cap=SOLVE_CAP):
    """Solve from a start until two consecutive five-iteration segments meet the stability rule."""
    directory.mkdir(parents=True, exist_ok=True)
    seed = atmosphere(start, row)
    save_product_structured_atmosphere(seed, directory / "iter_0000.npz", device="cpu", dtype="float64")
    arrays, mass, temperature = tio(directory / "iter_0000.npz")
    state = dict(previous=(0, mass, temperature, arrays), consecutive=0, flux={}, stable=None)

    def hook(n, setup, step):
        flux = np.abs(step.remapped.finalization.temperature_correction_result.flux_error_percent)
        state["flux"][n] = dict(flux_error_median_percent=float(np.median(flux)),
                                flux_error_p95_percent=float(np.percentile(flux, 95)),
                                flux_error_max_percent=float(np.max(flux)))
        if n % RULE["segment_iterations"]:
            return None
        path = directory / f"iter_{n:04d}.npz"
        save_product_structured_atmosphere(step.remapped.atmosphere, path, device="cpu", dtype="float64")
        arrays, mass, temperature = tio(path)
        pn, pm, pt, ps = state["previous"]
        spectral = difference(arrays, ps)
        flux_pass = all(all(state["flux"][k][a] <= gate[b] for a, b in NAMES_FLUX.items()) for k in range(pn + 1, n + 1))
        stable = (max(spectral.values()) < RULE["spectral_change"]
                  and float(np.percentile(abs(temperature / pt - 1), 95)) < RULE["temperature_p95"]
                  and float(np.percentile(abs(np.log10(mass / pm)), 95)) < RULE["mass_dex_p95"] and flux_pass)
        state["consecutive"] = state["consecutive"] + 1 if stable else 0
        state["previous"] = (n, mass, temperature, arrays)
        if state["consecutive"] >= RULE["consecutive_segments"]:
            state["stable"] = n
            raise _Stable
        return None

    config = dataclasses.replace(_solver_config(seed, iterations_per_trial=cap, structured_atmosphere_path=None,
                                                debug_state_path=None), enable_convergence_stop=False)
    try:
        run_atmosphere_model(config, after_iteration_hook=hook)
    except _Stable:
        return directory / f"iter_{state['stable']:04d}.npz", state["stable"]
    return None, None


class SolvedModel:
    """Atmosphere solved from the network start at the trial labels, then synthesized like the fast model."""

    def __init__(self, fast, directory):
        self.fast, self.directory = fast, directory
        self.gate = json.loads(GATE.read_text())["thresholds"]
        self.records = []

    def __call__(self, parameters):
        key = self.fast.key(parameters)
        started = time.time()
        profile, row = self.fast.start(key)
        path, iteration = solve_stable(self.directory / f"solve_{len(self.records):02d}", profile, row, self.gate)
        record = dict(parameters=dict(zip(PARAMETERS, map(float, parameters))), stable_iteration=iteration,
                      seconds=time.time() - started)
        self.records.append(record)
        if path is None:
            raise RuntimeError(f"solve not stable within {SOLVE_CAP} iterations at {record['parameters']}")
        return self.fast.project(self.fast.synthesize(path), *parameters[7:])


def rv_start(obs, model, labels):
    native = model.native(model.key(labels))
    chi = lambda v: apogee._profiled_chi2(obs, model.project(native, v, apogee.SIGMA_START_KM_S, 3.0))
    coarse = np.arange(-30.0, 30.01, 1.0)
    v0 = coarse[int(np.argmin([chi(v) for v in coarse]))]
    fine = np.arange(v0 - 1.0, v0 + 1.01, 0.1)
    return float(fine[int(np.argmin([chi(v) for v in fine]))])


def arm_name(args):
    stem = "uncalibrated" if args.offsets is None else Path(args.offsets).stem
    return f"{stem}_{args.start}"


def stage_fit(args):
    from fitter import (FitConfiguration, NormalizedSpectrum, PhysicalAtmosphereConfiguration,
                        fit_normalized_spectrum, refine_with_physical_atmosphere)

    star = star_record(args.star)
    record = {} if args.offsets is None else json.loads(Path(args.offsets).read_text())
    offsets = record.get("offsets", {})
    out = OUT / arm_name(args) / str(args.star)
    out.mkdir(parents=True, exist_ok=True)
    obs = Observation(args.star)
    bundle = bundle_for(offsets, args.device, args.dtype, out / "scratch", record.get("line_offsets"))
    model = NetworkModel(obs, bundle, args.device, args.dtype, out)
    aspcap = star["raw"]
    first = (dict(teff=aspcap["teff"], logg=aspcap["logg"], m_h=aspcap["m_h_atm"], alpha_m=aspcap["alpha_m_atm"],
                  c_m=aspcap["c_m_atm"], n_m=aspcap["n_m_atm"], vmic_km_s=aspcap["v_micro"])
             if args.start == "aspcap" else dict(FIXED_START))
    lower = np.array([BOUNDS[p][0] if p in BOUNDS else -np.inf for p in PARAMETERS])
    upper = np.array([BOUNDS[p][1] if p in BOUNDS else np.inf for p in PARAMETERS])
    labels = np.clip([first[p] for p in PARAMETERS[:7]], lower[:7], upper[:7])
    started = time.time()
    rv0 = rv_start(obs, model, labels)
    initial = np.array([*labels, rv0, apogee.SIGMA_START_KM_S, 3.0])
    lower[7], upper[7] = rv0 - 5.0, rv0 + 5.0
    spectrum = NormalizedSpectrum(wavelength=obs.wave_nm, flux=obs.flux, inverse_variance=obs.weight, mask=obs.mask)
    configuration = FitConfiguration(names=PARAMETERS, initial=initial, lower=lower, upper=upper,
        derivative_steps=np.array([20.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.5]),
        trust_half_width=np.array([150.0, 0.4, 0.3, 0.2, 0.2, 0.3, 0.5, 1.0, 1.5, 3.0]),
        maximum_iterations=args.max_iterations)
    fast = fit_normalized_spectrum(spectrum, configuration, model, continuum_basis=obs.basis)
    fast.save(out / "fast")
    faces = lambda p: [n for n, v, lo, hi in zip(PARAMETERS, p, lower, upper) if v <= lo + 1e-6 or v >= hi - 1e-6]
    summary = dict(sdss_id=args.star, role=star["role"], start=args.start, offsets=offsets,
        start_parameters=dict(zip(PARAMETERS, map(float, initial))), n_pixels=int(obs.mask.sum()),
        continuum=dict(kind="per-chip cubic B-spline", knot_spacing_nm=KNOT_SPACING_NM, coefficients=obs.basis.shape[1]),
        fast=dict(parameters=dict(zip(PARAMETERS, map(float, fast.parameters))), faces=faces(fast.parameters),
                  chi2_per_pixel=float(fast.mean_weighted_squared_residual),
                  start_chi2_per_pixel=float(fast.objective_path[0]), stop_reason=fast.stop_reason,
                  evaluations=model.evaluations, syntheses=model.syntheses, seconds=time.time() - started))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(dict(star=args.star, fast=summary["fast"]), default=str), flush=True)

    solved = SolvedModel(model, out / "solves")
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
        summary["physical"] = dict(parameters=dict(zip(PARAMETERS, map(float, physical.parameters))),
            faces=faces(physical.parameters), chi2_per_pixel=float(physical.mean_weighted_squared_residual),
            successful=physical.successful, stationary=physical.physical_fit_stationary,
            gates_passed=physical.fast_physical_gates_passed, stop_reason=physical.stop_reason,
            solves=solved.records, seconds=time.time() - started)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(dict(star=args.star, physical=summary["physical"]), default=str), flush=True)


def stage_reference(args):
    """Solved atmosphere at the raw ASPCAP labels of a calibration star, with RV and broadening fitted."""
    from fitter import FitConfiguration, NormalizedSpectrum, fit_normalized_spectrum

    star = star_record(args.star)
    out = OUT / "calibration" / str(args.star)
    out.mkdir(parents=True, exist_ok=True)
    obs = Observation(args.star)
    model = NetworkModel(obs, bundle_for({}, args.device, args.dtype, out / "scratch"), args.device, args.dtype, out)
    raw = star["raw"]
    labels = np.clip([raw["teff"], raw["logg"], raw["m_h_atm"], raw["alpha_m_atm"], raw["c_m_atm"], raw["n_m_atm"],
                      raw["v_micro"]], [BOUNDS[p][0] for p in PARAMETERS[:7]], [BOUNDS[p][1] for p in PARAMETERS[:7]])
    key = model.key(labels)
    started = time.time()
    profile, row = model.start(key)
    path, iteration = solve_stable(out / "solve", profile, row, json.loads(GATE.read_text())["thresholds"])
    report = dict(sdss_id=args.star, labels=dict(zip(PARAMETERS[:7], map(float, key))), stable_iteration=iteration,
                  solve_seconds=time.time() - started)
    if path is not None:
        native = model.synthesize(path)
        rv0 = rv_start(obs, model, labels)
        spectrum = NormalizedSpectrum(wavelength=obs.wave_nm, flux=obs.flux, inverse_variance=obs.weight,
                                      mask=obs.mask)
        nuisance = fit_normalized_spectrum(spectrum, FitConfiguration(names=PARAMETERS[7:],
            initial=np.array([rv0, apogee.SIGMA_START_KM_S, 3.0]), lower=np.array([rv0 - 5.0, 0.0, 0.0]),
            upper=np.array([rv0 + 5.0, 10.0, 15.0]), derivative_steps=np.array([0.1, 0.1, 0.5]),
            trust_half_width=np.array([1.0, 1.5, 3.0]), maximum_iterations=8),
            lambda p: model.project(native, *p), continuum_basis=obs.basis)
        report.update(endpoint=str(path.relative_to(ROOT)), nuisance=dict(zip(PARAMETERS[7:], map(float, nuisance.parameters))),
                      chi2_per_pixel=float(nuisance.mean_weighted_squared_residual))
    (out / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def stage_calibrate(args):
    """Molecular log gf offsets by group that minimize the total chi2 of the calibration stars."""
    references = [json.loads(p.read_text()) for p in sorted((OUT / "calibration").glob("*/reference.json"))]
    references = [r for r in references if r.get("endpoint")]
    stars = [(r, Observation(r["sdss_id"])) for r in references]
    step, width, bound = 0.05, 0.3, 1.0
    history = []

    def residuals(offsets):
        bundle = bundle_for(dict(zip(OFFSET_GROUPS, offsets)), args.device, args.dtype, OUT / "calibration/scratch")
        parts, total, pixels = [], 0.0, 0
        for r, obs in stars:
            model = NetworkModel(obs, bundle, args.device, args.dtype, OUT / "calibration/scratch")
            flux = model.project(model.synthesize(ROOT / r["endpoint"]), *[r["nuisance"][p] for p in PARAMETERS[7:]])
            m = obs.mask
            design = flux[m, None] * obs.basis[m]
            w = np.sqrt(obs.weight[m])
            coefficients = np.linalg.lstsq((design * w[:, None]), (obs.flux[m] - flux[m]) * w, rcond=None)[0]
            residual = (flux[m] + design @ coefficients - obs.flux[m]) * w
            parts.append(residual)
            total, pixels = total + float(residual @ residual), pixels + m.sum()
        return np.concatenate(parts), total / pixels

    offsets = np.zeros(len(OFFSET_GROUPS))
    current, chi2 = residuals(offsets)
    history.append(dict(offsets=offsets.tolist(), chi2_per_pixel=chi2))
    for _ in range(args.max_iterations):
        jacobian = np.empty((current.size, offsets.size))
        for j in range(offsets.size):
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
    (OUT / "calibration/offsets.json").write_text(json.dumps(dict(offsets=dict(zip(OFFSET_GROUPS, offsets.tolist())),
        groups=OFFSET_GROUPS, stars=[r["sdss_id"] for r, _ in stars], history=history,
        derivative_step_dex=step, trust_half_width_dex=width, bound_dex=bound), indent=2) + "\n")


def profiled_chi2(obs, model, pixels):
    design = model[pixels, None] * obs.basis[pixels]
    w = np.sqrt(obs.weight[pixels])
    coefficients = np.linalg.lstsq(design * w[:, None], obs.flux[pixels] * w, rcond=None)[0]
    return float(np.mean(obs.weight[pixels] * (obs.flux[pixels] - design @ coefficients) ** 2))


def stage_collect(args):
    """Per-arm table of the fits against ASPCAP, and the chi2 of the ASPCAP model on common pixels."""
    sample = {s["sdss_id"]: s for s in json.loads((OUT / "sample.json").read_text())["stars"]}
    arms = {}
    for arm_dir in sorted(p for p in OUT.iterdir() if p.is_dir() and (p.name.endswith("_aspcap") or p.name.endswith("_fixed"))):
        rows = []
        for path in sorted(arm_dir.glob("*/summary.json")):
            fit = json.loads(path.read_text())
            if "physical" not in fit:
                continue
            star, physical = sample[fit["sdss_id"]], fit["physical"]
            refined = "parameters" in physical
            parameters = physical["parameters"] if refined else fit["fast"]["parameters"]
            row = dict(sdss_id=fit["sdss_id"], role=star["role"], temperature_bin=star["temperature_bin"], snr=star["snr"],
                       parameters=parameters, refined=refined,
                       ends_at_cap=any(s["stable_iteration"] is None for s in physical["solves"]),
                       faces=physical.get("faces", fit["fast"]["faces"]),
                       chi2_per_pixel=physical.get("chi2_per_pixel", fit["fast"]["chi2_per_pixel"]),
                       solves=len(physical["solves"]), syntheses=fit["fast"]["syntheses"],
                       continuum_coefficients=fit["continuum"]["coefficients"],
                       solve_seconds=[x["seconds"] for x in physical["solves"] if x["stable_iteration"]],
                       solve_iterations=[x["stable_iteration"] for x in physical["solves"] if x["stable_iteration"]],
                       seconds_per_synthesis=fit["fast"]["seconds"] / fit["fast"]["syntheses"],
                       seconds=fit["fast"]["seconds"] + physical["seconds"],
                       raw=star["raw"], calibrated=star["calibrated"])
            astra = apogee.read_astra_aspcap(fit["sdss_id"])
            model_path = path.parent / "physical_model_flux.npy"
            if astra is not None and model_path.is_file():
                obs = Observation(fit["sdss_id"])
                pixels = obs.mask & np.isfinite(astra["model_flux"]) & (astra["model_flux"] > 0)
                row["common_pixels"] = int(pixels.sum())
                row["chi2_common"] = profiled_chi2(obs, np.load(model_path), pixels)
                row["chi2_aspcap_model"] = profiled_chi2(obs, astra["model_flux"], pixels)
            rows.append(row)
        arms[arm_dir.name] = rows
    pairs = dict(teff="teff", logg="logg", m_h="m_h_atm", alpha_m="alpha_m_atm", c_m="c_m_atm", n_m="n_m_atm",
                 vmic_km_s="v_micro")
    stats = {}
    for arm, rows in arms.items():
        entry = dict(fits=len(rows), refined=sum(r["refined"] for r in rows), ends_at_cap=sum(r["ends_at_cap"] for r in rows),
                     on_bound=sum(any(f in BOUNDS and f not in ("vsini_km_s",) for f in r["faces"]) for r in rows))
        for role in ("test", "calibration"):
            subset = [r for r in rows if (r["role"] in ("test", "control")) == (role == "test")]
            if not subset:
                continue
            entry[role] = {}
            for p, k in pairs.items():
                d = np.array([r["parameters"][p] - r["raw"][k] for r in subset])
                entry[role][f"{p}_raw"] = dict(mean=float(d.mean()), rms=float(np.sqrt(np.mean(d ** 2))), std=float(d.std()))
            for p, k in (("teff", "teff"), ("logg", "logg"), ("m_h", "m_h_atm")):
                d = np.array([r["parameters"][p] - r["calibrated"][k] for r in subset])
                entry[role][f"{p}_calibrated"] = dict(mean=float(d.mean()), rms=float(np.sqrt(np.mean(d ** 2))))
            entry[role]["chi2_median"] = float(np.median([r["chi2_per_pixel"] for r in subset]))
            common = [r for r in subset if "chi2_common" in r]
            if common:
                entry[role]["chi2_common_median"] = float(np.median([r["chi2_common"] for r in common]))
                entry[role]["chi2_aspcap_model_median"] = float(np.median([r["chi2_aspcap_model"] for r in common]))
                entry[role]["common_stars"] = len(common)
        stats[arm] = entry
    (OUT / "summary.json").write_text(json.dumps(dict(stats=stats, arms=arms), indent=2) + "\n")
    print(json.dumps(stats, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sample", "fit", "reference", "calibrate", "collect"))
    parser.add_argument("--star", type=int)
    parser.add_argument("--offsets", help="JSON with the calibrated log gf offsets by group")
    parser.add_argument("--start", choices=("aspcap", "fixed"), default="aspcap")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--max-physical", type=int, default=4)
    args = parser.parse_args()
    dict(sample=stage_sample, fit=stage_fit, reference=stage_reference, calibrate=stage_calibrate,
         collect=stage_collect)[args.stage](args)


if __name__ == "__main__":
    main()
