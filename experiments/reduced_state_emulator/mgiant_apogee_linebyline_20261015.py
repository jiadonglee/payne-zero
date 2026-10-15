"""Line-by-line log gf calibration of the APOGEE CN, CO and OH lines (Sect. 5).

Each calibration star keeps the solved atmosphere, radial velocity and
broadening of its reference fit (``coauthor_mgiant_apogee_fit_20261013.py
reference``). The opacity of every line except the selected molecular lines
is evaluated once with the per-group calibrated list; each evaluation adds the
selected lines with additive log gf corrections, solves the transfer, applies
the rotation and instrument operators of the fit and profiles the per-chip
B-spline continuum by weighted least squares. The gradient with respect to
every correction comes from automatic differentiation.

The objective is the summed chi^2 of the calibration stars, divided by the
median chi^2 per pixel of these stars at zero correction, plus a Gaussian
prior of width ``PRIOR_DEX`` on every correction; corrections are bounded by
``BOUND_DEX``. The result is a per-line overlay on top of the per-group
offsets, written as an offsets file that the ``fit`` stage of the APOGEE
script reads.

Stages::

    check --star ID   the differentiable spectrum at zero correction against the fit's own model
    calibrate         L-BFGS over the corrections of every selected line
"""
from bench import environment as _environment

import argparse
import dataclasses
import json
from pathlib import Path
import time

import numpy as np
import torch

from coauthor_mgiant_apogee_fit_20261013 import (
    LINE_BY_LINE_SPECIES, OUT, ROOT, PARAMETERS, NetworkModel, Observation, apogee, bundle_for, molecular_table,
)

PRIOR_DEX = 0.25
BOUND_DEX = 1.0
MAX_ITERATIONS = 50
TOLERANCE = 1e-4
CALIBRATION_DIR = OUT / "linebyline"


class StarSpectrum:
    """Differentiable normalized model flux on the observed grid for one fixed atmosphere."""

    def __init__(self, reference, offsets, device, dtype):
        from payne_zero_synthesis import line_opacity, molecular_lines
        from payne_zero_synthesis.pipeline import SynthesisPipeline, _SYNTHESIS_TABLE_DIR
        from fitter import ObservedSpectrumOperator, RotationalBroadening

        self.device, self.dtype = torch.device(device), getattr(torch, dtype)
        self.obs = Observation(reference["sdss_id"])
        bundle = bundle_for(offsets, device, dtype, CALIBRATION_DIR / "scratch")
        self.pipeline = SynthesisPipeline(ROOT / reference["endpoint"], wl_start_nm=apogee.SYNTH_VAC_NM[0],
            wl_end_nm=apogee.SYNTH_VAC_NM[1], resolution=apogee.R_GRID, molecular_lines=True,
            device=self.device, dtype=self.dtype, window_invariants=bundle)
        baseline = self.pipeline.run(keep_slabs=True)
        self.wavelength = np.asarray(self.pipeline.wavelength_nm, np.float64)
        catalog = molecular_lines.build_catalog_from_arrays(selected_lines(offsets))
        catalog.center_index = line_opacity.nearest_grid_indices(self.wavelength, catalog.wavelength_nm).astype(np.int32)
        tables = np.load(_SYNTHESIS_TABLE_DIR / "line_profile_tables.npz", allow_pickle=False)
        self.invariants = molecular_lines.precompute_invariants(catalog=catalog, wavelength_grid_nm=self.wavelength,
            harris_profile_h0_table=tables["harris_profile_h0_table"],
            harris_profile_h1_table=tables["harris_profile_h1_table"],
            harris_profile_h2_table=tables["harris_profile_h2_table"], runtime_device=self.device)
        self.strength = self.invariants.classical_line_strength.detach().clone()
        tensor = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)
        self.continuum_absorption = tensor(baseline.continuum_absorption)
        self.continuum_scattering = tensor(baseline.continuum_scattering)
        self.state = dict(partition_normalized_populations=self.pipeline._partition_normalized_populations,
            mass_density=self.pipeline._mass_density, electron_density=self.pipeline._electron_density,
            temperature=self.pipeline._temperature, hc_over_kt=self.pipeline._hc_over_kt,
            microturbulence=self.pipeline._microturbulence,
            collision_density_proxy=self.pipeline._collision_density_proxy,
            continuum_opacity=self.continuum_absorption + self.continuum_scattering)
        with torch.no_grad():
            selected_baseline = self.selected_opacity(torch.zeros(len(catalog), dtype=self.dtype, device=self.device))
        self.fixed = tensor(baseline.line_mass_absorption_coefficient) - selected_baseline
        self.line_source = tensor(baseline.line_source)
        self.line_scattering = torch.zeros_like(self.line_source)
        self.per_wavelength = tensor(apogee_surface_factor(self.wavelength))
        self.rotation = RotationalBroadening(self.wavelength, maximum_vsini_km_s=20.0, device=self.device, dtype=self.dtype)
        self.operator = ObservedSpectrumOperator(self.wavelength, self.obs.wave_nm, resolving_power=apogee.INSTRUMENT_R,
                                                 device=self.device, dtype=self.dtype)
        nuisance = reference["nuisance"]
        self.vsini = max(float(nuisance["vsini_km_s"]), 0.0)
        self.operator.set_parameters(radial_velocity_km_s=float(nuisance["rv_km_s"]),
                                     broadening_sigma_km_s=float(abs(nuisance["sigma_extra_km_s"])))
        mask = self.obs.mask
        self.pixels = torch.as_tensor(np.flatnonzero(mask), device=self.device)
        self.flux = tensor(self.obs.flux[mask])
        self.root_weight = tensor(np.sqrt(self.obs.weight[mask]))
        self.basis = tensor(self.obs.basis[mask])

    def selected_opacity(self, corrections):
        from payne_zero_synthesis import molecular_lines
        invariants = dataclasses.replace(self.invariants, classical_line_strength=self.strength * 10.0 ** corrections)
        return molecular_lines.accumulate_molecular(invariants, self.state, apply_stim=True).to(self.dtype)

    def model(self, corrections):
        from payne_zero_synthesis import radiative_transfer
        line = self.fixed + self.selected_opacity(corrections)
        total, continuum, _ = radiative_transfer.solve_spectrum(self.continuum_absorption, self.continuum_scattering,
            line, self.line_scattering, self.line_source, self.pipeline.column_mass, self.pipeline.transfer_tables)
        total = self.rotation(total * self.per_wavelength, vsini_km_s=self.vsini)
        continuum = self.rotation(continuum * self.per_wavelength, vsini_km_s=self.vsini)
        _, _, normalized = self.operator.convolve_fluxes(total, continuum)
        return normalized

    def residual(self, corrections):
        """Weighted residual after the continuum is profiled."""
        flux = self.model(corrections)[self.pixels]
        design = flux[:, None] * self.basis
        coefficients = torch.linalg.lstsq(design * self.root_weight[:, None], (self.flux - flux) * self.root_weight).solution
        return (flux + design @ coefficients - self.flux) * self.root_weight


def selected_lines(offsets):
    """Per-line arrays of the APOGEE CN, CO and OH lines, with the scalar entries of the table."""
    merged, selected = molecular_table(offsets)
    n = merged["species_code"].size
    return {k: (v[selected] if np.ndim(v) and len(v) == n else v) for k, v in merged.items()}


def apogee_surface_factor(wavelength):
    from payne_zero_synthesis.api import _surface_flux_per_wavelength_nm
    return _surface_flux_per_wavelength_nm(wavelength, np.ones_like(wavelength))


def references():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "calibration").glob("*/reference.json"))]
    return [r for r in rows if r.get("endpoint")]


def group_offsets():
    return json.loads((OUT / "calibration/offsets.json").read_text())["offsets"]


def stage_check(args):
    """Differentiable spectrum at zero correction against NetworkModel.project of the same atmosphere."""
    reference = next(r for r in references() if r["sdss_id"] == args.star)
    offsets = group_offsets()
    star = StarSpectrum(reference, offsets, args.device, args.dtype)
    with torch.no_grad():
        ours = star.model(torch.zeros(star.strength.numel(), dtype=star.dtype, device=star.device)).cpu().numpy()
    model = NetworkModel(star.obs, bundle_for(offsets, args.device, args.dtype, CALIBRATION_DIR / "scratch"),
                         args.device, args.dtype, CALIBRATION_DIR / "scratch")
    theirs = model.project(model.synthesize(ROOT / reference["endpoint"]),
                           *[reference["nuisance"][p] for p in PARAMETERS[7:]])
    mask = star.obs.mask
    report = dict(sdss_id=args.star, selected_lines=int(star.strength.numel()),
                  max_abs_difference=float(np.max(abs(ours[mask] - theirs[mask]))))
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    (CALIBRATION_DIR / f"check_{args.star}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def stage_calibrate(args):
    offsets = group_offsets()
    stars = [StarSpectrum(r, offsets, args.device, args.dtype) for r in references()]
    count = stars[0].strength.numel()
    device, dtype = stars[0].device, stars[0].dtype
    with torch.no_grad():
        zero = torch.zeros(count, dtype=dtype, device=device)
        baseline = [float((s.residual(zero) ** 2).mean()) for s in stars]
    scale = float(np.median(baseline))
    pixels = sum(int(s.pixels.numel()) for s in stars)
    free = torch.zeros(count, dtype=dtype, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([free], lr=1.0, max_iter=1, history_size=20, line_search_fn="strong_wolfe")
    history = []

    def corrections():
        return BOUND_DEX * torch.tanh(free / BOUND_DEX)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for star in stars:
            loss = (star.residual(corrections()) ** 2).sum() / scale
            loss.backward()
            total += float(loss)
        prior = (corrections() / PRIOR_DEX).pow(2).sum()
        prior.backward()
        return torch.as_tensor(total + float(prior), dtype=dtype)

    started = time.time()
    objective = float(closure())
    history.append(dict(iteration=0, objective=objective, seconds=0.0))
    for iteration in range(1, MAX_ITERATIONS + 1):
        previous = objective
        optimizer.step(closure)
        with torch.no_grad():
            values = corrections()
            objective = (sum(float((s.residual(values) ** 2).sum()) for s in stars) / scale
                         + float((values / PRIOR_DEX).pow(2).sum()))
        history.append(dict(iteration=iteration, objective=objective, seconds=time.time() - started))
        print(json.dumps(history[-1]), flush=True)
        if previous - objective < TOLERANCE * previous:
            break
    with torch.no_grad():
        values = corrections().detach()
        chi2 = [float((s.residual(values) ** 2).mean()) for s in stars]
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    selected = selected_lines(offsets)
    np.savez(CALIBRATION_DIR / "line_offsets.npz", corrections_dex=values.cpu().numpy(),
             center_index_1based=selected["center_index_1based"], species_code=selected["species_code"],
             lower_excitation_cm=selected["lower_excitation_cm"])
    (CALIBRATION_DIR / "linebyline.json").write_text(json.dumps(dict(offsets=offsets,
        line_offsets=str((CALIBRATION_DIR / "line_offsets.npz").relative_to(ROOT)),
        stars=[s.obs.path.name for s in stars], selected_lines=count,
        selected_by_species={int(c): int((selected["species_code"] == c).sum()) for c in LINE_BY_LINE_SPECIES},
        prior_dex=PRIOR_DEX, bound_dex=BOUND_DEX, chi2_scale=scale, pixels=pixels,
        chi2_per_pixel_before=dict(median=float(np.median(baseline)), per_star=baseline),
        chi2_per_pixel_after=dict(median=float(np.median(chi2)), per_star=chi2),
        corrections=dict(nonzero=int((values.abs() > 1e-3).sum()), rms=float(values.pow(2).mean().sqrt()),
                         at_bound=int((values.abs() > 0.99 * BOUND_DEX).sum())),
        history=history), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "calibrate"))
    parser.add_argument("--star", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    dict(check=stage_check, calibrate=stage_calibrate)[args.stage](args)


if __name__ == "__main__":
    main()
