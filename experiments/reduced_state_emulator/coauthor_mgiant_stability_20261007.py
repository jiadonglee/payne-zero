"""Test the hydrostatic-seeded M-giant start with the existing spectral stop rule."""
from bench import environment as _environment

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from bench.run_reference import _solver_config
from payne_zero_atmosphere.runner import run_atmosphere_model
from payne_zero_atmosphere.synthesis_bridge import save_product_structured_atmosphere
from coauthor_benchmark_20261005 import atmosphere

ROW = dict(effective_temperature=3750.0, log_surface_gravity=0.75,
           metallicity=-0.5, alpha_enhancement=0.0, microturbulence_km_s=2.0)
RULE = dict(segment_iterations=5, consecutive_segments=2, temperature_p95=1e-4,
            mass_dex_p95=1e-3, spectral_change=5e-4, wavelength_nm=[665, 667],
            resolution=20000, cross_path_spectral_limit=5e-3)


def run(root, seed_path, row=ROW, cap=90, arm="physical"):
    out = root / arm
    out.mkdir(parents=True, exist_ok=True)
    with np.load(seed_path) as data:
        start = atmosphere(data["start"], row)
    save_product_structured_atmosphere(start, out / "iter_0000.npz", device="cpu", dtype="float64")
    with (out / "iterations.jsonl").open("w") as handle:
        def hook(n, setup, step):
            current = step.remapped.atmosphere
            flux = np.abs(step.remapped.finalization.temperature_correction_result.flux_error_percent)
            handle.write(json.dumps(dict(iteration=n,
                flux_error_median_percent=float(np.median(flux)),
                flux_error_p95_percent=float(np.percentile(flux, 95)),
                flux_error_max_percent=float(np.max(flux)))) + "\n")
            handle.flush()
            if n % 5 == 0:
                save_product_structured_atmosphere(current, out / f"iter_{n:04d}.npz", device="cpu", dtype="float64")
                print(json.dumps(dict(iteration=n, flux_p95=float(np.percentile(flux, 95)))), flush=True)
        config = dataclasses.replace(_solver_config(start, iterations_per_trial=cap,
            structured_atmosphere_path=None, debug_state_path=None), enable_convergence_stop=False)
        result = run_atmosphere_model(config, after_iteration_hook=hook)
    return dict(iterations=result.iterations_completed, labels=row, stability_rule=RULE,
                iteration_cap=cap, production_defaults_unchanged=True)


def evaluate(root, gate_path):
    from payne_zero_synthesis import build_structured_atmosphere, synthesize
    import torch
    torch.set_num_threads(8)
    while not (root / "solve.json").exists():
        time.sleep(30)
    setup = json.loads((root / "solve.json").read_text())
    if setup.get("arm_errors"):
        (root / "spectral_stability.json").write_text(json.dumps(setup,indent=2)+"\n")
        return
    gate = json.loads(gate_path.read_text())["thresholds"]
    names = {"flux_error_median_percent": "median_absolute_flux_error_percent",
             "flux_error_p95_percent": "p95_absolute_flux_error_percent",
             "flux_error_max_percent": "maximum_absolute_flux_error_percent"}
    report = dict(labels=setup["labels"], stability_rule=RULE, flux_thresholds=gate, arms={})
    endpoints = {}
    for arm in ("physical", "reference"):
        directory = root / arm
        iterations = {d["iteration"]: d for d in
                      (json.loads(line) for line in (directory / "iterations.jsonl").read_text().splitlines())}
        previous = None
        consecutive = 0
        changes = []
        frozen = None
        for path in sorted(directory.glob("iter_*.npz")):
            n = int(path.stem.split("_")[1])
            with np.load(path) as data:
                mass, temperature = data["column_mass"].copy(), data["temperature"].copy()
                columns = build_structured_atmosphere(temperature=temperature, column_mass=mass,
                    gas_pressure=data["gas_pressure"], electron_density=data["electron_density"],
                    elemental_abundances=data["elemental_abundances"], microturbulence=data["microturbulence"],
                    molecular_lines=True, device="cuda", dtype="float64")
            spectrum = synthesize(columns, wavelength_start_nm=665, wavelength_end_nm=667,
                                  resolution=20000, molecular_lines=True, device="cuda", dtype="float64")
            arrays = (np.asarray(spectrum.normalized_flux), np.asarray(spectrum.flux_total),
                      np.asarray(spectrum.flux_continuum))
            if not all(np.isfinite(a).all() for a in arrays):
                raise ValueError(f"nonfinite spectrum: {arm} iteration {n}")
            if previous is not None:
                pn, pm, pt, ps = previous
                spectral = [float(np.max(abs(arrays[0]-ps[0]))),
                            float(np.max(abs(arrays[1]-ps[1])/ps[2])),
                            float(np.max(abs(arrays[2]/ps[2]-1)))]
                dT = float(np.percentile(abs(temperature/pt-1), 95))
                dm = float(np.percentile(abs(np.log10(mass/pm)), 95))
                flux_pass = n-pn == 5 and all(
                    k in iterations and all(iterations[k][a] <= gate[b] for a,b in names.items())
                    for k in range(pn+1,n+1))
                stable = max(spectral)<5e-4 and dT<1e-4 and dm<1e-3 and flux_pass
                consecutive = consecutive+1 if stable else 0
                changes.append(dict(iteration=n, spectral_changes=spectral, temperature_p95=dT,
                                    mass_dex_p95=dm, flux_pass=flux_pass, stable=stable))
                print(json.dumps(dict(arm=arm, **changes[-1])), flush=True)
                if consecutive >= 2:
                    frozen = n
                    endpoints[arm] = arrays
                    break
            previous = n, mass, temperature, arrays
        report["arms"][arm] = dict(frozen_iteration=frozen, segments=changes)
    if len(endpoints)==2:
        a,b = endpoints["physical"],endpoints["reference"]
        cross = dict(normalized_flux=float(np.max(abs(a[0]-b[0]))),
                     flux_total=float(np.max(abs(a[1]-b[1])/b[2])),
                     flux_continuum=float(np.max(abs(a[2]/b[2]-1))))
        report["cross_path"] = cross
        report["spectral_consistency"] = max(cross.values())<=5e-3
    (root / "spectral_stability.json").write_text(json.dumps(report, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        result = run(args.root, args.seed)
        (args.root / "solve.json").write_text(json.dumps(result,indent=2)+"\n")
    else:
        evaluate(args.root, args.gate)


if __name__ == "__main__":
    main()
