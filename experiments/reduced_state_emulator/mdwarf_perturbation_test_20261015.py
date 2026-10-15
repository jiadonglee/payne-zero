"""Small perturbations of converged M-dwarf atmospheres under the production solver.

Each converged dwarf of the MARCS-grid recovery is restarted four times with
the production solver and no early stop: unperturbed (the control), with the
temperature raised and lowered by 1e-3 in every layer, and with an
alternating +/-1e-3 layer pattern. Every start is rebuilt from (m,T) as in
the M-giant tests. The deviation of a perturbed path from the control at
the same iteration measures whether the iteration contracts near the
solution.
"""
from bench import environment as _environment

import argparse
from concurrent.futures import ProcessPoolExecutor
import dataclasses
import json
from pathlib import Path
import time

import numpy as np

from bench.run_reference import _solver_config
from payne_zero_atmosphere.runner import run_atmosphere_model
from coauthor_benchmark_20261005 import atmosphere
from coauthor_mgiant_holdout_test_20261012 import rebuilt

STATES = (
    "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t4000", "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3800",
    "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3700", "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3600",
    "g+5.00_m+0.50_a+0.00_c+0.00_x1.00_t4000", "g+5.00_m+0.50_a+0.00_c+0.00_x1.00_t3800",
    "g+5.00_m+0.50_a+0.00_c+0.00_x1.00_t3600", "g+5.00_m+0.50_a+0.00_c+0.00_x1.00_t3500",
    "g+5.00_m+0.50_a+0.00_c+0.00_x1.00_t3400",
    "g+4.50_m-1.00_a+0.00_c+0.00_x1.00_t4000", "g+4.50_m-1.00_a+0.00_c+0.00_x1.00_t3800",
)
EPSILON = 1e-3
PERTURBATIONS = ("control", "plus", "minus", "alternating")
ITERATIONS = 60
SEGMENT = 5
FLOOR = 1e-5


def labels(candidate):
    g, m, _, _, x, t = candidate.split("_")
    return dict(effective_temperature=float(t[1:]), log_surface_gravity=float(g[1:]),
                metallicity=float(m[1:]), alpha_enhancement=0.0, microturbulence_km_s=float(x[1:]))


def shape(kind, n):
    if kind == "control":
        return np.zeros(n)
    if kind == "plus":
        return np.full(n, EPSILON)
    if kind == "minus":
        return np.full(n, -EPSILON)
    return EPSILON * (-1.0) ** np.arange(n)


def worker(payload):
    candidate, kind, product, root = payload
    out = root / candidate / kind
    out.mkdir(parents=True, exist_ok=True)
    if (out / "path.npz").exists():
        return
    row = labels(candidate)
    with np.load(product) as d:
        profile = np.column_stack((d["column_mass"], d["temperature"]))
    profile[:, 1] *= 1.0 + shape(kind, len(profile))
    start = atmosphere(rebuilt(profile, row)[0], row)
    temperature, mass, flux, started = [start.temperature.copy()], [start.column_mass.copy()], [], time.monotonic()

    def hook(n, setup, step):
        current = step.remapped.atmosphere
        temperature.append(np.asarray(current.temperature, dtype=float).copy())
        mass.append(np.asarray(current.column_mass, dtype=float).copy())
        f = np.abs(step.remapped.finalization.temperature_correction_result.flux_error_percent)
        flux.append(float(np.percentile(f, 95)))

    error = None
    try:
        config = dataclasses.replace(_solver_config(start, iterations_per_trial=ITERATIONS,
            structured_atmosphere_path=None, debug_state_path=None), enable_convergence_stop=False)
        run_atmosphere_model(config, after_iteration_hook=hook)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    np.savez(out / "path.npz", temperature=np.asarray(temperature), column_mass=np.asarray(mass),
             flux_p95_percent=np.asarray(flux))
    (out / "run.json").write_text(json.dumps(dict(candidate=candidate, perturbation=kind, labels=row,
        product=str(product), iterations=len(flux), error=error, seconds=time.monotonic() - started,
        production_defaults_unchanged=True), indent=2) + "\n")
    print(json.dumps(dict(candidate=candidate, perturbation=kind, iterations=len(flux), error=error)), flush=True)


def deviation(path, control):
    """p95 of |T/T_control - 1| over layers at each common iteration, on the control's column mass."""
    n = min(len(path["temperature"]), len(control["temperature"]))
    values = []
    for k in range(n):
        if not (np.isfinite(path["column_mass"][k]).all() and np.isfinite(path["temperature"][k]).all()):
            break
        x0, x = np.log10(control["column_mass"][k]), np.log10(path["column_mass"][k])
        inside = (x0 >= x.min()) & (x0 <= x.max())
        t = np.interp(x0[inside], x, path["temperature"][k])
        values.append(float(np.percentile(abs(t / control["temperature"][k][inside] - 1), 95)))
    return np.asarray(values)


def drift(control):
    """p95 of |T/T_start - 1| between the last and the first state of the control."""
    first = dict(temperature=control["temperature"][:1], column_mass=control["column_mass"][:1])
    last = dict(temperature=control["temperature"][-1:], column_mass=control["column_mass"][-1:])
    return float(deviation(last, first)[0])


def collect(root):
    rows = []
    for candidate in STATES:
        directory = root / candidate
        runs = {k: json.loads((directory / k / "run.json").read_text()) for k in PERTURBATIONS}
        control = dict(np.load(directory / "control" / "path.npz"))
        entry = dict(candidate=candidate, labels=labels(candidate),
                     control_iterations=runs["control"]["iterations"], control_error=runs["control"]["error"],
                     control_drift_p95=drift(control),
                     control_flux_p95_final=float(control["flux_p95_percent"][-1])
                     if len(control["flux_p95_percent"]) else None, perturbed={})
        for kind in PERTURBATIONS[1:]:
            path = dict(np.load(directory / kind / "path.npz"))
            d = deviation(path, control)
            factors = [d[k + SEGMENT] / d[k] for k in range(0, len(d) - SEGMENT, SEGMENT) if d[k] > FLOOR]
            finite = np.isfinite(path["column_mass"]).all(axis=1) & np.isfinite(path["temperature"]).all(axis=1)
            entry["perturbed"][kind] = dict(iterations=runs[kind]["iterations"], error=runs[kind]["error"],
                first_nonfinite_iteration=None if finite.all() else int(np.flatnonzero(~finite)[0]),
                deviation_start=float(d[0]), deviation_end=float(d[-1]),
                deviation_max=float(d.max()), deviation_every_segment=d[::SEGMENT].tolist(),
                factor_per_segment=factors,
                factor_median=float(np.median(factors)) if factors else None,
                factor_first=float(factors[0]) if factors else None)
        rows.append(entry)
    (root / "summary.json").write_text(json.dumps(dict(states=list(STATES), epsilon=EPSILON,
        iterations=ITERATIONS, segment=SEGMENT, floor=FLOOR, rows=rows), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "collect"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--products", type=Path, help="JSON map candidate id -> converged product path")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "collect":
        collect(root)
        return
    products = json.loads(args.products.read_text())
    payloads = [(c, k, products[c], root) for k in PERTURBATIONS for c in STATES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, payloads))
    collect(root)


if __name__ == "__main__":
    main()
