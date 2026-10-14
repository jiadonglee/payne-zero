"""Small abundance steps from independent MARCS solutions to two CNO targets."""
from bench import environment as _environment

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from bench.run_reference import _solver_config
from payne_zero_atmosphere.runner import run_atmosphere_model
from payne_zero_atmosphere.warm_start import INITIALIZER_OUTPUT_FIELDS
from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere
from coauthor_benchmark_20261005 import atmosphere, FIELDS
from coauthor_mgiant_stability_20261007 import run, evaluate
from coauthor_mgiant_cno_restart_20261007 import worker as strict_restart, spectrum, wait_for

TARGETS = ((3, 1), (5, 7))
CNO = ("carbon_enhancement", "nitrogen_enhancement", "oxygen_enhancement")


def rebuild(profile, row):
    return reconstruct_full_atmosphere(
        ReducedAtmosphere(profile[:, 0], profile[:, 1], {f: row[f] for f in FIELDS}),
        seed=atmosphere(profile, row), n_synchronizations=None,
        max_synchronizations=8, pressure_tolerance_dex=1e-3).atmosphere


def worker(payload):
    index, donor, source, root, gate = payload
    directory = root / f"case_{index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    target = json.loads((source / f"case_{index:02d}" / "solve.json").read_text())["labels"]
    previous = json.loads((source / f"case_{donor:02d}" / "spectral_stability.json").read_text())
    n = previous["arms"]["reference"]["frozen_iteration"]
    product = source / f"case_{donor:02d}" / "reference" / f"iter_{n:04d}.npz"
    with np.load(product) as data:
        profile = np.column_stack((data["column_mass"], data["temperature"]))
    report = dict(labels=target, donor_case=donor, donor_product=str(product),
                  abundance_step_dex=0.025, steps=[], arms={}, arm_errors={})
    try:
        for step in range(1, 9):
            row = dict(target)
            row.update({f: previous["labels"][f] + step / 8 *
                        (target[f] - previous["labels"][f]) for f in CNO})
            row["absolute_abundance_offsets"] = {
                z: row["metallicity"] + row[f] for z, f in zip((6, 7, 8), CNO)}
            initial = rebuild(profile, row)
            result = run_atmosphere_model(_solver_config(initial, iterations_per_trial=120,
                structured_atmosphere_path=directory / f"step_{step:02d}.npz", debug_state_path=None))
            final = result.atmosphere
            quality = all(np.isfinite(getattr(final, f)).all() and (getattr(final, f) > 0).all()
                          for f in INITIALIZER_OUTPUT_FIELDS)
            quality = bool(quality and (np.diff(final.column_mass) > 0).all())
            last = result.diagnostics["iteration_timings"][-1]
            flux_pass = all(last[k] <= v for k, v in gate.items())
            record = dict(step=step, labels=row, iterations=result.iterations_completed,
                          numerically_converged=bool(result.converged), quality=quality,
                          flux_pass=flux_pass)
            report["steps"].append(record)
            print(json.dumps(dict(case=index, **record)), flush=True)
            if not (result.converged and quality and flux_pass):
                raise ValueError(f"abundance step {step} did not pass the fixed numerical and flux criteria")
            profile = np.column_stack((final.column_mass, final.temperature))
        target["absolute_abundance_offsets"] = {int(z): v for z, v in target["absolute_abundance_offsets"].items()}
        seed = rebuild(profile, target)
        seed_path = directory / "reference_seed.npz"
        np.savez(seed_path, start=np.column_stack([getattr(seed, f) for f in INITIALIZER_OUTPUT_FIELDS]))
        report["arms"]["reference"] = run(directory, seed_path, row=target, cap=120, arm="reference")
        report["arms"]["physical"] = json.loads((source / f"case_{index:02d}" / "solve.json").read_text())["arms"]["physical"]
        (directory / "physical").symlink_to(source / f"case_{index:02d}" / "physical", target_is_directory=True)
    except Exception as exc:
        report["arm_errors"]["reference"] = f"{type(exc).__name__}: {exc}"
    (directory / "solve.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "evaluate"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=Path("flux_gate.json"))
    args = parser.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    gate = json.loads(args.gate.read_text())["thresholds"]
    if args.stage == "run":
        (root / "setup.json").write_text(json.dumps(dict(targets=TARGETS,
            source=str(source), role="development reference-path diagnostic",
            abundance_step_dex=0.025, steps=8, original_outcomes_preserved=True), indent=2) + "\n")
        with ProcessPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker, [(i, d, source, root, gate) for i, d in TARGETS]))
        return
    rows = []
    for index, _ in TARGETS:
        directory = root / f"case_{index:02d}"
        wait_for(directory / "solve.json")
        evaluate(directory, args.gate)
        strict_restart((index, root, gate))
        record = json.loads((directory / "strict_restart.json").read_text())
        if record["passes"]:
            a, b = spectrum(directory / "strict_restart.npz"), spectrum(Path(record["reference_product"]))
            metrics = dict(normalized_flux=float(np.max(abs(a[0] - b[0]))),
                flux_total=float(np.max(abs(a[1] - b[1]) / b[2])),
                flux_continuum=float(np.max(abs(a[2] / b[2] - 1))))
            record.update(spectral_metrics=metrics, passes=max(metrics.values()) <= 5e-3)
        rows.append(record)
    (root / "diagnostic.json").write_text(json.dumps(dict(expected=2, completed=2,
        accepted=sum(r["passes"] for r in rows), original_pilot_admitted=6,
        original_pilot_expected=8, role="development; original pilot outcomes retained",
        rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
