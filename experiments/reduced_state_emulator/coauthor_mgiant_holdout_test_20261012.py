"""Final M-giant test on random labels inside the holdout boxes.

Ten labels are drawn in the box of each holdout parent. Four starts are
solved at each label: the final two-profile network (rebuilt from (m,T)), the
final six-profile network (used directly), the native MARCS structure of the
holdout parent's grid node (rebuilt from (m,T)), and the parent's converged
atmosphere (rebuilt from (m,T)), which is the reference. A start succeeds at
the iteration where it meets the stability rule, provided its endpoint agrees
with the reference endpoint within the three-metric TiO limit.
"""
from bench import environment as _environment

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from bench.labels import StellarLabels
from payne_zero_atmosphere.warm_start import INITIALIZER_OUTPUT_FIELDS
from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere
from experiments.reduced_state_emulator.marcs_h5 import load_marcs_node
from coauthor_benchmark_20261005 import atmosphere, FIELDS
from coauthor_mgiant_stability_20261007 import run, RULE
from coauthor_mgiant_round2_corpus_20261009 import NAMES, BOX
from coauthor_mgiant_train_20261009 import predict

SEED = 20261013
PER_PARENT = 10
LIMITS = dict(effective_temperature=(3000.0, 4000.0), log_surface_gravity=(0.5, 2.5),
              metallicity=(-1.0, 0.5))
FREE = dict(alpha_enhancement=(-0.1, 0.1), microturbulence_km_s=(1.5, 2.5),
            carbon_enhancement=(-0.3, 0.1), nitrogen_enhancement=(-0.1, 0.5),
            oxygen_enhancement=(-0.1, 0.1))
ARMS = dict(two_profile=90, six_profile=90, marcs_node=90, reference=120)
COMPARED = ("two_profile", "six_profile", "marcs_node")
TEMPERATURE_BINS = ((3000, 3300), (3300, 3600), (3600, 4001))
NAMES_FLUX = {"flux_error_median_percent": "median_absolute_flux_error_percent",
              "flux_error_p95_percent": "p95_absolute_flux_error_percent",
              "flux_error_max_percent": "maximum_absolute_flux_error_percent"}


def holdout(setup_path):
    setup = json.loads(Path(setup_path).read_text())
    return [p for p in setup["parents"] if p["role"] == "holdout"]


def draw(parents, products):
    rng = np.random.default_rng(SEED)
    labels = []
    for parent in parents:
        for _ in range(PER_PARENT):
            row = {}
            for field, width in BOX.items():
                lo, hi = LIMITS[field]
                row[field] = float(np.clip(parent["labels"][field] + rng.uniform(-width, width), lo, hi))
            for field, (lo, hi) in FREE.items():
                row[field] = float(rng.uniform(lo, hi))
            row["absolute_abundance_offsets"] = {z: row["metallicity"] + row[f]
                for z, f in zip((6, 7, 8), NAMES[-3:])}
            row["holdout_parent"] = parent["candidate_id"]
            row["parent_labels"] = parent["labels"]
            row["reference_product"] = products[parent["candidate_id"]]
            labels.append(row)
    return labels


def rebuilt(profile, row):
    result = reconstruct_full_atmosphere(ReducedAtmosphere(profile[:, 0], profile[:, 1], {f: row[f] for f in FIELDS}),
        seed=atmosphere(profile[:, :2], row), n_synchronizations=None, max_synchronizations=8,
        pressure_tolerance_dex=1e-3)
    return np.column_stack([getattr(result.atmosphere, f) for f in INITIALIZER_OUTPUT_FIELDS]), result.n_evaluations


def worker(payload):
    index, row, starts, root = payload
    directory = root / f"case_{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "solve.json").exists():
        return
    report = dict(labels=row, arms={}, arm_errors={})
    for arm, cap in ARMS.items():
        started = time.monotonic()
        try:
            profile = starts[arm]
            if arm == "six_profile":
                start, rebuilds = profile, 0
            else:
                start, rebuilds = rebuilt(profile, row)
            seed_path = directory / f"{arm}_seed.npz"
            np.savez(seed_path, start=start)
            report["arms"][arm] = run(directory, seed_path, row=row, cap=cap, arm=arm)
            report["arms"][arm].update(rebuild_evaluations=rebuilds, seconds=time.monotonic() - started)
        except Exception as exc:
            report["arm_errors"][arm] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(dict(case=index, arm=arm, error=report["arm_errors"].get(arm))), flush=True)
    (directory / "solve.json").write_text(json.dumps(report, indent=2) + "\n")


def tio(path):
    from payne_zero_synthesis import build_structured_atmosphere, synthesize
    with np.load(path) as data:
        columns = build_structured_atmosphere(temperature=data["temperature"], column_mass=data["column_mass"],
            gas_pressure=data["gas_pressure"], electron_density=data["electron_density"],
            elemental_abundances=data["elemental_abundances"], microturbulence=data["microturbulence"],
            molecular_lines=True, device="cuda", dtype="float64")
        mass, temperature = data["column_mass"].copy(), data["temperature"].copy()
    spectrum = synthesize(columns, wavelength_start_nm=665, wavelength_end_nm=667, resolution=20000,
                          molecular_lines=True, device="cuda", dtype="float64")
    arrays = tuple(np.asarray(a) for a in (spectrum.normalized_flux, spectrum.flux_total, spectrum.flux_continuum))
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError(f"nonfinite spectrum: {path}")
    return arrays, mass, temperature


def difference(a, b):
    return dict(normalized_flux=float(np.max(abs(a[0] - b[0]))),
                flux_total=float(np.max(abs(a[1] - b[1]) / b[2])),
                flux_continuum=float(np.max(abs(a[2] / b[2] - 1))))


def stability(directory, arm, gate):
    """Iteration at which the arm meets the stability rule, its endpoint and its start spectrum."""
    out = directory / arm
    iterations = {d["iteration"]: d for d in
                  (json.loads(line) for line in (out / "iterations.jsonl").read_text().splitlines())}
    previous, consecutive, frozen, endpoint, first = None, 0, None, None, None
    for path in sorted(out.glob("iter_*.npz")):
        n = int(path.stem.split("_")[1])
        arrays, mass, temperature = tio(path)
        if first is None:
            first = arrays
        if previous is not None:
            pn, pm, pt, ps = previous
            spectral = difference(arrays, ps)
            dT = float(np.percentile(abs(temperature / pt - 1), 95))
            dm = float(np.percentile(abs(np.log10(mass / pm)), 95))
            flux_pass = n - pn == 5 and all(k in iterations and all(iterations[k][a] <= gate[b]
                for a, b in NAMES_FLUX.items()) for k in range(pn + 1, n + 1))
            stable = (max(spectral.values()) < RULE["spectral_change"] and dT < RULE["temperature_p95"]
                      and dm < RULE["mass_dex_p95"] and flux_pass)
            consecutive = consecutive + 1 if stable else 0
            if consecutive >= RULE["consecutive_segments"]:
                frozen, endpoint = n, arrays
                break
        previous = n, mass, temperature, arrays
    return frozen, endpoint, first


def evaluate(index, root, gate):
    directory = root / f"case_{index:03d}"
    if (directory / "evaluation.json").exists():
        return
    while not (directory / "solve.json").exists():
        time.sleep(30)
    solve = json.loads((directory / "solve.json").read_text())
    report = dict(index=index, labels=solve["labels"], arm_errors=solve["arm_errors"], arms={})
    results = {}
    for arm in ARMS:
        if arm in solve["arm_errors"]:
            continue
        try:
            results[arm] = stability(directory, arm, gate)
        except Exception as exc:
            report["arm_errors"][arm] = f"evaluation {type(exc).__name__}: {exc}"
    reference = results.get("reference", (None, None, None))
    report["reference_stable_iteration"] = reference[0]
    for arm in COMPARED:
        if arm not in results:
            report["arms"][arm] = dict(success=False, stable_iteration=None)
            continue
        frozen, endpoint, first = results[arm]
        entry = dict(stable_iteration=frozen)
        if reference[1] is not None:
            entry["start_difference"] = difference(first, reference[1])
            if endpoint is not None:
                entry["endpoint_difference"] = difference(endpoint, reference[1])
        entry["success"] = bool(frozen is not None and reference[1] is not None
                                and max(entry["endpoint_difference"].values()) <= RULE["cross_path_spectral_limit"])
        report["arms"][arm] = entry
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(index=index, reference=reference[0],
                          **{a: report["arms"][a]["stable_iteration"] for a in COMPARED})), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "evaluate", "collect"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True, help="round-1 setup.json")
    parser.add_argument("--parents", type=Path, help="MARCS structure summary with reference products")
    parser.add_argument("--networks", type=Path, help="directory with the final checkpoints")
    parser.add_argument("--marcs-grid", type=Path)
    parser.add_argument("--gate", type=Path, default=Path("flux_gate.json"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--indices", default=":", help="START:STOP slice of case indices; started cases are skipped")
    args = parser.parse_args()
    start, _, stop = args.indices.partition(":")
    selected = range(int(start or 0), int(stop) if stop else 10 ** 6)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        parents = holdout(args.holdout)
        products = {r["candidate_id"]: r["reference_product"]
                    for r in json.loads(args.parents.read_text())["rows"]}
        labels = draw(parents, products)
        setup_path = root / "setup.json"
        if not setup_path.exists():
            setup_path.write_text(json.dumps(dict(expected=len(labels), labels=labels, random_seed=SEED,
                per_parent=PER_PARENT, box=BOX, free_ranges=FREE, arms=ARMS, networks=str(args.networks),
                marcs_grid=str(args.marcs_grid), stability_rule=RULE,
                success="stable within the cap and endpoint within the TiO limit of the reference endpoint"),
                indent=2) + "\n")
        values = np.asarray([[row[f] for f in NAMES] for row in labels])
        torch.set_num_threads(1)
        two = predict(torch.load(args.networks / "checkpoint_2profile.pt", map_location="cpu", weights_only=False), values)
        six = predict(torch.load(args.networks / "checkpoint_6profile.pt", map_location="cpu", weights_only=False), values)
        payloads = []
        for i, row in enumerate(labels):
            if i % args.shards != args.shard or i not in selected or (root / f"case_{i:03d}").exists():
                continue
            base = {f: float(row["parent_labels"][f]) for f in FIELDS}
            marcs = load_marcs_node(args.marcs_grid, StellarLabels(**base), verify_sha256=False,
                                    expected_sha256=None, depth_coordinate="log_mass")
            with np.load(row["reference_product"]) as d:
                reference = np.column_stack((d["column_mass"], d["temperature"]))
            starts = dict(two_profile=two[i], six_profile=six[i],
                          marcs_node=np.column_stack((marcs.reduced_column_mass, marcs.reduced_temperature)),
                          reference=reference)
            payloads.append((i, row, starts, root))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, payloads))
        return
    while not (root / "setup.json").exists():
        time.sleep(30)
    setup = json.loads((root / "setup.json").read_text())
    if args.stage == "evaluate":
        import torch as _torch
        _torch.set_num_threads(8)
        gate = json.loads(args.gate.read_text())["thresholds"]
        pending = [i for i in range(args.shard, setup["expected"], args.shards) if i in selected]
        while pending:
            ready = [i for i in pending if (root / f"case_{i:03d}" / "solve.json").exists()]
            if not ready:
                time.sleep(60)
                continue
            evaluate(ready[0], root, gate)
            pending.remove(ready[0])
        return
    records = []
    for index in range(setup["expected"]):
        path = root / f"case_{index:03d}" / "evaluation.json"
        while not path.exists():
            time.sleep(30)
        records.append(json.loads(path.read_text()))
    with_reference = [r for r in records if r["reference_stable_iteration"] is not None]
    summary = dict(expected=setup["expected"], completed=len(records),
                   with_reference=len(with_reference), without_reference=len(records) - len(with_reference))
    for arm in COMPARED:
        stable = [r["arms"][arm]["stable_iteration"] for r in with_reference if r["arms"][arm]["success"]]
        starts = [max(r["arms"][arm]["start_difference"].values()) for r in with_reference
                  if "start_difference" in r["arms"][arm]]
        summary[arm] = dict(success=len(stable), success_fraction_with_reference=len(stable) / max(len(with_reference), 1),
            median_success_iteration=float(np.median(stable)) if stable else None,
            median_start_difference=float(np.median(starts)) if starts else None,
            by_temperature={f"{lo}-{min(hi, 4000)}": dict(
                labels=sum(lo <= r["labels"]["effective_temperature"] < hi for r in records),
                with_reference=sum(lo <= r["labels"]["effective_temperature"] < hi for r in with_reference),
                success=sum(lo <= r["labels"]["effective_temperature"] < hi and r["arms"][arm]["success"]
                            for r in with_reference)) for lo, hi in TEMPERATURE_BINS})
    summary["rows"] = records
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
