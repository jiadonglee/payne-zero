"""Final 84-input M-giant test on random labels inside the holdout boxes.

Twelve labels are drawn in the box of each holdout parent, with the
eight-label coordinates of the eight-label holdout test and every one of the
81 elements changed independently within +/-0.1 dex; labels with C/O >= 1 are
redrawn. Five starts are solved: the 84-input two-profile network (rebuilt),
the 84-input six-profile network (used directly), the final eight-label
two-profile network at the projected labels (rebuilt), the native MARCS
structure of the parent's grid node (rebuilt), and the parent's converged
atmosphere (rebuilt), which is the reference. Every rebuild and solve uses the
full composition. Success is the stability rule plus endpoint agreement with
the reference, as in the eight-label holdout test.
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
from experiments.reduced_state_emulator.marcs_h5 import load_marcs_node
from payne_zero_atmosphere.direct_abundance import DIRECT_XH_ATOMIC_NUMBERS
from coauthor_benchmark_20261005 import FIELDS
from coauthor_mgiant_stability_20261007 import run, RULE
from coauthor_mgiant_round2_corpus_20261009 import NAMES, BOX
from coauthor_mgiant_train_20261009 import predict
from coauthor_mgiant_holdout_test_20261012 import holdout, rebuilt, stability, difference, LIMITS, FREE
from mgiant_direct_abundance_pilot_20261014 import project
from mgiant_direct_abundance_bootstrap_20261015 import eight_label_xh, c_to_o, predict84

SEED = 20261018
PER_PARENT = 12
WIDTH = 0.1
ARMS = dict(two_profile=90, six_profile=90, eight_label=90, marcs_node=90, reference=120)
COMPARED = ("two_profile", "six_profile", "eight_label", "marcs_node")
TEMPERATURE_BINS = ((3000, 3300), (3300, 3600), (3600, 4001))


def draw(parents, products):
    rng = np.random.default_rng(SEED)
    labels = []
    for parent in parents:
        drawn = 0
        while drawn < PER_PARENT:
            base = {}
            for field, width in BOX.items():
                lo, hi = LIMITS[field]
                base[field] = float(np.clip(parent["labels"][field] + rng.uniform(-width, width), lo, hi))
            for field, (lo, hi) in FREE.items():
                base[field] = float(rng.uniform(lo, hi))
            delta = {z: float(rng.uniform(-WIDTH, WIDTH)) for z in DIRECT_XH_ATOMIC_NUMBERS}
            xh = {z: round(v + delta[z], 2) for z, v in eight_label_xh(base).items()}
            ratio = c_to_o(xh)
            if ratio >= 1.0:
                continue
            row = project(base["effective_temperature"], base["log_surface_gravity"], base["microturbulence_km_s"], xh)
            row.update(absolute_abundance_offsets=xh, base_labels=base, c_to_o=ratio,
                       holdout_parent=parent["candidate_id"], parent_labels=parent["labels"],
                       reference_product=products[parent["candidate_id"]])
            labels.append(row)
            drawn += 1
    return labels


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
            if arm == "six_profile":
                start, rebuilds = starts[arm], 0
            else:
                start, rebuilds = rebuilt(starts[arm], row)
            seed_path = directory / f"{arm}_seed.npz"
            np.savez(seed_path, start=start)
            report["arms"][arm] = run(directory, seed_path, row=row, cap=cap, arm=arm)
            report["arms"][arm].update(rebuild_evaluations=rebuilds, seconds=time.monotonic() - started)
        except Exception as exc:
            report["arm_errors"][arm] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(dict(case=index, arm=arm, error=report["arm_errors"].get(arm))), flush=True)
    (directory / "solve.json").write_text(json.dumps(report, indent=2) + "\n")


def evaluate(index, root, gate):
    directory = root / f"case_{index:03d}"
    if (directory / "evaluation.json").exists():
        return
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
        entry = dict(stable_iteration=frozen, seconds=solve["arms"][arm]["seconds"],
                     rebuild_evaluations=solve["arms"][arm]["rebuild_evaluations"])
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


def collect(root, setup):
    records = [json.loads((root / f"case_{i:03d}" / "evaluation.json").read_text()) for i in range(setup["expected"])]
    with_reference = [r for r in records if r["reference_stable_iteration"] is not None]
    summary = dict(expected=setup["expected"], with_reference=len(with_reference),
                   without_reference=len(records) - len(with_reference))
    for arm in COMPARED:
        success = [r for r in with_reference if r["arms"][arm]["success"]]
        starts = [max(r["arms"][arm]["start_difference"].values()) for r in with_reference
                  if "start_difference" in r["arms"][arm]]
        summary[arm] = dict(success=len(success), success_fraction_with_reference=len(success) / max(len(with_reference), 1),
            median_success_iteration=float(np.median([r["arms"][arm]["stable_iteration"] for r in success])) if success else None,
            median_start_difference=float(np.median(starts)) if starts else None,
            median_seconds=float(np.median([r["arms"][arm]["seconds"] for r in records if "seconds" in r["arms"][arm]])),
            by_temperature={f"{lo}-{min(hi, 4000)}": dict(
                labels=sum(lo <= r["labels"]["effective_temperature"] < hi for r in records),
                with_reference=sum(lo <= r["labels"]["effective_temperature"] < hi for r in with_reference),
                success=sum(lo <= r["labels"]["effective_temperature"] < hi for r in success))
                for lo, hi in TEMPERATURE_BINS})
    pairs = {}
    for a, b in (("two_profile", "eight_label"), ("six_profile", "eight_label"), ("two_profile", "six_profile")):
        both = [r for r in with_reference if r["arms"][a]["success"] and r["arms"][b]["success"]]
        pairs[f"{a}_vs_{b}"] = dict(paired=len(both),
            faster=sum(r["arms"][a]["stable_iteration"] < r["arms"][b]["stable_iteration"] for r in both),
            equal=sum(r["arms"][a]["stable_iteration"] == r["arms"][b]["stable_iteration"] for r in both),
            slower=sum(r["arms"][a]["stable_iteration"] > r["arms"][b]["stable_iteration"] for r in both),
            only_first=sum(r["arms"][a]["success"] and not r["arms"][b]["success"] for r in with_reference),
            only_second=sum(r["arms"][b]["success"] and not r["arms"][a]["success"] for r in with_reference))
    summary["pairs"] = pairs
    summary["rows"] = records
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "evaluate", "collect"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True, help="round-1 setup.json")
    parser.add_argument("--parents", type=Path, help="MARCS structure summary with reference products")
    parser.add_argument("--networks", type=Path, help="directory with the final 84-input checkpoints")
    parser.add_argument("--eight-label", type=Path, help="final eight-label two-profile checkpoint")
    parser.add_argument("--marcs-grid", type=Path)
    parser.add_argument("--gate", type=Path, default=Path("flux_gate.json"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        parents = holdout(args.holdout)
        products = {r["candidate_id"]: r["reference_product"] for r in json.loads(args.parents.read_text())["rows"]}
        labels = draw(parents, products)
        if not (root / "setup.json").exists():
            (root / "setup.json").write_text(json.dumps(dict(expected=len(labels), labels=labels, random_seed=SEED,
                per_parent=PER_PARENT, width_dex=WIDTH, box=BOX, free_ranges=FREE, arms=ARMS,
                networks=str(args.networks), eight_label=str(args.eight_label), marcs_grid=str(args.marcs_grid),
                stability_rule=RULE, c_to_o_limit=1.0,
                success="stable within the cap and endpoint within the TiO limit of the reference endpoint"),
                indent=2) + "\n")
        torch.set_num_threads(1)
        load = lambda p: torch.load(p, map_location="cpu", weights_only=False)
        two = predict84(load(args.networks / "checkpoint_2profile.pt"), labels)
        six = predict84(load(args.networks / "checkpoint_6profile.pt"), labels)
        eight = predict(load(args.eight_label), np.asarray([[row[f] for f in NAMES] for row in labels]))
        payloads = []
        for i, row in enumerate(labels):
            if i % args.shards != args.shard or (root / f"case_{i:03d}").exists():
                continue
            base = {f: float(row["parent_labels"][f]) for f in FIELDS}
            marcs = load_marcs_node(args.marcs_grid, StellarLabels(**base), verify_sha256=False,
                                    expected_sha256=None, depth_coordinate="log_mass")
            with np.load(row["reference_product"]) as d:
                reference = np.column_stack((d["column_mass"], d["temperature"]))
            starts = dict(two_profile=two[i], six_profile=six[i], eight_label=eight[i],
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
        torch.set_num_threads(8)
        gate = json.loads(args.gate.read_text())["thresholds"]
        pending = list(range(args.shard, setup["expected"], args.shards))
        while pending:
            ready = [i for i in pending if (root / f"case_{i:03d}" / "solve.json").exists()]
            if not ready:
                time.sleep(60)
                continue
            evaluate(ready[0], root, gate)
            pending.remove(ready[0])
        return
    collect(root, setup)


if __name__ == "__main__":
    main()
