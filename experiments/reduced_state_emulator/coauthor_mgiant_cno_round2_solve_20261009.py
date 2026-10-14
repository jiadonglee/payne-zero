"""Bootstrapping rounds 2 and 3: random CNO labels over an M-giant label box.

The learned start is the round-2 two-profile network; the reference start is
the nearest corpus atmosphere in scaled label distance. Labels inside a holdout
box are redrawn. Criteria and outcome classes are those of round 1. Round 2
draws over the full range; round 3 over the cool, high-gravity corner.
"""
from bench import environment as _environment

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch

from payne_zero_atmosphere.warm_start import INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH
from coauthor_mgiant_cno_pilot_20261007 import worker
from coauthor_mgiant_cno_restart_20261007 import wait_for
from coauthor_mgiant_cno_multiparent_20261008 import admit, outcome, TEMPERATURE_BINS
from coauthor_mgiant_round2_corpus_20261009 import NAMES, holdout_parents, in_holdout_box
from coauthor_mgiant_train_20261009 import predict

SCALES = np.asarray([250.0, 0.25, 0.20, 0.10, 0.50, 0.10, 0.10, 0.10])
RANGES = dict(effective_temperature=(3000.0, 4000.0), log_surface_gravity=(0.5, 2.5),
              metallicity=(-1.0, 0.5), alpha_enhancement=(-0.1, 0.1), microturbulence_km_s=(1.5, 2.5),
              carbon_enhancement=(-0.3, 0.1), nitrogen_enhancement=(-0.1, 0.5), oxygen_enhancement=(-0.1, 0.1))
ROUNDS = {2: dict(seed=20261011, count=172, ranges=RANGES, scales=SCALES),
          3: dict(seed=20261012, count=120, ranges=dict(RANGES, effective_temperature=(3000.0, 3300.0),
                                                        log_surface_gravity=(1.5, 2.5)),
                  scales=np.concatenate(([100.0], SCALES[1:])))}


def draw(holdout, seed, count, ranges):
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        row = {f: float(rng.uniform(*ranges[f])) for f in NAMES}
        if not in_holdout_box(row, holdout):
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "evaluate", "collect"))
    parser.add_argument("--round", type=int, choices=sorted(ROUNDS), default=2)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True, help="round-1 setup.json")
    parser.add_argument("--corpus", type=Path, required=True, help="round-2 corpus.npz")
    parser.add_argument("--network", type=Path, help="round-2 two-profile checkpoint")
    parser.add_argument("--gate", type=Path, default=Path("flux_gate.json"))
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        holdout = holdout_parents(args.holdout)
        plan = ROUNDS[args.round]
        labels = draw(holdout, plan["seed"], plan["count"], plan["ranges"])
        with np.load(args.corpus) as d:
            corpus_labels, corpus_profiles = d["labels"].copy(), d["atmosphere_profiles"][:, :, :2].copy()
            corpus_groups = d["parent_groups"].copy()
        values = np.asarray([[row[f] for f in NAMES] for row in labels])
        nearest = np.argmin((((values[:, None, :] - corpus_labels[None]) / plan["scales"]) ** 2).sum(-1), axis=1)
        torch.set_num_threads(1)
        learned = predict(torch.load(args.network, map_location="cpu", weights_only=False), values)
        payloads = []
        for i, row in enumerate(labels):
            row["absolute_abundance_offsets"] = {z: row["metallicity"] + row[f]
                for z, f in zip((6, 7, 8), NAMES[-3:])}
            row["development_group"] = f"round{args.round}_{i:03d}"
            row["reference_corpus_row"] = int(nearest[i])
            row["reference_parent_group"] = str(corpus_groups[nearest[i]])
            payloads.append((i, row, (learned[i], corpus_profiles[nearest[i]]), root))
        setup = dict(round=args.round, expected=plan["count"], labels=labels, random_seed=plan["seed"],
            ranges=plan["ranges"],
            reference_scales=plan["scales"].tolist(), holdout_box_rejection=True,
            predictor=str(args.network), reference=f"nearest row of {args.corpus}",
            role=f"bootstrapping round {args.round}, development training; holdout boxes excluded",
            workers=args.workers, physical_iteration_cap=90, reference_iteration_cap=120)
        (root / "setup.json").write_text(json.dumps(setup, indent=2) + "\n")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, payloads))
        return
    wait_for(root / "setup.json")
    setup = json.loads((root / "setup.json").read_text())
    if args.stage == "evaluate":
        gate = json.loads(args.gate.read_text())["thresholds"]
        for index in range(args.shard, setup["expected"], args.shards):
            admit(index, root, args.gate, gate)
        return
    records, labels, profiles, label_rows = [], [], [], []
    for index in range(setup["expected"]):
        directory = root / f"case_{index:02d}"
        wait_for(directory / "admission.json")
        row = setup["labels"][index]
        status = outcome(directory)
        records.append(dict(index=index, outcome=status, reference_parent_group=row["reference_parent_group"],
            **{f: row[f] for f in NAMES}))
        if status == "admitted":
            labels.append([row[f] for f in NAMES]); label_rows.append(json.dumps(row))
            with np.load(directory / "strict_restart_profiles.npz") as d:
                profiles.append(d["profiles"].copy())
    counts = {s: sum(r["outcome"] == s for r in records) for s in
              ("admitted", "no_reference", "learned_not_stable", "paths_disagree",
               "strict_restart_or_spectrum", "evaluation_error")}
    with_reference = len(records) - counts["no_reference"]
    by_temperature = {f"{lo}-{min(hi, 4000)}": dict(
        cases=sum(lo <= r["effective_temperature"] < hi for r in records),
        admitted=sum(lo <= r["effective_temperature"] < hi and r["outcome"] == "admitted" for r in records),
        no_reference=sum(lo <= r["effective_temperature"] < hi and r["outcome"] == "no_reference" for r in records))
        for lo, hi in TEMPERATURE_BINS}
    np.savez(root / "cno_development_corpus.npz", labels=np.asarray(labels).reshape(-1, 8),
        atmosphere_profiles=np.asarray(profiles).reshape(-1, 80, 6),
        standard_rosseland_optical_depth=np.broadcast_to(INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH, (len(labels), 80)),
        labels_json=np.asarray(label_rows, dtype=str),
        parent_groups=np.asarray([f"round{setup.get('round', 2)}_{r['index']:03d}" for r in records
                                  if r["outcome"] == "admitted"], dtype=str),
        roles=np.asarray(["development_training"] * len(labels)))
    (root / "summary.json").write_text(json.dumps(dict(expected=setup["expected"], completed=len(records),
        counts=counts, admitted_fraction=counts["admitted"] / len(records),
        admitted_fraction_with_reference=counts["admitted"] / with_reference if with_reference else None,
        by_temperature=by_temperature, population=f"round {setup.get('round', 2)}: random labels over {setup['ranges']}",
        rows=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
