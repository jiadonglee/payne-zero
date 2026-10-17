"""Bootstrapping the M-giant corpus with individual abundances (84 inputs).

Labels are drawn as in round 2 of the eight-label bootstrapping, with the
temperature stratified and every holdout box excluded; on top of the
eight-label composition, each of the 81 elements of the direct-abundance
encoding changes independently by a uniform amount within +/-w dex. Labels
with a number ratio C/O >= 1 are redrawn. Every abundance is passed to the
solver as an absolute [X/H].

The learned start is the final eight-label two-profile network at the
projected labels (round 1) or the 84-input two-profile network (round 2),
rebuilt at the full composition; the reference start is the nearest corpus
atmosphere in scaled distance of the projected labels, rebuilt at the full
composition. Stability, spectral agreement, strict restart and admission are
those of the eight-label rounds.

``assemble`` converts the eight-label corpus to the 84-input form and joins
the admitted atmospheres; the eight-label rows keep their training and
validation roles, and new rows are split by development group.
"""
from bench import environment as _environment

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from payne_zero_atmosphere.direct_abundance import DIRECT_XH_ATOMIC_NUMBERS
from payne_zero_atmosphere.warm_start import (
    INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH, compute_metal_log_number_abundances,
)
from coauthor_mgiant_cno_pilot_20261007 import worker
from coauthor_mgiant_cno_restart_20261007 import wait_for
from coauthor_mgiant_cno_multiparent_20261008 import admit, outcome, TEMPERATURE_BINS
from coauthor_mgiant_round2_corpus_20261009 import NAMES, holdout_parents, in_holdout_box
from coauthor_mgiant_train_20261009 import predict
from mgiant_direct_abundance_pilot_20261014 import project

RANGES = dict(log_surface_gravity=(0.5, 2.5), metallicity=(-1.0, 0.5), alpha_enhancement=(-0.1, 0.1),
              microturbulence_km_s=(1.5, 2.5), carbon_enhancement=(-0.3, 0.1),
              nitrogen_enhancement=(-0.1, 0.5), oxygen_enhancement=(-0.1, 0.1))
STRATA = ((3000.0, 3300.0), (3300.0, 3600.0), (3600.0, 4000.0))
SCALES = np.asarray([250.0, 0.25, 0.20, 0.10, 0.50, 0.10, 0.10, 0.10])
COOL_TEMPERATURE_SCALE = 100.0
ROUNDS = {1: dict(seed=20261015, groups=[dict(width=0.1, strata=(80, 88, 88))], learned="eight_label"),
          2: dict(seed=20261016, groups=[dict(width=0.1, strata=(40, 44, 44)),
                                         dict(width=0.2, strata=(40, 44, 44))], learned="direct_abundance")}
VALIDATION_FRACTION = 0.15
SPLIT_SEED = 20261017
ALPHA = (10, 12, 14, 16, 20, 22)


def eight_label_xh(row):
    """[X/H] of the 81 elements for an eight-label composition."""
    m = row["metallicity"]
    xh = {z: m for z in DIRECT_XH_ATOMIC_NUMBERS}
    for z in ALPHA:
        xh[z] = m + row["alpha_enhancement"]
    for z, f in zip((6, 7, 8), NAMES[-3:]):
        xh[z] = m + row[f]
    return xh


def c_to_o(xh):
    log_n = compute_metal_log_number_abundances(absolute_abundance_offsets=xh)
    return float(10 ** (log_n[6 - 3] - log_n[8 - 3]))


def features84(teff, logg, vmic, xh):
    """Label vector in the order of the direct-abundance encoding, T_eff first."""
    iron = xh[26]
    return [teff, logg, vmic, iron] + [xh[z] - iron for z in DIRECT_XH_ATOMIC_NUMBERS if z != 26]


def draw(holdout, plan):
    rng = np.random.default_rng(plan["seed"])
    rows = []
    for group, settings in enumerate(plan["groups"]):
        for (lo, hi), count in zip(STRATA, settings["strata"]):
            drawn = 0
            while drawn < count:
                base = dict(effective_temperature=float(rng.uniform(lo, hi)),
                            **{f: float(rng.uniform(*RANGES[f])) for f in NAMES[1:]})
                delta = {z: float(rng.uniform(-settings["width"], settings["width"])) for z in DIRECT_XH_ATOMIC_NUMBERS}
                if in_holdout_box(base, holdout):
                    continue
                xh = {z: round(v + delta[z], 2) for z, v in eight_label_xh(base).items()}
                ratio = c_to_o(xh)
                if ratio >= 1.0:
                    continue
                row = project(base["effective_temperature"], base["log_surface_gravity"],
                              base["microturbulence_km_s"], xh)
                row.update(absolute_abundance_offsets=xh, base_labels=base, width_dex=settings["width"],
                           width_group=group, c_to_o=ratio)
                rows.append(row)
                drawn += 1
    return rows


def predict84(checkpoint, rows):
    values = np.asarray([features84(r["effective_temperature"], r["log_surface_gravity"],
                                    r["microturbulence_km_s"], r["absolute_abundance_offsets"]) for r in rows])
    return predict(checkpoint, values)


def run(args):
    holdout = holdout_parents(args.holdout)
    plan = ROUNDS[args.round]
    labels = draw(holdout, plan)
    with np.load(args.corpus) as d:
        corpus_labels = d["projected_labels"].copy() if "projected_labels" in d.files else d["labels"].copy()
        corpus_profiles = d["atmosphere_profiles"][:, :, :2].copy()
        corpus_groups = d["parent_groups"].copy()
    values = np.asarray([[row[f] for f in NAMES] for row in labels])
    scales = np.broadcast_to(SCALES, values.shape).copy()
    scales[values[:, 0] < 3300.0, 0] = COOL_TEMPERATURE_SCALE
    nearest = np.argmin((((values[:, None, :] - corpus_labels[None]) / scales[:, None, :]) ** 2).sum(-1), axis=1)
    torch.set_num_threads(1)
    checkpoint = torch.load(args.network, map_location="cpu", weights_only=False)
    learned = predict(checkpoint, values) if plan["learned"] == "eight_label" else predict84(checkpoint, labels)
    payloads = []
    for i, row in enumerate(labels):
        row["development_group"] = f"direct_round{args.round}_{i:03d}"
        row["reference_corpus_row"] = int(nearest[i])
        row["reference_parent_group"] = str(corpus_groups[nearest[i]])
        payloads.append((i, row, (learned[i], corpus_profiles[nearest[i]]), args.root))
    setup = dict(round=args.round, expected=len(labels), labels=labels, random_seed=plan["seed"], groups=plan["groups"],
                 strata=STRATA, ranges=RANGES, reference_scales=SCALES.tolist(),
                 cool_temperature_scale=COOL_TEMPERATURE_SCALE, holdout_box_rejection=True, c_to_o_limit=1.0,
                 learned=plan["learned"], predictor=str(args.network), reference=f"nearest row of {args.corpus}",
                 role=f"direct-abundance bootstrapping round {args.round}, development training; holdout boxes excluded",
                 workers=args.workers, physical_iteration_cap=90, reference_iteration_cap=120)
    if not (args.root / "setup.json").exists():
        (args.root / "setup.json").write_text(json.dumps(setup, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, [p for p in payloads if p[0] % args.shards == args.shard
                               and not (args.root / f"case_{p[0]:02d}").exists()]))


def collect(root):
    setup = json.loads((root / "setup.json").read_text())
    records, rows, profiles = [], [], []
    for index in range(setup["expected"]):
        directory = root / f"case_{index:02d}"
        wait_for(directory / "admission.json")
        row = setup["labels"][index]
        status = outcome(directory)
        records.append(dict(index=index, outcome=status, width_dex=row["width_dex"],
                            reference_parent_group=row["reference_parent_group"], c_to_o=row["c_to_o"],
                            **{f: row[f] for f in NAMES}))
        if status == "admitted":
            rows.append(row)
            with np.load(directory / "strict_restart_profiles.npz") as d:
                profiles.append(d["profiles"].copy())
    counts = lambda subset: {s: sum(r["outcome"] == s for r in subset) for s in
        ("admitted", "no_reference", "learned_not_stable", "paths_disagree", "strict_restart_or_spectrum",
         "evaluation_error")}
    groups = {}
    for width in sorted({r["width_dex"] for r in records}):
        subset = [r for r in records if r["width_dex"] == width]
        c = counts(subset)
        with_reference = len(subset) - c["no_reference"]
        groups[f"{width:.1f}"] = dict(cases=len(subset), counts=c, admitted_fraction=c["admitted"] / len(subset),
            admitted_fraction_with_reference=c["admitted"] / with_reference if with_reference else None,
            by_temperature={f"{lo}-{min(hi, 4000)}": dict(
                cases=sum(lo <= r["effective_temperature"] < hi for r in subset),
                admitted=sum(lo <= r["effective_temperature"] < hi and r["outcome"] == "admitted" for r in subset),
                no_reference=sum(lo <= r["effective_temperature"] < hi and r["outcome"] == "no_reference"
                                 for r in subset)) for lo, hi in TEMPERATURE_BINS})
    xh = [[row["absolute_abundance_offsets"][str(z)] for z in DIRECT_XH_ATOMIC_NUMBERS] for row in rows]
    np.savez(root / "direct_development_corpus.npz",
             labels84=np.asarray([features84(r["effective_temperature"], r["log_surface_gravity"],
                 r["microturbulence_km_s"], {int(k): v for k, v in r["absolute_abundance_offsets"].items()})
                 for r in rows]).reshape(-1, 84),
             projected_labels=np.asarray([[r[f] for f in NAMES] for r in rows]).reshape(-1, 8),
             xh=np.asarray(xh).reshape(-1, 81), atmosphere_profiles=np.asarray(profiles).reshape(-1, 80, 6),
             standard_rosseland_optical_depth=INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH,
             parent_groups=np.asarray([r["development_group"] for r in rows], dtype=str),
             width_dex=np.asarray([r["width_dex"] for r in rows]))
    (root / "summary.json").write_text(json.dumps(dict(round=setup["round"], expected=setup["expected"],
        completed=len(records), groups=groups, rows=records), indent=2) + "\n")


def assemble(base_corpus, rounds, out):
    with np.load(base_corpus) as d:
        base_labels, base_profiles = d["labels"].copy(), d["atmosphere_profiles"].copy()
        base_groups, base_roles, tau = d["parent_groups"].copy(), d["roles"].copy(), d["standard_rosseland_optical_depth"].copy()
    rows = [dict(zip(NAMES, map(float, r))) for r in base_labels]
    labels84 = [features84(r["effective_temperature"], r["log_surface_gravity"], r["microturbulence_km_s"],
                           eight_label_xh(r)) for r in rows]
    projected, profiles = [list(r) for r in base_labels], list(base_profiles)
    groups, roles, sources = list(base_groups), list(base_roles), ["eight_label_corpus"] * len(rows)
    new_groups = []
    for path in rounds:
        with np.load(path) as d:
            if not np.allclose(d["standard_rosseland_optical_depth"], tau, rtol=1e-12, atol=0):
                raise ValueError(f"{path} uses a different optical-depth grid")
            labels84 += d["labels84"].tolist(); projected += d["projected_labels"].tolist()
            profiles += list(d["atmosphere_profiles"]); new_groups += d["parent_groups"].tolist()
            sources += [path.parent.name] * len(d["labels84"])
    order = np.random.default_rng(SPLIT_SEED).permutation(len(new_groups))
    validation = {new_groups[i] for i in order[:round(VALIDATION_FRACTION * len(new_groups))]}
    groups += new_groups
    roles += ["validation" if g in validation else "train" for g in new_groups]
    roles = np.asarray(roles)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "corpus.npz", labels=np.asarray(labels84), projected_labels=np.asarray(projected),
             atmosphere_profiles=np.asarray(profiles), standard_rosseland_optical_depth=tau,
             parent_groups=np.asarray(groups), roles=roles, sources=np.asarray(sources),
             label_fields=np.asarray(["effective_temperature", "log_surface_gravity", "microturbulence_km_s",
                                      "fe_h"] + [f"x{z}_fe" for z in DIRECT_XH_ATOMIC_NUMBERS if z != 26]))
    summary = dict(rows=len(labels84), eight_label_rows=len(rows), direct_rows=len(new_groups),
                   train=int((roles == "train").sum()), validation=int((roles == "validation").sum()),
                   direct_validation=len(validation), split_seed=SPLIT_SEED, rounds=[str(p) for p in rounds])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "evaluate", "collect", "assemble"))
    parser.add_argument("--round", type=int, choices=sorted(ROUNDS), default=1)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--holdout", type=Path, help="round-1 setup.json of the eight-label bootstrapping")
    parser.add_argument("--corpus", type=Path, help="corpus.npz for the reference start, or the base for assemble")
    parser.add_argument("--network", type=Path, help="two-profile checkpoint of the learned start")
    parser.add_argument("--rounds", type=Path, nargs="*", default=(), help="direct_development_corpus.npz files")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gate", type=Path, default=Path("flux_gate.json"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "assemble":
        assemble(args.corpus, args.rounds, args.out)
        return
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.stage == "run":
        run(args)
        return
    wait_for(args.root / "setup.json")
    if args.stage == "evaluate":
        setup = json.loads((args.root / "setup.json").read_text())
        gate = json.loads(args.gate.read_text())["thresholds"]
        pending = list(range(args.shard, setup["expected"], args.shards))
        while pending:
            ready = [i for i in pending if (args.root / f"case_{i:02d}" / "solve.json").exists()]
            if not ready:
                time.sleep(60)
                continue
            admit(ready[0], args.root, args.gate, gate)
            pending.remove(ready[0])
        return
    collect(args.root)


if __name__ == "__main__":
    main()
