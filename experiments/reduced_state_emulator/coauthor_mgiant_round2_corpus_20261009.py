"""M-giant training corpus for bootstrapping round 2.

``regenerate`` restarts the v4 cool-corpus atmospheres (scaled-solar CNO) from
their stored (m,T) and keeps the six initializer profiles of every restart that
passes the strict-restart checks. ``assemble`` joins them with the admitted CNO
development atmospheres, removes every row inside a holdout box, and splits the
rest by parent group into training and validation.
"""
from bench import environment as _environment

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

NAMES = ("effective_temperature", "log_surface_gravity", "metallicity",
         "alpha_enhancement", "microturbulence_km_s", "carbon_enhancement",
         "nitrogen_enhancement", "oxygen_enhancement")
BOX = dict(effective_temperature=75.0, log_surface_gravity=0.25, metallicity=0.25)
SEED = 20261010
VALIDATION_FRACTION = 0.15
STRUCTURE = dict(temperature_p95=3e-3, mass_dex_p95=7.7e-3)


def holdout_parents(setup_path):
    setup = json.loads(Path(setup_path).read_text())
    return [p["labels"] for p in setup["parents"] if p["role"] == "holdout"]


def in_holdout_box(row, holdout):
    return any(all(abs(row[f] - h[f]) <= w for f, w in BOX.items()) for h in holdout)


def restart(payload):
    from bench.run_reference import _solver_config
    from payne_zero_atmosphere.runner import run_atmosphere_model
    from payne_zero_atmosphere.warm_start import INITIALIZER_OUTPUT_FIELDS
    from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere
    from coauthor_benchmark_20261005 import atmosphere, FIELDS
    index, row, mass, temperature, gate, out = payload
    path = out / f"cool_{index:02d}.json"
    if path.exists():
        return
    report = dict(index=index, labels=row, passes=False)
    try:
        profile = np.column_stack((mass, temperature))
        rebuilt = reconstruct_full_atmosphere(ReducedAtmosphere(mass, temperature, {f: row[f] for f in FIELDS}),
            seed=atmosphere(profile, row), n_synchronizations=None, max_synchronizations=8,
            pressure_tolerance_dex=1e-3)
        config = _solver_config(rebuilt.atmosphere, iterations_per_trial=120,
            structured_atmosphere_path=None, debug_state_path=None)
        result = run_atmosphere_model(config)
        final = np.column_stack([getattr(result.atmosphere, f) for f in INITIALIZER_OUTPUT_FIELDS])
        last = result.diagnostics["iteration_timings"][-1]
        dT = float(np.percentile(abs(final[:, 1] / temperature - 1), 95))
        dm = float(np.percentile(abs(np.log10(final[:, 0] / mass)), 95))
        checks = dict(converged=bool(result.converged), finite=bool(np.isfinite(final).all()),
            positive=bool((final[:, :5] > 0).all()), monotone_column_mass=bool((np.diff(final[:, 0]) > 0).all()),
            flux_pass=all(last[k] <= v for k, v in gate.items()),
            structure=dT <= STRUCTURE["temperature_p95"] and dm <= STRUCTURE["mass_dex_p95"])
        report.update(checks=checks, iterations=result.iterations_completed,
                      temperature_p95=dT, mass_dex_p95=dm, passes=all(checks.values()))
        if report["passes"]:
            np.save(out / f"cool_{index:02d}_profiles.npy", final)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(index=index, passes=report["passes"], error=report.get("error"))), flush=True)


def cool_rows(path):
    with np.load(path, allow_pickle=True) as data:
        fields = [str(f) for f in data["label_fields"]]
        labels = [dict(zip(fields, map(float, row))) for row in data["labels"]]
        for row in labels:
            row.update(carbon_enhancement=0.0, nitrogen_enhancement=0.0, oxygen_enhancement=0.0)
        return labels, data["column_mass"].copy(), data["temperature"].copy(), [str(n) for n in data["node_ids"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("regenerate", "assemble"))
    parser.add_argument("--holdout", type=Path, required=True, help="round-1 setup.json")
    parser.add_argument("--cool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=Path("flux_gate.json"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cno", type=Path, nargs="*", default=())
    parser.add_argument("--cool-dir", type=Path, help="directory of regenerated cool rows (default: --out)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    holdout = holdout_parents(args.holdout)
    labels, mass, temperature, nodes = cool_rows(args.cool)
    keep = [i for i, row in enumerate(labels) if not in_holdout_box(row, holdout)]
    if args.stage == "regenerate":
        gate = json.loads(args.gate.read_text())["thresholds"]
        payloads = [(i, labels[i], mass[i], temperature[i], gate, args.out) for i in keep]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(restart, payloads))
        return
    cool_dir = args.cool_dir or args.out
    rows, profiles, groups, sources = [], [], [], []
    regenerated = dict(attempted=len(keep), excluded_by_holdout_box=len(labels) - len(keep), passed=0)
    for i in keep:
        report = json.loads((cool_dir / f"cool_{i:02d}.json").read_text())
        if report["passes"]:
            rows.append(labels[i]); profiles.append(np.load(cool_dir / f"cool_{i:02d}_profiles.npy"))
            groups.append(nodes[i]); sources.append("v4_cool_corpus")
            regenerated["passed"] += 1
    cno = {}
    tau = None
    for path in args.cno:
        with np.load(path) as data:
            tau_here = data["standard_rosseland_optical_depth"][0]
            if tau is not None and not np.array_equal(tau, tau_here):
                raise ValueError(f"{path} uses a different optical-depth grid")
            tau = tau_here.copy()
            entries = [json.loads(str(s)) for s in data["labels_json"]]
            excluded = 0
            for row, profile, group in zip(entries, data["atmosphere_profiles"], data["parent_groups"]):
                if in_holdout_box(row, holdout):
                    excluded += 1
                    continue
                rows.append({f: float(row[f]) for f in NAMES}); profiles.append(profile.copy())
                groups.append(str(group)); sources.append(str(path.name))
            cno[str(path)] = dict(rows=len(entries), excluded_by_holdout_box=excluded)
    unique = sorted(set(groups))
    order = np.random.default_rng(SEED).permutation(len(unique))
    validation_groups = {unique[i] for i in order[:max(1, round(VALIDATION_FRACTION * len(unique)))]}
    roles = np.asarray(["validation" if g in validation_groups else "train" for g in groups])
    np.savez(args.out / "corpus.npz", labels=np.asarray([[r[f] for f in NAMES] for r in rows]),
             atmosphere_profiles=np.asarray(profiles), standard_rosseland_optical_depth=tau,
             parent_groups=np.asarray(groups), roles=roles, sources=np.asarray(sources), label_fields=np.asarray(NAMES))
    summary = dict(rows=len(rows), train=int((roles == "train").sum()), validation=int((roles == "validation").sum()),
        parent_groups=len(unique), validation_groups=sorted(validation_groups), split_seed=SEED,
        holdout_box=BOX, cool_regeneration=regenerated, cno_sources=cno)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "validation_groups"}))


if __name__ == "__main__":
    main()
