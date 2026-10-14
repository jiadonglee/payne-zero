"""Per-element abundance changes at eight development M-giant parents.

Each parent keeps its own T_eff, log g and microturbulence. Four compositions
per parent change elements outside the eight-label relation by 0.1 dex with
random sign: Mg, Si and Ca; Ti alone; C, N and O; all 81 elements. A ninth
case per parent keeps the parent composition and is the null. Every
composition is passed to the solver as an absolute [X/H] entry for all 81
elements. Two starts are solved: the final eight-label two-profile network at
the projected labels, rebuilt at the full composition (cap 90), and the
parent's converged atmosphere rebuilt at the full composition (the reference,
cap 120). The null case solves the reference start only.
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
from payne_zero_atmosphere.warm_start import compute_metal_log_number_abundances
from coauthor_benchmark_20261005 import FIELDS
from coauthor_mgiant_stability_20261007 import run, RULE
from coauthor_mgiant_round2_corpus_20261009 import NAMES
from coauthor_mgiant_train_20261009 import predict
from coauthor_mgiant_cno_multiparent_20261008 import parents
from coauthor_mgiant_holdout_test_20261012 import rebuilt, stability, difference, NAMES_FLUX

SEED = 20261014
STEP = 0.1
PARENTS = (
    "g+1.50_m+0.00_a+0.00_c+0.00_x2.00_t3000", "g+2.50_m+0.00_a+0.00_c+0.00_x2.00_t3000",
    "g+2.50_m-0.50_a+0.00_c+0.00_x2.00_t3200", "g+0.50_m+0.00_a+0.00_c+0.00_x2.00_t3500",
    "g+1.50_m+0.50_a+0.00_c+0.00_x2.00_t3500", "g+2.50_m-1.00_a+0.00_c+0.00_x2.00_t3300",
    "g+1.50_m+0.00_a+0.00_c+0.00_x2.00_t3800", "g+2.50_m-1.00_a+0.00_c+0.00_x2.00_t4000",
)
CHANGES = dict(alpha_split=(12, 14, 20), titanium=(22,), cno=(6, 7, 8), all=DIRECT_XH_ATOMIC_NUMBERS)
PROJECTED_ALPHA = (10, 12, 14, 16, 20, 22)
ARMS = dict(eight_label=90, reference=120)
WINDOWS = dict(tio=(665.0, 667.0, 20000), h_band=(1555.0, 1570.0, 22500))
NETWORK_VALIDATION_T_P95 = 0.0053


def project(teff, logg, vmic, xh):
    """Eight-label coordinates of a full composition: [M/H] = [Fe/H], [alpha/M] the mean [X/Fe]."""
    iron = xh[26]
    return dict(effective_temperature=teff, log_surface_gravity=logg, metallicity=iron,
                alpha_enhancement=float(np.mean([xh[z] - iron for z in PROJECTED_ALPHA])),
                microturbulence_km_s=vmic, carbon_enhancement=xh[6] - iron,
                nitrogen_enhancement=xh[7] - iron, oxygen_enhancement=xh[8] - iron)


def draw(summary):
    rows, roles = parents(summary)
    by_id = {r["candidate_id"]: (r, role) for r, role in zip(rows, roles)}
    rng = np.random.default_rng(SEED)
    cases = []
    for change in ("null", *CHANGES):
        for candidate in PARENTS:
            parent, role = by_id[candidate]
            if role != "development":
                raise ValueError(f"{candidate} is not a development parent")
            labels = parent["labels"]
            delta = {z: 0.0 for z in DIRECT_XH_ATOMIC_NUMBERS}
            for z in CHANGES.get(change, ()):
                delta[z] = float(rng.choice((-STEP, STEP)))
            xh = {z: round(labels["metallicity"] + delta[z], 2) for z in DIRECT_XH_ATOMIC_NUMBERS}
            row = project(labels["effective_temperature"], labels["log_surface_gravity"],
                          labels["microturbulence_km_s"], xh)
            row.update(absolute_abundance_offsets=xh, change=change, delta=delta, parent=candidate,
                       parent_labels=labels, reference_product=parent["reference_product"])
            log_n = compute_metal_log_number_abundances(absolute_abundance_offsets=xh)
            row["c_to_o"] = float(10 ** (log_n[6 - 3] - log_n[8 - 3]))
            cases.append(row)
    return cases


def eight_label_identity(cases):
    """Largest difference between a null composition and the eight-label composition of its parent."""
    worst = 0.0
    for row in cases:
        if row["change"] != "null":
            continue
        p = row["parent_labels"]
        m = p["metallicity"]
        eight = compute_metal_log_number_abundances(metallicity=m, alpha_enhancement=p["alpha_enhancement"],
            absolute_abundance_offsets={6: m, 7: m, 8: m + p["alpha_enhancement"]})
        full = compute_metal_log_number_abundances(absolute_abundance_offsets=row["absolute_abundance_offsets"])
        present = [z - 3 for z in DIRECT_XH_ATOMIC_NUMBERS]
        worst = max(worst, float(np.max(abs(eight[present] - full[present]))))
    return worst


def worker(payload):
    index, row, starts, root = payload
    directory = root / f"case_{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "solve.json").exists():
        return
    report = dict(labels=row, arms={}, arm_errors={})
    for arm, profile in starts.items():
        started = time.monotonic()
        try:
            start, rebuilds = rebuilt(profile, row)
            seed_path = directory / f"{arm}_seed.npz"
            np.savez(seed_path, start=start)
            report["arms"][arm] = run(directory, seed_path, row=row, cap=ARMS[arm], arm=arm)
            report["arms"][arm].update(rebuild_evaluations=rebuilds, seconds=time.monotonic() - started)
        except Exception as exc:
            report["arm_errors"][arm] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(dict(case=index, arm=arm, error=report["arm_errors"].get(arm))), flush=True)
    (directory / "solve.json").write_text(json.dumps(report, indent=2) + "\n")


def synthesize_window(arrays, window, abundances=None):
    from payne_zero_synthesis import build_structured_atmosphere, synthesize
    columns = build_structured_atmosphere(temperature=arrays["temperature"], column_mass=arrays["column_mass"],
        gas_pressure=arrays["gas_pressure"], electron_density=arrays["electron_density"],
        elemental_abundances=arrays["elemental_abundances"] if abundances is None else abundances,
        microturbulence=arrays["microturbulence"], molecular_lines=True, device="cuda", dtype="float64")
    lo, hi, resolution = window
    spectrum = synthesize(columns, wavelength_start_nm=lo, wavelength_end_nm=hi, resolution=resolution,
                          molecular_lines=True, device="cuda", dtype="float64")
    return tuple(np.asarray(a) for a in (spectrum.normalized_flux, spectrum.flux_total, spectrum.flux_continuum))


def load(path):
    with np.load(path) as data:
        return {k: data[k].copy() for k in data.files}


def temperature_response(case, null):
    """|T/T_null - 1| on the null column-mass grid, inside the common range."""
    x0, x = np.log10(null["column_mass"]), np.log10(case["column_mass"])
    inside = (x0 >= x.min()) & (x0 <= x.max())
    relative = abs(np.interp(x0[inside], x, case["temperature"]) / null["temperature"][inside] - 1)
    return dict(max=float(relative.max()), p95=float(np.percentile(relative, 95)))


def evaluate(index, root, gate, nulls):
    directory = root / f"case_{index:03d}"
    if (directory / "evaluation.json").exists():
        return json.loads((directory / "evaluation.json").read_text())
    solve = json.loads((directory / "solve.json").read_text())
    row = solve["labels"]
    report = dict(index=index, change=row["change"], parent=row["parent"], labels=row,
                  arm_errors=dict(solve["arm_errors"]), arms={})
    results = {}
    for arm in solve["arms"]:
        try:
            results[arm] = stability(directory, arm, gate)
        except Exception as exc:
            report["arm_errors"][arm] = f"evaluation {type(exc).__name__}: {exc}"
    reference = results.get("reference", (None, None, None))
    report["reference_stable_iteration"] = reference[0]
    for arm in solve["arms"]:
        if arm in results:
            report["arms"][arm] = dict(stable_iteration=results[arm][0],
                                       seconds=solve["arms"][arm]["seconds"],
                                       iterations=solve["arms"][arm]["iterations"])
    if reference[0] is not None:
        endpoint = load(directory / "reference" / f"iter_{reference[0]:04d}.npz")
        report["endpoint_file"] = str(directory / "reference" / f"iter_{reference[0]:04d}.npz")
        if row["change"] == "null":
            report["parent_product_response"] = temperature_response(endpoint, load(row["reference_product"]))
        elif row["parent"] in nulls:
            null = load(nulls[row["parent"]])
            ratio = np.log10(endpoint["elemental_abundances"] / null["elemental_abundances"])
            report["composition_error_dex"] = float(max(abs(ratio[z - 1] - row["delta"][str(z)])
                                                        for z in DIRECT_XH_ATOMIC_NUMBERS))
            report["temperature_response"] = temperature_response(endpoint, null)
            report["structure_only_spectrum"] = {name: difference(synthesize_window(endpoint, window),
                synthesize_window(null, window, abundances=endpoint["elemental_abundances"]))
                for name, window in WINDOWS.items()}
    if "eight_label" in results:
        frozen, endpoint, first = results["eight_label"]
        entry = report["arms"]["eight_label"]
        if reference[1] is not None:
            entry["start_difference"] = difference(first, reference[1])
            if endpoint is not None:
                entry["endpoint_difference"] = difference(endpoint, reference[1])
        entry["success"] = bool(frozen is not None and reference[1] is not None
                                and max(entry["endpoint_difference"].values()) <= RULE["cross_path_spectral_limit"])
    elif row["change"] != "null":
        report["arms"]["eight_label"] = dict(success=False, stable_iteration=None)
    (directory / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(index=index, change=row["change"], reference=reference[0],
                          eight_label=report["arms"].get("eight_label", {}).get("stable_iteration"))), flush=True)
    return report


def collect(root, records, setup):
    summary = dict(expected=setup["expected"], eight_label_identity_dex=setup["eight_label_identity_dex"],
                   network_validation_temperature_p95=NETWORK_VALIDATION_T_P95,
                   null_reference_stable=sum(r["reference_stable_iteration"] is not None
                                             for r in records if r["change"] == "null"),
                   null_against_parent_product_max=max((r["parent_product_response"]["max"] for r in records
                                                        if "parent_product_response" in r), default=None),
                   changes={})
    for change in CHANGES:
        rows = [r for r in records if r["change"] == change]
        with_reference = [r for r in rows if r["reference_stable_iteration"] is not None]
        response = [r["temperature_response"] for r in with_reference if "temperature_response" in r]
        spectra = [r["structure_only_spectrum"] for r in with_reference if "structure_only_spectrum" in r]
        success = [r for r in with_reference if r["arms"]["eight_label"]["success"]]
        summary["changes"][change] = dict(
            labels=len(rows), with_reference=len(with_reference), eight_label_success=len(success),
            composition_error_dex_max=max((r["composition_error_dex"] for r in with_reference
                                           if "composition_error_dex" in r), default=None),
            temperature_response_max=max((x["max"] for x in response), default=None),
            temperature_response_p95_median=float(np.median([x["p95"] for x in response])) if response else None,
            structure_only_spectrum_max={name: max((max(s[name].values()) for s in spectra), default=None)
                                         for name in WINDOWS},
            median_iteration=dict(
                reference=float(np.median([r["reference_stable_iteration"] for r in with_reference]))
                if with_reference else None,
                eight_label=float(np.median([r["arms"]["eight_label"]["stable_iteration"] for r in success]))
                if success else None),
            median_seconds={arm: float(np.median([r["arms"][arm]["seconds"] for r in rows if arm in r["arms"]
                                                  and "seconds" in r["arms"][arm]]))
                            for arm in ARMS if any(arm in r["arms"] and "seconds" in r["arms"][arm] for r in rows)})
    summary["rows"] = records
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parents", type=Path, required=True, help="MARCS structure summary with reference products")
    parser.add_argument("--networks", type=Path, help="directory with the final checkpoints")
    parser.add_argument("--gate", type=Path, default=Path("flux_gate.json"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cases = draw(args.parents)
    setup_path = root / "setup.json"
    if not setup_path.exists():
        setup_path.write_text(json.dumps(dict(expected=len(cases), cases=cases, random_seed=SEED, step_dex=STEP,
            parents=PARENTS, changes={k: list(v) for k, v in CHANGES.items()}, projected_alpha=PROJECTED_ALPHA,
            arms=ARMS, windows=WINDOWS, networks=str(args.networks), stability_rule=RULE,
            eight_label_identity_dex=eight_label_identity(cases)), indent=2) + "\n")
    setup = json.loads(setup_path.read_text())
    if args.stage == "run":
        torch.set_num_threads(1)
        values = np.asarray([[row[f] for f in NAMES] for row in cases])
        two = predict(torch.load(args.networks / "checkpoint_2profile.pt", map_location="cpu", weights_only=False), values)
        payloads = []
        for i, row in enumerate(cases):
            if (root / f"case_{i:03d}").exists():
                continue
            with np.load(row["reference_product"]) as d:
                reference = np.column_stack((d["column_mass"], d["temperature"]))
            starts = dict(reference=reference) if row["change"] == "null" else dict(eight_label=two[i], reference=reference)
            payloads.append((i, row, starts, root))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(worker, payloads))
        return
    torch.set_num_threads(8)
    gate = json.loads(args.gate.read_text())["thresholds"]
    records, nulls = [], {}
    for i in range(setup["expected"]):
        while not (root / f"case_{i:03d}" / "solve.json").exists():
            time.sleep(60)
        record = evaluate(i, root, gate, nulls)
        if record["change"] == "null" and "endpoint_file" in record:
            nulls[record["parent"]] = record["endpoint_file"]
        records.append(record)
    collect(root, records, setup)


if __name__ == "__main__":
    main()
