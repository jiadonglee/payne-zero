"""Compare recovered giant temperatures with MARCS on common column-mass support."""
from bench import environment as _environment

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench.labels import StellarLabels
from experiments.reduced_state_emulator.marcs_h5 import inspect_marcs_grid, load_marcs_node

plt.rcParams.update({"font.size":11,"axes.labelsize":11,"xtick.labelsize":10,"ytick.labelsize":10})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--marcs-grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads(args.status.read_text())
    selected = set(status["eligible_candidate_ids"])
    schema = inspect_marcs_grid(args.marcs_grid, verify_sha256=False, expected_sha256=None)
    rows = []
    curves = []
    for path in sorted(args.cases.glob("*/t*/case.json")):
        record = json.loads(path.read_text())
        if record["candidate_id"] not in selected:
            continue
        labels = record["labels"]
        marcs = load_marcs_node(args.marcs_grid, StellarLabels(**labels),
                               verify_sha256=False, expected_sha256=None, schema=schema, depth_coordinate="log_mass")
        with np.load(record["primary"]["product_path"]) as data:
            mass = data["column_mass"].copy()
            temperature = data["temperature"].copy()
        native_mass = marcs.native_column_mass
        support = (native_mass >= mass[0]) & (native_mass <= mass[-1])
        x = np.log10(native_mass[support])
        native = marcs.native_temperature[support]
        predicted = 10**np.interp(x, np.log10(mass), np.log10(temperature))
        error = abs(predicted/native-1)
        rows.append(dict(candidate_id=record["candidate_id"], labels=labels,
                         reference_product=record["primary"]["product_path"],
                         matched_layers=int(support.sum()), maximum_relative_temperature_difference=float(error.max()),
                         maximum_depth_fraction=float((x[error.argmax()]-x[0])/(x[-1]-x[0])),
                         log_mass=x.tolist(), relative_temperature_difference=error.tolist()))
        curves.append((x,native,predicted))
    if len(rows) != len(selected):
        raise ValueError(f"matched {len(rows)} of {len(selected)} selected giants")
    errors = np.asarray([r["maximum_relative_temperature_difference"] for r in rows])
    median = float(np.median(errors))
    typical = int(np.argmin(abs(errors-median)))
    fig, axes = plt.subplots(1,2,figsize=(9,3.4),constrained_layout=True)
    x,native,predicted = curves[typical]
    axes[0].plot(x,native,color="#1565c0",label="MARCS")
    axes[0].plot(x,predicted,color="#d55e00",ls="--",label="Payne Zero")
    axes[0].set(xlabel=r"$\log_{10}(m/\mathrm{g\,cm^{-2}})$",ylabel="Temperature (K)")
    axes[0].legend(fontsize=9);axes[0].grid(alpha=.2)
    values=np.sort(errors)
    axes[1].plot(values,np.arange(1,len(values)+1)/len(values),color="#1565c0")
    axes[1].set(xlabel="Maximum relative temperature difference",ylabel="Fraction of recovered giants",ylim=(0,1.01))
    axes[1].grid(alpha=.2)
    args.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out/"fig_mstruct.pdf")
    fig.savefig(args.out/"fig_mstruct.png",dpi=180)
    report=dict(count=len(rows),median_maximum_relative_temperature_difference=median,
                maximum_in_deeper_log_mass_half=sum(r["maximum_depth_fraction"]>=.5 for r in rows),
                coordinate="common native MARCS log-column-mass samples, without extrapolation",
                marcs_optical_depth="tau5000; native Rosseland optical depth is unavailable in the extract",
                typical=rows[typical],rows=rows)
    (args.out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(count=len(rows),median=median,typical=rows[typical]["labels"])),flush=True)


if __name__=="__main__":
    main()
