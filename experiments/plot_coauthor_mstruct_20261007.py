"""Compare recovered giant temperatures with MARCS on common column-mass support."""
from bench import environment as _environment

import argparse
import json
from pathlib import Path

import sys

import numpy as np

from bench.labels import StellarLabels
from experiments.reduced_state_emulator.marcs_h5 import inspect_marcs_grid, load_marcs_node

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib.pyplot as plt  # noqa: E402
import paper_style as ps  # noqa: E402


def draw(x, native, predicted, errors, out):
    """Panel (a): the typical giant; panel (b): the cumulative maximum differences."""
    from matplotlib.ticker import PercentFormatter

    fig = plt.figure(figsize=(ps.PAGE, 2.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(2.2, 1), width_ratios=(1, 1))
    ax = fig.add_subplot(grid[0, 0])
    ax.plot(x, native, color=ps.ALTERNATIVE, lw=1.4, label="MARCS")
    ax.plot(x, predicted, color=ps.PRIMARY, lw=1.4, ls="--", label="Payne Zero")
    ax.set(ylabel="Temperature (K)")
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left")
    ps.panel(ax, "a")
    below = fig.add_subplot(grid[1, 0], sharex=ax)
    below.plot(x, predicted / native - 1, color=ps.PRIMARY, lw=1.2)
    below.axhline(0, color=ps.MUTED, lw=0.6)
    below.yaxis.set_major_locator(plt.MultipleLocator(0.03))
    below.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    below.set(xlabel=r"$\log_{10}(m/\mathrm{g\,cm^{-2}})$", ylabel="Payne Zero\n/ MARCS − 1")
    ax = fig.add_subplot(grid[:, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    values = np.sort(errors)
    ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", color=ps.PRIMARY)
    median = float(np.median(errors))
    ax.axvline(median, color=ps.MUTED, lw=0.7, ls=(0, (1, 2)))
    ps.end_label(ax, median, 0.08, f"median {median:.1%}", dx=3, ha="left")
    ax.set_xlim(0, None)
    ax.set(xlabel="Maximum relative temperature difference", ylabel="Fraction of recovered giants", ylim=(0, 1.02))
    ps.panel(ax, "b")
    out.mkdir(parents=True, exist_ok=True)
    ps.save(fig, out / "fig_mstruct.pdf")


def plot(args):
    """Redraw from summary.json, a local copy of the typical product and the MARCS grid."""
    summary = json.loads((args.out / "summary.json").read_text())
    typical = summary["typical"]
    marcs = load_marcs_node(args.marcs_grid, StellarLabels(**typical["labels"]), verify_sha256=False,
                            expected_sha256=None, depth_coordinate="log_mass")
    with np.load(args.product) as data:
        mass, temperature = data["column_mass"].copy(), data["temperature"].copy()
    support = (marcs.native_column_mass >= mass[0]) & (marcs.native_column_mass <= mass[-1])
    x = np.log10(marcs.native_column_mass[support])
    predicted = 10 ** np.interp(x, np.log10(mass), np.log10(temperature))
    errors = np.asarray([r["maximum_relative_temperature_difference"] for r in summary["rows"]])
    draw(x, marcs.native_temperature[support], predicted, errors, args.figs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--marcs-grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true", help="redraw from OUT/summary.json")
    parser.add_argument("--product", type=Path, help="local copy of the typical product (with --plot-only)")
    parser.add_argument("--figs", type=Path, default=Path("paper/figs"))
    args = parser.parse_args()
    if args.plot_only:
        plot(args)
        return
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
    x,native,predicted = curves[typical]
    draw(x, native, predicted, errors, args.out)
    report=dict(count=len(rows),median_maximum_relative_temperature_difference=median,
                maximum_in_deeper_log_mass_half=sum(r["maximum_depth_fraction"]>=.5 for r in rows),
                coordinate="common native MARCS log-column-mass samples, without extrapolation",
                marcs_optical_depth="tau5000; native Rosseland optical depth is unavailable in the extract",
                typical=rows[typical],rows=rows)
    (args.out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(count=len(rows),median=median,typical=rows[typical]["labels"])),flush=True)


if __name__=="__main__":
    main()
