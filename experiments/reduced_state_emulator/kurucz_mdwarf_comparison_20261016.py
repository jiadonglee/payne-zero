"""The original Fortran ATLAS12 on the M dwarfs that Payne Zero does not converge.

Each dwarf is started twice: from the MARCS node as rebuilt by Payne Zero
(the start of the Payne Zero recovery, written as a Kurucz deck), and from
the nearest model of the Kurucz initial-atmosphere grid (the default of the
``kurucz`` package). ATLAS12 runs in blocks of 30 iterations with the input
cards of the package's ``atlas12`` (molecules, TiO and H2O line opacity,
convection with overshoot), at most four blocks (120 iterations); each block
starts from the model of the previous one. Convergence is the criterion of the package's
``checkconv``: the largest relative temperature change between the last two
iterations in layers 40-75 is below 1e-4. SYNTHE is not run.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

MAX_BLOCKS = 4
DLNT_LIMIT = 1e-4
STARTS = ("payne_zero_marcs", "kurucz_grid")
CONTROLS = {"t3600_g4.50_m+0.00", "t3800_g4.50_m+0.00", "t3400_g5.00_m+0.50"}


def read_iterations(path):
    """Per-iteration arrays of the ATLAS12 iteration file (fort.66): 80 layers x 13 columns."""
    rows = [line.replace("D", "E").split() for line in Path(path).read_text().splitlines()]
    blocks, current = [], []
    for fields in rows:
        if len(fields) == 13:
            try:
                current.append([float(x) for x in fields])
            except ValueError:
                continue
            if len(current) == 80:
                blocks.append(np.asarray(current))
                current = []
    return np.asarray(blocks)


def read_deck(path):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("READ DECK6"))
    table = np.asarray([[float(x) for x in line.replace("D", "E").split()[:2]] for line in lines[start + 1:start + 81]])
    return table[:, 0], table[:, 1]


def temperature_difference(mass, temperature, reference_mass, reference_temperature):
    x0, x = np.log10(reference_mass), np.log10(mass)
    inside = (x0 >= x.min()) & (x0 <= x.max())
    relative = abs(np.interp(x0[inside], x, temperature) / reference_temperature[inside] - 1)
    return dict(max=float(relative.max()), p95=float(np.percentile(relative, 95)))


ATLAS12_READ_LINES = """MOLECULES ON
READ MOLECULES
READ PUNCH
READ LINES
CONVECTION OVER 1.25 0 36
ITERATIONS 1 PRINT 1 PUNCH 0
BEGIN
END
"""
ATLAS12_BLOCK = """MOLECULES ON
READ MOLECULES
READ PUNCH
TITLE ATLAS12 l/H=1.25
OPACITY ON LINES
OPACITY ON XLINES
CONVECTION OVER 1.25 0 36
ITERATIONS 30
PRINT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
PUNCH 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2
SCALE MODEL 80 -6.875 0.125 {teff:.0f}. {logg:.4f}
VTURB {vturb:.2E}
BEGIN
END
"""


def run_blocks(kurucz, work, current, teff, logg, vturb):
    """ATLAS12 blocks of the ``kurucz`` package, with the input kept apart from the output.

    Returns the number of blocks, whether the last block wrote a model, and the iteration files.
    """
    iteration_files = []
    for block in range(1, MAX_BLOCKS + 1):
        tmp = work / f"block{block}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        kurucz._setup_atlas12_links(tmp, current)
        with open(tmp / "read_lines.out", "w") as f:
            subprocess.run([str(kurucz.BIN_DIR / "atlas12.exe")], input=ATLAS12_READ_LINES, text=True, stdout=f, cwd=tmp)
        if (tmp / "fort.12").exists():
            shutil.move(tmp / "fort.12", tmp / "tmp.bin")
        for f in tmp.glob("fort.*"):
            f.unlink()
        (tmp / "fort.3").symlink_to(current)
        (tmp / "fort.19").symlink_to(kurucz.KURUCZ_HOME / "lines" / "nltelinobsat12.bin")
        (tmp / "fort.2").symlink_to(kurucz.KURUCZ_HOME / "lines" / "molecules.new")
        (tmp / "fort.12").symlink_to(tmp / "tmp.bin")
        with open(tmp / "atlas12.out", "w") as f:
            subprocess.run([str(kurucz.BIN_DIR / "atlas12.exe")], text=True, stdout=f, cwd=tmp,
                           input=ATLAS12_BLOCK.format(teff=teff, logg=logg, vturb=vturb * 1.0e5))
        if (tmp / "fort.66").exists():
            iteration_files.append(work / f"block{block}.iter")
            shutil.copy(tmp / "fort.66", iteration_files[-1])
        if not (tmp / "fort.7").exists():
            return block, False, iteration_files
        result = work / f"block{block}.atm"
        shutil.copy(tmp / "fort.7", result)
        subprocess.run([str(kurucz.BIN_DIR / "checkconv.exe")], cwd=tmp, capture_output=True)
        failed = (tmp / "model.failed").exists()
        (tmp / "tmp.bin").unlink(missing_ok=True)
        for f in tmp.glob("fort.*"):
            f.unlink()
        current = result
        if not failed:
            return block, True, iteration_files
    return MAX_BLOCKS, True, iteration_files


def worker(payload):
    state, start, kurucz_home, root, reference = payload
    sys.path.insert(0, str(kurucz_home))
    import kurucz

    name = f"{state['name']}_{start}"
    out = root / "runs" / name
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    initial = out / "start.atm"
    if start == "payne_zero_marcs":
        shutil.copy(root / state["deck"], initial)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            kurucz.generate_atmosphere(initial, state["effective_temperature"], state["log_surface_gravity"],
                                       state["metallicity"], 0.0, state["microturbulence_km_s"])
    started = time.monotonic()
    blocks, wrote_model, iteration_files = run_blocks(kurucz, out, initial, state["effective_temperature"],
                                                      state["log_surface_gravity"], state["microturbulence_km_s"])
    record = dict(name=name, state=state, start=start, crashed=not wrote_model, blocks=blocks,
                  seconds=time.monotonic() - started)
    if iteration_files:
        iterations = read_iterations(iteration_files[-1])
        if len(iterations) >= 2:
            temperature = iterations[:, :, 2]
            dlnt = abs(temperature[-2] - temperature[-1]) / temperature[-1]
            finite = bool(np.isfinite(iterations[-1]).all())
            record.update(last_block_iterations=int(len(iterations)), finite=finite,
                          max_dlnt_layers_40_75=float(dlnt[39:75].max()),
                          max_abs_flux_error=float(abs(iterations[-1][:, 11]).max()),
                          max_abs_flux_derivative_error=float(abs(iterations[-1][:, 12]).max()))
            record["converged"] = bool(wrote_model and finite and record["max_dlnt_layers_40_75"] < DLNT_LIMIT)
    record.setdefault("converged", False)
    final = out / f"block{blocks}.atm"
    if final.exists():
        mass, temperature = read_deck(final)
        record["final_temperature_range"] = [float(temperature.min()), float(temperature.max())]
        if reference is not None:
            with np.load(reference) as d:
                record["difference_from_payne_zero"] = temperature_difference(
                    mass, temperature, d["column_mass"], d["temperature"])
    (out / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record.get(k) for k in ("name", "crashed", "converged", "blocks", "max_dlnt_layers_40_75")}),
          flush=True)
    return record


def collect(root, gate_path):
    """The package criterion and, as a diagnostic, the Payne Zero flux gate on the last ATLAS12 iteration."""
    gate = json.loads(gate_path.read_text())["thresholds"]
    rows = []
    for path in sorted((root / "runs").glob("*/record.json")):
        record = json.loads(path.read_text())
        files = sorted(path.parent.glob("block*.iter"))
        error = abs(read_iterations(files[-1])[-1][:, 11]) if files else np.asarray([np.inf])
        statistics = dict(median=float(np.median(error)), p95=float(np.percentile(error, 95)), max=float(error.max()))
        record["flux_error_percent"] = statistics
        record["flux_gate"] = bool(statistics["median"] <= gate["median_absolute_flux_error_percent"]
                                   and statistics["p95"] <= gate["p95_absolute_flux_error_percent"]
                                   and statistics["max"] <= gate["maximum_absolute_flux_error_percent"])
        rows.append(record)
    controls = [r for r in rows if r["state"]["name"] in CONTROLS]
    failing = [r for r in rows if r["state"]["name"] not in CONTROLS]
    summary = dict(gate=gate, controls=sorted(CONTROLS),
                   counts={start: dict(
                       failing_states=sum(r["start"] == start for r in failing),
                       failing_converged=sum(r["start"] == start and r["converged"] for r in failing),
                       failing_flux_gate=sum(r["start"] == start and r["flux_gate"] for r in failing),
                       controls_converged=sum(r["start"] == start and r["converged"] for r in controls),
                       controls_flux_gate=sum(r["start"] == start and r["flux_gate"] for r in controls))
                       for start in STARTS},
                   control_difference_max=max(r["difference_from_payne_zero"]["max"] for r in controls
                                              if "difference_from_payne_zero" in r),
                   rows=rows)
    (root / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["counts"], indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory with states.json and decks/")
    parser.add_argument("--kurucz-home", type=Path)
    parser.add_argument("--collect", action="store_true", help="summarize finished runs against the flux gate")
    parser.add_argument("--gate", type=Path, default=Path("results/m_star_emulator_v1r2_marcs100/flux_gate.json"))
    parser.add_argument("--references", type=Path, help="JSON map state name -> converged Payne Zero product")
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.collect:
        collect(root, args.gate)
        return
    states = json.loads((root / "states.json").read_text())["states"]
    references = json.loads(args.references.read_text()) if args.references else {}
    payloads = [(s, start, args.kurucz_home.resolve(), root, references.get(s["name"])) for start in STARTS for s in states]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(worker, payloads))
    summary = dict(max_blocks=MAX_BLOCKS, iterations_per_block=30, dlnt_limit=DLNT_LIMIT, starts=STARTS, records=records)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
