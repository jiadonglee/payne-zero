"""The original Fortran ATLAS12 on the M dwarfs that Payne Zero does not converge.

Each dwarf is started twice: from the MARCS node as rebuilt by Payne Zero
(the start of the Payne Zero recovery, written as a Kurucz deck), and from
the nearest model of the Kurucz initial-atmosphere grid (the default of the
``kurucz`` package). ATLAS12 runs in blocks of 30 iterations with molecules,
TiO and H2O line opacity and the convection settings of the package, at most
four blocks (120 iterations). Convergence is the criterion of the package's
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
import sys
import time

import numpy as np

MAX_BLOCKS = 4
DLNT_LIMIT = 1e-4
STARTS = ("payne_zero_marcs", "kurucz_grid")


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


def worker(payload):
    state, start, kurucz_home, root, reference = payload
    sys.path.insert(0, str(kurucz_home))
    import kurucz

    kurucz.synthe = lambda *args, **kwargs: None
    if start == "payne_zero_marcs":
        deck = root / state["deck"]
        kurucz.generate_atmosphere = lambda output_file, *args, **kwargs: shutil.copy(deck, output_file)
    name = f"{state['name']}_{start}"
    log = io.StringIO()
    started = time.monotonic()
    with contextlib.redirect_stdout(log):
        result = kurucz.atlas12(name, state["effective_temperature"], state["log_surface_gravity"],
                                mh=state["metallicity"], am=0.0, vturb=state["microturbulence_km_s"],
                                max_iterations=MAX_BLOCKS - 1)
    text = log.getvalue()
    out = root / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "driver.log").write_text(text)
    base = kurucz_home / "grids" / name
    record = dict(name=name, state=state, start=start, crashed=result is None,
                  blocks=text.count("Running atlas12.exe (main calculation)"), seconds=time.monotonic() - started)
    iterations = sorted((base / "outfiles").glob("*.iter"))
    if iterations:
        shutil.copy(iterations[0], out / "last_block.iter")
        blocks = read_iterations(iterations[0])
        if len(blocks) >= 2:
            temperature = blocks[:, :, 2]
            dlnt = abs(temperature[-2] - temperature[-1]) / temperature[-1]
            finite = bool(np.isfinite(blocks[-1]).all())
            record.update(last_block_iterations=int(len(blocks)), finite=finite,
                          max_dlnt_layers_40_75=float(dlnt[39:75].max()),
                          max_abs_flux_error=float(abs(blocks[-1][:, 11]).max()),
                          max_abs_flux_derivative_error=float(abs(blocks[-1][:, 12]).max()))
            record["converged"] = bool(not record["crashed"] and finite and record["max_dlnt_layers_40_75"] < DLNT_LIMIT)
    record.setdefault("converged", False)
    atm = base / "atm" / f"{name}_t{int(state['effective_temperature']):05d}g{state['log_surface_gravity']:.2f}.atm"
    if atm.exists():
        shutil.copy(atm, out / "final.atm")
        mass, temperature = read_deck(atm)
        record["final_temperature_range"] = [float(temperature.min()), float(temperature.max())]
        if reference is not None:
            with np.load(reference) as d:
                record["difference_from_payne_zero"] = temperature_difference(
                    mass, temperature, d["column_mass"], d["temperature"])
    (out / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record.get(k) for k in ("name", "crashed", "converged", "blocks", "max_dlnt_layers_40_75")}),
          flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory with states.json and decks/")
    parser.add_argument("--kurucz-home", type=Path, required=True)
    parser.add_argument("--references", type=Path, help="JSON map state name -> converged Payne Zero product")
    parser.add_argument("--workers", type=int, default=9)
    args = parser.parse_args()
    root = args.root.resolve()
    states = json.loads((root / "states.json").read_text())["states"]
    references = json.loads(args.references.read_text()) if args.references else {}
    payloads = [(s, start, args.kurucz_home.resolve(), root, references.get(s["name"])) for start in STARTS for s in states]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(worker, payloads))
    summary = dict(max_blocks=MAX_BLOCKS, iterations_per_block=30, dlnt_limit=DLNT_LIMIT, starts=STARTS, records=records)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
