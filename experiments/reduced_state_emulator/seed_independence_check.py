#!/usr/bin/env python3
"""Seed-independence check for reduced-state reconstruction.

The reconstruction in ``reduced_state/reconstruct.py`` pins (m, T) and
re-derives P, n_e, kappa_R, g_rad by physical synchronization. This script
verifies that the *choice of seed* for those derived fields is immaterial:
the historical six-field emulator seed, the physics-default seed
(``P = g*m``, ``n_e = 1e-4*P/kT``), and deliberately perturbed physical
seeds (electron fraction 1e-6 / 1e-2, pressure scale x0.5 / x2) must all
synchronize to the same atmosphere within the sync tolerance.

Stars are drawn from the frozen truth corpus and spread over Teff so the
check covers the label grid, not one corner of it.

Two-phase design: on this workstation a process that has run the torch-based
emulator warm start segfaults when the numba continuum kernels load
afterwards (and vice versa is only safe in one import order). The emulator
seed decks are therefore built in a fresh subprocess that never imports the
solver pipeline; the main process never imports torch. The decks are cached
next to the output file, so reruns skip phase 1.

Usage::

    PYTHONPATH=. python experiments/reduced_state_emulator/seed_independence_check.py \
        --stars 12 --workers 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from continuity.closure import TARGET_FIELDS, load_corpus
from payne_zero_atmosphere.data_files import atmosphere_emulator_dir

LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)

ARMS = (
    ("emulator_seed", "emulator", None),
    ("physical_seed", "physical", None),
    ("physical_ne_1e-6", "physical", {"electron_fraction": 1.0e-6}),
    ("physical_ne_1e-2", "physical", {"electron_fraction": 1.0e-2}),
    ("physical_p_x0.5", "physical", {"pressure_scale": 0.5}),
    ("physical_p_x2", "physical", {"pressure_scale": 2.0}),
)

REFERENCE_ARM = "physical_seed"
# The adaptive pressure-only stop (any tolerance) halts as soon as pressure
# settles, which is one pass before n_e/kappa_R finish contracting: the
# runtime state computes total_nuclei = P/kT - n_e from the *seed's* electron
# density before the molecular solve, so n_e seed memory decays by ~2 orders
# of magnitude per synchronization pass. To demonstrate that all seeds share
# one fixed point, use the fixed-pass mode with enough passes for that
# contraction to bottom out, then compare at 1e-4 dex.
FIXED_SYNCHRONIZATIONS = 3
PASS_TOLERANCE_DEX = 1.0e-4

DEX_FIELDS = ("gas_pressure", "electron_density", "rosseland_opacity")
REL_FIELDS = ("radiative_acceleration",)


def corpus_path() -> Path:
    return atmosphere_emulator_dir() / "five_label" / "strict_truth_52199.npz"


def select_stars(labels: list[dict], count: int) -> list[int]:
    """Evenly Teff-spaced deterministic subset, endpoints included."""

    teff = np.asarray(
        [float(entry["effective_temperature"]) for entry in labels],
        dtype=np.float64,
    )
    order = np.argsort(teff, kind="stable")
    positions = np.linspace(0, len(order) - 1, min(count, len(order)))
    return sorted(int(order[int(round(p))]) for p in positions)


def _max_dex(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.any(a <= 0.0) or np.any(b <= 0.0):
        raise ValueError("dex comparison requires positive values")
    return float(np.max(np.abs(np.log10(a) - np.log10(b))))


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    scale = np.maximum(np.abs(b), 1.0e-300)
    return float(np.max(np.abs(a - b) / scale))


def build_emulator_decks(labels_json: Path, out_json: Path) -> None:
    """Phase 1 (subprocess): emulator warm-start decks for the given labels.

    Imports torch but never the solver pipeline; see the module docstring.
    """

    from payne_zero_atmosphere.warm_start import emulator_warm_start_model

    entries = json.loads(labels_json.read_text())
    decks = {}
    for entry in entries:
        print(f"building emulator deck for star {entry['star']}", flush=True)
        _atmosphere, deck = emulator_warm_start_model(
            device="cpu", **entry["labels"]
        )
        decks[str(entry["star"])] = deck
    out_json.write_text(json.dumps(decks))


def emulator_seed_atmospheres(
    decks: dict[str, str],
    stars: list[int],
    profiles: np.ndarray,
    field_index: dict[str, int],
) -> dict[int, "ModelAtmosphere"]:
    """Parse phase-1 decks and pin the exact truth (m, T), as the historical
    ``_seed_atmosphere`` did after calling the emulator."""

    from payne_zero_atmosphere.atmosphere_io import (
        ModelAtmosphere,
        parse_atmosphere_deck,
    )

    seeds: dict[int, ModelAtmosphere] = {}
    for star in stars:
        seeded = parse_atmosphere_deck(
            decks[str(star)], source=f"<emulator-seed star {star}>"
        )
        seeds[star] = dataclasses.replace(
            seeded,
            column_mass=np.asarray(
                profiles[star, :, field_index["column_mass"]], dtype=np.float64
            ),
            temperature=np.asarray(
                profiles[star, :, field_index["temperature"]], dtype=np.float64
            ),
        )
    return seeds


def _worker(payload):
    """Reconstruct one (star, arm) pair; runs in a pool worker."""

    index, star, column_mass, temperature, label_dict, seed, seed_kwargs, sync_passes = payload
    from reduced_state.reconstruct import (
        ReducedAtmosphere,
        reconstruct_full_atmosphere,
    )

    started = time.perf_counter()
    try:
        result = reconstruct_full_atmosphere(
            ReducedAtmosphere(
                column_mass=column_mass, temperature=temperature, labels=label_dict
            ),
            n_synchronizations=sync_passes,
            seed=seed,
            seed_kwargs=seed_kwargs,
        )
        atmosphere = result.atmosphere
        return index, {
            "star_index": star,
            "synchronized": bool(result.synchronized),
            "n_evaluations": int(result.n_evaluations),
            "seconds": time.perf_counter() - started,
            "fields": {
                field: np.asarray(getattr(atmosphere, field), dtype=np.float64)
                for field in DEX_FIELDS + REL_FIELDS
            },
        }
    except Exception as exc:  # a failed arm is a data point, not a batch abort
        return index, {
            "star_index": star,
            "synchronized": False,
            "seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def _report_progress(done: int, total: int, key, outcome: dict) -> None:
    status = "ok" if outcome.get("synchronized") else "FAILED"
    print(
        f"[{done}/{total}] star {key[0]} {key[1]}: {status} "
        f"({outcome['seconds']:.1f}s)",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stars", type=int, default=12)
    parser.add_argument(
        "--sync-passes",
        type=int,
        default=FIXED_SYNCHRONIZATIONS,
        help="fixed synchronization passes per arm (raise to show the "
        "residual keeps contracting on the slowest stars)",
    )
    # Each reconstruction loads the full opacity/line catalogs into memory
    # (~GBs per process); on a 16 GB workstation only sequential runs are
    # safe. Raise this only on a node with the RAM to match.
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/seed_independence_20260819.json"),
    )
    parser.add_argument(
        "--star-indices",
        type=int,
        nargs="*",
        default=None,
        help="explicit corpus row indices; overrides --stars selection",
    )
    parser.add_argument(
        "--build-emulator-decks",
        nargs=2,
        metavar=("LABELS_JSON", "OUT_JSON"),
        help=argparse.SUPPRESS,  # phase-1 subprocess entry point
    )
    args = parser.parse_args()

    if args.build_emulator_decks is not None:
        build_emulator_decks(*(Path(p) for p in args.build_emulator_decks))
        return

    profiles, _tau_std, _iters, labels = load_corpus(corpus_path())
    field_index = {name: i for i, name in enumerate(TARGET_FIELDS)}
    stars = (
        sorted(args.star_indices)
        if args.star_indices
        else select_stars(labels, args.stars)
    )
    print(f"selected {len(stars)} stars: {stars}", flush=True)

    decks_path = args.out.with_suffix(".emulator_decks.json")
    if not decks_path.is_file():
        labels_path = args.out.with_suffix(".emulator_labels.json")
        labels_path.write_text(
            json.dumps(
                [
                    {
                        "star": star,
                        "labels": {
                            field: float(labels[star][field])
                            for field in LABEL_FIELDS
                        },
                    }
                    for star in stars
                ]
            )
        )
        print("phase 1: building emulator seed decks in a subprocess...", flush=True)
        repo_root = Path(__file__).resolve().parents[2]
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--build-emulator-decks",
                str(labels_path),
                str(decks_path),
            ],
            check=True,
            env={
                **os.environ,
                "PYTHONPATH": str(repo_root)
                + os.pathsep
                + os.environ.get("PYTHONPATH", ""),
            },
        )
    emulator_seeds = emulator_seed_atmospheres(
        json.loads(decks_path.read_text()), stars, profiles, field_index
    )

    payloads = []
    for star in stars:
        label_dict = {field: float(labels[star][field]) for field in LABEL_FIELDS}
        column_mass = np.asarray(
            profiles[star, :, field_index["column_mass"]], dtype=np.float64
        )
        temperature = np.asarray(
            profiles[star, :, field_index["temperature"]], dtype=np.float64
        )
        for arm_name, seed, seed_kwargs in ARMS:
            payloads.append(
                (
                    (star, arm_name),
                    star,
                    column_mass,
                    temperature,
                    label_dict,
                    emulator_seeds[star] if seed == "emulator" else seed,
                    seed_kwargs,
                    args.sync_passes,
                )
            )

    results: dict[tuple[int, str], dict] = {}
    if args.workers <= 1:
        iterator = map(_worker, payloads)
        for done, (key, outcome) in enumerate(iterator, start=1):
            results[key] = outcome
            _report_progress(done, len(payloads), key, outcome)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for done, (key, outcome) in enumerate(
                executor.map(_worker, payloads), start=1
            ):
                results[key] = outcome
                _report_progress(done, len(payloads), key, outcome)

    report = {"stars": {}, "pass": True, "sync_passes": args.sync_passes}
    for star in stars:
        reference = results.get((star, REFERENCE_ARM))
        star_report = {"arms": {}, "pass": True}
        if reference is None or "fields" not in reference:
            star_report["pass"] = False
            star_report["error"] = f"reference arm {REFERENCE_ARM} failed"
            report["pass"] = False
        else:
            for arm_name, _seed, _kwargs in ARMS:
                outcome = results.get((star, arm_name))
                entry = {
                    "synchronized": bool(outcome.get("synchronized")),
                    "n_evaluations": outcome.get("n_evaluations"),
                    "seconds": outcome["seconds"],
                }
                if "error" in outcome:
                    entry["error"] = outcome["error"].splitlines()[0]
                    star_report["pass"] = False
                    report["pass"] = False
                else:
                    for field in DEX_FIELDS:
                        entry[f"{field}_max_dex_vs_reference"] = _max_dex(
                            outcome["fields"][field], reference["fields"][field]
                        )
                    for field in REL_FIELDS:
                        entry[f"{field}_max_rel_vs_reference"] = _max_rel(
                            outcome["fields"][field], reference["fields"][field]
                        )
                    worst = max(
                        entry[f"{field}_max_dex_vs_reference"] for field in DEX_FIELDS
                    )
                    entry["worst_dex_vs_reference"] = worst
                    if not outcome["synchronized"] or worst > PASS_TOLERANCE_DEX:
                        star_report["pass"] = False
                        report["pass"] = False
                star_report["arms"][arm_name] = entry
        report["stars"][str(star)] = star_report

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\noverall pass: {report['pass']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
