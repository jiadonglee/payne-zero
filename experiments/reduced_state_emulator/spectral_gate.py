"""Three-metric spectral gate on converged products from two initializers.

Iteration counts are not an acceptance criterion on their own. A faster
initializer is only useful if the solver still lands on the *same physical
answer*; if the converged atmosphere depends on where the iteration started,
the speedup is worthless. This gate measures that directly: synthesize from
each arm's converged structured atmosphere and compare.

The three metrics and the 5e-3 bar are Ting's, not invented here
(`emulator_v1_2/README.md`, `RESEARCH_LOG.md` Sec 10 -- the v1.2.1 milestone
passed all three below 5e-3, largest 0.00424). Their implementations are
imported from `emulator_v1_2.gates.compare_spectra` rather than reimplemented,
so a subtle redefinition cannot make this gate look easier than his:

``normalized_flux``   max absolute difference in normalized flux
``flux_total``        max full-flux difference divided by the reference continuum
``flux_continuum``    max relative continuum difference

Reference (baseline) is the production arm's converged product; candidate is
the learned arm's. Both solved the same star under the same policy, so a
difference above the bar means the initializer moved the fixed point.

Usage::

    export NUMBA_THREADING_LAYER=workqueue
    PYTHONPATH=.:.. .venv/bin/python -m experiments.reduced_state_emulator.spectral_gate \\
        --products-dir runs/reduced_state_emulator/products --count 12
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from emulator_v1_2.gates.compare_spectra import (
    _absolute_stats,
    _continuum_scaled_stats,
    _load_spectrum_npz,
    _relative_stats,
    _synthesize_one,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BAR = 5.0e-3
BASELINE_ARM = "production_six_field"
CANDIDATE_ARM = "learned_reduced_state"


def paired_slugs(products_dir: Path, baseline_arm=BASELINE_ARM, candidate_arm=CANDIDATE_ARM) -> list[str]:
    """Stars whose converged product exists for *both* arms.

    A star that failed in either arm has no converged product, so it cannot be
    gated. Those are reported as excluded rather than silently dropped: the
    learned arm's three failures are exactly the population most likely to
    differ spectrally, and hiding them would make the gate look better than it is.
    """

    baseline = {p.stem for p in (products_dir / baseline_arm).glob("*.npz")}
    candidate = {p.stem for p in (products_dir / candidate_arm).glob("*.npz")}
    return sorted(baseline & candidate)


def gate_one(
    slug: str,
    products_dir: Path,
    spectra_dir: Path,
    *,
    wavelength_start_nm: float,
    wavelength_end_nm: float,
    resolution: float,
    molecular_lines: bool,
    device: str | None,
    dtype: str,
    baseline_arm: str = BASELINE_ARM,
    candidate_arm: str = CANDIDATE_ARM,
) -> dict:
    paths = {}
    for arm in (baseline_arm, candidate_arm):
        out_path = spectra_dir / arm / f"{slug}.npz"
        if not out_path.is_file():
            _synthesize_one(
                products_dir / arm / f"{slug}.npz",
                out_path,
                wavelength_start_nm=wavelength_start_nm,
                wavelength_end_nm=wavelength_end_nm,
                resolution=resolution,
                molecular_lines=molecular_lines,
                device=device,
                dtype=dtype,
            )
        paths[arm] = out_path

    baseline = _load_spectrum_npz(paths[baseline_arm])
    candidate = _load_spectrum_npz(paths[candidate_arm])

    return {
        "slug": slug,
        "normalized_flux": _absolute_stats(
            candidate["normalized_flux"], baseline["normalized_flux"]
        ),
        "flux_total": _continuum_scaled_stats(
            candidate["flux_total"], baseline["flux_total"], baseline["flux_continuum"]
        ),
        "flux_continuum": _relative_stats(
            candidate["flux_continuum"], baseline["flux_continuum"]
        ),
    }


def _gate_worker(payload: tuple) -> dict:
    """Process-pool wrapper with only pickle-safe arguments."""

    (
        slug,
        products_dir,
        spectra_dir,
        wavelength_start_nm,
        wavelength_end_nm,
        resolution,
        molecular_lines,
        device,
        dtype,
        baseline_arm,
        candidate_arm,
    ) = payload
    return gate_one(
        slug,
        Path(products_dir),
        Path(spectra_dir),
        wavelength_start_nm=wavelength_start_nm,
        wavelength_end_nm=wavelength_end_nm,
        resolution=resolution,
        molecular_lines=molecular_lines,
        device=device,
        dtype=dtype,
        baseline_arm=baseline_arm,
        candidate_arm=candidate_arm,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--products-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "reduced_state_emulator" / "products",
    )
    parser.add_argument(
        "--spectra-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "reduced_state_emulator" / "spectra",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "spectral_gate.json")
    parser.add_argument("--count", type=int, default=None, help="gate only the first N pairs")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel spectrum workers; float64 cluster runs can use several",
    )
    parser.add_argument("--wavelength-start-nm", type=float, default=400.0)
    parser.add_argument("--wavelength-end-nm", type=float, default=900.0)
    parser.add_argument("--resolution", type=float, default=20000.0)
    parser.add_argument("--bar", type=float, default=DEFAULT_BAR)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=("float64", "float32"),
        default="float64",
        help=(
            "float64 matches Ting's gate and is what the reported numbers must "
            "use. float32 exists only so the pipeline can be exercised on Apple "
            "Metal, which has no float64 at all; its metric values are not "
            "comparable to the 5e-3 bar."
        ),
    )
    parser.add_argument("--no-molecular-lines", action="store_true")
    parser.add_argument("--baseline-arm", default=BASELINE_ARM)
    parser.add_argument("--candidate-arm", default=CANDIDATE_ARM)
    args = parser.parse_args(argv)

    slugs = paired_slugs(args.products_dir, args.baseline_arm, args.candidate_arm)
    baseline_only = {p.stem for p in (args.products_dir / args.baseline_arm).glob("*.npz")}
    candidate_only = {p.stem for p in (args.products_dir / args.candidate_arm).glob("*.npz")}
    excluded = sorted((baseline_only | candidate_only) - set(slugs))
    print(
        f"{len(slugs)} gateable pairs; {len(excluded)} stars excluded because one "
        f"arm produced no converged atmosphere",
        flush=True,
    )
    for slug in excluded:
        which = args.baseline_arm if slug in baseline_only else args.candidate_arm
        print(f"  excluded {slug} (only {which} converged)", flush=True)

    if args.count is not None:
        slugs = slugs[: args.count]
        print(f"gating the first {len(slugs)}", flush=True)

    payloads = [
        (
            slug,
            str(args.products_dir),
            str(args.spectra_dir),
            args.wavelength_start_nm,
            args.wavelength_end_nm,
            args.resolution,
            not args.no_molecular_lines,
            args.device,
            args.dtype,
            args.baseline_arm,
            args.candidate_arm,
        )
        for slug in slugs
    ]
    started = time.perf_counter()
    if args.workers <= 1:
        rows = []
        for index, payload in enumerate(payloads, start=1):
            row = _gate_worker(payload)
            rows.append(row)
            print(
                f"[{index}/{len(slugs)}] {row['slug']}  "
                f"norm={row['normalized_flux']['max']:.3e}  "
                f"total={row['flux_total']['max']:.3e}  "
                f"cont={row['flux_continuum']['max']:.3e}  "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_gate_worker, payloads))
        for index, row in enumerate(rows, start=1):
            print(
                f"[{index}/{len(rows)}] {row['slug']}  "
                f"norm={row['normalized_flux']['max']:.3e}  "
                f"total={row['flux_total']['max']:.3e}  "
                f"cont={row['flux_continuum']['max']:.3e}",
                flush=True,
            )

    fields = ("normalized_flux", "flux_total", "flux_continuum")
    summary = {
        "bar": args.bar,
        "dtype": args.dtype,
        "device": args.device,
        "baseline_arm": args.baseline_arm,
        "candidate_arm": args.candidate_arm,
        "window_nm": [args.wavelength_start_nm, args.wavelength_end_nm],
        "resolution": args.resolution,
        "gated_star_count": len(rows),
        "excluded_stars": excluded,
        "per_star": rows,
    }
    for field in fields:
        maxima = [row[field]["max"] for row in rows]
        summary[field] = {
            "max_over_stars": float(np.max(maxima)) if maxima else None,
            "median_over_stars": float(np.median(maxima)) if maxima else None,
            "stars_over_bar": int(sum(m > args.bar for m in maxima)),
            "passes": bool(maxima) and bool(np.max(maxima) <= args.bar),
        }
    summary["all_pass"] = all(summary[field]["passes"] for field in fields)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    print(f"{'metric':>18s} {'max':>11s} {'median':>11s} {'over bar':>9s} {'pass':>6s}")
    for field in fields:
        entry = summary[field]
        print(
            f"{field:>18s} {entry['max_over_stars']:>11.3e} "
            f"{entry['median_over_stars']:>11.3e} {entry['stars_over_bar']:>9d} "
            f"{str(entry['passes']):>6s}"
        )
    print(f"\nall three below {args.bar:g}: {summary['all_pass']}")
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
