"""Exhaust the remaining donor seeds at the two divergence boundaries.

Two probe groups, production physics unchanged:

- poor 3662.5 K ([M/H] -1.0): the fine walk only tried the 3700 K
  waypoint product as donor; the three newer waypoints (3687.5,
  3681.25, 3675 K) were never tried.
- rich 3275 K ([M/H] 0): the wall probe tried the 3325 K chain, the
  certified 3300 K and 3400 K products; the certified 3500 K product
  was never tried.

Each probe: reconstruct from one donor product, one production solve at
the wall temperature, record pass/diverge and the residual signature.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .cool_star_step_test import (
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from . import m_star_pipeline as pipeline
from . import m_star_bootstrap_v1r2_marcs100 as base
from . import m_star_iteration_tomography_v1 as tomography
from .m_star_bootstrap_v1 import _load_mt, _write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_donor_exhaust_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
TOMOGRAPHY_ROOT = REPO_ROOT / "results" / "m_star_iteration_tomography_v1"
DONOR_WALK_ROOT = REPO_ROOT / "results" / "m_star_donor_walk_3600_v1"

FINE_CONTINUATION_DIR = (
    REPO_ROOT
    / "results"
    / "m_star_fine_donor_walk_v1"
    / "cases"
    / "dwarf_g+4.50_m-1.00_t3600"
    / "products"
    / "continuation"
)

PROBES = [
    {
        "probe_id": "poor_3662.5_from_3687.5",
        "class": "dwarf",
        "log_surface_gravity": 4.5,
        "metallicity": -1.0,
        "microturbulence_km_s": 1.0,
        "target_temperature_K": 3662.5,
        "donor_glob": "t03687.5_*.npz",
    },
    {
        "probe_id": "poor_3662.5_from_3681.25",
        "class": "dwarf",
        "log_surface_gravity": 4.5,
        "metallicity": -1.0,
        "microturbulence_km_s": 1.0,
        "target_temperature_K": 3662.5,
        "donor_glob": "t03681.2_*.npz",
    },
    {
        "probe_id": "poor_3662.5_from_3675",
        "class": "dwarf",
        "log_surface_gravity": 4.5,
        "metallicity": -1.0,
        "microturbulence_km_s": 1.0,
        "target_temperature_K": 3662.5,
        "donor_glob": "t03675.0_*.npz",
    },
    {
        "probe_id": "rich_3275_from_3500",
        "class": "dwarf",
        "log_surface_gravity": 4.5,
        "metallicity": 0.0,
        "microturbulence_km_s": 1.0,
        "target_temperature_K": 3275.0,
        "donor_product": TOMOGRAPHY_ROOT
        / "cases"
        / "dwarf"
        / "g+4.50_m+0.00_a+0.00_c+0.00_x1.00"
        / "t3500"
        / "products"
        / "primary"
        / "t03500.0_g+4.50_m+0.00_a+0.00_x1.00.npz",
    },
]


def _run_probe(probe: dict[str, Any], result_root: Path) -> dict[str, Any]:
    _set_single_thread_environment()
    track_payload = pipeline.track_payload(
        stellar_class=probe["class"],
        log_surface_gravity=probe["log_surface_gravity"],
        metallicity=probe["metallicity"],
        microturbulence_km_s=probe["microturbulence_km_s"],
    )
    track = base._track_from_payload(track_payload)
    labels = track.labels(float(probe["target_temperature_K"]))
    if "donor_glob" in probe:
        matches = sorted(FINE_CONTINUATION_DIR.glob(probe["donor_glob"]))
        if not matches:
            return {**probe, "status": "missing_donor",
                    "donor_glob": probe["donor_glob"]}
        donor_product = matches[0]
    else:
        donor_product = Path(probe["donor_product"])
        if not donor_product.is_file():
            return {**probe, "status": "missing_donor",
                    "donor_product": str(donor_product)}
    seed_mass, seed_profile = _load_mt(donor_product)
    try:
        seed = _reconstruct_from_mt(labels, seed_mass, seed_profile)
    except Exception as exc:  # noqa: BLE001 - a failed probe is an outcome
        return {**probe, "status": "reconstruction_failed",
                "error": f"{type(exc).__name__}: {exc}"}
    started = time.perf_counter()
    record, _state = _solve_attempt(
        track=track,
        method="donor_exhaust_probe",
        schedule="wall_exhaustion",
        source_temperature=None,
        target_labels=labels,
        initial_atmosphere=seed,
        product_dir=result_root / "cases" / probe["probe_id"] / "products",
        iteration_cap=60,
        maximum_all_layer_relative_temperature_change=(
            pipeline.STRICT_ALL_LAYER_LIMIT
        ),
    )
    fd = (record.get("solver_diagnostics") or {}).get("final_diagnostics", {})
    return {
        **probe,
        "donor_product": str(donor_product),
        "status": "pass" if record.get("survives_solver") else "diverged",
        "converged": record.get("converged"),
        "iterations": record.get("iterations"),
        "p95_absolute_flux_error_percent": fd.get(
            "p95_absolute_flux_error_percent"
        ),
        "seconds": float(time.perf_counter() - started),
    }


def run_campaign(args: argparse.Namespace) -> int:
    result_root = Path(args.result_root)
    _set_single_thread_environment()
    results = []
    for probe in PROBES:
        if not Path(probe["donor_product"]).is_file():
            print(f"{probe['probe_id']}: donor missing, skipped")
            continue
        result = _run_probe(probe, result_root)
        results.append(result)
        print(
            f"{result['probe_id']}: {result.get('status')} "
            f"iters={result.get('iterations')} "
            f"p95={result.get('p95_absolute_flux_error_percent')}"
        )
    _write_json(
        result_root / "donor_exhaust.json",
        {
            "campaign": CAMPAIGN,
            "probes": results,
            "status": "complete",
        },
    )
    passing = [r for r in results if r.get("status") == "pass"]
    print(
        f"summary: {len(passing)}/{len(results)} probes converged -- "
        + (
            "walls breached at: " + ", ".join(r["probe_id"] for r in passing)
            if passing
            else "both walls hold under every remaining donor"
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    args = parser.parse_args(argv)
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
