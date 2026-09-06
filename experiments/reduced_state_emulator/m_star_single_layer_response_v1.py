"""Single-layer T-response sweep at the two certified-map walls.

For the wall-adjacent converged products -- 3300 K on the rich track and
3800 K on the metal-poor track -- build temperature-perturbed
atmospheres (T scaled by (1+delta) per layer, gas pressure held at the
baseline value), run exactly one production solver iteration on each,
and record the per-layer response of the convective state, Rosseland
opacity, electron density, and molecular-equilibrium Newton load.

The sweep is diagnostic: no physics is modified. It locates which
physical quantity responds first (and at which depth) as the
temperature moves toward the divergence walls, which is the evidence a
convection/opacity adaptation decision needs.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from . import m_star_pipeline as pipeline
from . import m_star_iteration_tomography_v1 as tomography
from .cool_star_step_test import (
    _clone_atmosphere,
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from .m_star_bootstrap_v1 import _write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_single_layer_response_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
TOMOGRAPHY_ROOT = REPO_ROOT / "results" / "m_star_iteration_tomography_v1"
SCALEOUT_ROOT = REPO_ROOT / "results" / "m_star_pipeline_scaleout_v1"
DELTAS = (-0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10)

BASELINES = {
    "rich_3300": {
        "product_dir": TOMOGRAPHY_ROOT
        / "cases"
        / "dwarf"
        / "g+4.50_m+0.00_a+0.00_c+0.00_x1.00"
        / "t3300"
        / "products"
        / "primary",
        "temperature_K": 3300.0,
        "track": dict(
            stellar_class="dwarf",
            log_surface_gravity=4.5,
            metallicity=0.0,
            microturbulence_km_s=1.0,
        ),
    },
    "poor_3800": {
        "product_dir": SCALEOUT_ROOT
        / "cases"
        / "dwarf_g+4.50_m-1.00_t3800"
        / "cap60"
        / "products"
        / "primary",
        "temperature_K": 3800.0,
        "track": dict(
            stellar_class="dwarf",
            log_surface_gravity=4.5,
            metallicity=-1.0,
            microturbulence_km_s=1.0,
        ),
    },
}


def _baseline_atmosphere(product_dir: Path, temperature_k: float):
    products = sorted(Path(product_dir).glob("*.npz"))
    if not products:
        raise FileNotFoundError(f"no baseline product in {product_dir}")
    track = pipeline.track_payload(**_baseline_track_spec(product_dir))
    labels = pipeline.labels_for(track, temperature_k)
    return pipeline.continuation_seed(products[0], labels), products[0]


def _baseline_track_spec(product_dir: Path) -> dict[str, str]:
    for name, spec in BASELINES.items():
        if str(spec["product_dir"]) == str(product_dir):
            return dict(spec["track"])
    raise KeyError(product_dir)


def _response_summary(iterations_dir: Path) -> dict[str, Any]:
    paths = sorted(iterations_dir.glob("iter_*.npz"))
    if not paths:
        return {"error": "no recorded iterations"}
    with np.load(paths[-1], allow_pickle=False) as data:
        log_tau = np.asarray(data["log_tau_standard"], dtype=np.float64)
        return {
            "log_tau": log_tau.tolist(),
            "sag": np.asarray(data["superadiabatic_gradient"], dtype=np.float64).tolist(),
            "flux_ratio": np.asarray(data["flux_ratio"], dtype=np.float64).tolist(),
            "log_kappa": np.log10(
                np.maximum(
                    np.asarray(data["rosseland_opacity_post"], dtype=np.float64),
                    1.0e-300,
                )
            ).tolist(),
            "log_ne": np.log10(
                np.maximum(
                    np.asarray(data["electron_density_post"], dtype=np.float64),
                    1.0e-300,
                )
            ).tolist(),
            "newton_max": int(
                np.max(data["molecular_newton_iterations"])
            ),
            "lstsq_count": int(
                np.sum(data["molecular_newton_used_lstsq"])
            ),
        }


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    _set_single_thread_environment()
    result_root = Path(args.result_root)
    sweep: dict[str, Any] = {"baselines": {}}
    for name, spec in BASELINES.items():
        baseline_atmosphere, baseline_product = _baseline_atmosphere(
            spec["product_dir"], spec["temperature_K"]
        )
        baseline_record = {
            "product": str(baseline_product),
            "temperature_K": spec["temperature_K"],
        }
        track = pipeline.track_payload(**spec["track"])
        labels = pipeline.labels_for(track, spec["temperature_K"])
        curves: dict[str, Any] = {}
        for delta in DELTAS:
            perturbed = _clone_atmosphere(baseline_atmosphere)
            perturbed.temperature = perturbed.temperature * (1.0 + float(delta))
            case_root = result_root / "cases" / name / f"delta_{delta:+.2f}"
            hook = tomography.make_tomography_hook(case_root / "iterations")
            started = time.perf_counter()
            record, _state = _solve_attempt(
                track=pipeline._track(track),
                method=f"response_sweep_delta_{delta:+.2f}",
                schedule="single_layer_response",
                source_temperature=None,
                target_labels=labels,
                initial_atmosphere=perturbed,
                product_dir=case_root / "products",
                iteration_cap=1,
                after_iteration_hook=hook,
            )
            curves[f"{delta:+.2f}"] = {
                "converged": record.get("converged"),
                "iterations": record.get("iterations"),
                "seconds": float(time.perf_counter() - started),
                "response": _response_summary(case_root / "iterations"),
            }
            print(
                f"{name} delta {delta:+.2f}: done in "
                f"{curves[f'{delta:+.2f}']['seconds']:.1f}s"
            )
        sweep["baselines"][name] = {
            "baseline": baseline_record,
            "curves": curves,
        }
    protocol = {
        "campaign": CAMPAIGN,
        "deltas": list(DELTAS),
        "baselines": {
            name: spec["temperature_K"] for name, spec in BASELINES.items()
        },
    }
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    sweep["protocol_hash"] = protocol["protocol_hash"]
    _write_json(result_root / "protocol.json", protocol)
    _write_json(result_root / "response_sweep.json", sweep)
    print(f"protocol hash {protocol['protocol_hash']}")
    return sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    args = parser.parse_args(argv)
    run_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
