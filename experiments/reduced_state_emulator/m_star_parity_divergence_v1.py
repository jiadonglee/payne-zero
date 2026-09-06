"""Divergence-provenance parity for the [M/H]-1.0 wall below 3800 K.

The user authorized a single-layer parity study to establish whether the
disconnection below 3800 K (and the 3600 K divergence) is a numerical
instability or the real edge of the current 1D MLT physics. This
campaign re-runs the two diverged downwalk steps -- 3800 -> 3775 at
25 K and the 12.5 K retry to 3787.5 -- with the per-iteration recorder
attached, then localizes the onset: the first iteration where the raw
correction runs away, the depth where it ignites, and the co-located
convective, molecular-Newton, and opacity state at that layer. Those
are the observables a physics verdict needs.
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
    _reconstruct_from_mt,
    _set_single_thread_environment,
    _solve_attempt,
)
from . import m_star_bootstrap_v1r2_marcs100 as base
from .m_star_bootstrap_v1 import _load_mt, _write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_parity_divergence_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_CHAIN_PRODUCT = (
    REPO_ROOT
    / "results"
    / "m_star_pipeline_scaleout_v1"
    / "cases"
    / "dwarf_g+4.50_m-1.00_t4000"
    / "cap60"
    / "products"
    / "primary"
)
RUNS = (
    {"arm": "step25_to_3775", "source_temperature_K": 3800.0, "target_temperature_K": 3775.0},
    {"arm": "step125_to_3787", "source_temperature_K": 3800.0, "target_temperature_K": 3787.5},
)


def _run_hooked_attempt(
    *,
    run: dict[str, Any],
    track,
    chain_mass: np.ndarray,
    chain_profile: np.ndarray,
    result_root: Path,
) -> dict[str, Any]:
    labels = track.labels(float(run["target_temperature_K"]))
    seed = _reconstruct_from_mt(labels, chain_mass, chain_profile)
    case_root = result_root / "cases" / run["arm"]
    started = time.perf_counter()
    record, _state = _solve_attempt(
        track=track,
        method=f"parity_divergence_{run['arm']}",
        schedule="parity_divergence",
        source_temperature=float(run["source_temperature_K"]),
        target_labels=labels,
        initial_atmosphere=seed,
        product_dir=case_root / "products",
        iteration_cap=60,
        maximum_all_layer_relative_temperature_change=(
            pipeline.STRICT_ALL_LAYER_LIMIT
        ),
        after_iteration_hook=tomography.make_tomography_hook(
            case_root / "iterations"
        ),
    )
    return {
        "arm": run["arm"],
        "source_temperature_K": run["source_temperature_K"],
        "target_temperature_K": run["target_temperature_K"],
        "converged": record.get("converged"),
        "iterations": record.get("iterations"),
        "seconds": float(time.perf_counter() - started),
        "final_diagnostics": (record.get("solver_diagnostics") or {}).get(
            "final_diagnostics", {}
        ),
        "iterations_dir": str(case_root / "iterations"),
    }


def _onset_analysis(iterations_dir: Path) -> dict[str, Any]:
    paths = sorted(iterations_dir.glob("iter_*.npz"))
    if not paths:
        return {"error": "no recorded iterations"}
    worst_ratio, worst_layer, worst_log_tau = [], [], []
    newton_at_worst, sag_at_worst, kappa_at_worst = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            raw = np.asarray(data["raw_temperature_correction"], dtype=np.float64)
            pre = np.maximum(
                np.asarray(data["temperature_pre"], dtype=np.float64), 1.0
            )
            log_tau = np.asarray(data["log_tau_working"], dtype=np.float64)
            newton = np.asarray(
                data["molecular_newton_iterations"], dtype=np.float64
            )
            sag = np.asarray(data["superadiabatic_gradient"], dtype=np.float64)
            kappa = np.log10(
                np.maximum(
                    np.asarray(data["rosseland_opacity_post"], dtype=np.float64),
                    1.0e-300,
                )
            )
        ratio = np.abs(raw) / pre
        index = int(np.argmax(ratio))
        worst_ratio.append(float(ratio[index]))
        worst_layer.append(index)
        worst_log_tau.append(float(log_tau[index]))
        newton_at_worst.append(float(newton[index]))
        sag_at_worst.append(float(sag[index]))
        kappa_at_worst.append(float(kappa[index]))
    iterations = list(range(1, len(paths) + 1))
    onset = next(
        (i for i, value in enumerate(worst_ratio) if value > 0.05),
        None,
    )
    summary = {
        "iterations": iterations,
        "worst_ratio": worst_ratio,
        "worst_log_tau": worst_log_tau,
        "worst_layer": worst_layer,
        "newton_at_worst": newton_at_worst,
        "sag_at_worst": sag_at_worst,
        "log_kappa_at_worst": kappa_at_worst,
        "onset_iteration": None if onset is None else onset + 1,
        "onset_log_tau": None if onset is None else worst_log_tau[onset],
        "onset_ratio": None if onset is None else worst_ratio[onset],
        "onset_newton": None if onset is None else newton_at_worst[onset],
        "onset_sag": None if onset is None else sag_at_worst[onset],
        "onset_log_kappa": None if onset is None else kappa_at_worst[onset],
    }
    return summary


def run_campaign(args: argparse.Namespace) -> int:
    result_root = Path(args.result_root)
    _set_single_thread_environment()
    chain_products = sorted(Path(args.chain_product_dir).glob("*.npz"))
    if not chain_products:
        raise FileNotFoundError(f"no chain product in {args.chain_product_dir}")
    track_payload = pipeline.track_payload(
        stellar_class="dwarf",
        log_surface_gravity=4.5,
        metallicity=-1.0,
        microturbulence_km_s=pipeline.MICROTURBULENCE["dwarf"],
    )
    track = base._track_from_payload(track_payload)
    start_mass, start_profile = _load_mt(chain_products[0])

    attempts = []
    for run in RUNS:
        attempts.append(
            _run_hooked_attempt(
                run=run,
                track=track,
                chain_mass=start_mass,
                chain_profile=start_profile,
                result_root=result_root,
            )
        )
        attempts[-1]["onset"] = _onset_analysis(
            Path(attempts[-1]["iterations_dir"])
        )
        onset = attempts[-1]["onset"]
        print(
            "{arm}: converged={conv} iters={iters} | onset iter {oi} at "
            "log tau {ot} ratio {or_:.3g} newton {on} sag {os_:.3g}".format(
                arm=attempts[-1]["arm"],
                conv=attempts[-1]["converged"],
                iters=attempts[-1]["iterations"],
                oi=onset.get("onset_iteration"),
                ot=None if onset.get("onset_log_tau") is None else round(onset["onset_log_tau"], 2),
                or_=onset.get("onset_ratio") or 0,
                on=onset.get("onset_newton"),
                os_=onset.get("onset_sag") or 0,
            )
        )
    protocol = {
        "campaign": CAMPAIGN,
        "runs": RUNS,
        "chain_seed_product_sha256": hashlib.sha256(
            chain_products[0].read_bytes()
        ).hexdigest(),
    }
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    output = {
        "campaign": CAMPAIGN,
        "protocol_hash": protocol["protocol_hash"],
        "attempts": attempts,
        "status": "complete",
    }
    _write_json(result_root / "divergence_provenance.json", output)
    _write_json(result_root / "protocol.json", protocol)
    print(f"protocol hash {protocol['protocol_hash']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument(
        "--chain-product-dir", default=str(DEFAULT_CHAIN_PRODUCT)
    )
    args = parser.parse_args(argv)
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
