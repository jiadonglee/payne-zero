"""Scale the certified cold-star pipeline across giants, caps, metallicities.

Every node runs the same pipeline entry point (``m_star_pipeline``:
native MARCS seed, production solve, strict self-restart, frozen flux
gate, path consistency, certification phase guard). Three arms:

- ``G`` giants: logg {1.5, 2.5} x Teff {3500, 3750, 4000} at [M/H] 0;
- ``M`` metallicity dwarfs: logg 4.5 x Teff {3600, 3800, 4000} x
  [M/H] {-1.0, +0.5};
- ``I`` iteration caps: three anchor nodes solved at caps 30/60/120 to
  show convergence-iteration counts and cap-independence of the outcome.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from . import m_star_pipeline as pipeline
from .cool_star_step_test import _set_single_thread_environment
from .m_star_bootstrap_v1 import _run_workers, _write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_pipeline_scaleout_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_GATE_PATH = (
    REPO_ROOT / "results" / "m_star_iteration_tomography_v1" / "flux_gate.json"
)
PREREGISTRATION_PATH = (
    REPO_ROOT
    / "notes"
    / "m_star_pipeline_scaleout_v1_preregistration_20260905.md"
)

GIANT_NODES = [
    (logg, teff)
    for logg in (1.5, 2.5)
    for teff in (3500.0, 3750.0, 4000.0)
]
METALLICITY_NODES = [
    (teff, metallicity)
    for teff in (3600.0, 3800.0, 4000.0)
    for metallicity in (-1.0, 0.5)
]
CAP_NODES = [
    ("giant", 2.5, 0.0, 3750.0),
    ("dwarf", 4.5, 0.0, 4000.0),
    ("dwarf", 4.5, 0.0, 3500.0),
]
CAPS = (30, 60, 120)


def _nodes() -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for logg, teff in GIANT_NODES:
        nodes.append(
            {
                "node_id": f"giant_g+{logg:.2f}_m+0.00_t{int(teff)}",
                "arm": "G",
                "stellar_class": "giant",
                "log_surface_gravity": logg,
                "metallicity": 0.0,
                "temperature_K": teff,
                "caps": (60,),
            }
        )
    for teff, metallicity in METALLICITY_NODES:
        nodes.append(
            {
                "node_id": f"dwarf_g+4.50_m{metallicity:+.2f}_t{int(teff)}",
                "arm": "M",
                "stellar_class": "dwarf",
                "log_surface_gravity": 4.5,
                "metallicity": metallicity,
                "temperature_K": teff,
                "caps": (60,),
            }
        )
    for stellar_class, logg, metallicity, teff in CAP_NODES:
        nodes.append(
            {
                "node_id": (
                    f"{stellar_class}_g+{logg:.2f}_m{metallicity:+.2f}"
                    f"_t{int(teff)}"
                ),
                "arm": "I",
                "stellar_class": stellar_class,
                "log_surface_gravity": logg,
                "metallicity": metallicity,
                "temperature_K": teff,
                "caps": CAPS,
            }
        )
    return nodes


def _case_worker(payload: tuple[Any, ...]) -> dict[str, Any]:
    (
        node,
        result_root_text,
        marcs_grid_text,
        gate,
        protocol_hash,
    ) = payload
    _set_single_thread_environment()
    result_root = Path(result_root_text)
    case_path = result_root / "cases" / f"{node['node_id']}_pipeline.json"
    if case_path.is_file():
        existing = json.loads(case_path.read_text())
        if existing.get("status") == "complete":
            return existing

    track = pipeline.track_payload(
        stellar_class=node["stellar_class"],
        log_surface_gravity=node["log_surface_gravity"],
        metallicity=node["metallicity"],
        microturbulence_km_s=pipeline.MICROTURBULENCE[node["stellar_class"]],
    )
    attempts: dict[str, Any] = {}
    try:
        for cap in node["caps"]:
            labels = pipeline.labels_for(track, node["temperature_K"])
            seed = pipeline.marcs_seed(
                labels, marcs_grid=Path(marcs_grid_text)
            )
            attempts[str(cap)] = pipeline.certified_solve(
                labels=labels,
                track=track,
                candidate_id=node["node_id"],
                initial_atmosphere=seed,
                product_dir=result_root / "cases" / node["node_id"] / f"cap{cap}",
                flux_gate=gate,
                iteration_cap=int(cap),
            )
        caps = sorted(attempts)
        outcome = attempts[caps[-1]]
        eligible_by_cap = {cap: bool(attempts[cap]["training_eligible"]) for cap in caps}
        converged_by_cap = {
            cap: bool((attempts[cap]["primary"] or {}).get("converged")) for cap in caps
        }
        cap_independent = len(set(eligible_by_cap.values())) == 1
        output = {
            "campaign": CAMPAIGN,
            "protocol_hash": protocol_hash,
            **node,
            "attempts": attempts,
            "eligible_by_cap": eligible_by_cap,
            "converged_by_cap": converged_by_cap,
            "cap_independent": cap_independent,
            "status": "complete",
        }
    except Exception as exc:  # noqa: BLE001 - a failed node is an outcome
        output = {
            "campaign": CAMPAIGN,
            "protocol_hash": protocol_hash,
            **node,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    _write_json(case_path, output)
    return output


def run_protocol(args: argparse.Namespace) -> dict[str, Any]:
    result_root = Path(args.result_root)
    gate = json.loads(Path(args.gate_path).read_text())
    protocol = {
        "campaign": CAMPAIGN,
        "preregistration": str(PREREGISTRATION_PATH),
        "arms": {"G": "giants", "M": "metallicity dwarfs", "I": "iteration caps"},
        "nodes": [node["node_id"] for node in _nodes()],
        "caps": list(CAPS),
        "flux_gate_source": {
            "campaign": gate.get("campaign"),
            "gate_hash": gate.get("gate_hash"),
            "thresholds": gate.get("thresholds"),
        },
        "certification_phase_guard_required": True,
    }
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    _write_json(result_root / "protocol.json", protocol)
    print(f"protocol hash {protocol['protocol_hash']}")
    return protocol


def run_campaign(args: argparse.Namespace) -> int:
    result_root = Path(args.result_root)
    protocol = json.loads((result_root / "protocol.json").read_text())
    gate = json.loads(Path(args.gate_path).read_text())
    payloads = []
    for node in _nodes():
        if args.arm and node["arm"] != args.arm:
            continue
        payloads.append(
            (
                node,
                str(result_root),
                str(Path(args.marcs_grid)),
                gate,
                protocol["protocol_hash"],
            )
        )
    started = time.perf_counter()
    records = _run_workers(_case_worker, payloads, workers=int(args.workers))
    for record in sorted(records, key=lambda row: row["node_id"]):
        if record.get("status") != "complete":
            print(f"{record['node_id']}: ERROR {record.get('error')}")
            continue
        attempts = record["attempts"]
        parts = []
        for cap in sorted(attempts, key=int):
            block = attempts[cap]
            primary = block.get("primary") or {}
            p95 = (
                (block.get("primary_flux_gate") or {})
                .get("metrics", {})
                .get("p95_absolute_flux_error_percent", {})
                .get("value")
            )
            guard = (block.get("phase_guard") or {})
            parts.append(
                f"cap{cap}: elig={block.get('training_eligible')} "
                f"iters={primary.get('iterations')} "
                f"p95={round(p95, 2) if p95 is not None else '—'} "
                f"guard={guard.get('primary')}/{guard.get('restart')}"
            )
        print(f"{record['node_id']}  {' | '.join(parts)}")
    print(f"wall seconds {time.perf_counter() - started:.1f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    common.add_argument(
        "--marcs-grid", default=str(pipeline.DEFAULT_MARCS_GRID)
    )
    common.add_argument("--gate-path", default=str(DEFAULT_GATE_PATH))
    common.add_argument("--workers", type=int, default=6)
    common.add_argument("--arm", default=None, help="restrict to arm G, M, or I")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("protocol", parents=[common])
    sub.add_parser("run", parents=[common])
    args = parser.parse_args(argv)

    if args.stage == "protocol":
        run_protocol(args)
        return 0
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
