"""Close the three cool undetermined points with gravity walks.

Each point's reference is walked from a converged corpus truth at the same
metallicity and nearly the same temperature but a different logg (the remedy
that worked for `t3250`): logg legs of 0.25 dex at fixed temperature, then
one 25 K temperature leg into the target.  The reference is then relaxed
under the frozen stop rule (cap 120, within the 240 target budget) and
compared against the point's existing emulator arm (cap 90) with the
standard acceptance: both arms stable, cross-arm TiO `<= 5e-3`, structure
differences reported.  Anchors are corpus products, never emulator output.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from emulator_v1_2.gates.compare_spectra import (
    _absolute_stats,
    _continuum_scaled_stats,
    _load_spectrum_npz,
    _relative_stats,
)

from .cool_star_step_test import (
    _reconstruct_from_mt,
    _set_single_thread_environment,
)
from .m_star_bootstrap_v1 import _load_mt, _write_json
from .m_star_stop_rule_v1 import (
    _continue_arm,
    _labels_for,
    _node_id,
)
from .m_star_sixty_budget_v1 import (
    EMULATOR_CAP,
    GATE_BAR,
    REFERENCE_RELAX_CAP,
    WALK_LEG_DEX,
    WALK_LEG_K,
    _cross_metrics,
    _corpus_product,
    _frozen_iteration,
    _flux_at,
    _track,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_gravity_walk_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_CORPUS = (
    REPO_ROOT / "results" / "m_star_cool_corpus_mgiant_v2" / "cool_truth_corpus.npz"
)
DEFAULT_FLUX_GATE = REPO_ROOT / "results" / "m_star_emulator_v1" / "flux_gate.json"
DEFAULT_SIXTY_ROOT = REPO_ROOT / "results" / "m_star_sixty_budget_v1"

# (teff, logg, metallicity, anchor (teff, logg), waypoint (logg, teff) list)
TARGETS = (
    (
        3525.0,
        2.25,
        -1.0,
        (3500.0, 2.5),
        ((2.25, 3500.0), (2.25, 3525.0)),
    ),
    (
        3175.0,
        2.25,
        0.5,
        (3200.0, 1.5),
        ((1.75, 3200.0), (2.0, 3200.0), (2.25, 3200.0), (2.25, 3175.0)),
    ),
    (
        3325.0,
        1.5,
        -0.5,
        (3300.0, 2.5),
        ((2.25, 3300.0), (2.0, 3300.0), (1.75, 3300.0), (1.5, 3300.0), (1.5, 3325.0)),
    ),
)


def _walk_and_relax(
    *,
    teff: float,
    logg: float,
    metallicity: float,
    anchor: tuple[float, float],
    waypoints: tuple[tuple[float, float], ...],
    corpus: Path,
    arm_dir: Path,
    gate: dict,
) -> dict[str, Any]:
    anchor_product = _corpus_product(
        corpus, anchor[1], metallicity, anchor[0]
    )
    if anchor_product is None:
        return {"status": "no_anchor"}
    _product, mass, temperature = anchor_product
    report: dict[str, Any] = {"anchor": str(_product), "legs": []}
    current = (anchor[1], anchor[0])
    for leg_logg, leg_teff in waypoints:
        labels = _labels_for(leg_teff, leg_logg, metallicity)
        seed = _reconstruct_from_mt(labels, mass, temperature)
        from .cool_star_step_test import _solve_attempt  # local import keeps hot path

        leg, _state = _solve_attempt(
            track=_track(leg_logg, metallicity),
            method="gravity_walk_leg",
            schedule="gravity_walk",
            source_temperature=float(teff),
            target_labels=labels,
            initial_atmosphere=seed,
            product_dir=arm_dir
            / "walk"
            / f"g{leg_logg:+05.2f}_t{int(leg_teff):04d}",
            iteration_cap=60,
            maximum_all_layer_relative_temperature_change=5.0e-4,
        )
        survived = bool(leg.get("survives_solver"))
        report["legs"].append(
            {"logg": leg_logg, "teff": leg_teff, "survived": survived}
        )
        if not survived:
            report["status"] = "walk_failed"
            report["reached"] = current
            return report
        mass, temperature = _load_mt(leg["product_path"])
        current = (leg_logg, leg_teff)
    labels = _labels_for(teff, logg, metallicity)
    start = _reconstruct_from_mt(labels, mass, temperature)
    _continue_arm(
        labels=labels,
        start_atmosphere=start,
        arm_dir=arm_dir,
        cap=REFERENCE_RELAX_CAP,
        gate=gate,
    )
    report["status"] = "ok"
    report["reference_frozen"] = _frozen_iteration(arm_dir, gate)
    return report


def _run_one(payload: tuple) -> dict[str, Any]:
    (
        teff,
        logg,
        metallicity,
        anchor,
        waypoints,
        result_root_text,
        corpus_text,
        gate,
    ) = payload
    _set_single_thread_environment()
    node = _node_id(teff, logg, metallicity)
    arm_dir = Path(result_root_text) / "points" / node / "reference_walk"
    if (arm_dir / "iterations.jsonl").is_file():
        return {"node_id": node, "status": "already done"}
    report = _walk_and_relax(
        teff=teff,
        logg=logg,
        metallicity=metallicity,
        anchor=anchor,
        waypoints=waypoints,
        corpus=Path(corpus_text),
        arm_dir=arm_dir,
        gate=gate,
    )
    report["node_id"] = node
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("walk", "evaluate"), default="walk"
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--flux-gate", type=Path, default=DEFAULT_FLUX_GATE)
    parser.add_argument(
        "--sixty-root", type=Path, default=DEFAULT_SIXTY_ROOT
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    gate = json.loads(args.flux_gate.read_text())

    if args.stage == "walk":
        from concurrent.futures import ProcessPoolExecutor

        payloads = [
            (
                teff,
                logg,
                metallicity,
                anchor,
                waypoints,
                str(args.result_root),
                str(args.corpus),
                gate,
            )
            for teff, logg, metallicity, anchor, waypoints in TARGETS
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for report in pool.map(_run_one, payloads):
                print(json.dumps(report, sort_keys=True), flush=True)
        return 0

    rows = []
    spectra_dir = args.result_root / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    for teff, logg, metallicity, _anchor, _waypoints in TARGETS:
        node = _node_id(teff, logg, metallicity)
        point_dir = args.result_root / "points" / node
        reference_frozen = _frozen_iteration(
            point_dir / "reference_walk", gate
        )
        emulator_arm = args.sixty_root / "points" / node / "emulator"
        emulator_frozen = _frozen_iteration(emulator_arm, gate)
        row = {
            "node_id": node,
            "teff_K": teff,
            "logg": logg,
            "metallicity": metallicity,
            "emulator_frozen": emulator_frozen,
            "reference_frozen": reference_frozen,
        }
        if emulator_frozen is not None and reference_frozen is not None:
            metrics = _cross_metrics(
                emulator_arm / f"iter_{emulator_frozen:04d}.npz",
                point_dir
                / "reference_walk"
                / f"iter_{reference_frozen:04d}.npz",
                spectra_dir,
                node,
            )
            row.update(metrics)
            row["verdict"] = (
                "pass"
                if max(metrics["tiO"].values()) <= GATE_BAR
                else "consistency_fail"
            )
            row["budget"] = (
                "within_60" if emulator_frozen <= 60 else "over_budget"
            )
        else:
            row["verdict"] = "undetermined"
        rows.append(row)
        print(json.dumps(row, default=str)[:400], flush=True)
    _write_json(args.result_root / "gravity_walk_verdicts.json", {"rows": rows})
    print(
        json.dumps(
            {
                "pass": sum(1 for r in rows if r["verdict"] == "pass"),
                "undetermined": sum(
                    1 for r in rows if r["verdict"] == "undetermined"
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
