"""Inventory every admitted cool-star atmosphere product for the safe-zone library.

Read-only walk over the m-star campaign records.  A product is admitted when its
own campaign record marks it training-eligible (frozen flux gate, restart leg,
path consistency) and the primary npz is present on disk.  The safe-zone
boundary table (giant 3500-4000 K at logg 0.5-2.5; dwarf [M/H]=+0.5 at
Teff >= 3600 K, 0.0 at >= 3300 K, -1.0 at >= 3800 K) decides library entry;
admitted rows outside the boundary are listed but excluded.  Converged waypoint
products that never passed the flux gate are reported separately and never
enter the library.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import csv
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "m_star_cold_library_inventory_v1"

CAMPAIGN_PRIORITY = (
    "m_star_pipeline_complete_v1",
    "m_star_pipeline_scaleout_v1",
    "m_star_iteration_tomography_v1",
    "m_star_mgiant_corner_truth_v1",
    "m_star_giant_supplement_v2_cap240",
    "m_star_giant_supplement_v1",
    "m_star_emulator_v1r2_marcs100",
    "m_star_emulator_v1r3_dwarf_iter120",
    "m_star_atlas_continuation_opened_tracks_v1",
)

FLAT_CASE_GLOBS = {
    "m_star_emulator_v1r2_marcs100": "cases/*/*/*/case.json",
    "m_star_emulator_v1r3_dwarf_iter120": "cases/*/*/*/case.json",
    "m_star_iteration_tomography_v1": "cases/*/*/*/case.json",
    "m_star_atlas_continuation_opened_tracks_v1": "cases/*/*/*/case.json",
    "m_star_giant_supplement_v1": "cases/*/*/*/case.json",
    "m_star_giant_supplement_v2_cap240": "cases/*/*/*/case.json",
    "m_star_mgiant_corner_truth_v1": "cases/*/*/case.json",
    "m_star_downwalk_v1": "cases/*/case.json",
}

WALK_CAMPAIGNS = (
    "m_star_donor_walk_3600_v1",
    "m_star_fine_donor_walk_v1",
    "m_star_wall_probe_v1",
)

FROZEN_GATE_FALLBACK = {
    "p95_absolute_flux_error_percent": 9.557343255085982,
    "maximum_absolute_flux_error_percent": 23.151468755609834,
    "median_absolute_flux_error_percent": 0.23421537419224742,
}

FLUX_METRIC_NAMES = (
    "median_absolute_flux_error_percent",
    "p95_absolute_flux_error_percent",
    "maximum_absolute_flux_error_percent",
)

GIANT_TEFF = (3000.0, 3100.0, 3200.0, 3300.0, 3400.0, 3500.0, 3600.0, 3700.0, 3750.0, 3800.0, 3900.0, 4000.0)
GIANT_LOGGS = (0.5, 1.5, 2.5)
GIANT_METALLICITIES = (0.5, 0.0, -0.5, -1.0)
VALIDATION_GIANT_TRACKS = {(1.5, -0.5), (2.5, 0.5)}
SEALED_GIANT_TRACKS = {(0.5, 0.5)}
DWARF_TEFF_BOUNDARY = {0.5: 3600.0, 0.0: 3300.0, -1.0: 3800.0}


def _repo_relative(product_path: str | None) -> str | None:
    if not product_path:
        return None
    marker = product_path.find("/results/")
    if marker >= 0:
        return product_path[marker + 1 :]
    if product_path.startswith("results/"):
        return product_path
    return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _labels_of(record: dict[str, Any]) -> dict[str, float] | None:
    labels = record.get("labels")
    if isinstance(labels, dict) and "effective_temperature" in labels:
        return labels
    track = record.get("track")
    if isinstance(track, dict) and "temperature_K" in record:
        return {
            "effective_temperature": float(record["temperature_K"]),
            "log_surface_gravity": float(track["log_surface_gravity"]),
            "metallicity": float(track["metallicity"]),
            "microturbulence_km_s": float(track["microturbulence_km_s"]),
        }
    return None


def _node_key(labels: dict[str, float]) -> tuple:
    teff = float(labels["effective_temperature"])
    logg = float(labels["log_surface_gravity"])
    metallicity = float(labels["metallicity"])
    vmic = float(labels["microturbulence_km_s"])
    stellar_class = "dwarf" if logg >= 3.5 else "giant"
    return (stellar_class, round(teff, 4), round(logg, 2), round(metallicity, 2), round(vmic, 2))


def _in_boundary(stellar_class: str, teff: float, logg: float, metallicity: float) -> bool:
    if stellar_class == "giant":
        # M giants: the v1r2 corpus contains gate-passing products down to
        # 3000 K; the EOS walls are a dwarf phenomenon.
        return 3000.0 <= teff <= 4000.0 and 0.5 <= logg <= 2.5
    if stellar_class == "dwarf":
        if logg < 4.5 or not 3300.0 <= teff <= 4000.0:
            return False
        floor = DWARF_TEFF_BOUNDARY.get(round(metallicity, 2))
        return floor is not None and teff >= floor
    return False


def _load_frozen_gate(repo: Path) -> dict[str, float]:
    gate_path = repo / "results" / "m_star_iteration_tomography_v1" / "flux_gate.json"
    if gate_path.is_file():
        payload = json.loads(gate_path.read_text())
        for value in payload.values():
            if isinstance(value, dict):
                metrics = value.get("metrics", value)
                thresholds = {}
                for name, metric in metrics.items():
                    if isinstance(metric, dict) and "threshold" in metric:
                        thresholds[name] = float(metric["threshold"])
                if "p95_absolute_flux_error_percent" in thresholds:
                    return thresholds
    return dict(FROZEN_GATE_FALLBACK)


def _flux_metrics(record: dict[str, Any]) -> dict[str, float | None]:
    metrics = (record.get("primary_flux_gate") or {}).get("metrics") or {}
    return {
        name: (metrics.get(name) or {}).get("value")
        for name in FLUX_METRIC_NAMES
    }


def _flat_case_rows(repo: Path, campaign: str, admitted: dict, notes: dict) -> None:
    for case_path in sorted((repo / "results" / campaign).glob(FLAT_CASE_GLOBS[campaign])):
        record = json.loads(case_path.read_text())
        labels = _labels_of(record)
        if labels is None:
            continue
        key = _node_key(labels)
        eligible = record.get("training_eligible") is True
        primary = record.get("primary") or {}
        restart = record.get("restart") or {}
        primary_rel = _repo_relative(primary.get("product_path"))
        restart_rel = _repo_relative(restart.get("product_path"))
        primary_ok = bool(primary_rel) and (repo / primary_rel).is_file()
        restart_ok = bool(restart_rel) and (repo / restart_rel).is_file()
        entry = {
            "campaign": campaign,
            "case": str(case_path.relative_to(repo)),
            "training_eligible": eligible,
            "primary_product": primary_rel if primary_ok else None,
            "restart_product": restart_rel if restart_ok else None,
            "primary_flux_gate_passes": (record.get("primary_flux_gate") or {}).get("passes"),
            "restart_flux_gate_passes": (record.get("restart_flux_gate") or {}).get("passes"),
            "primary_flux_metrics": _flux_metrics(record),
            "path_consistency_passes": (record.get("path_consistency") or {}).get("passes"),
            "phase_guard": record.get("phase_guard"),
            "primary_iterations": primary.get("iterations"),
            "restart_iterations": restart.get("iterations"),
            "status": record.get("status"),
        }
        if eligible and primary_ok:
            entry["admitted"] = True
            admitted.setdefault(key, []).append(entry)
        else:
            entry["admitted"] = False
            if eligible and not primary_ok:
                notes.setdefault("eligible_but_product_missing", []).append(entry)
            elif primary.get("converged") or restart.get("converged"):
                notes.setdefault("converged_not_eligible", []).append(entry)


def _attempt_rows(repo: Path, campaign: str, admitted: dict, notes: dict) -> None:
    suffix = "_complete.json" if campaign == "m_star_pipeline_complete_v1" else "_pipeline.json"
    for record_path in sorted((repo / "results" / campaign / "cases").glob(f"*{suffix}")):
        record = json.loads(record_path.read_text())
        for cap, attempt in sorted((record.get("attempts") or {}).items(), key=lambda kv: int(kv[0])):
            labels = _labels_of(attempt)
            if labels is None:
                continue
            key = _node_key(labels)
            primary = attempt.get("primary") or {}
            restart = attempt.get("restart") or {}
            phase_guard = attempt.get("phase_guard") or {}
            primary_rel = _repo_relative(primary.get("product_path"))
            restart_rel = _repo_relative(restart.get("product_path"))
            primary_ok = bool(primary_rel) and (repo / primary_rel).is_file()
            restart_ok = bool(restart_rel) and (repo / restart_rel).is_file()
            gates_pass = bool((attempt.get("primary_flux_gate") or {}).get("passes")) and bool(
                (attempt.get("restart_flux_gate") or {}).get("passes")
            )
            path_pass = bool((attempt.get("path_consistency") or {}).get("passes"))
            guard_pass = bool(phase_guard.get("primary")) and bool(phase_guard.get("restart"))
            eligible = attempt.get("training_eligible")
            if eligible is None:
                eligible = gates_pass and path_pass and guard_pass and restart_ok
            entry = {
                "campaign": campaign,
                "case": str(record_path.relative_to(repo)),
                "cap": int(cap),
                "training_eligible": bool(eligible),
                "admitted": bool(eligible and primary_ok and restart_ok),
                "primary_product": primary_rel if primary_ok else None,
                "restart_product": restart_rel if restart_ok else None,
                "primary_flux_gate_passes": (attempt.get("primary_flux_gate") or {}).get("passes"),
                "restart_flux_gate_passes": (attempt.get("restart_flux_gate") or {}).get("passes"),
                "primary_flux_metrics": _flux_metrics(attempt),
                "path_consistency_passes": path_pass,
                "phase_guard": phase_guard,
                "primary_iterations": primary.get("iterations"),
                "restart_iterations": restart.get("iterations"),
                "status": attempt.get("status"),
            }
            if entry["admitted"]:
                admitted.setdefault(key, []).append(entry)
            elif gates_pass and not restart_ok:
                notes.setdefault("gate_pass_but_restart_product_missing", []).append(entry)


def _walk_step_rows(repo: Path, campaign: str, waypoints: list) -> None:
    root = repo / "results" / campaign / "cases"
    if not root.is_dir():
        return
    for case_path in sorted(root.glob("*/case.json")):
        record = json.loads(case_path.read_text())
        steps = record.get("steps") or (record.get("continuation") or {}).get("steps") or []
        for step in steps:
            if not isinstance(step, dict):
                continue
            teff = step.get("target_temperature_K")
            if teff is None:
                labels = step.get("labels") or {}
                teff = labels.get("effective_temperature")
            if teff is None:
                continue
            track = record.get("track") or {}
            waypoints.append(
                {
                    "campaign": campaign,
                    "case": str(case_path.relative_to(repo)),
                    "teff_K": float(teff),
                    "logg": float(track.get("log_surface_gravity", 0.0)),
                    "metallicity": float(track.get("metallicity", 0.0)),
                    "status": step.get("status"),
                    "p95_flux_percent": step.get("p95_absolute_flux_error_percent"),
                    "donor": step.get("donor"),
                }
            )


def _canonical(rows: list[dict]) -> dict:
    def rank(entry: dict) -> tuple:
        try:
            priority = CAMPAIGN_PRIORITY.index(entry["campaign"])
        except ValueError:
            priority = len(CAMPAIGN_PRIORITY)
        return (priority, entry.get("primary_iterations") or 10**6)

    return sorted(rows, key=rank)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    repo = args.repo_root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    admitted: dict[tuple, list[dict]] = {}
    notes: dict[str, list] = {}
    for campaign in FLAT_CASE_GLOBS:
        if (repo / "results" / campaign).is_dir():
            _flat_case_rows(repo, campaign, admitted, notes)
    for campaign in ("m_star_pipeline_scaleout_v1", "m_star_pipeline_complete_v1"):
        if (repo / "results" / campaign).is_dir():
            _attempt_rows(repo, campaign, admitted, notes)

    waypoints: list[dict] = []
    for campaign in WALK_CAMPAIGNS + ("m_star_downwalk_v1",):
        _walk_step_rows(repo, campaign, waypoints)
    frozen_gate = _load_frozen_gate(repo)
    for waypoint in waypoints:
        p95 = waypoint.get("p95_flux_percent")
        waypoint["gate_pass_estimate"] = (
            p95 is not None and float(p95) <= frozen_gate["p95_absolute_flux_error_percent"]
        )

    rows = []
    for key in sorted(admitted):
        entries = admitted[key]
        canonical = _canonical(entries)
        stellar_class, teff, logg, metallicity, vmic = key
        rows.append(
            {
                "stellar_class": stellar_class,
                "teff_K": teff,
                "logg": logg,
                "metallicity": metallicity,
                "vmic_km_s": vmic,
                "in_boundary": _in_boundary(stellar_class, teff, logg, metallicity),
                "canonical_campaign": canonical["campaign"],
                "canonical_product": canonical["primary_product"],
                "canonical_restart_product": canonical["restart_product"],
                "primary_flux_metrics": canonical["primary_flux_metrics"],
                "primary_iterations": canonical["primary_iterations"],
                "sources": sorted({entry["campaign"] for entry in entries}),
            }
        )
    in_boundary = [row for row in rows if row["in_boundary"]]
    out_boundary = [row for row in rows if not row["in_boundary"]]

    def by_class_metallicity(selected: list[dict]) -> dict:
        table: dict[str, dict[str, int]] = {}
        for row in selected:
            cls = table.setdefault(row["stellar_class"], {})
            name = f"{row['metallicity']:+.1f}"
            cls[name] = cls.get(name, 0) + 1
        return table

    giant_gap = []
    admitted_giant_keys = {
        (row["logg"], row["metallicity"], row["teff_K"]) for row in in_boundary if row["stellar_class"] == "giant"
    }
    for logg in GIANT_LOGGS:
        for metallicity in GIANT_METALLICITIES:
            if (logg, metallicity) in VALIDATION_GIANT_TRACKS:
                role = "validation"
            elif (logg, metallicity) in SEALED_GIANT_TRACKS:
                role = "sealed"
            else:
                role = "train"
            if role != "train":
                continue
            for teff in GIANT_TEFF:
                if (logg, metallicity, teff) in admitted_giant_keys:
                    continue
                giant_gap.append(
                    {"logg": logg, "metallicity": metallicity, "teff_K": teff, "role": role}
                )

    summary = {
        "repo_root": str(repo),
        "frozen_flux_gate": frozen_gate,
        "admitted_total": len(rows),
        "admitted_in_boundary": len(in_boundary),
        "admitted_in_boundary_by_class_metallicity": by_class_metallicity(in_boundary),
        "admitted_out_of_boundary": len(out_boundary),
        "admitted_out_of_boundary_by_class_metallicity": by_class_metallicity(out_boundary),
        "converged_waypoint_steps": len(waypoints),
        "converged_waypoint_gate_pass_estimate": sum(1 for w in waypoints if w["gate_pass_estimate"]),
        "giant_gap_open_tracks": len(giant_gap),
        "notes_counts": {name: len(entries) for name, entries in notes.items()},
    }

    args.out.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.out / "inventory.json",
        {
            "summary": summary,
            "admitted_rows": rows,
            "in_boundary_rows": in_boundary,
            "out_of_boundary_rows": out_boundary,
            "giant_gap_nodes_to_fill": giant_gap,
            "converged_waypoints": waypoints,
            "notes": notes,
        },
    )
    fieldnames = [
        "stellar_class",
        "teff_K",
        "logg",
        "metallicity",
        "vmic_km_s",
        "in_boundary",
        "canonical_campaign",
        "canonical_product",
        "canonical_restart_product",
        "primary_iterations",
        "sources",
    ]
    with (args.out / "inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{key: row[key] for key in fieldnames} for row in rows]
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nwrote {args.out / 'inventory.json'}")
    print(f"wrote {args.out / 'inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
