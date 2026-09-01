"""Bounded native-MARCS M-star science case for Payne-Zero.

This is deliberately a separate campaign from ``cool_star_step_test``.  It
uses eight native MARCS nodes (four M dwarfs and four M giants), passes only
the converted ``(m, T)`` pair to Payne-Zero, and records both successful and
failed solver attempts.  The primary continuation arm rematerializes the
dependent fields at every temperature; full-state carry is retained only as a
diagnostic control.

The campaign is not a production-emulator training or routing change.  Its
default result namespace is ``results/m_star_science_case_v1``.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from typing import Any

import numpy as np

from bench.labels import StellarLabels
from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere

from .cool_star_step_test import (
    ANCHOR_TEMPERATURE,
    ITERATION_CAP,
    PRIMARY_ITERATION_CAP,
    TrackSpec,
    _atmosphere_quality,
    _clone_atmosphere,
    _failed_record,
    _reconstruct_from_mt,
    _retarget_full_state,
    _set_single_thread_environment,
    _solve_attempt,
)
from .marcs_h5 import (
    EXPECTED_MARCS_SHA256,
    MarcsGridSchema,
    MarcsH5Error,
    inspect_marcs_grid,
    load_marcs_node,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "m_star_science_case_v1"
DEFAULT_KORG_ROOT = Path("/Users/jdli/Project/jorg/Korg.jl-1.0.1")
DEFAULT_JULIA = Path("/opt/homebrew/bin/julia")
TEMPERATURES = (3000.0, 3300.0, 3500.0, 3750.0)
CONTINUATION_LADDER = (4000.0, 3750.0, 3500.0, 3300.0, 3000.0)
KORG_REFERENCE_WAVELENGTH_ANGSTROM = 5000.0
KORG_SMOKE_WINDOW_ANGSTROM = (5000.0, 5020.0)
KORG_FULL_WINDOW_NM = (400.0, 900.0)
KORG_RESOLUTION = 20_000.0
KORG_SYNTHESIS_STEP_ANGSTROM = 0.05


def build_mstar_tracks() -> list[TrackSpec]:
    """Return the two fixed non-temperature tracks in the protocol."""

    return [
        TrackSpec(
            log_surface_gravity=5.0,
            metallicity=0.0,
            alpha_enhancement=0.0,
            carbon_enhancement=0.0,
            microturbulence_km_s=1.0,
        ),
        TrackSpec(
            log_surface_gravity=1.5,
            metallicity=0.0,
            alpha_enhancement=0.0,
            carbon_enhancement=0.0,
            microturbulence_km_s=2.0,
        ),
    ]


def star_class(track: TrackSpec) -> str:
    return "M-dwarf" if track.log_surface_gravity >= 3.5 else "M-giant"


def case_id(track: TrackSpec, temperature: float) -> str:
    return f"{star_class(track).lower().replace('-', '_')}_t{int(temperature):04d}"


def protocol_manifest(
    *,
    marcs_grid: Path,
    marcs_sha256: str,
    result_root: Path,
    iteration_cap: int,
) -> dict[str, Any]:
    return {
        "campaign": "m_star_science_case_v1",
        "status": "development_bounded_science_case",
        "scope": {
            "claim": "validation on eight representative native MARCS M-star nodes",
            "not_claimed": [
                "broad support for all Teff < 4000 K M dwarfs or M giants",
                "production emulator retraining or routing support",
                "sealed holdout validation",
            ],
        },
        "native_marcs": {
            "path": str(marcs_grid),
            "sha256": marcs_sha256,
            "expected_sha256": EXPECTED_MARCS_SHA256,
            "nodes": "native label lookup only; 56-to-80 interpolation is depth-only",
            "depth_coordinate": "log_mass",
        },
        "cases": {
            "effective_temperature_K": list(TEMPERATURES),
            "dwarf": {
                "logg": 5.0,
                "microturbulence_km_s": 1.0,
            },
            "giant": {
                "logg": 1.5,
                "microturbulence_km_s": 2.0,
            },
            "metallicity": 0.0,
            "alpha_enhancement": 0.0,
            "carbon_enhancement": 0.0,
        },
        "solver": {
            "anchor_temperature_K": ANCHOR_TEMPERATURE,
            "continuation_ladder_K": list(CONTINUATION_LADDER),
            "direct_start": "native MARCS (m,T), then Payne-Zero rematerialization",
            "primary_continuation": "MARCS-anchored reduced/rematerialized (m,T)",
            "diagnostic_continuation": "MARCS-anchored full-state carry",
            "iteration_cap": int(iteration_cap),
            "within_15_iterations_recorded": True,
            "stopping_rule": "unchanged production solver stopping rule",
        },
        "korg": {
            "path": str(DEFAULT_KORG_ROOT),
            "version": "1.0.1",
            "geometry": "planar for both same-atmosphere and independent-MARCS comparisons",
            "smoke_window_A": list(KORG_SMOKE_WINDOW_ANGSTROM),
            "full_window_nm": list(KORG_FULL_WINDOW_NM),
            "resolution": KORG_RESOLUTION,
            "linelist": "Korg.get_VALD_solar_linelist(), with atomic-only filtered by !Korg.ismolecule",
            "molecular_coverage": "VALD solar molecular subset is present; no GES artifact dependency",
        },
        "paths": {
            "result_root": str(result_root),
            "records": str(result_root / "records.jsonl"),
            "cases": str(result_root / "cases.json"),
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _ensure_flux_imbalance_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Keep the solver's all-layer flux maximum when no layer vector was saved."""

    flux = record.get("flux_imbalance")
    if isinstance(flux, dict) and flux.get("available"):
        return record
    final = record.get("solver_diagnostics", {}).get("final_diagnostics", {})
    maximum = final.get("maximum_absolute_flux_error_percent")
    if maximum is not None:
        record["flux_imbalance"] = {
            "available": True,
            "vector_available": False,
            "max_percent": float(maximum),
            "source": "unchanged solver final diagnostic; maximum over all atmosphere layers",
        }
    return record


def _refresh_loaded_case_diagnostics(cases: list[dict[str, Any]]) -> None:
    """Normalize diagnostics when reusing a completed solver campaign."""

    for case in cases:
        for key in ("direct", "continuation_primary", "continuation_full_carry_diagnostic"):
            record = case.get(key)
            if isinstance(record, dict):
                _ensure_flux_imbalance_summary(record)


def _augment_record(
    record: dict[str, Any],
    *,
    case: str,
    role: str,
    node: Any | None,
) -> dict[str, Any]:
    """Attach full-layer flux and structural diagnostics to a solver record."""

    record = dict(record)
    record["case_id"] = case
    record["case_role"] = role
    iterations = record.get("iterations")
    record["within_15_iterations"] = bool(
        record.get("survives_solver")
        and iterations is not None
        and int(iterations) <= PRIMARY_ITERATION_CAP
    )
    record["failure_reason"] = None if record.get("survives_solver") else record.get("error")

    product = record.get("product_path")
    flux = None
    structure = None
    if product and Path(product).is_file():
        try:
            with np.load(product, allow_pickle=False) as data:
                if "flux_error_percent" in data:
                    flux = np.asarray(data["flux_error_percent"], dtype=np.float64)
                if node is not None and "temperature" in data and "column_mass" in data:
                    final_temperature = np.asarray(data["temperature"], dtype=np.float64)
                    final_mass = np.asarray(data["column_mass"], dtype=np.float64)
                    target_temperature = np.asarray(node.reduced_temperature, dtype=np.float64)
                    target_mass = np.asarray(node.reduced_column_mass, dtype=np.float64)
                    if final_temperature.shape == target_temperature.shape:
                        structure = {
                            "temperature_relative": {
                                "median": float(np.median(np.abs(final_temperature - target_temperature) / target_temperature)),
                                "p95": float(np.percentile(np.abs(final_temperature - target_temperature) / target_temperature, 95.0)),
                                "max": float(np.max(np.abs(final_temperature - target_temperature) / target_temperature)),
                            },
                            "column_mass_dex": {
                                "median": float(np.median(np.abs(np.log10(final_mass) - np.log10(target_mass)))),
                                "p95": float(np.percentile(np.abs(np.log10(final_mass) - np.log10(target_mass)), 95.0)),
                                "max": float(np.max(np.abs(np.log10(final_mass) - np.log10(target_mass)))),
                            },
                            "reference": "converted native MARCS (m,T); diagnostic only",
                        }
        except Exception as exc:  # keep the solver outcome, expose the diagnostic failure
            record["diagnostic_error"] = f"{type(exc).__name__}: {exc}"

    if flux is not None and flux.ndim == 1 and flux.size:
        finite = bool(np.all(np.isfinite(flux)))
        record["flux_imbalance"] = {
            "available": True,
            "finite": finite,
            "all_layers_percent": flux.tolist(),
            "median_percent": float(np.median(np.abs(flux))) if finite else None,
            "p95_percent": float(np.percentile(np.abs(flux), 95.0)) if finite else None,
            "max_percent": float(np.max(np.abs(flux))) if finite else None,
        }
    else:
        record["flux_imbalance"] = {"available": False}
    record["structure_difference_to_native_marcs"] = structure
    return _jsonable(_ensure_flux_imbalance_summary(record))


def _load_node_and_seed(
    schema: MarcsGridSchema,
    track: TrackSpec,
    temperature: float,
) -> tuple[Any, ModelAtmosphere]:
    print(
        f"[{track.track_id}] rematerializing native MARCS {temperature:g} K",
        flush=True,
    )
    labels = track.labels(temperature)
    node = load_marcs_node(
        schema.path,
        labels,
        carbon_enhancement=track.carbon_enhancement,
        verify_sha256=False,
        expected_sha256=None,
        schema=schema,
        depth_coordinate="log_mass",
    )
    seed = _reconstruct_from_mt(labels, node.reduced_column_mass, node.reduced_temperature)
    return node, seed


def _track_campaign(
    *,
    track: TrackSpec,
    schema: MarcsGridSchema,
    result_root: Path,
    iteration_cap: int,
    manifest_hash: str,
) -> dict[str, Any]:
    """Run both solver paths for one dwarf/giant track."""

    _set_single_thread_environment()
    print(f"[{track.track_id}] starting {star_class(track)} track", flush=True)
    track_root = result_root / "tracks" / track.track_id
    records: list[dict[str, Any]] = []
    nodes: dict[float, Any] = {}
    seeds: dict[float, ModelAtmosphere] = {}
    errors: list[str] = []

    # The 4000 K MARCS anchor is solved once.  It is the source of both
    # continuation paths and is not counted as one of the eight target cases.
    anchor_node = None
    anchor_state = None
    try:
        anchor_node, anchor_seed = _load_node_and_seed(schema, track, ANCHOR_TEMPERATURE)
        anchor_record, anchor_state = _solve_attempt(
            track=track,
            method="marcs_anchor_reduced",
            schedule="anchor",
            source_temperature=None,
            target_labels=track.labels(ANCHOR_TEMPERATURE),
            initial_atmosphere=anchor_seed,
            product_dir=track_root / "products" / "marcs_anchor_reduced",
            iteration_cap=iteration_cap,
        )
        print(
            f"[{track.track_id}] anchor 4000 K: {anchor_record.get('status')} "
            f"({anchor_record.get('iterations')} iterations)",
            flush=True,
        )
        records.append(_augment_record(anchor_record, case=f"{star_class(track).lower()}_t4000", role="anchor", node=anchor_node))
    except Exception as exc:  # preserve anchor failure and let all continuation cases be blocked
        errors.append(f"anchor: {type(exc).__name__}: {exc}")
        anchor_record = _failed_record(
            track=track,
            method="marcs_anchor_reduced",
            labels=track.labels(ANCHOR_TEMPERATURE),
            source_temperature=None,
            target_temperature=ANCHOR_TEMPERATURE,
            schedule="anchor",
            error=exc,
            status="anchor_load_or_reconstruction_failed",
        )
        records.append(_augment_record(anchor_record, case=f"{star_class(track).lower()}_t4000", role="anchor", node=None))

    # Decode all eight native target nodes before solving so the direct and
    # full-carry diagnostic paths share exactly the same frozen MARCS inputs.
    for temperature in TEMPERATURES:
        try:
            nodes[temperature], seeds[temperature] = _load_node_and_seed(schema, track, temperature)
        except Exception as exc:
            errors.append(f"{temperature:g} K node: {type(exc).__name__}: {exc}")

    direct_by_temperature: dict[float, dict[str, Any]] = {}
    continuation_by_temperature: dict[float, dict[str, Any]] = {}
    full_by_temperature: dict[float, dict[str, Any]] = {}

    # Direct MARCS two-field starts.
    for temperature in TEMPERATURES:
        labels = track.labels(temperature)
        case = case_id(track, temperature)
        node = nodes.get(temperature)
        try:
            print(f"[{track.track_id}] direct {temperature:g} K", flush=True)
            direct_record, _direct_state = _solve_attempt(
                track=track,
                method="marcs_direct_reduced",
                schedule="direct",
                source_temperature=None,
                target_labels=labels,
                initial_atmosphere=seeds.get(temperature),
                product_dir=track_root / "products" / "marcs_direct_reduced",
                iteration_cap=iteration_cap,
            )
            direct_record["marcs_input_diagnostics"] = _jsonable({
                "native_layers": int(node.native_column_mass.size) if node is not None else None,
                "native_indices": list(node.indices) if node is not None else None,
                "native_tau5000_range": [float(np.min(node.native_tau_5000)), float(np.max(node.native_tau_5000))] if node is not None else None,
                "payne_receives": ["column_mass", "temperature"],
            })
        except Exception as exc:
            direct_record = _failed_record(
                track=track,
                method="marcs_direct_reduced",
                labels=labels,
                source_temperature=None,
                target_temperature=temperature,
                schedule="direct",
                error=exc,
                status="marcs_load_or_reconstruction_failed",
            )
        direct_by_temperature[temperature] = _augment_record(direct_record, case=case, role="direct", node=node)
        records.append(direct_by_temperature[temperature])

    target_templates = {temperature: seed for temperature, seed in seeds.items()}

    # Primary reduced/rematerialized continuation.
    current = _clone_atmosphere(anchor_state) if anchor_state is not None else None
    current_temperature = ANCHOR_TEMPERATURE
    for temperature in CONTINUATION_LADDER[1:]:
        labels = track.labels(temperature)
        case = case_id(track, temperature)
        node = nodes.get(temperature)
        if current is None:
            continuation_record = _failed_record(
                track=track,
                method="marcs_continuation_reduced",
                labels=labels,
                source_temperature=current_temperature,
                target_temperature=temperature,
                schedule="4000-3750-3500-3300-3000K",
                error="continuation stopped because the MARCS 4000 K anchor or a previous step failed",
                status="blocked_by_previous_step",
            )
        else:
            try:
                print(
                    f"[{track.track_id}] continuation {current_temperature:g} -> {temperature:g} K",
                    flush=True,
                )
                initial = _reconstruct_from_mt(labels, current.column_mass, current.temperature)
                continuation_record, next_state = _solve_attempt(
                    track=track,
                    method="marcs_continuation_reduced",
                    schedule="4000-3750-3500-3300-3000K",
                    source_temperature=current_temperature,
                    target_labels=labels,
                    initial_atmosphere=initial,
                    product_dir=track_root / "products" / "marcs_continuation_reduced",
                    iteration_cap=iteration_cap,
                )
                if continuation_record.get("survives_solver") and next_state is not None:
                    current = next_state
                    current_temperature = temperature
                else:
                    current = None
            except Exception as exc:
                continuation_record = _failed_record(
                    track=track,
                    method="marcs_continuation_reduced",
                    labels=labels,
                    source_temperature=current_temperature,
                    target_temperature=temperature,
                    schedule="4000-3750-3500-3300-3000K",
                    error=exc,
                    status="rematerialization_failed",
                )
                current = None
        continuation_by_temperature[temperature] = _augment_record(continuation_record, case=case, role="continuation_primary", node=node)
        records.append(continuation_by_temperature[temperature])

    # Full-state carry control.  It uses only the target seed's metadata and
    # abundance bookkeeping; the resulting rows never choose the paper route.
    current = _clone_atmosphere(anchor_state) if anchor_state is not None else None
    current_temperature = ANCHOR_TEMPERATURE
    for temperature in CONTINUATION_LADDER[1:]:
        labels = track.labels(temperature)
        case = case_id(track, temperature)
        node = nodes.get(temperature)
        if current is None or temperature not in target_templates:
            full_record = _failed_record(
                track=track,
                method="marcs_continuation_full_carry",
                labels=labels,
                source_temperature=current_temperature,
                target_temperature=temperature,
                schedule="4000-3750-3500-3300-3000K",
                error="full-carry continuation stopped because a previous step failed",
                status="blocked_by_previous_step",
            )
        else:
            try:
                print(
                    f"[{track.track_id}] full-carry {current_temperature:g} -> {temperature:g} K",
                    flush=True,
                )
                initial = _retarget_full_state(current, target_templates[temperature])
                full_record, next_state = _solve_attempt(
                    track=track,
                    method="marcs_continuation_full_carry",
                    schedule="4000-3750-3500-3300-3000K",
                    source_temperature=current_temperature,
                    target_labels=labels,
                    initial_atmosphere=initial,
                    product_dir=track_root / "products" / "marcs_continuation_full_carry",
                    iteration_cap=iteration_cap,
                )
                if full_record.get("survives_solver") and next_state is not None:
                    current = next_state
                    current_temperature = temperature
                else:
                    current = None
            except Exception as exc:
                full_record = _failed_record(
                    track=track,
                    method="marcs_continuation_full_carry",
                    labels=labels,
                    source_temperature=current_temperature,
                    target_temperature=temperature,
                    schedule="4000-3750-3500-3300-3000K",
                    error=exc,
                    status="full_carry_initialization_failed",
                )
                current = None
        full_by_temperature[temperature] = _augment_record(full_record, case=case, role="continuation_full_carry_diagnostic", node=node)
        records.append(full_by_temperature[temperature])

    cases: list[dict[str, Any]] = []
    for temperature in TEMPERATURES:
        direct = direct_by_temperature[temperature]
        continuation = continuation_by_temperature[temperature]
        full = full_by_temperature[temperature]
        if continuation.get("survives_solver"):
            selected_route = "continuation_primary"
            selected = continuation
        elif direct.get("survives_solver"):
            selected_route = "direct"
            selected = direct
        else:
            selected_route = None
            selected = None
        cases.append({
            "case_id": case_id(track, temperature),
            "track_id": track.track_id,
            "class": star_class(track),
            "labels": track.labels(temperature).as_kwargs(),
            "native_marcs_indices": list(nodes[temperature].indices) if temperature in nodes else None,
            "direct": direct,
            "continuation_primary": continuation,
            "continuation_full_carry_diagnostic": full,
            "selected_route": selected_route,
            "selected_product_path": None if selected is None else selected.get("product_path"),
        })

    payload = {
        "track_id": track.track_id,
        "track": track.as_json(),
        "class": star_class(track),
        "manifest_hash": manifest_hash,
        "errors": errors,
        "records": records,
        "cases": cases,
    }
    _write_json(track_root / "track.json", payload)
    return payload


def _selected_cases(track_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for track_payload in track_payloads:
        rows.extend(track_payload["cases"])
    return sorted(rows, key=lambda row: (row["class"], float(row["labels"]["effective_temperature"])))


def _write_korg_inputs(result_root: Path, cases: list[dict[str, Any]]) -> Path:
    input_root = result_root / "korg_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    manifest_path = input_root / "manifest.tsv"
    fields = [
        "case_id", "class", "effective_temperature", "logg", "metallicity",
        "alpha_enhancement", "carbon_enhancement", "microturbulence_km_s",
        "product_path", "atmosphere_path",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for case in cases:
            product = case.get("selected_product_path")
            if not product or not Path(product).is_file():
                continue
            labels = case["labels"]
            atmosphere_path = input_root / f"{case['case_id']}.tsv"
            with np.load(product, allow_pickle=False) as data:
                required = ("temperature", "electron_density", "gas_pressure", "mass_density", "column_mass")
                if any(key not in data for key in required):
                    continue
                matrix = np.column_stack([np.asarray(data[key], dtype=np.float64) for key in required])
            np.savetxt(
                atmosphere_path,
                matrix,
                delimiter="\t",
                header="temperature_K\telectron_density_cm-3\tgas_pressure_dyn_cm-2\tmass_density_g_cm-3\tcolumn_mass_g_cm-2",
                comments="",
            )
            writer.writerow({
                "case_id": case["case_id"],
                "class": case["class"],
                "effective_temperature": labels["effective_temperature"],
                "logg": labels["log_surface_gravity"],
                "metallicity": labels["metallicity"],
                "alpha_enhancement": labels["alpha_enhancement"],
                "carbon_enhancement": 0.0,
                "microturbulence_km_s": case["track_id"].split("_x")[-1],
                "product_path": str(Path(product).resolve()),
                "atmosphere_path": str(atmosphere_path.resolve()),
            })
    return manifest_path


def _run_payne_zero_spectra(result_root: Path, cases: list[dict[str, Any]], *, smoke: bool = False) -> None:
    """Synthesize selected converged products for the Korg comparison."""

    from payne_zero_synthesis import synthesize

    pz_root = result_root / "spectra" / ("smoke" if smoke else "full")
    for case in cases:
        product = case.get("selected_product_path")
        if not product or not Path(product).is_file():
            case["payne_zero_spectra"] = {"status": "not_available"}
            continue
        output = {}
        for molecular in (False, True):
            arm = "molecular" if molecular else "atomic_only"
            path = pz_root / arm / f"{case['case_id']}.npz"
            if not path.is_file():
                try:
                    spectrum = synthesize(
                        product,
                        wavelength_start_nm=400.0 if not smoke else 500.0,
                        wavelength_end_nm=900.0 if not smoke else 502.0,
                        resolution=KORG_RESOLUTION,
                        molecular_lines=molecular,
                        device="cpu",
                        dtype="float64",
                    )
                    spectrum.save_npz(path)
                except Exception as exc:
                    output[arm] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    continue
            output[arm] = {"status": "ok", "path": str(path.resolve())}
        case["payne_zero_spectra"] = output


def _korg_stats(candidate: np.ndarray, reference: np.ndarray, *, absolute: bool = False) -> dict[str, float]:
    delta = np.abs(np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    if absolute:
        scaled = delta
    else:
        scaled = delta / np.maximum(np.abs(np.asarray(reference, dtype=np.float64)), 1.0e-300)
    return {
        "median": float(np.median(scaled)),
        "p95": float(np.percentile(scaled, 95.0)),
        "max": float(np.max(scaled)),
    }


def _load_h5_spectrum(path: str | Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        return {key: np.asarray(handle[key], dtype=np.float64) for key in handle.keys()}


def _compare_spectra(pz_path: str | Path, korg_path: str | Path) -> dict[str, Any]:
    pz = np.load(pz_path, allow_pickle=False)
    pz_wavelength = np.asarray(pz["wavelength_nm"], dtype=np.float64)
    pz_total = np.asarray(pz["flux_total"], dtype=np.float64)
    pz_continuum = np.asarray(pz["flux_continuum"], dtype=np.float64)
    pz_normalized = np.asarray(pz["normalized_flux"], dtype=np.float64)
    korg = _load_h5_spectrum(korg_path)
    wavelength_nm = np.asarray(korg["wavelength_A"], dtype=np.float64) / 10.0
    overlap = (pz_wavelength >= wavelength_nm[0]) & (pz_wavelength <= wavelength_nm[-1])
    if not np.any(overlap):
        raise ValueError(f"no wavelength overlap between {pz_path} and {korg_path}")
    pz_wavelength = pz_wavelength[overlap]
    interp = lambda values: np.interp(pz_wavelength, wavelength_nm, np.asarray(values, dtype=np.float64))
    k_total = interp(korg["flux_total"])
    k_continuum = interp(korg["flux_continuum"])
    k_normalized = interp(korg["normalized_flux"])
    continuum_mask = np.minimum(pz_normalized, k_normalized) >= 0.995
    line_mask = np.minimum(pz_normalized, k_normalized) <= 0.98
    if not np.any(continuum_mask):
        continuum_mask = np.ones_like(pz_normalized, dtype=bool)
    if not np.any(line_mask):
        line_mask = np.ones_like(pz_normalized, dtype=bool)
    total_scale = np.maximum(np.abs(k_continuum), 1.0e-300)
    total_delta = np.abs(pz_total[overlap] - k_total) / total_scale
    continuum_delta = np.abs(pz_continuum[overlap] - k_continuum) / np.maximum(np.abs(k_continuum), 1.0e-300)
    normalized_delta = np.abs(pz_normalized[overlap] - k_normalized)
    return {
        "wavelength_nm": [float(pz_wavelength[0]), float(pz_wavelength[-1])],
        "samples": int(pz_wavelength.size),
        "normalized_flux": _korg_stats(pz_normalized, k_normalized, absolute=True),
        "total_flux": {
            "median": float(np.median(total_delta)),
            "p95": float(np.percentile(total_delta, 95.0)),
            "max": float(np.max(total_delta)),
        },
        "continuum": {
            "median": float(np.median(continuum_delta)),
            "p95": float(np.percentile(continuum_delta, 95.0)),
            "max": float(np.max(continuum_delta)),
        },
        "continuum_region": {
            "fraction": float(np.mean(continuum_mask)),
            "normalized_flux": _korg_stats(pz_normalized[continuum_mask], k_normalized[continuum_mask], absolute=True),
        },
        "line_rich_region": {
            "fraction": float(np.mean(line_mask)),
            "normalized_flux": _korg_stats(pz_normalized[line_mask], k_normalized[line_mask], absolute=True),
        },
    }


def _attach_korg_metrics(result_root: Path, cases: list[dict[str, Any]], *, smoke: bool = False) -> None:
    korg_root = result_root / "korg" / ("smoke" if smoke else "full")
    for case in cases:
        case_id_text = case["case_id"]
        pz = case.get("payne_zero_spectra", {})
        if not pz:
            case["korg_comparison"] = {"status": "not_available"}
            continue
        comparison: dict[str, Any] = {"status": "ok", "arms": {}}
        for molecular in (False, True):
            korg_kind = "molecular" if molecular else "atomic_only"
            pz_arm = pz.get(korg_kind, {})
            if pz_arm.get("status") != "ok":
                comparison["arms"][korg_kind] = {"status": "not_available"}
                continue
            same_path = korg_root / "same_atmosphere" / korg_kind / f"{case_id_text}.h5"
            independent_path = korg_root / "independent_marcs" / korg_kind / f"{case_id_text}.h5"
            if not same_path.is_file() or not independent_path.is_file():
                comparison["arms"][korg_kind] = {"status": "korg_output_missing"}
                continue
            same_vs_pz = _compare_spectra(pz_arm["path"], same_path)
            independent_vs_pz = _compare_spectra(pz_arm["path"], independent_path)
            same_data = _load_h5_spectrum(same_path)
            indep_data = _load_h5_spectrum(independent_path)
            # Independent-MARCS geometry/structure is the second layer of
            # the comparison; this metric is Korg-vs-Korg, not PZ-vs-Korg.
            same_vs_independent = {
                "normalized_flux": _korg_stats(
                    np.interp(np.asarray(same_data["wavelength_A"]), indep_data["wavelength_A"], indep_data["normalized_flux"]),
                    same_data["normalized_flux"],
                    absolute=True,
                )
            }
            comparison["arms"][korg_kind] = {
                "status": "ok",
                "payne_zero_vs_same_atmosphere_korg": same_vs_pz,
                "payne_zero_vs_independent_marcs_korg": independent_vs_pz,
                "same_atmosphere_vs_independent_marcs_korg": same_vs_independent,
                "same_atmosphere_path": str(same_path.resolve()),
                "independent_marcs_path": str(independent_path.resolve()),
            }
        case["korg_comparison"] = comparison


def run_korg(
    *,
    result_root: Path,
    manifest_path: Path,
    julia: Path,
    korg_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    script = REPO_ROOT / "experiments" / "korg_mstar_compare.jl"
    output_root = result_root / "korg" / ("smoke" if smoke else "full")
    output_root.mkdir(parents=True, exist_ok=True)
    command = [
        str(julia), f"--project={korg_root}", str(script),
        "--manifest", str(manifest_path), "--out-dir", str(output_root),
        "--mode", "smoke" if smoke else "full",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    log = {
        "command": command,
        "returncode": int(completed.returncode),
        "seconds": float(time.perf_counter() - started),
        "stdout": completed.stdout[-20_000:],
        "stderr": completed.stderr[-20_000:],
    }
    _write_json(output_root / "run_log.json", log)
    if completed.returncode != 0:
        raise RuntimeError(f"Korg comparison failed; see {output_root / 'run_log.json'}")
    return log


def _refresh_case_json(cases_path: Path, cases: list[dict[str, Any]], manifest_hash: str) -> None:
    _write_json(cases_path, {
        "campaign": "m_star_science_case_v1",
        "manifest_hash": manifest_hash,
        "case_count": len(cases),
        "cases": _jsonable(cases),
    })


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    _set_single_thread_environment()
    marcs_grid = Path(args.marcs_grid).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    if not marcs_grid.is_file():
        raise FileNotFoundError(marcs_grid)
    marcs_sha256 = sha256_file(marcs_grid)
    if marcs_sha256 != EXPECTED_MARCS_SHA256:
        raise MarcsH5Error(f"MARCS SHA-256 mismatch: got {marcs_sha256}, expected {EXPECTED_MARCS_SHA256}")
    result_root.mkdir(parents=True, exist_ok=True)
    cases_path = result_root / "cases.json"
    if args.skip_solver:
        manifest_path = result_root / "manifest.json"
        if not manifest_path.is_file() or not cases_path.is_file():
            raise FileNotFoundError("--skip-solver requires existing manifest.json and cases.json")
        manifest = json.loads(manifest_path.read_text())
        cases_document = json.loads(cases_path.read_text())
        if manifest.get("campaign") != "m_star_science_case_v1" or cases_document.get("case_count") != 8:
            raise RuntimeError("existing result namespace is not a complete m_star_science_case_v1 campaign")
        manifest_hash = str(manifest.get("manifest_hash", cases_document.get("manifest_hash", "")))
        cases = list(cases_document["cases"])
        _refresh_loaded_case_diagnostics(cases)
        _refresh_case_json(cases_path, cases, manifest_hash)
    else:
        schema = inspect_marcs_grid(marcs_grid, verify_sha256=False, expected_sha256=None)
        manifest = protocol_manifest(
            marcs_grid=marcs_grid,
            marcs_sha256=marcs_sha256,
            result_root=result_root,
            iteration_cap=args.iteration_cap,
        )
        manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest["manifest_hash"] = manifest_hash
        _write_json(result_root / "manifest.json", manifest)

        track_arguments = [
            {
                "track": track,
                "schema": schema,
                "result_root": result_root,
                "iteration_cap": args.iteration_cap,
                "manifest_hash": manifest_hash,
            }
            for track in build_mstar_tracks()
        ]
        if args.workers == 1:
            track_payloads = [_track_campaign(**kwargs) for kwargs in track_arguments]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(_track_campaign, **kwargs) for kwargs in track_arguments]
                track_payloads = [future.result() for future in futures]
        cases = _selected_cases(track_payloads)
        _refresh_case_json(cases_path, cases, manifest_hash)

        all_records = []
        for payload in track_payloads:
            all_records.extend(payload["records"])
        _write_jsonl(result_root / "records.jsonl", all_records)

    if args.run_pz_spectra:
        _run_payne_zero_spectra(result_root, cases, smoke=args.smoke)
        _refresh_case_json(cases_path, cases, manifest_hash)
    if args.run_korg:
        korg_manifest = _write_korg_inputs(result_root, cases)
        run_korg(
            result_root=result_root,
            manifest_path=korg_manifest,
            julia=Path(args.julia).expanduser().resolve(),
            korg_root=Path(args.korg_root).expanduser().resolve(),
            smoke=args.smoke,
        )
        _attach_korg_metrics(result_root, cases, smoke=args.smoke)
        _refresh_case_json(cases_path, cases, manifest_hash)

    successful = [case for case in cases if case.get("selected_product_path")]
    direct_converged = sum(bool(case["direct"].get("survives_solver")) for case in cases)
    continuation_converged = sum(bool(case["continuation_primary"].get("survives_solver")) for case in cases)
    summary = {
        "campaign": "m_star_science_case_v1",
        "manifest_hash": manifest_hash,
        "case_count": len(cases),
        "selected_product_count": len(successful),
        "direct_solver_converged": direct_converged,
        "continuation_solver_converged": continuation_converged,
        "direct_within_15": sum(bool(case["direct"].get("within_15_iterations")) for case in cases),
        "continuation_within_15": sum(bool(case["continuation_primary"].get("within_15_iterations")) for case in cases),
        "failed_cases": [case["case_id"] for case in cases if not case.get("selected_product_path")],
        "production_routing_changed": False,
        "sealed_holdout_opened": False,
        "note": "A selected product enables comparison only; it does not promote a failed route or alter production support.",
    }
    _write_json(result_root / "campaign.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marcs-grid", type=Path, default=DEFAULT_MARCS_GRID)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--iteration-cap", type=int, default=ITERATION_CAP)
    parser.add_argument("--workers", type=int, default=1, help="independent dwarf/giant track workers")
    parser.add_argument("--julia", type=Path, default=DEFAULT_JULIA)
    parser.add_argument("--korg-root", type=Path, default=DEFAULT_KORG_ROOT)
    parser.add_argument("--smoke", action="store_true", help="5000--5020 A smoke comparison")
    parser.add_argument("--run-pz-spectra", action="store_true")
    parser.add_argument("--run-korg", action="store_true")
    parser.add_argument("--skip-solver", action="store_true", help="reuse an existing complete solver campaign")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    summary = run_campaign(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
