import json
from pathlib import Path

import numpy as np
import pytest

from experiments.reduced_state_emulator.m_star_bootstrap_v1 import (
    FLUX_METRICS,
    MINIMUM_REFERENCE_SUCCESSES,
    PATH_COLUMN_MASS_P95_DEX_LIMIT,
    PATH_TEMPERATURE_P95_LIMIT,
    SPINE_TEMPERATURES,
    _passes_flux_gate,
    _product_consistency,
    build_tracks,
    freeze_flux_gate,
    protocol_payload,
)


def _reference_record(node_id: str, role: str, scale: float = 1.0) -> dict:
    return {
        "node_id": node_id,
        "split_role": role,
        "survives_solver": True,
        "solver_diagnostics": {
            "final_diagnostics": {
                "median_absolute_flux_error_percent": 1.0 * scale,
                "p95_absolute_flux_error_percent": 2.0 * scale,
                "maximum_absolute_flux_error_percent": 4.0 * scale,
            }
        },
    }


def test_protocol_has_210_nodes_and_trackwise_balanced_roles(tmp_path: Path) -> None:
    tracks = build_tracks()
    assert len(tracks) == 10
    assert len(SPINE_TEMPERATURES) == 21
    assert {row["class"] for row in tracks} == {"giant", "dwarf"}
    for role, expected in {"train": 6, "validation": 2, "sealed": 2}.items():
        selected = [row for row in tracks if row["role"] == role]
        assert len(selected) == expected
        if role != "train":
            assert {row["class"] for row in selected} == {"giant", "dwarf"}

    protocol = protocol_payload(tmp_path)
    assert protocol["grid"]["target_node_count"] == 210
    assert protocol["boundaries"]["production_routing_changed"] is False
    assert protocol["boundaries"]["existing_sealed_holdout_opened"] is False
    assert protocol["split"]["sealed_not_run_by_default"] is True


def test_flux_gate_is_frozen_from_reference_only(tmp_path: Path) -> None:
    protocol = protocol_payload(tmp_path)
    open_tracks = [
        row
        for row in protocol["split"]["tracks"]
        if row["role"] in {"train", "validation"}
    ]
    records = [
        _reference_record(f"reference_{index}", row["role"], scale=1.0 + index / 100)
        for index, row in enumerate(open_tracks * 2)
    ]
    assert len(records) >= MINIMUM_REFERENCE_SUCCESSES
    cursor = 0
    for row in open_tracks:
        path = tmp_path / "reference" / row["track_id"] / "reference.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"records": records[cursor : cursor + 2]}))
        cursor += 2

    gate = freeze_flux_gate(
        tmp_path,
        protocol=protocol,
        roles={"train", "validation"},
    )
    assert gate["frozen"]
    assert gate["reference_success_count"] == len(records)
    largest_scale = 1.0 + (len(records) - 1) / 100
    assert gate["thresholds"]["maximum_absolute_flux_error_percent"] == pytest.approx(
        1.25 * 4.0 * largest_scale
    )
    assert "gate_hash" in gate


def test_flux_gate_fail_stops_when_reference_panel_is_incomplete(
    tmp_path: Path,
) -> None:
    protocol = protocol_payload(tmp_path)
    first = protocol["split"]["tracks"][0]
    path = tmp_path / "reference" / first["track_id"] / "reference.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"records": [_reference_record("only_one", first["role"])]})
    )
    gate = freeze_flux_gate(tmp_path, protocol=protocol, roles={"train"})
    assert gate["frozen"] is False
    assert gate["status"] == "fail_stop_insufficient_reference_solves"
    assert gate["thresholds"] == {}


def test_flux_gate_checks_all_three_metrics() -> None:
    thresholds = {metric: 10.0 for metric in FLUX_METRICS}
    gate = {"thresholds": thresholds}
    record = _reference_record("node", "train")
    assert _passes_flux_gate(record, gate)["passes"]
    record["solver_diagnostics"]["final_diagnostics"][
        "p95_absolute_flux_error_percent"
    ] = 11.0
    result = _passes_flux_gate(record, gate)
    assert result["passes"] is False
    assert (
        result["metrics"]["p95_absolute_flux_error_percent"]["passes"] is False
    )


def test_product_consistency_uses_preregistered_profile_limits(
    tmp_path: Path,
) -> None:
    mass = np.geomspace(1.0e-6, 100.0, 80)
    temperature = np.linspace(2800.0, 6000.0, 80)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez(first, column_mass=mass, temperature=temperature)
    np.savez(
        second,
        column_mass=mass * 10.0 ** (0.5 * PATH_COLUMN_MASS_P95_DEX_LIMIT),
        temperature=temperature * (1.0 + 0.5 * PATH_TEMPERATURE_P95_LIMIT),
    )
    result = _product_consistency(first, second)
    assert result["passes"]

    np.savez(
        second,
        column_mass=mass * 10.0 ** (2.0 * PATH_COLUMN_MASS_P95_DEX_LIMIT),
        temperature=temperature,
    )
    result = _product_consistency(first, second)
    assert result["passes"] is False
