import json
from pathlib import Path

from experiments.reduced_state_emulator.m_star_bootstrap_v1 import (
    FLUX_METRICS,
    protocol_payload as v1_protocol_payload,
)
from experiments.reduced_state_emulator.m_star_bootstrap_v1r1_policy60 import (
    CAMPAIGN,
    ITERATION_CAP,
    import_parent_flux_gate,
    protocol_payload,
)


def _write_parent(tmp_path: Path) -> Path:
    parent = tmp_path / "parent"
    parent.mkdir()
    protocol = v1_protocol_payload(parent)
    (parent / "protocol.json").write_text(json.dumps(protocol))
    gate = {
        "campaign": "m_star_emulator_v1",
        "status": "pass",
        "frozen": True,
        "gate_hash": "parent-gate-hash",
        "thresholds": {metric: index + 1.0 for index, metric in enumerate(FLUX_METRICS)},
    }
    (parent / "flux_gate.json").write_text(json.dumps(gate))
    return parent


def test_policy60_protocol_changes_only_declared_solver_policy(tmp_path: Path) -> None:
    parent = _write_parent(tmp_path)
    child = tmp_path / "child"
    payload = protocol_payload(child, parent)
    parent_protocol = json.loads((parent / "protocol.json").read_text())

    assert payload["campaign"] == CAMPAIGN
    assert payload["solver"]["iteration_cap"] == ITERATION_CAP == 60
    assert payload["solver"]["stopping_rule"] == (
        parent_protocol["solver"]["stopping_rule"]
    )
    assert payload["split"] == parent_protocol["split"]
    assert payload["training_eligibility"] == (
        parent_protocol["training_eligibility"]
    )
    assert payload["boundaries"]["existing_sealed_holdout_opened"] is False
    assert payload["boundaries"]["new_sealed_tracks_run"] is False


def test_policy60_imports_parent_gate_without_refitting(tmp_path: Path) -> None:
    parent = _write_parent(tmp_path)
    child = tmp_path / "child"
    child.mkdir()
    protocol = protocol_payload(child, parent)
    imported = import_parent_flux_gate(
        child,
        parent_root=parent,
        protocol=protocol,
    )
    parent_gate = json.loads((parent / "flux_gate.json").read_text())

    assert imported["frozen"] is True
    assert imported["thresholds_refit"] is False
    assert imported["thresholds"] == parent_gate["thresholds"]
    assert imported["source"]["gate_hash"] == parent_gate["gate_hash"]
    assert (child / "flux_gate.json").is_file()
