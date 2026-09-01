"""Unit tests for the bounded native-MARCS M-star campaign plumbing."""

from __future__ import annotations

from experiments.reduced_state_emulator.m_star_science_case import (
    CONTINUATION_LADDER,
    TEMPERATURES,
    _ensure_flux_imbalance_summary,
    build_mstar_tracks,
    case_id,
    protocol_manifest,
    star_class,
)


def test_mstar_protocol_has_exact_eight_native_nodes():
    tracks = build_mstar_tracks()
    assert [(track.log_surface_gravity, track.microturbulence_km_s) for track in tracks] == [
        (5.0, 1.0),
        (1.5, 2.0),
    ]
    assert len(tracks) * len(TEMPERATURES) == 8
    assert CONTINUATION_LADDER == (4000.0, 3750.0, 3500.0, 3300.0, 3000.0)
    assert star_class(tracks[0]) == "M-dwarf"
    assert star_class(tracks[1]) == "M-giant"
    assert case_id(tracks[0], 3300.0) == "m_dwarf_t3300"
    assert case_id(tracks[1], 3300.0) == "m_giant_t3300"


def test_mstar_manifest_freezes_solver_and_korg_scope(tmp_path):
    manifest = protocol_manifest(
        marcs_grid=tmp_path / "SDSS_MARCS_atmospheres.h5",
        marcs_sha256="frozen",
        result_root=tmp_path,
        iteration_cap=30,
    )
    assert manifest["cases"]["effective_temperature_K"] == [3000.0, 3300.0, 3500.0, 3750.0]
    assert manifest["solver"]["iteration_cap"] == 30
    assert "only" in manifest["native_marcs"]["nodes"]
    assert manifest["korg"]["geometry"].startswith("planar")
    assert "molecular" in manifest["korg"]["molecular_coverage"]


def test_flux_summary_falls_back_to_solver_all_layer_maximum():
    record = {
        "flux_imbalance": {"available": False},
        "solver_diagnostics": {
            "final_diagnostics": {"maximum_absolute_flux_error_percent": 12.5}
        },
    }
    _ensure_flux_imbalance_summary(record)
    assert record["flux_imbalance"] == {
        "available": True,
        "vector_available": False,
        "max_percent": 12.5,
        "source": "unchanged solver final diagnostic; maximum over all atmosphere layers",
    }
