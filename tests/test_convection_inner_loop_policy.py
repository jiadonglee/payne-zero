"""Experimental config defaults for the convective inner loop.

These checks read source so they do not import the atmosphere package
``__init__`` (Numba/NumPy pin).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_inner_loop_defaults_to_production_off() -> None:
    config = (REPO / "payne_zero_atmosphere" / "config.py").read_text()
    setup = (REPO / "payne_zero_atmosphere" / "run_setup.py").read_text()
    runner = (REPO / "payne_zero_atmosphere" / "runner.py").read_text()
    assert "convection_zone_inner_loop_passes: int = 0" in config
    assert "convection_zone_inner_loop_freeze_mask: bool = True" in setup or (
        "convection_zone_inner_loop_freeze_mask: bool = True" in config
    )
    assert "convection_zone_inner_loop_passes must be non-negative" in setup
    assert "convection_zone_inner_loop_passes=convection_zone_inner_loop_passes" in setup
    assert "if inner_passes > 0:" in runner
    assert "assigned_required_flux" in runner
