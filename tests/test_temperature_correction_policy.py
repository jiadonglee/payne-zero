"""Tests for the opt-in temperature-correction policy knob."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from payne_zero_atmosphere.config import (
    AtmosphereConfig,
    AtmosphereInput,
    AtmosphereOutput,
)
from payne_zero_atmosphere.run_setup import resolve_run_setup


def _config(damping: float) -> AtmosphereConfig:
    return AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=SimpleNamespace(layers=3, metadata={})
        ),
        outputs=AtmosphereOutput(),
        temperature_correction_damping=damping,
    )


def test_temperature_correction_damping_is_one_by_default():
    assert (
        AtmosphereConfig.__dataclass_fields__["temperature_correction_damping"].default
        == 1.0
    )


@pytest.mark.parametrize("damping", [0.0, -0.1, 1.01, float("nan")])
def test_resolve_run_setup_rejects_invalid_temperature_correction_damping(
    monkeypatch, damping: float
):
    monkeypatch.setattr(
        "payne_zero_atmosphere.run_setup.validate_atmosphere_seed",
        lambda _: None,
    )
    monkeypatch.setattr(
        "payne_zero_atmosphere.run_setup.initialize_microturbulence",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="temperature_correction_damping"):
        resolve_run_setup(_config(damping))
