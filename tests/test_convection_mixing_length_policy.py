"""Tests for the experimental convection mixing-length knob."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from payne_zero_atmosphere.config import (
    AtmosphereConfig,
    AtmosphereInput,
    AtmosphereOutput,
)
from payne_zero_atmosphere.run_setup import resolve_run_setup


def test_mixing_length_defaults_to_production_none():
    assert AtmosphereConfig.__dataclass_fields__["convection_mixing_length"].default is None


def _config(mixing_length) -> AtmosphereConfig:
    return AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=SimpleNamespace(layers=3, metadata={})
        ),
        outputs=AtmosphereOutput(),
        convection_mixing_length=mixing_length,
    )


def _resolved(mixing_length):
    monkeypatch_flags = pytest.MonkeyPatch()
    monkeypatch_flags.setattr(
        "payne_zero_atmosphere.run_setup.validate_atmosphere_seed", lambda _: None
    )
    monkeypatch_flags.setattr(
        "payne_zero_atmosphere.run_setup.initialize_microturbulence",
        lambda *args, **kwargs: None,
    )
    try:
        return resolve_run_setup(_config(mixing_length))
    finally:
        monkeypatch_flags.undo()


def test_resolve_run_setup_carries_and_normalizes():
    assert _resolved(None).convection.mixing_length == 1.25
    assert _resolved(1.0).convection.mixing_length == 1.0
    assert _resolved(1.5).convection.mixing_length == 1.5


@pytest.mark.parametrize("mixing_length", [0.0, -1.0, float("nan")])
def test_resolve_run_setup_rejects_invalid(mixing_length):
    with pytest.raises(ValueError, match="convection_mixing_length"):
        _resolved(mixing_length)
