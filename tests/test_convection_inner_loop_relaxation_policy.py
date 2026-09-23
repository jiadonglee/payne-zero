"""The inner-loop relaxation knob: 1.0 by default, validated in (0, 1]."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from payne_zero_atmosphere.config import (
    AtmosphereConfig,
    AtmosphereInput,
    AtmosphereOutput,
)
from payne_zero_atmosphere.run_setup import resolve_run_setup


def _config(relaxation: float) -> AtmosphereConfig:
    return AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=SimpleNamespace(layers=3, metadata={})
        ),
        outputs=AtmosphereOutput(),
        convection_zone_inner_loop_relaxation=relaxation,
    )


def test_inner_loop_relaxation_is_one_by_default():
    field = AtmosphereConfig.__dataclass_fields__["convection_zone_inner_loop_relaxation"]
    assert field.default == 1.0


@pytest.mark.parametrize("relaxation", [0.0, -0.2, 1.5, float("nan")])
def test_resolve_run_setup_rejects_invalid_inner_loop_relaxation(
    monkeypatch, relaxation: float
):
    monkeypatch.setattr(
        "payne_zero_atmosphere.run_setup.validate_atmosphere_seed",
        lambda _: None,
    )
    monkeypatch.setattr(
        "payne_zero_atmosphere.run_setup.initialize_microturbulence",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="convection_zone_inner_loop_relaxation"):
        resolve_run_setup(_config(relaxation))
