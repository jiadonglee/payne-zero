"""Tests for the experimental residual-guided solver-policy knobs.

These pin the default-inert contract (both flags off reproduce the
historical solver exactly) and the pure step-scale scheduler's behavior:
halve on a worsening p95 flux error, floor at 0.125, restore by 1.5x
after two consecutive non-worsening iterations, and treat non-finite
residuals as "no information".
"""

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
from payne_zero_atmosphere.temperature_correction import (
    FLUX_RESIDUAL_MIN_STEP_SCALE,
    next_flux_residual_step_scale,
)


def test_policy_flags_default_off():
    assert AtmosphereConfig.__dataclass_fields__["flux_residual_guided_damping"].default is False
    assert (
        AtmosphereConfig.__dataclass_fields__["require_improving_flux_residual"].default
        is False
    )


def _config(**kwargs) -> AtmosphereConfig:
    return AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=SimpleNamespace(layers=3, metadata={})
        ),
        outputs=AtmosphereOutput(),
        **kwargs,
    )


def test_resolve_run_setup_carries_the_flags(monkeypatch):
    monkeypatch.setattr(
        "payne_zero_atmosphere.run_setup.validate_atmosphere_seed",
        lambda _: None,
    )
    monkeypatch.setattr(
        "payne_zero_atmosphere.run_setup.initialize_microturbulence",
        lambda *args, **kwargs: None,
    )
    setup = resolve_run_setup(
        _config(
            flux_residual_guided_damping=True,
            require_improving_flux_residual=True,
        )
    )
    assert setup.flux_residual_guided_damping is True
    assert setup.require_improving_flux_residual is True


def test_scheduler_keeps_scale_without_history():
    assert next_flux_residual_step_scale(None, 12.0) == (1.0, 0)
    assert next_flux_residual_step_scale(12.0, float("nan")) == (1.0, 0)


def test_scheduler_halves_on_worsening_and_floors():
    scale, streak = next_flux_residual_step_scale(10.0, 16.0, current_scale=1.0)
    assert scale == pytest.approx(0.5) and streak == 0
    scale, _ = next_flux_residual_step_scale(
        10.0, 100.0, current_scale=0.2, improving_streak=0
    )
    assert scale == pytest.approx(FLUX_RESIDUAL_MIN_STEP_SCALE)


def test_scheduler_restores_after_three_improvements():
    # The first two non-worsening iterations keep the scale but arm the
    # streak.
    scale, streak = next_flux_residual_step_scale(
        10.0, 9.0, current_scale=0.25, improving_streak=0
    )
    assert scale == pytest.approx(0.25) and streak == 1
    scale, streak = next_flux_residual_step_scale(
        9.0, 8.0, current_scale=0.25, improving_streak=1
    )
    assert scale == pytest.approx(0.25) and streak == 2
    # The third consecutive improvement restores by 1.25x.
    scale, streak = next_flux_residual_step_scale(
        8.0, 7.0, current_scale=0.25, improving_streak=2
    )
    assert scale == pytest.approx(0.3125) and streak == 3


def test_scheduler_restoration_caps_at_one():
    scale, streak = next_flux_residual_step_scale(
        8.0, 7.0, current_scale=0.9, improving_streak=2
    )
    assert scale == pytest.approx(1.0) and streak == 3


def test_scheduler_tolerates_small_bounces():
    # A 20 percent bounce is inside the worsening tolerance: the scale is
    # kept (not halved) and the improvement streak restarts at one.
    scale, streak = next_flux_residual_step_scale(
        10.0, 12.0, current_scale=0.5, improving_streak=0
    )
    assert scale == pytest.approx(0.5) and streak == 1


def test_scheduler_halves_on_a_monotone_drift_past_tolerance():
    scale, streak = next_flux_residual_step_scale(
        10.0, 15.1, current_scale=1.0, improving_streak=2
    )
    assert scale == pytest.approx(0.5) and streak == 0


def test_scheduler_ignores_non_finite_residuals():
    scale, streak = next_flux_residual_step_scale(
        float("nan"), 12.0, current_scale=0.5, improving_streak=3
    )
    assert scale == pytest.approx(0.5) and streak == 0
