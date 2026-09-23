"""COOLTLUSTY-style inner loop: physics flux, mask freeze/release, default off."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


loop = _load(
    "convection_inner_loop_isolated",
    "payne_zero_atmosphere/convection_inner_loop.py",
)


def _structure(n: int = 10):
    pressure = np.logspace(3.0, 7.0, n)
    temperature = np.empty(n, dtype=np.float64)
    temperature[0] = 3200.0
    nabla_ad = np.full(n, 0.4)
    nabla = np.full(n, 0.4)
    nabla[4:] = 0.55
    for index in range(1, n):
        temperature[index] = temperature[index - 1] * np.exp(
            nabla[index] * np.log(pressure[index] / pressure[index - 1])
        )
    return temperature, pressure, nabla_ad


def _physics_factory(pressure: np.ndarray, nabla_ad: np.ndarray, scale: float = 0.05):
    calls = {"count": 0, "fluxes": []}

    def physics(temperature: np.ndarray) -> loop.ConvectivePhysicsSnapshot:
        calls["count"] += 1
        nabla = np.zeros_like(temperature)
        nabla[0] = 0.4
        for index in range(1, temperature.size):
            nabla[index] = np.log(
                max(temperature[index], 1.0) / max(temperature[index - 1], 1.0)
            ) / np.log(pressure[index] / pressure[index - 1])
        delta = np.maximum(nabla - nabla_ad, 0.0)
        flux = scale * delta**1.5
        calls["fluxes"].append(np.asarray(flux, dtype=np.float64).copy())
        return loop.ConvectivePhysicsSnapshot(
            convective_flux=flux,
            logarithmic_gradient=nabla,
            adiabatic_gradient=nabla_ad,
            total_pressure=pressure,
        )

    return physics, calls


def test_required_flux_is_a_target_not_the_stored_flux() -> None:
    temperature, pressure, nabla_ad = _structure()
    physics, calls = _physics_factory(pressure, nabla_ad)
    target = 1.0
    radiative = np.full(temperature.shape, 0.6)
    result = loop.run_convection_zone_inner_loop(
        temperature=temperature,
        radiative_eddington_flux=radiative,
        target_eddington_flux=target,
        physics=physics,
        config=loop.ConvectiveInnerLoopConfig(passes=4),
    )
    assert result.assigned_required_flux is False
    assert calls["count"] == result.physics_evaluations
    assert calls["count"] >= 5
    assert not np.allclose(result.convective_flux, result.required_convective_flux)
    assert np.allclose(result.convective_flux, calls["fluxes"][-1])
    assert np.all(result.required_convective_flux == pytest.approx(0.4))


def test_mask_is_frozen_then_released() -> None:
    temperature, pressure, nabla_ad = _structure()
    physics, _calls = _physics_factory(pressure, nabla_ad)
    result = loop.run_convection_zone_inner_loop(
        temperature=temperature,
        radiative_eddington_flux=np.full(temperature.shape, 0.6),
        target_eddington_flux=1.0,
        physics=physics,
        config=loop.ConvectiveInnerLoopConfig(
            passes=3,
            freeze_mask_during_passes=True,
            release_mask_after=True,
        ),
    )
    assert result.mask_was_frozen is True
    assert result.mask_was_released is True
    assert not np.all(result.convective_mask_initial)
    assert result.convective_mask_initial[0:3].sum() == 0
    source = (REPO / "payne_zero_atmosphere" / "convection_inner_loop.py").read_text()
    assert "assigned_required_flux=False" in source
    assert "convective_flux[:] = required" not in source
    assert "never write this into" in source.lower() or "Never write this" in source


def test_inner_loop_reduces_local_mismatch_without_assigning_flux() -> None:
    temperature, pressure, nabla_ad = _structure()
    physics, _calls = _physics_factory(pressure, nabla_ad)
    radiative = np.full(temperature.shape, 0.7)
    target = 1.0
    before = physics(temperature)
    mismatch_before = loop.local_energy_mismatch(
        radiative_eddington_flux=radiative,
        convective_flux=before.convective_flux,
        target_eddington_flux=target,
    )
    result = loop.run_convection_zone_inner_loop(
        temperature=temperature,
        radiative_eddington_flux=radiative,
        target_eddington_flux=target,
        physics=physics,
        config=loop.ConvectiveInnerLoopConfig(passes=6),
    )
    active = result.convective_mask_initial
    assert np.any(active)
    assert np.mean(np.abs(result.local_energy_mismatch[active])) < np.mean(
        np.abs(mismatch_before[active])
    )


def test_runner_does_not_call_inner_loop_on_the_default_path() -> None:
    runner = (REPO / "payne_zero_atmosphere" / "runner.py").read_text()
    config = (REPO / "payne_zero_atmosphere" / "config.py").read_text()
    assert "convection_zone_inner_loop_passes: int = 0" in config
    assert "if inner_passes > 0:" in runner
    assert "run_convection_zone_inner_loop" in runner
    experiment = (
        REPO
        / "experiments"
        / "reduced_state_emulator"
        / "m_star_convection_inner_loop_v1.py"
    ).read_text()
    assert "INNER_PASSES = 8" in experiment
    inner = (REPO / "payne_zero_atmosphere" / "convection_inner_loop.py").read_text()
    assert "DEFAULT_INNER_PASSES = 8" in inner
    tomography = (
        REPO
        / "experiments"
        / "reduced_state_emulator"
        / "m_star_iteration_tomography_v1.py"
    ).read_text()
    for marker in ("3500.0", "3400.0", "3300.0", "3600.0"):
        assert marker in tomography
        assert marker in experiment
    assert "g+4.50_m+0.00_a+0.00_c+0.00_x1.00" in experiment
    assert "g+4.50_m-0.50_a+0.00_c+0.00_x1.00" in experiment
