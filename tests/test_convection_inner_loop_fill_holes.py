"""Interior holes: subadiabatic layers inside a convective run."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

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
    "convection_inner_loop_fill_holes_isolated",
    "payne_zero_atmosphere/convection_inner_loop.py",
)

N_LAYERS = 30
SLAB_TOP = 10
HOLE = (18, 19)


def _structure():
    """A superadiabatic slab with written gradients below ∇_ad at 18–20.

    The centred read makes layers 18 and 19 subadiabatic and keeps 17 and 20
    superadiabatic.
    """

    pressure = np.logspace(3.0, 7.0, N_LAYERS)
    nabla_ad = 0.12 + 0.012 * np.arange(N_LAYERS)
    written = nabla_ad - 0.03
    written[SLAB_TOP:] = nabla_ad[SLAB_TOP:] + 0.012
    written[18:21] = nabla_ad[18:21] - 0.02
    temperature = np.empty(N_LAYERS)
    temperature[0] = 2500.0
    for index in range(1, N_LAYERS):
        temperature[index] = temperature[index - 1] * np.exp(
            written[index] * np.log(pressure[index] / pressure[index - 1])
        )
    return temperature, pressure, nabla_ad


def _physics(pressure, nabla_ad):
    def physics(temperature):
        nabla = np.gradient(np.log(temperature), np.log(pressure))
        delta = np.maximum(nabla - nabla_ad, 0.0)
        return loop.ConvectivePhysicsSnapshot(
            convective_flux=delta**1.5,
            logarithmic_gradient=nabla,
            adiabatic_gradient=nabla_ad,
            total_pressure=pressure,
        )

    return physics


def _run(*, fill: bool, radiative_fraction: float, passes: int = 8):
    temperature, pressure, nabla_ad = _structure()
    physics = _physics(pressure, nabla_ad)
    seed = physics(temperature)
    mask = loop.convective_mask_from_gradients(seed.logarithmic_gradient, nabla_ad)
    assert not mask[list(HOLE)].any() and mask[17] and mask[20]
    required = float(seed.convective_flux[12])
    radiative = required * radiative_fraction / (1.0 - radiative_fraction)
    result = loop.run_convection_zone_inner_loop(
        temperature=temperature,
        radiative_eddington_flux=np.full(N_LAYERS, radiative),
        target_eddington_flux=radiative + required,
        physics=physics,
        config=loop.ConvectiveInnerLoopConfig(
            passes=passes, correct_written_gradient=True, fill_interior_holes=fill,
        ),
    )
    return result, required


def test_hole_stays_far_from_balance_when_filling_is_off() -> None:
    result, required = _run(fill=False, radiative_fraction=0.02)
    assert not result.filled_layers.any()
    # neighbouring layers move the centred read inside the hole a little, but
    # the hole keeps carrying only a small part of what it has to
    assert np.all(result.convective_flux[list(HOLE)] / required < 0.2)


def test_filled_hole_carries_the_required_flux_and_releases_convective() -> None:
    result, required = _run(fill=True, radiative_fraction=0.02)
    assert np.flatnonzero(result.filled_layers).tolist() == list(HOLE)
    ratio = result.convective_flux[list(HOLE)] / required
    assert np.all(np.abs(ratio - 1.0) < 0.05)
    # the released mask is rebuilt from the gradient the loop produced
    assert result.convective_mask_final[list(HOLE)].all()
    assert not result.convective_mask_initial[list(HOLE)].any()
    assert result.assigned_required_flux is False


def test_gap_that_radiation_can_carry_is_not_filled() -> None:
    result, required = _run(fill=True, radiative_fraction=0.99)
    assert not result.filled_layers.any()
    assert np.all(result.convective_flux[list(HOLE)] / required < 0.2)


def test_hole_filling_is_off_by_default() -> None:
    assert loop.ConvectiveInnerLoopConfig().fill_interior_holes is False
    config = (REPO / "payne_zero_atmosphere" / "config.py").read_text()
    assert "convection_zone_inner_loop_fill_holes: bool = False" in config
    runner = (REPO / "payne_zero_atmosphere" / "runner.py").read_text()
    assert "fill_interior_holes=bool(setup.convection_zone_inner_loop_fill_holes)" in runner
    assert "inner.convective_mask_initial | inner.filled_layers" in runner
