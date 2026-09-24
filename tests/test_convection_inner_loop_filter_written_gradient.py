"""Written-gradient correction with the alternating component of the written gradient removed."""

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
    "convection_inner_loop_filter_written_gradient_isolated",
    "payne_zero_atmosphere/convection_inner_loop.py",
)

N_LAYERS = 30
SLAB_TOP = 10
WRITTEN_EXCESS = 0.012
INPUT_FLUX_OVER_REQUIRED = 1.6
RADIATIVE_FLUX = 0.05
MODE = 1.0e-3
WINDOW = slice(SLAB_TOP + 3, N_LAYERS - 4)


def _structure(*, mode: float = 0.0):
    """∇_ad rises 0.012 per layer; the slab is written 0.012 superadiabatic.

    ``mode`` adds ``mode (-1)^i`` to ln T inside the slab.
    """

    pressure = np.logspace(3.0, 7.0, N_LAYERS)
    nabla_ad = 0.12 + 0.012 * np.arange(N_LAYERS)
    written = nabla_ad - 0.03
    written[SLAB_TOP:] = nabla_ad[SLAB_TOP:] + WRITTEN_EXCESS
    temperature = np.empty(N_LAYERS)
    temperature[0] = 2500.0
    for index in range(1, N_LAYERS):
        temperature[index] = temperature[index - 1] * np.exp(
            written[index] * np.log(pressure[index] / pressure[index - 1])
        )
    layers = np.arange(N_LAYERS)
    slab = layers > SLAB_TOP
    temperature[slab] *= np.exp(mode * (-1.0) ** layers[slab])
    return temperature, pressure, nabla_ad


def _centred_physics(pressure: np.ndarray, nabla_ad: np.ndarray):
    """Read ∇ with a centred difference (one-sided at the ends), F ∝ δ^1.5."""

    def physics(temperature: np.ndarray) -> loop.ConvectivePhysicsSnapshot:
        nabla = np.gradient(np.log(temperature), np.log(pressure))
        delta = np.maximum(nabla - nabla_ad, 0.0)
        return loop.ConvectivePhysicsSnapshot(
            convective_flux=delta**1.5,
            logarithmic_gradient=nabla,
            adiabatic_gradient=nabla_ad,
            total_pressure=pressure,
        )

    return physics


def _run(temperature, pressure, nabla_ad, *, filtered: bool, passes: int = 8):
    physics = _centred_physics(pressure, nabla_ad)
    smooth, _, _ = _structure()
    required = float(physics(smooth).convective_flux[SLAB_TOP + 5]) / INPUT_FLUX_OVER_REQUIRED
    result = loop.run_convection_zone_inner_loop(
        temperature=temperature,
        radiative_eddington_flux=np.full(N_LAYERS, RADIATIVE_FLUX),
        target_eddington_flux=RADIATIVE_FLUX + required,
        physics=physics,
        config=loop.ConvectiveInnerLoopConfig(
            passes=passes, correct_written_gradient=True, filter_written_gradient=filtered,
        ),
    )
    return result, required


def _alternating(values: np.ndarray) -> float:
    index = np.arange(N_LAYERS)[WINDOW]
    residual = values[index] - 0.5 * (values[index - 1] + values[index + 1])
    return float(np.mean(residual * (-1.0) ** index) / 2.0)


def test_filter_removes_the_alternating_component_and_keeps_quadratics() -> None:
    layers = np.arange(40.0)
    mask = (layers >= 12) & (layers != 30)
    quadratic = 0.1 + 0.004 * layers + 3.0e-5 * layers**2
    alternating = 0.02 * (-1.0) ** layers
    assert np.max(np.abs(loop.remove_alternating_component(quadratic, mask) - quadratic)) < 1.0e-14
    filtered = loop.remove_alternating_component(quadratic + alternating, mask)
    assert np.max(np.abs(filtered[mask] - quadratic[mask])) < 1.0e-14
    assert np.array_equal(filtered[~mask], (quadratic + alternating)[~mask])
    short = np.zeros(40, dtype=bool)
    short[5:9] = True
    assert np.array_equal(loop.remove_alternating_component(alternating, short), alternating)


def _imposed_mode(filtered: bool, passes: int) -> float:
    """Alternating amplitude the added mode leaves after ``passes``, net of the smooth profile."""

    with_mode, pressure, nabla_ad = _structure(mode=MODE)
    smooth, _, _ = _structure()
    after_mode, _ = _run(with_mode, pressure, nabla_ad, filtered=filtered, passes=passes)
    after_smooth, _ = _run(smooth, pressure, nabla_ad, filtered=filtered, passes=passes)
    return _alternating(np.log(after_mode.temperature)) - _alternating(np.log(after_smooth.temperature))


def test_correction_keeps_an_alternating_mode_that_the_filter_removes() -> None:
    with_mode, _, _ = _structure(mode=MODE)
    smooth, _, _ = _structure()
    imposed = _alternating(np.log(with_mode)) - _alternating(np.log(smooth))
    assert abs(imposed / MODE - 1.0) < 0.01
    # the centred read does not see the mode, so the corrected update carries it
    assert abs(_imposed_mode(False, 2) / imposed - 1.0) < 0.01
    assert _imposed_mode(False, 8) / imposed > 0.7
    assert abs(_imposed_mode(True, 1) / imposed) < 0.01
    assert abs(_imposed_mode(True, 8) / imposed) < 0.01


def test_filtered_correction_still_meets_the_required_flux() -> None:
    temperature, pressure, nabla_ad = _structure()
    result, required = _run(temperature, pressure, nabla_ad, filtered=True)
    ratio = result.convective_flux[SLAB_TOP:] / required
    # the unfiltered correction meets the target to 1e-9 here by carrying the
    # decaying odd-even ripple that the one-sided read at the grid bottom sets
    # off; with that ripple filtered, a residual of order 1e-5 reaches upward
    assert np.max(np.abs(ratio[:10] - 1.0)) < 1.0e-4
    assert np.mean(np.abs(ratio - 1.0)) < 0.03
    assert result.assigned_required_flux is False


def test_written_gradient_filter_is_off_by_default() -> None:
    assert loop.ConvectiveInnerLoopConfig().filter_written_gradient is False
    config = (REPO / "payne_zero_atmosphere" / "config.py").read_text()
    assert "convection_zone_inner_loop_filter_written_gradient: bool = False" in config
    runner = (REPO / "payne_zero_atmosphere" / "runner.py").read_text()
    assert "filter_written_gradient=bool(" in runner
