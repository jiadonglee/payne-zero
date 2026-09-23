"""Inner loop under a centred MLT read-back with ∇_ad rising steeply with depth."""

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
    "convection_inner_loop_written_gradient_isolated",
    "payne_zero_atmosphere/convection_inner_loop.py",
)

N_LAYERS = 30
SLAB_TOP = 10
ISOLATED_LAYER = 5
WRITTEN_EXCESS = 0.012
INPUT_FLUX_OVER_REQUIRED = 1.6
RADIATIVE_FLUX = 0.05


def _structure(*, isolated_layer: bool = False):
    """∇_ad rises 0.012 per layer; the slab is written 0.012 superadiabatic.

    ``isolated_layer`` adds one layer that reads superadiabatic between two
    layers that read subadiabatic.
    """

    pressure = np.logspace(3.0, 7.0, N_LAYERS)
    nabla_ad = 0.12 + 0.012 * np.arange(N_LAYERS)
    written = nabla_ad - 0.03
    written[SLAB_TOP:] = nabla_ad[SLAB_TOP:] + WRITTEN_EXCESS
    if isolated_layer:
        pair = slice(ISOLATED_LAYER, ISOLATED_LAYER + 2)
        written[pair] = nabla_ad[pair] + 0.01
    temperature = np.empty(N_LAYERS)
    temperature[0] = 2500.0
    for index in range(1, N_LAYERS):
        temperature[index] = temperature[index - 1] * np.exp(
            written[index] * np.log(pressure[index] / pressure[index - 1])
        )
    return temperature, pressure, nabla_ad


def _centred_physics(pressure: np.ndarray, nabla_ad: np.ndarray):
    """Read ∇ with a centred difference (one-sided at the ends), F ∝ δ^1.5."""

    temperatures: list[np.ndarray] = []

    def physics(temperature: np.ndarray) -> loop.ConvectivePhysicsSnapshot:
        temperatures.append(np.asarray(temperature, dtype=np.float64).copy())
        nabla = np.gradient(np.log(temperature), np.log(pressure))
        delta = np.maximum(nabla - nabla_ad, 0.0)
        return loop.ConvectivePhysicsSnapshot(
            convective_flux=delta**1.5,
            logarithmic_gradient=nabla,
            adiabatic_gradient=nabla_ad,
            total_pressure=pressure,
        )

    return physics, temperatures


def _run(temperature, pressure, nabla_ad, *, correct: bool, passes: int):
    physics, temperatures = _centred_physics(pressure, nabla_ad)
    input_flux = physics(temperature).convective_flux
    temperatures.clear()
    required = float(input_flux[SLAB_TOP + 5]) / INPUT_FLUX_OVER_REQUIRED
    result = loop.run_convection_zone_inner_loop(
        temperature=temperature,
        radiative_eddington_flux=np.full(N_LAYERS, RADIATIVE_FLUX),
        target_eddington_flux=RADIATIVE_FLUX + required,
        physics=physics,
        config=loop.ConvectiveInnerLoopConfig(
            passes=passes, correct_written_gradient=correct,
        ),
    )
    return result, required, temperatures


def test_read_back_prescription_settles_above_the_required_flux() -> None:
    temperature, pressure, nabla_ad = _structure()
    result, required, temperatures = _run(
        temperature, pressure, nabla_ad, correct=False, passes=12,
    )
    interior = slice(SLAB_TOP + 1, N_LAYERS - 1)
    ratio = result.convective_flux[interior] / required
    # written excess settles at the target, the centred read adds half the
    # per-layer rise of ∇_ad on top: (0.0131 + 0.006)^1.5 / 0.0131^1.5
    assert np.all(ratio > 1.6)
    assert np.max(np.abs(temperatures[-1] - temperatures[-2])) < 1.0e-6
    # the first pass heats the slab although the input flux already exceeds
    # the requirement everywhere in it
    assert np.all(temperatures[1][SLAB_TOP + 1:] > temperature[SLAB_TOP + 1:])


def test_written_gradient_correction_meets_the_required_flux() -> None:
    temperature, pressure, nabla_ad = _structure()
    result, required, temperatures = _run(
        temperature, pressure, nabla_ad, correct=True, passes=8,
    )
    ratio = result.convective_flux[SLAB_TOP:] / required
    # away from the grid bottom the recomputed flux meets the target exactly
    assert np.max(np.abs(ratio[:10] - 1.0)) < 1.0e-9
    # the one-sided read at the grid bottom differs from the centred interior
    # offset; that step spreads upward as a decaying odd-even ripple
    assert np.mean(np.abs(ratio - 1.0)) < 0.03
    assert np.max(np.abs(ratio - 1.0)) < 0.15
    assert np.all(temperatures[1][SLAB_TOP + 1:] < temperature[SLAB_TOP + 1:])
    assert result.assigned_required_flux is False


def test_written_gradient_correction_leaves_an_uncontrollable_edge_bounded() -> None:
    temperature, pressure, nabla_ad = _structure(isolated_layer=True)
    physics, _ = _centred_physics(pressure, nabla_ad)
    mask = loop.convective_mask_from_gradients(
        physics(temperature).logarithmic_gradient, nabla_ad,
    )
    assert mask[ISOLATED_LAYER]
    assert not mask[ISOLATED_LAYER - 1] and not mask[ISOLATED_LAYER + 1]

    # the isolated layer's centred read spans its two held neighbours, so its
    # own temperature cannot move it; the layer takes one direct step and stays
    result, _required, temperatures = _run(
        temperature, pressure, nabla_ad, correct=True, passes=8,
    )
    edge = np.array([t[ISOLATED_LAYER] for t in temperatures])
    assert abs(edge[1] - edge[0]) > 0.1
    assert np.max(np.abs(np.diff(edge[1:]))) < 1.0e-9 * edge[0]
    assert result.convective_mask_initial[ISOLATED_LAYER]


def test_pass_record_reports_the_read_minus_written_gradient() -> None:
    temperature, pressure, nabla_ad = _structure()
    off, _, _ = _run(temperature, pressure, nabla_ad, correct=False, passes=3)
    on, _, _ = _run(temperature, pressure, nabla_ad, correct=True, passes=3)
    for result in (off, on):
        gap = result.pass_records[-1]["max_abs_read_minus_written_gradient"]
        assert 0.005 < gap < 0.02


def test_written_gradient_correction_is_off_by_default() -> None:
    assert loop.ConvectiveInnerLoopConfig().correct_written_gradient is False
    config = (REPO / "payne_zero_atmosphere" / "config.py").read_text()
    assert "convection_zone_inner_loop_correct_written_gradient: bool = False" in config
    runner = (REPO / "payne_zero_atmosphere" / "runner.py").read_text()
    assert "correct_written_gradient=bool(" in runner


def test_state_refresh_is_off_by_default_and_guarded() -> None:
    config = (REPO / "payne_zero_atmosphere" / "config.py").read_text()
    assert "convection_zone_inner_loop_refresh_state: bool = False" in config
    runner = (REPO / "payne_zero_atmosphere" / "runner.py").read_text()
    helper = runner[runner.index("def _refresh_state_at_current_temperature"):]
    helper = helper[: helper.index("def _physics_at_temperature")]
    assert "pressure_iteration_enabled=True" in helper
    callback = runner[runner.index("def _physics_at_temperature"):]
    callback = callback[: callback.index("samples = compute_convection_finite_difference_samples(")]
    assert "if setup.convection_zone_inner_loop_refresh_state:" in callback
    assert "_refresh_state_at_current_temperature(" in callback


def test_correction_hold_is_off_by_default_and_zeroes_only_held_layers() -> None:
    config = (REPO / "payne_zero_atmosphere" / "config.py").read_text()
    assert "convection_zone_inner_loop_hold_correction: bool = False" in config
    runner = (REPO / "payne_zero_atmosphere" / "runner.py").read_text()
    assert "correction_hold_layers: np.ndarray | None = None" in runner
    assert "hold_layers=correction_hold_layers," in runner
    assert "inner.convective_mask_initial | inner.filled_layers" in runner
    assert ") & inner.convective_mask_final" in runner
    correction = (REPO / "payne_zero_atmosphere" / "temperature_correction.py").read_text()
    assert "hold_layers: np.ndarray | None = None," in correction
    held_block = correction[correction.index("if hold_layers is not None:"):]
    held_block = held_block[: held_block.index("new_temperature = temperature + temperature_correction")]
    assert "temperature_correction[held] = 0.0" in held_block
