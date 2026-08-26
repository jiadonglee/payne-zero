"""Tests for the differentiable initializer and the deck quantizer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from payne_zero_atmosphere.warm_start import load_atmosphere_initializer

from payne_zero_diffatm import check_initializer
from payne_zero_diffatm.initializer import (
    OUTPUT_FIELDS,
    DifferentiableInitializer,
    default_checkpoint_path,
)
from payne_zero_diffatm.quantize import (
    _round_decimals,
    _round_significant,
    quantize_columns,
    quantize_prediction,
)


SOLAR = dict(
    effective_temperature=5777.0,
    log_surface_gravity=4.44,
    metallicity=0.0,
    alpha_enhancement=0.0,
    microturbulence_km_s=2.0,
)


def _labels(mapping, dtype=torch.float64):
    return {key: torch.tensor([value], dtype=dtype) for key, value in mapping.items()}


@pytest.fixture(scope="module")
def checkpoint():
    return torch.load(default_checkpoint_path(), map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def model(checkpoint):
    module = DifferentiableInitializer(checkpoint)
    module.eval()
    return module


@pytest.fixture(scope="module")
def reference_model(checkpoint):
    module = DifferentiableInitializer(checkpoint, network_dtype=torch.float32)
    module.eval()
    return module


@pytest.fixture(autouse=True)
def _clear_gradients(request):
    """The models are module-scoped, so ``.grad`` would accumulate across tests."""

    for name in ("model", "reference_model"):
        if name in request.fixturenames:
            request.getfixturevalue(name).zero_grad(set_to_none=True)
    yield


# --- structure ---------------------------------------------------------------


def test_model_shape_matches_the_checkpoint(model, checkpoint):
    assert model.layer_count == 80
    assert len(model.coordinate_fields) == 6
    assert model.standard_rosseland_optical_depth.numel() == 80
    assert model.basis.shape == tuple(checkpoint["pca"]["basis"].shape)


def test_released_weights_load_unchanged(model, checkpoint):
    for name, parameter in model.network.state_dict().items():
        expected = checkpoint["model"]["state_dict"][name]
        assert torch.equal(parameter.to(expected.dtype), expected)


def test_prediction_stack_orders_fields_consistently(model):
    prediction = model(_labels(SOLAR))
    stacked = prediction.stack()
    assert stacked.shape == (1, 80, 6)
    for index, field in enumerate(OUTPUT_FIELDS):
        assert torch.equal(stacked[..., index], getattr(prediction, field))


# --- agreement with the released initializer ---------------------------------


def test_decode_is_exact(model, checkpoint):
    worst = check_initializer.check_decode(model, checkpoint)
    for field, value in worst.items():
        assert value < check_initializer.DECODE_TOLERANCE, field


def test_float32_matches_the_released_initializer(reference_model):
    worst = check_initializer.check_against_reference(reference_model)
    for field, value in worst.items():
        bound = check_initializer.tolerance_for(
            field,
            check_initializer.DECODE_TOLERANCE,
            sinh_tolerance=check_initializer.REFERENCE_SINH_TOLERANCE,
        )
        assert value < bound, field


def test_only_the_sinh_field_needs_the_looser_bound(reference_model):
    """The exemption must stay narrow: everything else holds at 1e-14.

    If a future change widens the disagreement beyond radiative acceleration,
    ``tolerance_for`` would quietly absorb it, so pin the other five fields to
    the strict bound explicitly.
    """

    worst = check_initializer.check_against_reference(reference_model)
    for field, value in worst.items():
        if field in check_initializer.SINH_AMPLIFIED_FIELDS:
            continue
        assert value < check_initializer.DECODE_TOLERANCE, field


def test_float64_network_is_batch_invariant(model):
    worst = check_initializer.check_batch_invariance(model)
    for field, value in worst.items():
        assert value < check_initializer.BATCH_TOLERANCE, field


def test_float32_network_is_not_batch_invariant(reference_model):
    """The reason float64 is the default; if this ever passes, revisit it."""

    worst = check_initializer.check_batch_invariance(reference_model)
    assert max(worst.values()) > 1e-7


def test_labels_are_widened_before_arithmetic(model):
    """``5040 / Teff`` must be computed at float64, whatever the input width.

    Every label here is exactly representable in float32, so the two calls
    differ only in the precision of the division itself. Computing it at
    float32 — which is what a bare ``torch.tensor([...])`` used to produce —
    costs ~6e-8 in the feature, and the deck's four-significant-digit
    quantization turns that into a visible last-digit difference.
    """

    exact_in_float32 = dict(
        effective_temperature=5777.0,
        log_surface_gravity=4.5,
        metallicity=0.0,
        alpha_enhancement=0.0,
        microturbulence_km_s=2.0,
    )
    wide = model(_labels(exact_in_float32, dtype=torch.float64))
    narrow = model(_labels(exact_in_float32, dtype=torch.float32))
    for field in OUTPUT_FIELDS:
        assert torch.equal(getattr(wide, field), getattr(narrow, field)), field


def test_feature_division_is_not_done_in_float32(model):
    """Directly: the feature must be the float64 quotient.

    Not an exact equality — torch computes ``scalar / tensor`` by
    reciprocal-multiply and lands one unit in the last place away from Python's
    correctly rounded divide, a relative 1e-16. The float32 mistake this guards
    against is 5.5e-9, seven orders of magnitude larger, so a 1e-13 tolerance
    separates the two cleanly.
    """

    feature = model.features(_labels(SOLAR, dtype=torch.float32))[0, 0].item()
    expected = 5040.0 / 5777.0
    assert feature == pytest.approx(expected, rel=1e-13)

    in_float32 = float(np.float32(5040.0) / np.float32(5777.0))
    assert abs(in_float32 - expected) / expected > 1e-9, "the guarded error is this big"
    assert feature != pytest.approx(in_float32, rel=1e-13)


# --- physical validity -------------------------------------------------------


def test_decoded_columns_are_physical(model):
    labels = {
        "effective_temperature": torch.tensor([4200.0, 5777.0, 9800.0], dtype=torch.float64),
        "log_surface_gravity": torch.tensor([1.2, 4.44, 4.0], dtype=torch.float64),
        "metallicity": torch.tensor([-2.0, 0.0, -1.0], dtype=torch.float64),
        "alpha_enhancement": torch.tensor([0.4, 0.0, 0.2], dtype=torch.float64),
        "microturbulence_km_s": torch.tensor([1.5, 2.0, 3.0], dtype=torch.float64),
    }
    prediction = model(labels)
    for field in ("column_mass", "temperature", "gas_pressure",
                  "electron_density", "rosseland_opacity"):
        values = getattr(prediction, field)
        assert torch.isfinite(values).all(), field
        assert (values > 0).all(), f"{field} must be positive"
    assert torch.isfinite(prediction.radiative_acceleration).all()


def test_column_mass_is_strictly_increasing(model):
    prediction = model(_labels(SOLAR))
    assert (prediction.column_mass.diff(dim=-1) > 0).all()


def test_increment_guard_repairs_an_absorbed_increment(model):
    """A tiny increment against a large running sum vanishes in float64.

    The reference raises on the resulting duplicate layer
    (``warm_start.py:686``); the guard must instead keep the sequence strictly
    increasing so training does not die on one bad batch element.
    """

    coordinates = torch.zeros(1, 80, 6, dtype=torch.float64)
    coordinates[..., 0] = 5.0          # increments of 1e5
    coordinates[0, 40:45, 0] = -30.0   # increments of 1e-30, absorbed entirely
    prediction = model.decode(coordinates, torch.tensor([5777.0], dtype=torch.float64))

    naive = torch.cumsum(torch.pow(10.0, coordinates[..., 0]), dim=-1)
    assert (naive.diff(dim=-1) == 0).any(), "the setup must actually trigger absorption"
    assert (prediction.column_mass.diff(dim=-1) > 0).all()


def test_increment_guard_can_be_disabled(checkpoint):
    unguarded = DifferentiableInitializer(checkpoint, minimum_relative_increment=0.0)
    coordinates = torch.zeros(1, 80, 6, dtype=torch.float64)
    coordinates[..., 0] = 5.0
    coordinates[0, 40:45, 0] = -30.0
    prediction = unguarded.decode(coordinates, torch.tensor([5777.0], dtype=torch.float64))
    assert (prediction.column_mass.diff(dim=-1) == 0).any()


def test_guard_leaves_a_healthy_prediction_untouched(model, checkpoint):
    unguarded = DifferentiableInitializer(checkpoint, minimum_relative_increment=0.0)
    unguarded.eval()
    with torch.no_grad():
        guarded_out = model(_labels(SOLAR))
        plain_out = unguarded(_labels(SOLAR))
    assert torch.equal(guarded_out.column_mass, plain_out.column_mass)


def test_batch_size_mismatch_is_rejected(model):
    coordinates = torch.zeros(3, 80, 6, dtype=torch.float64)
    with pytest.raises(ValueError, match="scalar or match"):
        model.decode(coordinates, torch.tensor([5777.0, 4000.0], dtype=torch.float64))


def test_scalar_temperature_broadcasts_over_a_batch(model):
    coordinates = torch.zeros(3, 80, 6, dtype=torch.float64)
    prediction = model.decode(coordinates, torch.tensor([5777.0], dtype=torch.float64))
    assert prediction.temperature.shape == (3, 80)


# --- gradients ---------------------------------------------------------------


def test_gradients_reach_every_weight(model):
    all_finite, norms = check_initializer.check_gradients(model)
    assert all_finite
    assert len(norms) == 14


def test_gradient_matches_finite_difference(model):
    """Directional derivative check on a scalar built from the decode."""

    def loss_of(parameters_delta: float, direction) -> float:
        with torch.no_grad():
            for parameter, step in zip(model.network.parameters(), direction):
                parameter.add_(parameters_delta * step)
        with torch.no_grad():
            value = model(_labels(SOLAR)).temperature.log().sum().item()
        with torch.no_grad():
            for parameter, step in zip(model.network.parameters(), direction):
                parameter.sub_(parameters_delta * step)
        return value

    generator = torch.Generator().manual_seed(0)
    direction = [
        torch.randn(p.shape, generator=generator, dtype=p.dtype) * 1e-3
        for p in model.network.parameters()
    ]

    model.zero_grad(set_to_none=True)
    model(_labels(SOLAR)).temperature.log().sum().backward()
    analytic = sum(
        float((p.grad * d).sum()) for p, d in zip(model.network.parameters(), direction)
    )

    epsilon = 1e-4
    numeric = (loss_of(epsilon, direction) - loss_of(-epsilon, direction)) / (2 * epsilon)
    assert numeric == pytest.approx(analytic, rel=1e-5)


# --- quantization ------------------------------------------------------------


@pytest.mark.parametrize("digits,fmt", [(3, "%.3E"), (8, "%.8E")])
def test_round_significant_matches_printf(digits, fmt):
    generator = torch.Generator().manual_seed(3)
    magnitude = torch.randint(-30, 10, (5000,), generator=generator).double()
    values = (torch.rand(5000, generator=generator, dtype=torch.float64) * 2 - 1) * torch.pow(
        10.0, magnitude
    )
    got = _round_significant(values, digits).numpy()
    expected = np.array([float(fmt % v) for v in values.numpy()])
    # One unit in the last place: the deck path re-parses a decimal string while
    # this scales by a power of ten, and the two can land on adjacent doubles.
    np.testing.assert_allclose(got, expected, rtol=1e-15, atol=0)


def test_round_decimals_matches_printf():
    generator = torch.Generator().manual_seed(4)
    values = torch.rand(5000, generator=generator, dtype=torch.float64) * 12000
    got = _round_decimals(values, 1).numpy()
    expected = np.array([float("%.1f" % v) for v in values.numpy()])
    np.testing.assert_array_equal(got, expected)


def test_round_significant_keeps_special_values():
    values = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan")],
                          dtype=torch.float64)
    got = _round_significant(values, 3)
    assert got[0] == 0.0 and got[1] == 0.0
    assert torch.isinf(got[2]) and torch.isinf(got[3])
    assert torch.isnan(got[4])


def test_round_significant_preserves_sign():
    values = torch.tensor([-1.23456e-7, -9.87654e12], dtype=torch.float64)
    got = _round_significant(values, 3)
    assert (got < 0).all()


def test_quantization_resolution_is_worst_near_a_mantissa_of_one():
    """The floor under any initializer improvement, stated precisely.

    Four significant digits is a *half-step* relative error of 5e-5 when the
    mantissa is near 9.999, but 5e-4 when it is near 1.000 — ten times worse,
    and equal to the solver's own 5e-4 convergence threshold rather than five
    times below it.
    """

    worst = torch.tensor([1.00049e-5], dtype=torch.float64)
    quantized = _round_significant(worst, 3)
    assert quantized.item() == pytest.approx(1.000e-5, rel=1e-12)
    worst_relative = abs(quantized.item() - worst.item()) / worst.item()
    assert 4.5e-4 < worst_relative < 5.0e-4

    best = torch.tensor([9.99949e-5], dtype=torch.float64)
    best_relative = abs(
        _round_significant(best, 3).item() - best.item()
    ) / best.item()
    assert best_relative < 5.1e-5

    # Empirically, over a decade of mantissas, the bound is 5e-4.
    generator = torch.Generator().manual_seed(17)
    values = 1.0 + 9.0 * torch.rand(20000, generator=generator, dtype=torch.float64)
    relative = ((_round_significant(values, 3) - values).abs() / values).max().item()
    assert relative < 5.0e-4


def test_quantize_columns_applies_the_right_format_per_column(model):
    with torch.no_grad():
        prediction = model(_labels(SOLAR))
    quantized = quantize_columns(**prediction.as_dict(), straight_through=False)

    # Temperature is fixed-point to 0.1 K, not significant digits.
    np.testing.assert_array_equal(
        quantized["temperature"].numpy(),
        np.round(prediction.temperature.numpy(), 1),
    )
    # Pressure keeps four significant digits.
    pressure = quantized["gas_pressure"].numpy()
    expected = np.array([float("%.3E" % v) for v in prediction.gas_pressure[0].numpy()])
    np.testing.assert_allclose(pressure[0], expected, rtol=1e-15)


def test_straight_through_forward_equals_the_quantized_value(model):
    with torch.no_grad():
        prediction = model(_labels(SOLAR))
    hard = quantize_columns(**prediction.as_dict(), straight_through=False)
    soft = quantize_columns(**prediction.as_dict(), straight_through=True)
    for field in OUTPUT_FIELDS:
        torch.testing.assert_close(soft[field], hard[field], rtol=0, atol=0)


def test_straight_through_backward_is_the_identity(model):
    prediction = model(_labels(SOLAR))
    quantized = quantize_prediction(prediction)
    quantized.gas_pressure.sum().backward(retain_graph=True)
    through = [p.grad.clone() for p in model.network.parameters()]

    model.zero_grad(set_to_none=True)
    prediction = model(_labels(SOLAR))
    prediction.gas_pressure.sum().backward()
    direct = [p.grad.clone() for p in model.network.parameters()]

    for a, b in zip(through, direct):
        torch.testing.assert_close(a, b)


def test_quantization_without_straight_through_blocks_gradient(model):
    prediction = model(_labels(SOLAR))
    quantized = quantize_prediction(prediction, straight_through=False)
    quantized.gas_pressure.sum().backward()
    norms = [float(p.grad.norm()) for p in model.network.parameters()]
    assert max(norms) == 0.0, "round() has zero derivative; that is the point"


def test_quantized_column_mass_stays_increasing(model):
    with torch.no_grad():
        prediction = model(_labels(SOLAR))
    quantized = quantize_prediction(prediction, straight_through=False)
    assert (quantized.column_mass.diff(dim=-1) > 0).all()


# --- the eight-label family --------------------------------------------------


def test_cno8_checkpoint_loads_and_predicts():
    """The label plumbing is generic; confirm it on the other released family."""

    path = default_checkpoint_path("cno8")
    if not path.is_file():
        pytest.skip("cno8 checkpoint is not installed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    module = DifferentiableInitializer(checkpoint)
    module.eval()
    assert len(module.family_fields) == 8

    labels = dict(
        SOLAR,
        carbon_enhancement=0.1,
        nitrogen_enhancement=0.2,
        oxygen_enhancement=0.1,
    )
    with torch.no_grad():
        prediction = module(_labels(labels))
    assert prediction.temperature.shape == (1, 80)
    assert (prediction.temperature > 0).all()


def test_cno8_matches_its_released_initializer():
    path = default_checkpoint_path("cno8")
    if not path.is_file():
        pytest.skip("cno8 checkpoint is not installed")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    module = DifferentiableInitializer(checkpoint, network_dtype=torch.float32)
    module.eval()

    labels = dict(
        SOLAR,
        effective_temperature=4800.0,
        log_surface_gravity=2.5,
        metallicity=-0.5,
        alpha_enhancement=0.3,
        microturbulence_km_s=1.8,
        carbon_enhancement=0.1,
        nitrogen_enhancement=0.2,
        oxygen_enhancement=0.1,
    )
    reference = load_atmosphere_initializer(checkpoint_path=path, device="cpu")
    expected = reference.predict(**labels)
    with torch.no_grad():
        got = module(_labels(labels))
    for field in ("column_mass", "temperature", "gas_pressure",
                  "electron_density", "rosseland_opacity"):
        np.testing.assert_allclose(
            getattr(got, field)[0].numpy(),
            np.asarray(expected[field], dtype=np.float64),
            rtol=1e-14,
        )
