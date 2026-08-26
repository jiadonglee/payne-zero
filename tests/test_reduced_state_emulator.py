"""Tests for the learned reduced-state initializer.

The monotonicity property is the one that must hold for *arbitrary* network
output, not just trained output: ``reduced_state.reconstruct._seed_atmosphere``
raises on a non-increasing column mass, so a single violated star aborts a
sweep. The smoke run measured 39/60 violations for the direct parameterization,
which is exactly why the guarantee is tested against adversarial raw values
rather than against a fitted model.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from reduced_state.emulator import (  # noqa: E402
    LAYER_COUNT,
    ReducedStateEmulator,
    Standardization,
    decode_column_mass,
    decode_temperature,
    fit_standardization,
    label_features,
    load_checkpoint,
    predict_reduced_state,
    save_checkpoint,
)


def _standardization(seed: int = 0) -> Standardization:
    generator = np.random.default_rng(seed)
    features = generator.normal(size=(256, 5))
    # A plausible monotone log column mass and a monotone-ish log temperature.
    log_column_mass = np.linspace(-4.0, 1.5, LAYER_COUNT)[None, :] + generator.normal(
        scale=0.02, size=(256, LAYER_COUNT)
    )
    log_temperature = np.linspace(3.5, 4.0, LAYER_COUNT)[None, :] + generator.normal(
        scale=0.01, size=(256, LAYER_COUNT)
    )
    return fit_standardization(features, log_column_mass, log_temperature)


def test_label_features_uses_theta_not_temperature():
    labels = np.array([[5040.0, 4.44, 0.0, 0.0, 1.0], [10080.0, 2.0, -1.0, 0.3, 2.0]])
    features = label_features(labels)
    assert features[0, 0] == pytest.approx(1.0)
    assert features[1, 0] == pytest.approx(0.5)
    # The remaining four labels pass through untouched.
    np.testing.assert_allclose(features[:, 1:], labels[:, 1:])


@pytest.mark.parametrize("scale", [1.0, 50.0, 500.0])
def test_monotone_decode_is_increasing_for_adversarial_raw(scale):
    """Monotonicity must be structural, not learned."""

    standardization = _standardization()
    generator = torch.Generator().manual_seed(7)
    raw = (
        torch.randn(64, LAYER_COUNT, generator=generator, dtype=torch.float64) * scale
    )
    log_column_mass = decode_column_mass(raw, standardization, monotone=True)
    differences = torch.diff(log_column_mass, dim=1)
    assert torch.all(differences > 0.0), float(differences.min())
    assert torch.isfinite(log_column_mass).all()


def test_direct_decode_can_violate_monotonicity():
    """The contrast the ablation measures; guards the test above from vacuity."""

    standardization = _standardization()
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(64, LAYER_COUNT, generator=generator, dtype=torch.float64) * 5.0
    log_column_mass = decode_column_mass(raw, standardization, monotone=False)
    assert not torch.all(torch.diff(log_column_mass, dim=1) > 0.0)


def test_monotone_decode_recovers_the_mean_profile():
    """Zero raw input should land near the train-set mean, not far from it.

    The decode offsets the softplus by the inverse-softplus of the mean
    profile's own increments, so an untrained network starts at a physically
    sensible profile instead of having to learn the depth scale from zero.
    """

    standardization = _standardization()
    raw = torch.zeros(1, LAYER_COUNT, dtype=torch.float64)
    decoded = decode_column_mass(raw, standardization, monotone=True).numpy()[0]
    expected = standardization.log_column_mass_mean
    # Anchor is exact; the walk accumulates only the INCREMENT_FLOOR per step.
    assert decoded[0] == pytest.approx(expected[0])
    np.testing.assert_allclose(decoded, expected, atol=1.0e-3)


def test_decode_temperature_is_the_inverse_of_standardizing():
    standardization = _standardization()
    target = np.linspace(3.5, 4.0, LAYER_COUNT)[None, :]
    standardized = (
        target - standardization.log_temperature_mean
    ) / standardization.log_temperature_std
    decoded = decode_temperature(
        torch.as_tensor(standardized, dtype=torch.float64), standardization
    ).numpy()
    np.testing.assert_allclose(decoded, target, rtol=1.0e-12)


def test_predict_reduced_state_returns_physical_positive_profiles():
    standardization = _standardization()
    model = ReducedStateEmulator(width=32, depth=2, monotone=True)
    labels = np.array([[5777.0, 4.44, 0.0, 0.0, 1.0], [4000.0, 1.5, -2.0, 0.4, 2.0]])
    column_mass, temperature = predict_reduced_state(model, standardization, labels)
    assert column_mass.shape == (2, LAYER_COUNT)
    assert temperature.shape == (2, LAYER_COUNT)
    assert (column_mass > 0.0).all()
    assert (temperature > 0.0).all()
    assert (np.diff(column_mass, axis=1) > 0.0).all()


def test_checkpoint_round_trip_is_exact(tmp_path):
    standardization = _standardization()
    model = ReducedStateEmulator(width=32, depth=2, monotone=True)
    labels = np.array([[5777.0, 4.44, 0.0, 0.0, 1.0]])
    before = predict_reduced_state(model, standardization, labels)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, standardization, {"seed": 1})
    restored_model, restored_standardization, meta = load_checkpoint(path)

    assert meta["seed"] == 1
    assert restored_model.monotone is True
    after = predict_reduced_state(restored_model, restored_standardization, labels)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def test_model_runs_in_float64():
    """float32 batched matmul is not reduction-order stable at the 5e-4 scale
    the deck quantization operates on (progress Sec 4.1)."""

    model = ReducedStateEmulator(width=16, depth=2)
    for parameter in model.parameters():
        assert parameter.dtype == torch.float64


def test_batch_invariance():
    """A star's prediction must not depend on its batch companions."""

    standardization = _standardization()
    model = ReducedStateEmulator(width=32, depth=2, monotone=True)
    labels = np.array(
        [
            [5777.0, 4.44, 0.0, 0.0, 1.0],
            [4000.0, 1.5, -2.0, 0.4, 2.0],
            [9000.0, 4.0, -0.5, 0.1, 1.5],
        ]
    )
    batched = predict_reduced_state(model, standardization, labels)
    single = predict_reduced_state(model, standardization, labels[:1])
    np.testing.assert_allclose(batched[0][0], single[0][0], rtol=1.0e-14)
    np.testing.assert_allclose(batched[1][0], single[1][0], rtol=1.0e-14)
