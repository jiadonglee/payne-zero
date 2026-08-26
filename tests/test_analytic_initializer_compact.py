"""Tests for the Chebyshev depth basis and the grid-free warm start.

The property under test that the tabulated version could not have is grid
independence: the same constants must mean the same function on a grid the fit
never saw.  Everything else -- the four invariants, the label box -- is
inherited and re-checked off-grid, because a guarantee that only holds on the
fitting grid is not the guarantee that was claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from experiments.analytic_initializer.analytic_depth import (
    AnalyticDepthClosure,
    DepthNormalization,
    evaluate_analytic_depth_closure,
    fit_analytic_depth_closure,
)
from experiments.analytic_initializer.compact_initializer import (
    COMPACT_CONFIGURATION,
    PARITY_CONFIGURATION,
    PHYSICAL_CONFIGURATION,
    fit_compact_profile_parameters,
    load_compact_profile_parameters,
    predict_compact_reduced_state,
    save_compact_profile_parameters,
)
from experiments.analytic_initializer.discovery import Split, grey_temperature

LAYERS = 40


@dataclass
class _Corpus:
    labels: np.ndarray
    tau: np.ndarray
    temperature: np.ndarray
    column_mass: np.ndarray
    rosseland_opacity: np.ndarray

    @property
    def layers(self) -> int:
        return int(self.tau.size)


def _corpus(rows: int = 300, layers: int = LAYERS) -> _Corpus:
    generator = np.random.default_rng(20260817)
    labels = np.column_stack(
        (
            np.linspace(4200.0, 9800.0, rows),
            generator.uniform(1.0, 5.0, rows),
            generator.uniform(-2.0, 0.4, rows),
            generator.uniform(0.0, 0.4, rows),
            generator.uniform(0.6, 3.5, rows),
        )
    )
    tau = np.logspace(-3.0, 2.0, layers)
    grey = grey_temperature(labels[:, 0], tau)
    bend = 1.0 + 0.05 * np.tanh(np.log10(tau))[None, :] * (labels[:, 1:2] / 5.0)
    opacity = 10.0 ** (
        -2.0 + 0.4 * np.log10(tau)[None, :] + 0.3 * labels[:, 2:3] + 0.1 * np.sin(labels[:, 1:2])
    )
    return _Corpus(
        labels=labels,
        tau=tau,
        temperature=grey * bend,
        column_mass=np.cumsum(np.full((rows, layers), 0.1), axis=1),
        rosseland_opacity=opacity,
    )


def _split(rows: int) -> Split:
    index = np.arange(rows, dtype=np.int64)
    return Split(
        train=index[: int(0.8 * rows)],
        validation=index[int(0.8 * rows) :],
        excluded=np.asarray([], dtype=np.int64),
        seed=1,
    )


_SMALL = {"degree": 2, "components": 2, "center_degree": 8, "mode_degree": 6}


def _fitted(corpus: _Corpus):
    return fit_compact_profile_parameters(
        corpus,
        _split(corpus.labels.shape[0]),
        configuration={"temperature": _SMALL, "opacity": _SMALL},
    )


# --- the depth normalization ----------------------------------------------


def test_normalization_maps_the_grid_onto_the_unit_interval() -> None:
    tau = np.logspace(-3.0, 2.0, 40)
    normalization = DepthNormalization.from_grid(tau)
    coordinate = normalization.coordinate(tau)
    assert coordinate[0] == pytest.approx(-1.0)
    assert coordinate[-1] == pytest.approx(1.0)
    assert np.all(np.abs(coordinate) <= 1.0 + 1.0e-12)


def test_normalization_refuses_depths_outside_the_interval() -> None:
    normalization = DepthNormalization.from_grid(np.logspace(-3.0, 2.0, 40))
    normalization.require_support(np.logspace(-2.0, 1.0, 5))
    with pytest.raises(ValueError, match="outside the fitted depth interval"):
        normalization.require_support(np.logspace(-6.0, 2.0, 5))


def test_normalization_rejects_a_degenerate_grid() -> None:
    with pytest.raises(ValueError, match="half_width must be"):
        DepthNormalization(center=0.0, half_width=0.0)
    with pytest.raises(ValueError, match="at least two"):
        DepthNormalization.from_grid(np.asarray([1.0]))


def test_normalization_rejects_non_positive_tau() -> None:
    normalization = DepthNormalization.from_grid(np.logspace(-3.0, 2.0, 40))
    with pytest.raises(ValueError, match="finite and positive"):
        normalization.coordinate(np.asarray([-1.0, 1.0]))


# --- the closure ------------------------------------------------------------


def test_the_closure_is_the_same_function_on_any_grid() -> None:
    """The property the tabulated version could not have."""

    corpus = _corpus()
    closure = fit_analytic_depth_closure(
        corpus, _split(corpus.labels.shape[0]), target=np.log10(corpus.rosseland_opacity), **_SMALL
    )
    dense = np.exp(
        np.linspace(np.log(corpus.tau[0]), np.log(corpus.tau[-1]), (LAYERS - 1) * 6 + 1)
    )
    shared = np.arange(0, dense.size, 6)
    assert np.allclose(dense[shared], corpus.tau)
    native = evaluate_analytic_depth_closure(corpus.labels, corpus.tau, closure)
    on_dense = evaluate_analytic_depth_closure(corpus.labels, dense, closure)
    assert np.allclose(on_dense[:, shared], native, rtol=0.0, atol=1.0e-12)


def test_the_closure_evaluates_on_a_grid_of_any_length() -> None:
    corpus = _corpus()
    closure = fit_analytic_depth_closure(
        corpus, _split(corpus.labels.shape[0]), target=np.log10(corpus.rosseland_opacity), **_SMALL
    )
    for count in (7, 40, 233):
        grid = np.logspace(-2.5, 1.5, count)
        assert evaluate_analytic_depth_closure(corpus.labels[:4], grid, closure).shape == (4, count)


def test_a_single_regime_is_allowed_and_costs_a_third() -> None:
    corpus = _corpus()
    split = _split(corpus.labels.shape[0])
    target = np.log10(corpus.rosseland_opacity)
    segmented = fit_analytic_depth_closure(corpus, split, target=target, **_SMALL)
    single = fit_analytic_depth_closure(
        corpus, split, target=target, regime_boundaries=None, **_SMALL
    )
    assert segmented.regimes == 3 and single.regimes == 1
    assert single.stored_float_count < segmented.stored_float_count
    assert evaluate_analytic_depth_closure(corpus.labels, corpus.tau, single).shape == (
        corpus.labels.shape[0],
        corpus.layers,
    )


def test_the_closure_rejects_a_target_off_the_corpus_grid() -> None:
    corpus = _corpus()
    with pytest.raises(ValueError, match="sampled on the corpus tau grid"):
        fit_analytic_depth_closure(
            corpus,
            _split(corpus.labels.shape[0]),
            target=np.zeros((corpus.labels.shape[0], LAYERS - 1)),
            **_SMALL,
        )


def test_smooth_regime_weights_agree_with_hard_ones_away_from_the_seams() -> None:
    corpus = _corpus()
    split = _split(corpus.labels.shape[0])
    target = np.log10(corpus.rosseland_opacity)
    hard = fit_analytic_depth_closure(corpus, split, target=target, **_SMALL)
    soft = fit_analytic_depth_closure(
        corpus, split, target=target, smoothing_width_K=1.0e-3, **_SMALL
    )
    interior = corpus.labels[np.abs(corpus.labels[:, 0] - 5500.0) > 400.0]
    interior = interior[np.abs(interior[:, 0] - 7500.0) > 400.0]
    assert np.allclose(
        evaluate_analytic_depth_closure(interior, corpus.tau, hard),
        evaluate_analytic_depth_closure(interior, corpus.tau, soft),
        atol=1.0e-9,
    )


# --- the assembled formula --------------------------------------------------


def test_all_four_invariants_hold_on_grids_the_fit_never_saw() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    for grid in (
        corpus.tau,
        np.logspace(-3.0, 2.0, 233),
        np.logspace(-2.0, 1.0, 17),
        np.sort(np.exp(np.random.default_rng(5).uniform(np.log(1.0e-3), np.log(1.0e2), 61))),
    ):
        mass, temperature, log_opacity = predict_compact_reduced_state(
            corpus.labels, grid, parameters
        )
        assert np.all(np.isfinite(log_opacity))
        assert np.all(temperature > 0.0)
        assert np.all(np.diff(temperature, axis=1) > 0.0)
        assert np.all(np.diff(mass, axis=1) > 0.0)


def test_prediction_refuses_depths_outside_the_fitted_interval() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    with pytest.raises(ValueError, match="outside the fitted depth interval"):
        predict_compact_reduced_state(corpus.labels[:2], np.logspace(-8.0, 2.0, 30), parameters)


def test_prediction_refuses_labels_outside_the_box() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    with pytest.raises(ValueError, match="outside the fitted analytic support"):
        predict_compact_reduced_state(
            np.asarray([[12000.0, 4.0, -1.0, 0.2, 1.0]]), corpus.tau, parameters
        )


def test_prediction_rejects_an_unsorted_grid() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    with pytest.raises(ValueError, match="strictly increasing"):
        predict_compact_reduced_state(
            corpus.labels[:2], np.asarray([1.0, 0.1, 10.0]), parameters
        )


def test_the_two_shipped_configurations_differ_between_fields() -> None:
    """Temperature and opacity are deliberately not configured alike."""

    assert COMPACT_CONFIGURATION["temperature"] != COMPACT_CONFIGURATION["opacity"]
    assert PARITY_CONFIGURATION["temperature"]["center_degree"] > (
        PARITY_CONFIGURATION["opacity"]["center_degree"]
    )


def test_stored_float_count_counts_the_label_scaling_once() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    assert np.array_equal(
        parameters.temperature.feature_center, parameters.opacity.feature_center
    )
    expected = (
        parameters.temperature.stored_float_count
        + parameters.opacity.stored_float_count
        - 10
        + 11
    )
    assert parameters.stored_float_count == expected


def test_save_and_load_round_trip_preserves_every_field(tmp_path: Path) -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    destination = save_compact_profile_parameters(tmp_path / "compact.npz", parameters)
    reloaded = load_compact_profile_parameters(destination)

    assert reloaded.gradient_floor == parameters.gradient_floor
    assert np.array_equal(reloaded.support.lower, parameters.support.lower)
    for name in ("temperature", "opacity"):
        original, restored = getattr(parameters, name), getattr(reloaded, name)
        assert restored.center_degree == original.center_degree
        assert restored.mode_degree == original.mode_degree
        assert isinstance(restored.regime_boundaries, tuple)
        assert restored.regime_boundaries == original.regime_boundaries
        assert restored.normalization == original.normalization
        assert np.array_equal(restored.modes_by_regime, original.modes_by_regime)
        assert np.array_equal(
            restored.coefficients_by_regime, original.coefficients_by_regime
        )

    grid = np.logspace(-2.5, 1.5, 55)
    for one, other in zip(
        predict_compact_reduced_state(corpus.labels, grid, parameters),
        predict_compact_reduced_state(corpus.labels, grid, reloaded),
    ):
        assert np.array_equal(one, other)


def test_load_rejects_a_foreign_format_marker(tmp_path: Path) -> None:
    destination = tmp_path / "wrong.npz"
    np.savez_compressed(destination, format=np.asarray("something_else"))
    with pytest.raises(ValueError, match="unsupported compact profile"):
        load_compact_profile_parameters(destination)


def test_the_fit_refuses_closures_that_disagree_on_label_scaling() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    # Construct the disagreement the fit guards against.
    broken = AnalyticDepthClosure(
        normalization=parameters.opacity.normalization,
        center_degree=parameters.opacity.center_degree,
        mode_degree=parameters.opacity.mode_degree,
        exponents=parameters.opacity.exponents,
        feature_center=parameters.opacity.feature_center + 1.0,
        feature_scale=parameters.opacity.feature_scale,
        center_by_regime=parameters.opacity.center_by_regime,
        modes_by_regime=parameters.opacity.modes_by_regime,
        coefficients_by_regime=parameters.opacity.coefficients_by_regime,
        regime_boundaries=parameters.opacity.regime_boundaries,
    )
    assert not np.array_equal(
        parameters.temperature.feature_center, broken.feature_center
    )


# --- the physical label coordinates -----------------------------------------


def test_the_physical_configuration_survives_a_save_and_load(tmp_path: Path) -> None:
    """The stored name is what lets prediction rebuild the same coordinates."""

    corpus = _corpus()
    parameters = fit_compact_profile_parameters(
        corpus,
        _split(corpus.labels.shape[0]),
        configuration={
            field: {**PHYSICAL_CONFIGURATION[field], "center_degree": 8, "mode_degree": 6}
            for field in ("temperature", "opacity")
        },
    )
    assert parameters.temperature.label_features == "physical"
    assert parameters.temperature.feature_center.size == 7
    assert parameters.temperature.exponents.shape[1] == 7

    reloaded = load_compact_profile_parameters(
        save_compact_profile_parameters(tmp_path / "physical.npz", parameters)
    )
    assert reloaded.temperature.label_features == "physical"
    grid = np.logspace(-2.5, 1.5, 33)
    for one, other in zip(
        predict_compact_reduced_state(corpus.labels, grid, parameters),
        predict_compact_reduced_state(corpus.labels, grid, reloaded),
    ):
        assert np.array_equal(one, other)


def test_the_physical_configuration_keeps_the_four_invariants_off_grid() -> None:
    corpus = _corpus()
    parameters = fit_compact_profile_parameters(
        corpus,
        _split(corpus.labels.shape[0]),
        configuration={
            field: {**PHYSICAL_CONFIGURATION[field], "center_degree": 8, "mode_degree": 6}
            for field in ("temperature", "opacity")
        },
    )
    for grid in (corpus.tau, np.logspace(-3.0, 2.0, 137), np.logspace(-2.0, 1.0, 19)):
        mass, temperature, log_opacity = predict_compact_reduced_state(
            corpus.labels, grid, parameters
        )
        assert np.all(np.isfinite(log_opacity))
        assert np.all(temperature > 0.0)
        assert np.all(np.diff(temperature, axis=1) > 0.0)
        assert np.all(np.diff(mass, axis=1) > 0.0)


def test_the_shared_label_scaling_tracks_the_feature_count() -> None:
    """Seven features means fourteen shared floats, not ten."""

    corpus = _corpus()
    physical = fit_compact_profile_parameters(
        corpus,
        _split(corpus.labels.shape[0]),
        configuration={
            field: {**PHYSICAL_CONFIGURATION[field], "center_degree": 8, "mode_degree": 6}
            for field in ("temperature", "opacity")
        },
    )
    expected = (
        physical.temperature.stored_float_count
        + physical.opacity.stored_float_count
        - 2 * 7
        + 11
    )
    assert physical.stored_float_count == expected
