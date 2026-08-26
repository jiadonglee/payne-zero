"""Tests for the monotone temperature representation and the support guard.

These cover the two invariants H2 did not hold -- a strictly increasing
temperature, and a bounded label domain -- plus the closure refactor that let
a target live on an axis other than the tau grid.  Nothing here imports the
solver, so the module stays runnable in the lightweight environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from experiments.analytic_initializer.discovery import Split, grey_temperature
from experiments.analytic_initializer.monotone_initializer import (
    MonotoneProfileParameters,
    fit_monotone_profile_parameters,
    load_monotone_profile_parameters,
    predict_monotone_reduced_state,
    save_monotone_profile_parameters,
)
from experiments.analytic_initializer.monotone_temperature import (
    GRADIENT_FLOOR,
    LabelSupport,
    anchor_target,
    fit_label_support,
    log_increment_target,
    project_to_monotone,
    rebuild_temperature,
    require_label_support,
)
from experiments.analytic_initializer.profile_closure import (
    evaluate_profile_closure,
    fit_profile_closure,
    predict_profile_closure,
)
from experiments.analytic_initializer.profile_initializer import (
    AnalyticProfileParameters,
)

LAYERS = 16


@dataclass
class _Corpus:
    """The subset of the corpus interface the closure fit actually reads."""

    labels: np.ndarray
    tau: np.ndarray
    temperature: np.ndarray
    column_mass: np.ndarray
    rosseland_opacity: np.ndarray

    @property
    def layers(self) -> int:
        return int(self.tau.size)


def _corpus(rows: int = 240, layers: int = LAYERS) -> _Corpus:
    """A synthetic corpus spanning all three temperature regimes."""

    generator = np.random.default_rng(20260817)
    # Teff is spread across the 5500/7500 K seams so every regime is populated.
    labels = np.column_stack(
        (
            np.linspace(4200.0, 9800.0, rows),
            generator.uniform(1.0, 5.0, rows),
            generator.uniform(-2.0, 0.4, rows),
            generator.uniform(0.0, 0.4, rows),
            generator.uniform(0.6, 3.5, rows),
        )
    )
    # The grid stops at 1e-2 rather than 1e-6 so that every increment of the
    # grey profile clears the default floor.  Sub-floor increments are real --
    # the production grid has them at both ends -- but they belong in the
    # tests that are about the floor, not in the fixture every test shares.
    tau = np.logspace(-2.0, 2.0, layers)
    grey = grey_temperature(labels[:, 0], tau)
    # A smooth, strictly increasing perturbation of the grey profile.
    bend = 1.0 + 0.05 * np.tanh(np.log10(tau))[None, :] * (labels[:, 1:2] / 5.0)
    temperature = grey * bend
    opacity = 10.0 ** (
        -2.0
        + 0.4 * np.log10(tau)[None, :]
        + 0.3 * labels[:, 2:3]
        + 0.1 * np.sin(labels[:, 1:2])
    )
    mass = np.cumsum(np.full((rows, layers), 0.1), axis=1) * (1.0 + labels[:, 1:2] / 10.0)
    return _Corpus(
        labels=labels,
        tau=tau,
        temperature=temperature,
        column_mass=mass,
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


def _fitted(corpus: _Corpus) -> MonotoneProfileParameters:
    return fit_monotone_profile_parameters(
        corpus, _split(corpus.labels.shape[0]), degree=2, components=3
    )


# --- the transform ---------------------------------------------------------


def test_transform_round_trip_is_exact_above_the_floor() -> None:
    corpus = _corpus()
    rebuilt = rebuild_temperature(
        corpus.tau,
        corpus.labels[:, 0],
        anchor_target(corpus.tau, corpus.temperature, corpus.labels[:, 0]),
        log_increment_target(corpus.tau, corpus.temperature),
    )
    assert np.allclose(rebuilt, corpus.temperature, rtol=1.0e-12, atol=0.0)


def test_projection_makes_an_adversarial_profile_strictly_increasing() -> None:
    corpus = _corpus(rows=40)
    generator = np.random.default_rng(7)
    # Multiplicative noise large enough to fold the profile back on itself.
    broken = corpus.temperature * (1.0 + generator.normal(0.0, 0.3, corpus.temperature.shape))
    broken = np.abs(broken) + 1.0
    assert not np.all(np.diff(broken, axis=1) > 0.0)
    fixed = project_to_monotone(corpus.tau, corpus.labels[:, 0], broken)
    assert np.all(np.diff(fixed, axis=1) > 0.0)
    assert np.all(fixed > 0.0)


def test_projection_leaves_an_already_monotone_profile_alone() -> None:
    corpus = _corpus(rows=40)
    fixed = project_to_monotone(corpus.tau, corpus.labels[:, 0], corpus.temperature)
    assert np.allclose(fixed, corpus.temperature, rtol=1.0e-12, atol=0.0)


def test_only_the_floor_is_lossy_not_the_transform() -> None:
    """A near-isothermal top round trips exactly once the floor is lowered."""

    corpus = _corpus(rows=20)
    tau = np.logspace(-6.0, 2.0, corpus.layers)
    flat = grey_temperature(corpus.labels[:, 0], tau)
    gradient = np.diff(np.log(flat), axis=1) / np.diff(np.log(tau))
    assert gradient.min() < GRADIENT_FLOOR  # the fixture really is sub-floor

    lifted = project_to_monotone(tau, corpus.labels[:, 0], flat)
    assert not np.allclose(lifted, flat, rtol=1.0e-12, atol=0.0)

    exact = project_to_monotone(tau, corpus.labels[:, 0], flat, floor=1.0e-30)
    assert np.allclose(exact, flat, rtol=1.0e-12, atol=0.0)


def test_projection_lifts_a_flat_step_to_the_floor() -> None:
    tau = np.logspace(-3.0, 1.0, 5)
    effective = np.asarray([5000.0])
    profile = np.asarray([[3000.0, 3000.0, 4000.0, 4000.0, 5000.0]])
    fixed = project_to_monotone(tau, effective, profile, floor=1.0e-3)
    gradient = np.diff(np.log(fixed), axis=1) / np.diff(np.log(tau))
    assert np.all(gradient > 0.0)
    # The two flat steps land exactly on the floor as a gradient, and the real
    # ones are untouched.
    assert gradient[0, 0] == pytest.approx(1.0e-3, rel=1.0e-9)
    assert gradient[0, 2] == pytest.approx(1.0e-3, rel=1.0e-9)
    assert gradient[0, 1] == pytest.approx(
        np.log(4000.0 / 3000.0) / np.log(tau[2] / tau[1]), rel=1.0e-6
    )


def test_the_floor_means_the_same_thing_on_a_finer_grid() -> None:
    """The defect that a per-interval floor would have: grid dependence."""

    coarse = np.logspace(-4.0, 2.0, 20)
    fine = np.logspace(-4.0, 2.0, 20 * 8 - 7)  # same interval, shares every depth
    effective = np.asarray([6000.0])
    # A profile flat enough that the floor is what sets its slope everywhere.
    flat = np.full((1, coarse.size), 5000.0)
    flat_fine = np.full((1, fine.size), 5000.0)
    on_coarse = project_to_monotone(coarse, effective, flat, floor=1.0e-3)
    on_fine = project_to_monotone(fine, effective, flat_fine, floor=1.0e-3)
    assert np.allclose(on_fine[:, ::8], on_coarse, rtol=1.0e-12, atol=0.0)


def test_increment_target_rejects_a_non_positive_temperature() -> None:
    tau = np.logspace(-3.0, 1.0, 4)
    with pytest.raises(ValueError, match="finite and positive"):
        log_increment_target(tau, np.asarray([[100.0, 0.0, 200.0, 300.0]]))


def test_increment_target_rejects_a_non_positive_floor() -> None:
    tau = np.logspace(-3.0, 1.0, 4)
    with pytest.raises(ValueError, match="floor must be"):
        log_increment_target(tau, np.asarray([[1.0, 2.0, 3.0, 4.0]]), floor=0.0)


def test_rebuild_rejects_a_mismatched_increment_count() -> None:
    tau = np.logspace(-3.0, 1.0, 5)
    with pytest.raises(ValueError, match="len\\(tau\\) - 1"):
        rebuild_temperature(tau, np.asarray([5000.0]), np.zeros((1, 1)), np.zeros((1, 7)))


# --- the support guard -----------------------------------------------------


def test_support_box_accepts_inside_and_names_the_offending_label() -> None:
    corpus = _corpus()
    support = fit_label_support(corpus.labels)
    require_label_support(corpus.labels, support)
    outside = corpus.labels[0].copy()
    outside[0] = 12000.0
    with pytest.raises(ValueError, match="effective_temperature"):
        require_label_support(outside, support)


def test_support_box_reports_every_violated_label_at_once() -> None:
    corpus = _corpus()
    support = fit_label_support(corpus.labels)
    outside = corpus.labels[0].copy()
    outside[0] = 12000.0
    outside[2] = -9.0
    with pytest.raises(ValueError) as error:
        require_label_support(outside, support)
    assert "effective_temperature" in str(error.value)
    assert "metallicity" in str(error.value)


def test_support_box_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="must not fall below"):
        LabelSupport(lower=np.ones(5), upper=np.zeros(5))


def test_support_box_rejects_a_wrong_label_count() -> None:
    with pytest.raises(ValueError, match="shape"):
        LabelSupport(lower=np.zeros(4), upper=np.ones(4))


# --- the assembled formula -------------------------------------------------


def test_all_four_invariants_hold_for_every_row() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    mass, temperature, log_opacity = predict_monotone_reduced_state(
        corpus.labels, corpus.tau, parameters
    )
    assert np.all(np.isfinite(log_opacity))
    assert np.all(temperature > 0.0)
    assert np.all(np.diff(temperature, axis=1) > 0.0)
    assert np.all(np.diff(mass, axis=1) > 0.0)


def test_prediction_refuses_labels_outside_the_box_but_can_be_asked_anyway() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    outside = np.asarray([[12000.0, 4.0, -1.0, 0.2, 1.0]])
    with pytest.raises(ValueError, match="outside the fitted analytic support"):
        predict_monotone_reduced_state(outside, corpus.tau, parameters)
    # The escape hatch exists so the probe can quantify what the guard prevents.
    _, temperature, _ = predict_monotone_reduced_state(
        outside, corpus.tau, parameters, check_support=False
    )
    assert temperature.shape == (1, corpus.layers)


def test_prediction_refuses_a_grid_of_the_wrong_length() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    with pytest.raises(ValueError, match="does not match the fitted profile basis"):
        predict_monotone_reduced_state(
            corpus.labels[:2], np.logspace(-6.0, 2.0, LAYERS + 4), parameters
        )


def test_adopting_h2_constants_reproduces_them_exactly() -> None:
    corpus = _corpus()
    fitted = _fitted(corpus)
    analytic = AnalyticProfileParameters(
        temperature=fitted.temperature, opacity=fitted.opacity
    )
    adopted = MonotoneProfileParameters.from_analytic(
        analytic, fit_label_support(corpus.labels)
    )
    assert adopted.temperature is fitted.temperature
    assert adopted.opacity is fitted.opacity
    assert adopted.gradient_floor == GRADIENT_FLOOR


def test_save_and_load_round_trip_preserves_every_field(tmp_path: Path) -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    destination = save_monotone_profile_parameters(
        tmp_path / "monotone.npz", parameters
    )
    reloaded = load_monotone_profile_parameters(destination)

    assert reloaded.gradient_floor == parameters.gradient_floor
    assert np.array_equal(reloaded.support.lower, parameters.support.lower)
    assert np.array_equal(reloaded.support.upper, parameters.support.upper)
    for name in ("temperature", "opacity"):
        original = getattr(parameters, name)
        restored = getattr(reloaded, name)
        assert restored.degree == original.degree
        assert restored.components == original.components
        # The tuple type matters: a numpy array here silently changes how the
        # regime comparison broadcasts.
        assert isinstance(restored.regime_boundaries, tuple)
        assert restored.regime_boundaries == original.regime_boundaries
        assert restored.smoothing_width_K == original.smoothing_width_K
        assert np.array_equal(
            restored.coefficients_by_regime, original.coefficients_by_regime
        )
        assert np.array_equal(restored.basis_by_regime, original.basis_by_regime)

    before = predict_monotone_reduced_state(corpus.labels, corpus.tau, parameters)
    after = predict_monotone_reduced_state(corpus.labels, corpus.tau, reloaded)
    for one, other in zip(before, after):
        assert np.array_equal(one, other)


def test_load_rejects_a_foreign_format_marker(tmp_path: Path) -> None:
    destination = tmp_path / "wrong.npz"
    np.savez_compressed(destination, format=np.asarray("something_else"))
    with pytest.raises(ValueError, match="unsupported monotone profile"):
        load_monotone_profile_parameters(destination)


def test_stored_float_count_matches_a_hand_count() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    expected = 2 * 5 + 1
    for closure in (parameters.temperature, parameters.opacity):
        expected += (
            closure.feature_center.size
            + closure.feature_scale.size
            + closure.target_center_by_regime.size
            + closure.basis_by_regime.size
            + closure.coefficients_by_regime.size
        )
    assert parameters.stored_float_count == expected


# --- the closure refactor --------------------------------------------------


def test_closure_fits_a_target_that_is_not_on_the_tau_grid() -> None:
    """The increment ablation needs a target of depth len(tau) - 1."""

    corpus = _corpus()
    target = log_increment_target(corpus.tau, corpus.temperature)
    closure = fit_profile_closure(
        corpus, _split(corpus.labels.shape[0]), target=target, degree=2, components=3
    )
    predicted = evaluate_profile_closure(corpus.labels, closure)
    assert predicted.shape == (corpus.labels.shape[0], corpus.layers - 1)


def test_closure_fits_a_single_scalar_per_star() -> None:
    corpus = _corpus()
    target = anchor_target(corpus.tau, corpus.temperature, corpus.labels[:, 0])
    closure = fit_profile_closure(
        corpus, _split(corpus.labels.shape[0]), target=target, degree=2, components=1
    )
    assert evaluate_profile_closure(corpus.labels, closure).shape == (
        corpus.labels.shape[0],
        1,
    )


def test_tau_grid_wrapper_still_guards_the_length() -> None:
    corpus = _corpus()
    parameters = _fitted(corpus)
    assert np.array_equal(
        predict_profile_closure(corpus.labels, corpus.tau, parameters.opacity),
        evaluate_profile_closure(corpus.labels, parameters.opacity),
    )
    with pytest.raises(ValueError, match="tau length does not match"):
        predict_profile_closure(
            corpus.labels, np.logspace(-6.0, 2.0, LAYERS + 1), parameters.opacity
        )


def test_closure_rejects_a_target_with_the_wrong_row_count() -> None:
    corpus = _corpus()
    with pytest.raises(ValueError, match="one row per star"):
        fit_profile_closure(
            corpus,
            _split(corpus.labels.shape[0]),
            target=np.zeros((corpus.labels.shape[0] - 1, LAYERS)),
            degree=2,
            components=3,
        )
