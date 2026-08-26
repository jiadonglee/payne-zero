"""Tests for the shared deep-band diagnostics and the smooth regime switch."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from experiments.analytic_initializer.candidates import (
    temperature_regime_weights,
    temperature_regimes,
)
from experiments.analytic_initializer.deep_diagnostics import (
    convective_diagnostics,
    deep_window,
    error_bands,
)
from experiments.analytic_initializer.run_arm_comparison import _exact_mcnemar


@dataclass
class _Corpus:
    """The subset of the corpus interface the diagnostics actually read."""

    labels: np.ndarray
    temperature: np.ndarray
    column_mass: np.ndarray
    gas_pressure: np.ndarray
    rosseland_opacity: np.ndarray

    @property
    def layers(self) -> int:
        return int(self.temperature.shape[1])


def _radiative_corpus(layers: int = 80) -> _Corpus:
    """A single star with a smooth, everywhere-radiative stratification."""

    tau = np.logspace(-5.0, 2.0, layers)
    temperature = 5000.0 * (0.75 * (tau + 2.0 / 3.0)) ** 0.25
    # A shallow dlnT/dlnP keeps the true gradient well under the radiative one
    # only if opacity is large; here opacity is tiny so grad_rad stays small.
    pressure = 1.0e3 * tau
    return _Corpus(
        labels=np.array([[5000.0, 4.5, 0.0, 0.0, 1.0]]),
        temperature=temperature[None, :],
        column_mass=(tau / 0.34)[None, :],
        gas_pressure=pressure[None, :],
        rosseland_opacity=np.full((1, layers), 1.0e-6),
    )


def test_deep_window_is_the_production_band() -> None:
    assert deep_window(80) == (39, 75)


def test_deep_window_matches_the_production_convergence_stop() -> None:
    # The band is only meaningful if it is the one the solver actually gates on.
    # The solver package is imported lazily: these probes are meant to stay
    # runnable in a lightweight environment that cannot import it, which is the
    # same reason ``no_emulator_bridge`` defers its own solver imports.
    convergence = pytest.importorskip("payne_zero_atmosphere.convergence")

    layers = 80
    start, _stop = deep_window(layers)
    before = np.ones(layers)

    outside = before.copy()
    outside[start - 1] = 2.0
    assert convergence.deep_layer_relative_temperature_change(before, outside) == 0.0

    inside = before.copy()
    inside[start] = 2.0
    assert convergence.deep_layer_relative_temperature_change(before, inside) > 0.0


def test_deep_window_falls_back_to_every_layer_on_small_grids() -> None:
    assert deep_window(20) == (0, 20)


def test_error_bands_attribute_a_deep_only_error_to_the_deep_band() -> None:
    corpus = _radiative_corpus()
    start, stop = deep_window(corpus.layers)
    temperature = corpus.temperature.copy()
    temperature[:, stop - 1] *= 10.0 ** 0.2

    bands = error_bands(
        corpus,
        np.array([0]),
        mass=corpus.column_mass,
        temperature=temperature,
        log_opacity=np.log10(corpus.rosseland_opacity),
    )
    assert bands["temperature_surface"][0] == pytest.approx(0.0, abs=1.0e-12)
    assert bands["temperature_deep"][0] == pytest.approx(0.2, abs=1.0e-9)
    assert bands["temperature_deep_argmax_layer"][0] == stop - 1


def test_convective_diagnostics_report_no_onset_for_a_radiative_star() -> None:
    corpus = _radiative_corpus()
    subadiabatic, onset = convective_diagnostics(corpus, np.array([0]))
    assert not subadiabatic.any()
    assert onset[0] == -1


def test_convective_diagnostics_find_an_onset_when_opacity_is_large() -> None:
    # Raising opacity raises grad_rad without touching the profile, which is
    # exactly the Schwarzschild statement the diagnostic is meant to encode.
    corpus = _radiative_corpus()
    corpus.rosseland_opacity[:] = 1.0e4
    subadiabatic, onset = convective_diagnostics(corpus, np.array([0]))
    assert subadiabatic.any()
    assert 0 <= onset[0] < corpus.layers


def _labels(effective_temperature: np.ndarray) -> np.ndarray:
    rows = np.zeros((effective_temperature.size, 5), dtype=np.float64)
    rows[:, 0] = effective_temperature
    rows[:, 1] = 4.0
    return rows


def test_regime_weights_sum_to_one() -> None:
    labels = _labels(np.linspace(4000.0, 10500.0, 41))
    weights = temperature_regime_weights(labels, width_K=250.0)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert np.all(weights >= 0.0)


def test_narrow_regime_weights_reproduce_the_hard_assignment() -> None:
    # The smooth switch must be a strict generalization of the hard one, or a
    # smooth-versus-hard ablation would confound the seam with a refit.  The
    # grid deliberately misses the seams themselves; those are covered below.
    labels = _labels(np.linspace(4137.0, 10411.0, 64))
    assert not np.isin(labels[:, 0], (5500.0, 7500.0)).any()
    weights = temperature_regime_weights(labels, width_K=1.0e-3)
    hard = temperature_regimes(labels)
    np.testing.assert_allclose(weights, np.eye(3)[hard], atol=1.0e-9)


def test_regime_weights_split_evenly_exactly_on_a_boundary() -> None:
    # Landing exactly on a seam is the one place the smooth and hard switches
    # disagree however narrow the width: the gate is at its midpoint, so the
    # star is shared, while the hard rule owes it entirely to the upper regime.
    labels = _labels(np.array([5500.0, 7500.0]))
    weights = temperature_regime_weights(labels, width_K=1.0e-3)
    np.testing.assert_allclose(weights[0], [0.5, 0.5, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(weights[1], [0.0, 0.5, 0.5], atol=1.0e-12)
    np.testing.assert_array_equal(temperature_regimes(labels), [1, 2])


def test_shifted_boundaries_move_the_hard_assignment() -> None:
    labels = _labels(np.array([7000.0, 8000.0]))
    np.testing.assert_array_equal(temperature_regimes(labels), [1, 2])
    np.testing.assert_array_equal(
        temperature_regimes(labels, boundaries=(6300.0, 8700.0)), [1, 1]
    )


def test_exact_mcnemar_is_one_when_the_arms_never_disagree() -> None:
    assert _exact_mcnemar(0, 0) == 1.0
    # Symmetric disagreement is the null, however much of it there is.
    assert _exact_mcnemar(7, 7) == 1.0


def test_exact_mcnemar_matches_the_binomial_by_hand() -> None:
    # Three discordant pairs all favouring one arm: 2 * (1/8) = 0.25.
    assert _exact_mcnemar(3, 0) == pytest.approx(0.25)
    assert _exact_mcnemar(0, 3) == pytest.approx(0.25)
    # One-versus-three: two-sided tail is 2 * (C(4,0) + C(4,1)) / 2^4.
    assert _exact_mcnemar(1, 3) == pytest.approx(2.0 * 5.0 / 16.0)


def test_exact_mcnemar_is_symmetric_and_bounded() -> None:
    for first, second in ((1, 0), (4, 1), (9, 2), (20, 6)):
        assert _exact_mcnemar(first, second) == pytest.approx(
            _exact_mcnemar(second, first)
        )
        assert 0.0 < _exact_mcnemar(first, second) <= 1.0


def test_regime_boundaries_must_increase() -> None:
    labels = _labels(np.array([6000.0]))
    with pytest.raises(ValueError):
        temperature_regimes(labels, boundaries=(7500.0, 5500.0))
    with pytest.raises(ValueError):
        temperature_regime_weights(labels, boundaries=(7500.0, 5500.0))
