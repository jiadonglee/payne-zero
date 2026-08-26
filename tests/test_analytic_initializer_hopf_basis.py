"""Tests for the physical depth columns behind the recorded negative result.

The Hopf-augmented basis was measured and not adopted.  These tests pin the
properties the measurement relied on, so the conclusion stays checkable: the
columns are the exact grey solution, they vanish deep, and the classical one
reproduces the textbook Hopf function.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.analytic_initializer.run_hopf_basis_probe import (
    _design,
    classical_hopf_parameter,
    grey_hopf_column,
    physical_columns,
)

TAU = np.logspace(-7.0, 3.0, 80)


def test_the_eddington_case_is_identically_zero() -> None:
    """q = 2/3 is the grey atmosphere T_grey is already normalized by."""

    assert np.allclose(grey_hopf_column(TAU, 2.0 / 3.0), 0.0, atol=0.0)


def test_the_column_is_the_exact_grey_solution() -> None:
    """Not a fit: invert it and the Hopf parameter comes straight back."""

    q = 0.42
    column = grey_hopf_column(TAU, q)
    recovered = (TAU + 2.0 / 3.0) * 10.0 ** (4.0 * column) - TAU
    assert np.allclose(recovered, q, rtol=1.0e-10)


def test_physical_columns_vanish_deep() -> None:
    """They may only carry surface structure; the deep is left to Chebyshev."""

    for variant in ("classical_hopf", "hopf_plus_2q", "hopf_plus_5q", "hopf_2q_kernels"):
        block = physical_columns(TAU, variant)
        assert np.abs(block[-1]).max() < 1.0e-3, variant
        assert np.all(np.isfinite(block)), variant


def test_the_classical_parameter_matches_the_textbook_limits() -> None:
    assert classical_hopf_parameter(np.asarray(0.0)) == pytest.approx(0.577, abs=1.0e-3)
    assert classical_hopf_parameter(np.asarray(50.0)) == pytest.approx(0.710, abs=1.0e-6)
    # Monotone increasing between those limits.
    values = classical_hopf_parameter(np.linspace(0.0, 10.0, 200))
    assert np.all(np.diff(values) > 0.0)


def test_a_single_classical_column_is_perfectly_conditioned() -> None:
    block = physical_columns(TAU, "classical_hopf")
    assert block.shape[1] == 1
    assert np.linalg.cond(block) == pytest.approx(1.0)


def test_more_constant_q_columns_cost_conditioning() -> None:
    """Why the probe did not simply take the largest physical block."""

    small = np.linalg.cond(physical_columns(TAU, "hopf_plus_2q"))
    large = np.linalg.cond(physical_columns(TAU, "hopf_plus_5q"))
    assert large > 100.0 * small


def test_the_hybrid_design_has_exactly_the_requested_dimension() -> None:
    for variant in (None, "classical_hopf", "hopf_2q_kernels"):
        for dimension in (9, 13, 17):
            assert _design(TAU, variant, dimension).shape == (TAU.size, dimension)


def test_the_hybrid_design_refuses_a_dimension_below_its_physical_block() -> None:
    with pytest.raises(ValueError, match="too small"):
        _design(TAU, "hopf_plus_5q", 5)


def test_an_unknown_variant_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown physical basis variant"):
        physical_columns(TAU, "not_a_variant")


def test_the_physical_block_really_does_lower_the_representation_floor() -> None:
    """The half of the result that was positive, kept so it stays true."""

    tau = np.logspace(-6.0, 2.5, 80)
    log_tau = np.log(tau)
    # A target with genuine surface Hopf structure plus a deep departure.
    q = classical_hopf_parameter(tau) + 0.15 * np.tanh(log_tau)
    target = (grey_hopf_column(tau, q) + 0.01 * np.tanh(log_tau - 2.0))[None, :]

    def floor(design):
        coefficients, *_ = np.linalg.lstsq(design, target.T, rcond=None)
        return np.abs((design @ coefficients).T - target).max()

    assert floor(_design(tau, "hopf_2q_kernels", 11)) < floor(_design(tau, None, 11))
