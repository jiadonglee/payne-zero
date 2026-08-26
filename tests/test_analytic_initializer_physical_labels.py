"""Tests for the Saha label coordinates.

The claim these support is that the gain comes from the ionization physics and
not from having two more columns to fit with, so most of what is pinned here is
physical behaviour: the fractions saturate the right way, they move with
electron pressure in the right direction, and the degree cap that made them
cheap really does bound the table.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.analytic_initializer.discovery import label_features, polynomial_exponents
from experiments.analytic_initializer.physical_labels import (
    ALPHA_DONOR_FRACTION,
    HYDROGEN_POTENTIAL_EV,
    METAL_POTENTIAL_EV,
    PHYSICAL_DEGREE_CAPS,
    capped_polynomial_exponents,
    effective_metal_abundance,
    electron_pressure_proxy,
    feature_map,
    ionized_fraction,
    physical_label_features,
)


def _labels(teff=5772.0, logg=4.44, mh=0.0, am=0.0, vturb=1.0) -> np.ndarray:
    return np.asarray([[teff, logg, mh, am, vturb]], dtype=np.float64)


# --- the ionization fraction ------------------------------------------------


def test_hydrogen_is_neutral_when_cool_and_ionized_when_hot() -> None:
    assert ionized_fraction(_labels(teff=4000.0), HYDROGEN_POTENTIAL_EV)[0] < 1.0e-6
    assert ionized_fraction(_labels(teff=10500.0), HYDROGEN_POTENTIAL_EV)[0] > 0.9


def test_metals_ionize_far_earlier_than_hydrogen() -> None:
    """The whole point: two different potentials give two different sigmoids."""

    for teff in (4500.0, 5772.0, 7000.0):
        labels = _labels(teff=teff)
        assert (
            ionized_fraction(labels, METAL_POTENTIAL_EV)[0]
            > ionized_fraction(labels, HYDROGEN_POTENTIAL_EV)[0]
        )
    # And by the solar photosphere the metals are already mostly ionized while
    # hydrogen is not.
    assert ionized_fraction(_labels(), METAL_POTENTIAL_EV)[0] > 0.5
    assert ionized_fraction(_labels(), HYDROGEN_POTENTIAL_EV)[0] < 1.0e-3


def test_the_fraction_is_bounded_and_monotone_in_temperature() -> None:
    grid = np.column_stack(
        (
            np.linspace(4000.0, 10500.0, 400),
            np.full(400, 4.0),
            np.zeros(400),
            np.zeros(400),
            np.ones(400),
        )
    )
    for potential in (HYDROGEN_POTENTIAL_EV, METAL_POTENTIAL_EV):
        fraction = ionized_fraction(grid, potential)
        assert np.all((fraction >= 0.0) & (fraction <= 1.0))
        assert np.all(np.diff(fraction) > 0.0)
        assert np.all(np.isfinite(fraction))


def test_more_electron_pressure_suppresses_ionization() -> None:
    """Le Chatelier through the Saha equation: recombination wins at high P_e."""

    thin = ionized_fraction(_labels(teff=7000.0, logg=1.0), HYDROGEN_POTENTIAL_EV)[0]
    thick = ionized_fraction(_labels(teff=7000.0, logg=5.0), HYDROGEN_POTENTIAL_EV)[0]
    assert thin > thick


def test_metal_poor_stars_have_less_electron_pressure() -> None:
    rich = electron_pressure_proxy(_labels(mh=0.0))[0]
    poor = electron_pressure_proxy(_labels(mh=-2.5))[0]
    assert rich > poor
    # Half a dex of metals is half of half a dex of electron pressure.
    assert rich - poor == pytest.approx(0.5 * 2.5, rel=1.0e-9)


def test_alpha_enhancement_adds_donors_without_double_counting() -> None:
    plain = effective_metal_abundance(_labels(mh=0.0, am=0.0))[0]
    enhanced = effective_metal_abundance(_labels(mh=0.0, am=0.4))[0]
    assert enhanced > plain
    assert plain == pytest.approx(0.0, abs=1.0e-12)
    expected = np.log10((1.0 - ALPHA_DONOR_FRACTION) + ALPHA_DONOR_FRACTION * 10.0**0.4)
    assert enhanced == pytest.approx(expected, rel=1.0e-12)


def test_the_fraction_does_not_overflow_at_the_cool_extreme() -> None:
    with np.errstate(over="raise"):
        value = ionized_fraction(_labels(teff=4000.0, logg=5.3, mh=0.5), HYDROGEN_POTENTIAL_EV)
    assert np.all(np.isfinite(value))
    assert value[0] >= 0.0


def test_the_fraction_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        ionized_fraction(np.zeros((3, 4)), HYDROGEN_POTENTIAL_EV)
    with pytest.raises(ValueError, match="finite and positive"):
        ionized_fraction(_labels(), -1.0)


# --- the feature map --------------------------------------------------------


def test_physical_features_extend_rather_than_replace() -> None:
    """The control that failed: substituting is three times worse than adding."""

    labels = np.column_stack(
        (
            np.linspace(4200.0, 9800.0, 20),
            np.full(20, 3.0),
            np.linspace(-2.0, 0.4, 20),
            np.zeros(20),
            np.ones(20),
        )
    )
    features = physical_label_features(labels)
    assert features.shape == (20, 7)
    assert np.array_equal(features[:, :5], label_features(labels))


def test_the_feature_map_accepts_a_single_row() -> None:
    assert physical_label_features(np.asarray([5772.0, 4.44, 0.0, 0.0, 1.0])).shape == (1, 7)


def test_the_registry_round_trips_the_stored_name() -> None:
    labels = _labels()
    assert np.array_equal(feature_map("standard")(labels), label_features(labels))
    assert np.array_equal(feature_map("physical")(labels), physical_label_features(labels))
    with pytest.raises(ValueError, match="unknown label feature map"):
        feature_map("not_a_map")


# --- the capped exponent table ---------------------------------------------


def test_capping_reduces_the_table_without_touching_the_base_terms() -> None:
    full = polynomial_exponents(7, 3)
    capped = capped_polynomial_exponents(3, PHYSICAL_DEGREE_CAPS)
    assert capped.shape[0] == 104
    assert full.shape[0] == 120
    # Every capped row is a legal degree-3 row.
    assert np.all(capped.sum(axis=1) <= 3)
    # The two ionization columns never exceed first order, which is the whole
    # content of the cap.
    assert capped[:, 5].max() == 1
    assert capped[:, 6].max() == 1
    # Nothing that only involves the five standard labels was dropped.
    base_full = {tuple(row) for row in full if row[5] == 0 and row[6] == 0}
    base_capped = {tuple(row) for row in capped if row[5] == 0 and row[6] == 0}
    assert base_full == base_capped


def test_an_uncapped_table_reproduces_the_standard_one() -> None:
    capped = capped_polynomial_exponents(3, (3, 3, 3, 3, 3))
    standard = polynomial_exponents(5, 3)
    assert {tuple(row) for row in capped} == {tuple(row) for row in standard}


def test_the_table_starts_at_the_constant_term() -> None:
    table = capped_polynomial_exponents(3, PHYSICAL_DEGREE_CAPS)
    assert table[0].tolist() == [0] * 7
    assert np.all(np.diff(table.sum(axis=1)) >= 0)


def test_the_table_rejects_malformed_arguments() -> None:
    with pytest.raises(ValueError, match="total_degree"):
        capped_polynomial_exponents(-1, PHYSICAL_DEGREE_CAPS)
    with pytest.raises(ValueError, match="at least one feature"):
        capped_polynomial_exponents(3, ())
    with pytest.raises(ValueError, match="non-negative"):
        capped_polynomial_exponents(3, (3, -1))
