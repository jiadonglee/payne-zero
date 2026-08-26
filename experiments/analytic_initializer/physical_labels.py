"""Label coordinates with the ionization physics put in by hand.

The depth axis of the compact formula turned out not to be where the error
lives.  Splitting the held-out temperature error by stage gives 16 percent to
the depth basis, 37 percent to the rank-5 truncation and 48 percent to the map
from labels to mode amplitudes.  This module is about that 48 percent.

Two things were measured first, because they decide whether the term is worth
attacking at all.  The amplitude function is globally smooth -- k-nearest
neighbours in label space does *worse* than a global polynomial (0.0086 at
k=10 against 0.0086 for degree 3, and 0.0110 at k=40), so the right family is
parametric, not local.  And it is under-resolved rather than noisy: raising the
polynomial degree keeps paying, 0.00856 at degree 3, 0.00698 at 4, 0.00670 at
5, against a per-star oracle floor of 0.00449.

So the question is which coordinates let a cheap polynomial reach what an
expensive one reaches.  The answer is the Saha ionized fraction, and the
reason it works is that it is a sigmoid in effective temperature -- the shape a
total-degree polynomial has to spend many terms approximating.  Hydrogen
ionization sets where the convection zone begins and supplies the electrons
that H-minus opacity needs, so it is not a curve-fitting trick.

Measured on the held-out split, temperature residual p95 in dex, then column
mass p95 in dex:

    5 labels, degree 3      56 terms    0.00856    0.0870   <- what was there
    5 labels, degree 4     126 terms    0.00698    0.0636
    7 labels, degree 3     120 terms    0.00615    0.0585
    7 labels, capped       104 terms    0.00614    0.0597   <- this module

Capping means the two ionization features may appear at most linearly.  They
are already the nonlinear part; letting them multiply each other and everything
else three deep costs 16 more terms and buys nothing (0.00614 against 0.00615).

Three controls, all of which failed, are worth recording because each rules out
a cheaper story:

* Replacing the original labels with physical ones instead of adding to them is
  far worse -- 0.0268 against 0.0086.  Surface gravity and metallicity carry
  something the ionization fractions do not.
* Linear rather than logarithmic abundances (``10**[M/H]``, ``10**[a/M]``) buy
  nothing at all: 0.0086, unchanged.  It is specifically the sigmoid.
* A single ionization fraction gets most of it; the second is a small addition.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np

from .discovery import LABEL_FIELDS, label_features

#: Ionization potentials in eV.  Hydrogen is exact; the metal value is a stand
#: -in for the electron donors that matter in cool atmospheres -- Mg 7.65,
#: Si 8.15, Fe 7.90, Ca 6.11 -- and nothing here is sensitive to it at the
#: tenth of an eV.
HYDROGEN_POTENTIAL_EV = 13.598
METAL_POTENTIAL_EV = 7.6

#: Fraction of metal electron donors contributed by alpha elements, used to
#: fold ``[alpha/M]`` into a single effective metal abundance.
ALPHA_DONOR_FRACTION = 0.28

#: The Saha constant, ``log10(2 (2 pi m_e k)^{3/2} / h^3) - ...`` collected into
#: the usual -0.1762 for pressures in dyn/cm^2 and the partition function ratio
#: taken as unity, which the polynomial absorbs anyway.
_SAHA_OFFSET = 0.1762

#: The two ionization features are appended after the five standard ones.
PHYSICAL_FEATURE_FIELDS = LABEL_FIELDS + (
    "hydrogen_ionized_fraction",
    "metal_ionized_fraction",
)
#: Per-feature degree caps that reproduce the full table's accuracy at fewer
#: terms; see the module docstring.
PHYSICAL_DEGREE_CAPS = (3, 3, 3, 3, 3, 1, 1)


def effective_metal_abundance(labels: np.ndarray) -> np.ndarray:
    """Fold ``[alpha/M]`` into a single log abundance of electron donors."""

    values = np.asarray(labels, dtype=np.float64)
    return values[:, 2] + np.log10(
        (1.0 - ALPHA_DONOR_FRACTION) + ALPHA_DONOR_FRACTION * 10.0 ** values[:, 3]
    )


def electron_pressure_proxy(labels: np.ndarray) -> np.ndarray:
    """A ``log10 P_e`` scaling for the regime where metals donate electrons.

    When the electrons come from singly ionized metals rather than from
    hydrogen, charge balance gives ``n_e`` proportional to the square root of
    the product of the gas density and the metal abundance, so ``log P_e`` goes
    as half of ``log g`` plus half the metal abundance.  The additive constant
    is arbitrary here: every feature is standardized before use, and the
    polynomial absorbs any offset.
    """

    values = np.asarray(labels, dtype=np.float64)
    return 0.5 * (values[:, 1] + effective_metal_abundance(values)) - 0.5


def ionized_fraction(labels: np.ndarray, potential_eV: float) -> np.ndarray:
    """The Saha ionized fraction for a species of the given potential.

    ``log10(n_II / n_I) = -theta chi + 2.5 log10 T - log10 P_e - 0.1762``, with
    ``theta = 5040 / T``.  Returned as a fraction rather than a ratio so the
    feature is bounded in ``[0, 1]`` and saturates instead of diverging, which
    is the property that makes it cheap for a polynomial to use.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABEL_FIELDS):
        raise ValueError(f"labels must have shape (N, {len(LABEL_FIELDS)})")
    if not np.isfinite(potential_eV) or potential_eV <= 0.0:
        raise ValueError("potential_eV must be finite and positive")
    effective_temperature = values[:, 0]
    theta = 5040.0 / effective_temperature
    exponent = (
        -theta * potential_eV
        + 2.5 * np.log10(effective_temperature)
        - electron_pressure_proxy(values)
        - _SAHA_OFFSET
    )
    # The clip only prevents an overflow warning; 10**300 already saturates the
    # logistic to zero in double precision.
    return 1.0 / (1.0 + 10.0 ** (-np.clip(exponent, -300.0, 300.0)))


def physical_label_features(labels: np.ndarray) -> np.ndarray:
    """The five standard coordinates plus two Saha ionized fractions.

    Appended rather than substituted.  Substituting is much worse, and the
    module docstring records by how much.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    return np.column_stack(
        (
            label_features(values),
            ionized_fraction(values, HYDROGEN_POTENTIAL_EV),
            ionized_fraction(values, METAL_POTENTIAL_EV),
        )
    )


def capped_polynomial_exponents(
    total_degree: int, caps: Sequence[int] = PHYSICAL_DEGREE_CAPS
) -> np.ndarray:
    """Monomial exponents with a total degree and a per-feature ceiling.

    ``polynomial_exponents`` in ``discovery`` is the special case where every
    cap equals the total degree.  Capping matters here because the ionization
    features are already nonlinear: allowing them to appear cubically and to
    multiply everything else adds terms without adding accuracy.
    """

    if total_degree < 0:
        raise ValueError("total_degree must be non-negative")
    if not len(caps):
        raise ValueError("caps must name at least one feature")
    if any(cap < 0 for cap in caps):
        raise ValueError("caps must be non-negative")
    rows = [
        row
        for row in itertools.product(*[range(min(cap, total_degree) + 1) for cap in caps])
        if sum(row) <= total_degree
    ]
    # Ascending total degree keeps the table readable and puts the constant
    # term first, matching ``polynomial_exponents``.
    rows.sort(key=lambda row: (sum(row), row))
    return np.asarray(rows, dtype=np.int64)


#: Registry so a fitted closure can record which map it used and reload it.
FEATURE_MAPS = {
    "standard": label_features,
    "physical": physical_label_features,
}


def feature_map(name: str):
    """Look up a feature map by the name stored with a fitted closure."""

    if name not in FEATURE_MAPS:
        raise ValueError(f"unknown label feature map: {name}")
    return FEATURE_MAPS[name]
