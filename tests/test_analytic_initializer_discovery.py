"""Tests for the offline analytic-initializer discovery primitives."""

from __future__ import annotations

import json

import numpy as np

from experiments.analytic_initializer.discovery import (
    LABEL_FIELDS,
    collect_excluded_indices,
    fit_low_rank_surrogate,
    label_features,
    make_split,
    polynomial_exponents,
    polynomial_features,
)


def _labels(count: int = 24) -> np.ndarray:
    return np.column_stack(
        (
            np.linspace(4200.0, 9800.0, count),
            np.linspace(1.0, 5.0, count),
            np.linspace(-2.2, 0.3, count),
            np.linspace(-0.05, 0.45, count),
            np.linspace(0.6, 3.8, count),
        )
    )


def test_label_features_are_finite_and_dimensionless() -> None:
    values = label_features(_labels())
    assert values.shape == (24, len(LABEL_FIELDS))
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(values[:, 0], 5040.0 / _labels()[:, 0])
    np.testing.assert_allclose(values[:, -1], np.log10(_labels()[:, -1]))


def test_polynomial_library_has_expected_term_count_and_reuses_scaling() -> None:
    features = label_features(_labels())
    exponents = polynomial_exponents(features.shape[1], degree=2)
    first, center, scale = polynomial_features(features[:12], exponents)
    second, _, _ = polynomial_features(features[12:], exponents, center=center, scale=scale)
    assert exponents.shape == (21, 5)
    assert first.shape == (12, 21)
    assert second.shape == (12, 21)
    np.testing.assert_allclose(first[:, 0], 1.0)


def test_split_is_disjoint_and_reproducible() -> None:
    first = make_split(100, excluded=[1, 2, 3], seed=17)
    second = make_split(100, excluded=[3, 2, 1], seed=17)
    np.testing.assert_array_equal(first.train, second.train)
    np.testing.assert_array_equal(first.validation, second.validation)
    assert not np.intersect1d(first.train, first.validation).size
    assert not np.intersect1d(first.train, first.excluded).size
    assert not np.intersect1d(first.validation, first.excluded).size


def test_manifest_index_union_ignores_non_index_metadata(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "star_indices": [1, 4],
                "selection": {"star_indices": [4, 9]},
                "star_count": 10,
                "not_indices": [20],
            }
        ),
        encoding="utf-8",
    )
    indices, used = collect_excluded_indices([path], corpus_size=10)
    np.testing.assert_array_equal(indices, [1, 4, 9])
    assert used == [str(path)]


def test_low_rank_surrogate_recovers_a_smooth_separable_profile() -> None:
    labels = _labels(60)
    features = label_features(labels)
    x = (features - features.mean(axis=0)) / features.std(axis=0)
    depth = np.linspace(-1.0, 1.0, 8)
    target = (
        0.12 * x[:, 0, None]
        + 0.07 * x[:, 1, None] * depth[None, :]
        + 0.03 * x[:, 2, None] ** 2 * (1.0 - depth[None, :] ** 2)
    )
    result = fit_low_rank_surrogate(
        target,
        labels,
        np.arange(45),
        np.arange(45, 60),
        components=3,
        degree=2,
    )
    assert result["r2"] > 0.99
    # The design is nearly rank-three but the validation points are an
    # extrapolation of the monotone synthetic label path, so keep this as a
    # reconstruction sanity check rather than a machine-precision assertion.
    assert result["absolute_error_p95"] < 1.0e-4
