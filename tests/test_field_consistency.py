"""Fast tests for the four-field comparison metrics."""

from __future__ import annotations

import numpy as np

from experiments.reduced_state_emulator.field_consistency import _metric


def test_radiative_acceleration_metric_is_finite_at_zero():
    result = _metric(
        np.array([[0.0, 4.0]]),
        np.array([[0.0, 2.0]]),
        "radiative_acceleration",
    )
    values = result["normalized_error"]
    assert np.isfinite(values["median"])
    assert np.isfinite(values["p95"])
    assert values["max"] < 1.0


def test_positive_fields_report_relative_and_log_errors():
    result = _metric(
        np.array([[2.0, 8.0]]),
        np.array([[1.0, 4.0]]),
        "gas_pressure",
    )
    assert result["relative_error"]["median"] == 1.0
    assert np.isclose(result["log10_error"]["median"], np.log10(2.0))
