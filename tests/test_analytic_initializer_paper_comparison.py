from types import SimpleNamespace

import numpy as np
import pytest

from experiments.analytic_initializer.build_paper_dev60_comparison import (
    _network_parameter_count,
    _profile_errors,
    _solver_summary,
)
from experiments.analytic_initializer.run_h2_solver_funnel import _run_funnel
from paper.collect_numbers import _validate_analytic_comparison_arrays


def test_recorded_two_field_network_parameter_count():
    assert _network_parameter_count(labels=5, layers=80, width=512, depth=4) == 873_120


def test_profile_errors_use_the_manuscript_metrics():
    truth_mass = np.array([[1.0, 10.0]])
    truth_temperature = np.array([[100.0, 200.0]])
    mass = np.array([[1.0, 100.0]])
    temperature = np.array([[110.0, 180.0]])

    summary, temperature_error, mass_error = _profile_errors(
        mass, temperature, truth_mass, truth_temperature
    )

    np.testing.assert_allclose(temperature_error, [[0.1, 0.1]])
    np.testing.assert_allclose(mass_error, [[0.0, 1.0]])
    assert summary["temperature_relative_p50"] == pytest.approx(0.1)
    assert summary["column_mass_dex_p50"] == pytest.approx(0.5)


def test_solver_summary_counts_timeouts_as_failures():
    converged = np.array([True, False, True])
    iterations = np.array([4.0, np.nan, 6.0])

    summary = _solver_summary(
        converged, iterations, timeout_count=1, error_count=0
    )

    assert summary["converged_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["timeout_count"] == 1
    assert summary["mean_iterations_converged"] == 5.0


def test_paper_collector_rejects_json_npz_metric_drift():
    learned_temperature = np.array([[0.1, 0.2], [0.3, 0.4]])
    learned_mass = np.array([[0.2, 0.3], [0.4, 0.5]])
    analytic_temperature = learned_temperature + 0.1
    analytic_mass = learned_mass + 0.1
    learned_ok = np.array([True, False])
    analytic_ok = np.array([True, True])
    learned_iterations = np.array([3.0, np.nan])
    analytic_iterations = np.array([5.0, 4.0])

    def profile(temperature, mass):
        return {
            "temperature_relative_p50": float(np.percentile(temperature, 50)),
            "temperature_relative_p95": float(np.percentile(temperature, 95)),
            "column_mass_dex_p50": float(np.percentile(mass, 50)),
            "column_mass_dex_p95": float(np.percentile(mass, 95)),
        }

    def solver(ok, iterations):
        values = iterations[ok]
        return {
            "star_count": int(ok.size),
            "converged_count": int(ok.sum()),
            "failure_count": int((~ok).sum()),
            "mean_iterations_converged": float(values.mean()),
            "median_iterations_converged": float(np.median(values)),
            "p90_iterations_converged": float(np.percentile(values, 90)),
        }

    comparison = {
        "sample": {"star_indices": [1, 2]},
        "learned_two_field": {
            "profile_errors": profile(learned_temperature, learned_mass),
            "solver": solver(learned_ok, learned_iterations),
        },
        "analytic_parity": {
            "profile_errors": profile(analytic_temperature, analytic_mass),
            "solver": solver(analytic_ok, analytic_iterations),
        },
        "paired_solver": {
            "common_converged_count": 1,
            "learned_only_converged_count": 0,
            "analytic_only_converged_count": 1,
            "neither_converged_count": 0,
            "learned_fewer_iterations_count": 1,
            "analytic_fewer_iterations_count": 0,
            "tied_count": 0,
            "mean_analytic_minus_learned_iterations": 2.0,
            "median_analytic_minus_learned_iterations": 2.0,
        },
    }
    arrays = {
        "star_indices": np.array([1, 2]),
        "learned_temperature_relative_error": learned_temperature,
        "learned_column_mass_dex_error": learned_mass,
        "analytic_temperature_relative_error": analytic_temperature,
        "analytic_column_mass_dex_error": analytic_mass,
        "learned_converged": learned_ok,
        "analytic_converged": analytic_ok,
        "learned_iterations": learned_iterations,
        "analytic_iterations": analytic_iterations,
    }
    _validate_analytic_comparison_arrays(comparison, arrays)
    arrays["analytic_temperature_relative_error"] = analytic_temperature.copy()
    arrays["analytic_temperature_relative_error"][0, 0] += 1.0
    with pytest.raises(SystemExit, match="JSON/NPZ mismatch"):
        _validate_analytic_comparison_arrays(comparison, arrays)


def test_funnel_resume_reuses_complete_stream(tmp_path):
    path = tmp_path / "partial.jsonl"
    fields = (
        '"arm": "parity", "effective_temperature": 5000.0, '
        '"log_surface_gravity": 4.0, "metallicity": 0.0, '
        '"alpha_enhancement": 0.0, "microturbulence_km_s": 1.0'
    )
    path.write_text(
        f'{{"corpus_index": 2, "converged": true, {fields}}}\n'
        f'{{"corpus_index": 1, "converged": false, {fields}}}\n',
        encoding="utf-8",
    )
    corpus = SimpleNamespace(labels=np.tile([5000.0, 4.0, 0.0, 0.0, 1.0], (3, 1)))

    records = _run_funnel(
        corpus,
        np.array([1, 2]),
        arm="parity",
        reduced_state=None,
        timeout=1.0,
        jsonl_path=path,
        resume=True,
    )

    assert [record["corpus_index"] for record in records] == [1, 2]


def test_funnel_refuses_existing_rows_without_resume(tmp_path):
    path = tmp_path / "partial.jsonl"
    path.write_text(
        '{"corpus_index": 1, "arm": "parity", '
        '"effective_temperature": 5000.0, "log_surface_gravity": 4.0, '
        '"metallicity": 0.0, "alpha_enhancement": 0.0, '
        '"microturbulence_km_s": 1.0}\n',
        encoding="utf-8",
    )
    corpus = SimpleNamespace(labels=np.tile([5000.0, 4.0, 0.0, 0.0, 1.0], (2, 1)))

    with pytest.raises(SystemExit, match="already holds records"):
        _run_funnel(
            corpus,
            np.array([1]),
            arm="parity",
            reduced_state=None,
            timeout=1.0,
            jsonl_path=path,
            resume=False,
        )
