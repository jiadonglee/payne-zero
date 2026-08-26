"""Tests for the reference-solver benchmark harness.

These do not run the solver. A single solve costs ~40 s and 8 GB, so the pieces
that surround it are tested directly and the solve itself is covered by the
opt-in integration test in ``test_integration.py``.
"""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bench import environment
from bench.labels import (
    LABEL_FIELDS,
    StellarLabels,
    load,
    sample_uniform,
    save,
    support_bounds,
)
from bench import report
from bench import run_reference
from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere


# --- environment -------------------------------------------------------------


def test_configure_sets_workqueue_by_default(monkeypatch):
    monkeypatch.delenv("NUMBA_THREADING_LAYER", raising=False)
    applied = environment.configure()
    assert applied["NUMBA_THREADING_LAYER"] == "workqueue"


def test_configure_honours_an_explicit_layer(monkeypatch):
    monkeypatch.setenv("NUMBA_THREADING_LAYER", "tbb")
    applied = environment.configure()
    assert applied["NUMBA_THREADING_LAYER"] == "tbb", "caller's choice must win"


def test_configure_reports_thread_count_when_set(monkeypatch):
    monkeypatch.setenv("NUMBA_NUM_THREADS", "4")
    assert environment.configure()["NUMBA_NUM_THREADS"] == "4"


# --- labels ------------------------------------------------------------------


def test_support_bounds_cover_the_documented_range():
    bounds = support_bounds()
    assert set(bounds) == set(LABEL_FIELDS)
    low, high = bounds["effective_temperature"]
    # The checkpoint stores 5040/Teff, so the inversion must not swap the ends.
    assert low < high
    assert 3990 < low < 4010
    assert 10490 < high < 10510
    for field in LABEL_FIELDS:
        assert bounds[field][0] < bounds[field][1]


def test_sample_uniform_stays_strictly_inside_the_box():
    bounds = support_bounds()
    labels = sample_uniform(200, seed=1)
    assert len(labels) == 200
    for item in labels:
        for field in LABEL_FIELDS:
            low, high = bounds[field]
            value = getattr(item, field)
            assert low < value < high, f"{field}={value} escaped [{low}, {high}]"


def test_sample_uniform_is_reproducible():
    assert sample_uniform(10, seed=7) == sample_uniform(10, seed=7)
    assert sample_uniform(10, seed=7) != sample_uniform(10, seed=8)


def test_sample_uniform_rejects_a_nonpositive_count():
    with pytest.raises(ValueError):
        sample_uniform(0)


def test_slug_is_filesystem_safe_and_distinct():
    labels = sample_uniform(50, seed=3)
    slugs = {item.slug for item in labels}
    assert len(slugs) == 50
    for slug in slugs:
        assert "/" not in slug and " " not in slug


def test_labels_round_trip_through_jsonl(tmp_path):
    original = sample_uniform(5, seed=4)
    path = save(original, tmp_path / "nested" / "labels.jsonl")
    assert path.is_file(), "save must create missing parent directories"
    assert load(path) == original


def test_labels_load_accepts_a_json_list(tmp_path):
    original = sample_uniform(3, seed=5)
    path = tmp_path / "labels.json"
    path.write_text(json.dumps([item.as_kwargs() for item in original]))
    assert load(path) == original


def test_labels_load_ignores_blank_lines(tmp_path):
    path = tmp_path / "labels.jsonl"
    body = json.dumps(StellarLabels(5777.0, 4.44, 0.0, 0.0, 2.0).as_kwargs())
    path.write_text(f"\n{body}\n\n")
    assert len(load(path)) == 1


# --- record bookkeeping ------------------------------------------------------


def _trial(index, iterations, converged, **kwargs):
    return run_reference.TrialRecord(
        trial_index=index,
        initializer_label=None,
        iterations_completed=iterations,
        converged=converged,
        seconds=1.0,
        **kwargs,
    )


def _star(trials, labels=None):
    return run_reference.StarRecord(
        labels=labels or StellarLabels(5777.0, 4.44, 0.0, 0.0, 2.0),
        trials=trials,
        seconds=sum(t.seconds for t in trials),
    )


def test_star_record_first_trial_success():
    star = _star([_trial(0, 4, True)])
    assert star.converged
    assert not star.needed_retry
    assert star.total_iterations == 4
    assert star.converging_trial_iterations == 4


def test_star_record_counts_a_failed_trial_towards_cost():
    star = _star([_trial(0, 15, False), _trial(1, 6, True)])
    assert star.converged and star.needed_retry
    assert star.total_iterations == 21, "a failed trial still burns iterations"
    assert star.converging_trial_iterations == 6, "only the winning trial's own cost"


def test_star_record_total_failure():
    star = _star([_trial(0, 15, False), _trial(1, 15, False)])
    assert not star.converged
    assert star.converging_trial_iterations is None
    assert star.total_iterations == 30


def test_star_record_serializes_to_json():
    star = _star([_trial(0, 15, False, error="RuntimeError: boom"), _trial(1, 5, True)])
    encoded = json.loads(json.dumps(star.as_json()))
    assert encoded["trials_used"] == 2
    assert encoded["trials"][0]["error"].startswith("RuntimeError")
    assert encoded["labels"]["effective_temperature"] == 5777.0


# --- JSON coercion -----------------------------------------------------------


def test_as_plain_handles_every_type_the_runner_emits():
    plain = run_reference._as_plain(
        {
            "int": np.int64(3),
            "float": np.float64(1.5),
            "bool": np.bool_(True),
            "array": np.arange(3),
            "path": Path("/tmp/x.npz"),
            "nested": [{"a": np.float32(0.25)}, (np.int32(1), None)],
            "none": None,
            "str": "plain",
        }
    )
    json.dumps(plain)  # must not raise
    assert plain["int"] == 3
    assert plain["array"] == [0, 1, 2]
    assert plain["path"] == "/tmp/x.npz"
    assert plain["nested"][0]["a"] == 0.25
    assert plain["nested"][1] == [1, None]


def test_as_plain_maps_non_finite_values_to_null():
    """A diverged solve emits NaN flux errors; bare NaN is not valid JSON."""

    plain = run_reference._as_plain(
        {
            "inf": np.float64("inf"),
            "nan": np.float64("nan"),
            "array": np.array([1.0, np.nan, np.inf]),
            "python_nan": float("nan"),
        }
    )
    assert plain["inf"] is None and plain["nan"] is None
    assert plain["python_nan"] is None
    assert plain["array"] == [1.0, None, None]
    # Standard JSON: no bare NaN or Infinity tokens for jq or pandas to choke on.
    encoded = json.dumps(plain, allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_as_plain_keeps_finite_floats_exact():
    plain = run_reference._as_plain({"x": np.float64(1.5), "y": 0.0, "z": -2.25})
    assert plain == {"x": 1.5, "y": 0.0, "z": -2.25}


# --- solver configuration ----------------------------------------------------


def test_solver_config_matches_the_production_policy():
    config = run_reference._solver_config(
        object(),
        iterations_per_trial=15,
        structured_atmosphere_path=None,
        debug_state_path=None,
    )
    assert config.iterations == 15
    assert config.enable_molecules and config.enable_convection
    assert config.enable_convergence_stop
    assert config.minimum_iterations_before_convergence == 3
    assert config.required_consecutive_converged_iterations == 1
    assert config.maximum_deep_layer_relative_temperature_change == pytest.approx(5e-4)
    assert config.inputs.molecules_path.is_file()
    assert config.inputs.predicted_atomic_lines_path.is_file()


def test_solver_config_threads_output_paths(tmp_path):
    config = run_reference._solver_config(
        object(),
        iterations_per_trial=3,
        structured_atmosphere_path=tmp_path / "p.npz",
        debug_state_path=tmp_path / "d.npz",
    )
    assert config.outputs.structured_atmosphere_path == tmp_path / "p.npz"
    assert config.outputs.debug_state_path == tmp_path / "d.npz"


# --- run_star ------------------------------------------------------------


def test_run_star_rejects_a_nonfinite_converged_atmosphere(monkeypatch):
    """The structural stop only checks a temperature layer window, so a
    solver-reported convergence can still hide a non-finite state elsewhere
    (e.g. gas pressure). ``run_star`` must not report that trial as usable."""

    layer = np.array([1.0, 2.0])
    nonfinite_atmosphere = ModelAtmosphere(
        column_mass=layer,
        temperature=layer,
        gas_pressure=np.array([1.0, np.nan]),
        electron_density=layer,
        rosseland_opacity=layer,
        radiative_acceleration=layer,
        microturbulence=layer,
        convective_flux=layer,
        convective_velocity=layer,
    )
    monkeypatch.setattr(
        run_reference, "deterministic_initializer_labels", lambda **kw: (None,)
    )
    monkeypatch.setattr(
        run_reference,
        "emulator_warm_start_model",
        lambda **kw: (object(), object()),
    )
    monkeypatch.setattr(
        run_reference,
        "run_atmosphere_model",
        lambda config: SimpleNamespace(
            converged=True,
            iterations_completed=4,
            diagnostics={},
            atmosphere=nonfinite_atmosphere,
        ),
    )

    record = run_reference.run_star(StellarLabels(5777.0, 4.44, 0.0, 0.0, 2.0))

    assert not record.converged
    assert not record.trials[0].converged
    assert any("non-finite values" in w for w in record.warnings)


# --- run_many ----------------------------------------------------------------


def test_run_many_streams_records_and_survives_interruption(tmp_path, monkeypatch):
    calls = []

    def fake_run_star(labels, **options):
        calls.append(labels)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return _star([_trial(0, 4, True)], labels=labels)

    monkeypatch.setattr(run_reference, "run_star", fake_run_star)
    out = tmp_path / "records.jsonl"
    with pytest.raises(KeyboardInterrupt):
        run_reference.run_many(sample_uniform(5, seed=2), out_path=out)

    written = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert len(written) == 2, "records before the interruption must survive on disk"


def test_run_many_appends_rather_than_truncating(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_reference, "run_star",
        lambda labels, **o: _star([_trial(0, 4, True)], labels=labels),
    )
    out = tmp_path / "records.jsonl"
    out.write_text('{"pre": "existing"}\n')
    run_reference.run_many(sample_uniform(2, seed=2), out_path=out)
    assert len(out.read_text().splitlines()) == 3


def test_worker_payload_is_picklable():
    """ProcessPoolExecutor pickles this; a failure here breaks --workers > 1."""

    payload = (
        sample_uniform(1, seed=9)[0],
        {"iterations_per_trial": 15, "max_trials": 2,
         "trace_dir": Path("/tmp/traces"), "keep_product": False},
    )
    assert pickle.loads(pickle.dumps(payload)) == payload
    assert pickle.loads(pickle.dumps(run_reference._worker)) is run_reference._worker


# --- report ------------------------------------------------------------------


def _record(iterations, converged=True, trials=1, labels=None, residuals=(1e-3, 5e-4)):
    timings = [
        {
            "iteration": index + 1,
            "deep_layer_relative_temperature_change": value,
            "population_seconds": 0.1,
            "opacity_seconds": 5.0,
            "transfer_seconds": 0.1,
            "finalization_seconds": 0.3,
            "remap_seconds": 0.01,
            "total_seconds": 5.5,
        }
        for index, value in enumerate(residuals)
    ]
    star_labels = labels or StellarLabels(5777.0, 4.44, 0.0, 0.0, 2.0)
    trial_records = []
    for index in range(trials):
        is_last = index == trials - 1
        trial_records.append(
            run_reference.TrialRecord(
                trial_index=index,
                initializer_label=None,
                iterations_completed=iterations if is_last else 15,
                converged=converged and is_last,
                seconds=10.0,
                diagnostics={"iteration_timings": timings},
            )
        )
    return _star(trial_records, labels=star_labels).as_json()


def test_summarize_basic_counts():
    records = [_record(4), _record(6), _record(15, converged=False, trials=2)]
    summary = report.summarize(records)
    assert summary["star_count"] == 3
    assert summary["converged_count"] == 2
    assert summary["failure_count"] == 1
    assert summary["retry_count"] == 1
    assert summary["converging_trial_iterations"]["histogram"] == {"4": 1, "6": 1}


def test_summarize_headroom_excludes_the_floor():
    summary = report.summarize([_record(3), _record(3), _record(9)])
    head = summary["headroom"]
    assert head["stars_already_at_floor"] == 2
    assert head["total_iterations_burned"] == 15
    assert head["recoverable_iterations"] == 6, "only iterations above the floor count"


def test_summarize_contraction_flags_non_monotonic_trajectories():
    rising = _record(4, residuals=(1e-3, 2e-3, 4e-4))
    falling = _record(4, residuals=(1e-3, 5e-4, 2e-4))
    summary = report.summarize([rising, falling])
    assert summary["contraction"]["non_monotonic_fraction"] == pytest.approx(0.5)
    assert summary["contraction"]["q_ratio"]["p50"] > 0


def test_contraction_summary_is_robust_to_a_near_zero_denominator():
    """One residual that nearly vanishes must not dominate the summary.

    An arithmetic mean over q = r_next/r is meaningless here: a single solve
    that lands on a residual of 1e-12 produces a ratio in the millions. The
    geometric mean is the per-iteration factor that actually compounds.
    """

    normal = [_record(4, residuals=(1e-3, 5e-4, 2.5e-4)) for _ in range(20)]
    pathological = [_record(4, residuals=(1e-12, 1e-3))]
    q = report.summarize(normal + pathological)["contraction"]["q_ratio"]

    assert q["max"] > 1e6, "the outlier must still be visible"
    assert 0.4 < q["geometric_mean"] < 1.5, "but must not dominate the summary"
    assert q["p50"] == pytest.approx(0.5, abs=0.05)


def test_summarize_separates_first_from_later_iteration_cost():
    stages = report.summarize([_record(4, residuals=(1e-3, 5e-4))])["stage_seconds"]
    assert stages["first_iteration_mean"]["opacity_seconds"] == pytest.approx(5.0)
    assert stages["later_iteration_mean"]["opacity_seconds"] == pytest.approx(5.0)


def test_summarize_handles_all_stars_failing():
    """No converged star means empty percentile inputs; must not raise."""

    summary = report.summarize([_record(15, converged=False, trials=2) for _ in range(3)])
    assert summary["converged_count"] == 0
    assert math.isnan(summary["converging_trial_iterations"]["mean"])
    assert summary["total_iterations_including_retries"]["mean"] == 30
    report.format_report(summary)  # must render without raising


def test_summarize_handles_a_trial_that_raised():
    star = _star([_trial(0, 0, False, error="RuntimeError: boom")]).as_json()
    summary = report.summarize([star])
    assert summary["failure_count"] == 1
    assert summary["contraction"]["trajectory_count"] == 0
    report.format_report(summary)


def test_summarize_rejects_an_empty_record_set():
    with pytest.raises(ValueError):
        report.summarize([])


def test_tail_labels_identify_the_expensive_population():
    cheap = [_record(3, labels=StellarLabels(5000.0, 4.5, 0.0, 0.0, 2.0)) for _ in range(9)]
    expensive = [_record(15, labels=StellarLabels(10000.0, 1.0, 0.0, 0.0, 2.0))]
    tail = report.summarize(cheap + expensive)["tail_labels"]
    assert tail["tail_count"] == 1
    assert tail["labels"]["effective_temperature"]["tail_mean"] == 10000.0
    assert tail["labels"]["log_surface_gravity"]["tail_mean"] == 1.0


def test_load_records_and_format_report_round_trip(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text("\n".join(json.dumps(_record(n)) for n in (3, 5, 9)) + "\n")
    text = report.format_report(report.summarize(report.load_records(path)))
    assert "Stage 0 baseline" in text
    assert "recoverable iterations" in text


def test_report_cli_writes_json(tmp_path, capsys):
    records = tmp_path / "records.jsonl"
    records.write_text("\n".join(json.dumps(_record(n)) for n in (3, 5)) + "\n")
    summary_path = tmp_path / "summary.json"
    assert report.main([str(records), "--json", str(summary_path)]) == 0
    assert json.loads(summary_path.read_text())["star_count"] == 2
    assert "Stage 0 baseline" in capsys.readouterr().out
