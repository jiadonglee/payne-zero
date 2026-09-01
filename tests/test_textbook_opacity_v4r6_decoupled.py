"""Focused tests for the v4r6 decoupled grey-mass / convective-T seed."""

from __future__ import annotations

import inspect

import numpy as np

from experiments.analytic_initializer.textbook_opacity import (
    build_textbook_reduced_state_v4,
    build_textbook_reduced_state_v4r3,
    build_textbook_reduced_state_v4r6,
    build_textbook_reduced_state_v4r6_decoupled,
    predict_textbook_reduced_state_v4r6,
    predict_textbook_reduced_state_v4r6_decoupled,
    textbook_rosseland_opacity_v4r6,
)
from experiments.analytic_initializer.textbook_opacity_v4r6_decoupled_gates import (
    DEVELOPMENT_COOL_CONVERGED_MIN,
    DEVELOPMENT_HOT_CONVERGED_MIN,
    DEVELOPMENT_LOSSES_AMONG_GREY_MAX,
    DEVELOPMENT_NET_GAIN_MIN,
    DEVELOPMENT_TIMEOUTS_MAX,
    DEVELOPMENT_TOTAL_CONVERGED_MIN,
    FAIL_STOP_DEVELOPMENT,
    PASS_TO_FRESH_OPEN,
    development_gate,
    exact_mcnemar_one_sided,
)


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(
        [
            [4000.0, 4.5, 0.0, 0.0, 1.0],
            [8000.0, 3.0, 0.0, 0.0, 1.5],
            [12000.0, 4.0, -0.5, 0.2, 1.0],
        ]
    )
    tau = 10.0 ** np.linspace(-6.875, 3.0, 12)
    return labels, tau


def test_candidate_output_shapes_match_v4r6() -> None:
    labels, tau = _inputs()
    mass_v4r6, temperature_v4r6, log_opacity_v4r6 = predict_textbook_reduced_state_v4r6(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    mass, temperature, log_opacity = predict_textbook_reduced_state_v4r6_decoupled(
        labels, tau, substeps_per_layer=2
    )
    assert mass.shape == mass_v4r6.shape == (labels.shape[0], tau.size)
    assert temperature.shape == temperature_v4r6.shape
    assert log_opacity.shape == log_opacity_v4r6.shape


def test_candidate_values_are_finite_and_positive() -> None:
    labels, tau = _inputs()
    mass, temperature, diagnostics = build_textbook_reduced_state_v4r6_decoupled(
        labels, tau, substeps_per_layer=2
    )
    opacity = np.asarray(diagnostics["rosseland_opacity"])
    for values in (mass, temperature, opacity):
        assert np.all(np.isfinite(values))
        assert np.all(values > 0.0)
    _, _, log_opacity = predict_textbook_reduced_state_v4r6_decoupled(
        labels, tau, substeps_per_layer=2
    )
    assert np.all(np.isfinite(log_opacity))


def test_candidate_mass_is_bitwise_equal_to_grey() -> None:
    labels, tau = _inputs()
    mass_grey, _, _ = build_textbook_reduced_state_v4r6(
        labels, tau, include_convection=False, substeps_per_layer=2
    )
    mass, _, _ = build_textbook_reduced_state_v4r6_decoupled(
        labels, tau, substeps_per_layer=2
    )
    np.testing.assert_allclose(mass, mass_grey, rtol=0.0, atol=0.0)


def test_candidate_temperature_is_bitwise_equal_to_convective() -> None:
    labels, tau = _inputs()
    _, temperature_convective, _ = build_textbook_reduced_state_v4r6(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    _, temperature, _ = build_textbook_reduced_state_v4r6_decoupled(
        labels, tau, substeps_per_layer=2
    )
    np.testing.assert_allclose(temperature, temperature_convective, rtol=0.0, atol=0.0)
    _, temperature_grey, _ = build_textbook_reduced_state_v4r6(
        labels, tau, include_convection=False, substeps_per_layer=2
    )
    assert not np.allclose(temperature, temperature_grey)


def test_candidate_opacity_matches_fresh_v4r6_recompute() -> None:
    labels, tau = _inputs()
    mass, temperature, diagnostics = build_textbook_reduced_state_v4r6_decoupled(
        labels, tau, substeps_per_layer=2
    )
    pressure = np.asarray(diagnostics["gas_pressure"])
    gravity = 10.0 ** labels[:, 1]
    np.testing.assert_allclose(pressure, gravity[:, None] * mass, rtol=0.0, atol=0.0)
    recomputed = textbook_rosseland_opacity_v4r6(labels, temperature, pressure)
    np.testing.assert_allclose(
        np.asarray(diagnostics["rosseland_opacity"]),
        recomputed,
        rtol=1.0e-12,
        atol=0.0,
    )
    assert diagnostics["mass_reintegrated_after_convection"] is False


def test_candidate_path_does_not_read_stored_atmospheric_state() -> None:
    source = inspect.getsource(build_textbook_reduced_state_v4r6_decoupled)
    source += inspect.getsource(predict_textbook_reduced_state_v4r6_decoupled)
    for forbidden in (
        "load_strict_truth",
        "atmosphere_profiles",
        "electron_density",
        "column_mass",
        "checkpoint",
        "corpus.",
    ):
        assert forbidden not in source


def test_candidate_has_zero_fitted_parameters() -> None:
    signature = inspect.signature(build_textbook_reduced_state_v4r6_decoupled)
    assert list(signature.parameters) == [
        "labels",
        "tau",
        "constants",
        "substeps_per_layer",
    ]
    signature = inspect.signature(predict_textbook_reduced_state_v4r6_decoupled)
    assert list(signature.parameters) == [
        "labels",
        "tau",
        "constants",
        "substeps_per_layer",
    ]


def test_existing_grey_and_convective_outputs_remain_unchanged() -> None:
    labels, tau = _inputs()
    mass_grey_before, temperature_grey_before, _ = predict_textbook_reduced_state_v4r6(
        labels, tau, include_convection=False, substeps_per_layer=2
    )
    mass_conv_before, temperature_conv_before, _ = predict_textbook_reduced_state_v4r6(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    predict_textbook_reduced_state_v4r6_decoupled(labels, tau, substeps_per_layer=2)
    mass_grey_after, temperature_grey_after, _ = predict_textbook_reduced_state_v4r6(
        labels, tau, include_convection=False, substeps_per_layer=2
    )
    mass_conv_after, temperature_conv_after, _ = predict_textbook_reduced_state_v4r6(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    np.testing.assert_allclose(mass_grey_after, mass_grey_before, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        temperature_grey_after, temperature_grey_before, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(mass_conv_after, mass_conv_before, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        temperature_conv_after, temperature_conv_before, rtol=0.0, atol=0.0
    )
    assert not np.allclose(mass_grey_before, mass_conv_before, rtol=1.0e-12, atol=0.0)


def test_existing_v4_through_v4r6_builders_remain_callable() -> None:
    labels, tau = _inputs()
    mass_v4, _, _ = build_textbook_reduced_state_v4(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    mass_v4r3, _, _ = build_textbook_reduced_state_v4r3(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    mass_v4r6, _, _ = build_textbook_reduced_state_v4r6(
        labels, tau, include_convection=True, substeps_per_layer=2
    )
    assert mass_v4.shape == mass_v4r3.shape == mass_v4r6.shape


def _record(index: int, teff: float, logg: float, *, converged: bool, outcome: str) -> dict:
    return {
        "corpus_index": index,
        "effective_temperature": teff,
        "log_surface_gravity": logg,
        "converged": converged,
        "solver_outcome": outcome,
    }


def _sixty_records(converged_cool: int, converged_hot: int, timeouts: int = 0) -> list[dict]:
    records = []
    for index in range(27):
        ok = index < converged_cool
        outcome = "converged" if ok else "not_converged"
        records.append(_record(index, 5000.0, 4.5, converged=ok, outcome=outcome))
    remaining_timeouts = timeouts
    for index in range(27, 60):
        ok = (index - 27) < converged_hot
        if not ok and remaining_timeouts:
            outcome = "timeout"
            remaining_timeouts -= 1
        else:
            outcome = "converged" if ok else "not_converged"
        records.append(_record(index, 9000.0, 4.0, converged=ok, outcome=outcome))
    return records


def test_development_gate_thresholds_are_frozen() -> None:
    assert DEVELOPMENT_COOL_CONVERGED_MIN == 11
    assert DEVELOPMENT_HOT_CONVERGED_MIN == 30
    assert DEVELOPMENT_TOTAL_CONVERGED_MIN == 41
    assert DEVELOPMENT_LOSSES_AMONG_GREY_MAX == 2
    assert DEVELOPMENT_NET_GAIN_MIN == 4
    assert DEVELOPMENT_TIMEOUTS_MAX == 3


def test_development_gate_passes_only_with_all_thresholds() -> None:
    grey = {"records": _sixty_records(6, 31, timeouts=3)}
    fail_candidate = {
        "records": _sixty_records(6, 31, timeouts=3),
        "initializer_provenance": {"finite_seed_count": 60},
    }
    assert development_gate(fail_candidate, grey)["decision"] == FAIL_STOP_DEVELOPMENT

    pass_candidate = {
        "records": _sixty_records(11, 31, timeouts=3),
        "initializer_provenance": {"finite_seed_count": 60},
    }
    result = development_gate(pass_candidate, grey)
    assert result["paired_vs_grey"]["net_gain"] == 5
    assert result["paired_vs_grey"]["losses_among_control_successes"] == 0
    assert result["decision"] == PASS_TO_FRESH_OPEN


def test_one_sided_mcnemar_matches_the_fresh_open_rule() -> None:
    assert exact_mcnemar_one_sided(0, 0) == 1.0
    assert exact_mcnemar_one_sided(6, 0) < 0.05
    assert exact_mcnemar_one_sided(4, 0) > 0.05


def test_iter60_driver_does_not_overwrite_the_15iter_result() -> None:
    from experiments.analytic_initializer.run_textbook_opacity_v4r6_decoupled_dev60 import (
        OUTPUT as ITER15_OUTPUT,
    )
    from experiments.analytic_initializer.run_textbook_opacity_v4r6_decoupled_dev60_iter60 import (
        DECISION,
        FIFTEEN_CONTROL,
        ITERATIONS,
        OUTPUT as ITER60_OUTPUT,
    )

    assert ITERATIONS == 60
    assert ITER60_OUTPUT != ITER15_OUTPUT
    assert FIFTEEN_CONTROL == ITER15_OUTPUT
    assert "iter60" in ITER60_OUTPUT.name
    assert DECISION == "ITER60_DIAGNOSTIC_COMPLETE"
    assert ITER15_OUTPUT.name == "textbook_opacity_v4r6_decoupled_dev60_20260828.json"


def test_late_convergence_counts_recoveries_against_the_15iter_arm() -> None:
    from experiments.analytic_initializer.run_textbook_opacity_v4r6_decoupled_dev60_iter60 import (
        _late_convergence,
    )

    fifteen = [
        _record(1, 5000.0, 4.5, converged=True, outcome="converged"),
        _record(2, 5000.0, 4.5, converged=False, outcome="not_converged"),
        _record(3, 9000.0, 4.0, converged=False, outcome="not_converged"),
        _record(4, 9000.0, 4.0, converged=True, outcome="converged"),
    ]
    sixty = [
        _record(1, 5000.0, 4.5, converged=True, outcome="converged"),
        _record(2, 5000.0, 4.5, converged=True, outcome="converged"),
        _record(3, 9000.0, 4.0, converged=False, outcome="not_converged"),
        _record(4, 9000.0, 4.0, converged=False, outcome="timeout"),
    ]
    summary = _late_convergence(sixty, fifteen)
    assert summary["recovered_count"] == 1
    assert summary["recovered_indices"] == [2]
    assert summary["recovered_cool_count"] == 1
    assert summary["recovered_hot_count"] == 0
    assert summary["lost_count"] == 1
    assert summary["lost_indices"] == [4]
    assert summary["still_converged_count"] == 1
    assert summary["still_failed_count"] == 1
    assert summary["paired_vs_15iter_decoupled"]["net_gain"] == 0


def test_iter100_residual_driver_pins_the_six_timeouts() -> None:
    import json
    from pathlib import Path

    from experiments.analytic_initializer.run_textbook_opacity_v4r6_decoupled_dev60_iter60 import (
        OUTPUT as ITER60_OUTPUT,
    )
    from experiments.analytic_initializer.run_textbook_opacity_v4r6_decoupled_dev60_iter100_residual import (
        DECISION,
        ITERATIONS,
        OUTPUT,
        PER_STAR_TIMEOUT_SECONDS,
        RESIDUAL_INDICES,
    )

    assert ITERATIONS == 100
    assert PER_STAR_TIMEOUT_SECONDS == 3600
    assert OUTPUT != ITER60_OUTPUT
    assert "iter100_residual" in OUTPUT.name
    assert DECISION == "ITER100_RESIDUAL_DIAGNOSTIC_COMPLETE"
    assert RESIDUAL_INDICES == (6152, 33051, 33053, 44167, 46124, 48708)
    control = json.loads(Path(ITER60_OUTPUT).read_text(encoding="utf-8"))
    timeouts = tuple(
        sorted(
            int(item["corpus_index"])
            for item in control["records"]
            if item.get("solver_outcome") == "timeout"
        )
    )
    assert timeouts == RESIDUAL_INDICES


def test_residual_outcome_counts_recoveries_on_the_six_stars() -> None:
    from experiments.analytic_initializer.run_textbook_opacity_v4r6_decoupled_dev60_iter100_residual import (
        RESIDUAL_INDICES,
        _residual_outcome,
    )

    sixty = [
        _record(index, 5000.0, 4.5, converged=False, outcome="timeout")
        for index in RESIDUAL_INDICES
    ]
    hundred = [
        _record(RESIDUAL_INDICES[0], 5000.0, 4.5, converged=True, outcome="converged"),
        *[
            _record(index, 5000.0, 4.5, converged=False, outcome="timeout")
            for index in RESIDUAL_INDICES[1:]
        ],
    ]
    summary = _residual_outcome(hundred, sixty)
    assert summary["recovered_count"] == 1
    assert summary["recovered_indices"] == [RESIDUAL_INDICES[0]]
    assert summary["still_failed_count"] == 5
    assert summary["still_timeout_count"] == 5
    assert summary["paired_vs_60iter_residual"]["net_gain"] == 1
