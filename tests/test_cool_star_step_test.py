"""Unit tests for the cool-star experiment plumbing.

The real 619 MB MARCS file and the exact solver are intentionally not used by
the default test suite.  The integration test is opt-in with
``PAYNE_ZERO_RUN_SOLVER=1`` on a machine that has the optional HDF5 dependency
and the solver data assets.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from bench.labels import StellarLabels
from experiments.reduced_state_emulator.cool_star_step_test import (
    ANCHOR_TEMPERATURE,
    CONTINUATION_100_TARGETS,
    TARGET_TEMPERATURES,
    _analytic_grey_hydrostatic_mt,
    _atmosphere_quality,
    build_track_manifest,
    manifest_hash,
    manifest_payload,
)
from experiments.reduced_state_emulator.cool_star_extrapolated_target import (
    _positive_log_extrapolation,
)
from experiments.reduced_state_emulator.cool_star_adaptive import (
    backtrack_step,
    proposed_temperature,
)
from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere


def test_manifest_has_pilot_and_frozen_nine_track_confirmation_set():
    pilot = build_track_manifest("pilot")
    confirm = build_track_manifest("confirm")
    assert len(pilot) == 1
    assert pilot[0].log_surface_gravity == 5.0
    assert pilot[0].metallicity == 0.0
    assert len(confirm) == 9
    assert len({track.track_id for track in confirm}) == 9
    assert {(track.log_surface_gravity, track.metallicity) for track in confirm} == {
        (logg, metallicity)
        for logg in (4.5, 5.0, 5.5)
        for metallicity in (-1.0, 0.0, 0.5)
    }


def test_manifest_is_deterministically_hashable():
    payload = manifest_payload("confirm")
    assert payload["anchor_temperature"] == ANCHOR_TEMPERATURE
    assert payload["target_temperatures"] == list(TARGET_TEMPERATURES)
    assert payload["continuation_schedules"]["100K"] == [
        4000.0,
        *CONTINUATION_100_TARGETS,
    ]
    assert CONTINUATION_100_TARGETS == (3900.0, 3800.0, 3700.0, 3600.0, 3500.0)
    assert manifest_hash(payload) == manifest_hash(manifest_payload("confirm"))


def test_quality_gate_requires_positive_monotone_six_field_state():
    values = np.linspace(1.0, 2.0, 80)
    atmosphere = ModelAtmosphere(
        column_mass=values,
        temperature=values + 3000.0,
        gas_pressure=values,
        electron_density=values,
        rosseland_opacity=values,
        radiative_acceleration=values,
        microturbulence=values,
        convective_flux=values,
        convective_velocity=values,
    )
    result = _atmosphere_quality(atmosphere)
    assert result["valid"] is True
    invalid = ModelAtmosphere(
        column_mass=values[::-1],
        temperature=values + 3000.0,
        gas_pressure=values,
        electron_density=values,
        rosseland_opacity=values,
        radiative_acceleration=values,
        microturbulence=values,
        convective_flux=values,
        convective_velocity=values,
    )
    assert _atmosphere_quality(invalid)["valid"] is False


def test_log_predictor_extrapolates_one_temperature_step_and_preserves_order():
    anchor = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    source = np.asarray([2.0, 4.0, 8.0], dtype=np.float64)
    predicted = _positive_log_extrapolation(anchor, source)
    np.testing.assert_allclose(predicted, [4.0, 8.0, 16.0])
    assert np.all(np.diff(predicted) > 0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        _positive_log_extrapolation(anchor, np.asarray([2.0, 0.0, 8.0]))


def test_analytic_grey_hydrostatic_seed_is_valid_without_emulator_checkpoint():
    labels = StellarLabels(3500.0, 5.0, 0.0, 0.0, 1.0)
    result = _analytic_grey_hydrostatic_mt(labels)
    column_mass = result["column_mass"]
    temperature = result["temperature"]
    assert column_mass.shape == (80,)
    assert temperature.shape == (80,)
    assert np.all(np.isfinite(column_mass))
    assert np.all(np.isfinite(temperature))
    assert np.all(column_mass > 0.0)
    assert np.all(temperature > 0.0)
    assert np.all(np.diff(column_mass) > 0.0)
    assert result["diagnostics"]["analytic_model"] == "grey_hydrostatic"
    assert temperature[0] < temperature[-1]
    assert temperature[0] < labels.effective_temperature
    assert temperature[-1] > labels.effective_temperature


def test_adaptive_policy_halves_a_failed_cool_step_without_overshooting():
    assert proposed_temperature(4000.0, 3500.0, 500.0) == pytest.approx(3500.0)
    half_step = backtrack_step(500.0, 50.0)
    assert half_step == pytest.approx(250.0)
    assert proposed_temperature(4000.0, 3500.0, half_step) == pytest.approx(3750.0)
    assert proposed_temperature(3750.0, 3500.0, half_step) == pytest.approx(3500.0)
    assert backtrack_step(75.0, 50.0) == pytest.approx(50.0)


def test_adaptive_policy_rejects_invalid_step_bounds():
    with pytest.raises(ValueError, match="positive"):
        backtrack_step(0.0, 50.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        backtrack_step(50.0, 100.0)


def _make_synthetic_marcs(path: Path) -> None:
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001 - local NumPy/HDF5 ABI may differ
        pytest.skip(f"h5py unavailable for HDF5 loader test: {exc}")
    temperatures = np.linspace(3500.0, 3550.0, 56)
    total_number_density = np.geomspace(1.0e15, 1.0e18, 56)
    electron_density = total_number_density * 1.0e-3
    tau = np.geomspace(1.0e-6, 1.0e2, 56)
    height = np.linspace(1.0e8, -1.0e8, 56)
    encoded = np.empty((1, 1, 1, 1, 1, 5, 56), dtype=np.float64)
    encoded[0, 0, 0, 0, 0, 0] = temperatures
    encoded[0, 0, 0, 0, 0, 1] = np.log(electron_density)
    encoded[0, 0, 0, 0, 0, 2] = np.log(total_number_density)
    encoded[0, 0, 0, 0, 0, 3] = tau
    encoded[0, 0, 0, 0, 0, 4] = np.arcsinh(height)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "grid_parameter_names",
            data=np.asarray([b"Teff", b"logg", b"metallicity", b"alpha", b"carbon"]),
        )
        values = handle.create_group("grid_values")
        for index, value in enumerate(
            ([3500.0], [5.0], [0.0], [0.0], [0.0]), start=1
        ):
            values.create_dataset(str(index), data=np.asarray(value))
        handle.create_dataset("grid", data=encoded)


def test_marcs_loader_decodes_native_node_and_rejects_off_grid(tmp_path):
    path = tmp_path / "synthetic_marcs.h5"
    _make_synthetic_marcs(path)
    pytest.importorskip("h5py")
    from experiments.reduced_state_emulator.marcs_h5 import (
        BOLTZMANN_CGS,
        inspect_marcs_grid,
        load_marcs_node,
        MarcsH5Error,
    )

    schema = inspect_marcs_grid(path, verify_sha256=False, expected_sha256=None)
    labels = StellarLabels(3500.0, 5.0, 0.0, 0.0, 1.0)
    node = load_marcs_node(
        path,
        labels,
        schema=schema,
        verify_sha256=False,
        expected_sha256=None,
    )
    assert schema.grid_shape == (1, 1, 1, 1, 1, 5, 56)
    assert node.native_electron_density[0] == pytest.approx(1.0e12)
    expected_pressure = 1.0e15 * BOLTZMANN_CGS * 3500.0
    assert node.native_gas_pressure[0] == pytest.approx(expected_pressure)
    assert node.native_column_mass.shape == (56,)
    assert node.reduced_column_mass.shape == (80,)
    assert node.reduced_temperature.shape == (80,)
    assert np.all(np.isfinite(node.reduced_temperature))
    assert np.all(node.reduced_temperature > 0.0)
    assert np.all(np.diff(node.reduced_column_mass) > 0.0)
    tau_node = load_marcs_node(
        path,
        labels,
        schema=schema,
        verify_sha256=False,
        expected_sha256=None,
        depth_coordinate="tau5000",
    )
    assert tau_node.reduced_column_mass.shape == (80,)
    assert tau_node.reduced_temperature.shape == (80,)
    assert np.all(np.isfinite(tau_node.reduced_column_mass))
    assert np.all(np.isfinite(tau_node.reduced_temperature))
    assert np.all(tau_node.reduced_column_mass > 0.0)
    assert np.all(np.diff(tau_node.reduced_column_mass) > 0.0)
    with pytest.raises(ValueError, match="depth_coordinate"):
        load_marcs_node(
            path,
            labels,
            schema=schema,
            verify_sha256=False,
            expected_sha256=None,
            depth_coordinate="unknown",
        )
    with pytest.raises(MarcsH5Error, match="off the native MARCS grid"):
        load_marcs_node(
            path,
            StellarLabels(3501.0, 5.0, 0.0, 0.0, 1.0),
            schema=schema,
            verify_sha256=False,
            expected_sha256=None,
        )


@pytest.mark.skipif(
    os.environ.get("PAYNE_ZERO_RUN_SOLVER") != "1",
    reason="exact cool-star solver scan is opt-in and belongs on Garching",
)
def test_opt_in_pilot_solver_smoke(tmp_path):
    """Run the requested 4000 -> 3500 full-carry/recomputed smoke test."""

    from experiments.reduced_state_emulator.cool_star_step_test import (
        _production_atmosphere,
        _reconstruct_from_mt,
        _retarget_full_state,
        _solve_attempt,
        TrackSpec,
    )
    from experiments.reduced_state_emulator.marcs_h5 import (
        inspect_marcs_grid,
        load_marcs_node,
    )

    grid = Path(
        os.environ.get(
            "PAYNE_ZERO_MARCS_GRID",
            Path(__file__).resolve().parents[1] / "SDSS_MARCS_atmospheres.h5",
        )
    )
    schema = inspect_marcs_grid(grid, verify_sha256=True)
    track = TrackSpec(log_surface_gravity=5.0, metallicity=0.0)
    anchor_labels = track.labels(4000.0)
    target_labels = track.labels(3500.0)
    anchor_record, anchor_state = _solve_attempt(
        track=track,
        method="smoke_anchor",
        schedule="smoke",
        source_temperature=None,
        target_labels=anchor_labels,
        initial_atmosphere=_production_atmosphere(anchor_labels),
        product_dir=tmp_path / "products" / "smoke_anchor",
        iteration_cap=30,
    )
    assert anchor_record["survives_solver"]
    assert anchor_state is not None

    full_record, _full_state = _solve_attempt(
        track=track,
        method="smoke_full_carry",
        schedule="smoke",
        source_temperature=4000.0,
        target_labels=target_labels,
        initial_atmosphere=_retarget_full_state(
            anchor_state, _production_atmosphere(target_labels)
        ),
        product_dir=tmp_path / "products" / "smoke_full_carry",
        iteration_cap=30,
    )
    assert full_record["survives_solver"]

    marcs_node = load_marcs_node(
        grid,
        anchor_labels,
        schema=schema,
        verify_sha256=False,
        expected_sha256=None,
    )
    marcs_anchor_record, _marcs_anchor_state = _solve_attempt(
        track=track,
        method="smoke_marcs_anchor",
        schedule="smoke",
        source_temperature=None,
        target_labels=anchor_labels,
        initial_atmosphere=_reconstruct_from_mt(
            anchor_labels,
            marcs_node.reduced_column_mass,
            marcs_node.reduced_temperature,
        ),
        product_dir=tmp_path / "products" / "smoke_marcs_anchor",
        iteration_cap=30,
    )
    assert marcs_anchor_record["survives_solver"]

    reduced_record, _reduced_state = _solve_attempt(
        track=track,
        method="smoke_reduced_rematerialized",
        schedule="smoke",
        source_temperature=4000.0,
        target_labels=target_labels,
        initial_atmosphere=_reconstruct_from_mt(
            target_labels, anchor_state.column_mass, anchor_state.temperature
        ),
        product_dir=tmp_path / "products" / "smoke_reduced_rematerialized",
        iteration_cap=30,
    )
    assert reduced_record["survives_solver"]
