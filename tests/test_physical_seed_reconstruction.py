"""Tests for the physics-default reconstruction seed.

The reduced-state reconstruction historically seeded its derived-field
guesses from the six-field warm-start network. The physical seed replaces
that shortcut with labels + hydrostatic balance alone (``P = g*m``,
``n_e = 1e-4*P/kT``); these tests pin its contract without running the
solver. The solver-level seed-independence check lives in
``experiments/reduced_state_emulator/seed_independence_check.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from payne_zero_atmosphere.run_setup import (
    standard_rosseland_optical_depth_grid,
    validate_atmosphere_seed,
)
from payne_zero_atmosphere.warm_start import (
    compute_hydrogen_fraction,
    compute_metal_log_number_abundances,
)
from reduced_state.reconstruct import (
    ReducedAtmosphere,
    _physical_seed_atmosphere,
    _seed_atmosphere,
)


LABELS = {
    "effective_temperature": 5040.0,
    "log_surface_gravity": 2.5,
    "metallicity": -1.0,
    "alpha_enhancement": 0.3,
    "microturbulence_km_s": 2.0,
}


def _reduced() -> ReducedAtmosphere:
    tau = standard_rosseland_optical_depth_grid(80)
    temperature = LABELS["effective_temperature"] * (0.75 * (tau + 2.0 / 3.0)) ** 0.25
    return ReducedAtmosphere(
        column_mass=tau / 0.34, temperature=temperature, labels=dict(LABELS)
    )


def test_physical_seed_passes_solver_validation():
    atmosphere = _physical_seed_atmosphere(_reduced())
    validate_atmosphere_seed(atmosphere)


def test_physical_seed_uses_hydrostatic_pressure_and_positive_electron_seed():
    reduced = _reduced()
    atmosphere = _physical_seed_atmosphere(reduced)
    gravity = 10.0 ** LABELS["log_surface_gravity"]
    # The deck round-trip quantizes the seed fields to %.3E; the pinned
    # fields are restored exactly afterwards and must not be quantized.
    np.testing.assert_allclose(
        atmosphere.gas_pressure, gravity * reduced.column_mass, rtol=4.0e-4
    )
    assert np.all(atmosphere.electron_density > 0.0)
    assert np.all(atmosphere.rosseland_opacity > 0.0)
    np.testing.assert_array_equal(atmosphere.column_mass, reduced.column_mass)
    np.testing.assert_array_equal(atmosphere.temperature, reduced.temperature)
    np.testing.assert_allclose(
        atmosphere.microturbulence,
        np.full(80, LABELS["microturbulence_km_s"] * 1.0e5),
        rtol=4.0e-4,
    )


def test_physical_seed_metadata_and_abundances_match_production_wiring():
    atmosphere = _physical_seed_atmosphere(_reduced())
    assert atmosphere.metadata["effective_temperature"] == "5040.000000"
    assert atmosphere.metadata["log_surface_gravity"] == "2.500000"
    assert "opacity_flags" in atmosphere.metadata

    abundances = atmosphere.fixed_column_abundance_values
    # The deck writes abundances with 5 decimal places; allow the rounding.
    assert abundances[1] == pytest.approx(
        compute_hydrogen_fraction(
            metallicity=LABELS["metallicity"],
            alpha_enhancement=LABELS["alpha_enhancement"],
        ),
        abs=1.0e-5,
    )
    assert abundances[2] == pytest.approx(0.07837, abs=1.0e-7)
    expected_metals = compute_metal_log_number_abundances(
        metallicity=LABELS["metallicity"],
        alpha_enhancement=LABELS["alpha_enhancement"],
    )
    # The deck only writes metals above log abundance -10; the parser floors
    # the rest. Compare the written ones.
    for atomic_number in range(3, 100):
        expected = float(expected_metals[atomic_number - 3])
        if expected > -10.0:
            assert abundances[atomic_number] == pytest.approx(expected, abs=1.0e-5)


def test_physical_seed_rejects_non_monotonic_column_mass():
    reduced = _reduced()
    broken = ReducedAtmosphere(
        column_mass=reduced.column_mass[::-1],
        temperature=reduced.temperature,
        labels=reduced.labels,
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        _physical_seed_atmosphere(broken)


def test_seed_dispatch_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown seed mode"):
        _seed_atmosphere(_reduced(), seed="neural-five-field")


def test_physical_seed_dispatch_never_queries_six_field_emulator(monkeypatch):
    import reduced_state.reconstruct as reconstruction

    def fail_if_called(**_kwargs):
        raise AssertionError("six-field emulator was queried")

    monkeypatch.setattr(
        reconstruction, "emulator_warm_start_model", fail_if_called
    )
    atmosphere = reconstruction._seed_atmosphere(_reduced(), seed="physical")
    validate_atmosphere_seed(atmosphere)
