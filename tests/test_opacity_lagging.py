"""Unit tests for opacity lagging and the fixed-point invariant it must keep.

Opacity is 79% of solver wall time and is recomputed at full price on late
iterations where the atmosphere has stopped moving (iteration 2 onward costs a
flat ~1.85 s while the deep-layer temperature change falls by an order of
magnitude). Lagging freezes the opacity operator for a few iterations at a
time. That is only admissible if it cannot move the fixed point, which is what
these tests pin:

* the flag is off by default and the off path calls ``prepare_opacity_state``
  on every single iteration, exactly as before;
* convergence is never declared on an iteration that reused stale opacity.

No solve is run here. The expensive stages are replaced by stubs so the
scheduling, reuse, and stop policy are exercised on their own.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from payne_zero_atmosphere import runner
from payne_zero_atmosphere.config import (
    AtmosphereConfig,
    AtmosphereInput,
    AtmosphereOutput,
    DEFAULT_OPACITY_FLAGS,
)
from payne_zero_atmosphere.convergence import evaluate_convergence_stop
from payne_zero_atmosphere.line_opacity import LineOpacityState
from payne_zero_atmosphere.run_setup import (
    ConvectionSettings,
    RunSetup,
    TurbulenceSettings,
    resolve_run_setup,
)
from payne_zero_atmosphere.runner import (
    IterationCarry,
    OpacityState,
    opacity_recompute_scheduled,
    reuse_lagged_opacity_state,
)


LAYERS = 3
FREQUENCIES = 4


# --------------------------------------------------------------------------
# Fixtures: the smallest objects the code under test actually reads.
# --------------------------------------------------------------------------


def _atmosphere(temperature: float = 5000.0) -> SimpleNamespace:
    """A stand-in atmosphere with only the fields the loop body touches."""

    return SimpleNamespace(
        layers=LAYERS,
        column_mass=np.linspace(1.0, 3.0, LAYERS),
        temperature=np.full(LAYERS, temperature),
        gas_pressure=np.full(LAYERS, 1.0e4),
        thermal_energy_erg=np.full(LAYERS, 1.0e-12),
        microturbulence=np.full(LAYERS, 1.0),
    )


def _run_setup(
    *,
    enable_opacity_lagging: bool = False,
    opacity_recompute_interval: int = 1,
    iterations: int = 8,
    minimum_iterations_before_convergence: int = 3,
    required_consecutive_converged_iterations: int = 1,
) -> RunSetup:
    return RunSetup(
        atmosphere=_atmosphere(),
        iterations=iterations,
        enable_convergence_stop=True,
        minimum_iterations_before_convergence=minimum_iterations_before_convergence,
        required_consecutive_converged_iterations=(
            required_consecutive_converged_iterations
        ),
        maximum_deep_layer_relative_temperature_change=5.0e-4,
        maximum_all_layer_relative_temperature_change=None,
        surface_gravity_cgs=10.0**4.44,
        # Required, no default: RunSetup states the production grid explicitly
        # so nothing can construct a setup without deciding the stride.
        opacity_frequency_grid_stride=1,
        opacity_flags=list(DEFAULT_OPACITY_FLAGS),
        molecules_enabled=False,
        pressure_iteration_enabled=False,
        convection=ConvectionSettings(
            enabled=True,
            mixing_length=1.25,
            overshoot_weight=0.0,
            zero_top_layer_count=0,
        ),
        turbulence=TurbulenceSettings(
            enabled=False,
            density_coefficient=0.0,
            density_power=0.0,
            sound_speed_fraction=0.0,
            constant_velocity_km_s=0.0,
        ),
        surface_radiation_pressure_constant=0.0,
        effective_temperature=5777.0,
        log_surface_gravity=4.44,
        standard_rosseland_optical_depth=np.linspace(1.0e-6, 1.0, LAYERS),
        enable_opacity_lagging=enable_opacity_lagging,
        opacity_recompute_interval=opacity_recompute_interval,
    )


def _config(**kwargs) -> AtmosphereConfig:
    return AtmosphereConfig(
        inputs=AtmosphereInput(initial_atmosphere=_atmosphere()),
        outputs=AtmosphereOutput(),
        **kwargs,
    )


def _opacity_state(population_state, *, fill: float = 1.0) -> OpacityState:
    """A fully populated ``OpacityState`` with tiny, distinguishable arrays."""

    slab = np.full((LAYERS, FREQUENCIES), fill, dtype=np.float64)
    return OpacityState(
        population_state=population_state,
        continuum_atmosphere=SimpleNamespace(layers=LAYERS, tag="exact"),
        opacity_wavelength_grid_nm=np.linspace(400.0, 900.0, FREQUENCIES),
        opacity_frequency_hz=np.linspace(1.0e15, 2.0e15, FREQUENCIES),
        frequency_weights=np.full(FREQUENCIES, 0.25),
        active_continuum_indices=np.arange(FREQUENCIES),
        active_continuum_frequency_hz=np.linspace(1.0e15, 2.0e15, FREQUENCIES),
        continuum_absorption=slab.copy(),
        continuum_scattering=slab.copy() * 2.0,
        continuum_source=slab.copy() * 3.0,
        continuum_line_selection_threshold=slab.copy() * 4.0,
        continuum_reference_wavelength_nm=np.linspace(400.0, 900.0, FREQUENCIES),
        wavelength_bin_edges=np.arange(FREQUENCIES + 1),
        line_opacity=LineOpacityState(
            line_mass_absorption_coefficient=slab.copy() * 5.0,
            selected_line_count=7,
        ),
        rosseland_table=SimpleNamespace(tag="exact_table"),
        selected_line_catalog=SimpleNamespace(tag="selected"),
        transition_line_catalog=SimpleNamespace(tag="transition"),
    )


def _carry(setup: RunSetup) -> IterationCarry:
    return runner.initialize_iteration_carry(setup)


# --------------------------------------------------------------------------
# The flag is off by default.
# --------------------------------------------------------------------------


def test_opacity_lagging_is_off_by_default():
    config = _config()
    assert config.enable_opacity_lagging is False
    assert config.opacity_recompute_interval == 2, "the interval only bites when on"


def test_resolve_run_setup_defaults_to_recomputing_every_iteration():
    setup = _run_setup()
    assert setup.enable_opacity_lagging is False
    assert setup.opacity_recompute_interval == 1


def test_resolve_run_setup_rejects_a_non_positive_interval(monkeypatch):
    config = _config(opacity_recompute_interval=0)
    monkeypatch.setattr("payne_zero_atmosphere.run_setup.validate_atmosphere_seed", lambda _: None)
    monkeypatch.setattr(
        "payne_zero_atmosphere.run_setup.initialize_microturbulence",
        lambda *args, **kwargs: None,
    )
    config = dataclasses.replace(
        config,
        inputs=AtmosphereInput(
            initial_atmosphere=SimpleNamespace(layers=LAYERS, metadata={})
        ),
    )
    with pytest.raises(ValueError, match="opacity_recompute_interval"):
        resolve_run_setup(config)


# --------------------------------------------------------------------------
# The schedule.
# --------------------------------------------------------------------------


def _scheduled(index, **kwargs):
    defaults = {
        "total_iterations": 15,
        "enable_opacity_lagging": True,
        "opacity_recompute_interval": 2,
        "has_reusable_opacity": True,
        "force_exact_opacity": False,
    }
    defaults.update(kwargs)
    return opacity_recompute_scheduled(iteration_index=index, **defaults)


def test_scheduler_always_recomputes_when_lagging_is_off():
    for index in range(1, 16):
        assert _scheduled(index, enable_opacity_lagging=False) is True


def test_interval_of_one_recomputes_every_iteration():
    for index in range(1, 16):
        assert _scheduled(index, opacity_recompute_interval=1) is True


def test_interval_of_two_alternates_exact_and_lagged():
    exact = [index for index in range(1, 11) if _scheduled(index)]
    assert exact == [1, 3, 5, 7, 9]


def test_interval_of_three_recomputes_one_iteration_in_three():
    exact = [
        index
        for index in range(1, 11)
        if _scheduled(index, opacity_recompute_interval=3)
    ]
    assert exact == [1, 4, 7, 10]


def test_first_iteration_is_always_exact_because_nothing_can_be_reused():
    assert _scheduled(2, has_reusable_opacity=False) is True


def test_last_iteration_of_the_budget_is_never_lagged():
    # Iteration 8 would be lagged on an interval of 2, but a budget-exhausted
    # run returns that iteration's atmosphere, so it must be exact.
    assert _scheduled(8, total_iterations=15) is False
    assert _scheduled(8, total_iterations=8) is True


def test_forced_exact_opacity_overrides_the_schedule():
    assert _scheduled(2) is False
    assert _scheduled(2, force_exact_opacity=True) is True


# --------------------------------------------------------------------------
# What a lagged state reuses and what it recomputes.
# --------------------------------------------------------------------------


def test_lagged_state_reuses_the_opacity_slabs_and_takes_the_current_population(
    monkeypatch,
):
    previous_population = SimpleNamespace(
        setup=_run_setup(), runtime_state=SimpleNamespace(tag="old")
    )
    previous = _opacity_state(previous_population)

    current_population = SimpleNamespace(
        setup=_run_setup(), runtime_state=SimpleNamespace(tag="new")
    )
    monkeypatch.setattr(
        runner,
        "build_continuum_atmosphere_state",
        lambda atmosphere, state: SimpleNamespace(layers=LAYERS, tag="rebuilt"),
    )
    fresh_table = SimpleNamespace(tag="fresh_table")

    lagged = reuse_lagged_opacity_state(
        previous,
        population_state=current_population,
        rosseland_table=fresh_table,
    )

    # Reused: the whole monochromatic opacity operator, by reference.
    for field in (
        "continuum_absorption",
        "continuum_scattering",
        "continuum_source",
        "opacity_wavelength_grid_nm",
        "opacity_frequency_hz",
        "frequency_weights",
        "active_continuum_indices",
        "active_continuum_frequency_hz",
        "continuum_line_selection_threshold",
        "continuum_reference_wavelength_nm",
        "wavelength_bin_edges",
    ):
        assert getattr(lagged, field) is getattr(previous, field), field
    assert lagged.line_opacity is previous.line_opacity
    assert lagged.selected_line_catalog is previous.selected_line_catalog
    assert lagged.transition_line_catalog is previous.transition_line_catalog

    # Recomputed / current: population, continuum repack, carried table.
    assert lagged.population_state is current_population
    assert lagged.continuum_atmosphere.tag == "rebuilt"
    assert lagged.rosseland_table is fresh_table

    # And it is labelled as lagged, which is what the stop policy reads.
    assert lagged.opacity_recomputed is False
    assert previous.opacity_recomputed is True


def test_lagged_state_refuses_a_layer_count_change(monkeypatch):
    previous = _opacity_state(
        SimpleNamespace(setup=_run_setup(), runtime_state=SimpleNamespace())
    )
    other_setup = _run_setup()
    other_setup.atmosphere = SimpleNamespace(
        layers=LAYERS + 1,
        column_mass=np.linspace(1.0, 3.0, LAYERS + 1),
        temperature=np.full(LAYERS + 1, 5000.0),
    )
    monkeypatch.setattr(
        runner,
        "build_continuum_atmosphere_state",
        lambda atmosphere, state: SimpleNamespace(),
    )
    with pytest.raises(ValueError, match="layer count"):
        reuse_lagged_opacity_state(
            previous,
            population_state=SimpleNamespace(
                setup=other_setup, runtime_state=SimpleNamespace()
            ),
        )


# --------------------------------------------------------------------------
# Wiring: how many times does run_single_iteration compute opacity?
# --------------------------------------------------------------------------


@pytest.fixture
def stubbed_iteration(monkeypatch):
    """Replace every expensive stage of ``run_single_iteration`` with a stub.

    Returns the recompute spy: ``spy.calls`` counts real opacity computations.
    """

    spy = SimpleNamespace(calls=0, deep_change=1.0)

    monkeypatch.setattr(
        runner,
        "_copy_iteration_atmosphere",
        lambda atmosphere, gas_pressure=None: atmosphere,
    )
    monkeypatch.setattr(
        runner,
        "prepare_population_state",
        lambda config, **kwargs: SimpleNamespace(
            setup=kwargs["setup"],
            runtime_state=SimpleNamespace(),
            molecular_state=None,
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_continuum_atmosphere_state",
        lambda atmosphere, state: SimpleNamespace(layers=LAYERS),
    )

    def _prepare_opacity_state(config, *, population_state, **kwargs):
        spy.calls += 1
        return _opacity_state(population_state)

    monkeypatch.setattr(runner, "prepare_opacity_state", _prepare_opacity_state)
    monkeypatch.setattr(
        runner,
        "accumulate_transfer_state",
        lambda opacity, temperature_correction_state=None: SimpleNamespace(
            opacity_state=opacity,
            temperature_correction_state=SimpleNamespace(
                rosseland_opacity_table=SimpleNamespace()
            ),
        ),
    )
    monkeypatch.setattr(
        runner,
        "finalize_transfer_state",
        lambda transfer, **kwargs: SimpleNamespace(
            convection_result=SimpleNamespace(
                convective_flux=np.zeros(LAYERS),
                convective_velocity=np.zeros(LAYERS),
            ),
            radiative_pressure_state=SimpleNamespace(
                absolute_radiation_pressure=np.zeros(LAYERS),
                surface_radiation_pressure_constant=0.0,
            ),
            rosseland_opacity=np.ones(LAYERS),
            rosseland_optical_depth=np.ones(LAYERS),
        ),
    )
    monkeypatch.setattr(
        runner,
        "remap_finalized_iteration_state",
        lambda finalization, **kwargs: SimpleNamespace(
            finalization=SimpleNamespace(
                temperature_correction_result=SimpleNamespace(
                    flux_error_percent=np.zeros(LAYERS),
                    raw_temperature_correction=np.zeros(LAYERS),
                    flux_ratio=np.zeros(LAYERS),
                ),
                convection_result=SimpleNamespace(
                    logarithmic_temperature_pressure_gradient=np.zeros(LAYERS),
                    adiabatic_gradient=np.zeros(LAYERS),
                ),
            ),
            atmosphere=_atmosphere(),
            standard_rosseland_optical_depth=np.ones(LAYERS),
            integrated_radiation_pressure=np.zeros(LAYERS),
            turbulent_pressure=np.zeros(LAYERS),
        ),
    )
    monkeypatch.setattr(
        runner,
        "deep_layer_relative_temperature_change",
        lambda before, after: spy.deep_change,
    )
    monkeypatch.setattr(
        runner, "max_normalized_column_delta", lambda *args, **kwargs: spy.deep_change
    )
    return spy


def _drive(setup, config, iterations):
    carry = _carry(setup)
    used_exact = []
    for index in range(1, iterations + 1):
        step = runner.run_single_iteration(config, setup, carry, index)
        carry = step.carry
        used_exact.append(step.opacity_recomputed)
    return used_exact


def test_flag_off_recomputes_opacity_on_every_iteration(stubbed_iteration):
    setup = _run_setup(iterations=6)
    used_exact = _drive(setup, _config(), 6)

    assert stubbed_iteration.calls == 6, "the off path must not skip a single one"
    assert used_exact == [True] * 6


def test_flag_off_leaves_the_timing_payload_untouched(stubbed_iteration):
    setup = _run_setup(iterations=2)
    carry = _carry(setup)
    step = runner.run_single_iteration(_config(), setup, carry, 1)
    assert "opacity_recomputed" not in step.timing
    assert carry.previous_exact_opacity is None, "nothing is retained when off"


def test_flag_on_recomputes_only_on_the_schedule(stubbed_iteration):
    setup = _run_setup(
        enable_opacity_lagging=True, opacity_recompute_interval=2, iterations=6
    )
    used_exact = _drive(setup, _config(), 6)

    # 1, 3, 5 scheduled; 6 is the last iteration of the budget so it is forced.
    assert used_exact == [True, False, True, False, True, True]
    assert stubbed_iteration.calls == 4


def test_lagged_iterations_all_reuse_the_same_exact_slabs(stubbed_iteration):
    setup = _run_setup(
        enable_opacity_lagging=True, opacity_recompute_interval=4, iterations=8
    )
    carry = _carry(setup)
    absorption_seen = []
    for index in range(1, 5):
        step = runner.run_single_iteration(_config(), setup, carry, index)
        carry = step.carry
        absorption_seen.append(step.opacity.continuum_absorption)

    assert stubbed_iteration.calls == 1
    for slab in absorption_seen[1:]:
        assert slab is absorption_seen[0], "lagged states must not chain copies"


def test_flag_on_records_which_iterations_were_lagged(stubbed_iteration):
    setup = _run_setup(
        enable_opacity_lagging=True, opacity_recompute_interval=2, iterations=6
    )
    carry = _carry(setup)
    step = runner.run_single_iteration(_config(), setup, carry, 1)
    assert step.timing["opacity_recomputed"] == 1
    step = runner.run_single_iteration(_config(), setup, step.carry, 2)
    assert step.timing["opacity_recomputed"] == 0


# --------------------------------------------------------------------------
# The invariant: convergence is never declared off a lagged iteration.
# --------------------------------------------------------------------------


def _stop(**kwargs):
    defaults = {
        "enable_convergence_stop": True,
        "iteration_index": 5,
        "minimum_iterations_before_convergence": 3,
        "required_consecutive_converged_iterations": 1,
        "temperature_change_within_limit": True,
        "opacity_recomputed": True,
        "consecutive_converged_iterations": 0,
    }
    defaults.update(kwargs)
    return evaluate_convergence_stop(**defaults)


def test_exact_iteration_within_limits_converges():
    decision = _stop()
    assert decision.converged is True
    assert decision.consecutive_converged_iterations == 1


def test_lagged_iteration_within_limits_never_converges():
    decision = _stop(opacity_recomputed=False)
    assert decision.converged is False


def test_lagged_iteration_within_limits_forces_the_next_iteration_exact():
    decision = _stop(opacity_recomputed=False)
    assert decision.force_exact_opacity is True


def test_lagged_iteration_never_credits_the_consecutive_counter():
    decision = _stop(
        opacity_recomputed=False,
        required_consecutive_converged_iterations=2,
        consecutive_converged_iterations=1,
    )
    assert decision.converged is False
    assert decision.consecutive_converged_iterations == 1, "held, never incremented"


def test_lagged_iteration_still_clears_the_counter_when_the_state_moves():
    decision = _stop(
        opacity_recomputed=False,
        temperature_change_within_limit=False,
        consecutive_converged_iterations=1,
    )
    assert decision.consecutive_converged_iterations == 0
    assert decision.force_exact_opacity is False


def test_lagged_iteration_can_never_converge_for_any_input():
    for enable in (True, False):
        for index in (1, 3, 15):
            for within in (True, False):
                for consecutive in (0, 1, 5):
                    decision = _stop(
                        enable_convergence_stop=enable,
                        iteration_index=index,
                        temperature_change_within_limit=within,
                        consecutive_converged_iterations=consecutive,
                        opacity_recomputed=False,
                    )
                    assert decision.converged is False


def test_stop_policy_is_unchanged_when_every_iteration_is_exact():
    """The off path must reduce to the historical branch, case for case."""

    for enable in (True, False):
        for index in (1, 2, 3, 4):
            for within in (True, False):
                for consecutive in (0, 1, 2):
                    for required in (1, 2):
                        decision = _stop(
                            enable_convergence_stop=enable,
                            iteration_index=index,
                            minimum_iterations_before_convergence=3,
                            required_consecutive_converged_iterations=required,
                            temperature_change_within_limit=within,
                            consecutive_converged_iterations=consecutive,
                        )
                        if enable and index >= 3 and within:
                            expected = consecutive + 1
                        else:
                            expected = 0
                        assert (
                            decision.consecutive_converged_iterations == expected
                        )
                        assert decision.converged == bool(
                            enable and expected >= required
                        )
                        assert decision.force_exact_opacity is False


# --------------------------------------------------------------------------
# The invariant, end to end through the solver loop (no physics).
# --------------------------------------------------------------------------


@pytest.fixture
def stubbed_loop(monkeypatch):
    """Run ``_run_atmosphere_model``'s loop with a scripted iteration body."""

    def install(setup, residuals):
        seen = []

        def fake_run_single_iteration(config, run_setup, carry, iteration_index):
            recompute = opacity_recompute_scheduled(
                iteration_index=iteration_index,
                total_iterations=int(run_setup.iterations),
                enable_opacity_lagging=bool(run_setup.enable_opacity_lagging),
                opacity_recompute_interval=int(run_setup.opacity_recompute_interval),
                has_reusable_opacity=carry.previous_exact_opacity is not None,
                force_exact_opacity=bool(carry.force_exact_opacity),
            )
            if run_setup.enable_opacity_lagging and recompute:
                carry.previous_exact_opacity = SimpleNamespace()
                carry.force_exact_opacity = False
            change = residuals[min(iteration_index - 1, len(residuals) - 1)]
            seen.append((iteration_index, recompute, change))
            timing = {"iteration": iteration_index}
            if run_setup.enable_opacity_lagging:
                timing["opacity_recomputed"] = int(recompute)
            return runner.SingleIterationResult(
                carry=carry,
                remapped=SimpleNamespace(
                    finalization=SimpleNamespace(
                        temperature_correction_result=SimpleNamespace(
                            flux_error_percent=np.zeros(LAYERS)
                        )
                    ),
                    atmosphere=_atmosphere(),
                ),
                opacity=SimpleNamespace(
                    opacity_frequency_hz=np.zeros(FREQUENCIES),
                    population_state=SimpleNamespace(),
                ),
                transfer=SimpleNamespace(
                    frequency_start_index=0, frequency_stop_index=FREQUENCIES
                ),
                deep_layer_relative_temperature_change=change,
                all_layer_relative_temperature_change=change,
                timing=timing,
                opacity_recomputed=recompute,
            )

        monkeypatch.setattr(runner, "resolve_run_setup", lambda config: setup)
        monkeypatch.setattr(runner, "_require_supported_run_setup", lambda s: None)
        monkeypatch.setattr(
            runner, "run_single_iteration", fake_run_single_iteration
        )
        monkeypatch.setattr(
            runner,
            "finalize_remapped_iteration",
            lambda remapped, **kwargs: runner.AtmosphereRunResult(
                atmosphere=remapped.atmosphere,
                iterations_completed=kwargs["iterations_completed"],
                converged=kwargs["converged"],
                diagnostics=kwargs["diagnostics"],
            ),
        )
        return seen

    return install


def test_loop_converges_on_an_exact_iteration_when_lagging_is_off(stubbed_loop):
    setup = _run_setup(iterations=10)
    # Well inside the 5e-4 limit from iteration 2 onward.
    seen = stubbed_loop(setup, [1.0e-2] + [1.0e-6] * 9)
    result = runner._run_atmosphere_model(_config())

    assert result.converged is True
    assert result.iterations_completed == 3, "the minimum-iterations floor"
    assert all(recompute for _, recompute, _ in seen)


def test_loop_never_converges_on_a_lagged_iteration(stubbed_loop):
    setup = _run_setup(
        enable_opacity_lagging=True, opacity_recompute_interval=2, iterations=10
    )
    seen = stubbed_loop(setup, [1.0e-2] + [1.0e-6] * 9)
    result = runner._run_atmosphere_model(_config())

    assert result.converged is True
    last_index, last_recompute, _ = seen[-1]
    assert last_recompute is True, "the certifying iteration used exact opacity"
    assert result.iterations_completed == last_index
    assert result.diagnostics["converged_on_exact_opacity"] is True


def test_loop_holds_the_invariant_across_intervals_and_residual_scripts(stubbed_loop):
    """Sweep the schedule against residual histories that try to break it."""

    scripts = {
        "converges_immediately": [1.0e-6] * 12,
        "converges_late": [1.0e-2] * 6 + [1.0e-6] * 6,
        "flip_flops": [1.0e-6, 1.0e-2] * 6,
        "converges_only_on_even_iterations": [
            1.0e-2 if index % 2 else 1.0e-6 for index in range(12)
        ],
        "never_converges": [1.0e-2] * 12,
    }
    for interval in (2, 3, 4):
        for name, residuals in scripts.items():
            setup = _run_setup(
                enable_opacity_lagging=True,
                opacity_recompute_interval=interval,
                iterations=12,
            )
            seen = stubbed_loop(setup, residuals)
            result = runner._run_atmosphere_model(_config())
            if result.converged:
                _, last_recompute, _ = seen[-1]
                assert last_recompute is True, (
                    f"interval={interval} script={name}: convergence was declared "
                    "on a lagged iteration"
                )
            # Whether or not it converged, the returned atmosphere is always the
            # product of an exact-opacity iteration.
            assert seen[-1][1] is True, f"interval={interval} script={name}"


def test_loop_diagnostics_are_unchanged_on_the_default_path(stubbed_loop):
    setup = _run_setup(iterations=6)
    stubbed_loop(setup, [1.0e-2] * 6)
    result = runner._run_atmosphere_model(_config())

    for key in (
        "enable_opacity_lagging",
        "opacity_recompute_interval",
        "exact_opacity_iterations",
        "lagged_opacity_iterations",
        "converged_on_exact_opacity",
    ):
        assert key not in result.diagnostics
    for timing in result.diagnostics["iteration_timings"]:
        assert "opacity_recomputed" not in timing


def test_loop_diagnostics_count_exact_and_lagged_iterations(stubbed_loop):
    setup = _run_setup(
        enable_opacity_lagging=True, opacity_recompute_interval=2, iterations=6
    )
    stubbed_loop(setup, [1.0e-2] * 6)
    result = runner._run_atmosphere_model(_config())

    diagnostics = result.diagnostics
    assert diagnostics["enable_opacity_lagging"] is True
    assert diagnostics["opacity_recompute_interval"] == 2
    assert diagnostics["exact_opacity_iterations"] == 4  # 1, 3, 5, and forced 6
    assert diagnostics["lagged_opacity_iterations"] == 2
