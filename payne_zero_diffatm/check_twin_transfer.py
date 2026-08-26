"""Check the torch transfer twin against the classical numba kernel.

The reference target is built by running the production iteration-1 pipeline
for the sun trace star (5777/4.44/0/0/1.0): the jittered warm start from
``emulator_warm_start_model`` (same seed/policy as ``bench/run_reference.py``)
feeds ``prepare_population_state`` / ``prepare_opacity_state`` /
``accumulate_transfer_state`` (``runner.py:1057-1215``), and the twin is fed
the *same* opacity-state arrays. Every accumulator is then compared per layer.

What separates the twin from the reference, by construction:

1. **float32 grid operators.** The 51-point source iteration runs in float32
   in both, but torch's 51-element dot reductions are not ordered like the
   reference's ``np.dot`` — a few times 1e-7 relative on the grid source.
2. **Sweep count.** The reference exits its source iterations at a 1e-5
   relative-error criterion; the twin freezes each row at the same criterion
   (capped at 51 sweeps), so this should not contribute.
3. **Frequency-sum reassociation.** The reference accumulates chunked and
   serial; the twin sums the full frequency axis in one torch reduction —
   ~1e-16 relative.

The trace ``runs/twin_traces/<sun>/iter_1/.../debug_state.npz`` holds the same
accumulators from an independent production run and is reported as a
cross-check (its numba chunking may regroup the reductions at the ulp level).

Run::

    PYTHONPATH=. python -m payne_zero_diffatm.check_twin_transfer

First run pays the numba compile of the opacity+transfer kernels (minutes;
cached afterwards in ``.cache/`` / ``__pycache__``).
"""

from __future__ import annotations

# Must precede any Numba import. See bench/environment.py.
import bench.environment as _environment  # noqa: F401

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from bench.labels import StellarLabels
from bench.run_reference import (
    PRODUCTION_INITIALIZER_JITTER_SCALE,
    PRODUCTION_INITIALIZER_SEED,
    _solver_config,
)
from payne_zero_atmosphere.run_setup import resolve_run_setup
from payne_zero_atmosphere.runner import (
    accumulate_transfer_state,
    initialize_iteration_carry,
    prepare_opacity_state,
    prepare_population_state,
)
from payne_zero_atmosphere.warm_start import (
    deterministic_initializer_labels,
    emulator_warm_start_model,
)

from .twin_transfer import load_transfer_tables, transfer_moments


SUN = StellarLabels(5777.0, 4.44, 0.0, 0.0, 1.0)
COOL_DWARF = StellarLabels(4500.0, 4.7, -0.5, 0.2, 1.0)
HARD_REGION = StellarLabels(9000.0, 2.0, -1.0, 0.2, 2.0)
SUN_TRACE_DEBUG_STATE = (
    Path("runs/twin_traces")
    / SUN.slug
    / "iter_1"
    / SUN.slug
    / "trial_00"
    / "debug_state.npz"
)

# The twin should reproduce the classical accumulators to the float32-operator
# level, far below this bound (the solve's own tolerance is ~1e-3 in flux).
RELATIVE_TOLERANCE = 1.0e-3
# Batch invariance: fp64 accumulation plus row-local fp32 dots, so only
# reassociation-level noise is legitimate.
BATCH_TOLERANCE = 1.0e-12


def build_iteration_one_reference(labels: StellarLabels = SUN):
    """Iteration-1 population/opacity/transfer state for the sun trace star.

    Mirrors the ``carry.remapped is None`` branch of ``run_single_iteration``
    (``runner.py:291-346``) with the production solver configuration.
    """

    initializer_labels = deterministic_initializer_labels(
        **labels.as_kwargs(),
        max_trials=1,
        seed=PRODUCTION_INITIALIZER_SEED,
        jitter_scale=PRODUCTION_INITIALIZER_JITTER_SCALE,
        device="cpu",
    )
    warm_start_atmosphere, _deck = emulator_warm_start_model(
        **labels.as_kwargs(),
        device="cpu",
        initializer_label=initializer_labels[0],
    )
    config = _solver_config(
        warm_start_atmosphere,
        iterations_per_trial=1,
        structured_atmosphere_path=None,
        debug_state_path=None,
    )
    setup = resolve_run_setup(config)
    carry = initialize_iteration_carry(setup)
    iteration_setup = replace(
        setup,
        atmosphere=setup.atmosphere,
        surface_radiation_pressure_constant=(
            carry.previous_surface_radiation_pressure_constant
        ),
    )
    temperature_iteration_index = carry.iteration_itemp + 1
    population = prepare_population_state(
        config,
        temperature_iteration_index=temperature_iteration_index,
        setup=iteration_setup,
        molecular_thermal_energy_erg=carry.molecular_thermal_energy_reference,
    )
    opacity = prepare_opacity_state(
        config,
        population_state=population,
        temperature_iteration_index=temperature_iteration_index,
        rosseland_table=carry.previous_rosseland_table,
        selected_line_catalog=carry.selected_line_catalog,
        transition_line_catalog=carry.transition_line_catalog,
    )
    transfer = accumulate_transfer_state(opacity)
    return setup, opacity, transfer


def reference_accumulators(transfer) -> dict[str, np.ndarray]:
    """The raw mode-2 accumulators of the classical accumulation."""

    pressure = transfer.radiative_pressure_state
    correction = transfer.temperature_correction_state
    return {
        "rosseland_accumulator": np.asarray(transfer.rosseland_accumulator),
        "integrated_eddington_flux": np.asarray(pressure.integrated_eddington_flux),
        "radiation_energy_density": np.asarray(pressure.radiation_energy_density),
        "radiative_acceleration": np.asarray(pressure.radiative_acceleration),
        "surface_second_moment": np.atleast_1d(
            pressure.surface_radiation_pressure_constant
        ),
        "mean_intensity_minus_source_integral": np.asarray(
            correction.mean_intensity_minus_source_integral
        ),
        "absorption_heating_derivative": np.asarray(
            correction.absorption_heating_derivative
        ),
        "diagonal_lambda_accumulator": np.asarray(
            correction.diagonal_lambda_accumulator
        ),
        "temperature_correction_integrated_eddington_flux": np.asarray(
            correction.integrated_eddington_flux
        ),
    }


def twin_inputs(setup, opacity, *, temperature_scale: float = 1.0) -> dict:
    """Pack an OpacityState into the twin's (star, layer, freq) layout."""

    def layers_freq(array, dtype=torch.float64) -> torch.Tensor:
        return torch.as_tensor(
            np.ascontiguousarray(np.asarray(array)), dtype=dtype
        ).unsqueeze(0)

    atmosphere = setup.atmosphere
    temperature = torch.as_tensor(
        np.asarray(atmosphere.temperature) * temperature_scale, dtype=torch.float64
    ).unsqueeze(0)
    h_over_kt = torch.as_tensor(
        np.asarray(atmosphere.h_over_kt) / temperature_scale, dtype=torch.float64
    ).unsqueeze(0)
    effective_temperature = torch.tensor(
        [setup.effective_temperature], dtype=torch.float64
    )
    target = (
        5.6697e-5 / 12.5664 * effective_temperature**4
    )
    return {
        "continuum_absorption": layers_freq(opacity.continuum_absorption),
        "continuum_scattering": layers_freq(opacity.continuum_scattering),
        "continuum_source_or_planck": layers_freq(opacity.continuum_source),
        "line_opacity": layers_freq(
            opacity.line_opacity.line_mass_absorption_coefficient
        ),
        "column_mass": torch.as_tensor(
            np.asarray(atmosphere.column_mass), dtype=torch.float64
        ).unsqueeze(0),
        "temperature": temperature,
        "frequency_hz": torch.as_tensor(
            np.asarray(opacity.opacity_frequency_hz), dtype=torch.float64
        ),
        "frequency_weights": torch.as_tensor(
            np.asarray(opacity.frequency_weights), dtype=torch.float64
        ),
        "h_over_kt": h_over_kt,
        "effective_temperature": effective_temperature,
        "target_integrated_eddington_flux": target,
    }


def _relative(got: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    """(max pointwise relative, max absolute scaled by max|expected|)."""

    difference = np.abs(got - expected)
    scale = float(np.max(np.abs(expected)))
    pointwise = difference / np.maximum(np.abs(expected), 1.0e-300)
    return float(np.max(pointwise)), float(np.max(difference) / max(scale, 1.0e-300))


def check_against_reference(inputs, expected) -> dict[str, tuple[float, float]]:
    tables = load_transfer_tables()
    with torch.no_grad():
        result = transfer_moments(tables=tables, **inputs)
    worst: dict[str, tuple[float, float]] = {}
    for field, reference in expected.items():
        got = getattr(result, field)[0].detach().numpy()
        worst[field] = _relative(got, np.asarray(reference, dtype=np.float64))
    return worst


def check_trace_cross_check(expected) -> dict[str, tuple[float, float]] | None:
    """Live reference vs the independently captured iteration-1 trace."""

    if not SUN_TRACE_DEBUG_STATE.exists():
        return None
    trace = np.load(SUN_TRACE_DEBUG_STATE)
    fields = {
        "integrated_eddington_flux": "integrated_eddington_flux",
        "mean_intensity_minus_source_integral": "mean_intensity_minus_source_integral",
        "absorption_heating_derivative": "absorption_heating_derivative",
        "diagonal_lambda_accumulator": "diagonal_lambda_accumulator",
    }
    worst: dict[str, tuple[float, float]] = {}
    for accumulator, trace_key in fields.items():
        worst[accumulator] = _relative(
            np.asarray(expected[accumulator], dtype=np.float64),
            np.asarray(trace[trace_key], dtype=np.float64),
        )
    return worst


def check_batch_invariance(setup, opacity) -> float:
    """A star's accumulators must not depend on batch neighbours."""

    sun = twin_inputs(setup, opacity)
    perturbed = twin_inputs(setup, opacity, temperature_scale=1.01)

    def merge(first, second):
        merged = {}
        for key in first:
            if key in ("frequency_hz", "frequency_weights"):
                merged[key] = first[key]
            elif key in ("effective_temperature", "target_integrated_eddington_flux"):
                merged[key] = torch.cat([first[key], second[key]])
            else:
                merged[key] = torch.cat([first[key], second[key]], dim=0)
        return merged

    tables = load_transfer_tables()
    with torch.no_grad():
        batched = transfer_moments(tables=tables, **merge(sun, perturbed))
        single = transfer_moments(tables=tables, **sun)
    worst = 0.0
    for field in (
        "rosseland_accumulator",
        "integrated_eddington_flux",
        "mean_intensity_minus_source_integral",
        "absorption_heating_derivative",
        "diagonal_lambda_accumulator",
    ):
        got = getattr(batched, field)[0].numpy()
        want = getattr(single, field)[0].numpy()
        _, scaled = _relative(got, want)
        worst = max(worst, scaled)
    return worst


def check_gradients(setup, opacity) -> tuple[bool, float]:
    """Finite, nonzero temperature gradients on a frequency subset."""

    inputs = twin_inputs(setup, opacity)
    stride = max(1, inputs["frequency_hz"].shape[0] // 64)
    subset = slice(0, None, stride)
    for key in (
        "continuum_absorption",
        "continuum_scattering",
        "continuum_source_or_planck",
        "line_opacity",
    ):
        inputs[key] = inputs[key][:, :, subset]
    inputs["frequency_hz"] = inputs["frequency_hz"][subset]
    inputs["frequency_weights"] = inputs["frequency_weights"][subset]
    inputs["temperature"] = inputs["temperature"].requires_grad_(True)
    result = transfer_moments(tables=load_transfer_tables(), **inputs)
    loss = (
        result.integrated_eddington_flux.log().mean()
        + result.radiation_energy_density.log().mean()
        + result.rosseland_accumulator.log().mean()
        + result.diagonal_lambda_accumulator.abs().mean()
    )
    loss.backward()
    gradient = inputs["temperature"].grad
    norm = float(gradient.abs().max())
    ok = bool(torch.isfinite(gradient).all()) and norm > 0.0
    return ok, norm


def check_operator_dtype(inputs, expected) -> float:
    """fp64 grid operators vs the fp32 default (context, not a failure)."""

    with torch.no_grad():
        fp64 = transfer_moments(
            tables=load_transfer_tables(operator_dtype=torch.float64), **inputs
        )
    _, scaled = _relative(
        fp64.integrated_eddington_flux[0].numpy(),
        np.asarray(expected["integrated_eddington_flux"], dtype=np.float64),
    )
    return scaled


def check_sweep_count(inputs, expected, sweeps: int) -> float:
    """Twin at a reduced sweep cap vs the reference (context)."""

    with torch.no_grad():
        result = transfer_moments(
            tables=load_transfer_tables(), sweeps=sweeps, **inputs
        )
    _, scaled = _relative(
        result.integrated_eddington_flux[0].numpy(),
        np.asarray(expected["integrated_eddington_flux"], dtype=np.float64),
    )
    return scaled


def main() -> int:
    failures = []

    print("building iteration-1 reference state (numba compiles on first run)")
    setup, opacity, transfer = build_iteration_one_reference()
    layer_count, frequency_count = opacity.continuum_absorption.shape
    print(
        f"   layers={layer_count} frequencies={frequency_count} "
        f"teff={setup.effective_temperature:g}"
    )
    expected = reference_accumulators(transfer)
    inputs = twin_inputs(setup, opacity)

    print()
    print("1. twin vs classical accumulators, per layer (fp32 operators)")
    for field, (pointwise, scaled) in check_against_reference(inputs, expected).items():
        flag = "" if scaled < RELATIVE_TOLERANCE else "  <-- FAIL"
        print(f"   {field:44s} rel={pointwise:.3e} scaled={scaled:.3e}{flag}")
        if not scaled < RELATIVE_TOLERANCE:
            failures.append(f"reference:{field}")

    print()
    print("2. live reference vs captured iter-1 trace (cross-check, ulp-level)")
    trace = check_trace_cross_check(expected)
    if trace is None:
        print(f"   (trace not found: {SUN_TRACE_DEBUG_STATE})")
    else:
        for field, (pointwise, scaled) in trace.items():
            print(f"   {field:44s} rel={pointwise:.3e} scaled={scaled:.3e}")

    print()
    print("3. batch invariance (sun + perturbed sun batched vs sun alone)")
    batch = check_batch_invariance(setup, opacity)
    flag = "" if batch < BATCH_TOLERANCE else "  <-- FAIL"
    print(f"   max scaled discrepancy {batch:.3e}{flag}")
    if not batch < BATCH_TOLERANCE:
        failures.append("batch")

    print()
    print("4. gradients reach the temperature profile")
    ok, norm = check_gradients(setup, opacity)
    print(f"   finite={ok}  max|d accumulators / d temperature|={norm:.4g}")
    if not ok:
        failures.append("gradients")

    print()
    print("5. context: fp64 operators and reduced sweeps vs the same reference")
    fp64_scaled = check_operator_dtype(inputs, expected)
    print(f"   fp64 operators, integrated_eddington_flux scaled={fp64_scaled:.3e}")
    for sweeps in (8, 16):
        scaled = check_sweep_count(inputs, expected, sweeps)
        print(f"   sweeps={sweeps:<3d} integrated_eddington_flux scaled={scaled:.3e}")

    print()
    print("6. cool-dwarf and hard-region trace stars")
    for labels in (COOL_DWARF, HARD_REGION):
        case_setup, case_opacity, case_transfer = build_iteration_one_reference(labels)
        case_worst = check_against_reference(
            twin_inputs(case_setup, case_opacity), reference_accumulators(case_transfer)
        )
        maximum = max(scaled for _, scaled in case_worst.values())
        flag = "" if maximum < RELATIVE_TOLERANCE else "  <-- FAIL"
        print(f"   {labels.slug:48s} max scaled={maximum:.3e}{flag}")
        if not maximum < RELATIVE_TOLERANCE:
            failures.append(f"reference:{labels.slug}")

    print()
    if failures:
        print("FAIL:", ", ".join(failures))
        return 1
    print(
        f"PASS: accumulators match the classical kernel to {RELATIVE_TOLERANCE:g}, "
        "batch invariant, gradients finite"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
