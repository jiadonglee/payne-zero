"""Validate the operational correction/remap template at iteration one."""

from __future__ import annotations

import bench.environment as _environment  # noqa: F401

from pathlib import Path

import numpy as np
import torch

from bench.run_reference import (
    PRODUCTION_INITIALIZER_JITTER_SCALE,
    PRODUCTION_INITIALIZER_SEED,
    _solver_config,
)
from payne_zero_atmosphere.runner import initialize_iteration_carry, run_single_iteration
from payne_zero_atmosphere.run_setup import resolve_run_setup
from payne_zero_atmosphere.warm_start import (
    deterministic_initializer_labels,
    emulator_warm_start_model,
)

from .check_twin_transfer import SUN
from .twin_correction import (
    apply_reference_correction_template,
    load_reference_correction_template,
    reference_correction_template_from_iteration,
    save_reference_correction_template,
)


OUTPUT = Path("runs/twin_correction_templates/sun_iter1.npz")


def build_iteration(labels=SUN):
    initializer = deterministic_initializer_labels(
        **labels.as_kwargs(), max_trials=1, seed=PRODUCTION_INITIALIZER_SEED,
        jitter_scale=PRODUCTION_INITIALIZER_JITTER_SCALE, device="cpu",
    )[0]
    atmosphere, _ = emulator_warm_start_model(
        **labels.as_kwargs(), device="cpu", initializer_label=initializer
    )
    config = _solver_config(
        atmosphere, iterations_per_trial=1, structured_atmosphere_path=None,
        debug_state_path=None,
    )
    setup = resolve_run_setup(config)
    return run_single_iteration(config, setup, initialize_iteration_carry(setup), 1)


def _scaled(got, expected) -> float:
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(got - expected))) / max(
        float(np.max(np.abs(expected))), 1.0e-300
    )


def main() -> int:
    iteration = build_iteration()
    template = reference_correction_template_from_iteration(iteration)
    result = apply_reference_correction_template(
        template.input_temperature,
        template.input_column_mass,
        template.template_integrated_flux,
        template,
    )
    fields = {
        "temperature": (result.temperature[0], template.output_temperature),
        "column_mass": (result.column_mass[0], template.output_column_mass),
        "flux_error": (result.flux_error_percent[0], template.flux_error_percent),
    }
    errors = {name: _scaled(got, expected) for name, (got, expected) in fields.items()}
    if max(errors.values()) > 1.0e-12:
        print("FAIL:", errors)
        return 1

    save_reference_correction_template(template, OUTPUT)
    loaded = load_reference_correction_template(OUTPUT)
    if _scaled(loaded.output_temperature, template.output_temperature) != 0.0:
        print("FAIL: correction template persistence changed values")
        return 1

    temperature = template.input_temperature.clone().requires_grad_(True)
    flux = template.template_integrated_flux.clone().requires_grad_(True)
    differentiable = apply_reference_correction_template(
        temperature, template.input_column_mass, flux, template
    )
    differentiable.temperature.mean().backward()
    gradient_ok = all(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and bool(torch.any(gradient != 0.0))
        for gradient in (temperature.grad, flux.grad)
    )
    if not gradient_ok:
        print("FAIL: correction gradients are non-finite or zero")
        return 1

    batch = apply_reference_correction_template(
        template.input_temperature.repeat(2, 1),
        template.input_column_mass.repeat(2, 1),
        template.template_integrated_flux.repeat(2, 1),
        template,
    )
    batch_error = float(torch.max(torch.abs(batch.temperature[0] - batch.temperature[1])))
    if batch_error != 0.0:
        print(f"FAIL: batch error={batch_error:.3e}")
        return 1
    print(
        f"PASS: exact template errors={errors}, gradients finite/nonzero, "
        f"batch={batch_error:.1e}, template={OUTPUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
