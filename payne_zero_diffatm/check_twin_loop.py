"""End-to-end K=1 forward-anchor and gradient smoke check."""

from __future__ import annotations

import bench.environment as _environment  # noqa: F401

import torch

from .check_twin_correction import build_iteration
from .twin_loop import reference_solver_step_template_from_iteration, solver_step


def main() -> int:
    iteration = build_iteration()
    template = reference_solver_step_template_from_iteration(iteration)
    reference = template.correction
    with torch.no_grad():
        full = solver_step(
            reference.input_column_mass, reference.input_temperature, template
        )
    temperature_error = float(
        torch.max(torch.abs(full.correction.temperature[0] - reference.output_temperature))
    ) / float(torch.max(torch.abs(reference.output_temperature)))
    mass_error = float(
        torch.max(torch.abs(full.correction.column_mass[0] - reference.output_column_mass))
    ) / float(torch.max(torch.abs(reference.output_column_mass)))
    if max(temperature_error, mass_error) > 1.0e-12:
        print(f"FAIL: anchored forward T={temperature_error:.3e} m={mass_error:.3e}")
        return 1

    mass = reference.input_column_mass.clone().requires_grad_(True)
    temperature = reference.input_temperature.clone().requires_grad_(True)
    subset = torch.arange(0, template.frequency_hz.numel(), 431)
    result = solver_step(
        mass, temperature, template, frequency_indices=subset, sweeps=12
    )
    loss = (
        result.correction.temperature.log().mean()
        + result.correction.column_mass.log().mean()
        + result.transfer.integrated_eddington_flux.abs().mean()
    )
    loss.backward()
    gradients = (mass.grad, temperature.grad)
    gradient_ok = all(
        gradient is not None
        and bool(torch.isfinite(gradient).all())
        and bool(torch.any(gradient != 0.0))
        for gradient in gradients
    )
    if not gradient_ok:
        print("FAIL: K=1 gradients are non-finite or zero")
        return 1
    print(
        f"PASS: K=1 anchor T={temperature_error:.1e} m={mass_error:.1e}; "
        "end-to-end gradients finite/nonzero"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
