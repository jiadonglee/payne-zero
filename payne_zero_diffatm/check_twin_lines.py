"""Validate the operational reference-template line-opacity backend."""

from __future__ import annotations

import bench.environment as _environment  # noqa: F401

from pathlib import Path

import numpy as np
import torch

from .check_twin_transfer import SUN, build_iteration_one_reference
from .twin_lines import (
    line_opacity_from_template,
    load_reference_line_template,
    reference_line_template_from_opacity_state,
    save_reference_line_template,
)


OUTPUT = Path("runs/twin_line_templates/sun.npz")


def main() -> int:
    _, opacity, _ = build_iteration_one_reference(SUN)
    template = reference_line_template_from_opacity_state(opacity)
    population_widths = template.population_widths.unsqueeze(0)
    got = line_opacity_from_template(population_widths, template)[0]
    expected = torch.as_tensor(
        np.asarray(opacity.line_opacity.line_mass_absorption_coefficient),
        dtype=torch.float64,
    )
    absolute = float(torch.max(torch.abs(got - expected)))
    scaled = absolute / max(float(torch.max(torch.abs(expected))), 1.0e-300)
    if scaled > 1.0e-12:
        print(f"FAIL: template-state line opacity scaled error={scaled:.3e}")
        return 1

    save_reference_line_template(template, OUTPUT)
    reloaded = load_reference_line_template(OUTPUT)
    reload_error = float(
        torch.max(torch.abs(reloaded.line_opacity - template.line_opacity))
    ) / max(float(torch.max(torch.abs(template.line_opacity))), 1.0e-300)
    if reload_error > 1.0e-6:
        print(f"FAIL: persisted float32 template scaled error={reload_error:.3e}")
        return 1

    variable = population_widths.clone().requires_grad_(True)
    result = line_opacity_from_template(variable, template)
    scale = result.detach().abs().mean().clamp(min=1.0e-300)
    torch.log1p(result.abs() / scale).mean().backward()
    gradient_ok = bool(torch.isfinite(variable.grad).all()) and bool(
        torch.any(variable.grad != 0.0)
    )
    if not gradient_ok:
        print("FAIL: line-template gradients are non-finite or zero")
        return 1

    duplicated = torch.cat([population_widths, population_widths], dim=0)
    batch = line_opacity_from_template(duplicated, template)
    batch_error = float(torch.max(torch.abs(batch[0] - batch[1])))
    if batch_error != 0.0:
        print(f"FAIL: batch invariance absolute error={batch_error:.3e}")
        return 1

    perturbation = torch.linspace(
        0.99, 1.01, population_widths.shape[1], dtype=torch.float64
    )[None, :, None]
    changed = line_opacity_from_template(population_widths * perturbation, template)
    response = float(torch.max(torch.abs(changed - got[None, :, :])))
    if response == 0.0:
        print("FAIL: template backend did not respond to a population perturbation")
        return 1

    print(
        f"PASS: exact-state scaled={scaled:.1e}, persisted={reload_error:.3e}, "
        f"batch={batch_error:.1e}, gradient finite/nonzero, template={OUTPUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
