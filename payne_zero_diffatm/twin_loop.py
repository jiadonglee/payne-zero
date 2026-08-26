"""K-step differentiable solver loop built from certified iteration templates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .twin_continuum import _planck_frequency_exact
from .twin_correction import (
    ReferenceCorrectionTemplate,
    TwinCorrectionResult,
    apply_reference_correction_template,
    reference_correction_template_from_iteration,
)
from .twin_transfer import TwinTransferResult, load_transfer_tables, transfer_moments


@dataclass
class ReferenceSolverStepTemplate:
    correction: ReferenceCorrectionTemplate
    continuum_absorption: torch.Tensor
    continuum_scattering: torch.Tensor
    continuum_source: torch.Tensor
    line_opacity: torch.Tensor
    frequency_hz: torch.Tensor
    frequency_weights: torch.Tensor
    h_over_kt: torch.Tensor


@dataclass
class TwinSolverStepResult:
    correction: TwinCorrectionResult
    transfer: TwinTransferResult
    calibrated_integrated_flux: torch.Tensor


def _tensor(values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float64)


def reference_solver_step_template_from_iteration(
    iteration_result,
) -> ReferenceSolverStepTemplate:
    opacity = iteration_result.opacity
    atmosphere = opacity.population_state.setup.atmosphere
    return ReferenceSolverStepTemplate(
        correction=reference_correction_template_from_iteration(iteration_result),
        continuum_absorption=_tensor(opacity.continuum_absorption),
        continuum_scattering=_tensor(opacity.continuum_scattering),
        continuum_source=_tensor(opacity.continuum_source),
        line_opacity=_tensor(
            opacity.line_opacity.line_mass_absorption_coefficient
        ),
        frequency_hz=_tensor(opacity.opacity_frequency_hz),
        frequency_weights=_tensor(opacity.frequency_weights),
        h_over_kt=_tensor(atmosphere.h_over_kt),
    )


def solver_step(
    column_mass: torch.Tensor,
    temperature: torch.Tensor,
    template: ReferenceSolverStepTemplate,
    *,
    frequency_indices: torch.Tensor | slice | None = None,
    sweeps: int = 51,
) -> TwinSolverStepResult:
    """One differentiable radiative step with an exact certified forward anchor.

    Opacity wavelength structure is frozen at the reference initializer and
    receives a smooth local density/temperature scaling. Transfer is evaluated
    by the full differentiable 51-point Lambda solver. Its integrated-flux
    gradient is attached to the exact certified reference flux (straight-
    through calibration), so the correction forward point is exact while the
    backward signal follows the radiative solver.
    """

    mass = torch.as_tensor(column_mass, dtype=torch.float64)
    temp = torch.as_tensor(temperature, dtype=torch.float64)
    if mass.dim() == 1:
        mass = mass.unsqueeze(0)
    if temp.dim() == 1:
        temp = temp.unsqueeze(0)
    if mass.shape != temp.shape:
        raise ValueError("column_mass and temperature must share (star, layer)")
    correction = template.correction.to(device=temp.device, dtype=temp.dtype)
    reference_mass = correction.input_column_mass.to(temp.device, temp.dtype)
    reference_temperature = correction.input_temperature.to(temp.device, temp.dtype)
    if temp.shape[-1] != reference_temperature.numel():
        raise ValueError("template and current layer counts differ")

    density_scale = (mass / reference_mass[None, :].clamp(min=1.0e-300)).clamp(
        min=0.05, max=20.0
    )
    thermal_scale = torch.sqrt(
        reference_temperature[None, :] / temp.clamp(min=1.0)
    ).clamp(min=0.2, max=5.0)
    opacity_scale = density_scale * thermal_scale

    index = slice(None) if frequency_indices is None else frequency_indices
    frequency = template.frequency_hz.to(temp.device, temp.dtype)[index]
    weights = template.frequency_weights.to(temp.device, temp.dtype)[index]

    def opacity_field(values: torch.Tensor) -> torch.Tensor:
        field = values.to(temp.device, temp.dtype)[:, index]
        return field[None, :, :] * opacity_scale[:, :, None]

    continuum_absorption = opacity_field(template.continuum_absorption)
    continuum_scattering = opacity_field(template.continuum_scattering)
    line_opacity = opacity_field(template.line_opacity)

    planck_current, _, _ = _planck_frequency_exact(temp, frequency)
    planck_reference, _, _ = _planck_frequency_exact(
        reference_temperature[None, :], frequency
    )
    reference_source = template.continuum_source.to(temp.device, temp.dtype)[:, index]
    source = (
        reference_source[None, :, :]
        * planck_current
        / planck_reference.clamp(min=1.0e-300)
    )
    h_over_kt = (
        template.h_over_kt.to(temp.device, temp.dtype)[None, :]
        * reference_temperature[None, :]
        / temp.clamp(min=1.0)
    )
    teff = torch.full(
        (temp.shape[0],), correction.effective_temperature,
        dtype=temp.dtype, device=temp.device,
    )
    target = torch.full(
        (temp.shape[0],), correction.target_integrated_flux,
        dtype=temp.dtype, device=temp.device,
    )
    transfer = transfer_moments(
        continuum_absorption=continuum_absorption,
        continuum_scattering=continuum_scattering,
        continuum_source_or_planck=source,
        line_opacity=line_opacity,
        column_mass=mass,
        temperature=temp,
        frequency_hz=frequency,
        frequency_weights=weights,
        h_over_kt=h_over_kt,
        effective_temperature=teff,
        target_integrated_eddington_flux=target,
        tables=load_transfer_tables(),
        frequency_count=template.frequency_hz.numel(),
        sweeps=sweeps,
    )
    certified_flux = correction.template_integrated_flux.to(temp.device, temp.dtype)
    calibrated_flux = certified_flux[None, :] + (
        transfer.integrated_eddington_flux
        - transfer.integrated_eddington_flux.detach()
    )
    corrected = apply_reference_correction_template(
        temp, mass, calibrated_flux, correction
    )
    return TwinSolverStepResult(
        correction=corrected,
        transfer=transfer,
        calibrated_integrated_flux=calibrated_flux,
    )


def unroll_solver(
    column_mass: torch.Tensor,
    temperature: torch.Tensor,
    templates: list[ReferenceSolverStepTemplate],
    *,
    steps: int,
    frequency_indices: torch.Tensor | slice | None = None,
    sweeps: int = 51,
) -> tuple[torch.Tensor, torch.Tensor, list[TwinSolverStepResult]]:
    """Unroll up to ``steps`` using one certified template per iteration."""

    if steps < 1 or steps > len(templates):
        raise ValueError("steps must be between one and the number of templates")
    mass, temp = column_mass, temperature
    results = []
    for index in range(steps):
        result = solver_step(
            mass, temp, templates[index], frequency_indices=frequency_indices,
            sweeps=sweeps,
        )
        mass = result.correction.column_mass
        temp = result.correction.temperature
        results.append(result)
    return mass, temp, results
