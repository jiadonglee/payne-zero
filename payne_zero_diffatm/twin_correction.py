"""Operational differentiable temperature-correction template for K=1 training.

The certified solver supplies the exact iteration-1 correction/remap at a
frozen initializer state. This module retains that exact forward point and
adds a local differentiable response to changes in temperature, column mass,
and integrated flux. Real-solver evaluation remains the authority outside the
local training neighbourhood.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class ReferenceCorrectionTemplate:
    input_temperature: torch.Tensor
    input_column_mass: torch.Tensor
    output_temperature: torch.Tensor
    output_column_mass: torch.Tensor
    output_gas_pressure: torch.Tensor
    output_rosseland_opacity: torch.Tensor
    output_radiative_acceleration: torch.Tensor
    output_integrated_radiation_pressure: torch.Tensor
    template_integrated_flux: torch.Tensor
    convective_flux: torch.Tensor
    flux_error_percent: torch.Tensor
    temperature_correction: torch.Tensor
    target_integrated_flux: float
    effective_temperature: float

    def to(self, *, device=None, dtype=torch.float64):
        fields = {}
        for name, value in vars(self).items():
            fields[name] = (
                value.to(device=device, dtype=dtype)
                if isinstance(value, torch.Tensor)
                else value
            )
        return ReferenceCorrectionTemplate(**fields)


@dataclass
class TwinCorrectionResult:
    temperature: torch.Tensor
    column_mass: torch.Tensor
    gas_pressure: torch.Tensor
    rosseland_opacity: torch.Tensor
    radiative_acceleration: torch.Tensor
    integrated_radiation_pressure: torch.Tensor
    flux_error_percent: torch.Tensor
    temperature_correction: torch.Tensor


def _tensor(values) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values), dtype=torch.float64)


def reference_correction_template_from_iteration(
    iteration_result,
) -> ReferenceCorrectionTemplate:
    """Extract an exact correction/remap template from ``SingleIterationResult``."""

    opacity = iteration_result.opacity
    source = opacity.population_state.setup.atmosphere
    remapped = iteration_result.remapped
    output = remapped.atmosphere
    correction = remapped.finalization.temperature_correction_result
    target = (
        5.6697e-5 / 12.5664
        * float(opacity.population_state.setup.effective_temperature) ** 4
    )
    return ReferenceCorrectionTemplate(
        input_temperature=_tensor(source.temperature),
        input_column_mass=_tensor(source.column_mass),
        output_temperature=_tensor(output.temperature),
        output_column_mass=_tensor(output.column_mass),
        output_gas_pressure=_tensor(output.gas_pressure),
        output_rosseland_opacity=_tensor(output.rosseland_opacity),
        output_radiative_acceleration=_tensor(output.radiative_acceleration),
        output_integrated_radiation_pressure=_tensor(
            remapped.integrated_radiation_pressure
        ),
        template_integrated_flux=_tensor(
            iteration_result.transfer.temperature_correction_state.integrated_eddington_flux
        ),
        convective_flux=_tensor(correction.convective_flux),
        flux_error_percent=_tensor(correction.flux_error_percent),
        temperature_correction=_tensor(correction.temperature_correction),
        target_integrated_flux=float(target),
        effective_temperature=float(opacity.population_state.setup.effective_temperature),
    )


def save_reference_correction_template(
    template: ReferenceCorrectionTemplate, path: Path | str
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: value.detach().cpu().numpy()
        for name, value in vars(template).items()
        if isinstance(value, torch.Tensor)
    }
    arrays["target_integrated_flux"] = np.float64(template.target_integrated_flux)
    arrays["effective_temperature"] = np.float64(template.effective_temperature)
    np.savez_compressed(path, **arrays)
    return path


def load_reference_correction_template(
    path: Path | str, *, device="cpu", dtype=torch.float64
) -> ReferenceCorrectionTemplate:
    with np.load(path, allow_pickle=False) as arrays:
        values = {
            name: torch.as_tensor(np.asarray(arrays[name]), dtype=dtype, device=device)
            for name in (
                "input_temperature", "input_column_mass", "output_temperature",
                "output_column_mass", "output_gas_pressure",
                "output_rosseland_opacity", "output_radiative_acceleration",
                "output_integrated_radiation_pressure", "template_integrated_flux",
                "convective_flux", "flux_error_percent", "temperature_correction",
            )
        }
        return ReferenceCorrectionTemplate(
            **values,
            target_integrated_flux=float(arrays["target_integrated_flux"]),
            effective_temperature=float(arrays["effective_temperature"]),
        )


def apply_reference_correction_template(
    current_temperature: torch.Tensor,
    current_column_mass: torch.Tensor,
    current_integrated_flux: torch.Tensor,
    template: ReferenceCorrectionTemplate,
) -> TwinCorrectionResult:
    """Apply the exact template plus a local flux-response linearization."""

    temperature = torch.as_tensor(current_temperature)
    column_mass = torch.as_tensor(current_column_mass)
    integrated_flux = torch.as_tensor(current_integrated_flux)
    if temperature.dim() == 1:
        temperature = temperature.unsqueeze(0)
    if column_mass.dim() == 1:
        column_mass = column_mass.unsqueeze(0)
    if integrated_flux.dim() == 1:
        integrated_flux = integrated_flux.unsqueeze(0)
    if temperature.shape != column_mass.shape or temperature.shape != integrated_flux.shape:
        raise ValueError("temperature, column mass, and flux must share (star, layer)")
    ref = template.to(device=temperature.device, dtype=temperature.dtype)
    if temperature.shape[-1] != ref.input_temperature.numel():
        raise ValueError("template and current state layer counts differ")

    target = float(ref.target_integrated_flux)
    current_flux_error = (
        (integrated_flux + ref.convective_flux[None, :] - target)
        / max(target, 1.0e-300) * 100.0
    )
    error_delta = current_flux_error - ref.flux_error_percent[None, :]
    safe_error = torch.where(
        ref.flux_error_percent.abs() >= 0.1,
        ref.flux_error_percent,
        torch.where(
            ref.flux_error_percent >= 0.0,
            torch.full_like(ref.flux_error_percent, 0.1),
            torch.full_like(ref.flux_error_percent, -0.1),
        ),
    )
    reference_delta = ref.output_temperature - ref.input_temperature
    secant = reference_delta / safe_error
    fallback = -0.0025 * ref.input_temperature
    response = torch.where(
        ref.flux_error_percent.abs() >= 0.1, secant, fallback
    ).clamp(
        min=-ref.effective_temperature / 25.0,
        max=ref.effective_temperature / 25.0,
    )
    delta_temperature = (
        reference_delta[None, :] + response[None, :] * error_delta
    )
    delta_temperature = delta_temperature.clamp(
        min=-ref.effective_temperature / 12.5,
        max=ref.effective_temperature / 12.5,
    )
    next_temperature = (temperature + delta_temperature).clamp(min=1.0)

    reference_column_ratio = (
        ref.output_column_mass / ref.input_column_mass.clamp(min=1.0e-300)
    )
    next_column_mass = column_mass * reference_column_ratio[None, :]

    def broadcast(field: torch.Tensor) -> torch.Tensor:
        return field[None, :].expand(temperature.shape[0], -1)

    return TwinCorrectionResult(
        temperature=next_temperature,
        column_mass=next_column_mass,
        gas_pressure=broadcast(ref.output_gas_pressure),
        rosseland_opacity=broadcast(ref.output_rosseland_opacity),
        radiative_acceleration=broadcast(ref.output_radiative_acceleration),
        integrated_radiation_pressure=broadcast(
            ref.output_integrated_radiation_pressure
        ),
        flux_error_percent=current_flux_error,
        temperature_correction=delta_temperature,
    )
