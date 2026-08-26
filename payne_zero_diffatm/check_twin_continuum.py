"""Per-branch parity checks for the differentiable continuum-opacity twin."""

from __future__ import annotations

import bench.environment as _environment  # noqa: F401

import numpy as np
import torch

from payne_zero_atmosphere.constants import LIGHT_SPEED_NM_PER_S
from payne_zero_atmosphere.config import DEFAULT_OPACITY_FLAGS
from payne_zero_atmosphere.continuum_opacity import (
    build_continuum_atmosphere_state,
    build_opacity_sampling_grid,
    compute_aluminum_neutral_opacity_columns,
    compute_carbon_neutral_opacity_columns,
    compute_continuum_opacity_columns,
    compute_helium_neutral_opacity_columns,
    compute_helium_ionized_opacity_columns,
    compute_hot_metal_opacity_columns,
    compute_iron_neutral_opacity_columns,
    compute_lukewarm_metal_opacity_columns,
    compute_magnesium_neutral_opacity_columns,
    compute_silicon_neutral_opacity_columns,
    compute_heminus_opacity_columns,
    compute_hminus_opacity_columns,
    compute_hydrogen_opacity_columns,
    compute_molecular_hydrogen_ion_opacity_columns,
    compute_molecular_continuum_opacity_columns,
)

from .check_twin_molecules import CASES, build_reference
from .twin_continuum import (
    TwinContinuumState,
    TwinContinuumTables,
    continuum_opacity,
    _aluminum_neutral_absorption,
    _carbon_neutral_absorption,
    _helium_neutral_absorption,
    _helium_ionized_absorption,
    _heminus_absorption,
    _hot_metal_absorption,
    _iron_neutral_absorption,
    _lukewarm_metal_absorption,
    _magnesium_neutral_absorption,
    _silicon_neutral_absorption,
    _hminus_absorption,
    _hydrogen_absorption,
    _molecular_hydrogen_ion_absorption,
    _molecular_continuum_absorption,
)


TOLERANCE = 1.0e-10


def _scaled(got, expected) -> float:
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(got - expected)) / max(np.max(np.abs(expected)), 1e-300))


def _state(setup, population, tables):
    runtime = population.runtime_state
    return TwinContinuumState(
        temperature=torch.as_tensor(setup.atmosphere.temperature).unsqueeze(0),
        mass_density=torch.as_tensor(runtime.mass_density).unsqueeze(0),
        electron_density=torch.as_tensor(runtime.electron_density).unsqueeze(0),
        gas_pressure=torch.as_tensor(runtime.gas_pressure).unsqueeze(0),
        elemental_abundances_by_layer=torch.as_tensor(
            runtime.elemental_abundances_by_layer
        ).unsqueeze(0),
        ion_stage_populations_by_packed_slot=torch.as_tensor(
            runtime.ion_stage_populations_by_packed_slot
        ).unsqueeze(0),
        partition_normalized_populations_by_packed_slot=torch.as_tensor(
            runtime.partition_normalized_populations_by_packed_slot
        ).unsqueeze(0),
        hydrogen_departure_coefficients=torch.as_tensor(
            runtime.hydrogen_departure_coefficients
        ).unsqueeze(0),
        tables=tables,
    )


def main() -> int:
    failures = []
    tables = TwinContinuumTables()
    for labels in CASES:
        setup, _, population = build_reference(labels)
        wavelength, _ = build_opacity_sampling_grid(labels.effective_temperature)
        frequency = LIGHT_SPEED_NM_PER_S / wavelength
        frequency_tensor = torch.as_tensor(frequency)
        reference_state = build_continuum_atmosphere_state(
            setup.atmosphere, population.runtime_state
        )
        twin_state = _state(setup, population, tables)
        with torch.no_grad():
            hydrogen = _hydrogen_absorption(twin_state, frequency_tensor, tables)
            hminus = _hminus_absorption(twin_state, frequency_tensor, tables)
            h2plus = _molecular_hydrogen_ion_absorption(
                twin_state, frequency_tensor, tables
            )
            heminus = _heminus_absorption(twin_state, frequency_tensor, tables)
            hei = _helium_neutral_absorption(twin_state, frequency_tensor, tables)
            heii = _helium_ionized_absorption(twin_state, frequency_tensor, tables)
            aluminum = _aluminum_neutral_absorption(
                twin_state, frequency_tensor, tables
            )
            hot_metal = _hot_metal_absorption(twin_state, frequency_tensor, tables)
            molecular = _molecular_continuum_absorption(twin_state, frequency_tensor, tables)
            carbon = _carbon_neutral_absorption(twin_state, frequency_tensor, tables)
            magnesium = _magnesium_neutral_absorption(twin_state, frequency_tensor, tables)
            silicon = _silicon_neutral_absorption(twin_state, frequency_tensor, tables)
            iron = _iron_neutral_absorption(twin_state, frequency_tensor, tables)
            lukewarm = _lukewarm_metal_absorption(twin_state, frequency_tensor, tables)
            assembled = continuum_opacity(
                twin_state, frequency_tensor, tables=tables,
                flags=DEFAULT_OPACITY_FLAGS, frequency_chunk=5000,
            )
        expected_hydrogen = compute_hydrogen_opacity_columns(reference_state, frequency)
        expected_hminus = compute_hminus_opacity_columns(reference_state, frequency)
        expected_assembled = compute_continuum_opacity_columns(
            reference_state, frequency, opacity_flags=DEFAULT_OPACITY_FLAGS
        )
        checks = {
            "assembled absorption": (assembled[0][0].numpy(), expected_assembled[0]),
            "assembled scattering": (assembled[1][0].numpy(), expected_assembled[1]),
            "assembled source": (assembled[2][0].numpy(), expected_assembled[2]),
            "H absorption": (hydrogen[0][0].numpy(), expected_hydrogen[0]),
            "H source": (hydrogen[1][0].numpy(), expected_hydrogen[1]),
            "H- absorption": (hminus[0][0].numpy(), expected_hminus[0]),
            "H- source": (hminus[1][0].numpy(), expected_hminus[1]),
            "H2+": (h2plus[0].numpy(), compute_molecular_hydrogen_ion_opacity_columns(reference_state, frequency)[0]),
            "He-": (heminus[0].numpy(), compute_heminus_opacity_columns(reference_state, frequency)[0]),
            "He I": (hei[0][0].numpy(), compute_helium_neutral_opacity_columns(reference_state, frequency)[0]),
            "He I source": (hei[1][0].numpy(), compute_helium_neutral_opacity_columns(reference_state, frequency)[1]),
            "He II": (heii[0].numpy(), compute_helium_ionized_opacity_columns(reference_state, frequency)[0]),
            "lukewarm metal": (lukewarm[0].numpy(), compute_lukewarm_metal_opacity_columns(reference_state, frequency)[0]),
            "Al I": (aluminum[0].numpy(), compute_aluminum_neutral_opacity_columns(reference_state, frequency)[0]),
            "hot metal": (hot_metal[0].numpy(), compute_hot_metal_opacity_columns(reference_state, frequency)[0]),
            "molecular": (molecular[0].numpy(), compute_molecular_continuum_opacity_columns(reference_state, frequency)[0]),
            "C I": (carbon[0].numpy(), compute_carbon_neutral_opacity_columns(reference_state, frequency)[0]),
            "Mg I": (magnesium[0].numpy(), compute_magnesium_neutral_opacity_columns(reference_state, frequency)[0]),
            "Si I": (silicon[0].numpy(), compute_silicon_neutral_opacity_columns(reference_state, frequency)[0]),
            "Fe I": (iron[0].numpy(), compute_iron_neutral_opacity_columns(reference_state, frequency)[0]),
        }
        maximum = 0.0
        for name, (got, expected) in checks.items():
            error = _scaled(got, expected)
            maximum = max(maximum, error)
            if error >= TOLERANCE:
                failures.append(f"{labels.slug}:{name}")
                difference = np.abs(np.asarray(got) - np.asarray(expected))
                index = np.unravel_index(np.argmax(difference), difference.shape)
                print(
                    f"  {name}: scaled={error:.3e} index={index} "
                    f"got={np.asarray(got)[index]:.6e} "
                    f"expected={np.asarray(expected)[index]:.6e}"
                )
        print(f"{labels.slug}: implemented branches max_scaled={maximum:.3e}")
    if failures:
        print("FAIL:", ", ".join(failures))
        return 1

    frequency_subset = frequency_tensor[::211][:128]
    temperature = twin_state.temperature.detach().clone().requires_grad_(True)
    partition = (
        twin_state.partition_normalized_populations_by_packed_slot
        .detach().clone().requires_grad_(True)
    )
    gradient_state = TwinContinuumState(
        temperature=temperature,
        mass_density=twin_state.mass_density,
        electron_density=twin_state.electron_density,
        gas_pressure=twin_state.gas_pressure,
        elemental_abundances_by_layer=twin_state.elemental_abundances_by_layer,
        ion_stage_populations_by_packed_slot=(
            twin_state.ion_stage_populations_by_packed_slot
        ),
        partition_normalized_populations_by_packed_slot=partition,
        hydrogen_departure_coefficients=twin_state.hydrogen_departure_coefficients,
        tables=tables,
    )
    gradient_outputs = continuum_opacity(
        gradient_state, frequency_subset, tables=tables,
        flags=DEFAULT_OPACITY_FLAGS,
    )
    scale = gradient_outputs[0].detach().abs().mean().clamp(min=1.0e-300)
    objective = torch.log1p(gradient_outputs[0].abs() / scale).mean()
    temperature_gradient, partition_gradient = torch.autograd.grad(
        objective, (temperature, partition)
    )
    for name, gradient in (
        ("temperature", temperature_gradient),
        ("partition", partition_gradient),
    ):
        if not torch.isfinite(gradient).all() or not bool(torch.any(gradient != 0.0)):
            print(f"FAIL: non-finite or zero {name} gradient")
            return 1

    def duplicate(value):
        return torch.cat([value, value], dim=0)

    batch_state = TwinContinuumState(
        temperature=duplicate(twin_state.temperature),
        mass_density=duplicate(twin_state.mass_density),
        electron_density=duplicate(twin_state.electron_density),
        gas_pressure=duplicate(twin_state.gas_pressure),
        elemental_abundances_by_layer=duplicate(
            twin_state.elemental_abundances_by_layer
        ),
        ion_stage_populations_by_packed_slot=duplicate(
            twin_state.ion_stage_populations_by_packed_slot
        ),
        partition_normalized_populations_by_packed_slot=duplicate(
            twin_state.partition_normalized_populations_by_packed_slot
        ),
        hydrogen_departure_coefficients=duplicate(
            twin_state.hydrogen_departure_coefficients
        ),
        tables=tables,
    )
    with torch.no_grad():
        batch_outputs = continuum_opacity(
            batch_state, frequency_subset, tables=tables,
            flags=DEFAULT_OPACITY_FLAGS,
        )
    batch_error = max(
        float(torch.max(torch.abs(output[0] - output[1])))
        for output in batch_outputs
    )
    if batch_error != 0.0:
        print(f"FAIL: batch invariance absolute error={batch_error:.3e}")
        return 1
    print(
        "PASS: assembled gradients finite/nonzero; "
        f"batch invariance absolute error={batch_error:.1e}"
    )
    print(f"PASS: implemented continuum branches match to {TOLERANCE:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
