"""Validate the torch molecular-equilibrium twin against iteration-1 reference states."""

from __future__ import annotations

import bench.environment as _environment  # noqa: F401

import numpy as np
import torch

from bench.labels import StellarLabels
from bench.run_reference import (
    PRODUCTION_INITIALIZER_JITTER_SCALE,
    PRODUCTION_INITIALIZER_SEED,
    _solver_config,
)
from payne_zero_atmosphere.run_setup import resolve_run_setup
from payne_zero_atmosphere.runner import prepare_population_state
from payne_zero_atmosphere.runtime_state import build_runtime_state
from payne_zero_atmosphere.warm_start import (
    deterministic_initializer_labels,
    emulator_warm_start_model,
)

from .twin_molecules import TwinMoleculeTables, solve_molecular_equilibrium


CASES = (
    StellarLabels(5777.0, 4.44, 0.0, 0.0, 1.0),
    StellarLabels(4500.0, 4.7, -0.5, 0.2, 1.0),
    StellarLabels(9000.0, 2.0, -1.0, 0.2, 2.0),
)
RELATIVE_TOLERANCE = 1.0e-6


def build_reference(labels: StellarLabels):
    initializer_labels = deterministic_initializer_labels(
        **labels.as_kwargs(), max_trials=1, seed=PRODUCTION_INITIALIZER_SEED,
        jitter_scale=PRODUCTION_INITIALIZER_JITTER_SCALE, device="cpu"
    )
    atmosphere, _ = emulator_warm_start_model(
        **labels.as_kwargs(), device="cpu", initializer_label=initializer_labels[0]
    )
    config = _solver_config(
        atmosphere, iterations_per_trial=1, structured_atmosphere_path=None,
        debug_state_path=None
    )
    setup = resolve_run_setup(config)
    seed = build_runtime_state(setup.atmosphere)
    population = prepare_population_state(config, setup=setup)
    if population.molecular_state is None:
        raise RuntimeError("reference population did not run molecular equilibrium")
    return setup, seed, population


def scaled_error(got, expected) -> float:
    got = np.asarray(got, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(got - expected)) / max(np.max(np.abs(expected)), 1e-300))


def main() -> int:
    tables = TwinMoleculeTables()
    failures = []
    cold_inputs = None
    fields = (
        "electron_density", "total_nuclei_number_density", "mass_density",
        "molecular_populations", "partition_normalized_molecular_populations",
        "molecular_equation_densities",
    )
    for labels in CASES:
        setup, seed, reference = build_reference(labels)
        abundances = seed.elemental_abundances_by_layer[0]
        inputs = {
            "temperature": torch.as_tensor(setup.atmosphere.temperature).unsqueeze(0),
            "gas_pressure": torch.as_tensor(seed.gas_pressure).unsqueeze(0),
            "electron_density": torch.as_tensor(seed.electron_density).unsqueeze(0),
            "abundances": torch.as_tensor(abundances),
            "table": tables,
        }
        with torch.no_grad():
            twin = solve_molecular_equilibrium(**inputs)
        molecular = reference.molecular_state
        expected = {
            "electron_density": reference.runtime_state.electron_density,
            "total_nuclei_number_density": reference.runtime_state.total_nuclei_number_density,
            "mass_density": reference.runtime_state.mass_density,
            "molecular_populations": molecular.molecular_populations,
            "partition_normalized_molecular_populations": molecular.partition_normalized_molecular_populations,
            "molecular_equation_densities": molecular.molecular_equation_densities,
        }
        errors = {field: scaled_error(getattr(twin, field)[0].numpy(), expected[field]) for field in fields}
        maximum = max(errors.values())
        unconverged = int((~twin.converged).sum())
        flag = "" if maximum < RELATIVE_TOLERANCE and unconverged == 0 else "  <-- FAIL"
        print(f"{labels.slug}: max_scaled={maximum:.3e} unconverged={unconverged}{flag}")
        for field, error in errors.items():
            print(f"  {field:48s} {error:.3e}")
        if flag:
            failures.append(labels.slug)
        if labels == CASES[1]:
            cold_inputs = inputs

    assert cold_inputs is not None
    temperature = cold_inputs["temperature"].clone().requires_grad_(True)
    gradient_inputs = dict(cold_inputs, temperature=temperature, max_iterations=40)
    result = solve_molecular_equilibrium(**gradient_inputs)
    loss = result.electron_density.log().mean() + result.mass_density.log().mean()
    loss.backward()
    gradient_ok = bool(torch.isfinite(temperature.grad).all()) and bool(
        (temperature.grad != 0).any()
    )
    print(f"gradient finite/nonzero={gradient_ok}")
    if not gradient_ok:
        failures.append("gradient")
    if failures:
        print("FAIL:", ", ".join(failures))
        return 1
    print(f"PASS: molecular twin matches reference to {RELATIVE_TOLERANCE:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
