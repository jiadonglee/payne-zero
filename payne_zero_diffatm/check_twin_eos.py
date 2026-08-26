"""Check the twin EOS against the reference population path.

What is compared, and why the trace alone is not enough:

1. **Twin vs reference atomic path, same input atmosphere (the graded check).**
   The ``runs/twin_traces/<slug>/iter_1/`` debug snapshots were written after
   iteration 1's remap *and* with molecules enabled, while the population
   arrays in them were computed on the pre-remap atmosphere through the
   molecular equilibrium. Neither the remap nor the molecules are this
   module's physics. The like-for-like target is therefore produced directly:
   the iteration-1 input atmosphere is rebuilt exactly as
   ``runs/capture_traces.py`` did (same labels, same initializer policy), and
   the reference EOS functions (``build_runtime_state`` →
   ``update_charge_square_density`` → ``populate_all_species`` with
   ``molecules_enabled=False`` → ``update_doppler_line_strength_factors``) are
   run on it. The twin must match that to round-off-level on the continuous
   state and to better than 1e-4 relative on electron and mass density.

2. **Molecular gap (context, not graded).** The same reference chain run with
   molecules enabled (``prepare_population_state``) shows how far the atomic
   fixed point is from the molecule-coupled one — the known gap for the cool
   dwarf, quantified on the electron density.

3. **Twin vs iter_1 trace (context).** Conflates the molecular gap with the
   iteration-1 remap; reported only to bound the end-to-end difference.

4. **Gradients and batching.** Finite, nonzero gradients w.r.t. temperature
   and gas_pressure through the whole module, and batch invariance (a star's
   result must not depend on batch neighbours).

Run::

    PYTHONPATH=. .venv-linux/bin/python -m payne_zero_diffatm.check_twin_eos
"""

from __future__ import annotations

# Must precede any numba/torch import ordering concerns; see bench/environment.py.
from bench import environment as _environment  # noqa: F401

from pathlib import Path

import numpy as np
import torch

from bench.labels import StellarLabels
from bench.run_reference import (
    PRODUCTION_INITIALIZER_JITTER_SCALE,
    PRODUCTION_INITIALIZER_SEED,
)
from payne_zero_atmosphere.config import (
    AtmosphereConfig,
    AtmosphereInput,
    AtmosphereOutput,
)
from payne_zero_atmosphere.doppler import update_doppler_line_strength_factors
from payne_zero_atmosphere.equation_of_state import populate_all_species
from payne_zero_atmosphere.runner import prepare_population_state
from payne_zero_atmosphere.run_setup import (
    initialize_microturbulence,
    standard_rosseland_optical_depth_grid,
)
from payne_zero_atmosphere.runtime_state import (
    build_elemental_abundances_by_layer,
    build_runtime_state,
    update_charge_square_density,
)
from payne_zero_atmosphere.source_catalogs import molecular_equilibrium_catalog_path
from payne_zero_atmosphere.warm_start import (
    deterministic_initializer_labels,
    emulator_warm_start_model,
)

from .twin_eos import TwinEosTables, TwinPopulationState, solve_populations


STARS = [
    StellarLabels(5777.0, 4.44, 0.0, 0.0, 1.0),
    StellarLabels(4500.0, 4.70, -0.50, 0.20, 1.0),
]

TRACE_TEMPLATE = (
    "runs/twin_traces/{slug}/iter_1/{slug}/trial_00/debug_state.npz"
)

# Graded tolerance: electron density and mass density vs the reference atomic
# path. Both sides converge the same damped fixed point to the same 1e-4
# relative-update stop, so they agree far tighter than the stop itself.
DENSITY_TOLERANCE = 1.0e-4
# Populations: max |log10 ratio| over slots where the reference is
# significant. "Same order" is the requirement; the twin lands far inside it.
POPULATION_DEX_TOLERANCE = 0.1


def _warm_start_atmosphere(labels: StellarLabels):
    """The exact iteration-1 input atmosphere from the capture policy."""

    initializer_label = deterministic_initializer_labels(
        **labels.as_kwargs(),
        max_trials=1,
        seed=PRODUCTION_INITIALIZER_SEED,
        jitter_scale=PRODUCTION_INITIALIZER_JITTER_SCALE,
        device="cpu",
    )[0]
    atmosphere, _deck = emulator_warm_start_model(
        **labels.as_kwargs(), device="cpu", initializer_label=initializer_label
    )
    if not np.any(atmosphere.microturbulence > 0.0):
        initialize_microturbulence(
            atmosphere,
            effective_temperature=labels.effective_temperature,
            log_surface_gravity=labels.log_surface_gravity,
            standard_rosseland_optical_depth=standard_rosseland_optical_depth_grid(
                atmosphere.layers
            ),
        )
    return atmosphere


def _reference_atomic(atmosphere):
    """The reference population phase with molecules disabled."""

    state = build_runtime_state(atmosphere)
    update_charge_square_density(
        thermal_energy_erg=atmosphere.thermal_energy_erg, state=state
    )
    populate_all_species(
        temperature_k=atmosphere.temperature,
        thermal_energy_erg=atmosphere.thermal_energy_erg,
        state=state,
        molecules_enabled=False,
        pressure_iteration_enabled=True,
        temperature_iteration_index=1,
        temperature_iteration_cache={},
    )
    fractional_doppler_widths, population_over = (
        update_doppler_line_strength_factors(
            thermal_energy_erg=atmosphere.thermal_energy_erg,
            microturbulence=atmosphere.microturbulence,
            state=state,
        )
    )
    return state, fractional_doppler_widths, population_over


def _reference_molecular(atmosphere):
    """The reference population phase with molecules enabled."""

    config = AtmosphereConfig(
        inputs=AtmosphereInput(
            initial_atmosphere=atmosphere,
            molecules_path=molecular_equilibrium_catalog_path(),
        ),
        outputs=AtmosphereOutput(),
        enable_molecules=True,
    )
    return prepare_population_state(config)


def _max_rel(got: np.ndarray, expected: np.ndarray) -> float:
    # Slots with zero isotope mass carry inf Doppler widths on both sides by
    # construction (doppler.py:44-54); those match by finiteness pattern and
    # are excluded from the relative metric.
    if not np.array_equal(np.isfinite(got), np.isfinite(expected)):
        return float("inf")
    finite = np.isfinite(expected)
    if not np.any(finite):
        return 0.0
    return float(
        np.max(
            np.abs(got[finite] - expected[finite])
            / np.maximum(np.abs(expected[finite]), 1.0e-300)
        )
    )


def _max_dex(got: np.ndarray, expected: np.ndarray) -> float:
    """Largest order-of-magnitude disagreement over significant slots."""

    significant = expected > 1.0e-12 * np.max(expected)
    significant &= got > 0.0
    if not np.any(significant):
        return float("nan")
    return float(
        np.max(np.abs(np.log10(got[significant]) - np.log10(expected[significant])))
    )


def _twin_input(atmospheres, tables: TwinEosTables) -> TwinPopulationState:
    temperature = torch.tensor(
        np.stack([a.temperature for a in atmospheres]), dtype=torch.float64
    )
    gas_pressure = torch.tensor(
        np.stack([a.gas_pressure for a in atmospheres]), dtype=torch.float64
    )
    electron_density = torch.tensor(
        np.stack([a.electron_density for a in atmospheres]), dtype=torch.float64
    )
    microturbulence = torch.tensor(
        np.stack([a.microturbulence for a in atmospheres]), dtype=torch.float64
    )
    abundances = torch.tensor(
        np.stack(
            [build_elemental_abundances_by_layer(a)[0] for a in atmospheres]
        ),
        dtype=torch.float64,
    )
    return solve_populations(
        temperature,
        gas_pressure,
        electron_density,
        abundances,
        microturbulence,
        tables=tables,
    )


def check_against_atomic_reference(
    twin: TwinPopulationState, references
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Graded comparison: twin vs reference atomic path per star."""

    failures: list[str] = []
    report: dict[str, dict[str, float]] = {}
    for index, (labels, (state, doppler, population_over)) in enumerate(
        zip(STARS, references)
    ):
        metrics = {
            "electron_density": _max_rel(
                twin.electron_density[index].numpy(), state.electron_density
            ),
            "total_nuclei_number_density": _max_rel(
                twin.total_nuclei_number_density[index].numpy(),
                state.total_nuclei_number_density,
            ),
            "mass_density": _max_rel(
                twin.mass_density[index].numpy(), state.mass_density
            ),
            "charge_square_density": _max_rel(
                twin.charge_square_density[index].numpy(), state.charge_square_density
            ),
            "mean_nuclear_mass_amu": _max_rel(
                twin.mean_nuclear_mass_amu[index].numpy(), state.mean_nuclear_mass_amu
            ),
            "fractional_doppler_widths": _max_rel(
                twin.fractional_doppler_widths[index].numpy(), doppler
            ),
            "ion_stage_populations (dex)": _max_dex(
                twin.ion_stage_populations_by_packed_slot[index].numpy(),
                state.ion_stage_populations_by_packed_slot,
            ),
            "partition_normalized_pops (dex)": _max_dex(
                twin.partition_normalized_populations_by_packed_slot[index].numpy(),
                state.partition_normalized_populations_by_packed_slot,
            ),
            "population_over_density_width (dex)": _max_dex(
                twin.partition_normalized_population_over_mass_density_and_fractional_doppler_width[
                    index
                ].numpy(),
                population_over,
            ),
        }
        report[labels.slug] = metrics
        for field in (
            "electron_density",
            "total_nuclei_number_density",
            "mass_density",
            "charge_square_density",
        ):
            if not metrics[field] < DENSITY_TOLERANCE:
                failures.append(f"{labels.slug}:{field}")
        for field in (
            "ion_stage_populations (dex)",
            "partition_normalized_pops (dex)",
        ):
            if not metrics[field] < POPULATION_DEX_TOLERANCE:
                failures.append(f"{labels.slug}:{field}")
    return report, failures


def main() -> int:
    print("building iteration-1 input atmospheres (capture policy) ...")
    atmospheres = [_warm_start_atmosphere(labels) for labels in STARS]

    print("running reference atomic EOS (numba) ...")
    references = [_reference_atomic(atmosphere) for atmosphere in atmospheres]

    print("running twin EOS (torch, both stars batched) ...")
    tables = TwinEosTables()
    with torch.no_grad():
        twin = _twin_input(atmospheres, tables)

    failures: list[str] = []

    print()
    print("1. twin vs reference atomic path, max relative difference per array")
    report, check_failures = check_against_atomic_reference(twin, references)
    failures.extend(check_failures)
    for slug, metrics in report.items():
        print(f"   {slug}")
        for field, value in metrics.items():
            limit = (
                DENSITY_TOLERANCE if "(dex)" not in field else POPULATION_DEX_TOLERANCE
            )
            flag = "" if value < limit else "  <-- FAIL"
            print(f"     {field:38s} {value:.3e}{flag}")
    for index, labels in enumerate(STARS):
        unconverged = int((~twin.converged[index]).sum())
        worst_sweeps = int(twin.iterations_used[index].max()) + 1
        print(
            f"   {labels.slug}: twin fixed point unconverged points: {unconverged}, "
            f"max sweeps used: {worst_sweeps}"
        )
        if unconverged:
            failures.append(f"{labels.slug}:unconverged")

    print()
    print("2. molecular gap in the reference itself (context, not graded)")
    for labels, atmosphere, (state, _, _) in zip(STARS, atmospheres, references):
        molecular = _reference_molecular(atmosphere)
        molecular_state = molecular.runtime_state
        gap = _max_rel(
            np.asarray(molecular_state.electron_density),
            np.asarray(state.electron_density),
        )
        gap_nuclei = _max_rel(
            np.asarray(molecular_state.total_nuclei_number_density),
            np.asarray(state.total_nuclei_number_density),
        )
        print(
            f"   {labels.slug}: electron_density {gap:.3e}, "
            f"total_nuclei_number_density {gap_nuclei:.3e} "
            "(molecules-enabled vs atomic reference)"
        )

    print()
    print("3. twin vs iter_1 trace (post-remap + molecules; context only)")
    for index, labels in enumerate(STARS):
        trace_path = Path(TRACE_TEMPLATE.format(slug=labels.slug))
        with np.load(trace_path, allow_pickle=False) as trace:
            ne_diff = _max_rel(
                twin.electron_density[index].numpy(), trace["electron_density"]
            )
            mass_diff = _max_rel(
                twin.mass_density[index].numpy(), trace["mass_density"]
            )
            pop_dex = _max_dex(
                twin.ion_stage_populations_by_packed_slot[index].numpy(),
                trace["ion_stage_populations_by_packed_slot"],
            )
        print(
            f"   {labels.slug}: electron_density {ne_diff:.3e}, "
            f"mass_density {mass_diff:.3e}, ion_stage_populations {pop_dex:.3f} dex"
        )

    print()
    print("4. gradients reach temperature and gas_pressure")
    temperature = torch.tensor(
        np.stack([a.temperature for a in atmospheres]),
        dtype=torch.float64,
        requires_grad=True,
    )
    gas_pressure = torch.tensor(
        np.stack([a.gas_pressure for a in atmospheres]),
        dtype=torch.float64,
        requires_grad=True,
    )
    electron_density = torch.tensor(
        np.stack([a.electron_density for a in atmospheres]), dtype=torch.float64
    )
    microturbulence = torch.tensor(
        np.stack([a.microturbulence for a in atmospheres]), dtype=torch.float64
    )
    abundances = torch.tensor(
        np.stack([build_elemental_abundances_by_layer(a)[0] for a in atmospheres]),
        dtype=torch.float64,
    )
    out = solve_populations(
        temperature,
        gas_pressure,
        electron_density,
        abundances,
        microturbulence,
        tables=tables,
    )
    loss = (
        out.electron_density.log().mean()
        + out.mass_density.log().mean()
        + out.total_nuclei_number_density.clamp(min=1.0e-300).log().mean()
        + out.ion_stage_populations_by_packed_slot.clamp(min=1.0e-300).log().mean()
    )
    loss.backward()
    for name, tensor in (("temperature", temperature), ("gas_pressure", gas_pressure)):
        grad = tensor.grad
        finite = bool(torch.isfinite(grad).all())
        nonzero = bool((grad != 0.0).any())
        print(
            f"   {name:14s} grad finite: {finite}, nonzero: {nonzero}, "
            f"max |grad| {grad.abs().max():.3e}"
        )
        if not (finite and nonzero):
            failures.append(f"gradient:{name}")

    print()
    print("5. batch invariance (single stars vs the pair)")
    singles = [
        _twin_input([atmosphere], tables) for atmosphere in atmospheres
    ]
    worst = 0.0
    for index in range(len(STARS)):
        for field in (
            "electron_density",
            "mass_density",
            "ion_stage_populations_by_packed_slot",
            "partition_normalized_populations_by_packed_slot",
        ):
            worst = max(
                worst,
                float(
                    torch.max(
                        torch.abs(
                            getattr(twin, field)[index]
                            - getattr(singles[index], field)[0]
                        )
                    )
                ),
            )
    print(f"   max absolute difference: {worst:.3e}")
    if not worst == 0.0:
        failures.append("batch_invariance")

    print()
    if failures:
        print("FAIL:", ", ".join(failures))
        return 1
    print(
        f"PASS: electron/mass density within {DENSITY_TOLERANCE:g} of the "
        f"reference atomic path, populations within {POPULATION_DEX_TOLERANCE:g} "
        "dex, gradients finite, batch invariant"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
