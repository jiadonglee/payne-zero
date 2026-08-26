"""Build a Payne-grid MARCS ``(m,T)`` seed using Payne Rosseland opacity.

This is a diagnostic bridge between an external MARCS profile and the fixed
Payne-Zero Rosseland grid.  It first materializes the dependent fields from
the native MARCS ``(m,T)`` using the exact physics path, integrates the
resulting Payne Rosseland opacity, and then uses the solver's own remap
convention to place ``(m,T)`` on the standard grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from bench.labels import StellarLabels
from payne_zero_atmosphere.radiative_transfer import integrate_on_depth_grid, remap_to_grid
from payne_zero_atmosphere.run_setup import standard_rosseland_optical_depth_grid
from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere

from .marcs_h5 import load_marcs_node


def _log_tau_extrapolating_profile(
    source_tau: np.ndarray,
    source_values: np.ndarray,
    target_tau: np.ndarray,
) -> np.ndarray:
    """Interpolate positive profiles in log tau, with linear edge slopes."""

    source_tau = np.asarray(source_tau, dtype=np.float64)
    values = np.asarray(source_values, dtype=np.float64)
    target_tau = np.asarray(target_tau, dtype=np.float64)
    if (
        np.any(~np.isfinite(source_tau))
        or np.any(source_tau <= 0.0)
        or np.any(np.diff(source_tau) <= 0.0)
        or np.any(~np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("log-tau remap requires finite positive increasing inputs")
    source_coordinate = np.log10(source_tau)
    target_coordinate = np.log10(target_tau)
    log_values = np.log(values)
    spline = PchipInterpolator(source_coordinate, log_values, extrapolate=False)
    result = np.empty_like(target_coordinate)
    inside = (target_coordinate >= source_coordinate[0]) & (
        target_coordinate <= source_coordinate[-1]
    )
    result[inside] = spline(target_coordinate[inside])
    left_slope = (log_values[1] - log_values[0]) / (
        source_coordinate[1] - source_coordinate[0]
    )
    right_slope = (log_values[-1] - log_values[-2]) / (
        source_coordinate[-1] - source_coordinate[-2]
    )
    left = target_coordinate < source_coordinate[0]
    right = target_coordinate > source_coordinate[-1]
    result[left] = log_values[0] + left_slope * (
        target_coordinate[left] - source_coordinate[0]
    )
    result[right] = log_values[-1] + right_slope * (
        target_coordinate[right] - source_coordinate[-1]
    )
    return np.exp(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marcs-grid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=3750.0)
    parser.add_argument(
        "--source-profile",
        type=Path,
        default=None,
        help="reuse source_column_mass/source_temperature/source_rosseland_tau",
    )
    args = parser.parse_args()

    result = None
    if args.source_profile is None:
        labels = StellarLabels(
            effective_temperature=args.temperature,
            log_surface_gravity=5.0,
            metallicity=0.0,
            alpha_enhancement=0.0,
            microturbulence_km_s=1.0,
        )
        node = load_marcs_node(
            args.marcs_grid,
            labels,
            depth_coordinate="log_mass",
        )
        reduced = ReducedAtmosphere(
            column_mass=node.reduced_column_mass,
            temperature=node.reduced_temperature,
            labels=labels.as_kwargs(),
        )
        result = reconstruct_full_atmosphere(
            reduced,
            n_synchronizations=None,
            max_synchronizations=8,
            pressure_tolerance_dex=1.0e-3,
            allow_extrapolation=True,
        )
        source_mass = node.reduced_column_mass
        source_temperature = node.reduced_temperature
        opacity = np.asarray(result.atmosphere.rosseland_opacity, dtype=np.float64)
        rosseland_tau = integrate_on_depth_grid(
            source_mass,
            opacity,
            surface_value=float(opacity[0] * source_mass[0]),
        )
    else:
        with np.load(args.source_profile, allow_pickle=False) as profile:
            source_mass = np.asarray(profile["source_column_mass"], dtype=np.float64)
            source_temperature = np.asarray(
                profile["source_temperature"], dtype=np.float64
            )
            rosseland_tau = np.asarray(
                profile["source_rosseland_tau"], dtype=np.float64
            )
        opacity = None
    target_tau = standard_rosseland_optical_depth_grid(80)
    solver_remapped_mass, _ = remap_to_grid(
        rosseland_tau, source_mass, target_tau
    )
    solver_remapped_temperature, _ = remap_to_grid(
        rosseland_tau, source_temperature, target_tau
    )
    remapped_mass = _log_tau_extrapolating_profile(
        rosseland_tau, source_mass, target_tau
    )
    remapped_temperature = _log_tau_extrapolating_profile(
        rosseland_tau, source_temperature, target_tau
    )
    if (
        not np.all(np.isfinite(remapped_mass))
        or not np.all(np.isfinite(remapped_temperature))
        or np.any(remapped_mass <= 0.0)
        or np.any(np.diff(remapped_mass) <= 0.0)
        or np.any(remapped_temperature <= 0.0)
    ):
        raise ValueError("physical Rosseland remap produced an invalid profile")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        column_mass=remapped_mass,
        temperature=remapped_temperature,
        solver_remap_column_mass=solver_remapped_mass,
        solver_remap_temperature=solver_remapped_temperature,
        source_column_mass=source_mass,
        source_temperature=source_temperature,
        source_rosseland_tau=rosseland_tau,
        source_rosseland_opacity=(
            np.asarray(opacity, dtype=np.float64)
            if opacity is not None
            else np.zeros_like(rosseland_tau)
        ),
        target_rosseland_tau=target_tau,
    )
    summary = {
        "temperature": args.temperature,
        "remap_mode": "log_tau_linear_edges",
        "n_synchronizations": None if result is None else result.n_synchronizations,
        "pressure_change_dex_by_pass": (
            [] if result is None else result.pressure_change_dex_by_pass
        ),
        "source_tau_range": [float(rosseland_tau[0]), float(rosseland_tau[-1])],
        "source_log_tau_range": [
            float(np.log10(max(rosseland_tau[0], 1.0e-300))),
            float(np.log10(max(rosseland_tau[-1], 1.0e-300))),
        ],
        "target_log_tau_range": [-6.875, 3.0],
        "remapped_column_mass_endpoints": [
            float(remapped_mass[0]),
            float(remapped_mass[-1]),
        ],
        "remapped_temperature_endpoints": [
            float(remapped_temperature[0]),
            float(remapped_temperature[-1]),
        ],
        "solver_remap_column_mass_endpoints": [
            float(solver_remapped_mass[0]),
            float(solver_remapped_mass[-1]),
        ],
        "solver_remap_temperature_endpoints": [
            float(solver_remapped_temperature[0]),
            float(solver_remapped_temperature[-1]),
        ],
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
