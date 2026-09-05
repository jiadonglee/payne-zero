"""Structured atmosphere-solver inputs, outputs, and controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .atmosphere_io import ModelAtmosphere


@dataclass(frozen=True)
class AtmosphereInput:
    """Input sources for one atmosphere solve."""

    initial_atmosphere: ModelAtmosphere
    molecules_path: Path | None = None
    selected_line_catalog_path: Path | None = None
    detailed_line_catalog_path: Path | None = None
    predicted_atomic_lines_path: Path | None = None
    observed_atomic_lines_path: Path | None = None
    high_excitation_lines_path: Path | None = None
    diatomic_lines_path: Path | None = None
    titanium_oxide_lines_path: Path | None = None
    water_lines_path: Path | None = None
    h3plus_lines_path: Path | None = None


@dataclass(frozen=True)
class AtmosphereOutput:
    """Output targets produced by one atmosphere solve."""

    structured_atmosphere_path: Path | None = None
    diagnostics_path: Path | None = None
    debug_state_path: Path | None = None


@dataclass(frozen=True)
class AtmosphereConfig:
    """Top-level atmosphere-solver configuration."""

    inputs: AtmosphereInput
    outputs: AtmosphereOutput
    iterations: int = 1
    enable_molecules: bool = False
    enable_convection: bool = True
    enable_convergence_stop: bool = False
    minimum_iterations_before_convergence: int = 3
    required_consecutive_converged_iterations: int = 1
    maximum_deep_layer_relative_temperature_change: float = 5.0e-4
    maximum_all_layer_relative_temperature_change: float | None = None
    molecular_convection_thermal_tracks_perturbation: bool = True
    # Keep every ``stride``-th point of the 30000-point opacity-sampling grid
    # (``continuum_opacity.build_opacity_sampling_grid``). The opacity stage
    # costs time linear in the number of sampled points, so this trades
    # frequency resolution for wall time. 1 is the production grid and the only
    # value that reproduces released results.
    opacity_frequency_grid_stride: int = 1
    # Opacity lagging (off by default; the default path is bit-identical to the
    # historical solver). When enabled, iterations that are not on the recompute
    # schedule reuse the previous exact iteration's opacity slabs instead of
    # recomputing them. The convergence stop is never allowed to fire on such an
    # iteration -- see ``runner.opacity_recompute_scheduled`` and the stop
    # decision in ``runner._run_atmosphere_model``.
    enable_opacity_lagging: bool = False
    opacity_recompute_interval: int = 2
    # Experimental global damping of the temperature correction. The
    # production value is 1.0; values below 1.0 are only for solver-policy
    # experiments and do not change the default path.
    temperature_correction_damping: float = 1.0
    # Experimental residual-guided step scaling (off by default; the default
    # path is bit-identical to the historical solver). When enabled, the
    # global temperature-correction step is rescaled each iteration from the
    # p95 absolute-flux-error trend: halve on worsening, restore gradually
    # after repeated improvements. See
    # ``temperature_correction.next_flux_residual_step_scale``.
    flux_residual_guided_damping: bool = False
    # Experimental companion to the stopping rule (off by default; the
    # default path is unchanged). When enabled together with
    # ``enable_convergence_stop``, convergence additionally requires the p95
    # absolute flux error not to be worse than the previous iteration's.
    require_improving_flux_residual: bool = False


DEFAULT_OPACITY_FLAGS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
