"""Cool-star temperature-step experiment for Payne-Zero.

The experiment is intentionally a runner around the existing solver, not a
new solver.  Each track first solves a 4000 K production anchor and a MARCS
``(m,T)`` cross-check.  Direct target starts and continuation paths then use
the same production settings with a 30-iteration ceiling.  Products are
written per method, and the existing three-metric spectral gate is applied
after the solve stage.

Example::

    PYTHONPATH=. .venv/bin/python -m \
      experiments.reduced_state_emulator.cool_star_step_test \
      --marcs-grid SDSS_MARCS_atmospheres.h5 --stage pilot --workers 1

The 619 MB MARCS file is never loaded as one NumPy array.  The parent validates
its SHA-256 and each worker reads only the requested ``(5, 56)`` native node.
"""

from __future__ import annotations

# Set Numba's threading policy before importing the atmosphere runner.
from bench import environment as _environment  # noqa: F401,E402

import argparse
import dataclasses
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import traceback
import warnings

import numpy as np

from bench.labels import StellarLabels
from bench.run_reference import _as_plain, _atmosphere_is_finite, _solver_config
from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere
from payne_zero_atmosphere.runner import run_atmosphere_model
from payne_zero_atmosphere.warm_start import INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH
from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere

from .marcs_h5 import (
    EXPECTED_MARCS_SHA256,
    MARCS_DEPTH_COORDINATES,
    MarcsH5Error,
    inspect_marcs_grid,
    load_marcs_node,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "cool_star_step_test"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "cool_star_step_test"
DEFAULT_TWO_FIELD_CHECKPOINT_DIR = (
    REPO_ROOT / "artifacts" / "reduced_state_emulator" / "physical"
)
DEFAULT_TWO_FIELD_SEEDS = (20260807, 20260808, 20260809)
ANCHOR_TEMPERATURE = 4000.0
TARGET_TEMPERATURES = (3900.0, 3800.0, 3750.0, 3700.0, 3600.0, 3500.0)
CONTINUATION_100_TARGETS = (3900.0, 3800.0, 3700.0, 3600.0, 3500.0)
CONFIRM_LOGG = (4.5, 5.0, 5.5)
CONFIRM_METALLICITY = (-1.0, 0.0, 0.5)
ALPHA_ENHANCEMENT = 0.0
CARBON_ENHANCEMENT = 0.0
MICROTURBULENCE_KM_S = 1.0
ITERATION_CAP = 30
PRIMARY_ITERATION_CAP = 15
SPECTRAL_BAR = 5.0e-3
PLANCK_CONSTANT_CGS = 6.62607015e-27
SPEED_OF_LIGHT_CGS = 2.99792458e10
ANALYTIC_HYDROGEN_RHO_OVER_G = 1.0e-7
ANALYTIC_KAPPA_CALIBRATION_TEMPERATURE = 4000.0
ANALYTIC_TAU5000_MIN = 1.0e-5
ANALYTIC_TAU5000_MAX = 70.0
ANALYTIC_KAPPA_CM2_G = 3.4e5
DIRECT_METHODS = (
    "production_six_field_target",
    "learned_two_field_target",
    "marcs_target_reduced",
    "analytic_target_reduced",
    "anchor_full_carry",
    "anchor_reduced_rematerialized",
)
CONTINUATION_METHODS = (
    "continuation_250_full_carry",
    "continuation_250_reduced_rematerialized",
    "continuation_100_full_carry",
    "continuation_100_reduced_rematerialized",
)
ALL_METHODS = DIRECT_METHODS + CONTINUATION_METHODS


@dataclasses.dataclass(frozen=True)
class TrackSpec:
    """A fixed non-temperature MARCS/Payne-Zero trajectory."""

    log_surface_gravity: float
    metallicity: float
    alpha_enhancement: float = ALPHA_ENHANCEMENT
    carbon_enhancement: float = CARBON_ENHANCEMENT
    microturbulence_km_s: float = MICROTURBULENCE_KM_S

    @property
    def track_id(self) -> str:
        return (
            f"g{self.log_surface_gravity:+05.2f}"
            f"_m{self.metallicity:+05.2f}"
            f"_a{self.alpha_enhancement:+05.2f}"
            f"_c{self.carbon_enhancement:+05.2f}"
            f"_x{self.microturbulence_km_s:04.2f}"
        )

    def labels(self, effective_temperature: float) -> StellarLabels:
        return StellarLabels(
            effective_temperature=float(effective_temperature),
            log_surface_gravity=float(self.log_surface_gravity),
            metallicity=float(self.metallicity),
            alpha_enhancement=float(self.alpha_enhancement),
            microturbulence_km_s=float(self.microturbulence_km_s),
        )

    def as_json(self) -> dict[str, float | str]:
        return {
            "track_id": self.track_id,
            "log_surface_gravity": self.log_surface_gravity,
            "metallicity": self.metallicity,
            "alpha_enhancement": self.alpha_enhancement,
            "carbon_enhancement": self.carbon_enhancement,
            "microturbulence_km_s": self.microturbulence_km_s,
        }

    def label_metadata(self, effective_temperature: float) -> dict[str, str]:
        labels = self.labels(effective_temperature)
        return {
            "effective_temperature": f"{labels.effective_temperature:.6f}",
            "log_surface_gravity": f"{labels.log_surface_gravity:.6f}",
            "metallicity": f"{labels.metallicity:.6f}",
            "alpha_enhancement": f"{labels.alpha_enhancement:.6f}",
            "microturbulence_km_s": f"{labels.microturbulence_km_s:.6f}",
        }


def build_track_manifest(stage: str) -> list[TrackSpec]:
    """Return the pilot or frozen 3x3 confirmation track set."""

    if stage not in {"pilot", "confirm"}:
        raise ValueError("stage must be pilot or confirm")
    if stage == "pilot":
        return [TrackSpec(log_surface_gravity=5.0, metallicity=0.0)]
    return [
        TrackSpec(log_surface_gravity=logg, metallicity=metallicity)
        for logg in CONFIRM_LOGG
        for metallicity in CONFIRM_METALLICITY
    ]


def manifest_payload(stage: str, *, marcs_depth_coordinate: str = "log_mass") -> dict:
    if marcs_depth_coordinate not in MARCS_DEPTH_COORDINATES:
        raise ValueError(
            f"marcs_depth_coordinate must be one of {MARCS_DEPTH_COORDINATES!r}"
        )
    tracks = build_track_manifest(stage)
    return {
        "stage": stage,
        "anchor_temperature": ANCHOR_TEMPERATURE,
        "target_temperatures": list(TARGET_TEMPERATURES),
        "tracks": [track.as_json() for track in tracks],
        "continuation_schedules": {
            "250K": [4000.0, 3750.0, 3500.0],
            "100K": [4000.0, *CONTINUATION_100_TARGETS],
        },
        "policy": {
            "iteration_cap": ITERATION_CAP,
            "primary_iteration_cap": PRIMARY_ITERATION_CAP,
            "spectral_bar": SPECTRAL_BAR,
            "native_marcs_nodes_only": True,
            "marcs_depth_coordinate": marcs_depth_coordinate,
            "sealed_holdout_opened": False,
        },
    }


def manifest_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _set_single_thread_environment() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[name] = "1"
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # A direct in-process smoke call may have initialized the pool
            # already; the environment variables still constrain native BLAS.
            pass
    except ImportError:  # pragma: no cover - torch is a core dependency
        pass


def _clone_atmosphere(atmosphere: ModelAtmosphere) -> ModelAtmosphere:
    """Copy every mutable atmosphere field before a solver attempt."""

    return ModelAtmosphere(
        column_mass=np.asarray(atmosphere.column_mass, dtype=np.float64).copy(),
        temperature=np.asarray(atmosphere.temperature, dtype=np.float64).copy(),
        gas_pressure=np.asarray(atmosphere.gas_pressure, dtype=np.float64).copy(),
        electron_density=np.asarray(
            atmosphere.electron_density, dtype=np.float64
        ).copy(),
        rosseland_opacity=np.asarray(
            atmosphere.rosseland_opacity, dtype=np.float64
        ).copy(),
        radiative_acceleration=np.asarray(
            atmosphere.radiative_acceleration, dtype=np.float64
        ).copy(),
        microturbulence=np.asarray(atmosphere.microturbulence, dtype=np.float64).copy(),
        convective_flux=np.asarray(atmosphere.convective_flux, dtype=np.float64).copy(),
        convective_velocity=np.asarray(
            atmosphere.convective_velocity, dtype=np.float64
        ).copy(),
        metadata=dict(atmosphere.metadata),
        fixed_column_abundance_values=dict(
            atmosphere.fixed_column_abundance_values
        ),
    )


def _atmosphere_quality(atmosphere: ModelAtmosphere | None) -> dict[str, object]:
    """Independent finite/positive/monotone check for the six physical fields."""

    if atmosphere is None:
        return {"available": False, "finite": False, "positive": False, "monotone_column_mass": False, "valid": False}
    fields = {
        "column_mass": atmosphere.column_mass,
        "temperature": atmosphere.temperature,
        "gas_pressure": atmosphere.gas_pressure,
        "electron_density": atmosphere.electron_density,
        "rosseland_opacity": atmosphere.rosseland_opacity,
        "radiative_acceleration": atmosphere.radiative_acceleration,
    }
    finite = all(
        np.all(np.isfinite(np.asarray(values, dtype=np.float64)))
        for values in fields.values()
    )
    positive = all(
        np.all(np.asarray(values, dtype=np.float64) > 0.0)
        for values in fields.values()
    ) if finite else False
    monotone = bool(
        finite
        and np.all(np.asarray(atmosphere.column_mass, dtype=np.float64) > 0.0)
        and np.all(np.diff(np.asarray(atmosphere.column_mass, dtype=np.float64)) > 0.0)
    )
    return {
        "available": True,
        "finite": bool(finite),
        "positive": bool(positive),
        "monotone_column_mass": monotone,
        "valid": bool(finite and positive and monotone),
        "layers": int(atmosphere.layers),
    }


def _state_difference(
    candidate: ModelAtmosphere | None, reference: ModelAtmosphere | None
) -> dict[str, float] | None:
    """Compare final six-field arrays on their common 80-layer grid."""

    if candidate is None or reference is None:
        return None
    fields = (
        "column_mass",
        "temperature",
        "gas_pressure",
        "electron_density",
        "rosseland_opacity",
        "radiative_acceleration",
    )
    result: dict[str, float] = {}
    for field in fields:
        left = np.asarray(getattr(candidate, field), dtype=np.float64)
        right = np.asarray(getattr(reference, field), dtype=np.float64)
        if left.shape != right.shape:
            return None
        result[field] = float(
            np.max(np.abs(left - right) / np.maximum(np.abs(right), 1.0e-300))
        )
    return result


def _marcs_diagnostics(node) -> dict[str, object]:
    """Record decoded MARCS quantities without passing them to the solver."""

    fields = node.native_fields
    return {
        "native_layers": int(node.native_column_mass.size),
        "native_indices": list(node.indices),
        "native_finite": bool(
            all(np.all(np.isfinite(values)) for values in fields.values())
        ),
        "native_positive_density_pressure": bool(
            np.all(fields["electron_density"] > 0.0)
            and np.all(fields["total_number_density"] > 0.0)
            and np.all(fields["gas_pressure"] > 0.0)
        ),
        "native_column_mass_monotone": bool(
            np.all(np.diff(fields["column_mass"]) > 0.0)
        ),
        "native_tau5000_range": [
            float(np.min(fields["tau_5000"])),
            float(np.max(fields["tau_5000"])),
        ],
        "converted_layers": int(node.reduced_column_mass.size),
    }


def _residual_summary(diagnostics: dict) -> dict[str, object]:
    timings = diagnostics.get("iteration_timings", [])
    if not isinstance(timings, list):
        timings = []
    first = timings[0] if timings else {}
    final = timings[-1] if timings else {}
    fields = (
        "deep_layer_relative_temperature_change",
        "all_layer_relative_temperature_change",
        "median_absolute_flux_error_percent",
        "p95_absolute_flux_error_percent",
        "maximum_absolute_flux_error_percent",
    )
    return {
        "first_iteration": {
            field: _as_plain(first.get(field)) for field in fields
        },
        "final_iteration": {
            field: _as_plain(final.get(field)) for field in fields
        },
        "final_diagnostics": {
            field: _as_plain(diagnostics.get(field))
            for field in (
                "deep_layer_relative_temperature_change",
                "all_layer_relative_temperature_change",
                "median_absolute_flux_error_percent",
                "p95_absolute_flux_error_percent",
                "maximum_absolute_flux_error_percent",
                "total_seconds",
            )
        },
    }


def _failed_record(
    *,
    track: TrackSpec,
    method: str,
    labels: StellarLabels,
    source_temperature: float | None,
    target_temperature: float,
    schedule: str,
    error: BaseException | str,
    status: str = "failed_before_solver",
) -> dict:
    message = str(error)
    if isinstance(error, BaseException):
        message = f"{type(error).__name__}: {message}"
    return {
        "record_type": "cool_star_step",
        "track_id": track.track_id,
        "track": track.as_json(),
        "method": method,
        "schedule": schedule,
        "labels": labels.as_kwargs(),
        "source_temperature": source_temperature,
        "target_temperature": target_temperature,
        "delta_temperature": (
            None
            if source_temperature is None
            else float(source_temperature - target_temperature)
        ),
        "converged": False,
        "solver_converged": False,
        "iterations": None,
        "seconds": 0.0,
        "product_path": None,
        "product_written": False,
        "state_quality": _atmosphere_quality(None),
        "initial_quality": None,
        "survives_solver": False,
        "spectral_pass": None,
        "survives": False,
        "primary_pass": False,
        "recovered_pass": False,
        "status": status,
        "error": message,
    }


def _solve_attempt(
    *,
    track: TrackSpec,
    method: str,
    schedule: str,
    source_temperature: float | None,
    target_labels: StellarLabels,
    initial_atmosphere: ModelAtmosphere | None,
    product_dir: Path,
    iteration_cap: int,
    maximum_all_layer_relative_temperature_change: float | None = None,
    after_iteration_hook=None,
    config_overrides: dict | None = None,
) -> tuple[dict, ModelAtmosphere | None]:
    """Run one exact solver attempt and return its final state for continuation.

    ``after_iteration_hook`` is forwarded to the runner unchanged; it lets a
    campaign record per-iteration diagnostics without touching the solver.
    ``config_overrides`` applies experimental solver-policy fields (e.g.
    ``temperature_correction_damping``) on top of the production config
    without touching it for anyone else.
    """

    if initial_atmosphere is None:
        return (
            _failed_record(
                track=track,
                method=method,
                labels=target_labels,
                source_temperature=source_temperature,
                target_temperature=target_labels.effective_temperature,
                schedule=schedule,
                error="no usable initial atmosphere",
            ),
            None,
        )
    initial_quality = _atmosphere_quality(initial_atmosphere)
    product_dir = Path(product_dir)
    product_dir.mkdir(parents=True, exist_ok=True)
    product_path = product_dir / f"{target_labels.slug}.npz"
    product_path.unlink(missing_ok=True)
    started = time.perf_counter()
    captured_warnings: list[str] = []
    result = None
    error = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = _solver_config(
                _clone_atmosphere(initial_atmosphere),
                iterations_per_trial=int(iteration_cap),
                structured_atmosphere_path=product_path,
                debug_state_path=None,
            )
            if maximum_all_layer_relative_temperature_change is not None:
                config = dataclasses.replace(
                    config,
                    maximum_all_layer_relative_temperature_change=float(
                        maximum_all_layer_relative_temperature_change
                    ),
                )
            if config_overrides:
                config = dataclasses.replace(config, **config_overrides)
            result = run_atmosphere_model(
                config, after_iteration_hook=after_iteration_hook
            )
        captured_warnings = [str(item.message) for item in caught]
    except Exception as exc:  # noqa: BLE001 - a failed step is an outcome
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    seconds = time.perf_counter() - started
    if result is None:
        record = _failed_record(
            track=track,
            method=method,
            labels=target_labels,
            source_temperature=source_temperature,
            target_temperature=target_labels.effective_temperature,
            schedule=schedule,
            error=error or "solver returned no result",
        )
        record.update(
            {
                "seconds": seconds,
                "initial_quality": initial_quality,
                "warnings": captured_warnings,
            }
        )
        return record, None

    final_atmosphere = result.atmosphere
    quality = _atmosphere_quality(final_atmosphere)
    solver_converged = bool(result.converged) and _atmosphere_is_finite(
        final_atmosphere
    )
    product_written = product_path.is_file()
    survives_solver = bool(solver_converged and quality["valid"] and product_written)
    iterations = int(result.iterations_completed)
    record = {
        "record_type": "cool_star_step",
        "track_id": track.track_id,
        "track": track.as_json(),
        "method": method,
        "schedule": schedule,
        "labels": target_labels.as_kwargs(),
        "source_temperature": source_temperature,
        "target_temperature": target_labels.effective_temperature,
        "delta_temperature": (
            None
            if source_temperature is None
            else float(source_temperature - target_labels.effective_temperature)
        ),
        "converged": bool(solver_converged),
        "solver_converged": bool(solver_converged),
        "iterations": iterations,
        "seconds": float(seconds),
        "product_path": str(product_path) if product_written else None,
        "product_written": bool(product_written),
        "initial_quality": initial_quality,
        "state_quality": quality,
        "survives_solver": survives_solver,
        "spectral_pass": None,
        "survives": False,
        "primary_pass": False,
        "recovered_pass": False,
        "status": "solver_pass" if survives_solver else "solver_fail",
        "warnings": captured_warnings,
        "solver_diagnostics": _residual_summary(result.diagnostics),
        "error": None,
    }
    if maximum_all_layer_relative_temperature_change is not None:
        record["solver_policy"] = {
            "maximum_all_layer_relative_temperature_change": float(
                maximum_all_layer_relative_temperature_change
            )
        }
    if not solver_converged:
        record["error"] = "solver did not satisfy its formal convergence criterion"
    elif not quality["valid"]:
        record["error"] = "final six-field atmosphere failed the independent quality gate"
    elif not product_written:
        record["error"] = "converged solver did not write a structured product"
    return record, _clone_atmosphere(final_atmosphere) if solver_converged else None


def _production_atmosphere(labels: StellarLabels) -> ModelAtmosphere:
    """Build the six-field initializer, explicitly permitting test extrapolation."""

    from payne_zero_atmosphere.warm_start import emulator_warm_start_model

    atmosphere, _deck = emulator_warm_start_model(
        device="cpu", allow_extrapolation=True, **labels.as_kwargs()
    )
    return atmosphere


def _reconstruct_from_mt(
    labels: StellarLabels,
    column_mass: np.ndarray,
    temperature: np.ndarray,
) -> ModelAtmosphere:
    reduced = ReducedAtmosphere(
        column_mass=np.asarray(column_mass, dtype=np.float64).copy(),
        temperature=np.asarray(temperature, dtype=np.float64).copy(),
        labels=labels.as_kwargs(),
    )
    result = reconstruct_full_atmosphere(
        reduced,
        n_synchronizations=None,
        max_synchronizations=8,
        pressure_tolerance_dex=1.0e-3,
        allow_extrapolation=True,
    )
    if not result.synchronized:
        raise MarcsH5Error(
            "Payne-Zero full-state reconstruction did not reach its pressure "
            f"tolerance after {result.n_synchronizations} synchronizations"
        )
    return result.atmosphere


def _planck_flux_integrand(
    wavelength_nm: np.ndarray, temperature: float
) -> np.ndarray:
    wavelength_cm = np.asarray(wavelength_nm, dtype=np.float64) * 1.0e-7
    exponent = (
        PLANCK_CONSTANT_CGS
        * SPEED_OF_LIGHT_CGS
        / (wavelength_cm * 1.380649e-16 * float(temperature))
    )
    exponent = np.clip(exponent, 0.0, 700.0)
    return (
        2.0
        * PLANCK_CONSTANT_CGS
        * SPEED_OF_LIGHT_CGS**2
        / wavelength_cm**5
        / np.expm1(exponent)
    ) * 1.0e-7


def _trapezoid(values: np.ndarray, coordinates: np.ndarray) -> float:
    """NumPy 1.26/2.x compatible trapezoid integral."""

    integrator = getattr(np, "trapezoid", None) or np.trapz
    return float(integrator(values, coordinates))


def _analytic_hminus_kappa(
    wavelength_nm: np.ndarray, temperature: float, rho_over_g: float
) -> np.ndarray:
    lam = np.asarray(wavelength_nm, dtype=np.float64) * 1.0e-7
    lyman_cm = 91.18e-7
    edge = np.where(lam <= lyman_cm, 6.30e-18 * (lam / lyman_cm) ** 3, 0.0)
    peak = np.exp(-0.5 * ((np.log(lam) - np.log(8.5e-5)) / 0.40) ** 2)
    infrared = 2.0 * np.maximum(lam / 8.5e-5, 1.0) ** 3
    shape = edge + peak + infrared
    scaled = shape * float(rho_over_g) * (
        (4000.0 / max(float(temperature), 1.0)) ** 2
    )
    return np.maximum(scaled, 1.0e-14)


def _analytic_hminus_kappa_mean(temperature: float, rho_over_g: float) -> float:
    wavelength_nm = np.geomspace(50.0, 10000.0, 8192)
    weights = _planck_flux_integrand(wavelength_nm, float(temperature))
    kappa = _analytic_hminus_kappa(wavelength_nm, float(temperature), rho_over_g)
    denominator = _trapezoid(weights, wavelength_nm)
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("analytic opacity flux normalization failed")
    value = _trapezoid(kappa * weights, wavelength_nm) / denominator
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("analytic opacity mean is invalid")
    return value


def _analytic_grey_hydrostatic_mt(labels: StellarLabels) -> dict[str, object]:
    gravity = 10.0 ** float(labels.log_surface_gravity)
    kappa_mean = ANALYTIC_KAPPA_CM2_G * (
        4000.0 / float(labels.effective_temperature)
    ) ** 2
    flux = 5.6697e-5 / 12.5664 * float(labels.effective_temperature) ** 4
    radiative_acceleration = kappa_mean * flux / SPEED_OF_LIGHT_CGS
    tau = np.geomspace(
        ANALYTIC_TAU5000_MIN,
        ANALYTIC_TAU5000_MAX,
        len(INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH),
    )
    temperature = float(labels.effective_temperature) * (
        0.75 * (tau + 2.0 / 3.0)
    ) ** 0.25
    column_mass = ((gravity + radiative_acceleration) / kappa_mean) * tau
    diagnostics = {
        "analytic_model": "grey_hydrostatic",
        "kappa_normalization_source": "fixed_hminus_rosseland_estimate",
        "calibrated_kappa_cm2_g": float(kappa_mean),
        "tau_min": float(np.min(tau)),
        "tau_max": float(np.max(tau)),
        "rosseland_depth_min": None,
        "rosseland_depth_max": None,
        "flux_weighted_kappa_cm2_g": float(kappa_mean),
        "surface_g_rad_over_g": float(radiative_acceleration / gravity),
        "temperature_min_k": float(np.min(temperature)),
        "temperature_max_k": float(np.max(temperature)),
        "column_mass_min_g_cm2": float(np.min(column_mass)),
        "column_mass_max_g_cm2": float(np.max(column_mass)),
    }
    if (
        column_mass.shape != (80,)
        or temperature.shape != (80,)
        or np.any(~np.isfinite(column_mass))
        or np.any(~np.isfinite(temperature))
        or np.any(column_mass <= 0.0)
        or np.any(temperature <= 0.0)
        or np.any(np.diff(column_mass) <= 0.0)
    ):
        raise ValueError("analytic grey hydrostatic initializer produced an invalid m,T profile")
    return {
        "column_mass": column_mass,
        "temperature": temperature,
        "diagnostics": diagnostics,
    }


def _load_two_field_predictor(checkpoint_dir: Path, seeds: tuple[int, ...]):
    from reduced_state.emulator import load_physical_checkpoint

    models = []
    for seed in seeds:
        path = Path(checkpoint_dir) / f"checkpoint_physical_seed{int(seed)}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        models.append(load_physical_checkpoint(path)[:2])
    return models


def _predict_two_field(
    labels: StellarLabels,
    models,
) -> tuple[np.ndarray, np.ndarray]:
    from reduced_state.emulator import predict_physical_state

    label_array = np.asarray(
        [
            [
                labels.effective_temperature,
                labels.log_surface_gravity,
                labels.metallicity,
                labels.alpha_enhancement,
                labels.microturbulence_km_s,
            ]
        ],
        dtype=np.float64,
    )
    masses = []
    temperatures = []
    for model, standardization in models:
        mass, temperature = predict_physical_state(
            model, standardization, label_array
        )
        masses.append(mass[0])
        temperatures.append(temperature[0])
    column_mass = np.median(np.stack(masses), axis=0)
    temperature = np.median(np.stack(temperatures), axis=0)
    if (
        column_mass.shape != (80,)
        or temperature.shape != (80,)
        or np.any(~np.isfinite(column_mass))
        or np.any(~np.isfinite(temperature))
        or np.any(column_mass <= 0.0)
        or np.any(temperature <= 0.0)
        or np.any(np.diff(column_mass) <= 0.0)
    ):
        raise ValueError("two-field emulator returned an invalid m,T profile")
    return column_mass, temperature


def _retarget_full_state(
    full_anchor: ModelAtmosphere, target_template: ModelAtmosphere
) -> ModelAtmosphere:
    """Carry all six anchor fields while changing only target metadata."""

    carried = _clone_atmosphere(full_anchor)
    carried.metadata = dict(target_template.metadata)
    carried.fixed_column_abundance_values = dict(
        target_template.fixed_column_abundance_values
    )
    return carried


def _record_blocked(
    *,
    track: TrackSpec,
    method: str,
    labels: StellarLabels,
    source_temperature: float | None,
    schedule: str,
) -> dict:
    return _failed_record(
        track=track,
        method=method,
        labels=labels,
        source_temperature=source_temperature,
        target_temperature=labels.effective_temperature,
        schedule=schedule,
        error="continuation stopped because a previous step failed",
        status="blocked_by_previous_step",
    )


def _run_continuation(
    *,
    track: TrackSpec,
    method: str,
    schedule: str,
    targets: tuple[float, ...],
    mode: str,
    anchor_state: ModelAtmosphere | None,
    target_templates: dict[float, ModelAtmosphere],
    run_root: Path,
    iteration_cap: int,
) -> list[dict]:
    records: list[dict] = []
    current = _clone_atmosphere(anchor_state) if anchor_state is not None else None
    current_temperature = ANCHOR_TEMPERATURE
    for target_temperature in targets:
        target_labels = track.labels(target_temperature)
        if current is None:
            records.append(
                _record_blocked(
                    track=track,
                    method=method,
                    labels=target_labels,
                    source_temperature=current_temperature,
                    schedule=schedule,
                )
            )
            continue
        try:
            if mode == "full_carry":
                initial = _retarget_full_state(
                    current, target_templates[target_temperature]
                )
            elif mode == "reduced_rematerialized":
                initial = _reconstruct_from_mt(
                    target_labels, current.column_mass, current.temperature
                )
            else:
                raise ValueError(f"unknown continuation mode {mode}")
        except Exception as exc:  # noqa: BLE001 - record and stop this path
            record = _failed_record(
                track=track,
                method=method,
                labels=target_labels,
                source_temperature=current_temperature,
                target_temperature=target_temperature,
                schedule=schedule,
                error=exc,
            )
            record["status"] = "rematerialization_failed"
            records.append(record)
            current = None
            continue
        record, next_state = _solve_attempt(
            track=track,
            method=method,
            schedule=schedule,
            source_temperature=current_temperature,
            target_labels=target_labels,
            initial_atmosphere=initial,
            product_dir=run_root / "products" / method,
            iteration_cap=iteration_cap,
        )
        records.append(record)
        if record["survives_solver"] and next_state is not None:
            current = next_state
            current_temperature = target_temperature
        else:
            current = None
    return records


def _run_track_worker(payload: tuple) -> dict:
    """Build one track's anchors and all of its direct/continuation arms."""

    _set_single_thread_environment()
    track_payload, schema, options = payload
    track = TrackSpec(
        log_surface_gravity=float(track_payload["log_surface_gravity"]),
        metallicity=float(track_payload["metallicity"]),
        alpha_enhancement=float(track_payload["alpha_enhancement"]),
        carbon_enhancement=float(track_payload["carbon_enhancement"]),
        microturbulence_km_s=float(track_payload["microturbulence_km_s"]),
    )
    run_root = Path(options["run_root"])
    iteration_cap = int(options["iteration_cap"])
    records: list[dict] = []
    marcs_depth_coordinate = str(options.get("marcs_depth_coordinate", "log_mass"))
    references: dict[str, object] = {
        "track_id": track.track_id,
        "anchor_production_product": None,
        "anchor_marcs_product": None,
        "target_products": {},
        "errors": [],
    }
    target_templates: dict[float, ModelAtmosphere] = {}
    target_production_states: dict[float, ModelAtmosphere] = {}
    anchor_production_state: ModelAtmosphere | None = None

    try:
        anchor_labels = track.labels(ANCHOR_TEMPERATURE)
        anchor_template = _production_atmosphere(anchor_labels)
        anchor_record, anchor_production_state = _solve_attempt(
            track=track,
            method="anchor_production",
            schedule="anchor",
            source_temperature=None,
            target_labels=anchor_labels,
            initial_atmosphere=anchor_template,
            product_dir=run_root / "products" / "anchor_production",
            iteration_cap=iteration_cap,
        )
        records.append(anchor_record)
        if anchor_record["product_written"]:
            references["anchor_production_product"] = anchor_record["product_path"]

        anchor_marcs = load_marcs_node(
            schema.path,
            anchor_labels,
            carbon_enhancement=track.carbon_enhancement,
            verify_sha256=False,
            expected_sha256=None,
            schema=schema,
            depth_coordinate=marcs_depth_coordinate,
        )
        anchor_marcs_seed = _reconstruct_from_mt(
            anchor_labels,
            anchor_marcs.reduced_column_mass,
            anchor_marcs.reduced_temperature,
        )
        anchor_marcs_record, _anchor_marcs_state = _solve_attempt(
            track=track,
            method="anchor_marcs_reduced",
            schedule="anchor",
            source_temperature=None,
            target_labels=anchor_labels,
            initial_atmosphere=anchor_marcs_seed,
            product_dir=run_root / "products" / "anchor_marcs_reduced",
            iteration_cap=iteration_cap,
        )
        anchor_marcs_record["marcs_input_diagnostics"] = _marcs_diagnostics(
            anchor_marcs
        )
        records.append(anchor_marcs_record)
        if anchor_marcs_record["product_written"]:
            references["anchor_marcs_product"] = anchor_marcs_record["product_path"]
    except Exception as exc:  # noqa: BLE001 - preserve track-level failure
        references["errors"].append(f"anchor: {type(exc).__name__}: {exc}")

    two_field_models = None
    two_field_error = None
    checkpoint_dir = options.get("two_field_checkpoint_dir")
    if checkpoint_dir:
        try:
            seeds = tuple(int(seed) for seed in options["two_field_seeds"])
            two_field_models = _load_two_field_predictor(Path(checkpoint_dir), seeds)
        except Exception as exc:  # noqa: BLE001 - arm is explicitly unavailable
            two_field_error = f"{type(exc).__name__}: {exc}"

    for target_temperature in TARGET_TEMPERATURES:
        target_labels = track.labels(target_temperature)
        try:
            target_templates[target_temperature] = _production_atmosphere(target_labels)
            target_record, _target_state = _solve_attempt(
                track=track,
                method="production_six_field_target",
                schedule="direct",
                source_temperature=ANCHOR_TEMPERATURE,
                target_labels=target_labels,
                initial_atmosphere=target_templates[target_temperature],
                product_dir=run_root / "products" / "production_six_field_target",
                iteration_cap=iteration_cap,
            )
            if _target_state is not None:
                target_production_states[target_temperature] = _target_state
        except Exception as exc:  # noqa: BLE001
            target_record = _failed_record(
                track=track,
                method="production_six_field_target",
                labels=target_labels,
                source_temperature=ANCHOR_TEMPERATURE,
                target_temperature=target_temperature,
                schedule="direct",
                error=exc,
            )
            target_record["status"] = "target_initializer_failed"
        target_record["initializer_regime"] = (
            "in_domain" if target_temperature >= ANCHOR_TEMPERATURE else "extrapolation"
        )
        records.append(target_record)

        try:
            target_marcs = load_marcs_node(
                schema.path,
                target_labels,
                carbon_enhancement=track.carbon_enhancement,
                verify_sha256=False,
                expected_sha256=None,
                schema=schema,
                depth_coordinate=marcs_depth_coordinate,
            )
            target_marcs_seed = _reconstruct_from_mt(
                target_labels,
                target_marcs.reduced_column_mass,
                target_marcs.reduced_temperature,
            )
            marcs_record, _marcs_state = _solve_attempt(
                track=track,
                method="marcs_target_reduced",
                schedule="direct",
                source_temperature=ANCHOR_TEMPERATURE,
                target_labels=target_labels,
                initial_atmosphere=target_marcs_seed,
                product_dir=run_root / "products" / "marcs_target_reduced",
                iteration_cap=iteration_cap,
            )
            marcs_record["marcs_input_diagnostics"] = _marcs_diagnostics(
                target_marcs
            )
            marcs_record["final_atmosphere_difference_to_production_reference"] = (
                _state_difference(
                    _marcs_state,
                    target_production_states.get(target_temperature),
                )
            )
        except Exception as exc:  # noqa: BLE001
            marcs_record = _failed_record(
                track=track,
                method="marcs_target_reduced",
                labels=target_labels,
                source_temperature=ANCHOR_TEMPERATURE,
                target_temperature=target_temperature,
                schedule="direct",
                error=exc,
            )
            marcs_record["status"] = "marcs_or_reconstruction_failed"
        records.append(marcs_record)
        references["target_products"][str(target_temperature)] = {
            "production": target_record.get("product_path"),
            "marcs": marcs_record.get("product_path"),
        }

        try:
            analytic = _analytic_grey_hydrostatic_mt(target_labels)
            analytic_seed = _reconstruct_from_mt(
                target_labels,
                analytic["column_mass"],
                analytic["temperature"],
            )
            analytic_record, _analytic_state = _solve_attempt(
                track=track,
                method="analytic_target_reduced",
                schedule="direct",
                source_temperature=ANCHOR_TEMPERATURE,
                target_labels=target_labels,
                initial_atmosphere=analytic_seed,
                product_dir=run_root / "products" / "analytic_target_reduced",
                iteration_cap=iteration_cap,
            )
            analytic_record["analytic_input_diagnostics"] = {
                **analytic["diagnostics"],
                "uses_emulator_at_target": False,
                "note": "target grey/hydrostatic (m,T); other fields rebuilt",
            }
            analytic_record[
                "final_atmosphere_difference_to_production_reference"
            ] = _state_difference(
                _analytic_state,
                target_production_states.get(target_temperature),
            )
        except Exception as exc:  # noqa: BLE001
            analytic_record = _failed_record(
                track=track,
                method="analytic_target_reduced",
                labels=target_labels,
                source_temperature=ANCHOR_TEMPERATURE,
                target_temperature=target_temperature,
                schedule="direct",
                error=exc,
            )
            analytic_record["status"] = "analytic_or_reconstruction_failed"
        analytic_record["initializer_regime"] = "analytic_no_target_emulator"
        records.append(analytic_record)

        if two_field_models is None:
            records.append(
                _failed_record(
                    track=track,
                    method="learned_two_field_target",
                    labels=target_labels,
                    source_temperature=ANCHOR_TEMPERATURE,
                    target_temperature=target_temperature,
                    schedule="direct",
                    error=two_field_error or "two-field checkpoint was not configured",
                    status="two_field_unavailable",
                )
            )
        else:
            try:
                mass, temperature = _predict_two_field(target_labels, two_field_models)
                reduced_seed = _reconstruct_from_mt(target_labels, mass, temperature)
                two_record, _two_state = _solve_attempt(
                    track=track,
                    method="learned_two_field_target",
                    schedule="direct",
                    source_temperature=ANCHOR_TEMPERATURE,
                    target_labels=target_labels,
                    initial_atmosphere=reduced_seed,
                    product_dir=run_root / "products" / "learned_two_field_target",
                    iteration_cap=iteration_cap,
                )
                two_record[
                    "final_atmosphere_difference_to_production_reference"
                ] = _state_difference(
                    _two_state,
                    target_production_states.get(target_temperature),
                )
            except Exception as exc:  # noqa: BLE001
                two_record = _failed_record(
                    track=track,
                    method="learned_two_field_target",
                    labels=target_labels,
                    source_temperature=ANCHOR_TEMPERATURE,
                    target_temperature=target_temperature,
                    schedule="direct",
                    error=exc,
                )
                two_record["status"] = "two_field_prediction_or_reconstruction_failed"
            two_record["initializer_regime"] = "extrapolation"
            records.append(two_record)

        if anchor_production_state is None:
            for method in ("anchor_full_carry", "anchor_reduced_rematerialized"):
                records.append(
                    _record_blocked(
                        track=track,
                        method=method,
                        labels=target_labels,
                        source_temperature=ANCHOR_TEMPERATURE,
                        schedule="direct",
                    )
                )
        else:
            full_seed = _retarget_full_state(
                anchor_production_state, target_templates[target_temperature]
            )
            full_record, _full_state = _solve_attempt(
                track=track,
                method="anchor_full_carry",
                schedule="direct",
                source_temperature=ANCHOR_TEMPERATURE,
                target_labels=target_labels,
                initial_atmosphere=full_seed,
                product_dir=run_root / "products" / "anchor_full_carry",
                iteration_cap=iteration_cap,
            )
            full_record[
                "final_atmosphere_difference_to_production_reference"
            ] = _state_difference(
                _full_state,
                target_production_states.get(target_temperature),
            )
            records.append(full_record)
            try:
                reduced_seed = _reconstruct_from_mt(
                    target_labels,
                    anchor_production_state.column_mass,
                    anchor_production_state.temperature,
                )
                reduced_record, _reduced_state = _solve_attempt(
                    track=track,
                    method="anchor_reduced_rematerialized",
                    schedule="direct",
                    source_temperature=ANCHOR_TEMPERATURE,
                    target_labels=target_labels,
                    initial_atmosphere=reduced_seed,
                    product_dir=run_root / "products" / "anchor_reduced_rematerialized",
                    iteration_cap=iteration_cap,
                )
                reduced_record[
                    "final_atmosphere_difference_to_production_reference"
                ] = _state_difference(
                    _reduced_state,
                    target_production_states.get(target_temperature),
                )
            except Exception as exc:  # noqa: BLE001
                reduced_record = _failed_record(
                    track=track,
                    method="anchor_reduced_rematerialized",
                    labels=target_labels,
                    source_temperature=ANCHOR_TEMPERATURE,
                    target_temperature=target_temperature,
                    schedule="direct",
                    error=exc,
                )
                reduced_record["status"] = "rematerialization_failed"
            records.append(reduced_record)

    # The two continuation policies use the same converged 4000 K production
    # anchor, so their difference is the representation carried between steps.
    records.extend(
        _run_continuation(
            track=track,
            method="continuation_250_full_carry",
            schedule="250K",
            targets=(3750.0, 3500.0),
            mode="full_carry",
            anchor_state=anchor_production_state,
            target_templates=target_templates,
            run_root=run_root,
            iteration_cap=iteration_cap,
        )
    )
    records.extend(
        _run_continuation(
            track=track,
            method="continuation_250_reduced_rematerialized",
            schedule="250K",
            targets=(3750.0, 3500.0),
            mode="reduced_rematerialized",
            anchor_state=anchor_production_state,
            target_templates=target_templates,
            run_root=run_root,
            iteration_cap=iteration_cap,
        )
    )
    records.extend(
        _run_continuation(
            track=track,
            method="continuation_100_full_carry",
            schedule="100K",
            targets=CONTINUATION_100_TARGETS,
            mode="full_carry",
            anchor_state=anchor_production_state,
            target_templates=target_templates,
            run_root=run_root,
            iteration_cap=iteration_cap,
        )
    )
    records.extend(
        _run_continuation(
            track=track,
            method="continuation_100_reduced_rematerialized",
            schedule="100K",
            targets=CONTINUATION_100_TARGETS,
            mode="reduced_rematerialized",
            anchor_state=anchor_production_state,
            target_templates=target_templates,
            run_root=run_root,
            iteration_cap=iteration_cap,
        )
    )

    for record in records:
        record["track_manifest_hash"] = options["manifest_hash"]
    return {
        "track_id": track.track_id,
        "manifest_hash": options["manifest_hash"],
        "track": track.as_json(),
        "references": references,
        "records": records,
    }


def _jsonable(value):
    return _as_plain(value)


def _load_existing_track(path: Path, expected_hash: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if payload.get("manifest_hash") == expected_hash else None


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row)) + "\n")
    temporary.replace(path)


def _spectrum_gate(
    *,
    records: list[dict],
    references: dict[str, dict],
    run_root: Path,
    bar: float,
    workers: int,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Apply the existing 400--900 nm, R=20,000 three-metric gate."""

    from .spectral_gate import gate_one

    metrics: dict[str, dict] = {}
    reference_checks: dict[str, dict] = {}
    spectra_dir = run_root / "spectra"

    for track_id, reference in references.items():
        anchor_baseline = reference.get("anchor_production_product")
        anchor_candidate = reference.get("anchor_marcs_product")
        if anchor_baseline and anchor_candidate:
            try:
                check = gate_one(
                    Path(anchor_baseline).stem,
                    run_root / "products",
                    spectra_dir,
                    wavelength_start_nm=400.0,
                    wavelength_end_nm=900.0,
                    resolution=20000.0,
                    molecular_lines=True,
                    device="cpu",
                    dtype="float64",
                    baseline_arm="anchor_production",
                    candidate_arm="anchor_marcs_reduced",
                )
                check["pass"] = all(
                    check[field]["max"] <= bar
                    for field in ("normalized_flux", "flux_total", "flux_continuum")
                )
            except Exception as exc:  # noqa: BLE001
                check = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            check = {"pass": False, "error": "one or both 4000 K anchors missing"}
        reference_checks[f"{track_id}:4000"] = check

        for temperature_text, target_reference in reference.get(
            "target_products", {}
        ).items():
            baseline = target_reference.get("production")
            candidate = target_reference.get("marcs")
            key = f"{track_id}:{temperature_text}"
            if baseline and candidate:
                try:
                    check = gate_one(
                        Path(baseline).stem,
                        run_root / "products",
                        spectra_dir,
                        wavelength_start_nm=400.0,
                        wavelength_end_nm=900.0,
                        resolution=20000.0,
                        molecular_lines=True,
                        device="cpu",
                        dtype="float64",
                        baseline_arm="production_six_field_target",
                        candidate_arm="marcs_target_reduced",
                    )
                    check["pass"] = all(
                        check[field]["max"] <= bar
                        for field in (
                            "normalized_flux",
                            "flux_total",
                            "flux_continuum",
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    check = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
            else:
                check = {
                    "pass": False,
                    "error": "one or both target reference products missing",
                }
            reference_checks[key] = check

    for record in records:
        product = record.get("product_path")
        method = record.get("method")
        target_temperature = record.get("target_temperature")
        track_id = record.get("track_id")
        if not product or not record.get("survives_solver"):
            continue
        if method == "anchor_production":
            continue
        if method == "anchor_marcs_reduced":
            baseline_arm = "anchor_production"
            candidate_arm = "anchor_marcs_reduced"
            baseline_product = references[track_id].get("anchor_production_product")
            ref_key = f"{track_id}:4000"
        else:
            baseline_arm = "production_six_field_target"
            candidate_arm = method
            baseline_product = references[track_id].get("target_products", {}).get(
                str(float(target_temperature)), {}
            ).get("production")
            ref_key = f"{track_id}:{float(target_temperature)}"
        if not baseline_product:
            record["spectral_pass"] = False
            record["status"] = "reference_unresolved"
            continue
        try:
            gate = gate_one(
                Path(baseline_product).stem,
                run_root / "products",
                spectra_dir,
                wavelength_start_nm=400.0,
                wavelength_end_nm=900.0,
                resolution=20000.0,
                molecular_lines=True,
                device="cpu",
                dtype="float64",
                baseline_arm=baseline_arm,
                candidate_arm=candidate_arm,
            )
            pass_gate = all(
                gate[field]["max"] <= bar
                for field in ("normalized_flux", "flux_total", "flux_continuum")
            )
            gate["pass"] = pass_gate
            metrics[f"{track_id}:{method}:{target_temperature}"] = gate
            reference_resolved = bool(reference_checks.get(ref_key, {}).get("pass"))
            record["spectral_pass"] = bool(pass_gate)
            record["reference_resolved"] = reference_resolved
            record["survives"] = bool(
                record["survives_solver"] and pass_gate and reference_resolved
            )
            record["primary_pass"] = bool(
                record["survives"]
                and record.get("iterations") is not None
                and record["iterations"] <= PRIMARY_ITERATION_CAP
            )
            record["recovered_pass"] = bool(
                record["survives"]
                and record.get("iterations") is not None
                and record["iterations"] <= ITERATION_CAP
            )
            record["status"] = "pass" if record["survives"] else "spectral_fail"
        except Exception as exc:  # noqa: BLE001
            record["spectral_pass"] = False
            record["status"] = "spectral_gate_failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
    return metrics, reference_checks


def _summarize_records(
    records: list[dict],
    *,
    tracks: list[TrackSpec],
    reference_checks: dict[str, dict],
) -> dict:
    direct = [record for record in records if record.get("schedule") == "direct"]
    by_method: dict[str, dict] = {}
    for method in ALL_METHODS + ("anchor_production", "anchor_marcs_reduced"):
        rows = [record for record in records if record.get("method") == method]
        iterations = [row["iterations"] for row in rows if row.get("iterations") is not None]
        seconds = [float(row.get("seconds", 0.0)) for row in rows]
        direct_rows = [row for row in direct if row.get("method") == method]
        by_method[method] = {
            "attempts": len(rows),
            "solver_convergence_rate": (
                sum(bool(row.get("solver_converged")) for row in rows) / len(rows)
                if rows else None
            ),
            "survival_rate": (
                sum(bool(row.get("survives")) for row in rows) / len(rows)
                if rows else None
            ),
            "primary_rate": (
                sum(bool(row.get("primary_pass")) for row in rows) / len(rows)
                if rows else None
            ),
            "recovered_rate": (
                sum(bool(row.get("recovered_pass")) for row in rows) / len(rows)
                if rows else None
            ),
            "mean_iterations": float(np.mean(iterations)) if iterations else None,
            "p90_iterations": float(np.percentile(iterations, 90.0)) if iterations else None,
            "total_seconds": float(np.sum(seconds)),
            "direct_attempts": len(direct_rows),
            "direct_passes": sum(bool(row.get("survives")) for row in direct_rows),
            "direct_max_delta_by_track": {
                track.track_id: max(
                    (
                        float(row["delta_temperature"])
                        for row in direct_rows
                        if row.get("track_id") == track.track_id
                        and row.get("survives")
                    ),
                    default=None,
                )
                for track in tracks
            },
        }

    robust_max: dict[str, float | None] = {}
    for method in DIRECT_METHODS:
        values = []
        for delta in sorted(
            {float(row["delta_temperature"]) for row in direct if row.get("method") == method and row.get("delta_temperature") is not None},
            reverse=True,
        ):
            passed = all(
                any(
                    row.get("method") == method
                    and row.get("track_id") == track.track_id
                    and row.get("delta_temperature") == delta
                    and row.get("survives")
                    for row in direct
                )
                for track in tracks
            )
            if passed:
                values.append(delta)
        robust_max[method] = max(values) if values else None

    continuation_baseline = {
        "continuation_250_full_carry": "anchor_full_carry",
        "continuation_250_reduced_rematerialized": "anchor_reduced_rematerialized",
        "continuation_100_full_carry": "anchor_full_carry",
        "continuation_100_reduced_rematerialized": "anchor_reduced_rematerialized",
    }
    continuation_extra_cost: dict[str, dict] = {}
    for method, baseline_method in continuation_baseline.items():
        path_rows = [row for row in records if row.get("method") == method]
        baseline_rows = [
            row for row in direct
            if row.get("method") == baseline_method
        ]
        by_track = {}
        for track in tracks:
            path = [row for row in path_rows if row.get("track_id") == track.track_id]
            baseline = [
                row for row in baseline_rows if row.get("track_id") == track.track_id
            ]
            path_iterations = sum(
                int(row["iterations"])
                for row in path
                if row.get("iterations") is not None
            )
            baseline_iterations = sum(
                int(row["iterations"])
                for row in baseline
                if row.get("iterations") is not None
            )
            path_seconds = float(sum(float(row.get("seconds", 0.0)) for row in path))
            baseline_seconds = float(
                sum(float(row.get("seconds", 0.0)) for row in baseline)
            )
            by_track[track.track_id] = {
                "continuation_steps": len(path),
                "continuation_seconds": path_seconds,
                "direct_comparison_seconds": baseline_seconds,
                "extra_seconds": path_seconds - baseline_seconds,
                "continuation_iterations": path_iterations,
                "direct_comparison_iterations": baseline_iterations,
                "extra_iterations": path_iterations - baseline_iterations,
            }
        continuation_extra_cost[method] = {
            "baseline_direct_method": baseline_method,
            "by_track": by_track,
            "mean_extra_seconds": float(
                np.mean([value["extra_seconds"] for value in by_track.values()])
            ) if by_track else None,
            "mean_extra_iterations": float(
                np.mean([value["extra_iterations"] for value in by_track.values()])
            ) if by_track else None,
        }

    reference_4000_all = all(
        bool(reference_checks.get(f"{track.track_id}:4000", {}).get("pass"))
        for track in tracks
    )
    methods_passing_500 = [
        method
        for method in DIRECT_METHODS
        if robust_max.get(method) == 500.0 and len(tracks) == 9
    ]
    claim = (
        "9/9 轨道的 500 K direct jump 已通过："
        + ", ".join(methods_passing_500)
        if methods_passing_500
        else "不能据此宣称 9/9 轨道的 500 K direct jump 已稳健通过"
    )
    return {
        "track_count": len(tracks),
        "reference_4000_resolved_all_tracks": reference_4000_all,
        "reference_checks": reference_checks,
        "methods": by_method,
        "robust_max_direct_delta_temperature": robust_max,
        "methods_passing_500k_direct": methods_passing_500,
        "continuation_extra_cost": continuation_extra_cost,
        "claim_guardrail": claim,
    }


def _write_markdown_summary(path: Path, summary: dict) -> None:
    lines = [
        "# Cool-star temperature-step test",
        "",
        f"- Tracks: {summary['track_count']}",
        f"- 4000 K reference resolved for all tracks: {summary['reference_4000_resolved_all_tracks']}",
        f"- Guardrail: {summary['claim_guardrail']}",
        "",
        "| method | direct max step on all tracks | solver rate | survival rate | mean iterations |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in summary["methods"].items():
        robust = summary["robust_max_direct_delta_temperature"].get(method)
        lines.append(
            f"| {method} | {robust if robust is not None else 'n/a'} | "
            f"{values['solver_convergence_rate']} | {values['survival_rate']} | "
            f"{values['mean_iterations']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_experiment(args: argparse.Namespace) -> dict:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    manifest = manifest_payload(
        args.stage, marcs_depth_coordinate=args.marcs_depth_coordinate
    )
    manifest["marcs_grid"] = str(Path(args.marcs_grid).expanduser().resolve())
    manifest["marcs_sha256"] = sha256_file(args.marcs_grid)
    if manifest["marcs_sha256"] != EXPECTED_MARCS_SHA256:
        raise MarcsH5Error(
            f"MARCS SHA-256 mismatch: got {manifest['marcs_sha256']}, "
            f"expected {EXPECTED_MARCS_SHA256}"
        )
    schema = inspect_marcs_grid(
        args.marcs_grid,
        verify_sha256=False,
        expected_sha256=None,
    )
    frozen_hash = manifest_hash(manifest)
    run_root = Path(args.run_root).expanduser().resolve()
    result_root = Path(args.result_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    _write_json(result_root / "manifest.json", {**manifest, "manifest_hash": frozen_hash})

    tracks = build_track_manifest(args.stage)
    options = {
        "run_root": str(run_root),
        "iteration_cap": int(args.iteration_cap),
        "manifest_hash": frozen_hash,
        "two_field_checkpoint_dir": (
            str(Path(args.two_field_checkpoint_dir).expanduser().resolve())
            if args.two_field_checkpoint_dir is not None
            else None
        ),
        "two_field_seeds": tuple(
            int(value) for value in args.two_field_seeds.split(",") if value.strip()
        ),
        "marcs_depth_coordinate": args.marcs_depth_coordinate,
    }
    track_dir = run_root / "tracks"
    track_dir.mkdir(parents=True, exist_ok=True)
    payloads = []
    completed: dict[str, dict] = {}
    for track in tracks:
        path = track_dir / f"{track.track_id}.json"
        existing = _load_existing_track(path, frozen_hash) if args.resume else None
        if existing is not None:
            completed[track.track_id] = existing
            continue
        payloads.append((track.as_json(), schema, options))

    if args.dry_run:
        print(json.dumps({**manifest, "manifest_hash": frozen_hash}, indent=2))
        return {"manifest": manifest, "manifest_hash": frozen_hash, "dry_run": True}

    if args.workers <= 1:
        result_iterator = (_run_track_worker(payload) for payload in payloads)
        for result in result_iterator:
            _write_json(track_dir / f"{result['track_id']}.json", result)
            completed[result["track_id"]] = result
            print(
                f"completed {result['track_id']} ({len(result['records'])} records)",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=context
        ) as executor:
            futures = {
                executor.submit(_run_track_worker, payload): payload[0]["track_id"]
                for payload in payloads
            }
            for future in as_completed(futures):
                result = future.result()
                _write_json(track_dir / f"{result['track_id']}.json", result)
                completed[result["track_id"]] = result
                print(
                    f"completed {result['track_id']} ({len(result['records'])} records)",
                    flush=True,
                )

    records = [
        record
        for track in tracks
        for record in completed[track.track_id]["records"]
    ]
    references = {
        track.track_id: completed[track.track_id]["references"] for track in tracks
    }
    _write_json(run_root / "records.json", records)
    _write_jsonl(run_root / "records.jsonl", records)
    _write_json(run_root / "references.json", references)

    if args.skip_spectra:
        metrics = {}
        reference_checks = {}
    else:
        metrics, reference_checks = _spectrum_gate(
            records=records,
            references=references,
            run_root=run_root,
            bar=float(args.spectral_bar),
            workers=int(args.spectrum_workers),
        )
    summary = _summarize_records(
        records, tracks=tracks, reference_checks=reference_checks
    )
    summary.update(
        {
            "stage": args.stage,
            "manifest_hash": frozen_hash,
            "marcs_grid": str(Path(args.marcs_grid).expanduser().resolve()),
            "marcs_sha256": manifest["marcs_sha256"],
            "iteration_cap": int(args.iteration_cap),
            "primary_iteration_cap": PRIMARY_ITERATION_CAP,
            "spectral_bar": float(args.spectral_bar),
            "spectral_metrics": metrics,
            "skip_spectra": bool(args.skip_spectra),
        }
    )
    _write_json(result_root / "summary.json", summary)
    _write_json(
        result_root / "direct_summary.json",
        {
            "stage": args.stage,
            "manifest_hash": frozen_hash,
            "methods": {
                method: summary["methods"][method] for method in DIRECT_METHODS
            },
            "robust_max_direct_delta_temperature": {
                method: summary["robust_max_direct_delta_temperature"][method]
                for method in DIRECT_METHODS
            },
        },
    )
    _write_json(
        result_root / "continuation_summary.json",
        {
            "stage": args.stage,
            "manifest_hash": frozen_hash,
            "methods": {
                method: summary["methods"][method] for method in CONTINUATION_METHODS
            },
            "extra_cost": summary["continuation_extra_cost"],
        },
    )
    _write_markdown_summary(result_root / "summary.md", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marcs-grid", type=Path, default=DEFAULT_MARCS_GRID)
    parser.add_argument("--stage", choices=("pilot", "confirm"), required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--iteration-cap", type=int, default=ITERATION_CAP)
    parser.add_argument("--spectral-bar", type=float, default=SPECTRAL_BAR)
    parser.add_argument(
        "--marcs-depth-coordinate",
        choices=MARCS_DEPTH_COORDINATES,
        default="log_mass",
        help=(
            "coordinate used to place native MARCS (m,T) on the Payne grid; "
            "tau5000 uses the native optical-depth profile with linear edge "
            "extrapolation"
        ),
    )
    parser.add_argument("--spectrum-workers", type=int, default=1)
    parser.add_argument("--skip-spectra", action="store_true")
    parser.add_argument(
        "--two-field-checkpoint-dir",
        type=Path,
        default=DEFAULT_TWO_FIELD_CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--two-field-seeds",
        default=",".join(str(seed) for seed in DEFAULT_TWO_FIELD_SEEDS),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_experiment(args)
    if summary.get("dry_run"):
        return 0
    print(
        f"wrote cool-star summary for {summary['track_count']} tracks; "
        f"guardrail: {summary['claim_guardrail']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
