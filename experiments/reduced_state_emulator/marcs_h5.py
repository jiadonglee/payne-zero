"""Read native MARCS nodes for the cool-star continuation experiment.

The public ``SDSS_MARCS_atmospheres.h5`` file is a Korg-style tensor.  Its
stored parameter axes are reversed relative to ``grid_parameter_names``:
``(carbon, alpha, metallicity, logg, Teff, quantity, depth)``.  This module
only reads one native node at a time.  It deliberately does not interpolate in
label space; the only interpolation is the monotone 56-to-80 layer conversion
of that node's ``(m, T)`` profile.

MARCS pressure, number densities, optical depth, and height are retained as
diagnostics.  Payne-Zero receives only the converted ``(m, T)`` pair and the
requested labels; the other four fields are rebuilt by
``reconstruct_full_atmosphere``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.interpolate import PchipInterpolator

from bench.labels import StellarLabels
from reduced_state.reconstruct import ReducedAtmosphere


EXPECTED_MARCS_SHA256 = (
    "9e1b13ec5698e7ca8e068f8e6505ca49749be2fe79a67f25ef0a96077b5f3139"
)
EXPECTED_PARAMETER_NAMES = (
    "Teff",
    "logg",
    "metallicity",
    "alpha",
    "carbon",
)
HDF5_PARAMETER_NAMES = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "carbon_enhancement",
)
HDF5_STORAGE_AXES = (
    "carbon_enhancement",
    "alpha_enhancement",
    "metallicity",
    "log_surface_gravity",
    "effective_temperature",
)
MARCS_QUANTITY_NAMES = (
    "temperature",
    "ln_electron_number_density",
    "ln_total_number_density",
    "tau_5000",
    "asinh_height",
)
PAYNE_LAYER_COUNT = 80
PAYNE_LOG_TAU_START = -6.875
PAYNE_LOG_TAU_STEP = 0.125
BOLTZMANN_CGS = 1.380649e-16
MARCS_DEPTH_COORDINATES = ("log_mass", "tau5000")


class MarcsH5Error(ValueError):
    """Raised when the MARCS file or requested node is not usable."""


@dataclass(frozen=True)
class MarcsGridSchema:
    """Small, serializable description of the HDF5 tensor schema."""

    path: Path
    sha256: str | None
    parameter_names: tuple[str, ...]
    storage_axes: tuple[str, ...]
    quantity_names: tuple[str, ...]
    grid_values: dict[str, np.ndarray]
    grid_shape: tuple[int, ...]


@dataclass(frozen=True)
class MarcsNode:
    """One native MARCS node and the Payne-Zero reduced-state conversion."""

    labels: StellarLabels
    carbon_enhancement: float
    indices: tuple[int, ...]
    native_temperature: np.ndarray
    native_electron_density: np.ndarray
    native_total_number_density: np.ndarray
    native_tau_5000: np.ndarray
    native_height: np.ndarray
    native_gas_pressure: np.ndarray
    native_column_mass: np.ndarray
    reduced_column_mass: np.ndarray
    reduced_temperature: np.ndarray
    source_path: Path
    source_sha256: str | None

    @property
    def reduced(self) -> ReducedAtmosphere:
        """Return only the two fields allowed into the Payne-Zero path."""

        return ReducedAtmosphere(
            column_mass=self.reduced_column_mass.copy(),
            temperature=self.reduced_temperature.copy(),
            labels=self.labels.as_kwargs(),
        )

    @property
    def native_fields(self) -> dict[str, np.ndarray]:
        """Decoded MARCS fields for diagnostics, never for reconstruction."""

        return {
            "temperature": self.native_temperature.copy(),
            "electron_density": self.native_electron_density.copy(),
            "total_number_density": self.native_total_number_density.copy(),
            "tau_5000": self.native_tau_5000.copy(),
            "height": self.native_height.copy(),
            "gas_pressure": self.native_gas_pressure.copy(),
            "column_mass": self.native_column_mass.copy(),
        }


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Return a deterministic SHA-256 digest without loading the HDF5 file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_h5py():
    try:
        import h5py  # type: ignore
    except (ImportError, OSError, ValueError) as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The cool-star MARCS experiment needs optional dependency h5py>=3.11; "
            "install it with `pip install -e '.[cool-test]'` in the active "
            "environment."
        ) from exc
    return h5py


def _decode_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8")
    return str(value)


def _read_parameter_names(handle) -> tuple[str, ...]:
    values = np.asarray(handle["grid_parameter_names"])
    return tuple(_decode_string(value) for value in values.tolist())


def _read_grid_values(handle, parameter_names: tuple[str, ...]) -> dict[str, np.ndarray]:
    group = handle["grid_values"]
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(parameter_names, start=1):
        key = str(index)
        if key not in group:
            raise MarcsH5Error(f"missing HDF5 grid_values/{key} for {name}")
        values = np.asarray(group[key], dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise MarcsH5Error(f"grid axis {name} is not a finite non-empty vector")
        if np.any(np.diff(values) <= 0.0):
            raise MarcsH5Error(f"grid axis {name} is not strictly increasing")
        semantic_name = HDF5_PARAMETER_NAMES[index - 1]
        result[semantic_name] = values
    return result


def inspect_marcs_grid(
    path: Path,
    *,
    verify_sha256: bool = True,
    expected_sha256: str | None = EXPECTED_MARCS_SHA256,
) -> MarcsGridSchema:
    """Validate and inspect the small HDF5 metadata without reading the tensor."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path) if verify_sha256 else None
    if expected_sha256 is not None and digest != expected_sha256:
        if digest is None:
            digest = sha256_file(path)
        raise MarcsH5Error(
            f"MARCS SHA-256 mismatch for {path}: got {digest}, "
            f"expected {expected_sha256}"
        )

    h5py = _require_h5py()
    with h5py.File(path, "r") as handle:
        parameter_names = _read_parameter_names(handle)
        if parameter_names != EXPECTED_PARAMETER_NAMES:
            raise MarcsH5Error(
                f"unexpected MARCS parameter names {parameter_names!r}; "
                f"expected {EXPECTED_PARAMETER_NAMES!r}"
            )
        grid_values = _read_grid_values(handle, parameter_names)
        if "grid" not in handle:
            raise MarcsH5Error("MARCS HDF5 file has no /grid dataset")
        grid_shape = tuple(int(value) for value in handle["grid"].shape)
        expected_shape = tuple(
            grid_values[name].size for name in HDF5_STORAGE_AXES
        ) + (len(MARCS_QUANTITY_NAMES), 56)
        if grid_shape != expected_shape:
            raise MarcsH5Error(
                f"unexpected /grid shape {grid_shape}; expected {expected_shape}"
            )
    return MarcsGridSchema(
        path=path,
        sha256=digest,
        parameter_names=parameter_names,
        storage_axes=HDF5_STORAGE_AXES,
        quantity_names=MARCS_QUANTITY_NAMES,
        grid_values=grid_values,
        grid_shape=grid_shape,
    )


def _nearest_native_index(
    values: np.ndarray,
    requested: float,
    *,
    name: str,
    tolerance: float,
) -> int:
    differences = np.abs(values - float(requested))
    index = int(np.argmin(differences))
    scale = max(1.0, abs(float(values[index])), abs(float(requested)))
    if float(differences[index]) > tolerance * scale:
        raise MarcsH5Error(
            f"{name}={requested:g} is off the native MARCS grid; nearest node "
            f"is {values[index]:g}"
        )
    return index


def _native_indices(
    schema: MarcsGridSchema,
    labels: StellarLabels,
    *,
    carbon_enhancement: float,
    tolerance: float,
) -> tuple[int, ...]:
    requested = {
        "effective_temperature": labels.effective_temperature,
        "log_surface_gravity": labels.log_surface_gravity,
        "metallicity": labels.metallicity,
        "alpha_enhancement": labels.alpha_enhancement,
        "carbon_enhancement": carbon_enhancement,
    }
    semantic_indices = {
        name: _nearest_native_index(
            schema.grid_values[name], requested[name], name=name, tolerance=tolerance
        )
        for name in HDF5_PARAMETER_NAMES
    }
    return tuple(semantic_indices[name] for name in HDF5_STORAGE_AXES)


def _as_increasing_profile(
    column_mass: np.ndarray,
    *profiles: np.ndarray,
) -> tuple[np.ndarray, ...]:
    mass = np.asarray(column_mass, dtype=np.float64)
    if mass.ndim != 1 or mass.size < 2:
        raise MarcsH5Error("MARCS column mass must be a one-dimensional profile")
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise MarcsH5Error("MARCS column mass must be finite and positive")
    differences = np.diff(mass)
    if np.all(differences > 0.0):
        order = np.arange(mass.size)
    elif np.all(differences < 0.0):
        order = np.arange(mass.size - 1, -1, -1)
    else:
        raise MarcsH5Error("MARCS column mass is not strictly monotonic")
    return tuple(np.asarray(profile, dtype=np.float64)[order] for profile in (mass, *profiles))


def _convert_to_payne_layers(
    native_column_mass: np.ndarray,
    native_temperature: np.ndarray,
    *,
    native_tau_5000: np.ndarray | None = None,
    layer_count: int = PAYNE_LAYER_COUNT,
    depth_coordinate: str = "log_mass",
) -> tuple[np.ndarray, np.ndarray]:
    if layer_count < 2:
        raise ValueError("layer_count must be at least 2")
    if depth_coordinate not in MARCS_DEPTH_COORDINATES:
        raise ValueError(
            f"depth_coordinate must be one of {MARCS_DEPTH_COORDINATES!r}, "
            f"got {depth_coordinate!r}"
        )
    profiles = [native_temperature]
    if native_tau_5000 is not None:
        profiles.append(native_tau_5000)
    ordered = _as_increasing_profile(native_column_mass, *profiles)
    mass = ordered[0]
    temperature = ordered[1]
    if temperature.shape != mass.shape:
        raise MarcsH5Error("MARCS temperature and column-mass shapes differ")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise MarcsH5Error("MARCS temperature must be finite and positive")
    log_temperature = np.log(temperature)

    if depth_coordinate == "log_mass":
        source_coordinate = np.log(mass)
        target_coordinate = np.linspace(
            source_coordinate[0], source_coordinate[-1], int(layer_count)
        )
        converted_log_mass = target_coordinate
        interpolator = PchipInterpolator(
            source_coordinate, log_temperature, extrapolate=False
        )
        target_log_temperature = np.asarray(
            interpolator(target_coordinate), dtype=np.float64
        )
    else:
        if native_tau_5000 is None:
            raise MarcsH5Error(
                "tau5000 conversion requires the native MARCS tau_5000 profile"
            )
        tau = ordered[2]
        if (
            np.any(~np.isfinite(tau))
            or np.any(tau <= 0.0)
            or np.any(np.diff(tau) <= 0.0)
        ):
            raise MarcsH5Error(
                "MARCS tau_5000 must be finite, positive, and strictly increasing"
            )
        source_coordinate = np.log10(tau)
        target_coordinate = PAYNE_LOG_TAU_START + PAYNE_LOG_TAU_STEP * np.arange(
            int(layer_count), dtype=np.float64
        )

        def linear_edge_extrapolating_pchip(values: np.ndarray) -> np.ndarray:
            """PCHIP inside the MARCS range, linear in log-depth outside it."""

            spline = PchipInterpolator(source_coordinate, values, extrapolate=False)
            result = np.empty_like(target_coordinate)
            inside = (target_coordinate >= source_coordinate[0]) & (
                target_coordinate <= source_coordinate[-1]
            )
            result[inside] = spline(target_coordinate[inside])
            left_slope = (values[1] - values[0]) / (
                source_coordinate[1] - source_coordinate[0]
            )
            right_slope = (values[-1] - values[-2]) / (
                source_coordinate[-1] - source_coordinate[-2]
            )
            left = target_coordinate < source_coordinate[0]
            right = target_coordinate > source_coordinate[-1]
            result[left] = values[0] + left_slope * (
                target_coordinate[left] - source_coordinate[0]
            )
            result[right] = values[-1] + right_slope * (
                target_coordinate[right] - source_coordinate[-1]
            )
            return result

        converted_log_mass = linear_edge_extrapolating_pchip(np.log(mass))
        target_log_temperature = linear_edge_extrapolating_pchip(log_temperature)

    converted_mass = np.exp(converted_log_mass)
    converted_temperature = np.exp(target_log_temperature)
    if (
        np.any(~np.isfinite(converted_mass))
        or np.any(~np.isfinite(converted_temperature))
        or np.any(np.diff(converted_mass) <= 0.0)
        or np.any(converted_mass <= 0.0)
        or np.any(converted_temperature <= 0.0)
    ):
        raise MarcsH5Error("56-to-80 MARCS conversion produced an invalid profile")
    return converted_mass, converted_temperature


def load_marcs_node(
    path: Path,
    labels: StellarLabels,
    *,
    carbon_enhancement: float = 0.0,
    layer_count: int = PAYNE_LAYER_COUNT,
    tolerance: float = 1.0e-8,
    verify_sha256: bool = True,
    expected_sha256: str | None = EXPECTED_MARCS_SHA256,
    schema: MarcsGridSchema | None = None,
    depth_coordinate: str = "log_mass",
) -> MarcsNode:
    """Load one native node and convert only its ``(m,T)`` to 80 layers."""

    labels = labels if isinstance(labels, StellarLabels) else StellarLabels(**dict(labels))
    schema = schema or inspect_marcs_grid(
        path, verify_sha256=verify_sha256, expected_sha256=expected_sha256
    )
    indices = _native_indices(
        schema,
        labels,
        carbon_enhancement=float(carbon_enhancement),
        tolerance=float(tolerance),
    )
    h5py = _require_h5py()
    with h5py.File(schema.path, "r") as handle:
        # This is a single 5 x 56 node read, not a 619 MB grid read.
        encoded = np.asarray(handle["grid"][indices], dtype=np.float64)
    if encoded.shape != (len(MARCS_QUANTITY_NAMES), 56):
        raise MarcsH5Error(
            f"native MARCS node has shape {encoded.shape}; expected (5, 56)"
        )
    if np.any(~np.isfinite(encoded)):
        raise MarcsH5Error("native MARCS node contains non-finite encoded values")

    temperature = encoded[0]
    electron_density = np.exp(encoded[1])
    total_number_density = np.exp(encoded[2])
    tau_5000 = encoded[3]
    height = np.sinh(encoded[4])
    gas_pressure = total_number_density * BOLTZMANN_CGS * temperature
    gravity = 10.0 ** float(labels.log_surface_gravity)
    column_mass = gas_pressure / gravity

    (
        column_mass,
        temperature,
        electron_density,
        total_number_density,
        tau_5000,
        height,
        gas_pressure,
    ) = _as_increasing_profile(
        column_mass,
        temperature,
        electron_density,
        total_number_density,
        tau_5000,
        height,
        gas_pressure,
    )
    for name, values in {
        "temperature": temperature,
        "electron_density": electron_density,
        "total_number_density": total_number_density,
        "tau_5000": tau_5000,
        "height": height,
        "gas_pressure": gas_pressure,
    }.items():
        if np.any(~np.isfinite(values)):
            raise MarcsH5Error(f"decoded MARCS {name} contains non-finite values")
    if np.any(electron_density <= 0.0) or np.any(total_number_density <= 0.0):
        raise MarcsH5Error("decoded MARCS number densities must be positive")
    if np.any(gas_pressure <= 0.0):
        raise MarcsH5Error("decoded MARCS gas pressure must be positive")

    reduced_mass, reduced_temperature = _convert_to_payne_layers(
        column_mass,
        temperature,
        native_tau_5000=tau_5000,
        layer_count=layer_count,
        depth_coordinate=depth_coordinate,
    )
    return MarcsNode(
        labels=labels,
        carbon_enhancement=float(carbon_enhancement),
        indices=indices,
        native_temperature=temperature,
        native_electron_density=electron_density,
        native_total_number_density=total_number_density,
        native_tau_5000=tau_5000,
        native_height=height,
        native_gas_pressure=gas_pressure,
        native_column_mass=column_mass,
        reduced_column_mass=reduced_mass,
        reduced_temperature=reduced_temperature,
        source_path=schema.path,
        source_sha256=schema.sha256,
    )


def labels_from_mapping(values: Mapping[str, float]) -> StellarLabels:
    """Build public five-label coordinates from a JSON-friendly mapping."""

    return StellarLabels(
        effective_temperature=float(values["effective_temperature"]),
        log_surface_gravity=float(values["log_surface_gravity"]),
        metallicity=float(values["metallicity"]),
        alpha_enhancement=float(values["alpha_enhancement"]),
        microturbulence_km_s=float(values["microturbulence_km_s"]),
    )


__all__ = [
    "BOLTZMANN_CGS",
    "EXPECTED_MARCS_SHA256",
    "HDF5_PARAMETER_NAMES",
    "HDF5_STORAGE_AXES",
    "MARCS_QUANTITY_NAMES",
    "MARCS_DEPTH_COORDINATES",
    "MarcsGridSchema",
    "MarcsH5Error",
    "MarcsNode",
    "PAYNE_LAYER_COUNT",
    "PAYNE_LOG_TAU_START",
    "PAYNE_LOG_TAU_STEP",
    "inspect_marcs_grid",
    "labels_from_mapping",
    "load_marcs_node",
    "sha256_file",
]
