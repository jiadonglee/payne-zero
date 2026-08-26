"""Learned `labels -> (m, T)(tau)` initializer, the reduced-state analogue of
the production six-field emulator.

Part 2 (`notes/reduced_state_progress.md`) established that *exact truth*
`(m, T)` reconstructs a solver-compatible atmosphere: 100% convergence, mean
3.33 iterations against the six-field oracle's 3.23. That closed the question
"is `(m, T)` sufficient information". It left open the only question that
decides whether the reduced state is deployable: **can a network predict
`(m, T)` accurately enough that the reconstruction still lands in the basin?**

This module is the predictor. It is deliberately not a "continuous" coordinate
network: Part 3 swept intermediate-grid resolution over seven orders of
magnitude of representation error and found restart behaviour flat within
sampling noise, so depth-continuity has no measured benefit to buy. What is
retained from the original Part 4/5 brief is the part the evidence still
supports -- predicting two fields instead of six, on the production grid, with
monotonicity guaranteed by construction rather than by a runtime guard.

Two parameterizations are provided so Part 5's ablation is a flag, not a
rewrite:

``monotone=True``   log10 m = m0 + cumsum(eps + softplus(raw));   strictly
                    increasing for any raw, so ``_seed_atmosphere``'s
                    ``diff(column_mass) <= 0`` guard can never fire.
``monotone=False``  log10 m predicted directly, 80 independent outputs.

The plan's amendment A4 argued a softplus is unnecessary for the *six-field*
decoder because its decode is a cumulative sum of ``10**c``, positive by
construction. That argument does not transfer: predicting log10 m directly has
no such structure, so the guard is real here. The ablation measures whether it
matters in practice.

Everything is float64. The production initializer's float32 batched matmul is
not reduction-order stable, and the deck quantization amplifies a 5e-6 upstream
difference to ~6e-4 -- the convergence threshold's own scale
(`solver-in-the-loop-progress.md` Sec 4.1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)

LAYER_COUNT = 80
INCREMENT_FLOOR = 1.0e-6
PHYSICAL_PARAMETERIZATION = "grey_temperature_mass_increment_v1"


def label_features(labels: np.ndarray) -> np.ndarray:
    """Map raw labels to network inputs.

    ``5040 / Teff`` rather than Teff is the production initializer's own
    feature (`warm_start.py`); it linearizes the Saha/Boltzmann exponentials
    that dominate the atmosphere's temperature response. The remaining four
    labels enter directly -- they are already O(1) and roughly linear in their
    physical effect.
    """

    labels = np.asarray(labels, dtype=np.float64)
    theta = 5040.0 / labels[:, 0]
    return np.column_stack([theta, labels[:, 1:]])


@dataclass(frozen=True)
class Standardization:
    """Train-only feature and target moments. Applied in float64."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    log_temperature_mean: np.ndarray
    log_temperature_std: np.ndarray
    log_column_mass_mean: np.ndarray
    log_column_mass_std: np.ndarray

    def as_dict(self) -> dict:
        return {k: np.asarray(v).tolist() for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, payload: dict) -> "Standardization":
        return cls(**{k: np.asarray(v, dtype=np.float64) for k, v in payload.items()})


def fit_standardization(
    features: np.ndarray, log_column_mass: np.ndarray, log_temperature: np.ndarray
) -> Standardization:
    """Per-layer target moments; the profiles' scale varies strongly with depth."""

    return Standardization(
        feature_mean=features.mean(axis=0),
        feature_std=features.std(axis=0) + 1.0e-12,
        log_temperature_mean=log_temperature.mean(axis=0),
        log_temperature_std=log_temperature.std(axis=0) + 1.0e-12,
        log_column_mass_mean=log_column_mass.mean(axis=0),
        log_column_mass_std=log_column_mass.std(axis=0) + 1.0e-12,
    )


def production_tau_grid(layer_count: int = LAYER_COUNT) -> np.ndarray:
    """Return the fixed Rosseland grid used by the production solver."""

    return 10.0 ** (-6.875 + 0.125 * np.arange(int(layer_count), dtype=np.float64))


def grey_temperature(
    effective_temperature: np.ndarray | float,
    optical_depth: np.ndarray,
) -> np.ndarray:
    """Return the Eddington grey temperature used by the six-field decoder."""

    return (
        np.asarray(effective_temperature, dtype=np.float64)[..., None]
        * (0.75 * (np.asarray(optical_depth, dtype=np.float64) + 2.0 / 3.0)) ** 0.25
    )


@dataclass(frozen=True)
class PhysicalStandardization:
    """Train-only moments for grey-temperature and mass-increment coordinates."""

    feature_mean: np.ndarray
    feature_std: np.ndarray
    log_temperature_ratio_mean: np.ndarray
    log_temperature_ratio_std: np.ndarray
    log_mass_increment_mean: np.ndarray
    log_mass_increment_std: np.ndarray
    log_column_mass_mean: np.ndarray
    log_column_mass_std: np.ndarray

    def as_dict(self) -> dict:
        return {key: np.asarray(value).tolist() for key, value in self.__dict__.items()}

    @classmethod
    def from_dict(cls, payload: dict) -> "PhysicalStandardization":
        payload = dict(payload)
        # Checkpoints made before the cumulative-profile loss did not need
        # these moments for inference. Keep them loadable for comparison.
        payload.setdefault(
            "log_column_mass_mean",
            np.zeros_like(payload["log_mass_increment_mean"], dtype=np.float64),
        )
        payload.setdefault(
            "log_column_mass_std",
            np.ones_like(payload["log_mass_increment_std"], dtype=np.float64),
        )
        return cls(
            **{
                key: np.asarray(value, dtype=np.float64)
                for key, value in payload.items()
            }
        )


def fit_physical_standardization(
    features: np.ndarray,
    column_mass: np.ndarray,
    temperature: np.ndarray,
    effective_temperature: np.ndarray,
    optical_depth: np.ndarray | None = None,
) -> PhysicalStandardization:
    """Fit moments for the physically better-conditioned two-field targets."""

    tau = production_tau_grid() if optical_depth is None else np.asarray(optical_depth)
    mass = np.asarray(column_mass, dtype=np.float64)
    temp = np.asarray(temperature, dtype=np.float64)
    increments = np.empty_like(mass)
    increments[:, 0] = mass[:, 0]
    increments[:, 1:] = np.diff(mass, axis=1)
    if np.any(increments <= 0.0):
        raise ValueError("column_mass must have strictly positive increments")
    log_temperature_ratio = np.log10(
        temp / grey_temperature(effective_temperature, tau)
    )
    log_mass_increment = np.log10(increments)
    log_column_mass = np.log10(mass)
    return PhysicalStandardization(
        feature_mean=np.asarray(features, dtype=np.float64).mean(axis=0),
        feature_std=np.asarray(features, dtype=np.float64).std(axis=0) + 1.0e-12,
        log_temperature_ratio_mean=log_temperature_ratio.mean(axis=0),
        log_temperature_ratio_std=log_temperature_ratio.std(axis=0) + 1.0e-12,
        log_mass_increment_mean=log_mass_increment.mean(axis=0),
        log_mass_increment_std=log_mass_increment.std(axis=0) + 1.0e-12,
        log_column_mass_mean=log_column_mass.mean(axis=0),
        log_column_mass_std=log_column_mass.std(axis=0) + 1.0e-12,
    )


class ReducedStateEmulator(nn.Module):
    """5 labels -> (log10 m, log10 T) on the fixed 80-point production grid."""

    def __init__(
        self,
        *,
        width: int = 512,
        depth: int = 4,
        monotone: bool = True,
        layer_count: int = LAYER_COUNT,
    ) -> None:
        super().__init__()
        self.monotone = bool(monotone)
        self.layer_count = int(layer_count)

        layers: list[nn.Module] = [nn.Linear(5, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.temperature_head = nn.Linear(width, self.layer_count)
        # monotone: one absolute anchor + (layer_count - 1) positive increments.
        self.column_mass_head = nn.Linear(width, self.layer_count)
        self.to(torch.float64)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return standardized log10 T and *unstandardized* log10 m.

        The asymmetry is deliberate. Temperature is predicted in standardized
        space and destandardized outside; column mass must be assembled in
        physical log space for the cumulative sum to mean anything, so the
        monotone branch destandardizes internally via ``decode_column_mass``.
        """

        hidden = self.trunk(features)
        return self.temperature_head(hidden), self.column_mass_head(hidden)


def decode_column_mass(
    raw: torch.Tensor, standardization: Standardization, *, monotone: bool
) -> torch.Tensor:
    """Raw head output -> log10 column mass, monotone by construction if asked."""

    mean = torch.as_tensor(standardization.log_column_mass_mean, dtype=raw.dtype)
    std = torch.as_tensor(standardization.log_column_mass_std, dtype=raw.dtype)
    if not monotone:
        return raw * std + mean

    # Anchor the first layer in standardized space, then walk up in physical
    # log space with strictly positive steps. The reference increments (the
    # train-set mean profile's own differences) keep the softplus operating
    # near its linear region instead of having to learn the ~0.1 dex per-layer
    # scale from zero.
    anchor = raw[:, :1] * std[:1] + mean[:1]
    reference = torch.diff(mean)
    steps = INCREMENT_FLOOR + torch.nn.functional.softplus(raw[:, 1:] + _inverse_softplus(reference))
    return torch.cat([anchor, anchor + torch.cumsum(steps, dim=1)], dim=1)


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """softplus^-1, stable for the large-positive values that dominate here."""

    return value + torch.log(-torch.expm1(-value))


def decode_temperature(
    raw: torch.Tensor, standardization: Standardization
) -> torch.Tensor:
    mean = torch.as_tensor(standardization.log_temperature_mean, dtype=raw.dtype)
    std = torch.as_tensor(standardization.log_temperature_std, dtype=raw.dtype)
    return raw * std + mean


def predict_reduced_state(
    model: ReducedStateEmulator,
    standardization: Standardization,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched inference. Returns (column_mass, temperature) in physical units."""

    features = label_features(labels)
    standardized = (features - standardization.feature_mean) / standardization.feature_std
    with torch.no_grad():
        raw_temperature, raw_column_mass = model(
            torch.as_tensor(standardized, dtype=torch.float64)
        )
        log_temperature = decode_temperature(raw_temperature, standardization)
        log_column_mass = decode_column_mass(
            raw_column_mass, standardization, monotone=model.monotone
        )
        return (
            np.power(10.0, log_column_mass.numpy()),
            np.power(10.0, log_temperature.numpy()),
        )


class PhysicalReducedStateEmulator(nn.Module):
    """Predict grey-temperature and positive mass-increment coordinates."""

    def __init__(
        self,
        *,
        width: int = 512,
        depth: int = 4,
        layer_count: int = LAYER_COUNT,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.layer_count = int(layer_count)
        layers: list[nn.Module] = [nn.Linear(5, width), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.SiLU()]
        self.trunk = nn.Sequential(*layers)
        self.temperature_ratio_head = nn.Linear(width, self.layer_count)
        self.mass_increment_head = nn.Linear(width, self.layer_count)
        self.to(dtype=dtype)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.temperature_ratio_head(hidden), self.mass_increment_head(hidden)


def decode_physical_state(
    raw_temperature_ratio: torch.Tensor,
    raw_mass_increment: torch.Tensor,
    standardization: PhysicalStandardization,
    effective_temperature: torch.Tensor,
    optical_depth: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode the physical coordinates into positive ``(m, T)`` profiles."""

    tau = production_tau_grid() if optical_depth is None else np.asarray(optical_depth)
    dtype = raw_temperature_ratio.dtype
    device = raw_temperature_ratio.device
    tau_t = torch.as_tensor(tau, dtype=dtype, device=device)
    ratio_mean = torch.as_tensor(
        standardization.log_temperature_ratio_mean, dtype=dtype, device=device
    )
    ratio_std = torch.as_tensor(
        standardization.log_temperature_ratio_std, dtype=dtype, device=device
    )
    mass_mean = torch.as_tensor(
        standardization.log_mass_increment_mean, dtype=dtype, device=device
    )
    mass_std = torch.as_tensor(
        standardization.log_mass_increment_std, dtype=dtype, device=device
    )
    log_ratio = raw_temperature_ratio * ratio_std + ratio_mean
    log_increment = raw_mass_increment * mass_std + mass_mean
    grey = effective_temperature[:, None] * (
        0.75 * (tau_t[None, :] + 2.0 / 3.0)
    ) ** 0.25
    temperature = grey * 10.0 ** torch.clamp(log_ratio, -3.0, 3.0)
    increments = 10.0 ** torch.clamp(log_increment, -30.0, 30.0)
    column_mass = torch.cumsum(increments, dim=1)
    return column_mass, temperature


def predict_physical_state(
    model: PhysicalReducedStateEmulator,
    standardization: PhysicalStandardization,
    labels: np.ndarray,
    optical_depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched physical-coordinate inference in solver units."""

    labels = np.asarray(labels, dtype=np.float64)
    features = label_features(labels)
    standardized = (
        features - standardization.feature_mean
    ) / standardization.feature_std
    with torch.no_grad():
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        raw_ratio, raw_increment = model(
            torch.as_tensor(standardized, dtype=model_dtype, device=model_device)
        )
        column_mass, temperature = decode_physical_state(
            raw_ratio,
            raw_increment,
            standardization,
            torch.as_tensor(labels[:, 0], dtype=model_dtype, device=model_device),
            optical_depth=optical_depth,
        )
    return column_mass.cpu().numpy(), temperature.cpu().numpy()


def save_checkpoint(
    path, model: ReducedStateEmulator, standardization: Standardization, meta: dict
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "standardization": standardization.as_dict(),
            "monotone": model.monotone,
            "layer_count": model.layer_count,
            "width": model.temperature_head.in_features,
            "depth": sum(1 for m in model.trunk if isinstance(m, nn.Linear)),
            "meta": meta,
        },
        path,
    )


def load_checkpoint(path) -> tuple[ReducedStateEmulator, Standardization, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = ReducedStateEmulator(
        width=payload["width"],
        depth=payload["depth"],
        monotone=payload["monotone"],
        layer_count=payload["layer_count"],
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return (
        model,
        Standardization.from_dict(payload["standardization"]),
        payload.get("meta", {}),
    )


def save_physical_checkpoint(
    path,
    model: PhysicalReducedStateEmulator,
    standardization: PhysicalStandardization,
    meta: dict,
) -> None:
    """Save a self-describing physical-coordinate checkpoint."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": PHYSICAL_PARAMETERIZATION,
            "parameterization": PHYSICAL_PARAMETERIZATION,
            "state_dict": model.state_dict(),
            "standardization": standardization.as_dict(),
            "layer_count": model.layer_count,
            "width": model.temperature_ratio_head.in_features,
            "depth": sum(1 for module in model.trunk if isinstance(module, nn.Linear)),
            "meta": meta,
        },
        path,
    )


def load_physical_checkpoint(
    path,
) -> tuple[PhysicalReducedStateEmulator, PhysicalStandardization, dict]:
    """Load and validate a physical-coordinate checkpoint."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("parameterization") != PHYSICAL_PARAMETERIZATION:
        raise ValueError(
            f"{path} is not a {PHYSICAL_PARAMETERIZATION} checkpoint"
        )
    model = PhysicalReducedStateEmulator(
        width=payload["width"],
        depth=payload["depth"],
        layer_count=payload["layer_count"],
        dtype=torch.float64,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return (
        model,
        PhysicalStandardization.from_dict(payload["standardization"]),
        payload.get("meta", {}),
    )


def load_corpus(path) -> dict:
    """Read the five-label truth corpus into arrays this module's shapes expect."""

    with np.load(Path(path), allow_pickle=False) as data:
        labels_json = [json.loads(str(entry)) for entry in data["labels_json"]]
        profiles = np.asarray(data["atmosphere_profiles"], dtype=np.float64)
        iterations = np.asarray(data["iterations_to_convergence"], dtype=np.int64)
        slugs = [str(entry) for entry in data["slugs"]]
    labels = np.array(
        [[entry[field] for field in LABEL_FIELDS] for entry in labels_json],
        dtype=np.float64,
    )
    roles = np.array(
        [entry.get("intended_outer_role", "train") for entry in labels_json]
    )
    return {
        "labels": labels,
        "labels_json": labels_json,
        "roles": roles,
        "column_mass": profiles[:, :, 0],
        "temperature": profiles[:, :, 1],
        "iterations": iterations,
        "slugs": slugs,
    }
