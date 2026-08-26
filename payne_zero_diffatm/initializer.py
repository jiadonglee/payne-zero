"""Batched, differentiable form of the released atmosphere initializer.

Reproduces ``payne_zero_atmosphere.warm_start.AtmosphereInitializer.predict``
(``warm_start.py:614-693``) as a torch module that is differentiable in the
network parameters and evaluates a whole batch of stars at once. The released
weights load unchanged, so solver-in-the-loop training starts from the shipped
model rather than from scratch.

Fidelity notes, which matter because the reference is the thing being matched:

* The reference runs the multilayer perceptron in float32 and only then widens
  to float64 for the principal-component decode. That is reproduced exactly;
  doing the whole chain in float64 shifts the output at the 1e-7 level.
* The decode clamps are kept as they are in the reference. They sit far outside
  the range a sane prediction occupies, so they are guards rather than active
  nonlinearities, but removing them would change behaviour at the edges.
* Column mass is a cumulative sum of ``10**c``, which is positive, so the
  sequence is increasing by construction. The reference's non-monotonic guard
  (``warm_start.py:686``) can only fire when an increment underflows against a
  much larger running sum; ``minimum_relative_increment`` handles that case
  smoothly instead of raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from payne_zero_atmosphere.data_files import atmosphere_emulator_dir


# Order of the six decoded fields, matching
# checkpoint["coordinates"]["target_fields"].
OUTPUT_FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)

# Per-coordinate clamps from warm_start.py:678-683.
_COLUMN_MASS_CLAMP = 30.0
_TEMPERATURE_CLAMP = 3.0
_POWER_CLAMP = 30.0
_ACCELERATION_CLAMP = 20.0


@dataclass(frozen=True)
class InitializerPrediction:
    """One batch of decoded atmospheres, all shaped ``(batch, layers)``."""

    column_mass: torch.Tensor
    temperature: torch.Tensor
    gas_pressure: torch.Tensor
    electron_density: torch.Tensor
    rosseland_opacity: torch.Tensor
    radiative_acceleration: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {field: getattr(self, field) for field in OUTPUT_FIELDS}

    def stack(self) -> torch.Tensor:
        """Return ``(batch, layers, 6)`` in ``OUTPUT_FIELDS`` order."""
        return torch.stack([getattr(self, f) for f in OUTPUT_FIELDS], dim=-1)


def default_checkpoint_path(family: str = "five_label") -> Path:
    return atmosphere_emulator_dir() / family / "checkpoint.pt"


class DifferentiableInitializer(nn.Module):
    """The released initializer as a trainable, batched torch module.

    ``forward`` takes public labels and returns a decoded atmosphere batch. The
    network parameters are ordinary leaves, so an optimizer can update them
    directly; everything else (principal-component basis, normalizations, the
    standard optical-depth grid) is registered as a buffer and stays fixed.
    """

    def __init__(
        self,
        checkpoint: dict,
        *,
        device: torch.device | str = "cpu",
        minimum_relative_increment: float = 1.0e-12,
        network_dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.family_fields: tuple[str, ...] = tuple(checkpoint["labels"]["fields"])
        self.minimum_relative_increment = float(minimum_relative_increment)
        self.network_dtype = network_dtype

        config = checkpoint["model"]["config"]
        layers: list[nn.Module] = []
        width_in = int(config["input_dim"])
        for _ in range(int(config["hidden_layers"])):
            layers.extend((nn.Linear(width_in, int(config["width"])), nn.SiLU()))
            width_in = int(config["width"])
        layers.append(nn.Linear(width_in, int(config["output_dim"])))
        self.network = nn.Sequential(*layers)
        self.network.load_state_dict(checkpoint["model"]["state_dict"], strict=True)
        self.network.to(network_dtype)

        def buffer(values, dtype=torch.float64):
            return torch.as_tensor(values, dtype=dtype)

        self.register_buffer("label_mean", buffer(checkpoint["labels"]["mean"]))
        self.register_buffer("label_std", buffer(checkpoint["labels"]["std"]))

        pca = checkpoint["pca"]
        self.register_buffer("coefficient_mean", buffer(pca["coefficient_mean"]))
        self.register_buffer("coefficient_std", buffer(pca["coefficient_std"]))
        self.register_buffer("basis", buffer(pca["basis"]))
        self.register_buffer("coordinate_mean", buffer(pca["coordinate_mean"]))
        self.register_buffer("coordinate_std", buffer(pca["coordinate_std"]))

        coordinates = checkpoint["coordinates"]
        self.register_buffer(
            "standard_rosseland_optical_depth",
            buffer(coordinates["standard_rosseland_optical_depth"]),
        )
        self.acceleration_scale = float(coordinates["acceleration_scale"])
        self.coordinate_fields: tuple[str, ...] = tuple(coordinates["fields"])
        self.layer_count = int(self.coordinate_mean.numel() // len(self.coordinate_fields))

        self.to(device)

    @classmethod
    def from_checkpoint(
        cls,
        path: Path | str | None = None,
        *,
        family: str = "five_label",
        device: torch.device | str = "cpu",
        **kwargs,
    ) -> "DifferentiableInitializer":
        checkpoint_path = Path(path) if path is not None else default_checkpoint_path(family)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return cls(checkpoint, device=device, **kwargs)

    # -- label handling -----------------------------------------------------

    def _label(self, labels: dict[str, torch.Tensor], name: str) -> torch.Tensor:
        """Fetch one label as a float64 column.

        Widening happens *before* any arithmetic. A caller who passes
        ``torch.tensor([5777.0])`` gets float32 by default, and computing
        ``5040 / Teff`` at that width costs ~6e-8 in the feature — which the
        deck's four-significant-digit quantization then turns into a visible
        last-digit difference from the released initializer.
        """

        return torch.as_tensor(
            labels[name], dtype=torch.float64, device=self.basis.device
        ).reshape(-1)

    def features(self, labels: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build the checkpoint's feature vector from public labels.

        The first checkpoint coordinate is ``5040 K / Teff``, not the
        temperature itself (``warm_start.py:719``).
        """

        columns = [5040.0 / self._label(labels, "effective_temperature")]
        columns.extend(
            self._label(labels, _PUBLIC_NAME[field]) for field in self.family_fields[1:]
        )
        return torch.stack(columns, dim=-1)

    # -- forward ------------------------------------------------------------

    def coordinates(self, features: torch.Tensor) -> torch.Tensor:
        """Map standardized features to ``(batch, layers, 6)`` raw coordinates."""

        standardized = (features - self.label_mean) / self.label_std
        # The released initializer evaluates this network in float32
        # (warm_start.py:653). Setting network_dtype=torch.float32 reproduces it
        # bit-for-bit one star at a time; the default float64 is used for
        # training because float32 matrix products are not reduction-order
        # stable, so a star's prediction would otherwise depend on which batch
        # it happened to share a step with.
        raw = self.network(standardized.to(self.network_dtype)).to(torch.float64)
        coefficients = raw * self.coefficient_std + self.coefficient_mean
        standardized_coordinates = coefficients @ self.basis
        flattened = standardized_coordinates * self.coordinate_std + self.coordinate_mean
        return flattened.reshape(-1, self.layer_count, len(self.coordinate_fields))

    def decode(
        self,
        coordinates: torch.Tensor,
        effective_temperature: torch.Tensor,
    ) -> InitializerPrediction:
        """Decode raw coordinates into physical columns (``warm_start.py:672-693``)."""

        temperature = torch.as_tensor(
            effective_temperature, dtype=torch.float64, device=coordinates.device
        ).reshape(-1, 1)
        if temperature.shape[0] not in (1, coordinates.shape[0]):
            raise ValueError(
                "effective_temperature must be scalar or match the coordinate batch"
            )

        increments = torch.pow(
            10.0, coordinates[..., 0].clamp(-_COLUMN_MASS_CLAMP, _COLUMN_MASS_CLAMP)
        )
        column_mass = torch.cumsum(increments, dim=-1)
        column_mass = self._enforce_increasing(column_mass, increments)

        grey_temperature = temperature * (
            0.75 * (self.standard_rosseland_optical_depth + 2.0 / 3.0)
        ) ** 0.25
        layer_temperature = grey_temperature * torch.pow(
            10.0, coordinates[..., 1].clamp(-_TEMPERATURE_CLAMP, _TEMPERATURE_CLAMP)
        )

        powered = torch.pow(
            10.0, coordinates[..., 2:5].clamp(-_POWER_CLAMP, _POWER_CLAMP)
        )
        acceleration = self.acceleration_scale * torch.sinh(
            coordinates[..., 5].clamp(-_ACCELERATION_CLAMP, _ACCELERATION_CLAMP)
        )

        return InitializerPrediction(
            column_mass=column_mass,
            temperature=layer_temperature,
            gas_pressure=powered[..., 0],
            electron_density=powered[..., 1],
            rosseland_opacity=powered[..., 2],
            radiative_acceleration=acceleration,
        )

    def _enforce_increasing(
        self, column_mass: torch.Tensor, increments: torch.Tensor
    ) -> torch.Tensor:
        """Guard the one way the cumulative sum can stop increasing.

        ``10**c`` is positive, so the sum rises by construction in exact
        arithmetic. In floating point an increment far smaller than the running
        total is absorbed, leaving two equal layers — which is what the
        reference rejects outright (``warm_start.py:686``). Rebuilding the sum
        from increments floored at a relative fraction of the running total
        keeps it strictly increasing and keeps a usable gradient, instead of
        raising.
        """

        if self.minimum_relative_increment <= 0.0:
            return column_mass
        floor = self.minimum_relative_increment * column_mass
        safe = torch.maximum(increments, floor)
        return torch.cumsum(safe, dim=-1)

    def forward(self, labels: dict[str, torch.Tensor]) -> InitializerPrediction:
        features = self.features(labels)
        return self.decode(self.coordinates(features), labels["effective_temperature"])


# Checkpoint field name -> public keyword used by the solver interfaces.
_PUBLIC_NAME = {
    "temperature_ratio_5040_k_over_temperature": "effective_temperature",
    "log10_surface_gravity_cgs": "log_surface_gravity",
    "metallicity": "metallicity",
    "alpha_enhancement": "alpha_enhancement",
    "microturbulence_km_s": "microturbulence_km_s",
    "carbon_enhancement": "carbon_enhancement",
    "nitrogen_enhancement": "nitrogen_enhancement",
    "oxygen_enhancement": "oxygen_enhancement",
}
