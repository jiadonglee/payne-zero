"""Small label-local adapters for solver-aligned reduced-state fine-tuning."""

from __future__ import annotations

import torch
from torch import nn


class LabelRbfAdapter(nn.Module):
    """Layer offsets localized around standardized label-space centers."""

    def __init__(
        self, centers: torch.Tensor, *, layer_count: int = 80, sigma: float = 0.35
    ) -> None:
        super().__init__()
        centers = torch.as_tensor(centers, dtype=torch.float64)
        self.register_buffer("centers", centers)
        self.sigma = float(sigma)
        self.temperature_offsets = nn.Parameter(
            torch.zeros(centers.shape[0], layer_count, dtype=torch.float64)
        )
        self.mass_offsets = nn.Parameter(
            torch.zeros(centers.shape[0], layer_count, dtype=torch.float64)
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.as_tensor(features, dtype=self.centers.dtype)
        distance_squared = torch.sum(
            (features[:, None, :] - self.centers[None, :, :]) ** 2, dim=-1
        )
        weights = torch.exp(-0.5 * distance_squared / self.sigma**2)
        return weights @ self.temperature_offsets, weights @ self.mass_offsets
