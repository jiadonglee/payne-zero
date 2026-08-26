"""Batch decoder for the shipped five-label initializer's ``(m,T)`` teacher.

The teacher is used only to make a training target for a standalone reduced
state model.  It is not imported by the solver or by reduced-state inference.
The decoder follows ``AtmosphereInitializer.predict`` and keeps the production
checkpoint's float32 neural forward pass and float64 physical decode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from payne_zero_atmosphere.warm_start import (
    DEFAULT_FIVE_LABEL_WEIGHTS_PATH,
    INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH,
    load_atmosphere_initializer,
)


def predict_production_mT_batch(
    labels: np.ndarray,
    *,
    batch_size: int = 2048,
    checkpoint_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the production checkpoint's column mass and temperature.

    ``labels`` has columns ``Teff, logg, [M/H], alpha, microturbulence``. The
    output is the latent decoder result before the fixed-width deck round trip;
    that distinction is negligible for a training target and avoids creating a
    Python deck object for every row in the 52k-star corpus.
    """

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 5:
        raise ValueError("labels must have shape (n_stars, 5)")
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] <= 0.0):
        raise ValueError("labels must be finite and have positive Teff")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    import torch

    initializer = load_atmosphere_initializer(
        checkpoint_path=checkpoint_path or DEFAULT_FIVE_LABEL_WEIGHTS_PATH,
        device="cpu",
    )
    checkpoint = initializer.checkpoint
    label_mean = np.asarray(checkpoint["labels"]["mean"], dtype=np.float64)
    label_std = np.asarray(checkpoint["labels"]["std"], dtype=np.float64)
    if label_mean.shape != (5,) or label_std.shape != (5,):
        raise ValueError("the production teacher is not a five-label checkpoint")

    features = np.column_stack(
        [5040.0 / values[:, 0], values[:, 1], values[:, 2], values[:, 3], values[:, 4]]
    )
    standardized_features = (features - label_mean) / label_std
    pca = checkpoint["pca"]
    coefficient_mean = np.asarray(pca["coefficient_mean"], dtype=np.float64)
    coefficient_std = np.asarray(pca["coefficient_std"], dtype=np.float64)
    basis = np.asarray(pca["basis"], dtype=np.float64)
    coordinate_mean = np.asarray(pca["coordinate_mean"], dtype=np.float64)
    coordinate_std = np.asarray(pca["coordinate_std"], dtype=np.float64)
    acceleration_scale = float(checkpoint["coordinates"]["acceleration_scale"])
    tau = np.asarray(INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH, dtype=np.float64)
    grey_factor = (0.75 * (tau + 2.0 / 3.0)) ** 0.25

    masses: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            stop = min(len(values), start + int(batch_size))
            model_input = torch.as_tensor(
                standardized_features[start:stop], dtype=torch.float32
            )
            standardized_coefficients = (
                initializer.model(model_input).detach().cpu().numpy()
            )
            coefficients = (
                standardized_coefficients * coefficient_std + coefficient_mean
            )
            standardized_coordinates = coefficients @ basis
            flattened = (
                standardized_coordinates * coordinate_std + coordinate_mean
            )
            coordinates = flattened.reshape(
                stop - start, 80, len(checkpoint["coordinates"]["fields"])
            )
            increments = 10.0 ** np.clip(coordinates[:, :, 0], -30.0, 30.0)
            mass = np.cumsum(increments, axis=1)
            grey_temperature = values[start:stop, 0, None] * grey_factor[None, :]
            temperature = grey_temperature * 10.0 ** np.clip(
                coordinates[:, :, 1], -3.0, 3.0
            )
            mass = np.asarray(mass, dtype=np.float64)
            temperature = np.asarray(temperature, dtype=np.float64)
            if np.any(~np.isfinite(mass)) or np.any(~np.isfinite(temperature)):
                raise ValueError("production teacher decoded non-finite profiles")
            masses.append(mass)
            temperatures.append(temperature)

    column_mass = np.concatenate(masses, axis=0)
    temperature = np.concatenate(temperatures, axis=0)
    if np.any(np.diff(column_mass, axis=1) <= 0.0):
        raise ValueError("production teacher decoded a non-monotonic profile")
    return column_mass, temperature
