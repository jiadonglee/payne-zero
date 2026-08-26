"""Deck quantization of an initializer prediction, made differentiable.

Every warm start reaches the solver through the fixed-width deck formatter and
the deck parser (``warm_start.py:494`` and ``emulator_warm_start_model``). That
round trip is not an implementation detail: the comment at ``warm_start.py:506``
is explicit that the finite digits quantize the prediction and that the
certified baselines converged through exactly that quantization.

So the solver never sees the initializer's real output. It sees:

| column | format | resolution |
| --- | --- | --- |
| column mass | ``%.8E`` | 9 significant digits |
| temperature | ``%.1f`` | 0.1 K absolute |
| gas pressure | ``%.3E`` | 4 significant digits |
| electron density | ``%.3E`` | 4 significant digits |
| Rosseland opacity | ``%.3E`` | 4 significant digits |
| radiative acceleration | ``%.3E`` | 4 significant digits |

Four significant digits is a relative resolution of about 1e-4, which is only a
factor of five below the 5e-4 convergence threshold the solver tests. A
differentiable initializer that ignores this is optimizing a quantity the solver
cannot resolve.

Rounding has zero gradient almost everywhere, so ``quantize_prediction`` uses a
straight-through estimator: the forward value is exactly the quantized one and
the backward pass is the identity. That is the standard choice and it is the
right one here, because the quantization is a fixed property of the interface
rather than something the network should learn to exploit.
"""

from __future__ import annotations

import torch

from .initializer import InitializerPrediction


# Digits after the point in the ``%.<n>E`` deck formats (warm_start.py:575-579).
COLUMN_MASS_EXPONENT_DIGITS = 8
STANDARD_EXPONENT_DIGITS = 3
TEMPERATURE_DECIMALS = 1


def _round_significant(values: torch.Tensor, exponent_digits: int) -> torch.Tensor:
    """Round to ``exponent_digits + 1`` significant digits, as ``%.<n>E`` does.

    Zero and non-finite entries pass through: ``log10`` is undefined there and
    the formatter emits them exactly.
    """

    magnitude = values.abs()
    usable = torch.isfinite(values) & (magnitude > 0.0)
    safe = torch.where(usable, magnitude, torch.ones_like(magnitude))
    decade = torch.floor(torch.log10(safe))
    scale = torch.pow(10.0, decade - exponent_digits)
    rounded = torch.round(values / scale) * scale
    return torch.where(usable, rounded, values)


def _round_decimals(values: torch.Tensor, decimals: int) -> torch.Tensor:
    """Round to a fixed number of decimal places, as ``%.<n>f`` does."""

    scale = 10.0**decimals
    return torch.round(values * scale) / scale


def _straight_through(values: torch.Tensor, quantized: torch.Tensor) -> torch.Tensor:
    """Forward the quantized value, backward the identity."""

    return values + (quantized - values).detach()


def quantize_columns(
    *,
    column_mass: torch.Tensor,
    temperature: torch.Tensor,
    gas_pressure: torch.Tensor,
    electron_density: torch.Tensor,
    rosseland_opacity: torch.Tensor,
    radiative_acceleration: torch.Tensor,
    straight_through: bool = True,
) -> dict[str, torch.Tensor]:
    """Apply the deck's per-column resolution to a set of atmosphere columns."""

    quantized = {
        "column_mass": _round_significant(column_mass, COLUMN_MASS_EXPONENT_DIGITS),
        "temperature": _round_decimals(temperature, TEMPERATURE_DECIMALS),
        "gas_pressure": _round_significant(gas_pressure, STANDARD_EXPONENT_DIGITS),
        "electron_density": _round_significant(electron_density, STANDARD_EXPONENT_DIGITS),
        "rosseland_opacity": _round_significant(rosseland_opacity, STANDARD_EXPONENT_DIGITS),
        "radiative_acceleration": _round_significant(
            radiative_acceleration, STANDARD_EXPONENT_DIGITS
        ),
    }
    if not straight_through:
        return quantized
    originals = {
        "column_mass": column_mass,
        "temperature": temperature,
        "gas_pressure": gas_pressure,
        "electron_density": electron_density,
        "rosseland_opacity": rosseland_opacity,
        "radiative_acceleration": radiative_acceleration,
    }
    return {
        name: _straight_through(originals[name], value)
        for name, value in quantized.items()
    }


def quantize_prediction(
    prediction: InitializerPrediction, *, straight_through: bool = True
) -> InitializerPrediction:
    """Return the prediction as the solver will actually receive it."""

    return InitializerPrediction(
        **quantize_columns(**prediction.as_dict(), straight_through=straight_through)
    )
