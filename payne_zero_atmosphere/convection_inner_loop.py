"""COOLTLUSTY-style convective inner loop between global radiation iterations.

The production solver evaluates convection once per global iteration, then
applies the Avrett-style temperature correction.  This experimental loop sits
between those stages:

    global radiative transfer
        → convective inner loop (T-gradient, EOS/MLT recompute, local mismatch)
        → global temperature correction / next radiation field

It does not replace mixing length, opacity, or the EOS.  Two hard limits:

* The required convective flux ``σ T_eff^4 - F_rad`` is a target only.
  The stored convective flux is whatever the physics callback recomputes.
* The convective-zone mask may be frozen while the inner loop runs, but it
  must be released afterwards and rebuilt from the Schwarzschild criterion.

Default configuration keeps this loop off, so the production path is
unchanged.  Hubeny warns that near final convergence the inner correction
can cancel the global iteration and oscillate; that is a recorded diagnostic,
not a reason to freeze the zone permanently.

Two gradient stencils meet in this loop.  A pass writes ``∇`` as the
one-sided difference ``ln(T_i/T_{i-1}) / ln(P_i/P_{i-1})``; the MLT physics
reads ``∇`` back with a centred derivative.  Where ``∇_ad`` changes quickly
with depth (the H₂ dissociation zone of M dwarfs) the read-back value
exceeds the written one by about half the per-layer change of ``∇_ad``.
Prescribing the read-back target directly as the written gradient then
settles on a state with ``F_conv > F_required``.  With
``correct_written_gradient`` the pass instead shifts the written gradient by
the change the read-back gradient needs, so the loop settles where the
recomputed MLT flux meets the target.

The centred read-back is insensitive to a layer-alternating temperature
perturbation, so the corrected update carries such a component from pass to
pass, and where ``∇_ad`` follows the local temperature (the H₂ zone) the
target feeds it.  With ``filter_written_gradient`` the alternating part of
the written gradient inside the working mask is removed before the update.

A subadiabatic layer inside a convective run can still be unable to carry
its flux by radiation: its radiative gradient ``∇ H_target / H_rad`` exceeds
``∇_ad``, which is the Schwarzschild criterion stated with the gradient
radiation would need.  With ``fill_interior_holes`` such layers join the
frozen working mask, so the loop adjusts their gradient as well; the
convective flux is still whatever MLT gives for the resulting structure, and
the released mask is still rebuilt from ``∇ - ∇_ad > 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np


ZERO_TOP_LAYER_COUNT = 3
DEFAULT_INNER_PASSES = 8
DEFAULT_TEMPERATURE_RELATIVE_CAP = 0.15
DEFAULT_SUPERADIABATIC_CAP = 2.0
MLT_FLUX_EXPONENT = 1.5


@dataclass(frozen=True)
class ConvectivePhysicsSnapshot:
    """Quantities the inner loop reads.  ``convective_flux`` must be physical."""

    convective_flux: np.ndarray
    logarithmic_gradient: np.ndarray
    adiabatic_gradient: np.ndarray
    total_pressure: np.ndarray
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)


PhysicsCallback = Callable[[np.ndarray], ConvectivePhysicsSnapshot]


@dataclass(frozen=True)
class ConvectiveInnerLoopConfig:
    passes: int = DEFAULT_INNER_PASSES
    freeze_mask_during_passes: bool = True
    release_mask_after: bool = True
    temperature_relative_cap: float = DEFAULT_TEMPERATURE_RELATIVE_CAP
    zero_top_layer_count: int = ZERO_TOP_LAYER_COUNT
    superadiabatic_cap: float = DEFAULT_SUPERADIABATIC_CAP
    # Shift the written one-sided gradient by the change the read-back
    # gradient needs, instead of writing the read-back target directly.
    correct_written_gradient: bool = False
    # With ``correct_written_gradient``, remove the layer-alternating part of
    # the written gradient inside the working mask before the update.
    filter_written_gradient: bool = False
    # Add subadiabatic layers inside a convective run whose radiative gradient
    # exceeds the adiabatic one to the working mask.
    fill_interior_holes: bool = False


@dataclass(frozen=True)
class ConvectiveInnerLoopResult:
    temperature: np.ndarray
    convective_flux: np.ndarray
    required_convective_flux: np.ndarray
    local_energy_mismatch: np.ndarray
    convective_mask_initial: np.ndarray
    convective_mask_final: np.ndarray
    mask_was_frozen: bool
    mask_was_released: bool
    assigned_required_flux: bool
    physics_evaluations: int
    pass_records: tuple[dict[str, float], ...]
    extras: Mapping[str, np.ndarray] = field(default_factory=dict)
    filled_layers: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))


def required_convective_eddington_flux(
    *,
    target_eddington_flux: float,
    radiative_eddington_flux: np.ndarray,
) -> np.ndarray:
    """Target only: ``H_target - H_rad``.  Never write this into ``F_conv``."""

    return np.maximum(
        float(target_eddington_flux)
        - np.asarray(radiative_eddington_flux, dtype=np.float64),
        0.0,
    )


def convective_mask_from_gradients(
    logarithmic_gradient: np.ndarray,
    adiabatic_gradient: np.ndarray,
    *,
    zero_top_layer_count: int = ZERO_TOP_LAYER_COUNT,
) -> np.ndarray:
    """Schwarzschild mask; the top layers stay radiative as in MLT."""

    superadiabatic = (
        np.asarray(logarithmic_gradient, dtype=np.float64)
        - np.asarray(adiabatic_gradient, dtype=np.float64)
    )
    mask = superadiabatic > 0.0
    top = max(int(zero_top_layer_count), 0)
    if top > 0:
        mask = mask.copy()
        mask[: min(top, mask.size)] = False
    return mask


def interior_layers_radiation_cannot_carry(
    *,
    mask: np.ndarray,
    logarithmic_gradient: np.ndarray,
    adiabatic_gradient: np.ndarray,
    radiative_eddington_flux: np.ndarray,
    target_eddington_flux: float,
) -> np.ndarray:
    """Unmasked layers between masked layers where ``∇ H/H_rad > ∇_ad``."""

    active = np.asarray(mask, dtype=bool)
    masked = np.flatnonzero(active)
    filled = np.zeros_like(active)
    if masked.size < 2:
        return filled
    radiative = np.asarray(radiative_eddington_flux, dtype=np.float64)
    gradient = np.asarray(logarithmic_gradient, dtype=np.float64)
    adiabatic = np.asarray(adiabatic_gradient, dtype=np.float64)
    for layer in range(int(masked[0]) + 1, int(masked[-1])):
        if active[layer]:
            continue
        if radiative[layer] <= 0.0:
            filled[layer] = True
            continue
        radiative_gradient = gradient[layer] * float(target_eddington_flux) / radiative[layer]
        filled[layer] = bool(radiative_gradient > adiabatic[layer])
    return filled


def _neighbour_superadiabatic_seed(
    superadiabatic: np.ndarray,
    schwarzschild_mask: np.ndarray,
    layer: int,
) -> float:
    """Mean read-back excess of the nearest Schwarzschild layers above and below."""

    masked = np.flatnonzero(np.asarray(schwarzschild_mask, dtype=bool))
    above = masked[masked < layer]
    below = masked[masked > layer]
    values = []
    if above.size:
        values.append(float(superadiabatic[above[-1]]))
    if below.size:
        values.append(float(superadiabatic[below[0]]))
    positive = [value for value in values if value > 0.0]
    return float(np.mean(positive)) if positive else 0.0


def local_energy_mismatch(
    *,
    radiative_eddington_flux: np.ndarray,
    convective_flux: np.ndarray,
    target_eddington_flux: float,
) -> np.ndarray:
    """``H_rad + H_conv - H_target`` from the recomputed convective flux."""

    return (
        np.asarray(radiative_eddington_flux, dtype=np.float64)
        + np.asarray(convective_flux, dtype=np.float64)
        - float(target_eddington_flux)
    )


def superadiabatic_for_required_flux(
    current_superadiabatic: float,
    current_flux: float,
    required_flux: float,
    *,
    exponent: float = MLT_FLUX_EXPONENT,
    floor: float = 1.0e-6,
    cap: float = DEFAULT_SUPERADIABATIC_CAP,
) -> float:
    """Invert a monotonic MLT-like ``F ∝ δ^α`` for a trial superadiabatic.

    The result is a trial gradient, not a flux.  Callers must recompute
    ``F_conv`` from the original MLT after updating the structure.
    """

    required = max(float(required_flux), 0.0)
    flux = max(float(current_flux), 0.0)
    delta = max(float(current_superadiabatic), 0.0)
    if required <= 0.0:
        return 0.0
    if flux <= 0.0 or delta <= 0.0:
        return float(np.clip(max(delta, floor), 0.0, cap))
    scaled = delta * (required / flux) ** (1.0 / float(exponent))
    return float(np.clip(scaled, 0.0, cap))


def apply_logarithmic_gradient_to_temperature(
    *,
    temperature: np.ndarray,
    total_pressure: np.ndarray,
    nabla: np.ndarray,
    mask: np.ndarray,
    relative_cap: float = DEFAULT_TEMPERATURE_RELATIVE_CAP,
) -> np.ndarray:
    """Integrate ``∇ = d ln T / d ln P`` inward through the masked layers.

    Pressure is held fixed during the inner loop.  Unmasked layers keep their
    original temperature.  A newly convective slab anchors on the layer above.
    """

    original = np.asarray(temperature, dtype=np.float64)
    proposed = original.copy()
    pressure = np.asarray(total_pressure, dtype=np.float64)
    gradient = np.asarray(nabla, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    cap = float(relative_cap)
    for index in range(1, original.size):
        if not active[index]:
            proposed[index] = original[index]
            continue
        dln_pressure = np.log(
            np.maximum(pressure[index], 1.0e-300)
            / np.maximum(pressure[index - 1], 1.0e-300)
        )
        reference = proposed[index - 1] if active[index - 1] else original[index - 1]
        trial = float(reference) * np.exp(float(gradient[index]) * float(dln_pressure))
        lo = float(original[index]) * (1.0 - cap)
        hi = float(original[index]) * (1.0 + cap)
        proposed[index] = min(max(trial, lo), hi)
    proposed[0] = original[0]
    return proposed


def written_logarithmic_gradient(
    temperature: np.ndarray,
    total_pressure: np.ndarray,
) -> np.ndarray:
    """``∇`` on the stencil ``apply_logarithmic_gradient_to_temperature`` writes.

    Entry ``i`` is ``ln(T_i/T_{i-1}) / ln(P_i/P_{i-1})``; entry 0 has no layer
    above it and repeats entry 1.
    """

    temperature = np.asarray(temperature, dtype=np.float64)
    pressure = np.asarray(total_pressure, dtype=np.float64)
    gradient = np.zeros_like(temperature)
    if temperature.size < 2:
        return gradient
    gradient[1:] = np.log(
        np.maximum(temperature[1:], 1.0e-300) / np.maximum(temperature[:-1], 1.0e-300)
    ) / np.log(
        np.maximum(pressure[1:], 1.0e-300) / np.maximum(pressure[:-1], 1.0e-300)
    )
    gradient[0] = gradient[1]
    return gradient


def remove_alternating_component(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove the ``(-1)^i`` component of ``values`` inside each run of ``mask``.

    Where the stencil fits inside the run the 5-point filter
    ``(-1, 4, 10, 4, -1) / 16`` is used; at the two layers next to each end of
    a run, a least-squares fit of quadratic plus ``(-1)^j`` terms over the
    five nearest layers of the run gives the component to subtract.  Both
    leave quadratic profiles unchanged and use no value outside the run.
    Runs shorter than five layers and layers outside ``mask`` are unchanged.
    """

    values = np.asarray(values, dtype=np.float64)
    out = values.copy()
    index = np.flatnonzero(np.asarray(mask, dtype=bool))
    if index.size == 0:
        return out
    weights = np.array([-1.0, 4.0, 10.0, 4.0, -1.0]) / 16.0
    for run in np.split(index, np.flatnonzero(np.diff(index) > 1) + 1):
        if run.size < 5:
            continue
        for position, layer in enumerate(run):
            if 2 <= position <= run.size - 3:
                out[layer] = float(np.dot(weights, values[layer - 2:layer + 3]))
                continue
            window = run[:5] if position < 2 else run[-5:]
            offset = (window - layer).astype(np.float64)
            design = np.stack(
                [np.ones(5), offset, offset * offset, (-1.0) ** window], axis=1
            )
            coefficients, *_ = np.linalg.lstsq(design, values[window], rcond=None)
            out[layer] = float(values[layer] - coefficients[3] * (-1.0) ** layer)
    return out


def _pass_record(
    *,
    pass_index: int,
    mismatch: np.ndarray,
    mask: np.ndarray,
    temperature: np.ndarray,
    previous_temperature: np.ndarray,
    read_gradient: np.ndarray,
    total_pressure: np.ndarray,
) -> dict[str, float]:
    active = np.asarray(mask, dtype=bool)
    delta_t = np.asarray(temperature, dtype=np.float64) - np.asarray(
        previous_temperature, dtype=np.float64
    )
    read_minus_written = np.asarray(read_gradient, dtype=np.float64) - (
        written_logarithmic_gradient(temperature, total_pressure)
    )
    abs_mismatch = np.abs(np.asarray(mismatch, dtype=np.float64))
    if np.any(active):
        active_mismatch = abs_mismatch[active]
        active_dt = np.abs(delta_t[active])
    else:
        active_mismatch = abs_mismatch
        active_dt = np.abs(delta_t)
    return {
        "pass_index": float(pass_index),
        "n_masked_layers": float(np.count_nonzero(active)),
        "max_abs_mismatch": float(np.max(active_mismatch) if active_mismatch.size else 0.0),
        "p95_abs_mismatch": float(
            np.percentile(active_mismatch, 95.0) if active_mismatch.size else 0.0
        ),
        "max_abs_relative_temperature_change": float(
            np.max(active_dt / np.maximum(np.asarray(previous_temperature)[active], 1.0))
            if np.any(active)
            else 0.0
        ),
        "mean_temperature_change": float(np.mean(delta_t[active]) if np.any(active) else 0.0),
        "max_abs_read_minus_written_gradient": float(
            np.max(np.abs(read_minus_written[active])) if np.any(active) else 0.0
        ),
    }


def run_convection_zone_inner_loop(
    *,
    temperature: np.ndarray,
    radiative_eddington_flux: np.ndarray,
    target_eddington_flux: float,
    physics: PhysicsCallback,
    config: ConvectiveInnerLoopConfig | None = None,
) -> ConvectiveInnerLoopResult:
    """Adjust the convective-zone temperature gradient; recompute MLT flux.

    ``physics(temperature)`` must run the original EOS/MLT path and return a
    new ``convective_flux``.  This function never assigns that flux to the
    required residual.
    """

    settings = ConvectiveInnerLoopConfig() if config is None else config
    if int(settings.passes) < 0:
        raise ValueError("passes must be non-negative")
    temperature = np.asarray(temperature, dtype=np.float64).copy()
    radiative = np.asarray(radiative_eddington_flux, dtype=np.float64)
    required = required_convective_eddington_flux(
        target_eddington_flux=target_eddington_flux,
        radiative_eddington_flux=radiative,
    )
    snapshot = physics(temperature)
    physics_evaluations = 1
    if snapshot.convective_flux.shape != temperature.shape:
        raise ValueError("physics convective_flux must match the temperature grid")
    mask_initial = convective_mask_from_gradients(
        snapshot.logarithmic_gradient,
        snapshot.adiabatic_gradient,
        zero_top_layer_count=settings.zero_top_layer_count,
    )
    def _filled(schwarzschild: np.ndarray, state: ConvectivePhysicsSnapshot) -> np.ndarray:
        if not settings.fill_interior_holes:
            return np.zeros_like(schwarzschild)
        return interior_layers_radiation_cannot_carry(
            mask=schwarzschild,
            logarithmic_gradient=state.logarithmic_gradient,
            adiabatic_gradient=state.adiabatic_gradient,
            radiative_eddington_flux=radiative,
            target_eddington_flux=target_eddington_flux,
        )

    schwarzschild_mask = mask_initial.copy()
    filled_layers = _filled(schwarzschild_mask, snapshot)
    working_mask = mask_initial | filled_layers
    pass_records: list[dict[str, float]] = []
    current = snapshot
    for pass_index in range(int(settings.passes)):
        previous_temperature = temperature.copy()
        if not settings.freeze_mask_during_passes:
            schwarzschild_mask = convective_mask_from_gradients(
                current.logarithmic_gradient,
                current.adiabatic_gradient,
                zero_top_layer_count=settings.zero_top_layer_count,
            )
            filled_layers = _filled(schwarzschild_mask, current)
            working_mask = schwarzschild_mask | filled_layers
        read_nabla = np.asarray(current.logarithmic_gradient, dtype=np.float64)
        nabla = read_nabla.copy()
        superadiabatic = nabla - np.asarray(current.adiabatic_gradient, dtype=np.float64)
        if settings.correct_written_gradient:
            written_nabla = written_logarithmic_gradient(
                temperature, current.total_pressure
            )
            if settings.filter_written_gradient:
                written_nabla = remove_alternating_component(written_nabla, working_mask)
            # The centred read at the lower edge of a masked slab spans the
            # held layer below, so moving the edge layer does not change it;
            # the edge keeps the direct prescription.
            correctable = working_mask & np.append(working_mask[1:], True)
        for layer in np.flatnonzero(working_mask):
            if filled_layers[layer] and float(current.convective_flux[layer]) <= 0.0:
                # A filled layer without convective flux starts from the
                # read-back excess of its convective neighbours.
                trial_delta = min(
                    _neighbour_superadiabatic_seed(superadiabatic, schwarzschild_mask, layer),
                    float(settings.superadiabatic_cap),
                )
            else:
                trial_delta = superadiabatic_for_required_flux(
                    float(superadiabatic[layer]),
                    float(current.convective_flux[layer]),
                    float(required[layer]),
                    cap=settings.superadiabatic_cap,
                )
            target_nabla = float(current.adiabatic_gradient[layer]) + trial_delta
            if settings.correct_written_gradient and correctable[layer]:
                nabla[layer] = float(written_nabla[layer]) + (
                    target_nabla - float(read_nabla[layer])
                )
            else:
                nabla[layer] = target_nabla
        temperature = apply_logarithmic_gradient_to_temperature(
            temperature=temperature,
            total_pressure=current.total_pressure,
            nabla=nabla,
            mask=working_mask,
            relative_cap=settings.temperature_relative_cap,
        )
        current = physics(temperature)
        physics_evaluations += 1
        mismatch = local_energy_mismatch(
            radiative_eddington_flux=radiative,
            convective_flux=current.convective_flux,
            target_eddington_flux=target_eddington_flux,
        )
        pass_records.append(
            _pass_record(
                pass_index=pass_index,
                mismatch=mismatch,
                mask=working_mask,
                temperature=temperature,
                previous_temperature=previous_temperature,
                read_gradient=current.logarithmic_gradient,
                total_pressure=current.total_pressure,
            )
        )

    mismatch = local_energy_mismatch(
        radiative_eddington_flux=radiative,
        convective_flux=current.convective_flux,
        target_eddington_flux=target_eddington_flux,
    )
    mask_final = convective_mask_from_gradients(
        current.logarithmic_gradient,
        current.adiabatic_gradient,
        zero_top_layer_count=settings.zero_top_layer_count,
    )
    if not settings.release_mask_after:
        mask_final = working_mask
    return ConvectiveInnerLoopResult(
        temperature=temperature,
        convective_flux=np.asarray(current.convective_flux, dtype=np.float64).copy(),
        required_convective_flux=required,
        local_energy_mismatch=mismatch,
        convective_mask_initial=mask_initial,
        convective_mask_final=mask_final,
        mask_was_frozen=bool(settings.freeze_mask_during_passes),
        mask_was_released=bool(settings.release_mask_after),
        assigned_required_flux=False,
        physics_evaluations=int(physics_evaluations),
        pass_records=tuple(pass_records),
        extras=dict(current.extras),
        filled_layers=np.asarray(filled_layers, dtype=bool).copy(),
    )
