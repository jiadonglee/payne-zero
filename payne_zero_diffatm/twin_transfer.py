"""Batched, differentiable port of the monochromatic transfer stage.

Torch restatement of the per-frequency kernel ``_transfer_moments_compiled``
(``payne_zero_atmosphere/transfer_kernels.py:495-791``) and the mode-2
accumulations inlined in ``accumulate_transfer_range_compiled``
(``transfer_kernels.py:793-995``), which ``runner.accumulate_transfer_state``
(``payne_zero_atmosphere/runner.py:1057-1215``) drives over the frequency grid.
The reference loops over frequencies in Numba; here the per-frequency work is
batched over ``star * frequency`` rows with the sequential depth/grid scans
going through :mod:`.grid_math`.

Fidelity notes, which matter because the reference is the thing being matched:

* The fixed 51-point operators (``mean_intensity_operator``,
  ``eddington_flux_operator``, ``second_moment_weights``) are cast to float32
  on load, and the whole source iteration on the fixed grid runs in float32
  (``transfer_kernels.py:558-636``); everything off the grid is float64.
  ``load_transfer_tables(operator_dtype=torch.float64)`` reruns the grid block
  in float64 for comparison; the default matches the reference.
* The reference iterates the grid source correction
  (``transfer_kernels.py:597-636``) and the deep-layer diffusion correction
  (``transfer_kernels.py:698-758``) until a relative-error criterion
  (``1e-5``) is met, capped at 51 sweeps. This port runs the same capped loop
  but freezes each row with a ``torch.where`` mask once *its* criterion is
  met, so a converged row keeps exactly the state the reference would have
  stopped with; rows that never converge get the same 51 sweeps. Remaining
  differences are float32 reduction order inside the 51-element dot products,
  at the few-times-1e-7 level.
* The depth slices in the diffusion branch start at layer
  ``max(mapped-2, 0)`` / ``mapped-1`` (``transfer_kernels.py:694-696``), which
  varies per row. The tangent derivative is evaluated on the full layer grid
  and the slice-start layer is overwritten with the forward secant; interior
  tangent evaluations only read layers ``(i-1, i, i+1)``, so every other layer
  of the sliced derivative is identical to the full-grid one
  (``radiative_transfer.py:148-191``).
* ``mapped_layer_count`` is data-dependent in the reference
  (``transfer_kernels.py:559-561`` and the ``remap_to_grid`` return,
  ``radiative_transfer.py:372``: ``min(count(tau <= grid_top), L-1)``, with 1
  when even the surface layer lies beyond the grid). It is computed per row
  with ``searchsorted`` and the branch structure is applied with masks; the
  ``mapped == layer_count`` branch (``transfer_kernels.py:654-687``) is dead
  for an 80-layer atmosphere because the remap return is clamped to ``L-1``,
  but the masks still honour it.
* The exponential integral of order 3 (``transfer_kernels.py:422-493``) is
  vectorized with ``torch.where`` over its three rational-approximation
  branches, including the ``Teff <= 4250 K, 0.005 < dtau < 0.02`` zeroing
  quirk (``transfer_kernels.py:941-946``).
* The accumulators returned here are the *raw* mode-2 sums, before the mode-3
  conversions in ``radiative_pressure.py:72-101`` and
  ``rosseland_mean.py:60-78``. ``temperature_correction_integrated_eddington_flux``
  (``transfer_kernels.py:915-917``) deposits the same ``H*w`` as
  ``integrated_eddington_flux`` (``transfer_kernels.py:889-891``), so it is
  exposed as a property rather than a second tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .grid_math import (
    differentiate_on_depth_grid,
    integrate_on_depth_grid,
    remap_to_grid,
)

# Planck prefactor from runner.py:1153-1158 (non-exact-constant form).
_PLANCK_PREFACTOR = 1.47439e-2
# The reference floors (transfer_kernels.py:533, 570-571, 612-633, 683-686).
_OPACITY_FLOOR = 1.0e-300
_SOURCE_FLOOR = 1.0e-38
_GRID_SOURCE_FLOOR = 1.0e-37
_NEGATIVE_FLUX_FLOOR = 1.0e-99
# Source-iteration convergence criterion (transfer_kernels.py:627, 757).
_CONVERGENCE_RELATIVE_ERROR = 1.0e-5
# Small-dtau series cutoff for the diagonal-Lambda term
# (transfer_kernels.py:928) and the cool-star zeroing quirk (:941-946).
_DEPTH_STEP_SERIES_CUTOFF = 0.01
_QUIRK_EFFECTIVE_TEMPERATURE = 4250.0
_QUIRK_DEPTH_STEP_RANGE = (0.005, 0.02)


def _divided_or_zero(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """The reference's ``a / b if b != 0 else 0`` idiom, autograd-safe.

    Same construction as ``grid_math._divided_or_zero``: the denominator is
    sanitized before dividing so the discarded branch carries no inf/NaN into
    autograd.
    """

    safe = torch.where(
        denominator != 0.0, denominator, torch.ones_like(denominator)
    )
    return torch.where(
        denominator != 0.0, numerator / safe, torch.zeros_like(numerator)
    )


def _signed_floor_fp(value: torch.Tensor, floor: float) -> torch.Tensor:
    """The reference's ``abs(x) < floor -> +-floor`` guard, keeping the sign.

    transfer_kernels.py:612-623 (float32 grid iteration) and
    temperature_correction.py:169-172 (the same idiom in float64).
    """

    small = value.abs() < floor
    sign_fix = torch.where(
        value >= 0.0, torch.full_like(value, floor), torch.full_like(value, -floor)
    )
    return torch.where(small, sign_fix, value)


@dataclass(frozen=True)
class TwinTransferTables:
    """Fixed optical-depth operators used by the transfer solve.

    Torch mirror of ``RadiativeTransferTables``
    (``radiative_transfer.py:385-393``). ``transfer_optical_depth_grid`` stays
    float64; the operators take ``operator_dtype`` (float32 by default, as in
    ``radiative_transfer.py:423-439``).
    """

    transfer_optical_depth_grid: torch.Tensor  # (grid,) float64
    mean_intensity_operator: torch.Tensor  # (grid, grid) operator_dtype
    eddington_flux_operator: torch.Tensor  # (grid, grid) operator_dtype
    second_moment_weights: torch.Tensor  # (grid,) operator_dtype

    @property
    def grid_count(self) -> int:
        return int(self.transfer_optical_depth_grid.shape[0])

    @property
    def operator_dtype(self) -> torch.dtype:
        return self.mean_intensity_operator.dtype


def load_transfer_tables(
    path: str | Path | None = None,
    *,
    operator_dtype: torch.dtype = torch.float32,
) -> TwinTransferTables:
    """Load the packaged transfer operators with the reference's precisions.

    Mirrors ``load_radiative_transfer_tables``
    (``radiative_transfer.py:396-440``): the depth grid is float64 and the
    51-point operators are float32 by default. Pass
    ``operator_dtype=torch.float64`` to run the grid block in float64 for
    comparison; only the grid iteration and the operator products change.
    """

    table_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parent.parent
        / "atmosphere_tables"
        / "radiative_transfer_tables.npz"
    )
    with np.load(table_path, allow_pickle=False) as arrays:
        grid = torch.as_tensor(
            np.asarray(arrays["transfer_optical_depth_grid"], dtype=np.float64)
        )
        mean_intensity = torch.as_tensor(
            np.asarray(arrays["mean_intensity_operator"], dtype=np.float64)
        ).to(operator_dtype)
        eddington_flux = torch.as_tensor(
            np.asarray(arrays["eddington_flux_operator"], dtype=np.float64)
        ).to(operator_dtype)
        second_moment = torch.as_tensor(
            np.asarray(arrays["second_moment_weights"], dtype=np.float64)
        ).to(operator_dtype)
    return TwinTransferTables(
        transfer_optical_depth_grid=grid,
        mean_intensity_operator=mean_intensity,
        eddington_flux_operator=eddington_flux,
        second_moment_weights=second_moment,
    )


def planck_and_stimulated_emission(
    frequency_hz: torch.Tensor,
    h_over_kt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized B_nu and stimulated emission, runner.py:1139-1160.

    ``frequency_hz`` is ``(freq,)`` and ``h_over_kt`` is ``(star, layer)``;
    both outputs are ``(star, layer, freq)`` float64. The formula is the
    non-exact-constant form ``B = 1.47439e-2 * (nu/1e15)^3 * e^-x / max(1 -
    e^-x, 1e-300)`` with ``x = nu * h_over_kt``.
    """

    frequency = torch.as_tensor(frequency_hz, dtype=torch.float64)
    hkt = torch.as_tensor(h_over_kt, dtype=torch.float64)
    argument = frequency.reshape((1,) * hkt.ndim + (-1,)) * hkt.unsqueeze(-1)
    exponential = torch.exp(-argument)
    stimulated = (1.0 - exponential).clamp(min=1.0e-300)
    frequency_15 = frequency / 1.0e15
    planck = (
        _PLANCK_PREFACTOR
        * frequency_15.reshape((1,) * hkt.ndim + (-1,)) ** 3
        * exponential
        / stimulated
    )
    return planck, stimulated


def exponential_integral(order: int, argument: torch.Tensor) -> torch.Tensor:
    """Vectorized exponential-integral approximation, transfer_kernels.py:422-493.

    Same rational approximations as ``exponential_integral_approximation``
    (``temperature_correction.py:89-166``): three branches in ``x`` for the
    first order, then the recurrence ``E_{n+1} = (e^-x - x*E_n)/n``. ``x <= 0``
    returns 0, as in the reference.
    """

    a0, a1, a2, a3, a4, a5 = (
        -44178.5471728217, 57721.7247139444, 9938.31388962037,
        1842.11088668, 101.093806161906, 5.03416184097568,
    )
    b0, b1, b2, b3, b4 = (
        76537.3323337614, 32597.1881290275, 6106.10794245759,
        635.419418378382, 37.2298352833327,
    )
    c0, c1, c2, c3, c4, c5, c6 = (
        4.65627107975096e-7, 0.999979577051595, 9.04161556946329,
        24.3784088791317, 23.0192559391333, 6.90522522784444, 0.430967839469389,
    )
    d1, d2, d3, d4, d5, d6 = (
        10.0411643829054, 32.4264210695138, 41.2807841891424,
        20.4494785013794, 3.31909213593302, 0.103400130404874,
    )
    e0, e1, e2, e3, e4, e5, e6 = (
        -0.999999999998447, -26.6271060431811, -241.055827097015,
        -895.927957772937, -1298.85688746484, -545.374158883133,
        -5.66575206533869,
    )
    f1, f2, f3, f4, f5, f6 = (
        28.6271060422192, 292.310039388533, 1332.78537748257,
        2777.61949509163, 2404.01713225909, 631.6574832808,
    )
    x = torch.as_tensor(argument, dtype=torch.float64)
    positive_x = x.clamp(min=1.0e-300)
    exponential = torch.exp(-x.clamp(min=0.0))
    large = (
        exponential
        + exponential
        * (e0 + (e1 + (e2 + (e3 + (e4 + (e5 + e6 / positive_x) / positive_x)
                    / positive_x) / positive_x) / positive_x) / positive_x)
        / (positive_x + f1 + (f2 + (f3 + (f4 + (f5 + f6 / positive_x) / positive_x)
                              / positive_x) / positive_x) / positive_x)
    ) / positive_x
    middle = (
        exponential
        * (c6 + (c5 + (c4 + (c3 + (c2 + (c1 + c0 * positive_x) * positive_x)
                       * positive_x) * positive_x) * positive_x) * positive_x)
        / (d6 + (d5 + (d4 + (d3 + (d2 + (d1 + positive_x) * positive_x)
                       * positive_x) * positive_x) * positive_x) * positive_x)
    )
    small = (
        (a0 + (a1 + (a2 + (a3 + (a4 + a5 * positive_x) * positive_x)
               * positive_x) * positive_x) * positive_x)
        / (b0 + (b1 + (b2 + (b3 + (b4 + positive_x) * positive_x)
                 * positive_x) * positive_x) * positive_x)
        - torch.log(positive_x)
    )
    first_order = torch.where(
        x > 4.0, large, torch.where(x > 1.0, middle, small)
    )
    first_order = torch.where(x <= 0.0, torch.zeros_like(x), first_order)
    value = first_order
    for index in range(1, max(int(order), 1)):
        value = (exponential - x * value) / float(index)
    return torch.where(x <= 0.0, torch.zeros_like(x), value)


@dataclass
class TwinTransferResult:
    """Frequency-integrated accumulators plus per-frequency transfer moments.

    The accumulators are the raw mode-2 sums of
    ``accumulate_transfer_range_compiled`` (``transfer_kernels.py:886-995``),
    per ``(star, layer)``:

    * ``rosseland_accumulator``: ``sum_nu (dB/dT)/max(kappa,1e-300) * w``
      with ``dB/dT = B * nu * h/(kT) / max(T*stimulated, 1e-300)``
      (``transfer_kernels.py:979-995``).
    * ``integrated_eddington_flux``: ``sum_nu H_nu * w`` (:888-891).
    * ``radiation_energy_density``: ``sum_nu J_nu * w`` (:887-888).
    * ``radiative_acceleration``: ``sum_nu kappa * H_nu * w`` (:892-896).
    * ``surface_second_moment``: ``sum_nu K_nu * w``, the surface second
      moment of the grid source (``transfer_kernels.py:684-686, 788-790``),
      i.e. the reference's ``surface_radiation_pressure_constant`` before the
      mode-3 conversion (:897-899).
    * ``mean_intensity_minus_source_integral``:
      ``sum_nu kappa * (J - S) * w`` (:912-914).
    * ``absorption_heating_derivative``:
      ``sum_nu (dkappa/dm)/max(kappa,1e-300) * H_nu * w`` with the tangent
      derivative on column mass (:902-911).
    * ``diagonal_lambda_accumulator``: ``sum_nu kappa * (D-1) /
      max(1 - sigma*D, 1e-300) * (1-sigma) * dB/dT * w`` where ``D`` is the
      sequential diagonal-Lambda estimate built from the order-3 exponential
      integral of the optical-depth steps (:919-977).

    The per-frequency moments are ``(star, layer, freq)``: ``mean_intensity``
    (J), ``eddington_flux`` (H, the Eddington flux as defined by the kernel —
    the reference's ``H = (1/3) dS/dtau`` convention),
    ``source_function`` (S), ``mean_intensity_minus_source`` (J - S),
    ``optical_depth`` and ``total_opacity``. ``mapped_layer_count`` is
    ``(star, freq)``: how many atmosphere layers fall inside the fixed
    51-point grid for that frequency.
    """

    rosseland_accumulator: torch.Tensor
    integrated_eddington_flux: torch.Tensor
    radiation_energy_density: torch.Tensor
    radiative_acceleration: torch.Tensor
    surface_second_moment: torch.Tensor
    mean_intensity_minus_source_integral: torch.Tensor
    absorption_heating_derivative: torch.Tensor
    diagonal_lambda_accumulator: torch.Tensor
    mean_intensity: torch.Tensor
    eddington_flux: torch.Tensor
    source_function: torch.Tensor
    mean_intensity_minus_source: torch.Tensor
    optical_depth: torch.Tensor
    total_opacity: torch.Tensor
    mapped_layer_count: torch.Tensor

    @property
    def temperature_correction_integrated_eddington_flux(self) -> torch.Tensor:
        """The temperature-correction copy of ``sum_nu H_nu * w``.

        Deposited separately in the reference (``transfer_kernels.py:915-917``)
        but element-for-element the same sum as
        ``integrated_eddington_flux`` (:889-891).
        """

        return self.integrated_eddington_flux


def _forward_secant(
    grid: torch.Tensor, values: torch.Tensor, index: torch.Tensor
) -> torch.Tensor:
    """Forward secant at a per-row layer index, the slice-start derivative.

    The sliced ``_differentiate_on_depth_grid_compiled`` takes the secant at
    its first layer (``radiative_transfer.py:160-163``); the full-grid
    evaluation this patches would use the interior tangent there instead.
    """

    count = values.shape[-1]
    index0 = index.clamp(0, count - 1).unsqueeze(-1)
    index1 = (index + 1).clamp(0, count - 1).unsqueeze(-1)
    grid0 = torch.gather(grid, -1, index0)
    grid1 = torch.gather(grid, -1, index1)
    value0 = torch.gather(values, -1, index0)
    value1 = torch.gather(values, -1, index1)
    return _divided_or_zero(value1 - value0, grid1 - grid0)


def _sliced_tangent_derivative(
    grid: torch.Tensor, values: torch.Tensor, start_index: torch.Tensor
) -> torch.Tensor:
    """Tangent derivative of the per-row slice ``[start_index, layers)``.

    Every layer past ``start_index`` agrees with the full-grid tangent
    derivative (it only reads its two neighbours); the slice start takes the
    forward secant. Layers before ``start_index`` carry the full-grid values
    and are never read by the caller.
    """

    full = differentiate_on_depth_grid(grid, values)
    secant = _forward_secant(grid, values, start_index)
    layer_index = torch.arange(
        values.shape[-1], device=values.device
    ).unsqueeze(0)
    is_start = layer_index == start_index.unsqueeze(-1)
    return torch.where(is_start, secant, full)


def _iterate_grid_source(
    thermal_grid: torch.Tensor,
    scattering_grid: torch.Tensor,
    mean_intensity_operator: torch.Tensor,
    active: torch.Tensor,
    sweeps: int,
) -> torch.Tensor:
    """Lambda-operator source iteration on the fixed grid, batched over rows.

    Ports ``transfer_kernels.py:586-636``. The sweep walks the grid deep to
    surface with in-place (Gauss-Seidel) updates; each row freezes once its
    sweep-relative-error maximum falls to ``1e-5`` or below, matching the
    reference's per-frequency early exit. Everything runs in the operator
    dtype (float32 by default).
    """

    grid_count = thermal_grid.shape[-1]
    source = thermal_grid.clone()
    diagonal = 1.0 - scattering_grid * mean_intensity_operator.diagonal()
    thermal_term = (1.0 - scattering_grid) * thermal_grid
    eye = torch.eye(grid_count, dtype=torch.bool, device=source.device)
    for _ in range(sweeps):
        if not bool(active.any()):
            break
        max_relative_error = torch.zeros_like(active, dtype=source.dtype)
        for grid_index in range(grid_count - 1, -1, -1):
            mean_source = (source * mean_intensity_operator[grid_index]).sum(-1)
            numerator = (
                mean_source * scattering_grid[:, grid_index]
                + thermal_term[:, grid_index]
                - source[:, grid_index]
            )
            denominator = _signed_floor_fp(
                diagonal[:, grid_index], _GRID_SOURCE_FLOOR
            )
            correction = numerator / denominator
            relative_error = (
                correction / _signed_floor_fp(source[:, grid_index], _GRID_SOURCE_FLOOR)
            ).abs()
            max_relative_error = torch.maximum(max_relative_error, relative_error)
            updated = (source[:, grid_index] + correction).clamp(
                min=_GRID_SOURCE_FLOOR
            )
            column_mask = active.unsqueeze(-1) & eye[grid_index].unsqueeze(0)
            source = torch.where(column_mask, updated.unsqueeze(-1), source)
        active = active & (max_relative_error > _CONVERGENCE_RELATIVE_ERROR)
    return source


def _iterate_deep_source(
    optical_depth: torch.Tensor,
    thermal_source: torch.Tensor,
    planck: torch.Tensor,
    scattering_fraction: torch.Tensor,
    source: torch.Tensor,
    mapped_layer_count: torch.Tensor,
    active: torch.Tensor,
    sweeps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deep-layer diffusion iteration, batched over rows.

    Ports ``transfer_kernels.py:689-758``. Layers beyond the mapped grid range
    take ``H = (1/3) dS/dtau`` and ``J - S = dH/dtau`` with the tangent
    derivative; the source relaxes toward ``(1-sigma)*S_thermal +
    sigma*J``. Non-positive sources or fluxes reset the slice to the Planck
    function, exactly as the reference does, and each row freezes once its
    accumulated relative source change drops below ``1e-5``.

    ``source`` enters with the grid-remapped head already in place; it is
    returned with the deep layers updated (and any Planck resets applied, on
    head layers too, as in the reference's single shared array). Also
    returned: the final J, H and (J - S) over the deep slice.
    """

    layer_count = optical_depth.shape[-1]
    layer_index = torch.arange(layer_count, device=optical_depth.device).unsqueeze(0)
    derivative_start = (mapped_layer_count - 2).clamp(min=0)
    mean_start = mapped_layer_count - 1
    first_deep = torch.where(
        mapped_layer_count == 1,
        torch.zeros_like(mapped_layer_count),
        mapped_layer_count,
    )
    ge_derivative = layer_index >= derivative_start.unsqueeze(-1)
    ge_first_deep = layer_index >= first_deep.unsqueeze(-1)

    thermal = thermal_source.clone()
    mean_intensity = torch.zeros_like(source)
    eddington_flux = torch.zeros_like(source)
    mean_intensity_minus_source = torch.zeros_like(source)
    for _ in range(sweeps):
        if not bool(active.any()):
            break
        active_rows = active.unsqueeze(-1)
        invalid = ((source <= 0.0) & ge_derivative & active_rows).any(-1)
        reset = invalid.unsqueeze(-1) & ge_derivative & active_rows
        thermal = torch.where(reset, planck, thermal)
        source = torch.where(reset, planck, source)
        derivative = _sliced_tangent_derivative(optical_depth, source, derivative_start)
        flux = derivative / 3.0
        negative_flux = ((flux <= 0.0) & ge_derivative & active_rows).any(-1)
        reset = negative_flux.unsqueeze(-1) & ge_derivative & active_rows
        thermal = torch.where(reset, planck, thermal)
        source = torch.where(reset, planck, source)
        invalid = invalid | negative_flux
        if bool(negative_flux.any()):
            derivative = _sliced_tangent_derivative(
                optical_depth, source, derivative_start
            )
            flux = derivative / 3.0

        mean_derivative = _sliced_tangent_derivative(optical_depth, flux, mean_start)
        mms = torch.where(
            ge_first_deep & invalid.unsqueeze(-1),
            torch.zeros_like(mean_derivative),
            mean_derivative,
        )
        updated_mean_intensity = mms + source
        updated_source = (
            (1.0 - scattering_fraction) * thermal
            + scattering_fraction * updated_mean_intensity
        )
        accumulated_error = (
            (
                (updated_source - source).abs()
                / updated_source.abs().clamp(min=_OPACITY_FLOOR)
            )
            * (ge_first_deep & active_rows)
        ).sum(-1)
        apply = ge_first_deep & active_rows
        source = torch.where(apply, updated_source, source)
        mean_intensity = torch.where(active_rows, updated_mean_intensity, mean_intensity)
        eddington_flux = torch.where(active_rows, flux, eddington_flux)
        mean_intensity_minus_source = torch.where(active_rows, mms, mean_intensity_minus_source)
        active = active & (accumulated_error >= _CONVERGENCE_RELATIVE_ERROR)
    return source, mean_intensity, eddington_flux, mean_intensity_minus_source


def _transfer_moments_rows(
    continuum_absorption: torch.Tensor,
    continuum_source: torch.Tensor,
    line_opacity_stimulated: torch.Tensor,
    continuum_scattering: torch.Tensor,
    column_mass: torch.Tensor,
    planck: torch.Tensor,
    tables: TwinTransferTables,
    sweeps: int,
) -> dict[str, torch.Tensor]:
    """The per-frequency kernel, batched over (star * frequency) rows.

    Ports ``_transfer_moments_compiled`` (``transfer_kernels.py:495-791``) with
    the reference's ``line_source = planck`` calling convention (:852-871).
    All inputs are ``(row, layer)`` float64. Returns per-row ``(row, layer)``
    moments plus the ``(row,)`` surface second moment and mapped layer count.
    """

    layer_count = column_mass.shape[-1]
    grid = tables.transfer_optical_depth_grid
    grid_count = tables.grid_count
    layer_index = torch.arange(layer_count, device=column_mass.device).unsqueeze(0)

    total_opacity = (
        continuum_absorption + line_opacity_stimulated + continuum_scattering
    ).clamp(min=_OPACITY_FLOOR)
    scattering_fraction = continuum_scattering / total_opacity
    thermal_absorption = continuum_absorption + line_opacity_stimulated
    thermal_source = torch.where(
        thermal_absorption > 0.0,
        _divided_or_zero(
            continuum_absorption * continuum_source
            + line_opacity_stimulated * planck,
            thermal_absorption,
        ),
        planck,
    )
    optical_depth = integrate_on_depth_grid(
        column_mass,
        total_opacity,
        surface_value=total_opacity[..., 0] * column_mass[..., 0],
    )

    # mapped_layer_count (radiative_transfer.py:372 and
    # transfer_kernels.py:561): min(count(tau <= grid_top), L-1), or 1 when
    # even the surface optical depth lies beyond the grid.
    grid_top = grid[-1]
    inside = torch.searchsorted(
        optical_depth, grid_top.expand(optical_depth.shape[0], 1).contiguous()
    ).squeeze(-1)
    mapped_layer_count = inside.clamp(max=layer_count - 1)
    mapped_layer_count = torch.where(
        optical_depth[:, 0] > grid_top,
        torch.ones_like(mapped_layer_count),
        mapped_layer_count,
    )

    # --- Fixed-grid block (transfer_kernels.py:558-652), run in the operator
    # dtype. Rows with mapped == 1 skip it in the reference; their results are
    # masked out at the combination below.
    operator_dtype = tables.operator_dtype
    thermal_grid = (
        remap_to_grid(optical_depth, thermal_source, grid)
        .to(operator_dtype)
        .clamp(min=_SOURCE_FLOOR)
    )
    scattering_grid = (
        remap_to_grid(optical_depth, scattering_fraction, grid)
        .to(operator_dtype)
        .clamp(min=0.0)
    )
    below_surface = grid.unsqueeze(0) < optical_depth[:, :1]
    thermal_grid = torch.where(
        below_surface,
        thermal_source[:, :1].clamp(min=_SOURCE_FLOOR).to(operator_dtype),
        thermal_grid,
    )
    scattering_grid = torch.where(
        below_surface,
        scattering_fraction[:, :1].clamp(min=0.0).to(operator_dtype),
        scattering_grid,
    )
    source_grid = _iterate_grid_source(
        thermal_grid,
        scattering_grid,
        tables.mean_intensity_operator,
        active=mapped_layer_count > 1,
        sweeps=sweeps,
    )
    source_grid_f64 = source_grid.to(torch.float64)
    source_head = remap_to_grid(grid, source_grid_f64, optical_depth)

    # Grid moments from the converged source (transfer_kernels.py:654-686 and
    # 764-790 are the same block for the full and partial cases).
    operator_column = source_grid.unsqueeze(-1)
    mean_source_vec = (
        tables.mean_intensity_operator @ operator_column
    ).squeeze(-1)
    flux_vec = (tables.eddington_flux_operator @ operator_column).squeeze(-1)
    grid_mms = (-source_grid + mean_source_vec).to(torch.float64)
    grid_flux = flux_vec.to(torch.float64)
    mms_head = remap_to_grid(grid, grid_mms, optical_depth)
    flux_head = remap_to_grid(grid, grid_flux, optical_depth)
    surface_moment_grid = (
        tables.second_moment_weights * source_grid
    ).sum(-1).to(torch.float64)

    # --- Deep-layer diffusion block (transfer_kernels.py:689-762). Runs for
    # every row; fully-mapped rows are masked out at the combination.
    source_initial = torch.where(
        (layer_index < mapped_layer_count.unsqueeze(-1))
        & (mapped_layer_count.unsqueeze(-1) > 1),
        source_head,
        thermal_source,
    )
    source, mean_intensity_deep, flux_deep, mms_deep = _iterate_deep_source(
        optical_depth,
        thermal_source,
        planck,
        scattering_fraction,
        source_initial,
        mapped_layer_count,
        active=mapped_layer_count < layer_count,
        sweeps=sweeps,
    )

    # --- Combine grid head and diffusion tail per row.
    use_grid_layer = (layer_index < mapped_layer_count.unsqueeze(-1)) & (
        mapped_layer_count.unsqueeze(-1) > 1
    )
    fully_mapped = (mapped_layer_count == layer_count).unsqueeze(-1)
    # transfer_kernels.py:681-683 (full) versus :784-787 (partial): the
    # partial branch floors the source before adding.
    mean_intensity_head = torch.where(
        fully_mapped,
        (mms_head + source).clamp(min=_SOURCE_FLOOR),
        (mms_head + source.clamp(min=_SOURCE_FLOOR)).clamp(min=_SOURCE_FLOOR),
    )
    mean_intensity = torch.where(
        use_grid_layer, mean_intensity_head, mean_intensity_deep
    )
    eddington_flux = torch.where(use_grid_layer, flux_head, flux_deep)
    mean_intensity_minus_source = torch.where(use_grid_layer, mms_head, mms_deep)
    surface_second_moment = torch.where(
        mapped_layer_count == 1,
        mean_intensity_deep[:, 0] / 3.0,
        surface_moment_grid,
    )
    return {
        "optical_depth": optical_depth,
        "total_opacity": total_opacity,
        "scattering_fraction": scattering_fraction,
        "source_function": source,
        "mean_intensity": mean_intensity,
        "eddington_flux": eddington_flux,
        "mean_intensity_minus_source": mean_intensity_minus_source,
        "surface_second_moment": surface_second_moment,
        "mapped_layer_count": mapped_layer_count,
    }


def transfer_moments(
    continuum_absorption: torch.Tensor,
    continuum_scattering: torch.Tensor,
    continuum_source_or_planck: torch.Tensor,
    line_opacity: torch.Tensor,
    column_mass: torch.Tensor,
    temperature: torch.Tensor,
    *,
    frequency_hz: torch.Tensor,
    frequency_weights: torch.Tensor,
    h_over_kt: torch.Tensor,
    effective_temperature: torch.Tensor | float,
    target_integrated_eddington_flux: torch.Tensor | float,
    tables: TwinTransferTables,
    planck_source: torch.Tensor | None = None,
    stimulated_emission: torch.Tensor | None = None,
    frequency_count: int | None = None,
    sweeps: int = 51,
) -> TwinTransferResult:
    """Frequency-integrated transfer accumulators, batched over stars.

    All layer-resolved inputs are ``(star, layer, freq)`` float64;
    ``column_mass``, ``temperature`` and ``h_over_kt`` are ``(star, layer)``;
    ``frequency_hz`` and ``frequency_weights`` are shared ``(freq,)``.
    ``line_opacity`` is the line mass absorption coefficient *before* the
    stimulated-emission factor (the kernel multiplies it in,
    ``transfer_kernels.py:843-850``); pass float32 values upcast to float64 to
    match the reference bit-for-bit (``runner.py:1180-1183``).

    ``planck_source`` / ``stimulated_emission`` may be supplied directly
    (``(star, layer, freq)``); otherwise they are computed from
    ``frequency_hz`` and ``h_over_kt`` via
    :func:`planck_and_stimulated_emission` (``runner.py:1139-1160``).

    ``sweeps`` caps both source iterations (default 51, the reference's cap);
    converged rows freeze early via masked updates regardless.
    """

    ca = torch.as_tensor(continuum_absorption, dtype=torch.float64)
    cs = torch.as_tensor(continuum_scattering, dtype=torch.float64)
    csource = torch.as_tensor(continuum_source_or_planck, dtype=torch.float64)
    line = torch.as_tensor(line_opacity, dtype=torch.float64)
    mass = torch.as_tensor(column_mass, dtype=torch.float64)
    temp = torch.as_tensor(temperature, dtype=torch.float64)
    hkt = torch.as_tensor(h_over_kt, dtype=torch.float64)
    frequency = torch.as_tensor(frequency_hz, dtype=torch.float64).reshape(-1)
    weights = torch.as_tensor(frequency_weights, dtype=torch.float64).reshape(-1)
    teff = torch.as_tensor(effective_temperature, dtype=torch.float64).reshape(-1)
    target = torch.as_tensor(
        target_integrated_eddington_flux, dtype=torch.float64
    ).reshape(-1)

    star_count, layer_count, freq_count = ca.shape
    if planck_source is None or stimulated_emission is None:
        planck_full, stimulated_full = planck_and_stimulated_emission(frequency, hkt)
        if planck_source is None:
            planck_source = planck_full
        if stimulated_emission is None:
            stimulated_emission = stimulated_full
    planck = torch.as_tensor(planck_source, dtype=torch.float64)
    stimulated = torch.as_tensor(stimulated_emission, dtype=torch.float64)
    total_frequency_count = freq_count if frequency_count is None else int(frequency_count)

    # (star, layer, freq) -> (row = star*freq, layer).
    def to_rows(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 1).reshape(star_count * freq_count, layer_count)

    def to_layers(tensor: torch.Tensor) -> torch.Tensor:
        return (
            tensor.unsqueeze(1)
            .expand(star_count, freq_count, layer_count)
            .reshape(star_count * freq_count, layer_count)
        )

    line_stimulated = to_rows(line * stimulated)
    moments = _transfer_moments_rows(
        to_rows(ca),
        to_rows(csource),
        line_stimulated,
        to_rows(cs),
        to_layers(mass),
        to_rows(planck),
        tables,
        sweeps,
    )

    row_count = star_count * freq_count
    weight_rows = (
        weights.unsqueeze(0).expand(star_count, freq_count).reshape(row_count, 1)
    )
    frequency_rows = (
        frequency.unsqueeze(0).expand(star_count, freq_count).reshape(row_count, 1)
    )
    hkt_rows = to_layers(hkt)
    temp_rows = to_layers(temp)
    teff_rows = (
        teff.unsqueeze(-1)
        .expand(star_count, freq_count)
        .reshape(row_count, 1)
    )
    planck_rows = to_rows(planck)
    stimulated_rows = to_rows(stimulated)

    optical_depth = moments["optical_depth"]
    total_opacity = moments["total_opacity"]
    scattering_fraction = moments["scattering_fraction"]
    source = moments["source_function"]
    mean_intensity = moments["mean_intensity"]
    eddington_flux = moments["eddington_flux"]
    mean_intensity_minus_source = moments["mean_intensity_minus_source"]

    # Negative-flux floor (transfer_kernels.py:873-884).
    any_negative_flux = (eddington_flux < 0.0).any(-1, keepdim=True)
    eddington_flux = torch.where(
        any_negative_flux, eddington_flux.clamp(min=_NEGATIVE_FLUX_FLOOR), eddington_flux
    )
    mean_intensity = torch.where(
        any_negative_flux, mean_intensity.clamp(min=_NEGATIVE_FLUX_FLOOR), mean_intensity
    )
    source = torch.where(
        any_negative_flux, source.clamp(min=_NEGATIVE_FLUX_FLOOR), source
    )

    def frequency_sum(term: torch.Tensor) -> torch.Tensor:
        return (term * weight_rows).reshape(star_count, freq_count, layer_count).sum(1)

    # RADIAP mode 2 (transfer_kernels.py:886-899).
    radiation_energy_density = frequency_sum(mean_intensity)
    integrated_eddington_flux = frequency_sum(eddington_flux)
    radiative_acceleration = frequency_sum(total_opacity * eddington_flux)
    surface_second_moment = (
        (moments["surface_second_moment"] * weight_rows.squeeze(-1))
        .reshape(star_count, freq_count)
        .sum(1)
    )

    # Temperature-correction mode 2 (transfer_kernels.py:901-977).
    opacity_derivative = differentiate_on_depth_grid(to_layers(mass), total_opacity)
    absorption_heating_derivative = frequency_sum(
        opacity_derivative
        / total_opacity.clamp(min=_OPACITY_FLOOR)
        * eddington_flux
    )
    mean_intensity_minus_source_integral = frequency_sum(
        total_opacity * mean_intensity_minus_source
    )

    # Diagonal-Lambda term (:919-977): the sequential depth recurrence only
    # couples adjacent layers, so it vectorizes as a shifted sum.
    depth_step = torch.cat(
        [
            optical_depth[:, 1:] - optical_depth[:, :-1],
            torch.full_like(optical_depth[:, :1], 1.0e-10),
        ],
        dim=-1,
    ).clamp(min=1.0e-10)
    series_term = (
        (0.922784335098467 - torch.log(depth_step)) * depth_step / 4.0
        + depth_step * depth_step / 12.0
        - depth_step**3.0 / 96.0
        + depth_step**4.0 / 720.0
    )
    e3 = torch.where(
        depth_step < 10.0,
        exponential_integral(3, depth_step),
        torch.zeros_like(depth_step),
    )
    quirk = (
        (teff_rows <= _QUIRK_EFFECTIVE_TEMPERATURE)
        & (depth_step > _QUIRK_DEPTH_STEP_RANGE[0])
        & (depth_step < _QUIRK_DEPTH_STEP_RANGE[1])
    )
    e3 = torch.where(quirk, torch.zeros_like(e3), e3)
    exponential_term = 0.5 * (depth_step + e3 - 0.5) / depth_step
    next_term = torch.where(
        depth_step <= _DEPTH_STEP_SERIES_CUTOFF, series_term, exponential_term
    )
    diagonal_mean_intensity = (
        torch.cat([torch.zeros_like(next_term[:, :1]), next_term[:, :-1]], dim=-1)
        + next_term
    )
    planck_derivative = (
        planck_rows
        * frequency_rows
        * hkt_rows
        / (temp_rows * stimulated_rows).clamp(min=_OPACITY_FLOOR)
    )
    if total_frequency_count == 1:
        planck_derivative = (
            target.unsqueeze(-1)
            .expand(star_count, freq_count)
            .reshape(row_count, 1)
            * 16.0
            / temp_rows.clamp(min=_OPACITY_FLOOR)
        )
    diagonal_lambda_accumulator = frequency_sum(
        total_opacity
        * (diagonal_mean_intensity - 1.0)
        / (1.0 - scattering_fraction * diagonal_mean_intensity).clamp(
            min=_OPACITY_FLOOR
        )
        * (1.0 - scattering_fraction)
        * planck_derivative
    )

    # Rosseland-mean mode 2 (transfer_kernels.py:979-995).
    source_derivative = (
        planck_rows
        * frequency_rows
        * hkt_rows
        / (temp_rows * stimulated_rows).clamp(min=_OPACITY_FLOOR)
    )
    if total_frequency_count == 1:
        source_derivative = (
            4.0 * (5.6697e-5 / 3.14159) * temp_rows**3
        )
    rosseland_accumulator = frequency_sum(
        source_derivative / total_opacity.clamp(min=_OPACITY_FLOOR)
    )

    def from_rows(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(star_count, freq_count, layer_count).permute(0, 2, 1)

    return TwinTransferResult(
        rosseland_accumulator=rosseland_accumulator,
        integrated_eddington_flux=integrated_eddington_flux,
        radiation_energy_density=radiation_energy_density,
        radiative_acceleration=radiative_acceleration,
        surface_second_moment=surface_second_moment,
        mean_intensity_minus_source_integral=mean_intensity_minus_source_integral,
        absorption_heating_derivative=absorption_heating_derivative,
        diagonal_lambda_accumulator=diagonal_lambda_accumulator,
        mean_intensity=from_rows(mean_intensity),
        eddington_flux=from_rows(eddington_flux),
        source_function=from_rows(source),
        mean_intensity_minus_source=from_rows(mean_intensity_minus_source),
        optical_depth=from_rows(optical_depth),
        total_opacity=from_rows(total_opacity),
        mapped_layer_count=moments["mapped_layer_count"].reshape(
            star_count, freq_count
        ),
    )
