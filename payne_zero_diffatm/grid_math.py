"""Batched, differentiable port of the depth-grid calculus.

Torch float64 restatement of the depth-grid helpers in
``payne_zero_atmosphere/radiative_transfer.py`` (lines 26-372): the parabolic
interval coefficients, integration, tangent-based differentiation, and the
piecewise-quadratic remap. The twin needs these batched over stars (and over
frequencies for the transfer solve) and differentiable in the values, where the
reference is a 1-D NumPy loop.

Fidelity notes, which matter because the reference is the thing being matched:

* Every guarded division in the reference (``a / b if b != 0 else 0``) becomes
  :func:`_divided_or_zero`, which sanitizes the denominator before dividing so
  the discarded branch carries no inf/NaN into autograd.
* The sequential coefficient-blending loop (``radiative_transfer.py:96-110``)
  only ever reads *unblended* neighbour values — the loop ascends while reading
  from ``index + 1`` — so it vectorizes exactly with no reordering error.
* The remap's forward/backward coefficient cache
  (``radiative_transfer.py:241-292``) is value-transparent: the cached forward
  quadratic and the recomputed backward quadratic run identical float formulas
  on identical points, so recomputing both per interval is bit-for-bit the
  same. What is *not* transparent is the carried ``source_index``: the
  reference assumes a non-decreasing target grid. This port locates each target
  independently with ``searchsorted``, which agrees with the reference on every
  non-decreasing target grid; unsorted targets are out of contract.
* The reference remap reads uninitialized memory for a one-point source grid
  (``grid_1based[0]`` is never set). That case is undefined there; here a
  one-point source remaps to a constant.

All functions take the depth grid along the last axis. The grid may be a
shared 1-D tensor or batched with the same leading shape as the values.
"""

from __future__ import annotations

import torch


# Fallback branch threshold from radiative_transfer.py:181-187: below this the
# tangent denominators are treated as degenerate and the mean slope is used.
_TANGENT_DENOMINATOR_FLOOR = 1.0e-30


def _as_float64(tensor: torch.Tensor | float) -> torch.Tensor:
    """Accept anything tensor-like and land on float64."""

    return torch.as_tensor(tensor, dtype=torch.float64)


def _divided_or_zero(
    numerator: torch.Tensor, denominator: torch.Tensor
) -> torch.Tensor:
    """The reference's ``a / b if b != 0 else 0`` idiom, autograd-safe.

    Dividing first and masking after would leave inf/NaN in the discarded
    branch, and ``0 * nan`` resurfaces as NaN in backward, so the denominator
    is sanitized before the division.
    """

    safe = torch.where(
        denominator != 0.0, denominator, torch.ones_like(denominator)
    )
    return torch.where(
        denominator != 0.0, numerator / safe, torch.zeros_like(numerator)
    )


def _broadcast_grid_values(
    grid: torch.Tensor, values: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast a shared 1-D grid against batched values, or vice versa."""

    if grid.ndim == 1 and values.ndim > 1:
        grid = grid.expand(values.shape)
    elif values.ndim == 1 and grid.ndim > 1:
        values = values.expand(grid.shape)
    return torch.broadcast_tensors(grid, values)


def parabolic_coefficients(
    values: torch.Tensor,
    grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parabolic interval coefficients, batched over the last axis.

    Ports ``radiative_transfer.py:26-115``. Returns ``(constant, linear,
    quadratic)``, each shaped like ``values``, where interval ``i`` is the
    polynomial in the grid coordinate anchored at layer ``i``.
    """

    value = _as_float64(values)
    coordinate = _as_float64(grid)
    coordinate, value = _broadcast_grid_values(coordinate, value)
    count = value.shape[-1]
    if count == 0:
        empty = value.clone()
        return empty, empty.clone(), empty.clone()
    if count == 1:
        return value.clone(), torch.zeros_like(value), torch.zeros_like(value)

    # End intervals are linear (radiative_transfer.py:44-53).
    linear_first = _divided_or_zero(
        value[..., 1] - value[..., 0], coordinate[..., 1] - coordinate[..., 0]
    )
    constant_first = value[..., 0] - coordinate[..., 0] * linear_first
    linear_last = _divided_or_zero(
        value[..., -1] - value[..., -2], coordinate[..., -1] - coordinate[..., -2]
    )
    constant_last = value[..., -1] - coordinate[..., -1] * linear_last
    if count == 2:
        constant = torch.stack([constant_first, constant_last], dim=-1)
        linear = torch.stack([linear_first, linear_last], dim=-1)
        return constant, linear, torch.zeros_like(constant)

    # Interior parabolas through layers (i-1, i, i+1)
    # (radiative_transfer.py:57-83).
    grid_left, grid_mid, grid_right = (
        coordinate[..., :-2],
        coordinate[..., 1:-1],
        coordinate[..., 2:],
    )
    value_left, value_mid, value_right = (
        value[..., :-2],
        value[..., 1:-1],
        value[..., 2:],
    )
    local_width = grid_mid - grid_left
    next_width = grid_right - grid_mid
    wide_width = grid_right - grid_left
    slope = _divided_or_zero(value_mid - value_left, local_width)
    first_term = _divided_or_zero(value_right, next_width * wide_width)
    left_term = _divided_or_zero(value_left, wide_width)
    right_term = _divided_or_zero(value_mid, next_width)
    second_term = _divided_or_zero(left_term - right_term, local_width)
    quadratic_inner = first_term + second_term
    linear_inner = slope - (grid_mid + grid_left) * quadratic_inner
    constant_inner = (
        value_left - grid_left * slope + grid_mid * grid_left * quadratic_inner
    )

    # The two innermost intervals after the first are forced linear
    # (radiative_transfer.py:85-94): the reference fits its parabolas from the
    # deep end and seeds the surface with slopes.
    def _linear_coefficients(deep: int) -> tuple[torch.Tensor, torch.Tensor]:
        linear = _divided_or_zero(
            value[..., deep + 1] - value[..., deep],
            coordinate[..., deep + 1] - coordinate[..., deep],
        )
        constant = value[..., deep] - coordinate[..., deep] * linear
        return constant, linear

    zero_inner = torch.zeros_like(quadratic_inner[..., :1])
    constant_1, linear_1 = _linear_coefficients(1)
    quadratic_inner = torch.cat([zero_inner, quadratic_inner[..., 1:]], dim=-1)
    linear_inner = torch.cat([linear_1.unsqueeze(-1), linear_inner[..., 1:]], dim=-1)
    constant_inner = torch.cat(
        [constant_1.unsqueeze(-1), constant_inner[..., 1:]], dim=-1
    )
    if count > 3:
        constant_2, linear_2 = _linear_coefficients(2)
        quadratic_inner = torch.cat(
            [quadratic_inner[..., :1], zero_inner, quadratic_inner[..., 2:]], dim=-1
        )
        linear_inner = torch.cat(
            [linear_inner[..., :1], linear_2.unsqueeze(-1), linear_inner[..., 2:]],
            dim=-1,
        )
        constant_inner = torch.cat(
            [constant_inner[..., :1], constant_2.unsqueeze(-1), constant_inner[..., 2:]],
            dim=-1,
        )

    constant = torch.cat(
        [constant_first.unsqueeze(-1), constant_inner, constant_last.unsqueeze(-1)],
        dim=-1,
    )
    linear = torch.cat(
        [linear_first.unsqueeze(-1), linear_inner, linear_last.unsqueeze(-1)], dim=-1
    )
    quadratic = torch.cat(
        [zero_inner, quadratic_inner, zero_inner], dim=-1
    )

    # Curvature-weighted blend toward the deeper neighbour
    # (radiative_transfer.py:96-110). The reference loop ascends while reading
    # index + 1, so every read sees the pre-blend value and the loop is exactly
    # parallel. Intervals whose quadratic is zero are skipped.
    quadratic_self = quadratic[..., 1:-1]
    quadratic_next = quadratic[..., 2:]
    weight = _divided_or_zero(
        quadratic_next.abs(), quadratic_next.abs() + quadratic_self.abs()
    )
    keep = quadratic_self == 0.0
    constant_blend = constant[..., 2:] + weight * (
        constant[..., 1:-1] - constant[..., 2:]
    )
    linear_blend = linear[..., 2:] + weight * (linear[..., 1:-1] - linear[..., 2:])
    quadratic_blend = quadratic[..., 2:] + weight * (
        quadratic[..., 1:-1] - quadratic[..., 2:]
    )
    constant = torch.cat(
        [constant[..., :1], torch.where(keep, constant[..., 1:-1], constant_blend),
         constant[..., -1:]],
        dim=-1,
    )
    linear = torch.cat(
        [linear[..., :1], torch.where(keep, linear[..., 1:-1], linear_blend),
         linear[..., -1:]],
        dim=-1,
    )
    quadratic = torch.cat(
        [quadratic[..., :1], torch.where(keep, quadratic[..., 1:-1], quadratic_blend),
         quadratic[..., -1:]],
        dim=-1,
    )

    # The last interior interval takes the deep-end linear coefficients
    # (radiative_transfer.py:112-114).
    constant = torch.cat([constant[..., :-2], constant[..., -1:], constant[..., -1:]], dim=-1)
    linear = torch.cat([linear[..., :-2], linear[..., -1:], linear[..., -1:]], dim=-1)
    quadratic = torch.cat([quadratic[..., :-2], quadratic[..., -1:], quadratic[..., -1:]], dim=-1)
    return constant, linear, quadratic


def integrate_on_depth_grid(
    grid: torch.Tensor,
    values: torch.Tensor,
    *,
    surface_value: torch.Tensor | float,
) -> torch.Tensor:
    """Cumulative parabolic integral from the surface inward.

    Ports ``radiative_transfer.py:118-145``. ``surface_value`` seeds the
    integral at the outermost layer; the callers' convention is
    ``values[..., 0] * grid[..., 0]`` (e.g. ``rosseland_mean.py:72-78``,
    ``radiative_pressure.py:94-98``), with ``0.0`` where the integrand vanishes
    at the surface. Accepts a scalar or anything broadcasting against the
    leading shape of ``values``.
    """

    coordinate = _as_float64(grid)
    value = _as_float64(values)
    coordinate, value = _broadcast_grid_values(coordinate, value)
    count = value.shape[-1]
    surface = _as_float64(surface_value)
    if surface.ndim < value.ndim:
        surface = surface.reshape(surface.shape + (1,) * (value.ndim - surface.ndim))
    if count == 0:
        return value.clone()
    head = surface + torch.zeros_like(value[..., :1])
    if count == 1:
        return head
    constant, linear, quadratic = parabolic_coefficients(value, coordinate)
    grid_left = coordinate[..., :-1]
    grid_right = coordinate[..., 1:]
    segment = constant[..., :-1] + 0.5 * linear[..., :-1] * (grid_right + grid_left)
    segment = segment + (quadratic[..., :-1] / 3.0) * (
        (grid_right + grid_left) * grid_right + grid_left * grid_left
    )
    segment = segment * (grid_right - grid_left)
    return torch.cat([head, head + torch.cumsum(segment, dim=-1)], dim=-1)


def differentiate_on_depth_grid(
    grid: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Tangent-based derivative on a monotonic depth grid.

    Ports ``radiative_transfer.py:148-191``: end intervals take the secant,
    interior layers blend the left and right secants through a tangent
    averaging that falls back to the plain mean slope when the tangent
    denominators degenerate.
    """

    coordinate = _as_float64(grid)
    value = _as_float64(values)
    coordinate, value = _broadcast_grid_values(coordinate, value)
    count = value.shape[-1]
    if count < 2:
        return torch.zeros_like(value)

    low_width = coordinate[..., 1] - coordinate[..., 0]
    high_width = coordinate[..., -1] - coordinate[..., -2]
    derivative_first = _divided_or_zero(value[..., 1] - value[..., 0], low_width)
    derivative_last = _divided_or_zero(value[..., -1] - value[..., -2], high_width)
    if count == 2:
        return torch.stack([derivative_first, derivative_last], dim=-1)

    # Sign of grid growth at the surface (radiative_transfer.py:164-166); the
    # reference divides by the signed width, so this is +/-1, else 1.0.
    direction = torch.where(
        low_width != 0.0,
        _divided_or_zero(low_width.abs(), low_width),
        torch.ones_like(low_width),
    )

    grid_left, grid_mid, grid_right = (
        coordinate[..., :-2],
        coordinate[..., 1:-1],
        coordinate[..., 2:],
    )
    value_left, value_mid, value_right = (
        value[..., :-2],
        value[..., 1:-1],
        value[..., 2:],
    )
    scale = torch.maximum(
        torch.maximum(value_left.abs(), value_mid.abs()), value_right.abs()
    )
    scale = torch.where(
        grid_mid != 0.0,
        scale
        / torch.where(grid_mid != 0.0, grid_mid.abs(), torch.ones_like(grid_mid)),
        scale,
    )
    scale = torch.where(scale == 0.0, torch.ones_like(scale), scale)

    right_width = grid_right - grid_mid
    left_width = grid_mid - grid_left
    widths_ok = (right_width != 0.0) & (left_width != 0.0)
    right_slope = _divided_or_zero(value_right - value_mid, right_width) / scale
    left_slope = _divided_or_zero(value_mid - value_left, left_width) / scale

    direction_inner = direction.unsqueeze(-1)
    right_denominator = (
        direction_inner * torch.sqrt(1.0 + right_slope * right_slope) + 1.0
    )
    left_denominator = (
        direction_inner * torch.sqrt(1.0 + left_slope * left_slope) + 1.0
    )
    denominators_ok = (right_denominator.abs() >= _TANGENT_DENOMINATOR_FLOOR) & (
        left_denominator.abs() >= _TANGENT_DENOMINATOR_FLOOR
    )
    fallback = 0.5 * (left_slope + right_slope) * scale

    right_denominator_safe = torch.where(
        denominators_ok, right_denominator, torch.ones_like(right_denominator)
    )
    left_denominator_safe = torch.where(
        denominators_ok, left_denominator, torch.ones_like(left_denominator)
    )
    right_tangent = right_slope / right_denominator_safe
    left_tangent = left_slope / left_denominator_safe
    tangent_denominator = 1.0 - right_tangent * left_tangent
    tangent_ok = tangent_denominator.abs() >= _TANGENT_DENOMINATOR_FLOOR
    tangent_denominator_safe = torch.where(
        tangent_ok, tangent_denominator, torch.ones_like(tangent_denominator)
    )
    derivative_inner = torch.where(
        tangent_ok,
        (right_tangent + left_tangent) / tangent_denominator_safe * scale,
        fallback,
    )
    derivative_inner = torch.where(denominators_ok, derivative_inner, fallback)
    derivative_inner = torch.where(
        widths_ok, derivative_inner, torch.zeros_like(derivative_inner)
    )
    return torch.cat(
        [
            derivative_first.unsqueeze(-1),
            derivative_inner,
            derivative_last.unsqueeze(-1),
        ],
        dim=-1,
    )


def _quadratic_through_three(
    grid_left: torch.Tensor,
    grid_mid: torch.Tensor,
    grid_right: torch.Tensor,
    value_left: torch.Tensor,
    value_mid: torch.Tensor,
    value_right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quadratic through three layers, the remap's forward/backward fit.

    The forward (``radiative_transfer.py:301-335``) and backward
    (``radiative_transfer.py:247-288``) branches are the same float formula on
    their respective triples, which is what makes the reference's coefficient
    cache value-transparent.
    """

    width = grid_mid - grid_left
    right_width = grid_right - grid_mid
    wide_width = grid_right - grid_left
    slope = _divided_or_zero(value_mid - value_left, width)
    first_term = _divided_or_zero(value_right, right_width * wide_width)
    left_term = _divided_or_zero(value_left, wide_width)
    right_term = _divided_or_zero(value_mid, right_width)
    second_term = _divided_or_zero(left_term - right_term, width)
    quadratic = first_term + second_term
    linear = slope - (grid_mid + grid_left) * quadratic
    constant = (
        value_left - grid_left * slope + grid_mid * grid_left * quadratic
    )
    return quadratic, linear, constant


def remap_to_grid(
    source_grid: torch.Tensor,
    source_values: torch.Tensor,
    target_grid: torch.Tensor,
) -> torch.Tensor:
    """Piecewise-quadratic remap onto a non-decreasing target grid.

    Ports ``radiative_transfer.py:194-372``. Per target point the interval is
    located independently (the reference carries its interval cursor forward,
    which agrees with an independent search exactly when the target grid is
    non-decreasing — every caller's target grid is). Interval convention: the
    first two source intervals are linear, the last is the backward parabola,
    the interior blends the forward and backward parabolas by curvature weight
    ``|q_fwd| / (|q_fwd| + |q_bwd|)``, and beyond the source ends the outermost
    linear interval extrapolates.

    Unlike the reference this returns only the remapped values; the reference's
    second return (the mapped layer count) has no batched meaning and no caller
    uses it.

    The target grid may be a shared 1-D tensor or batched with leading
    dimensions broadcasting against the source's.
    """

    old_grid = _as_float64(source_grid)
    old_values = _as_float64(source_values)
    new_grid = _as_float64(target_grid)
    old_grid, old_values = _broadcast_grid_values(old_grid, old_values)
    if new_grid.ndim == 1 and old_grid.ndim > 1:
        new_grid = new_grid.expand(old_grid.shape[:-1] + new_grid.shape)
    leading = torch.broadcast_shapes(old_grid.shape[:-1], new_grid.shape[:-1])
    old_grid = old_grid.expand(leading + old_grid.shape[-1:]).contiguous()
    old_values = old_values.expand(leading + old_values.shape[-1:])
    new_grid = new_grid.expand(leading + new_grid.shape[-1:]).contiguous()

    source_count = old_grid.shape[-1]
    if source_count == 0 or new_grid.shape[-1] == 0:
        return torch.zeros_like(new_grid)
    if source_count == 1:
        # Undefined in the reference (reads an uninitialized pad slot); a
        # constant is the only sane batched meaning.
        return old_values.broadcast_to(new_grid.shape).contiguous()

    # k = number of source points <= target: the target sits in
    # [grid[k-1], grid[k]), with k == 0 below the grid and k == n above it.
    interval = torch.searchsorted(old_grid, new_grid, right=True)

    def _points(index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        clamped = index.clamp(0, source_count - 1)
        return (
            torch.gather(old_grid, -1, clamped),
            torch.gather(old_values, -1, clamped),
        )

    # Linear on interval (k-1, k), clamped to the outermost intervals; used for
    # k <= 2, k >= n (radiative_transfer.py:225-239 and 351-366).
    linear_start = (interval - 1).clamp(0, source_count - 2)
    grid_a, value_a = _points(linear_start)
    grid_b, value_b = _points(linear_start + 1)
    linear_linear = _divided_or_zero(value_b - value_a, grid_b - grid_a)
    linear_constant = value_b - grid_b * linear_linear

    # Backward parabola through (k-2, k-1, k), forward through (k-1, k, k+1).
    backward_grid_left, backward_value_left = _points(interval - 2)
    backward_grid_mid, backward_value_mid = _points(interval - 1)
    backward_grid_right, backward_value_right = _points(interval)
    backward_quadratic, backward_linear, backward_constant = _quadratic_through_three(
        backward_grid_left,
        backward_grid_mid,
        backward_grid_right,
        backward_value_left,
        backward_value_mid,
        backward_value_right,
    )
    forward_grid_left, forward_value_left = _points(interval - 1)
    forward_grid_mid, forward_value_mid = _points(interval)
    forward_grid_right, forward_value_right = _points(interval + 1)
    forward_quadratic, forward_linear, forward_constant = _quadratic_through_three(
        forward_grid_left,
        forward_grid_mid,
        forward_grid_right,
        forward_value_left,
        forward_value_mid,
        forward_value_right,
    )

    # Curvature weight (radiative_transfer.py:336-347).
    weight = _divided_or_zero(
        forward_quadratic.abs(), forward_quadratic.abs() + backward_quadratic.abs()
    )
    blend_constant = forward_constant + weight * (backward_constant - forward_constant)
    blend_linear = forward_linear + weight * (backward_linear - forward_linear)
    blend_quadratic = forward_quadratic + weight * (backward_quadratic - forward_quadratic)

    use_linear = (interval <= 2) | (interval >= source_count)
    use_backward_only = interval == source_count - 1
    constant = torch.where(
        use_linear,
        linear_constant,
        torch.where(use_backward_only, backward_constant, blend_constant),
    )
    linear = torch.where(
        use_linear,
        linear_linear,
        torch.where(use_backward_only, backward_linear, blend_linear),
    )
    quadratic = torch.where(
        use_linear,
        torch.zeros_like(blend_quadratic),
        torch.where(use_backward_only, backward_quadratic, blend_quadratic),
    )
    return constant + (linear + quadratic * new_grid) * new_grid
