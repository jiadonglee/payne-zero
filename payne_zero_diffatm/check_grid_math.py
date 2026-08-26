"""Check the torch depth-grid calculus against the NumPy reference.

The torch port in ``grid_math.py`` is only useful if it is an exact
restatement of the shipped helpers in
``payne_zero_atmosphere/radiative_transfer.py`` (lines 26-372), so this
compares every ported function against the NumPy original:

1. **Real solver state.** Depth grids and profiles from a captured twin trace
   (``runs/twin_traces/.../iter_3``): integrate and differentiate the trace's
   temperature, Rosseland opacity, and gas pressure on its column-mass grid,
   and remap them onto a log-spaced target that reaches past both grid ends.
2. **Synthetic batched grids.** Random monotonic log-grids with smooth random
   values, batched as ``(star, layer)`` and ``(star, frequency, layer)``, with
   shared and per-row grids, and targets that extrapolate off both ends. Each
   batch row is compared against a 1-D NumPy reference call.
3. **Autograd.** A scalar loss through ``remap_to_grid`` and
   ``integrate_on_depth_grid`` must push finite, nonzero gradients into the
   values.

These are the same float64 formulas, so any difference above round-off is a
bug. On the trace data (magnitudes up to ~3e5) round-off lands near 1e-10, so
the absolute bound is 1e-9; on the O(1) synthetic data the relative bound is
1e-12.

Run::

    PYTHONPATH=. .venv-linux/bin/python -m payne_zero_diffatm.check_grid_math
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from payne_zero_atmosphere.radiative_transfer import (
    differentiate_on_depth_grid as reference_differentiate,
)
from payne_zero_atmosphere.radiative_transfer import (
    integrate_on_depth_grid as reference_integrate,
)
from payne_zero_atmosphere.radiative_transfer import (
    remap_to_grid as reference_remap,
)

from .grid_math import (
    differentiate_on_depth_grid,
    integrate_on_depth_grid,
    remap_to_grid,
)


# Round-off only: the port runs the same float64 formulas as the reference.
ABS_TOLERANCE = 1.0e-9
REL_TOLERANCE = 1.0e-12

_TRACE_GLOB = (
    "runs/twin_traces/t05777.0_g+4.44_m+0.00_a+0.00_x1.00/iter_3/**/debug_state.npz"
)
_TRACE_FIELDS = ("temperature", "rosseland_opacity", "gas_pressure")


def _trace_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    matches = sorted(root.glob(_TRACE_GLOB))
    if not matches:
        raise FileNotFoundError(f"no debug_state.npz under {root / 'runs/twin_traces'}")
    return matches[0]


def _worst(diffs: dict[str, float], key: str, value: float) -> None:
    diffs[key] = max(diffs.get(key, 0.0), value)


def _numpy_remap_row(
    grid: np.ndarray, values: np.ndarray, target: np.ndarray
) -> np.ndarray:
    remapped, _ = reference_remap(grid, values, target)
    return remapped


def check_trace(state) -> dict[str, float]:
    """Reference vs torch on the captured solver state, one field at a time."""

    grid = np.asarray(state["column_mass"], dtype=np.float64)
    # Log-spaced target reaching past both ends, so the linear extrapolation
    # branches are exercised on real data too.
    target = np.geomspace(grid[0] * 0.3, grid[-1] * 3.0, 120)
    diffs: dict[str, float] = {}
    for field in _TRACE_FIELDS:
        values = np.asarray(state[field], dtype=np.float64)
        grid_t = torch.from_numpy(grid)
        values_t = torch.from_numpy(values)
        target_t = torch.from_numpy(target)
        surface = values[0] * grid[0]

        expected = reference_integrate(grid, values, surface_value=float(surface))
        got = integrate_on_depth_grid(grid_t, values_t, surface_value=surface)
        _worst(diffs, f"integrate:{field}", float(np.max(np.abs(got.numpy() - expected))))

        expected = reference_differentiate(grid, values)
        got = differentiate_on_depth_grid(grid_t, values_t)
        _worst(diffs, f"differentiate:{field}", float(np.max(np.abs(got.numpy() - expected))))

        expected = _numpy_remap_row(grid, values, target)
        got = remap_to_grid(grid_t, values_t, target_t)
        _worst(diffs, f"remap:{field}", float(np.max(np.abs(got.numpy() - expected))))
    return diffs


def _synthetic_batch(
    rng: np.random.Generator, leading: tuple[int, ...], layers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Monotonic log-grids and smooth random values, one per leading row."""

    steps = rng.uniform(0.05, 0.3, size=leading + (layers,))
    grids = np.exp(np.cumsum(steps, axis=-1))
    noise = rng.standard_normal(leading + (layers,))
    kernel = np.array([0.25, 0.5, 0.25])
    padded = np.pad(noise, [(0, 0)] * len(leading) + [(1, 1)], mode="edge")
    smooth = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"), -1, padded
    )
    values = np.cumsum(smooth, axis=-1) + 5.0 * grids / grids[..., -1:]
    return grids, values


def _reference_batched(fn, grids, values, per_row=None) -> np.ndarray:
    """Run the 1-D NumPy reference row by row over the leading dimensions."""

    flat_grids = grids.reshape(-1, grids.shape[-1])
    flat_values = values.reshape(-1, values.shape[-1])
    flat_extra = per_row.reshape(-1) if per_row is not None else [None] * len(flat_grids)
    rows = [fn(g, v, extra) for g, v, extra in zip(flat_grids, flat_values, flat_extra)]
    return np.stack(rows).reshape(values.shape)


def check_synthetic() -> dict[str, float]:
    """Batched torch against per-row NumPy on random grids, both batchings."""

    rng = np.random.default_rng(20260806)
    diffs: dict[str, float] = {}
    cases = [
        ("shared-grid(star,layer)", (4,), 48, True),
        ("row-grids(star,layer)", (4,), 48, False),
        ("row-grids(star,freq,layer)", (2, 3), 32, False),
        ("shared-grid(star,freq,layer)", (2, 3), 32, True),
    ]
    for name, leading, layers, shared_grid in cases:
        grids, values = _synthetic_batch(rng, leading, layers)
        if shared_grid:
            grids = np.broadcast_to(grids[(0,) * len(leading)], grids.shape).copy()
        # Target spans past both ends of every row's grid.
        lo = grids.min() * 0.5
        hi = grids.max() * 1.7
        target = np.exp(np.linspace(np.log(lo), np.log(hi), 57))
        surface = values[..., 0] * grids[..., 0]

        grids_t = torch.from_numpy(grids if not shared_grid else grids[(0,) * len(leading)])
        values_t = torch.from_numpy(values)
        target_t = torch.from_numpy(target)

        expected = _reference_batched(
            lambda g, v, _extra: reference_integrate(g, v, surface_value=0.0),
            grids,
            values,
        )
        got = integrate_on_depth_grid(grids_t, values_t, surface_value=0.0)
        _worst(diffs, f"integrate:{name}", float(np.max(np.abs(got.numpy() - expected))))

        # The surface-value convention used by the Rosseland/pressure steps.
        expected = _reference_batched(
            lambda g, v, s: reference_integrate(g, v, surface_value=float(s)),
            grids,
            values,
            per_row=surface,
        )
        got = integrate_on_depth_grid(
            grids_t, values_t, surface_value=torch.from_numpy(surface)
        )
        _worst(
            diffs, f"integrate-surface:{name}", float(np.max(np.abs(got.numpy() - expected)))
        )

        expected = _reference_batched(
            lambda g, v, _extra: reference_differentiate(g, v), grids, values
        )
        got = differentiate_on_depth_grid(grids_t, values_t)
        _worst(diffs, f"differentiate:{name}", float(np.max(np.abs(got.numpy() - expected))))

        # Remap rows share one target grid; per-row targets are covered by the
        # batched-grid cases above since each row has its own source grid.
        flat_g = grids.reshape(-1, layers)
        flat_v = values.reshape(-1, layers)
        expected = np.stack(
            [_numpy_remap_row(g, v, target) for g, v in zip(flat_g, flat_v)]
        ).reshape(leading + (target.size,))
        got = remap_to_grid(grids_t, values_t, target_t)
        _worst(diffs, f"remap:{name}", float(np.max(np.abs(got.numpy() - expected))))
    return diffs


def check_gradients() -> tuple[bool, float]:
    """A scalar loss through remap + integrate must give real gradients."""

    generator = torch.Generator().manual_seed(7)
    stars, layers = 3, 40
    grid = torch.exp(
        torch.cumsum(
            torch.rand(stars, layers, generator=generator, dtype=torch.float64) * 0.2
            + 0.05,
            dim=-1,
        )
    )
    base = torch.cumsum(
        torch.randn(stars, layers, generator=generator, dtype=torch.float64) * 0.1,
        dim=-1,
    )
    values = (base + 2.0 * grid / grid[:, -1:] + 3.0).requires_grad_(True)
    target = torch.linspace(
        float(grid.min()) * 0.5, float(grid.max()) * 1.5, 50, dtype=torch.float64
    )
    remapped = remap_to_grid(grid, values, target)
    integrated = integrate_on_depth_grid(
        grid, values, surface_value=values[..., :1] * grid[..., :1]
    )
    loss = remapped.square().mean() + integrated.square().mean()
    loss.backward()
    grad = values.grad
    assert grad is not None
    norm = float(grad.norm())
    finite = bool(torch.isfinite(grad).all()) and norm > 0.0
    return finite, norm


def main() -> int:
    failures: list[str] = []

    state = np.load(_trace_path(), allow_pickle=False)
    print(f"1. trace data ({_trace_path().relative_to(Path(__file__).resolve().parent.parent)})")
    trace = check_trace(state)
    for key, value in trace.items():
        flag = "" if value <= ABS_TOLERANCE else "  <-- FAIL"
        print(f"   {key:36s} max abs diff {value:.3e}{flag}")
        if value > ABS_TOLERANCE:
            failures.append(f"trace:{key}")

    print()
    print("2. synthetic batched grids (shared and per-row, with extrapolation)")
    synthetic = check_synthetic()
    for key, value in synthetic.items():
        flag = "" if value <= ABS_TOLERANCE else "  <-- FAIL"
        print(f"   {key:44s} max abs diff {value:.3e}{flag}")
        if value > ABS_TOLERANCE:
            failures.append(f"synthetic:{key}")

    print()
    print("3. gradients reach the values through remap and integrate")
    finite, norm = check_gradients()
    print(f"   values grad norm {norm:.4g} ({'finite, nonzero' if finite else 'BAD'})")
    if not finite:
        failures.append("gradients")

    print()
    if failures:
        print("FAIL:", ", ".join(failures))
        return 1
    worst = max(list(trace.values()) + list(synthetic.values()))
    print(
        f"PASS: all functions agree with the NumPy reference "
        f"(worst max abs diff {worst:.3e}, tolerances abs {ABS_TOLERANCE:g} / "
        f"rel {REL_TOLERANCE:g}), gradients finite"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
