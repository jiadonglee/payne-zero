"""Render the M-star iteration tomography into per-case diagnostic figures.

Reads the per-iteration NPZ series written by
``m_star_iteration_tomography_v1`` and draws, per case:

1. ``|dT_raw|/T`` over (iteration, log tau) -- where the correction lives and
   whether it decays;
2. ``superadiabatic_gradient`` over the same axes -- convective-topology
   flips;
3. molecular-equilibrium Newton pass count -- EOS stiffness by depth;
4. per-iteration scalars: p95 flux error, signflip fraction of the applied
   correction, and max |dT_raw|/T.

Usage::

    python -m experiments.reduced_state_emulator.m_star_iteration_tomography_v1_figures \
        --result-root results/m_star_iteration_tomography_v1 --out figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE_LABELS = {
    "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3500": "A 3500 K / 4.5 / 0.0",
    "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3400": "B 3400 K / 4.5 / 0.0",
    "g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3300": "C 3300 K / 4.5 / 0.0",
    "g+4.50_m-0.50_a+0.00_c+0.00_x1.00_t3600": "D 3600 K / 4.5 / -0.5",
}


def load_series(case_dir: Path, leg: str) -> dict[str, np.ndarray] | None:
    iteration_dir = case_dir / "iterations" / leg
    paths = sorted(iteration_dir.glob("iter_*.npz"))
    if not paths:
        return None
    series: dict[str, list] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                series.setdefault(key, []).append(data[key])
    return {
        key: (np.stack(value) if np.asarray(value[0]).ndim else np.asarray(value))
        for key, value in series.items()
    }


def case_figure(case_dir: Path, case_id: str, out_dir: Path) -> Path | None:
    primary = load_series(case_dir, "primary")
    if primary is None:
        return None
    log_tau = primary["log_tau_standard"][0]
    iterations = primary["iteration"]

    raw_ratio = np.abs(primary["raw_temperature_correction"]) / np.maximum(
        primary["temperature_pre"], 1.0
    )
    sag = primary["superadiabatic_gradient"]
    newton = primary["molecular_newton_iterations"]
    applied = primary["applied_temperature_correction"]
    signflip = np.asarray(
        [
            np.mean(np.sign(row[1:]) != np.sign(row[:-1]))
            for row in applied
        ]
    )
    p95_flux = primary["timing_p95_absolute_flux_error_percent"]

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    figure.suptitle(
        f"{CASE_LABELS.get(case_id, case_id)}  ({len(iterations)} iterations)"
    )

    def heatmap(axis, values, title, cmap="viridis"):
        image = axis.pcolormesh(
            log_tau, iterations, values, shading="nearest", cmap=cmap
        )
        axis.set_xlabel("log tau_R")
        axis.set_ylabel("iteration")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)

    heatmap(axes[0][0], np.log10(np.maximum(raw_ratio, 1.0e-16)),
            "log10 |dT_raw|/T")
    heatmap(axes[0][1], sag, "nabla - nabla_ad", cmap="RdBu_r")
    heatmap(axes[1][0], newton, "molecular Newton passes", cmap="magma")

    axis = axes[1][1]
    axis.plot(iterations, p95_flux, "o-", label="p95 |flux error| %")
    axis.plot(iterations, signflip * 100.0, "s-", label="signflip fraction %")
    axis.plot(
        iterations,
        np.max(raw_ratio, axis=1) * 100.0,
        "^-",
        label="max |dT_raw|/T %",
    )
    axis.set_xlabel("iteration")
    axis.set_yscale("log")
    axis.legend(fontsize=8)
    axis.set_title("iteration scalars")
    axis.grid(alpha=0.3)

    figure.tight_layout()
    out_path = out_dir / f"mstar_tomography_v1_{case_id}.png"
    figure.savefig(out_path, dpi=150)
    plt.close(figure)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="results/m_star_iteration_tomography_v1")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args(argv)

    result_root = Path(args.result_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for case_dir in sorted((result_root / "cases" / "dwarf").glob("*/*/")):
        case_id = f"{case_dir.parent.name}_{case_dir.name}"
        path = case_figure(case_dir, case_id, out_dir)
        if path is not None:
            written.append(str(path))
            print(path)
    if not written:
        print("no completed cases found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
