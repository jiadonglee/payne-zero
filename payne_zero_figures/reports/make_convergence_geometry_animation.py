"""Create an intuitive animation of the selected initializer convergence.

Unlike the PCA figure, this animation uses directly interpretable quantities:

* RMS error in log temperature versus RMS error in log gas pressure;
* layer-by-layer fractional temperature and pressure errors relative to the
  same solver-restarted reference atmosphere; and
* the solver's actual deep-layer stopping residual.

The archive already contains prefix-replayed states, so this script does not
call or modify the production solver.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.reduced_state_emulator.convergence_geometry import (
    ARM_NAMES,
    state_distance,
    state_vector,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

from payne_zero_figures.data import sha256 as _sha256  # noqa: E402

DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "convergence_geometry_20260814"
DEFAULT_ARCHIVE = DEFAULT_RESULT_ROOT / "trajectory_states.npz"
DEFAULT_SELECTION = DEFAULT_RESULT_ROOT / "candidate_selection.json"
THRESHOLD = 5.0e-4

COLORS = {
    "learned_reduced_state": "#0072B2",
    "production_six_field": "#D55E00",
    "interpolated_full_state": "#009E73",
}
LABELS = {
    "learned_reduced_state": "Two-field (m, T)",
    "production_six_field": "Six-field",
    "interpolated_full_state": "Full-state interpolation",
}
LINESTYLES = {
    "learned_reduced_state": "-",
    "production_six_field": "--",
    "interpolated_full_state": "-.",
}
MARKERS = {
    "learned_reduced_state": "o",
    "production_six_field": "s",
    "interpolated_full_state": "^",
}


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _load_data(archive_path: Path, selection_path: Path) -> dict:
    selection = json.loads(selection_path.read_text())
    with np.load(archive_path, allow_pickle=False) as data:
        archive_names = [str(value) for value in data["arm_names"]]
        if tuple(archive_names) != ARM_NAMES:
            raise ValueError(f"unexpected arm order: {archive_names}")
        states = np.asarray(data["states"], dtype=np.float64)
        residuals = np.asarray(data["residuals"], dtype=np.float64)
        counts = np.asarray(data["state_counts"], dtype=np.int64)
        reference = np.asarray(data["reference_state"], dtype=np.float64)
    if states.ndim != 4 or states.shape[0] != len(ARM_NAMES) or states.shape[2:] != (80, 2):
        raise ValueError(f"unexpected trajectory state shape: {states.shape}")
    if residuals.shape != (len(ARM_NAMES), states.shape[1] - 1):
        raise ValueError(f"unexpected residual shape: {residuals.shape}")
    if reference.shape != (160,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference state must be a finite 160-dimensional vector")
    if np.any(counts < 2) or np.any(counts > states.shape[1]):
        raise ValueError(f"invalid state counts: {counts}")
    for arm_index, count in enumerate(counts):
        if not np.all(np.isfinite(states[arm_index, : int(count)])):
            raise ValueError("trajectory archive contains non-finite physical states")

    reference_log_t = reference[:80]
    reference_log_p = reference[80:]
    diagnostics = {}
    for arm_index, arm in enumerate(ARM_NAMES):
        count = int(counts[arm_index])
        physical = states[arm_index, :count]
        log_t = np.log10(physical[:, :, 0])
        log_p = np.log10(physical[:, :, 1])
        delta_log_t = log_t - reference_log_t[None, :]
        delta_log_p = log_p - reference_log_p[None, :]
        temperature_percent = 100.0 * np.expm1(np.log(10.0) * delta_log_t)
        pressure_percent = 100.0 * np.expm1(np.log(10.0) * delta_log_p)
        distance_rows = []
        for state in physical:
            distance_rows.append(
                state_distance(
                    state_vector(state[:, 0], state[:, 1]),
                    reference,
                )
            )
        diagnostics[arm] = {
            "count": count,
            "delta_log_t": delta_log_t,
            "delta_log_p": delta_log_p,
            "temperature_percent": temperature_percent,
            "pressure_percent": pressure_percent,
            "temperature_rms_dex": np.asarray(
                [row["temperature_rms_dex"] for row in distance_rows]
            ),
            "pressure_rms_dex": np.asarray(
                [row["pressure_rms_dex"] for row in distance_rows]
            ),
            "combined_rms_dex": np.asarray(
                [row["combined_rms_dex"] for row in distance_rows]
            ),
            "residuals": residuals[arm_index, : count - 1],
        }
    return {
        "selection": selection,
        "states": states,
        "counts": counts,
        "reference": reference,
        "diagnostics": diagnostics,
    }


def _set_limits(values: list[np.ndarray], *, symmetric: bool = False, floor: float = 0.0):
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    low = float(np.min(finite))
    high = float(np.max(finite))
    if symmetric:
        bound = max(abs(low), abs(high), floor)
        bound *= 1.12
        return -bound, bound
    span = max(high - low, floor)
    pad = 0.08 * span
    return low - pad, high + pad


def _make_figure(data: dict, output_dir: Path):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    diagnostics = data["diagnostics"]
    counts = data["counts"]
    max_iteration = int(np.max(counts) - 1)
    selection = data["selection"]["selected"]
    slug = str(selection["slug"])

    t_values = [diagnostics[arm]["temperature_rms_dex"] for arm in ARM_NAMES]
    p_values = [diagnostics[arm]["pressure_rms_dex"] for arm in ARM_NAMES]
    x_lim = (0.0, max(float(np.max(value)) for value in t_values) * 1.18)
    y_lim = (0.0, max(float(np.max(value)) for value in p_values) * 1.18)
    t_profile_lim = _set_limits(
        [diagnostics[arm]["temperature_percent"] for arm in ARM_NAMES],
        symmetric=True,
        floor=0.5,
    )
    p_profile_lim = _set_limits(
        [diagnostics[arm]["pressure_percent"] for arm in ARM_NAMES],
        symmetric=True,
        floor=5.0,
    )
    residual_max = max(
        float(np.max(diagnostics[arm]["residuals"])) for arm in ARM_NAMES
    )
    residual_min = min(
        float(np.min(diagnostics[arm]["residuals"])) for arm in ARM_NAMES
    )

    fig = plt.figure(figsize=(9.0, 5.4))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1.15, 1.0, 1.0),
        height_ratios=(1.0, 1.0),
        left=0.075,
        right=0.985,
        bottom=0.10,
        top=0.86,
        wspace=0.36,
        hspace=0.48,
    )
    ax_error = fig.add_subplot(grid[:, 0])
    ax_temperature = fig.add_subplot(grid[0, 1])
    ax_pressure = fig.add_subplot(grid[1, 1], sharex=ax_temperature)
    ax_residual = fig.add_subplot(grid[:, 2])

    layer = np.arange(1, 81)
    reference_marker = ax_error.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=100,
        color="black",
        edgecolor="white",
        linewidth=0.5,
        zorder=7,
        label="Reference fixed point",
    )
    ax_error.set_xlim(*x_lim)
    ax_error.set_ylim(*y_lim)
    ax_error.set_xlabel(r"RMS $|\Delta\log_{10} T|$ (dex)")
    ax_error.set_ylabel(r"RMS $|\Delta\log_{10} P_{\rm gas}|$ (dex)")
    ax_error.set_title("Distance to reference", pad=5)
    ax_error.grid(True, color="#dddddd", linewidth=0.5, zorder=0)

    ax_temperature.axhline(0.0, color="#777777", lw=0.6, ls=":")
    ax_pressure.axhline(0.0, color="#777777", lw=0.6, ls=":")
    ax_temperature.set_ylim(*t_profile_lim)
    ax_pressure.set_ylim(*p_profile_lim)
    ax_temperature.set_ylabel(r"$100(T/T_{\rm ref}-1)$ (%)")
    ax_pressure.set_ylabel(r"$100(P/P_{\rm ref}-1)$ (%)")
    ax_pressure.set_xlabel("Atmospheric layer")
    ax_temperature.set_title("Layer-by-layer profile error", pad=5)
    ax_temperature.tick_params(labelbottom=False)
    for axis in (ax_temperature, ax_pressure):
        axis.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)
        axis.set_xlim(1, 80)

    ax_residual.set_yscale("log")
    ax_residual.set_xlim(0.7, max_iteration + 0.4)
    ax_residual.set_ylim(residual_min * 0.75, residual_max * 1.35)
    ax_residual.set_xlabel("Solver iteration")
    ax_residual.set_ylabel(r"Deep-layer max $|\Delta T/T|$")
    ax_residual.set_title("Actual stopping diagnostic", pad=5)
    ax_residual.axhline(
        THRESHOLD,
        color="#333333",
        lw=0.9,
        ls=":",
        label=r"Stopping threshold ($5\times10^{-4}$)",
    )

    error_lines = {}
    error_points = {}
    temperature_lines = {}
    pressure_lines = {}
    residual_lines = {}
    residual_points = {}
    for arm in ARM_NAMES:
        style = {
            "color": COLORS[arm],
            "lw": 1.6,
            "ls": LINESTYLES[arm],
            "marker": MARKERS[arm],
            "ms": 4.0,
            "label": LABELS[arm],
        }
        error_line, = ax_error.plot([], [], **style)
        error_point, = ax_error.plot([], [], marker=MARKERS[arm], color=COLORS[arm], ms=7, ls="None")
        temperature_line, = ax_temperature.plot([], [], **style)
        pressure_line, = ax_pressure.plot([], [], **style)
        residual_line, = ax_residual.plot([], [], **style)
        residual_point, = ax_residual.plot([], [], marker=MARKERS[arm], color=COLORS[arm], ms=7, ls="None")
        error_lines[arm] = error_line
        error_points[arm] = error_point
        temperature_lines[arm] = temperature_line
        pressure_lines[arm] = pressure_line
        residual_lines[arm] = residual_line
        residual_points[arm] = residual_point

    ax_error.legend(frameon=False, loc="upper right", handlelength=2.4)
    ax_temperature.legend(frameon=False, loc="best", handlelength=2.4)
    ax_residual.legend(frameon=False, loc="upper right", handlelength=2.4)
    title = fig.text(
        0.075,
        0.975,
        "",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    subtitle = fig.text(0.075, 0.935, "", ha="left", va="top", fontsize=8)

    for axis in (ax_error, ax_temperature, ax_pressure, ax_residual):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    ax_error.text(-0.18, 1.05, "A", transform=ax_error.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_temperature.text(-0.18, 1.05, "B", transform=ax_temperature.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_residual.text(-0.18, 1.05, "C", transform=ax_residual.transAxes, fontsize=10, fontweight="bold", va="top")

    def update(frame: int):
        current_iteration = int(frame)
        title.set_text(f"Selected case: {slug}")
        subtitle.set_text(
            f"Iteration {current_iteration}: each marker is the actual solver state; "
            f"two-field stops at N={int(counts[0]-1)}, six-field at N={int(counts[1]-1)}, "
            f"interpolation at N={int(counts[2]-1)}"
        )
        artists = [title, subtitle]
        for arm in ARM_NAMES:
            arm_data = diagnostics[arm]
            state_index = min(current_iteration, arm_data["count"] - 1)
            error_lines[arm].set_data(
                arm_data["temperature_rms_dex"][: state_index + 1],
                arm_data["pressure_rms_dex"][: state_index + 1],
            )
            error_points[arm].set_data(
                [arm_data["temperature_rms_dex"][state_index]],
                [arm_data["pressure_rms_dex"][state_index]],
            )
            temperature_lines[arm].set_data(
                layer,
                arm_data["temperature_percent"][state_index],
            )
            pressure_lines[arm].set_data(
                layer,
                arm_data["pressure_percent"][state_index],
            )
            residual_count = min(current_iteration, arm_data["residuals"].size)
            residual_lines[arm].set_data(
                np.arange(1, residual_count + 1),
                arm_data["residuals"][:residual_count],
            )
            if residual_count:
                residual_points[arm].set_data(
                    [residual_count],
                    [arm_data["residuals"][residual_count - 1]],
                )
            else:
                residual_points[arm].set_data([], [])
            artists.extend(
                [
                    error_lines[arm],
                    error_points[arm],
                    temperature_lines[arm],
                    pressure_lines[arm],
                    residual_lines[arm],
                    residual_points[arm],
                ]
            )
        return artists

    update(0)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_frame = output_dir / "convergence_geometry_animation_final.png"
    final_pdf = output_dir / "convergence_geometry_animation_final.pdf"
    first_frame = output_dir / "convergence_geometry_animation_iteration1.png"
    animation_path = output_dir / "convergence_geometry_animation.gif"
    mp4_path = output_dir / "convergence_geometry_animation.mp4"
    fig.savefig(first_frame, dpi=300, bbox_inches="tight")
    update(1)
    fig.savefig(first_frame, dpi=300, bbox_inches="tight")
    update(max_iteration)
    fig.savefig(final_frame, dpi=300, bbox_inches="tight")
    fig.savefig(final_pdf, bbox_inches="tight")

    from matplotlib.animation import FuncAnimation, PillowWriter

    frame_sequence = list(range(max_iteration + 1))
    animation = FuncAnimation(
        fig,
        update,
        frames=frame_sequence,
        interval=1000,
        blit=False,
        repeat=False,
    )
    animation.save(animation_path, writer=PillowWriter(fps=1), dpi=120)
    mp4_written = False
    try:
        from matplotlib.animation import FFMpegWriter

        animation.save(
            mp4_path,
            writer=FFMpegWriter(fps=1, bitrate=1800),
            dpi=140,
        )
        mp4_written = True
    except (FileNotFoundError, RuntimeError):
        if mp4_path.exists():
            mp4_path.unlink()
    plt.close(fig)
    return {
        "first_frame": first_frame,
        "final_frame": final_frame,
        "final_pdf": final_pdf,
        "gif": animation_path,
        "mp4": mp4_path if mp4_written else None,
        "frame_count": len(frame_sequence),
        "max_iteration": max_iteration,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output_dir / ".mplconfig"))
    data = _load_data(args.archive, args.selection)
    outputs = _make_figure(data, args.output_dir)
    config = {
        "format": "payne_zero_initializer_convergence_animation_v1",
        "code": {
            "animation_script": _relative(Path(__file__)),
            "animation_script_sha256": _sha256(Path(__file__)),
        },
        "archive": _relative(args.archive),
        "archive_sha256": _sha256(args.archive),
        "selection": _relative(args.selection),
        "selected_slug": data["selection"]["selected"]["slug"],
        "state": {
            "layers": 80,
            "coordinates": [
                "RMS absolute log-temperature error (dex)",
                "RMS absolute log-gas-pressure error (dex)",
                "fractional layer errors relative to reference",
            ],
            "reference": "same-solver restart from corpus atmosphere",
        },
        "frames": {
            "initial_frame": 0,
            "solver_iteration_frames": list(range(int(outputs["max_iteration"]) + 1)),
            "fps": 1,
            "frame_count": int(outputs["frame_count"]),
        },
        "outputs": {
            key: None if value is None else _relative(value)
            for key, value in outputs.items()
            if key in {"first_frame", "final_frame", "final_pdf", "gif", "mp4"}
        },
    }
    (args.output_dir / "animation_config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
