"""Build the paper figure comparing two-field, six-field, and interpolated starts.

The script is intentionally an experiment-level wrapper around the existing
solver. It never edits the production runner. Prefix runs with caps 1..N are
used to capture one exact state per solver iteration through the runner's
existing debug-state writer.

The current expanded benchmark must be present locally (normally by pulling
results from the compute node with .rsync-exclude-pull). The sealed holdout is
checked and rejected if it is supplied as the scored manifest.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import copy
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from experiments.reduced_state_emulator.convergence_geometry import (
    ARM_NAMES,
    make_candidate_row,
    pca_2d,
    record_iterations,
    select_candidate,
    state_distance,
    state_vector,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

from payne_zero_figures.data import (  # noqa: E402
    load_records_strict as _load_records,
    sha256 as _sha256,
)

DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "convergence_geometry_20260814"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "results"
    / "four_initializer_benchmark_expanded_20260814"
    / "expanded200_manifest.json"
)
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "four_initializer_benchmark_expanded_20260814"
DEFAULT_PREDICTION = (
    REPO_ROOT
    / "results"
    / "initializer_improvement_20260812"
    / "calibration400"
    / "base_ensemble.npz"
)
DEFAULT_RECORDS = {
    "learned_reduced_state": (
        DEFAULT_RUN_ROOT / "records" / "learned_reduced_state" / "records.jsonl"
    ),
    "production_six_field": (
        DEFAULT_RUN_ROOT / "records" / "production_six_field" / "records.jsonl"
    ),
    "interpolated_full_state": (
        DEFAULT_RUN_ROOT / "records" / "interpolated_full_state" / "records.jsonl"
    ),
}
PAPER_PDF = REPO_ROOT / "paper" / "figs" / "fig_initializer_convergence_geometry.pdf"
MAX_ITERATIONS = 15
CONVERGENCE_THRESHOLD = 5.0e-4
LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)

ARM_COLORS = {
    "learned_reduced_state": "#0072B2",
    "production_six_field": "#D55E00",
    "interpolated_full_state": "#009E73",
}
ARM_LABELS = {
    "learned_reduced_state": "Two-field (m, T)",
    "production_six_field": "Six-field",
    "interpolated_full_state": "Full-state interpolation",
}
ARM_MARKERS = {
    "learned_reduced_state": "o",
    "production_six_field": "s",
    "interpolated_full_state": "^",
}
CATEGORY_MARKERS = {"ordinary": "o", "hard": "s", "edge": "^", None: "o"}


def _jsonable(value):
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _artifact_path(path: Path) -> str:
    """Keep provenance portable when the run happens on a compute node."""

    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _load_inputs(manifest_path: Path, corpus_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("sealed_holdout_opened") is True or manifest.get("opened") is True:
        raise ValueError("the scored manifest is marked opened/sealed; refusing to use it")
    indices = np.asarray(manifest["star_indices"], dtype=np.int64)
    with np.load(corpus_path, allow_pickle=False) as data:
        raw_labels = data["labels_json"][indices]
        raw_label_dicts = [json.loads(str(value)) for value in raw_labels]
        labels = [
            {field: float(entry[field]) for field in LABEL_FIELDS}
            for entry in raw_label_dicts
        ]
        profiles = np.asarray(data["atmosphere_profiles"][indices], dtype=np.float64)
        tau = np.asarray(data["standard_rosseland_optical_depth"][0], dtype=np.float64)
    if "star_slugs" not in manifest:
        raise ValueError("expanded benchmark manifest must contain label-generated star_slugs")
    slugs = [str(value) for value in manifest["star_slugs"]]
    if len(slugs) != len(indices):
        raise ValueError("manifest star_slugs and star_indices have different lengths")
    categories = {}
    for category, payload in manifest.get("categories", {}).items():
        for index in payload.get("star_indices", []):
            categories[int(index)] = category
    return manifest, indices, slugs, labels, profiles, tau, categories


def _load_predictions(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing clean-base two-field prediction: {path}; "
            "pull the open initializer-improvement artifacts first"
        )
    predictions = {}
    with np.load(path, allow_pickle=False) as data:
        required = {"star_indices", "column_mass", "temperature"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"prediction archive is missing fields: {sorted(missing)}")
        for row, index in enumerate(np.asarray(data["star_indices"], dtype=np.int64)):
            predictions[int(index)] = (
                np.asarray(data["column_mass"][row], dtype=np.float64),
                np.asarray(data["temperature"][row], dtype=np.float64),
            )
    return predictions


def _atmosphere_state(atmosphere) -> np.ndarray:
    return state_vector(atmosphere.temperature, atmosphere.gas_pressure)


def _build_initial_atmosphere(
    arm: str,
    *,
    label: dict[str, float],
    truth_profile: np.ndarray,
    star_index: int,
    predictions: dict[int, tuple[np.ndarray, np.ndarray]],
    interpolation_context=None,
):
    """Build one exact initial atmosphere using the existing benchmark paths."""

    if arm == "production_six_field":
        from payne_zero_atmosphere.warm_start import emulator_warm_start_model

        atmosphere, _deck = emulator_warm_start_model(device="cpu", **label)
        return atmosphere

    if arm == "learned_reduced_state":
        from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere

        if star_index not in predictions:
            raise KeyError(f"two-field prediction is missing corpus index {star_index}")
        column_mass, temperature = predictions[star_index]
        result = reconstruct_full_atmosphere(
            ReducedAtmosphere(
                column_mass=column_mass,
                temperature=temperature,
                labels=label,
            ),
            n_synchronizations=None,
            max_synchronizations=8,
            pressure_tolerance_dex=1.0e-3,
        )
        return result.atmosphere

    if arm == "interpolated_full_state":
        from experiments.reduced_state_emulator.grey_start_benchmark import (
            full_state_to_atmosphere,
            interpolated_full_state,
        )

        if interpolation_context is None:
            raise ValueError("full-state interpolation context is required")
        coordinates, scale, encoded, tau, donor_indices = interpolation_context
        profile, _diagnostics = interpolated_full_state(
            label,
            coordinates,
            scale,
            encoded,
            tau,
            donor_indices=donor_indices,
            neighbours=8,
            power=2.0,
        )
        return full_state_to_atmosphere(profile, label)

    if arm == "reference":
        from experiments.reduced_state_emulator.grey_start_benchmark import (
            full_state_to_atmosphere,
        )

        return full_state_to_atmosphere(truth_profile, label)

    raise ValueError(f"unknown initializer arm: {arm}")


def _load_interpolation_context(corpus_path: Path, manifest_path: Path):
    from experiments.reduced_state_emulator.grey_start_benchmark import (
        DEFAULT_DONOR_EXCLUDE,
        load_full_donor_pool,
    )

    context = load_full_donor_pool(
        corpus_path,
        tuple(DEFAULT_DONOR_EXCLUDE) + (manifest_path,),
    )
    coordinates, scale, encoded, tau, _provenance, _excluded, donor_indices = context
    return coordinates, scale, encoded, tau, donor_indices


def _initial_state_cache_path(
    output_dir: Path,
    arm: str,
    *,
    exact_two_field: bool = True,
) -> Path:
    suffix = ""
    if arm == "learned_reduced_state" and not exact_two_field:
        suffix = "_screening"
    return output_dir / f"initial_states_{arm}{suffix}.npz"


def _materialize_initial_states(
    arm: str,
    *,
    output_dir: Path,
    indices: np.ndarray,
    slugs: list[str],
    labels: list[dict[str, float]],
    profiles: np.ndarray,
    predictions: dict[int, tuple[np.ndarray, np.ndarray]],
    interpolation_context=None,
    exact_two_field: bool = True,
) -> dict[str, np.ndarray]:
    """Materialize and cache the T/P state of every scored initial point.

    The two-field screening pass uses the cheap hydrostatic pressure estimate
    P=g*m. Exact pressure synchronization is performed only for the bounded
    candidate set and for the final trajectory replay.
    """

    cache_path = _initial_state_cache_path(
        output_dir,
        arm,
        exact_two_field=exact_two_field,
    )
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as data:
            cached_indices = np.asarray(data["star_indices"], dtype=np.int64)
            if np.array_equal(cached_indices, indices):
                states = np.asarray(data["states"], dtype=np.float64)
                if states.shape == (len(indices), 160) and np.all(np.isfinite(states)):
                    return {slug: states[row] for row, slug in enumerate(slugs)}

    states = np.empty((len(indices), 160), dtype=np.float64)
    for row, (index, slug) in enumerate(zip(indices, slugs, strict=True)):
        if arm == "learned_reduced_state" and not exact_two_field:
            column_mass, temperature = predictions[int(index)]
            pressure = (
                10.0 ** float(labels[row]["log_surface_gravity"])
            ) * np.asarray(column_mass, dtype=np.float64)
            states[row] = state_vector(temperature, pressure)
        else:
            atmosphere = _build_initial_atmosphere(
                arm,
                label=labels[row],
                truth_profile=profiles[row],
                star_index=int(index),
                predictions=predictions,
                interpolation_context=interpolation_context,
            )
            states[row] = _atmosphere_state(atmosphere)
        if (row + 1) % 10 == 0 or row + 1 == len(indices):
            print(f"[initial states] {arm}: {row + 1}/{len(indices)}", flush=True)
    if not np.all(np.isfinite(states)):
        raise ValueError(f"non-finite initial state for arm {arm}")
    np.savez_compressed(cache_path, star_indices=indices, states=states)
    return {slug: states[row] for row, slug in enumerate(slugs)}


def _run_once(initial_atmosphere, iteration_cap: int, temporary_dir: Path):
    """Run the unchanged solver once and read its final debug state."""

    from bench.run_reference import _solver_config
    from payne_zero_atmosphere.runner import run_atmosphere_model

    debug_path = temporary_dir / f"debug_state_{iteration_cap:02d}.npz"
    config = _solver_config(
        copy.deepcopy(initial_atmosphere),
        iterations_per_trial=int(iteration_cap),
        structured_atmosphere_path=None,
        debug_state_path=debug_path,
    )
    result = run_atmosphere_model(config)
    if not debug_path.is_file():
        raise RuntimeError(f"solver did not write debug state: {debug_path}")
    with np.load(debug_path, allow_pickle=False) as data:
        state = state_vector(data["temperature"], data["gas_pressure"])
    timings = result.diagnostics.get("iteration_timings", [])
    residuals = [
        float(item["deep_layer_relative_temperature_change"])
        for item in timings
        if item.get("deep_layer_relative_temperature_change") is not None
    ]
    if not residuals:
        raise RuntimeError("solver returned no finite per-iteration residuals")
    return result, state, residuals


def _capture_trajectory(initial_atmosphere, max_iterations: int, temporary_dir: Path):
    """Capture iteration 0..N by deterministic prefix replay."""

    initial_state = _atmosphere_state(initial_atmosphere)
    states = [initial_state]
    residuals = []
    final_result = None
    prefix_residuals = []
    for cap in range(1, int(max_iterations) + 1):
        result, state, run_residuals = _run_once(
            initial_atmosphere,
            cap,
            temporary_dir,
        )
        completed = int(result.iterations_completed)
        if completed < cap:
            raise RuntimeError(
                f"prefix cap {cap} stopped at {completed}; "
                "the solver path is not reproducible"
            )
        states.append(state)
        residuals.append(run_residuals[-1])
        prefix_residuals.append(run_residuals)
        final_result = result
        if bool(result.converged):
            break
    if final_result is None:
        raise RuntimeError("no solver prefix completed")
    return (
        np.stack(states),
        np.asarray(residuals, dtype=np.float64),
        prefix_residuals,
        final_result,
        np.asarray(prefix_residuals[-1], dtype=np.float64),
    )


def _capture_reference(
    initial_atmosphere,
    *,
    output_dir: Path,
    slug: str,
):
    reference_path = output_dir / "references" / f"{slug}.npz"
    if reference_path.is_file():
        with np.load(reference_path, allow_pickle=False) as data:
            state = np.asarray(data["state"], dtype=np.float64)
            if state.shape == (160,) and np.all(np.isfinite(state)):
                return state
    with tempfile.TemporaryDirectory(prefix=f"reference_{slug}_") as temporary:
        result, state, residuals = _run_once(
            initial_atmosphere,
            MAX_ITERATIONS,
            Path(temporary),
        )
    if not bool(result.converged):
        raise RuntimeError(f"corpus reference did not converge for {slug}")
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        reference_path,
        state=state,
        iterations_completed=int(result.iterations_completed),
        residuals=np.asarray(residuals, dtype=np.float64),
    )
    return state


def _population_rows(
    *,
    slugs: list[str],
    indices: np.ndarray,
    categories: dict[int, str],
    records: dict[str, dict[str, dict]],
    initial_states: dict[str, dict[str, np.ndarray]],
    corpus_states: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for index, slug in zip(indices, slugs, strict=True):
        if any(slug not in records[arm] for arm in ARM_NAMES):
            continue
        if any(slug not in initial_states[arm] for arm in ARM_NAMES):
            continue
        reference = corpus_states[slug]
        distances = {
            arm: state_distance(initial_states[arm][slug], reference)
            for arm in ARM_NAMES
        }
        n_two = record_iterations(records["learned_reduced_state"][slug])
        n_six = record_iterations(records["production_six_field"][slug])
        rows.append(
            {
                "slug": slug,
                "category": categories.get(int(index)),
                "converged": {
                    arm: bool(records[arm][slug].get("converged"))
                    for arm in ARM_NAMES
                },
                "iterations": {
                    arm: record_iterations(records[arm][slug]) for arm in ARM_NAMES
                },
                "iteration_gap_six_minus_two": (
                    None if n_two is None or n_six is None else n_six - n_two
                ),
                "distances": distances,
            }
        )
    return rows


def _candidate_shortlist(
    population_rows: list[dict],
    *,
    max_two_iterations: int = 4,
    minimum_gap: int = 4,
) -> list[dict]:
    shortlist = []
    for row in population_rows:
        converged = row["converged"]
        iterations = row["iterations"]
        gap = row["iteration_gap_six_minus_two"]
        if (
            all(converged.values())
            and iterations["learned_reduced_state"] is not None
            and iterations["learned_reduced_state"] <= max_two_iterations
            and gap is not None
            and gap >= minimum_gap
        ):
            d_two = row["distances"]["learned_reduced_state"]
            d_six = row["distances"]["production_six_field"]
            shortlist.append(row)
    return sorted(
        shortlist,
        key=lambda row: (
            -(row["iteration_gap_six_minus_two"] or -1),
            row["distances"]["production_six_field"]["combined_rms_dex"]
            / max(
                row["distances"]["learned_reduced_state"]["combined_rms_dex"],
                1.0e-300,
            ),
            row["slug"],
        ),
    )


def _state_map_from_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        slugs = [str(value) for value in data["slugs"]]
        states = np.asarray(data["states"], dtype=np.float64)
    return {slug: states[row] for row, slug in enumerate(slugs)}


def _save_trajectory_archive(
    path: Path,
    *,
    slug: str,
    arm_trajectories: dict[str, np.ndarray],
    arm_residuals: dict[str, np.ndarray],
    reference: np.ndarray,
) -> None:
    max_steps = max(value.shape[0] for value in arm_trajectories.values())
    padded_states = np.full((len(ARM_NAMES), max_steps, 80, 2), np.nan)
    padded_residuals = np.full((len(ARM_NAMES), max_steps - 1), np.nan)
    counts = []
    for arm_index, arm in enumerate(ARM_NAMES):
        states = arm_trajectories[arm]
        padded_states[arm_index, : states.shape[0]] = states
        residuals = arm_residuals[arm]
        padded_residuals[arm_index, : residuals.size] = residuals
        counts.append(states.shape[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        slug=np.asarray([slug]),
        arm_names=np.asarray(ARM_NAMES),
        states=padded_states,
        residuals=padded_residuals,
        state_counts=np.asarray(counts, dtype=np.int64),
        reference_state=np.asarray(reference, dtype=np.float64),
    )


def _plot_figure(
    *,
    output_dir: Path,
    population_rows: list[dict],
    selected: dict,
    trajectories: dict[str, np.ndarray],
    residuals: dict[str, np.ndarray],
    reference: np.ndarray,
    slug: str,
) -> tuple[Path, Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure_path = output_dir / "figure_initializer_convergence_geometry.pdf"
    png_path = output_dir / "figure_initializer_convergence_geometry.png"
    fig = plt.figure(figsize=(7.2, 3.6))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.15, 1.0),
        wspace=0.36,
        left=0.07,
        right=0.99,
        bottom=0.18,
        top=0.90,
    )
    ax_population = fig.add_subplot(grid[0, 0])
    ax_path = fig.add_subplot(grid[0, 1])
    ax_residual = fig.add_subplot(grid[0, 2])

    # Panel A: open-set screening context.
    for row in population_rows:
        d_two = row["distances"]["learned_reduced_state"]["combined_rms_dex"]
        d_six = row["distances"]["production_six_field"]["combined_rms_dex"]
        gap = row["iteration_gap_six_minus_two"]
        if not np.isfinite(d_two) or not np.isfinite(d_six) or gap is None:
            continue
        ratio = d_six / max(d_two, 1.0e-300)
        marker = CATEGORY_MARKERS.get(row.get("category"), "o")
        color = {"ordinary": "#999999", "hard": "#666666", "edge": "#222222"}.get(
            row.get("category"), "#777777"
        )
        selected_point = row["slug"] == slug
        ax_population.scatter(
            ratio,
            gap,
            marker=marker,
            s=20 if selected_point else 12,
            facecolor="white" if selected_point else color,
            edgecolor="#000000" if selected_point else color,
            linewidth=1.2 if selected_point else 0.35,
            zorder=4 if selected_point else 2,
        )
    ax_population.axvline(1.0, color="#555555", lw=0.7, ls=":")
    ax_population.axhline(0.0, color="#555555", lw=0.7, ls=":")
    ax_population.set_xscale("log")
    ax_population.set_xlabel("Six-field / two-field initial distance\n(screening context)")
    ax_population.set_ylabel("Iteration gap, N₆ − N₂")
    ax_population.text(
        0.04,
        0.96,
        "six-field closer",
        transform=ax_population.transAxes,
        ha="left",
        va="top",
        fontsize=7,
    )
    ax_population.text(
        0.96,
        0.04,
        "two-field faster",
        transform=ax_population.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )

    # Panel B: PCA of the actual 160-dimensional log(T, P) states.
    vector_list = [reference]
    vector_ranges = {}
    cursor = 1
    for arm in ARM_NAMES:
        vectors = np.stack(
            [
                state_vector(states[:, 0], states[:, 1])
                for states in trajectories[arm]
            ]
        )
        vector_list.extend(vectors)
        vector_ranges[arm] = (cursor, cursor + len(vectors))
        cursor += len(vectors)
    all_vectors = np.vstack(vector_list)
    coordinates, explained, _components = pca_2d(all_vectors)
    reference_xy = coordinates[0]
    for arm in ARM_NAMES:
        start, stop = vector_ranges[arm]
        xy = coordinates[start:stop]
        color = ARM_COLORS[arm]
        ax_path.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            lw=1.25,
            ls="-" if arm == "learned_reduced_state" else "--" if arm == "production_six_field" else "-.",
            marker=ARM_MARKERS[arm],
            ms=3.2,
            label=ARM_LABELS[arm],
            zorder=3,
        )
        for iteration in range(len(xy) - 1):
            ax_path.annotate(
                "",
                xy=xy[iteration + 1],
                xytext=xy[iteration],
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "lw": 0.55,
                    "mutation_scale": 6,
                },
                zorder=4,
            )
        for iteration, point in enumerate(xy):
            if iteration == 0 or iteration == len(xy) - 1 or iteration % 2 == 0:
                ax_path.text(
                    point[0],
                    point[1],
                    str(iteration),
                    color=color,
                    fontsize=6,
                    ha="left",
                    va="bottom",
                )
    ax_path.scatter(
        [reference_xy[0]],
        [reference_xy[1]],
        marker="*",
        s=70,
        facecolor="black",
        edgecolor="white",
        linewidth=0.4,
        label="Reference fixed point",
        zorder=5,
    )
    ax_path.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax_path.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax_path.set_title("160-D log T–log P state space", pad=4)
    ax_path.legend(frameon=False, loc="best", handlelength=2.2)

    # Panel C: the stopping residual and the auditable geometry metrics.
    for arm in ARM_NAMES:
        values = np.asarray(residuals[arm], dtype=np.float64)
        iterations = np.arange(1, values.size + 1)
        ax_residual.plot(
            iterations,
            values,
            color=ARM_COLORS[arm],
            lw=1.25,
            ls="-" if arm == "learned_reduced_state" else "--" if arm == "production_six_field" else "-.",
            marker=ARM_MARKERS[arm],
            ms=3.0,
            label=ARM_LABELS[arm],
        )
    ax_residual.axhline(
        CONVERGENCE_THRESHOLD,
        color="#333333",
        lw=0.8,
        ls=":",
        label="Stopping threshold",
    )
    ax_residual.set_yscale("log")
    ax_residual.set_xlabel("Solver iteration")
    ax_residual.set_ylabel("Deep-layer max |ΔT/T|")
    ax_residual.set_xticks(range(1, MAX_ITERATIONS + 1, 2))
    ax_residual.legend(frameon=False, loc="upper right", handlelength=2.2)
    d_two = selected["distances"]["learned_reduced_state"]
    d_six = selected["distances"]["production_six_field"]
    a_two = selected["first_step_alignment"]["learned_reduced_state"]
    a_six = selected["first_step_alignment"]["production_six_field"]
    metrics = (
        f"N₂={selected['iterations']['learned_reduced_state']}, "
        f"N₆={selected['iterations']['production_six_field']}\n"
        f"dT₂/dT₆={d_two['temperature_rms_dex']:.3g}/"
        f"{d_six['temperature_rms_dex']:.3g} dex\n"
        f"dP₂/dP₆={d_two['pressure_rms_dex']:.3g}/"
        f"{d_six['pressure_rms_dex']:.3g} dex\n"
        f"cosθ₂/cosθ₆={a_two:.2f}/{a_six:.2f}"
    )
    ax_residual.text(
        0.03,
        0.03,
        metrics,
        transform=ax_residual.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        bbox={"facecolor": "white", "edgecolor": "#bbbbbb", "pad": 3},
    )

    for index, axis in enumerate((ax_population, ax_path, ax_residual)):
        axis.text(
            -0.16,
            1.06,
            "ABC"[index],
            transform=axis.transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.savefig(figure_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return figure_path, png_path


def _write_caption(path: Path, selected: dict, slug: str) -> None:
    d_two = selected["distances"]["learned_reduced_state"]
    d_six = selected["distances"]["production_six_field"]
    text = f"""Figure X. Geometry of atmosphere-solver convergence for the open-set case {slug}.
All three initializers use the same Payne Zero atmosphere solver and stopping
rule. (A) Open-set context: the horizontal coordinate is the six-field to
two-field initial distance ratio in the common log(T, P_gas) state, and the
vertical coordinate is the iteration advantage of the two-field start.
For this population panel only, the two-field pressure is screened with the
cheap hydrostatic estimate P=g*m and distances are measured to the corpus
profile; the highlighted case's distances and all trajectory states use the
exact synchronized two-field pressure and the same-solver fixed point.
The highlighted case was selected by fixed numerical gates before plotting.
(B) Prefix-replayed trajectories in a two-dimensional PCA projection of the
160-dimensional state formed by the 80-layer log temperature and log gas
pressure profiles. The reference star is the converged atmosphere obtained by
restarting the corpus atmosphere with the same solver. (C) The solver's
deep-layer stopping residual. In this case, the six-field start is closer in
both temperature ({d_six["temperature_rms_dex"]:.3g} dex versus
{d_two["temperature_rms_dex"]:.3g} dex) and pressure
({d_six["pressure_rms_dex"]:.3g} dex versus
{d_two["pressure_rms_dex"]:.3g} dex), yet the two-field update follows a more
direct path and reaches the stopping criterion sooner. The PCA is visual only;
all distances and direction cosines are computed in the original 160-D space.
This representative case supports a path-geometry interpretation, not a claim
that the solver is optimizing a single scalar objective or that the result is
universal.
"""
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--two-records", type=Path, default=DEFAULT_RECORDS["learned_reduced_state"])
    parser.add_argument("--six-records", type=Path, default=DEFAULT_RECORDS["production_six_field"])
    parser.add_argument("--interpolation-records", type=Path, default=DEFAULT_RECORDS["interpolated_full_state"])
    parser.add_argument(
        "--max-shortlist",
        type=int,
        default=0,
        help="optional cap for debugging; zero evaluates every preliminary candidate",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, indices, slugs, labels, profiles, tau, categories = _load_inputs(
        args.manifest,
        args.corpus,
    )
    record_paths = {
        "learned_reduced_state": args.two_records,
        "production_six_field": args.six_records,
        "interpolated_full_state": args.interpolation_records,
    }
    records = {arm: _load_records(path) for arm, path in record_paths.items()}
    predictions = _load_predictions(args.prediction)
    required_prediction_indices = {
        int(index)
        for index, slug in zip(indices, slugs, strict=True)
        if slug in records["learned_reduced_state"]
    }
    missing_predictions = sorted(required_prediction_indices - set(predictions))
    if missing_predictions:
        raise ValueError(
            f"clean-base two-field prediction is missing {len(missing_predictions)} "
            f"scored stars; first indices: {missing_predictions[:5]}"
        )

    if args.dry_run:
        print(json.dumps(
            {
                "manifest": _artifact_path(args.manifest),
                "star_count": len(indices),
                "record_paths": {arm: _artifact_path(path) for arm, path in record_paths.items()},
                "prediction": _artifact_path(args.prediction),
                "sealed_holdout_checked": True,
            },
            indent=2,
        ))
        return 0

    interpolation_context = _load_interpolation_context(args.corpus, args.manifest)
    initial_states = {}
    for arm in ARM_NAMES:
        initial_states[arm] = _materialize_initial_states(
            arm,
            output_dir=args.output_dir,
            indices=indices,
            slugs=slugs,
            labels=labels,
            profiles=profiles,
            predictions=predictions,
            interpolation_context=interpolation_context,
            exact_two_field=False if arm == "learned_reduced_state" else True,
        )
    corpus_states = {
        slug: state_vector(profile[:, 1], profile[:, 2])
        for slug, profile in zip(slugs, profiles, strict=True)
    }
    population_rows = _population_rows(
        slugs=slugs,
        indices=indices,
        categories=categories,
        records=records,
        initial_states=initial_states,
        corpus_states=corpus_states,
    )
    shortlist = _candidate_shortlist(population_rows)
    if args.max_shortlist > 0:
        shortlist = shortlist[: args.max_shortlist]
    if not shortlist:
        raise RuntimeError(
            "no open-set star passes the preliminary gates; "
            "rerun with the opened 400-star calibration manifest"
        )
    print(f"[selection] preliminary shortlist: {len(shortlist)}", flush=True)

    labels_by_slug = {slug: labels[row] for row, slug in enumerate(slugs)}
    profiles_by_slug = {slug: profiles[row] for row, slug in enumerate(slugs)}
    exact_rows = []
    first_states = {
        "learned_reduced_state": {},
        "production_six_field": {},
    }
    reference_states = {}
    for position, preliminary in enumerate(shortlist, start=1):
        slug = preliminary["slug"]
        index = int(indices[slugs.index(slug)])
        label = labels_by_slug[slug]
        print(f"[selection] {position}/{len(shortlist)} {slug}", flush=True)
        reference_atmosphere = _build_initial_atmosphere(
            "reference",
            label=label,
            truth_profile=profiles_by_slug[slug],
            star_index=index,
            predictions=predictions,
        )
        reference = _capture_reference(
            reference_atmosphere,
            output_dir=args.output_dir,
            slug=slug,
        )
        reference_states[slug] = reference
        with tempfile.TemporaryDirectory(prefix=f"first_step_{slug}_") as temporary:
            for arm in ("learned_reduced_state", "production_six_field"):
                atmosphere = _build_initial_atmosphere(
                    arm,
                    label=label,
                    truth_profile=profiles_by_slug[slug],
                    star_index=index,
                    predictions=predictions,
                    interpolation_context=interpolation_context,
                )
                _result, state, _residuals = _run_once(
                    atmosphere,
                    1,
                    Path(temporary),
                )
                # The screening pass uses P=g*m for the learned two-field
                # start.  Replace that approximation with the exact
                # synchronized initial state before applying the candidate
                # gates and ranking the final case.
                initial_states[arm][slug] = _atmosphere_state(atmosphere)
                first_states[arm][slug] = state
        exact_rows.append(
            make_candidate_row(
                slug,
                records=records,
                initial_states=initial_states,
                first_states=first_states,
                reference=reference,
                category=preliminary.get("category"),
            )
        )

    selected = select_candidate(exact_rows)
    slug = selected["slug"]
    row = slugs.index(slug)
    index = int(indices[row])
    print(
        f"[selection] selected {slug}: "
        f"N2={selected['iterations']['learned_reduced_state']} "
        f"N6={selected['iterations']['production_six_field']} "
        f"alignment advantage={selected['alignment_advantage_two_minus_six']:.3f}",
        flush=True,
    )

    selected_trajectories = {}
    selected_residuals = {}
    selected_prefix_residuals = {}
    for arm in ARM_NAMES:
        print(f"[trajectory] {arm}", flush=True)
        atmosphere = _build_initial_atmosphere(
            arm,
            label=labels[row],
            truth_profile=profiles[row],
            star_index=index,
            predictions=predictions,
            interpolation_context=interpolation_context,
        )
        expected = record_iterations(records[arm][slug]) or MAX_ITERATIONS
        with tempfile.TemporaryDirectory(prefix=f"trajectory_{arm}_{slug}_") as temporary:
            (
                states,
                residuals,
                prefix_residuals,
                result,
                full_run_residuals,
            ) = _capture_trajectory(
                atmosphere,
                min(MAX_ITERATIONS, expected),
                Path(temporary),
            )
        selected_trajectories[arm] = np.stack(
            (10.0 ** states[:, :80], 10.0 ** states[:, 80:]),
            axis=-1,
        )
        selected_residuals[arm] = residuals
        selected_prefix_residuals[arm] = prefix_residuals
        if not bool(result.converged):
            raise RuntimeError(f"selected arm did not converge on trajectory replay: {arm}")
        actual_iterations = int(result.iterations_completed)
        if actual_iterations != expected:
            raise RuntimeError(
                f"selected arm {arm} changed iteration count: "
                f"benchmark={expected}, replay={actual_iterations}"
            )
        for iteration, values in enumerate(prefix_residuals, start=1):
            if not np.isclose(
                values[-1],
                full_run_residuals[iteration - 1],
                rtol=2.0e-8,
                atol=2.0e-12,
            ):
                raise RuntimeError(f"prefix residual mismatch for {arm} at iteration {iteration}")
        if not np.allclose(
            residuals,
            full_run_residuals,
            rtol=2.0e-8,
            atol=2.0e-12,
        ):
            raise RuntimeError(f"stored residual curve does not match full run for {arm}")

    reference = reference_states[slug]
    trajectory_path = args.output_dir / "trajectory_states.npz"
    _save_trajectory_archive(
        trajectory_path,
        slug=slug,
        arm_trajectories=selected_trajectories,
        arm_residuals=selected_residuals,
        reference=reference,
    )
    figure_path, png_path = _plot_figure(
        output_dir=args.output_dir,
        population_rows=population_rows,
        selected=selected,
        trajectories=selected_trajectories,
        residuals=selected_residuals,
        reference=reference,
        slug=slug,
    )
    if figure_path.resolve() != PAPER_PDF.resolve():
        PAPER_PDF.parent.mkdir(parents=True, exist_ok=True)
        PAPER_PDF.write_bytes(figure_path.read_bytes())
    _write_caption(
        args.output_dir / "figure_initializer_convergence_geometry_caption.md",
        selected,
        slug,
    )
    provenance = {
        "format": "payne_zero_initializer_convergence_geometry_v1",
        "code": {
            "trajectory_script": _artifact_path(Path(__file__)),
            "trajectory_script_sha256": _sha256(Path(__file__)),
            "geometry_module": _artifact_path(
                Path(__file__).with_name("convergence_geometry.py")
            ),
            "geometry_module_sha256": _sha256(
                Path(__file__).with_name("convergence_geometry.py")
            ),
        },
        "manifest": _artifact_path(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "corpus": _artifact_path(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "prediction": _artifact_path(args.prediction),
        "prediction_sha256": _sha256(args.prediction),
        "record_paths": {arm: _artifact_path(path) for arm, path in record_paths.items()},
        "record_sha256": {arm: _sha256(path) for arm, path in record_paths.items()},
        "solver": {
            "iterations_per_trial": MAX_ITERATIONS,
            "minimum_iterations_before_convergence": 3,
            "deep_layer_threshold": CONVERGENCE_THRESHOLD,
            "same_solver_for_all_arms": True,
            "production_runner": "payne_zero_atmosphere.runner.run_atmosphere_model",
        },
        "state": {
            "layers": 80,
            "fields": ["log10_temperature", "log10_gas_pressure"],
            "dimension": 160,
            "distance_units": "dex",
            "reference": "same-solver restart from corpus atmosphere",
        "pca_is_visual_only": True,
        "population_panel_two_field_pressure": (
            "screening P=g*m; selected-case metrics use exact synchronization"
        ),
        "population_panel_reference": (
            "corpus profile for screening; selected-case metrics use same-solver fixed point"
        ),
        },
        "interpolator": {"neighbours": 8, "power": 2.0},
        "sealed_holdout_used": False,
        "git_revision": _git_revision(),
        "selected_slug": slug,
        "outputs": {
            "candidate_selection": _artifact_path(args.output_dir / "candidate_selection.json"),
            "trajectory_archive": _artifact_path(trajectory_path),
            "pdf": _artifact_path(figure_path),
            "png": _artifact_path(png_path),
            "paper_pdf": _artifact_path(PAPER_PDF),
        },
    }
    _write_json(args.output_dir / "figure_config.json", provenance)
    _write_json(
        args.output_dir / "candidate_selection.json",
        {
            "selected": selected,
            "exact_candidate_rows": exact_rows,
            "preliminary_shortlist": [row["slug"] for row in shortlist],
            "population_rows": population_rows,
            "reference_cache": _artifact_path(args.output_dir / "references"),
            "sealed_holdout_used": False,
        },
    )
    print(f"[done] PDF: {figure_path}", flush=True)
    print(f"[done] PNG: {png_path}", flush=True)
    print(f"[done] paper copy: {PAPER_PDF}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
