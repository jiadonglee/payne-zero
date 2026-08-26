"""Solver benchmark for analytic Eddington-grey starting atmospheres."""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import gc
import json
import signal
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from bench.labels import LABEL_FIELDS, StellarLabels
from bench.report import load_records, summarize
from bench.run_reference import StarRecord, TrialRecord


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
DEFAULT_MANIFEST = REPO_ROOT / "results" / "reconstruction_metrics.json"
DEFAULT_PREDICTION = (
    REPO_ROOT
    / "artifacts"
    / "reduced_state_emulator"
    / "physical"
    / "predicted_physical_ensemble.npz"
)
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "grey_start_benchmark_20260812"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / "grey_start_benchmark_20260812"
PERTURBATION_SEEDS = (20260812, 20260813, 20260814)
G_RAD_FLOOR = 2.0577175465785027
FULL_STATE_FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)

# The interpolated-grid arm is the baseline a MARCS/ATLAS user actually has:
# no network, no truth profile for the target, just the nearest already-converged
# atmospheres in the corpus. Structure is set by these four labels; the fifth
# (microturbulence) enters the deck, not the interpolation metric.
INTERPOLATION_LABELS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
)
DEFAULT_INTERPOLATION_NEIGHBOURS = 8
DEFAULT_INTERPOLATION_POWER = 2.0
# Every manifest that has ever served as an evaluation, audit or sealed set. The
# donor pool is the corpus minus their union, so the interpolator is never handed
# a star that any arm is scored on. This is marginally conservative -- the
# network was trained before the later manifests were carved out -- and the bias
# it introduces runs against interpolation, which is the safe direction.
DEFAULT_DONOR_EXCLUDE = (
    REPO_ROOT / "results" / "reconstruction_metrics.json",
    REPO_ROOT / "results" / "sealed_solver_subset_20260808.json",
    REPO_ROOT / "results" / "sealed_audit_20260808.json",
    REPO_ROOT / "results" / "sealed_audit_20260811.json",
    REPO_ROOT / "results" / "sealed_initializer_holdout_20260812.json",
    REPO_ROOT / "results" / "initializer_calibration_20260812.json",
)


def analytic_grey_atmosphere(
    labels: dict[str, float], *, perturbation_seed: int | None = None
):
    """Build a legal 80-layer seed without a neural model or truth profile."""

    from payne_zero_atmosphere.atmosphere_io import parse_atmosphere_deck
    from payne_zero_atmosphere.run_setup import standard_rosseland_optical_depth_grid
    from payne_zero_atmosphere.warm_start import format_warm_start_deck

    tau = standard_rosseland_optical_depth_grid(80)
    temperature = float(labels["effective_temperature"]) * (
        0.75 * (tau + 2.0 / 3.0)
    ) ** 0.25
    column_mass = tau / 0.34

    if perturbation_seed is not None:
        generator = np.random.default_rng(int(perturbation_seed))

        def smooth(scale: float) -> np.ndarray:
            raw = generator.normal(size=tau.size + 12)
            kernel_x = np.arange(-6, 7, dtype=np.float64)
            kernel = np.exp(-0.5 * (kernel_x / 2.5) ** 2)
            kernel /= kernel.sum()
            values = np.convolve(raw, kernel, mode="valid")[: tau.size]
            values -= values.mean()
            values /= max(float(np.std(values)), 1.0e-12)
            return scale * values

        temperature *= np.exp(smooth(0.05))
        increments = np.diff(np.concatenate(([0.0], column_mass)))
        column_mass = np.cumsum(increments * np.exp(smooth(0.15)))

    gravity = 10.0 ** float(labels["log_surface_gravity"])
    gas_pressure = np.maximum(gravity * column_mass, 1.0e-20)
    electron_density = np.maximum(
        1.0e-4 * gas_pressure / (1.38054e-16 * temperature), 1.0e-20
    )
    rosseland_opacity = np.full_like(column_mass, 0.34)
    radiative_acceleration = np.zeros_like(column_mass)
    table = np.zeros((tau.size, 9), dtype=np.float64)
    table[:, 0] = column_mass
    table[:, 1] = temperature
    table[:, 2] = gas_pressure
    table[:, 3] = electron_density
    table[:, 4] = rosseland_opacity
    table[:, 5] = radiative_acceleration
    table[:, 6] = float(labels["microturbulence_km_s"]) * 1.0e5
    deck = format_warm_start_deck(
        effective_temperature=float(labels["effective_temperature"]),
        log_surface_gravity=float(labels["log_surface_gravity"]),
        metallicity=float(labels["metallicity"]),
        alpha_enhancement=float(labels["alpha_enhancement"]),
        layer_table=table,
        title=(
            "Payne Zero analytic Eddington-grey benchmark"
            if perturbation_seed is None
            else f"Payne Zero perturbed grey benchmark seed {perturbation_seed}"
        ),
    )
    return parse_atmosphere_deck(deck, source="<analytic-grey-benchmark>")


def interpolation_coordinates(labels: np.ndarray) -> np.ndarray:
    """Metric coordinates for donor search: log Teff plus the three abundances."""

    values = np.atleast_2d(np.asarray(labels, dtype=np.float64))
    coordinates = np.empty((values.shape[0], len(INTERPOLATION_LABELS)))
    coordinates[:, 0] = np.log10(np.maximum(values[:, 0], 1.0))
    coordinates[:, 1:] = values[:, 1 : len(INTERPOLATION_LABELS)]
    return coordinates


def inverse_temperature_interpolation_coordinates(labels: np.ndarray) -> np.ndarray:
    """Coordinates required by the complete-state benchmark: 5040/Teff plus labels."""

    values = np.atleast_2d(np.asarray(labels, dtype=np.float64))
    if values.shape[1] < len(INTERPOLATION_LABELS):
        raise ValueError("interpolation labels are missing required coordinates")
    coordinates = np.empty((values.shape[0], len(INTERPOLATION_LABELS)))
    if np.any(~np.isfinite(values[:, 0])) or np.any(values[:, 0] <= 0.0):
        raise ValueError("effective temperature must be finite and positive")
    coordinates[:, 0] = 5040.0 / values[:, 0]
    coordinates[:, 1:] = values[:, 1 : len(INTERPOLATION_LABELS)]
    return coordinates


def interpolated_reduced_state(
    label_dict: dict[str, float],
    donor_coordinates: np.ndarray,
    donor_scale: np.ndarray,
    donor_reduced: np.ndarray,
    *,
    neighbours: int = DEFAULT_INTERPOLATION_NEIGHBOURS,
    power: float = DEFAULT_INTERPOLATION_POWER,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inverse-distance interpolation of ``(m, T)`` over the nearest donors.

    Both fields are interpolated in the log, which keeps column mass positive
    and monotone whenever the donors are, and matches the coordinates the
    reduced-state emulator itself is trained in.
    """

    target = interpolation_coordinates(
        np.asarray([[label_dict[field] for field in INTERPOLATION_LABELS]])
    )[0]
    distance = np.sqrt(
        np.sum(((donor_coordinates - target) / donor_scale) ** 2, axis=1)
    )
    count = int(min(max(neighbours, 1), distance.size))
    nearest = np.argpartition(distance, count - 1)[:count]
    nearest = nearest[np.argsort(distance[nearest])]
    nearest_distance = distance[nearest]

    if nearest_distance[0] <= 1.0e-12:
        weights = np.zeros(count, dtype=np.float64)
        weights[0] = 1.0
    else:
        weights = 1.0 / nearest_distance**power
        weights /= weights.sum()

    log_mass = np.log10(np.maximum(donor_reduced[nearest, :, 0], 1.0e-300))
    log_temperature = np.log10(np.maximum(donor_reduced[nearest, :, 1], 1.0e-300))
    diagnostics = {
        "neighbours": count,
        "power": float(power),
        "nearest_distance": float(nearest_distance[0]),
        "farthest_used_distance": float(nearest_distance[-1]),
        "top_weight": float(weights[0]),
    }
    return (
        10.0 ** (weights @ log_mass),
        10.0 ** (weights @ log_temperature),
        diagnostics,
    )


def encode_interpolated_full_state(
    profiles: np.ndarray,
    effective_temperature: np.ndarray | float,
    tau: np.ndarray,
) -> np.ndarray:
    """Encode complete converged atmospheres in interpolation-safe coordinates.

    The first coordinate is ``log10(m)``; a positive weighted combination of
    monotone donor profiles remains monotone on the common depth grid. The
    temperature coordinate is relative to the analytic grey profile. The
    other positive fields are interpolated in log space and ``g_rad`` uses a
    bounded asinh coordinate around the same floor used by the diagnostics.
    """

    values = np.asarray(profiles, dtype=np.float64)
    if values.ndim not in (2, 3) or values.shape[-1] != 6:
        raise ValueError("profiles must have shape (80, 6) or (n, 80, 6)")
    single = values.ndim == 2
    if single:
        values = values[None, ...]
    if not np.all(np.isfinite(values)) or np.any(values[..., :5] <= 0.0):
        raise ValueError("complete atmosphere profiles must be finite and positive")
    if np.any(np.diff(values[..., 0], axis=1) <= 0.0):
        raise ValueError("complete atmosphere column mass must be strictly monotone")

    depth = np.asarray(tau, dtype=np.float64)
    if depth.shape != (values.shape[1],) or not np.all(np.isfinite(depth)):
        raise ValueError("tau must match the 80-layer profile depth grid")
    teff = np.asarray(effective_temperature, dtype=np.float64)
    if teff.ndim == 0:
        teff = np.full(values.shape[0], float(teff))
    teff = teff.reshape(-1)
    if teff.size != values.shape[0] or not np.all(np.isfinite(teff)) or np.any(teff <= 0.0):
        raise ValueError("effective_temperature must be finite and positive")

    grey_temperature = teff[:, None] * (0.75 * (depth[None, :] + 2.0 / 3.0)) ** 0.25
    encoded = np.empty_like(values)
    encoded[..., 0] = np.log10(values[..., 0])
    encoded[..., 1] = np.log10(values[..., 1] / grey_temperature)
    encoded[..., 2:5] = np.log10(values[..., 2:5])
    encoded[..., 5] = np.arcsinh(values[..., 5] / G_RAD_FLOOR)
    return encoded[0] if single else encoded


def decode_interpolated_full_state(
    encoded: np.ndarray,
    effective_temperature: np.ndarray | float,
    tau: np.ndarray,
) -> np.ndarray:
    """Decode a full-state interpolation coordinate into six physical fields."""

    coordinates = np.asarray(encoded, dtype=np.float64)
    if coordinates.ndim not in (2, 3) or coordinates.shape[-1] != 6:
        raise ValueError("encoded state must have shape (80, 6) or (n, 80, 6)")
    single = coordinates.ndim == 2
    if single:
        coordinates = coordinates[None, ...]
    depth = np.asarray(tau, dtype=np.float64)
    if depth.shape != (coordinates.shape[1],):
        raise ValueError("tau must match the encoded profile depth grid")
    teff = np.asarray(effective_temperature, dtype=np.float64)
    if teff.ndim == 0:
        teff = np.full(coordinates.shape[0], float(teff))
    teff = teff.reshape(-1)
    if teff.size != coordinates.shape[0] or not np.all(np.isfinite(teff)) or np.any(teff <= 0.0):
        raise ValueError("effective_temperature must be finite and positive")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("encoded state must be finite")

    decoded = np.empty_like(coordinates)
    decoded[..., 0] = 10.0 ** np.clip(coordinates[..., 0], -300.0, 300.0)
    grey_temperature = teff[:, None] * (0.75 * (depth[None, :] + 2.0 / 3.0)) ** 0.25
    decoded[..., 1] = grey_temperature * 10.0 ** np.clip(coordinates[..., 1], -30.0, 30.0)
    decoded[..., 2:5] = 10.0 ** np.clip(coordinates[..., 2:5], -300.0, 300.0)
    decoded[..., 5] = G_RAD_FLOOR * np.sinh(np.clip(coordinates[..., 5], -700.0, 700.0))
    if not np.all(np.isfinite(decoded)) or np.any(decoded[..., :5] <= 0.0):
        raise ValueError("decoded complete atmosphere is not finite and positive")
    if np.any(np.diff(decoded[..., 0], axis=1) <= 0.0):
        raise ValueError("decoded complete atmosphere has nonmonotone column mass")
    return decoded[0] if single else decoded


def interpolated_full_state(
    label_dict: dict[str, float],
    donor_coordinates: np.ndarray,
    donor_scale: np.ndarray,
    donor_encoded: np.ndarray,
    tau: np.ndarray,
    *,
    donor_indices: np.ndarray | None = None,
    neighbours: int = DEFAULT_INTERPOLATION_NEIGHBOURS,
    power: float = DEFAULT_INTERPOLATION_POWER,
) -> tuple[np.ndarray, dict]:
    """Inverse-distance interpolate all six atmosphere fields."""

    target = inverse_temperature_interpolation_coordinates(
        np.asarray([[label_dict[field] for field in INTERPOLATION_LABELS]])
    )[0]
    coordinates = np.asarray(donor_coordinates, dtype=np.float64)
    scale = np.asarray(donor_scale, dtype=np.float64)
    encoded = np.asarray(donor_encoded, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != len(INTERPOLATION_LABELS):
        raise ValueError("donor coordinates have the wrong shape")
    if encoded.ndim != 3 or encoded.shape[0] != coordinates.shape[0] or encoded.shape[2] != 6:
        raise ValueError("donor encoded profiles have the wrong shape")
    distance = np.sqrt(np.sum(((coordinates - target) / scale) ** 2, axis=1))
    count = int(min(max(neighbours, 1), distance.size))
    nearest = np.argpartition(distance, count - 1)[:count]
    nearest = nearest[np.argsort(distance[nearest])]
    nearest_distance = distance[nearest]
    if nearest_distance[0] <= 1.0e-12:
        weights = np.zeros(count, dtype=np.float64)
        weights[0] = 1.0
    else:
        weights = 1.0 / nearest_distance**power
        weights /= weights.sum()
    interpolated = np.sum(weights[:, None, None] * encoded[nearest], axis=0)
    profile = decode_interpolated_full_state(
        interpolated,
        float(label_dict["effective_temperature"]),
        tau,
    )
    diagnostics = {
        "neighbours": count,
        "power": float(power),
        "nearest_distance": float(nearest_distance[0]),
        "farthest_used_distance": float(nearest_distance[-1]),
        "top_weight": float(weights[0]),
        "donor_indices": (
            [int(value) for value in np.asarray(donor_indices)[nearest]]
            if donor_indices is not None
            else [int(value) for value in nearest]
        ),
        "donor_distances": [float(value) for value in nearest_distance],
        "weights": [float(value) for value in weights],
    }
    return profile, diagnostics


def full_state_to_atmosphere(
    profile: np.ndarray,
    label_dict: dict[str, float],
):
    """Pass an interpolated full state through the canonical deck parser."""

    from payne_zero_atmosphere.atmosphere_io import parse_atmosphere_deck
    from payne_zero_atmosphere.warm_start import format_warm_start_deck

    values = np.asarray(profile, dtype=np.float64)
    if values.shape != (80, 6):
        raise ValueError("interpolated full state must have shape (80, 6)")
    if not np.all(np.isfinite(values)) or np.any(values[:, :5] <= 0.0):
        raise ValueError("interpolated full state is not finite and positive")
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError("interpolated full state has nonmonotone column mass")
    table = np.zeros((80, 9), dtype=np.float64)
    table[:, :6] = values
    table[:, 6] = float(label_dict["microturbulence_km_s"]) * 1.0e5
    deck = format_warm_start_deck(
        effective_temperature=float(label_dict["effective_temperature"]),
        log_surface_gravity=float(label_dict["log_surface_gravity"]),
        metallicity=float(label_dict["metallicity"]),
        alpha_enhancement=float(label_dict["alpha_enhancement"]),
        layer_table=table,
        title="Payne Zero interpolated full-state benchmark",
    )
    return parse_atmosphere_deck(deck, source="<interpolated-full-state-benchmark>")


def load_donor_pool(corpus: Path, exclude: tuple[Path, ...]):
    """Corpus ``(m, T)`` profiles and metric coordinates, minus every scored set."""

    excluded: set[int] = set()
    used: list[dict] = []
    for path in exclude:
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        indices = [int(value) for value in manifest.get("star_indices", [])]
        excluded.update(indices)
        try:
            recorded = str(path.relative_to(REPO_ROOT))
        except ValueError:  # a manifest from outside the checkout is still legal
            recorded = str(path)
        used.append({"path": recorded, "star_count": len(indices)})

    # Memory discipline: every ``data[key]`` decompresses that whole array, and
    # the corpus profiles are ~200 MB. Touch the profiles exactly once, and drop
    # the four dependent fields in the same indexing pass rather than
    # materialising a full-width copy of the donor rows first.
    with np.load(corpus, allow_pickle=False) as data:
        profiles = data["atmosphere_profiles"]
        total = int(profiles.shape[0])
        keep = np.setdiff1d(
            np.arange(total, dtype=np.int64),
            np.fromiter(excluded, dtype=np.int64, count=len(excluded)),
        )
        reduced = np.ascontiguousarray(profiles[keep, :, :2], dtype=np.float64)
        del profiles
        raw_labels = data["labels_json"][keep]

    # Parse straight into the metric table; 50k label dicts kept alive at once
    # cost more than the profiles they describe.
    table = np.empty((raw_labels.size, len(INTERPOLATION_LABELS)), dtype=np.float64)
    for row, value in enumerate(raw_labels):
        entry = json.loads(str(value))
        table[row] = [entry[field] for field in INTERPOLATION_LABELS]
    del raw_labels
    coordinates = interpolation_coordinates(table)
    scale = np.maximum(coordinates.std(axis=0), 1.0e-12)
    provenance = {
        "corpus_star_count": total,
        "donor_star_count": int(keep.size),
        "excluded_star_count": int(total - keep.size),
        "excluded_manifests": used,
        "metric_labels": list(INTERPOLATION_LABELS),
        "metric_scale": [float(value) for value in scale],
    }
    return coordinates, scale, reduced, provenance, excluded


def load_full_donor_pool(
    corpus: Path,
    exclude: tuple[Path, ...],
):
    """Load the leakage-free corpus encoded for complete-state interpolation."""

    excluded: set[int] = set()
    used: list[dict] = []
    for path in exclude:
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        indices = [int(value) for value in manifest.get("star_indices", [])]
        excluded.update(indices)
        try:
            recorded = str(path.relative_to(REPO_ROOT))
        except ValueError:
            recorded = str(path)
        used.append({"path": recorded, "star_count": len(indices)})

    with np.load(corpus, allow_pickle=False) as data:
        profiles = data["atmosphere_profiles"]
        total = int(profiles.shape[0])
        keep = np.setdiff1d(
            np.arange(total, dtype=np.int64),
            np.fromiter(excluded, dtype=np.int64, count=len(excluded)),
        )
        selected_profiles = np.ascontiguousarray(profiles[keep], dtype=np.float64)
        tau = np.asarray(data["standard_rosseland_optical_depth"][0], dtype=np.float64)
        raw_labels = data["labels_json"][keep]

    table = np.empty((raw_labels.size, len(INTERPOLATION_LABELS)), dtype=np.float64)
    effective_temperature = np.empty(raw_labels.size, dtype=np.float64)
    for row, value in enumerate(raw_labels):
        entry = json.loads(str(value))
        table[row] = [entry[field] for field in INTERPOLATION_LABELS]
        effective_temperature[row] = float(entry["effective_temperature"])
    del raw_labels
    encoded = encode_interpolated_full_state(selected_profiles, effective_temperature, tau)
    del selected_profiles
    coordinates = inverse_temperature_interpolation_coordinates(table)
    scale = np.maximum(coordinates.std(axis=0), 1.0e-12)
    provenance = {
        "corpus_star_count": total,
        "donor_star_count": int(keep.size),
        "excluded_star_count": int(total - keep.size),
        "excluded_manifests": used,
        "metric_labels": list(INTERPOLATION_LABELS),
        "metric_scale": [float(value) for value in scale],
        "encoded_fields": [
            "log10_column_mass",
            "log10_temperature_over_grey",
            "log10_gas_pressure",
            "log10_electron_density",
            "log10_rosseland_opacity",
            "asinh_radiative_acceleration_over_floor",
        ],
        "radiative_acceleration_floor": G_RAD_FLOOR,
    }
    return coordinates, scale, encoded, tau, provenance, excluded, keep


def representative_positions(labels: np.ndarray, count: int = 12) -> list[int]:
    """Fixed maximin subset spanning the five-label panel."""

    values = np.asarray(labels, dtype=np.float64)
    scaled = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1.0e-12)
    centre = np.argmin(np.sum(scaled**2, axis=1))
    selected = [int(centre)]
    distance = np.sum((scaled - scaled[centre]) ** 2, axis=1)
    while len(selected) < min(int(count), len(values)):
        distance[selected] = -1.0
        next_index = int(np.argmax(distance))
        selected.append(next_index)
        distance = np.minimum(
            distance, np.sum((scaled - scaled[next_index]) ** 2, axis=1)
        )
    return selected


def _label_dict(entry: dict) -> dict[str, float]:
    return {field: float(entry[field]) for field in LABEL_FIELDS}


def _worker(payload):
    arm, profile, label_dict, prediction, perturbation_seed, options = payload
    from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere
    from experiments.reduced_state_emulator.restart import run_restart_trial

    started = time.perf_counter()
    try:
        if arm == "production_six_field":
            from payne_zero_atmosphere.warm_start import emulator_warm_start_model

            atmosphere, _deck = emulator_warm_start_model(device="cpu", **label_dict)
        elif arm in ("learned_reduced_state", "interpolated_grid"):
            # Both arms hand the same rematerialisation path a predicted (m, T);
            # only the predictor differs, which is the comparison we want.
            result = reconstruct_full_atmosphere(
                ReducedAtmosphere(
                    column_mass=prediction[0],
                    temperature=prediction[1],
                    labels=label_dict,
                ),
                n_synchronizations=None,
                max_synchronizations=8,
                pressure_tolerance_dex=1.0e-3,
            )
            atmosphere = result.atmosphere
        elif arm == "interpolated_full_state":
            atmosphere = full_state_to_atmosphere(prediction, label_dict)
        elif arm == "truth_reduced_state":
            result = reconstruct_full_atmosphere(
                ReducedAtmosphere(
                    column_mass=profile[:, 0],
                    temperature=profile[:, 1],
                    labels=label_dict,
                ),
                n_synchronizations=None,
                max_synchronizations=8,
                pressure_tolerance_dex=1.0e-3,
            )
            atmosphere = result.atmosphere
        elif arm in ("grey15", "grey30", "grey60", "grey_perturbed"):
            atmosphere = analytic_grey_atmosphere(
                label_dict, perturbation_seed=perturbation_seed
            )
        else:
            raise ValueError(f"unknown arm {arm}")
    except Exception as exc:  # one bad initializer is a result, not a batch abort
        elapsed = time.perf_counter() - started
        source = options.get("source")
        return StarRecord(
            labels=StellarLabels(**label_dict),
            seconds=elapsed,
            trials=[
                TrialRecord(
                    trial_index=0,
                    initializer_label={"source": source} if source else None,
                    iterations_completed=0,
                    converged=False,
                    seconds=elapsed,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            ],
        ).as_json()
    return run_restart_trial(
        StellarLabels(**label_dict), atmosphere, **options
    ).as_json()


def _load_inputs(corpus: Path, manifest: Path, prediction_path: Path):
    indices = np.asarray(json.loads(manifest.read_text())["star_indices"], dtype=np.int64)
    with np.load(corpus, allow_pickle=False) as data:
        labels_json = [json.loads(str(value)) for value in data["labels_json"][indices]]
        profiles = np.asarray(data["atmosphere_profiles"][indices], dtype=np.float64)
    labels = [_label_dict(entry) for entry in labels_json]
    predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if prediction_path.is_file():
        with np.load(prediction_path, allow_pickle=False) as data:
            for row, index in enumerate(np.asarray(data["star_indices"], dtype=np.int64)):
                predictions[int(index)] = (
                    np.asarray(data["column_mass"][row], dtype=np.float64),
                    np.asarray(data["temperature"][row], dtype=np.float64),
                )
    return indices, labels, profiles, predictions


def _record_key(record: dict) -> tuple[str, str | None]:
    source = None
    if record.get("trials"):
        label = record["trials"][0].get("initializer_label")
        source = label.get("source") if label else None
    return record["slug"], source


def _completed_records(path: Path) -> dict[tuple[str, str | None], dict]:
    return {_record_key(record): record for record in load_records(path)} if path.is_file() else {}


def _shutdown_pool(executor: ProcessPoolExecutor) -> None:
    """Shut down a worker pool without leaving solver processes behind.

    Each worker holds the line lists and opacity tables -- a couple of GB -- so
    an orphaned pool is not a tidiness problem, it is the next run's
    out-of-memory kill. ``shutdown`` alone waits for in-flight solves, which is
    exactly what we cannot afford when we are being torn down, so terminate
    whatever is still resident afterwards.
    """

    processes = list(getattr(executor, "_processes", {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=10.0)
        if process.is_alive():
            process.kill()


@contextmanager
def _terminate_pool_on_signal(executor: ProcessPoolExecutor | None):
    """Make SIGTERM/SIGINT run the pool teardown instead of orphaning workers."""

    if executor is None:
        yield
        return

    def handler(signum, _frame):
        _shutdown_pool(executor)
        raise KeyboardInterrupt(f"terminated by signal {signum}")

    previous = {
        number: signal.signal(number, handler)
        for number in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        yield
    finally:
        for number, original in previous.items():
            signal.signal(number, original)


def run_arm(
    arm: str,
    payloads: list[tuple],
    *,
    workers: int,
    records_path: Path,
) -> list[dict]:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_records(records_path)
    payload_keys = [
        (StellarLabels(**payload[2]).slug, payload[5]["source"]) for payload in payloads
    ]
    pending = [
        payload for payload, key in zip(payloads, payload_keys, strict=True)
        if key not in completed
    ]
    if workers <= 1:
        iterator = (_worker(payload) for payload in pending)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_worker, pending)
    try:
        with _terminate_pool_on_signal(executor), records_path.open("a") as handle:
            for number, record in enumerate(iterator, start=1):
                completed[_record_key(record)] = record
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(
                    f"[{arm} {number}/{len(pending)}] {record['slug']} "
                    f"converged={record['converged']} "
                    f"iters={record['converging_trial_iterations']} "
                    f"{record['seconds']:.1f}s",
                    flush=True,
                )
    finally:
        if executor is not None:
            _shutdown_pool(executor)
    return [completed[key] for key in payload_keys if key in completed]


def iteration_runtime_summary(records: list[dict]) -> dict:
    first = []
    later = []
    for record in records:
        for trial in record.get("trials", []):
            for timing in trial.get("diagnostics", {}).get("iteration_timings", []):
                value = timing.get("total_seconds")
                if value is None or not np.isfinite(float(value)):
                    continue
                (first if int(timing.get("iteration", 0)) == 1 else later).append(
                    float(value)
                )
    return {
        "first_iteration_mean_seconds": float(np.mean(first)) if first else None,
        "later_iteration_mean_seconds": float(np.mean(later)) if later else None,
        "first_iteration_count": len(first),
        "later_iteration_count": len(later),
    }


def has_finite_final_iteration(record: dict) -> bool:
    """Whether an unconverged solve still had a numeric state at its last step."""

    trials = record.get("trials", [])
    if not trials:
        return False
    timings = trials[0].get("diagnostics", {}).get("iteration_timings", [])
    if not timings:
        return False
    value = timings[-1].get("deep_layer_relative_temperature_change")
    return value is not None and np.isfinite(float(value))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=(
            "production_six_field",
            "learned_reduced_state",
            "truth_reduced_state",
            "interpolated_grid",
            "interpolated_full_state",
            "grey15",
            "grey30",
            "grey60",
            "grey_perturbed",
        ),
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--interpolation-neighbours",
        type=int,
        default=DEFAULT_INTERPOLATION_NEIGHBOURS,
    )
    parser.add_argument(
        "--interpolation-power", type=float, default=DEFAULT_INTERPOLATION_POWER
    )
    args = parser.parse_args(argv)

    indices, labels, profiles, predictions = _load_inputs(
        args.corpus, args.manifest, args.prediction
    )
    if args.count is not None:
        indices, labels, profiles = indices[: args.count], labels[: args.count], profiles[: args.count]
    positions = list(range(len(indices)))
    perturbation_seeds: list[int | None] = [None] * len(positions)
    iterations = {"grey30": 30, "grey60": 60}.get(args.arm, 15)
    extension_selection = None

    donor_provenance = None
    donor_diagnostics: dict[int, dict] = {}
    if args.arm == "learned_reduced_state":
        missing = [int(index) for index in indices if int(index) not in predictions]
        if missing:
            raise SystemExit(f"prediction is missing {len(missing)} requested stars")
    elif args.arm == "interpolated_grid":
        # Interpolate in the parent: the donor pool is ~50k x 80 x 2 and must not
        # be shipped to every worker process.
        coordinates, scale, reduced, donor_provenance, excluded = load_donor_pool(
            args.corpus, DEFAULT_DONOR_EXCLUDE
        )
        leaked = sorted(set(int(index) for index in indices) - excluded)
        if leaked:
            raise SystemExit(
                f"{len(leaked)} scored stars are still in the donor pool "
                f"(first: {leaked[:5]}); add this manifest to DEFAULT_DONOR_EXCLUDE"
            )
        for position, label in enumerate(labels):
            column_mass, temperature, diagnostics = interpolated_reduced_state(
                label,
                coordinates,
                scale,
                reduced,
                neighbours=args.interpolation_neighbours,
                power=args.interpolation_power,
            )
            predictions[int(indices[position])] = (column_mass, temperature)
            donor_diagnostics[int(indices[position])] = diagnostics
        print(
            f"[interpolated_grid] donor pool {donor_provenance['donor_star_count']} "
            f"stars; nearest-donor distance median "
            f"{np.median([d['nearest_distance'] for d in donor_diagnostics.values()]):.4f}",
            flush=True,
        )
        # Interpolation is finished, and every worker about to be spawned needs
        # its own copy of the solver. Release the donor pool before that happens
        # rather than holding a few hundred MB alongside N solver processes.
        del coordinates, scale, reduced
        gc.collect()
    elif args.arm == "interpolated_full_state":
        # Keep this pool separate from the older two-field interpolation arm:
        # it interpolates the complete six-field state in stable coordinates and
        # hands the decoded profile directly to the canonical deck parser.
        exclude = tuple(DEFAULT_DONOR_EXCLUDE) + (args.manifest,)
        (
            coordinates,
            scale,
            encoded,
            tau,
            donor_provenance,
            excluded,
            donor_indices,
        ) = load_full_donor_pool(args.corpus, exclude)
        leaked = sorted(set(int(index) for index in indices) - excluded)
        if leaked:
            raise SystemExit(
                f"{len(leaked)} scored stars are still in the full-state donor pool "
                f"(first: {leaked[:5]})"
            )
        for position, label in enumerate(labels):
            profile, diagnostics = interpolated_full_state(
                label,
                coordinates,
                scale,
                encoded,
                tau,
                donor_indices=donor_indices,
                neighbours=args.interpolation_neighbours,
                power=args.interpolation_power,
            )
            predictions[int(indices[position])] = profile
            donor_diagnostics[int(indices[position])] = diagnostics
        print(
            f"[interpolated_full_state] donor pool "
            f"{donor_provenance['donor_star_count']} stars; nearest-donor distance "
            f"median {np.median([d['nearest_distance'] for d in donor_diagnostics.values()]):.4f}",
            flush=True,
        )
        del coordinates, scale, encoded, tau, donor_indices
        gc.collect()
    if args.arm in ("grey30", "grey60"):
        parent_arm = "grey15" if args.arm == "grey30" else "grey30"
        parent_path = args.run_root / "records" / parent_arm / "records.jsonl"
        if not parent_path.is_file():
            raise SystemExit(f"{parent_arm} records are required before {args.arm}")
        parent_failed = [
            record for record in load_records(parent_path) if not record["converged"]
        ]
        eligible = parent_failed
        excluded_nonfinite = []
        if args.arm == "grey60":
            eligible = [record for record in parent_failed if has_finite_final_iteration(record)]
            excluded_nonfinite = [
                record for record in parent_failed if not has_finite_final_iteration(record)
            ]
        failed = {record["slug"] for record in eligible}
        positions = [
            p for p in positions if StellarLabels(**labels[p]).slug in failed
        ]
        extension_selection = {
            "parent_arm": parent_arm,
            "parent_failed_count": len(parent_failed),
            "eligible_finite_count": len(eligible),
            "excluded_nonfinite_count": len(excluded_nonfinite),
            "excluded_nonfinite_slugs": [record["slug"] for record in excluded_nonfinite],
        }
    elif args.arm == "grey_perturbed":
        table = np.asarray([[row[field] for field in LABEL_FIELDS] for row in labels])
        base_positions = representative_positions(table, count=12)
        positions = [p for p in base_positions for _ in PERTURBATION_SEEDS]
        perturbation_seeds = [seed for _ in base_positions for seed in PERTURBATION_SEEDS]

    payloads = []
    for row, position in enumerate(positions):
        label = labels[position]
        seed = perturbation_seeds[row] if args.arm == "grey_perturbed" else None
        source = args.arm if seed is None else f"grey_perturbed_{seed}"
        profile_dir = args.run_root / "profiles" / source
        product_dir = args.run_root / "products" / source
        options = {
            "source": source,
            "iterations_per_trial": iterations,
            "product_dir": str(product_dir),
            "atmosphere_profile_dir": str(profile_dir),
        }
        prediction = predictions.get(int(indices[position]), (None, None))
        payloads.append(
            (args.arm, profiles[position], label, prediction, seed, options)
        )

    records_path = args.run_root / "records" / args.arm / "records.jsonl"
    records = run_arm(args.arm, payloads, workers=args.workers, records_path=records_path)
    result = {
        "arm": args.arm,
        "star_indices": [int(indices[position]) for position in positions],
        "perturbation_seeds": (
            [int(seed) for seed in perturbation_seeds]
            if args.arm == "grey_perturbed"
            else None
        ),
        "iterations_per_trial": iterations,
        "summary": summarize(records) if records else None,
        "iteration_runtime": iteration_runtime_summary(records),
    }
    if extension_selection is not None:
        result["extension_selection"] = extension_selection
    if donor_provenance is not None:
        nearest = np.asarray(
            [entry["nearest_distance"] for entry in donor_diagnostics.values()]
        )
        result["donor_pool"] = donor_provenance
        result["interpolation"] = {
            "neighbours": args.interpolation_neighbours,
            "power": args.interpolation_power,
            "nearest_donor_distance": {
                "median": float(np.median(nearest)),
                "p95": float(np.percentile(nearest, 95)),
                "max": float(np.max(nearest)),
            },
            "per_star": {
                str(index): entry for index, entry in donor_diagnostics.items()
            },
        }
    args.result_root.mkdir(parents=True, exist_ok=True)
    out = args.result_root / f"convergence_{args.arm}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
