"""Pure geometry and candidate-selection helpers for initializer trajectories.

The production atmosphere solver is intentionally not imported here. Keeping
the distance, direction, PCA, and selection rules independent makes them cheap
to test and prevents the figure code from accidentally changing the solver.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


ARM_NAMES = (
    "learned_reduced_state",
    "production_six_field",
    "interpolated_full_state",
)


def state_vector(temperature: np.ndarray, gas_pressure: np.ndarray) -> np.ndarray:
    """Return the common 160-dimensional (log10 T, log10 P) state."""

    temperature = np.asarray(temperature, dtype=np.float64)
    gas_pressure = np.asarray(gas_pressure, dtype=np.float64)
    if temperature.shape != gas_pressure.shape:
        raise ValueError("temperature and gas_pressure must have the same shape")
    if temperature.ndim != 1 or temperature.size < 2:
        raise ValueError("temperature and gas_pressure must be one-dimensional")
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("temperature must be finite and positive")
    if not np.all(np.isfinite(gas_pressure)) or np.any(gas_pressure <= 0.0):
        raise ValueError("gas_pressure must be finite and positive")
    return np.concatenate((np.log10(temperature), np.log10(gas_pressure)))


def split_state(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a common state vector into log-temperature and log-pressure."""

    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or vector.size % 2:
        raise ValueError("state vector must be one-dimensional with an even size")
    half = vector.size // 2
    if not np.all(np.isfinite(vector)):
        raise ValueError("state vector must be finite")
    return vector[:half], vector[half:]


def state_distance(vector: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Measure field-balanced RMS distances in dex."""

    vector = np.asarray(vector, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if vector.shape != reference.shape:
        raise ValueError("state and reference must have the same shape")
    temperature, pressure = split_state(vector - reference)
    temperature_rms = float(np.sqrt(np.mean(temperature**2)))
    pressure_rms = float(np.sqrt(np.mean(pressure**2)))
    combined = float(np.sqrt(0.5 * (temperature_rms**2 + pressure_rms**2)))
    return {
        "temperature_rms_dex": temperature_rms,
        "pressure_rms_dex": pressure_rms,
        "combined_rms_dex": combined,
    }


def direction_cosine(
    previous: np.ndarray,
    following: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Cosine between the solver step and the direct direction to reference."""

    previous = np.asarray(previous, dtype=np.float64)
    following = np.asarray(following, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if previous.shape != following.shape or previous.shape != reference.shape:
        raise ValueError("all states must have the same shape")
    step = following - previous
    target = reference - previous
    step_norm = float(np.linalg.norm(step))
    target_norm = float(np.linalg.norm(target))
    if step_norm == 0.0 or target_norm == 0.0:
        return float("nan")
    return float(np.dot(step, target) / (step_norm * target_norm))


def progress_fraction(
    previous: np.ndarray,
    following: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Fractional reduction in distance to reference after one solver step."""

    before = float(np.linalg.norm(np.asarray(previous) - np.asarray(reference)))
    after = float(np.linalg.norm(np.asarray(following) - np.asarray(reference)))
    if before == 0.0:
        return float("nan")
    return float((before - after) / before)


def record_iterations(record: Mapping) -> int | None:
    """Read the first successful trial's iteration count from a benchmark row."""

    value = record.get("converging_trial_iterations")
    if value is None:
        return None
    value = int(value)
    return value if value > 0 else None


def _required_float(row: Mapping, key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"candidate metric {key!r} is not finite")
    return value


def make_candidate_row(
    slug: str,
    *,
    records: Mapping[str, Mapping[str, Mapping]],
    initial_states: Mapping[str, Mapping[str, np.ndarray]],
    first_states: Mapping[str, Mapping[str, np.ndarray]],
    reference: np.ndarray,
    category: str | None = None,
) -> dict:
    """Build all auditable metrics and gates for one star."""

    required_arms = ARM_NAMES
    missing = [
        arm
        for arm in required_arms
        if slug not in records.get(arm, {}) or slug not in initial_states.get(arm, {})
    ]
    if missing:
        raise KeyError(f"missing {slug} data for arms: {missing}")

    iterations = {
        arm: record_iterations(records[arm][slug]) for arm in required_arms
    }
    converged = {
        arm: bool(records[arm][slug].get("converged")) for arm in required_arms
    }
    distances = {
        arm: state_distance(initial_states[arm][slug], reference)
        for arm in required_arms
    }
    first_alignment = {}
    first_progress = {}
    for arm in ("learned_reduced_state", "production_six_field"):
        if slug not in first_states.get(arm, {}):
            first_alignment[arm] = float("nan")
            first_progress[arm] = float("nan")
            continue
        first_alignment[arm] = direction_cosine(
            initial_states[arm][slug], first_states[arm][slug], reference
        )
        first_progress[arm] = progress_fraction(
            initial_states[arm][slug], first_states[arm][slug], reference
        )

    n_two = iterations["learned_reduced_state"]
    n_six = iterations["production_six_field"]
    gap = None if n_two is None or n_six is None else n_six - n_two
    d_two = distances["learned_reduced_state"]
    d_six = distances["production_six_field"]
    alignment_advantage = (
        first_alignment["learned_reduced_state"]
        - first_alignment["production_six_field"]
    )
    distance_advantage = float(
        np.sqrt(
            d_two["combined_rms_dex"]
            / max(d_six["combined_rms_dex"], 1.0e-300)
        )
    )
    gates = {
        "all_three_converged": all(converged.values()),
        "two_field_at_most_four_iterations": n_two is not None and n_two <= 4,
        "six_field_at_least_four_iterations_slower": gap is not None and gap >= 4,
        "six_field_closer_in_temperature": (
            d_six["temperature_rms_dex"] < d_two["temperature_rms_dex"]
        ),
        "six_field_closer_in_pressure": (
            d_six["pressure_rms_dex"] < d_two["pressure_rms_dex"]
        ),
        "two_field_first_step_more_aligned": (
            np.isfinite(alignment_advantage) and alignment_advantage > 0.0
        ),
        "two_field_first_step_moves_toward_reference": (
            np.isfinite(first_progress["learned_reduced_state"])
            and first_progress["learned_reduced_state"] > 0.0
        ),
    }
    return {
        "slug": slug,
        "category": category,
        "converged": converged,
        "iterations": iterations,
        "iteration_gap_six_minus_two": gap,
        "distances": distances,
        "first_step_alignment": first_alignment,
        "first_step_progress": first_progress,
        "alignment_advantage_two_minus_six": float(alignment_advantage),
        "distance_advantage_two_over_six": distance_advantage,
        "gates": gates,
        "eligible": bool(all(gates.values())),
    }


def select_candidate(rows: Iterable[Mapping]) -> dict:
    """Select the strongest eligible case with a deterministic lexicographic rank."""

    eligible = [dict(row) for row in rows if bool(row.get("eligible"))]
    if not eligible:
        raise ValueError("no candidate satisfies the convergence-geometry gates")

    def key(row: Mapping):
        gap = _required_float(row, "iteration_gap_six_minus_two")
        alignment = _required_float(row, "alignment_advantage_two_minus_six")
        distance = _required_float(row, "distance_advantage_two_over_six")
        return (gap, alignment, distance, str(row["slug"]))

    return max(eligible, key=key)


def pca_2d(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project finite state vectors to two principal components.

    Returns coordinates, explained-variance fractions, and the two component
    row vectors. The caller may include the reference state in states so that
    the plotted origin is part of the fitted geometry.
    """

    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("states must be a two-dimensional matrix with >=2 rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("states must be finite")
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    components = components[:2]
    coordinates = centered @ components.T
    variance = singular_values**2
    total = float(variance.sum())
    explained = variance[:2] / total if total > 0.0 else np.zeros(2)
    return coordinates, explained, components
