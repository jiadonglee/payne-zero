"""Data and fitting primitives for analytic initializer discovery.

The fitting stage is intentionally not a neural emulator.  It decomposes the
converged profiles into physically normalized residuals and asks how many
smooth depth modes and low-order label terms are needed.  The output is a
diagnostic about formula complexity; it is not a production initializer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)

TARGET_FIELDS = (
    "log_temperature_over_grey",
    "log_column_mass_over_grey",
    "log_rosseland_opacity",
)

DEFAULT_CORPUS = Path(
    "source_data_files/atmosphere_emulator/five_label/strict_truth_52199.npz"
)


@dataclass(frozen=True)
class Corpus:
    """The minimal five-label corpus view used by the discovery stage."""

    path: Path
    slugs: np.ndarray
    labels: np.ndarray
    tau: np.ndarray
    temperature: np.ndarray
    column_mass: np.ndarray
    gas_pressure: np.ndarray
    electron_density: np.ndarray
    rosseland_opacity: np.ndarray

    @property
    def size(self) -> int:
        return int(self.labels.shape[0])

    @property
    def layers(self) -> int:
        return int(self.tau.size)


@dataclass(frozen=True)
class Split:
    """Deterministic, manifest-aware fit/validation split."""

    train: np.ndarray
    validation: np.ndarray
    excluded: np.ndarray
    seed: int


def _parse_label_rows(payload: np.ndarray) -> np.ndarray:
    rows = []
    for raw in payload:
        record = json.loads(str(raw))
        rows.append([float(record[field]) for field in LABEL_FIELDS])
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABEL_FIELDS):
        raise ValueError("labels_json did not decode to the five-label schema")
    return values


def load_strict_truth(path: Path | str = DEFAULT_CORPUS) -> Corpus:
    """Load only the arrays needed for the analytic discovery stage."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        target_fields = tuple(str(value) for value in data["target_fields"])
        expected = (
            "column_mass",
            "temperature",
            "gas_pressure",
            "electron_density",
            "rosseland_opacity",
            "radiative_acceleration",
        )
        if target_fields != expected:
            raise ValueError(
                "unexpected atmosphere field order: " f"{target_fields!r}"
            )
        tau_by_star = np.asarray(data["standard_rosseland_optical_depth"], dtype=np.float64)
        if tau_by_star.ndim != 2 or not np.allclose(tau_by_star, tau_by_star[0]):
            raise ValueError("the discovery corpus must use one common tau grid")
        profiles = np.asarray(data["atmosphere_profiles"], dtype=np.float64)
        if profiles.ndim != 3 or profiles.shape[2] != 6:
            raise ValueError("atmosphere_profiles must have shape (N, 80, 6)")
        labels = _parse_label_rows(data["labels_json"])
        slugs = np.asarray(data["slugs"], dtype=str)
        verified = np.asarray(data["depth_grid_verified"])

    if profiles.shape[0] != labels.shape[0] or profiles.shape[0] != slugs.size:
        raise ValueError("corpus arrays have inconsistent row counts")
    if not np.all(verified):
        raise ValueError("unverified depth-grid rows are not allowed")
    if np.any(~np.isfinite(profiles)) or np.any(profiles[:, :, :5] <= 0.0):
        raise ValueError("corpus contains non-finite or non-positive fields")
    if np.any(np.diff(profiles[:, :, 0], axis=1) <= 0.0):
        raise ValueError("corpus contains non-monotone column-mass profiles")

    return Corpus(
        path=source,
        slugs=slugs,
        labels=labels,
        tau=np.asarray(tau_by_star[0], dtype=np.float64),
        temperature=np.asarray(profiles[:, :, 1], dtype=np.float64),
        column_mass=np.asarray(profiles[:, :, 0], dtype=np.float64),
        gas_pressure=np.asarray(profiles[:, :, 2], dtype=np.float64),
        electron_density=np.asarray(profiles[:, :, 3], dtype=np.float64),
        rosseland_opacity=np.asarray(profiles[:, :, 4], dtype=np.float64),
    )


def _walk_manifest_indices(value: object, key: str | None = None) -> Iterable[int]:
    """Yield row indices from known manifest fields without reading products."""

    index_keys = {
        "star_indices",
        "indices",
        "solver_star_indices",
        "spectral_star_indices",
    }
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            yield from _walk_manifest_indices(child_value, str(child_key))
        return
    if key not in index_keys or not isinstance(value, (list, tuple)):
        return
    for item in value:
        if isinstance(item, (int, np.integer)):
            yield int(item)


def collect_excluded_indices(
    manifest_paths: Sequence[Path | str], *, corpus_size: int
) -> tuple[np.ndarray, list[str]]:
    """Collect the union of known evaluation rows from metadata manifests."""

    values: set[int] = set()
    used: list[str] = []
    for raw_path in manifest_paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        found = {index for index in _walk_manifest_indices(payload) if 0 <= index < corpus_size}
        if found:
            values.update(found)
            used.append(str(path))
    return np.asarray(sorted(values), dtype=np.int64), used


def make_split(
    corpus_size: int,
    *,
    excluded: Sequence[int] = (),
    validation_fraction: float = 0.2,
    seed: int = 20260816,
) -> Split:
    """Make a reproducible split after removing evaluation rows."""

    if corpus_size < 2:
        raise ValueError("corpus_size must be at least two")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    excluded_array = np.unique(np.asarray(excluded, dtype=np.int64))
    if np.any(excluded_array < 0) or np.any(excluded_array >= corpus_size):
        raise ValueError("excluded indices are outside the corpus")
    available = np.setdiff1d(np.arange(corpus_size, dtype=np.int64), excluded_array)
    if available.size < 2:
        raise ValueError("not enough rows remain after exclusions")
    generator = np.random.default_rng(int(seed))
    shuffled = generator.permutation(available)
    validation_count = max(1, int(round(validation_fraction * shuffled.size)))
    validation = np.sort(shuffled[:validation_count])
    train = np.sort(shuffled[validation_count:])
    return Split(train=train, validation=validation, excluded=excluded_array, seed=int(seed))


def label_features(labels: np.ndarray) -> np.ndarray:
    """Return dimensionless, smooth label coordinates for formula discovery."""

    values = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABEL_FIELDS):
        raise ValueError("labels must have shape (N, 5)")
    if np.any(~np.isfinite(values)) or np.any(values[:, 0] <= 0.0) or np.any(values[:, 4] <= 0.0):
        raise ValueError("labels must be finite with positive Teff and microturbulence")
    return np.column_stack(
        (
            5040.0 / values[:, 0],
            values[:, 1],
            values[:, 2],
            values[:, 3],
            np.log10(values[:, 4]),
        )
    )


def grey_temperature(effective_temperature: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Eddington-grey temperature on the production Rosseland grid."""

    return np.asarray(effective_temperature, dtype=np.float64)[:, None] * (
        0.75 * (np.asarray(tau, dtype=np.float64)[None, :] + 2.0 / 3.0)
    ) ** 0.25


def normalized_targets(corpus: Corpus) -> dict[str, np.ndarray]:
    """Construct stable residual targets rather than raw six-field outputs."""

    grey = grey_temperature(corpus.labels[:, 0], corpus.tau)
    tau_mass = np.maximum(corpus.tau[None, :] / 0.34, 1.0e-300)
    return {
        "log_temperature_over_grey": np.log10(corpus.temperature / grey),
        "log_column_mass_over_grey": np.log10(corpus.column_mass) - np.log10(tau_mass),
        "log_rosseland_opacity": np.log10(corpus.rosseland_opacity),
    }


def polynomial_exponents(feature_count: int, degree: int) -> np.ndarray:
    """Return all monomial exponent vectors through ``degree``."""

    if feature_count < 1 or degree < 0:
        raise ValueError("feature_count must be positive and degree non-negative")
    rows: list[tuple[int, ...]] = []
    for total_degree in range(degree + 1):
        for combination in itertools.combinations_with_replacement(
            range(feature_count), total_degree
        ):
            exponents = [0] * feature_count
            for index in combination:
                exponents[index] += 1
            rows.append(tuple(exponents))
    return np.asarray(rows, dtype=np.int64)


def polynomial_features(
    features: np.ndarray, exponents: np.ndarray, *, center: np.ndarray | None = None, scale: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a low-order monomial library after fit-set standardization."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("features must be two-dimensional")
    if center is None:
        center = values.mean(axis=0)
    if scale is None:
        scale = values.std(axis=0)
    center = np.asarray(center, dtype=np.float64)
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1.0e-12)
    normalized = (values - center) / scale
    terms = []
    for exponent in np.asarray(exponents, dtype=np.int64):
        term = np.ones(normalized.shape[0], dtype=np.float64)
        for feature_index, power in enumerate(exponent):
            if power:
                term *= normalized[:, feature_index] ** int(power)
        terms.append(term)
    return np.column_stack(terms), center, scale


def _r2(y_true: np.ndarray, y_pred: np.ndarray, center: np.ndarray) -> float:
    denominator = float(np.sum((y_true - center) ** 2))
    if denominator <= 0.0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def fit_low_rank_surrogate(
    target: np.ndarray,
    labels: np.ndarray,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    *,
    components: int = 8,
    degree: int = 3,
    ridge: float = 1.0e-8,
) -> dict[str, object]:
    """Fit a diagnostic low-rank polynomial surrogate and score held-out rows."""

    values = np.asarray(target, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("target must have shape (N, layers)")
    train = np.asarray(train_indices, dtype=np.int64)
    validation = np.asarray(validation_indices, dtype=np.int64)
    if train.size == 0 or validation.size == 0:
        raise ValueError("both train and validation splits must be non-empty")
    if np.intersect1d(train, validation).size:
        raise ValueError("train and validation splits overlap")
    if np.any(train >= values.shape[0]) or np.any(validation >= values.shape[0]):
        raise ValueError("split indices exceed target rows")

    x = label_features(labels)
    exponents = polynomial_exponents(x.shape[1], int(degree))
    phi_train, feature_center, feature_scale = polynomial_features(x[train], exponents)
    phi_validation, _, _ = polynomial_features(
        x[validation], exponents, center=feature_center, scale=feature_scale
    )

    target_center = values[train].mean(axis=0)
    centered_train = values[train] - target_center
    _, singular_values, right_vectors = np.linalg.svd(
        centered_train, full_matrices=False
    )
    component_count = min(int(components), right_vectors.shape[0], values.shape[1])
    if component_count < 1:
        raise ValueError("components must be positive")
    basis = right_vectors[:component_count]
    coefficient_target = centered_train @ basis.T

    gram = phi_train.T @ phi_train
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty,
        phi_train.T @ coefficient_target,
    )
    prediction = target_center + (phi_validation @ coefficients) @ basis
    truth = values[validation]
    error = prediction - truth
    explained = np.cumsum(singular_values**2) / np.sum(singular_values**2)
    layer_p95 = np.percentile(np.abs(error), 95.0, axis=0)
    return {
        "components": int(component_count),
        "degree": int(degree),
        "term_count": int(exponents.shape[0]),
        "ridge": float(ridge),
        "r2": _r2(truth, prediction, target_center),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "absolute_error_p95": float(np.percentile(np.abs(error), 95.0)),
        "absolute_error_max": float(np.max(np.abs(error))),
        "layer_error_p95_median": float(np.median(layer_p95)),
        "layer_error_p95_max": float(np.max(layer_p95)),
        "low_rank_explained_variance": float(explained[component_count - 1]),
    }


def file_sha256(path: Path | str) -> str:
    """Hash a source artifact for result provenance."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
