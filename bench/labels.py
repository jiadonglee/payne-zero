"""Stellar-label sets for reference-solver benchmarking.

Support bounds are read from the released five-label initializer checkpoint
rather than hardcoded, so a checkpoint refresh cannot silently desynchronize the
benchmark from the model's actual domain. The checkpoint stores effective
temperature as the ratio ``5040 K / Teff`` (see
``payne_zero_atmosphere/warm_start.py:719``), so that coordinate is inverted
here to give a temperature in kelvin.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from payne_zero_atmosphere.data_files import atmosphere_emulator_dir


LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)


@dataclass(frozen=True)
class StellarLabels:
    """One five-label request, in the solver's public coordinates."""

    effective_temperature: float
    log_surface_gravity: float
    metallicity: float
    alpha_enhancement: float
    microturbulence_km_s: float

    def as_kwargs(self) -> dict[str, float]:
        return asdict(self)

    @property
    def slug(self) -> str:
        return (
            f"t{self.effective_temperature:07.1f}"
            f"_g{self.log_surface_gravity:+05.2f}"
            f"_m{self.metallicity:+05.2f}"
            f"_a{self.alpha_enhancement:+05.2f}"
            f"_x{self.microturbulence_km_s:04.2f}"
        )


def support_bounds(checkpoint_path: Path | None = None) -> dict[str, tuple[float, float]]:
    """Return five-label support in public coordinates, from the checkpoint."""

    import torch

    path = checkpoint_path or (atmosphere_emulator_dir() / "five_label" / "checkpoint.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint["labels"]["bounds"]

    ratio_low, ratio_high = (float(v) for v in raw["temperature_ratio_5040_k_over_temperature"])
    # The ratio is inversely proportional to temperature, so the bounds swap.
    return {
        "effective_temperature": (5040.0 / ratio_high, 5040.0 / ratio_low),
        "log_surface_gravity": tuple(float(v) for v in raw["log10_surface_gravity_cgs"]),
        "metallicity": tuple(float(v) for v in raw["metallicity"]),
        "alpha_enhancement": tuple(float(v) for v in raw["alpha_enhancement"]),
        "microturbulence_km_s": tuple(float(v) for v in raw["microturbulence_km_s"]),
    }


def sample_uniform(
    count: int,
    *,
    seed: int = 0,
    margin: float = 1.0e-3,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> list[StellarLabels]:
    """Sample labels uniformly inside the support box.

    ``margin`` insets each edge by a fraction of its width so that sampling
    never lands on the boundary, where the initializer clips and warns.

    This covers the box, not the stellar locus: it deliberately includes
    combinations such as hot dwarfs at low gravity that no real star occupies.
    That is intended for a first characterization of *where the solver
    struggles*, which is the point of the baseline. Use a physically
    distributed catalog for the IID benchmark slice.
    """

    if count < 1:
        raise ValueError("count must be positive")
    box = bounds or support_bounds()
    generator = np.random.default_rng(seed)
    draws = generator.random((count, len(LABEL_FIELDS)))

    columns = []
    for index, field in enumerate(LABEL_FIELDS):
        low, high = box[field]
        inset = margin * (high - low)
        columns.append(low + inset + draws[:, index] * (high - low - 2.0 * inset))
    stacked = np.stack(columns, axis=1)
    return [StellarLabels(*(float(v) for v in row)) for row in stacked]


def sample_iid_from_corpus(
    count: int,
    *,
    seed: int = 0,
    corpus_path: Path | None = None,
) -> list[StellarLabels]:
    """Resample labels from the five-label training corpus.

    The corpus (`strict_truth_52199.npz`, the v1.3 training-data release) is
    rank-coupled to the H–R diagram — it is the distribution the initializer
    was actually trained on, so it defines the IID benchmark slice: the
    production solve measured on the population the initializer is for.  Rows
    are drawn with replacement; every field is taken from the same corpus row
    to preserve the joint distribution.
    """

    import json as _json

    path = corpus_path or (
        atmosphere_emulator_dir() / "five_label" / "strict_truth_52199.npz"
    )
    with np.load(path, allow_pickle=False) as data:
        labels_json = data["labels_json"]
        generator = np.random.default_rng(seed)
        indices = generator.integers(0, len(labels_json), size=count)
        rows = [_json.loads(str(labels_json[int(i)])) for i in indices]
    return [
        StellarLabels(**{field: float(row[field]) for field in LABEL_FIELDS})
        for row in rows
    ]


def sample_boundary(
    count: int,
    *,
    seed: int = 0,
    band: float = 0.05,
    margin: float = 1.0e-3,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> list[StellarLabels]:
    """Sample with at least one coordinate pinned near a support-box face.

    Each draw picks how many coordinates to pin (geometric: mostly one), which
    ones, and which side, then places them uniformly within ``band`` of that
    face; the rest are uniform over the box as in `sample_uniform`.  This is
    the boundary slice: it concentrates on the edges where the initializer
    extrapolates worst, without sampling only corners.
    """

    if count < 1:
        raise ValueError("count must be positive")
    box = bounds or support_bounds()
    generator = np.random.default_rng(seed)
    labels = sample_uniform(count, seed=seed + 1, margin=margin, bounds=box)
    table = np.array([[getattr(item, field) for field in LABEL_FIELDS] for item in labels])

    for row in table:
        # Mostly one pinned coordinate, occasionally two, rarely three.
        pin_count = 1 + int(generator.random() < 0.25) + int(generator.random() < 0.0625)
        pin_fields = generator.choice(len(LABEL_FIELDS), size=pin_count, replace=False)
        for field_index in pin_fields:
            low, high = box[LABEL_FIELDS[int(field_index)]]
            width = high - low
            inset = margin * width
            edge_band = band * width
            if generator.random() < 0.5:
                row[int(field_index)] = low + inset + generator.random() * edge_band
            else:
                row[int(field_index)] = high - inset - generator.random() * edge_band
    return [StellarLabels(*(float(v) for v in row)) for row in table]


def sample_hard_region(
    count: int,
    *,
    seed: int = 0,
    logg_range: tuple[float, float] = (0.7, 2.8),
    metallicity_range: tuple[float, float] = (-2.5, -0.5),
    margin: float = 1.0e-3,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> list[StellarLabels]:
    """Sample the low-gravity, metal-poor region where the tail lives.

    The Stage 0 report (§4.2) found the ≥30-iteration tail concentrated at
    mean logg 2.38 and mean [M/H] −1.19, versus 3.11 and −1.03 overall.  This
    slice samples uniformly inside that sub-box (logg and [M/H] ranges above,
    the other three coordinates uniform over the support box) to measure the
    hard region with statistical power instead of catching it incidentally.
    """

    if count < 1:
        raise ValueError("count must be positive")
    box = bounds or support_bounds()
    box = dict(box)
    box["log_surface_gravity"] = logg_range
    box["metallicity"] = metallicity_range
    return sample_uniform(count, seed=seed, margin=margin, bounds=box)


def load(path: Path | str) -> list[StellarLabels]:
    """Read labels from a JSON list or JSON Lines file."""

    text = Path(path).read_text()
    stripped = text.lstrip()
    if stripped.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [StellarLabels(**{field: float(r[field]) for field in LABEL_FIELDS}) for r in records]


def save(labels: list[StellarLabels], path: Path | str) -> Path:
    """Write labels as JSON Lines."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as handle:
        for item in labels:
            handle.write(json.dumps(item.as_kwargs()) + "\n")
    return destination
