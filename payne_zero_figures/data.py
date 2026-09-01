"""Artifact loading shared by every figure in the repository.

Nothing here computes physics.  Every function reads something already written
to ``results/`` or ``runs/`` by a harness, so a figure can never disagree with
the tables it sits next to.

One caution carried over from the scripts this module replaces.  There were two
``_load_records`` implementations with *opposite* duplicate handling: the
evidence figures rely on last-wins, because ``run_many_restarts`` appends and
the file legitimately holds reruns, while the convergence-geometry figure
rejects duplicate slugs as a corrupt-input check.  Merging them would have
silently changed one or the other, so both survive here under separate names --
:func:`load_records` and :func:`load_records_strict`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
RUNS = REPO / "runs"
EMULATOR_RUNS = RUNS / "reduced_state_emulator"
ARTIFACTS = REPO / "artifacts" / "reduced_state_emulator"
FIGURES = REPO / "figures"


# --------------------------------------------------------------------------
# Generic readers
# --------------------------------------------------------------------------


def sha256(path: Path) -> str:
    """Streaming digest, for the provenance manifests the reports emit."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=np.float64) for key in data.files}


# --------------------------------------------------------------------------
# Solver records
# --------------------------------------------------------------------------


def load_records(path: Path) -> dict:
    """Slug-keyed records, last occurrence winning.

    ``run_many_restarts`` appends, so the file holds reruns and the last entry
    for a slug is the current one.
    """

    rows = [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]
    return {row["slug"]: row for row in rows}


def load_records_strict(path: Path) -> dict[str, dict]:
    """Slug-keyed records, rejecting duplicates and empty files.

    Used where a repeated slug means the inputs are wrong rather than rerun.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing benchmark records: {path}; pull the open benchmark results first"
        )
    rows: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        slug = str(row["slug"])
        if slug in rows:
            raise ValueError(f"duplicate slug in {path}: {slug}")
        rows[slug] = row
    if not rows:
        raise ValueError(f"benchmark records are empty: {path}")
    return rows


def iteration_counts(path: Path) -> dict[str, int]:
    """Converging-trial iteration count per star, converged stars only."""

    return {slug: row["converging_trial_iterations"]
            for slug, row in load_records(path).items() if row["converged"]}


def residual_traces(path: Path) -> list[np.ndarray]:
    """Per-iteration deep-layer temperature residual, one array per star.

    This is the quantity the solver's stopping criterion reads, so it is the
    only trace that shows contraction as the solver actually measures it.
    """

    traces = []
    for row in load_records(path).values():
        if not row["converged"]:
            continue
        timings = row["trials"][-1]["diagnostics"]["iteration_timings"]
        traces.append(np.array(
            [t["deep_layer_relative_temperature_change"] for t in timings]))
    return traces


# --------------------------------------------------------------------------
# Spectra
# --------------------------------------------------------------------------

SPECTRA = EMULATOR_RUNS / "spectra"


def arm_spectrum(arm: str, slug: str, *, root: Path = SPECTRA) -> dict:
    """One converged spectrum: wavelength and normalized flux."""

    with np.load(root / arm / f"{slug}.npz") as data:
        return {key: data[key] for key in ("wavelength_nm", "normalized_flux")}


def metric_trace(
    slug: str,
    field: str,
    *,
    spectra_root: Path = SPECTRA,
    learned_root: Path | None = None,
    production_root: Path | None = None,
) -> tuple:
    """Per-wavelength |difference| in the units the named metric is defined in."""

    spectra = {}
    roots = {
        "learned": (
            learned_root
            if learned_root is not None
            else spectra_root / "learned_reduced_state"
        ),
        "production": (
            production_root
            if production_root is not None
            else spectra_root / "production_six_field"
        ),
    }
    for arm, root in roots.items():
        with np.load(root / f"{slug}.npz") as data:
            spectra[arm] = {k: data[k] for k in
                            ("wavelength_nm", "normalized_flux", "flux_total",
                             "flux_continuum")}
    wavelength = spectra["learned"]["wavelength_nm"]
    if field == "normalized_flux":
        delta = np.abs(spectra["learned"]["normalized_flux"]
                       - spectra["production"]["normalized_flux"])
    elif field == "flux_total":
        delta = np.abs(spectra["learned"]["flux_total"]
                       - spectra["production"]["flux_total"]) / np.maximum(
            np.abs(spectra["production"]["flux_continuum"]), 1e-300)
    else:
        delta = np.abs(spectra["learned"]["flux_continuum"]
                       - spectra["production"]["flux_continuum"]) / np.maximum(
            np.abs(spectra["production"]["flux_continuum"]), 1e-300)
    return wavelength, delta


def binned_max(wavelength: np.ndarray, delta: np.ndarray, bins: int = 220) -> tuple:
    """Max per wavelength bin.

    Plotting all 16,219 points as a line renders a solid block -- the ink says
    "everything is at the top" when almost all of it is two decades lower.  The
    metric is a maximum, so the max-per-bin envelope is both readable and the
    quantity actually being gated; nothing that matters is hidden by it.
    """

    edges = np.linspace(wavelength[0], wavelength[-1], bins + 1)
    index = np.clip(np.digitize(wavelength, edges) - 1, 0, bins - 1)
    peak = np.full(bins, np.nan)
    for b in range(bins):
        mask = index == b
        if mask.any():
            peak[b] = delta[mask].max()
    return 0.5 * (edges[:-1] + edges[1:]), peak
