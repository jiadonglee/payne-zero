"""Star sampling shared by the reduced-state parity and depth-resolution experiments.

Both experiments need the same kind of sample from the truth corpus: mostly
IID, with a stratified slice oversampling the low-logg/metal-poor region
where Ting's reconstruction drift and the continuity harness's broken-seed
population both concentrate (``solver-in-the-loop-prior-work.md`` Sec 4.2,
``solver-in-the-loop-continuity.md`` Sec 5). The bounds match
``bench.labels.sample_hard_region`` for consistency with the existing
baseline's hard-region slice.
"""

from __future__ import annotations

import numpy as np

# Matches bench.labels.sample_hard_region's bounds.
HARD_LOGG_RANGE = (0.7, 2.8)
HARD_METALLICITY_RANGE = (-2.5, -0.5)


def sample_star_indices(
    labels: list[dict], count: int, seed: int, hard_fraction: float = 1.0 / 3.0
) -> np.ndarray:
    """Mostly IID plus a stratified hard-region slice (low logg, metal-poor)."""

    rng = np.random.default_rng(seed)
    logg = np.array([entry["log_surface_gravity"] for entry in labels])
    metallicity = np.array([entry["metallicity"] for entry in labels])
    hard_mask = (
        (logg >= HARD_LOGG_RANGE[0])
        & (logg <= HARD_LOGG_RANGE[1])
        & (metallicity >= HARD_METALLICITY_RANGE[0])
        & (metallicity <= HARD_METALLICITY_RANGE[1])
    )
    hard_indices = np.where(hard_mask)[0]
    n_hard = min(int(round(count * hard_fraction)), hard_indices.size)
    hard_pick = rng.choice(hard_indices, size=n_hard, replace=False)

    remaining = np.setdiff1d(np.arange(len(labels)), hard_pick)
    n_iid = count - n_hard
    iid_pick = rng.choice(remaining, size=n_iid, replace=False)

    picked = np.concatenate([iid_pick, hard_pick])
    rng.shuffle(picked)
    return picked
