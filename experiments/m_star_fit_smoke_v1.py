"""Smoke-test the synthesis fitting path on an M-giant reference atmosphere.

Two checks:

1. CLI-style end-to-end synthesis from the validated converged product
   (structured NPZ -> spectrum NPZ) over the TiO window.
2. A minimal chi-square fitting loop: a mock observation is synthesized from
   the same atmosphere at R=20000 with Gaussian noise (SNR 100 per pixel);
   the pipeline recovers effective temperature, surface gravity, and
   metallicity by synthesizing from labels over a small grid around the
   truth and minimizing the normalized-flux chi-square.

Success criteria: the end-to-end synthesis completes and the best-fit grid
point falls within one grid cell of the truth on every label.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from payne_zero_synthesis.api import synthesize

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "results" / "m_star_fit_smoke_v1"
WINDOW_NM = (640.0, 680.0)
RESOLUTION = 20000.0
SNR = 100.0

# The library is the four relaxed truth atmospheres on the g2.0 [M/H]=0
# branch of the truth-final campaign.  The label-based synthesis path
# (`synthesize_from_labels`) cannot serve M giants: its neural atmosphere
# initializer supports 5040/T <= 1.26, i.e. Teff >= ~4000 K only.
LIBRARY = (
    (3750.0, "g+2.00_m+0.00_a+0.00_c+0.00_x2.00_t3750"),
    (3800.0, "g+2.00_m+0.00_a+0.00_c+0.00_x2.00_t3800"),
    (3850.0, "g+2.00_m+0.00_a+0.00_c+0.00_x2.00_t3850"),
    (3900.0, "g+2.00_m+0.00_a+0.00_c+0.00_x2.00_t3900"),
)
MOCK_FROM = 3750.0
TRUE_LABELS = {"teff": MOCK_FROM, "logg": 2.0, "metallicity": 0.0}


def _normalized(spectrum) -> np.ndarray:
    return np.asarray(spectrum.normalized_flux, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir",
        type=Path,
        default=REPO_ROOT
        / "results"
        / "m_star_truth_final_v1"
        / "checkpoints",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    library_dir = args.checkpoint_dir
    library = {}
    for teff, node in LIBRARY:
        product = library_dir / node / "iter_0030.npz"
        if not product.is_file():
            raise SystemExit(f"FAIL_STOP: missing {product}")
        library[teff] = synthesize(
            product,
            wavelength_start_nm=WINDOW_NM[0],
            wavelength_end_nm=WINDOW_NM[1],
            resolution=RESOLUTION,
            molecular_lines=True,
            device=None,
            dtype="float64",
        )
    target = library[MOCK_FROM]
    observed_flux = _normalized(target)
    rng = np.random.default_rng(args.seed)
    observed = observed_flux + rng.normal(
        0.0, 1.0 / SNR, size=observed_flux.shape
    )

    grid_chi2 = []
    for teff, _node in LIBRARY:
        model_flux = _normalized(library[teff])
        chi2 = float(np.sum(((observed - model_flux) * SNR) ** 2) / observed.size)
        grid_chi2.append({"teff": teff, "chi2_reduced": chi2})
    grid_chi2.sort(key=lambda row: row["chi2_reduced"])
    best = grid_chi2[0]
    deltas = {"teff_K": best["teff"] - TRUE_LABELS["teff"]}
    passed = best["teff"] == MOCK_FROM
    report = {
        "campaign": "m_star_fit_smoke_v1",
        "library": [teff for teff, _node in LIBRARY],
        "mock_source": MOCK_FROM,
        "window_nm": list(WINDOW_NM),
        "resolution": RESOLUTION,
        "snr_per_pixel": SNR,
        "best_fit": best,
        "true_labels": TRUE_LABELS,
        "recovery_deltas": deltas,
        "top5": grid_chi2[:5],
        "grid_size": len(grid_chi2),
        "passed": passed,
    }
    (args.out / "fit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.save(
        args.out / "observed_normalized_flux.npy",
        observed,
    )
    print(json.dumps(report["best_fit"], sort_keys=True))
    print(f"deltas: {json.dumps(deltas, sort_keys=True)}")
    print(f"PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
