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
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from payne_zero_synthesis.api import synthesize, synthesize_from_labels

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRODUCT = (
    REPO_ROOT
    / "results"
    / "m_star_emulator_mgiant_v3"
    / "spectral_stage"
    / "products"
    / "production_six_field"
    / "t03750.0_g+2.00_m+0.00_a+0.00_x2.00.npz"
)
DEFAULT_OUT = REPO_ROOT / "results" / "m_star_fit_smoke_v1"
WINDOW_NM = (640.0, 680.0)
RESOLUTION = 20000.0
SNR = 100.0

TEFF_GRID = (3650.0, 3700.0, 3750.0, 3800.0, 3850.0)
LOGG_GRID = (1.75, 2.0, 2.25)
METALLICITY_GRID = (-0.25, 0.0, 0.25)
TRUE_LABELS = {"teff": 3750.0, "logg": 2.0, "metallicity": 0.0}


def _normalized(spectrum) -> np.ndarray:
    return np.asarray(spectrum.normalized_flux, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", type=Path, default=DEFAULT_PRODUCT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args(argv)
    if not args.product.is_file():
        raise SystemExit(f"FAIL_STOP: missing {args.product}")
    args.out.mkdir(parents=True, exist_ok=True)

    target = synthesize(
        args.product,
        wavelength_start_nm=WINDOW_NM[0],
        wavelength_end_nm=WINDOW_NM[1],
        resolution=RESOLUTION,
        molecular_lines=True,
        device=None,
        dtype="float64",
    )
    observed_flux = _normalized(target)
    rng = np.random.default_rng(args.seed)
    observed = observed_flux + rng.normal(
        0.0, 1.0 / SNR, size=observed_flux.shape
    )

    grid_chi2 = []
    best = None
    for teff, logg, metallicity in itertools.product(
        TEFF_GRID, LOGG_GRID, METALLICITY_GRID
    ):
        model = synthesize_from_labels(
            effective_temperature=teff,
            log_surface_gravity=logg,
            metallicity=metallicity,
            microturbulence_km_s=2.0,
            wavelength_start_nm=WINDOW_NM[0],
            wavelength_end_nm=WINDOW_NM[1],
            resolution=RESOLUTION,
            molecular_lines=True,
            device=None,
            dtype="float64",
        )
        model_flux = _normalized(model)
        chi2 = float(
            np.sum(
                ((observed - model_flux) * SNR) ** 2
            )
            / observed.size
        )
        entry = {
            "teff": teff,
            "logg": logg,
            "metallicity": metallicity,
            "chi2_reduced": chi2,
        }
        grid_chi2.append(entry)
        if best is None or chi2 < best["chi2_reduced"]:
            best = entry
    grid_chi2.sort(key=lambda row: row["chi2_reduced"])

    deltas = {
        "teff_K": best["teff"] - TRUE_LABELS["teff"],
        "logg_dex": best["logg"] - TRUE_LABELS["logg"],
        "metallicity_dex": best["metallicity"] - TRUE_LABELS["metallicity"],
    }
    passed = (
        abs(deltas["teff_K"]) <= 50.0
        and abs(deltas["logg_dex"]) <= 0.25
        and abs(deltas["metallicity_dex"]) <= 0.25
    )
    report = {
        "campaign": "m_star_fit_smoke_v1",
        "product": str(args.product),
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
