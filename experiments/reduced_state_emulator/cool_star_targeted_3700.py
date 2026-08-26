"""Targeted 3750 -> 3700 K continuation diagnostic for the cool-star test.

This runner exists because the first pilot accidentally included 3750 K in
the nominal 100 K continuation.  It resumes from the completed 3750 K
continuation products and asks the actual missing question: can the next
100 K step reach 3700 K?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bench.labels import StellarLabels
from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere
from payne_zero_synthesis.atmosphere import load_atmosphere_npz

from .cool_star_step_test import (
    TrackSpec,
    _production_atmosphere,
    _reconstruct_from_mt,
    _retarget_full_state,
    _solve_attempt,
)


def _model_from_product(path: Path, labels: StellarLabels) -> ModelAtmosphere:
    product = load_atmosphere_npz(path)
    zeros = np.zeros_like(product["temperature"])
    positive = np.maximum(zeros + 1.0, 1.0e-300)
    metadata = {
        "effective_temperature": f"{labels.effective_temperature:.6f}",
        "log_surface_gravity": f"{labels.log_surface_gravity:.6f}",
        "metallicity": f"{labels.metallicity:.6f}",
        "alpha_enhancement": f"{labels.alpha_enhancement:.6f}",
        "microturbulence_km_s": f"{labels.microturbulence_km_s:.6f}",
        "surface_radiation_pressure_line": "PRADK 5.0000E-01",
    }
    return ModelAtmosphere(
        column_mass=np.asarray(product["column_mass"], dtype=np.float64),
        temperature=np.asarray(product["temperature"], dtype=np.float64),
        gas_pressure=np.asarray(product["gas_pressure"], dtype=np.float64),
        electron_density=np.asarray(product["electron_density"], dtype=np.float64),
        rosseland_opacity=positive,
        radiative_acceleration=np.maximum(zeros + 1.0e-30, 1.0e-300),
        microturbulence=np.asarray(product["microturbulence"], dtype=np.float64),
        convective_flux=np.zeros_like(zeros),
        convective_velocity=np.zeros_like(zeros),
        metadata=metadata,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-product", type=Path, required=True)
    parser.add_argument("--mode", choices=("full_carry", "reduced_rematerialized"), required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--source-temperature", type=float, default=3750.0)
    parser.add_argument("--target-temperature", type=float, default=3700.0)
    parser.add_argument("--iteration-cap", type=int, default=30)
    args = parser.parse_args()

    track = TrackSpec(log_surface_gravity=5.0, metallicity=0.0)
    target_labels = track.labels(args.target_temperature)
    source_labels = track.labels(args.source_temperature)
    source = _model_from_product(args.source_product, source_labels)
    template = _production_atmosphere(target_labels)
    if args.mode == "full_carry":
        seed = _retarget_full_state(source, template)
    else:
        seed = _reconstruct_from_mt(
            target_labels, source.column_mass, source.temperature
        )
    record, _state = _solve_attempt(
        track=track,
        method=f"targeted_100_{args.mode}",
        schedule="targeted_100K",
        source_temperature=args.source_temperature,
        target_labels=target_labels,
        initial_atmosphere=seed,
        product_dir=args.out_root / "products" / args.mode,
        iteration_cap=args.iteration_cap,
    )
    out_path = args.out_root / f"{args.mode}_record.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
