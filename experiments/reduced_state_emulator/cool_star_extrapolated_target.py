"""Test a predictor-corrector start for a direct 4000 -> 3500 K jump.

The predictor uses two already converged states, at 4000 and 3750 K, and
extrapolates the positive atmosphere fields one more 250 K step in log space.
The exact Payne-Zero solver is then run at 3500 K.  This is an initializer
experiment only: it does not alter the solver, its physics, or its convergence
criterion.

For the reduced route only ``(m,T)`` is extrapolated; the other four fields are
rebuilt with the existing exact reconstruction path before the solve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from payne_zero_atmosphere.atmosphere_io import ModelAtmosphere

from .cool_star_step_test import (
    TrackSpec,
    _atmosphere_quality,
    _clone_atmosphere,
    _production_atmosphere,
    _reconstruct_from_mt,
    _solve_attempt,
)
from .cool_star_targeted_3700 import _model_from_product


def _positive_log_extrapolation(
    anchor: np.ndarray, source: np.ndarray, *, step_ratio: float = 1.0
) -> np.ndarray:
    """Extrapolate a positive profile by one source-to-anchor log step."""

    left = np.asarray(anchor, dtype=np.float64)
    right = np.asarray(source, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("predictor profiles must be one-dimensional and aligned")
    if (
        np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
        or np.any(left <= 0.0)
        or np.any(right <= 0.0)
    ):
        raise ValueError("predictor profiles must be finite and positive")
    prediction = np.exp(np.log(right) + float(step_ratio) * (np.log(right) - np.log(left)))
    if np.any(~np.isfinite(prediction)) or np.any(prediction <= 0.0):
        raise ValueError("log-space extrapolation produced an invalid profile")
    return prediction


def _predict_full_state(
    anchor: ModelAtmosphere,
    source: ModelAtmosphere,
    target_template: ModelAtmosphere,
) -> ModelAtmosphere:
    """Predict a complete positive warm start and attach target metadata."""

    predicted = _clone_atmosphere(source)
    for field in (
        "column_mass",
        "temperature",
        "gas_pressure",
        "electron_density",
        "rosseland_opacity",
        "radiative_acceleration",
    ):
        setattr(
            predicted,
            field,
            _positive_log_extrapolation(
                getattr(anchor, field), getattr(source, field)
            ),
        )
    # The product loader uses positive placeholders for fields that are not
    # serialized.  Keep the source auxiliary fields and only retarget labels.
    predicted.metadata = dict(target_template.metadata)
    predicted.fixed_column_abundance_values = dict(
        target_template.fixed_column_abundance_values
    )
    quality = _atmosphere_quality(predicted)
    if not quality["valid"]:
        raise ValueError(f"predicted full state failed quality gate: {quality}")
    return predicted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-product", type=Path, required=True)
    parser.add_argument("--source-product", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("full_predictor", "reduced_predictor"), required=True
    )
    parser.add_argument("--anchor-temperature", type=float, default=4000.0)
    parser.add_argument("--source-temperature", type=float, default=3750.0)
    parser.add_argument("--target-temperature", type=float, default=3500.0)
    parser.add_argument("--logg", type=float, default=5.0)
    parser.add_argument("--metallicity", type=float, default=0.0)
    parser.add_argument("--iteration-cap", type=int, default=30)
    args = parser.parse_args()

    track = TrackSpec(log_surface_gravity=args.logg, metallicity=args.metallicity)
    anchor_labels = track.labels(args.anchor_temperature)
    source_labels = track.labels(args.source_temperature)
    target_labels = track.labels(args.target_temperature)
    anchor = _model_from_product(args.anchor_product, anchor_labels)
    source = _model_from_product(args.source_product, source_labels)
    target_template = _production_atmosphere(target_labels)

    if args.mode == "full_predictor":
        seed = _predict_full_state(anchor, source, target_template)
    else:
        mass = _positive_log_extrapolation(
            anchor.column_mass, source.column_mass
        )
        temperature = _positive_log_extrapolation(
            anchor.temperature, source.temperature
        )
        seed = _reconstruct_from_mt(target_labels, mass, temperature)

    record, _state = _solve_attempt(
        track=track,
        method=f"{args.mode}_4000_to_3500",
        schedule="predictor_corrector_250K_extrapolation",
        source_temperature=args.anchor_temperature,
        target_labels=target_labels,
        initial_atmosphere=seed,
        product_dir=args.out_root / "products" / args.mode,
        iteration_cap=args.iteration_cap,
    )
    record["predictor"] = {
        "anchor_temperature": args.anchor_temperature,
        "source_temperature": args.source_temperature,
        "target_temperature": args.target_temperature,
        "step_ratio": 1.0,
        "mode": args.mode,
        "anchor_product": str(args.anchor_product),
        "source_product": str(args.source_product),
    }
    output = {
        "format": "payne_zero_cool_star_extrapolated_target_v1",
        "track": track.as_json(),
        "record": record,
    }
    output_path = args.out_root / f"{args.mode}_record.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    print(f"wrote {output_path}")
    return 0 if record.get("survives_solver", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
