"""Compose two standalone physical ``(m,T)`` predictions by label region.

This is a reproducible model-selection step, not a solver correction.  The
base prediction is used everywhere except an explicit label-defined region,
where the override prediction is used.  Both inputs must refer to the same
sealed index manifest and carry the same truth arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compose(
    base: dict[str, np.ndarray],
    override: dict[str, np.ndarray],
    *,
    temperature_min: float | None = None,
    logg_max: float | None = None,
    metallicity_max: float | None = None,
    alpha_min: float | None = None,
    alpha_max: float | None = None,
    vmic_min: float | None = None,
    vmic_max: float | None = None,
    solver_tail_region: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.array_equal(base["star_indices"], override["star_indices"]):
        raise ValueError("base and override star indices differ")
    if not np.array_equal(base["labels"], override["labels"]):
        raise ValueError("base and override labels differ")
    for key in ("truth_column_mass", "truth_temperature"):
        if not np.array_equal(base[key], override[key]):
            raise ValueError(f"base and override {key} differ")
    bounds = (temperature_min, logg_max, metallicity_max, alpha_min, alpha_max, vmic_min, vmic_max)
    if solver_tail_region and any(value is not None for value in bounds):
        raise SystemExit("--solver-tail-region cannot be combined with label-box flags")
    labels = base["labels"]
    if solver_tail_region:
        mask = (labels[:, 1] < 3.2) | (labels[:, 4] > 3.0) | (labels[:, 3] > 0.4)
    else:
        mask = np.ones(len(labels), dtype=bool)
        clauses = ((0, temperature_min, np.greater), (1, logg_max, np.less),
                   (2, metallicity_max, np.less), (3, alpha_min, np.greater),
                   (3, alpha_max, np.less), (4, vmic_min, np.greater),
                   (4, vmic_max, np.less))
        for column, value, operator in clauses:
            if value is not None:
                mask &= operator(labels[:, column], float(value))
    mass = np.array(base["column_mass"], copy=True)
    temperature = np.array(base["temperature"], copy=True)
    mass[mask] = override["column_mass"][mask]
    temperature[mask] = override["temperature"][mask]
    return mass, temperature, mask


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-prediction", type=Path, required=True)
    parser.add_argument("--override-prediction", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature-min", type=float)
    parser.add_argument("--logg-max", type=float)
    parser.add_argument("--metallicity-max", type=float)
    parser.add_argument("--alpha-min", type=float)
    parser.add_argument("--alpha-max", type=float)
    parser.add_argument("--vmic-min", type=float)
    parser.add_argument("--vmic-max", type=float)
    parser.add_argument("--solver-tail-region", action="store_true")
    args = parser.parse_args(argv)
    box_values = (args.temperature_min, args.logg_max, args.metallicity_max,
                  args.alpha_min, args.alpha_max, args.vmic_min, args.vmic_max)
    if not args.solver_tail_region and all(value is None for value in box_values):
        args.temperature_min, args.logg_max, args.metallicity_max = 6000.0, 2.5, -1.5

    with np.load(args.base_prediction, allow_pickle=False) as data:
        base = {key: np.asarray(data[key]) for key in data.files}
    with np.load(args.override_prediction, allow_pickle=False) as data:
        override = {key: np.asarray(data[key]) for key in data.files}
    mass, temperature, mask = compose(
        base,
        override,
        temperature_min=args.temperature_min,
        logg_max=args.logg_max,
        metallicity_max=args.metallicity_max,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        vmic_min=args.vmic_min,
        vmic_max=args.vmic_max,
        solver_tail_region=args.solver_tail_region,
    )
    violations = int(np.any(np.diff(mass, axis=1) <= 0.0, axis=1).sum())
    if violations:
        raise SystemExit(f"composed prediction is not strictly monotonic: {violations}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        star_indices=base["star_indices"],
        labels=base["labels"],
        column_mass=mass,
        temperature=temperature,
        truth_column_mass=base["truth_column_mass"],
        truth_temperature=base["truth_temperature"],
        base_column_mass=base["column_mass"],
        base_temperature=base["temperature"],
        override_column_mass=override["column_mass"],
        override_temperature=override["temperature"],
        override_mask=mask,
    )
    provenance = {
        "base_prediction": str(args.base_prediction),
        "base_prediction_sha256": _sha256(args.base_prediction),
        "override_prediction": str(args.override_prediction),
        "override_prediction_sha256": _sha256(args.override_prediction),
        "prediction": str(args.out),
        "prediction_sha256": _sha256(args.out),
        "ensemble_policy": "base_with_label_defined_regional_override",
        "region": {
            "solver_tail_region": args.solver_tail_region,
            "temperature_min": args.temperature_min,
            "logg_max": args.logg_max,
            "metallicity_max": args.metallicity_max,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "vmic_min": args.vmic_min,
            "vmic_max": args.vmic_max,
            "selected_count": int(np.sum(mask)),
        },
        "star_count": int(len(mask)),
        "monotonicity_violations": violations,
    }
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.out} (regional_override={int(np.sum(mask))})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
