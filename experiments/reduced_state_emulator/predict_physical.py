"""Materialize physical-coordinate reduced-state predictions.

This remains a separate process from the solver workers. It can combine several
seed checkpoints by taking the layer-wise median of their physical ``m`` and
``T`` predictions; the median of monotonic profiles is monotonic as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reduced_state.emulator import (
    load_corpus,
    load_physical_checkpoint,
    predict_physical_state,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
DEFAULT_INDICES = REPO_ROOT / "results" / "reconstruction_metrics.json"
DEFAULT_CHECKPOINT_DIR = (
    REPO_ROOT / "artifacts" / "reduced_state_emulator" / "physical"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _indices(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    return np.asarray(payload["star_indices"], dtype=np.int64)


def combine_physical_predictions(
    masses: list[np.ndarray],
    temperatures: list[np.ndarray],
    labels: np.ndarray,
    seeds: tuple[int, ...],
    *,
    regional_seed: int | None = None,
    regional_temperature_min: float | None = None,
    regional_logg_max: float | None = None,
    regional_metallicity_max: float | None = None,
    alpha_min: float | None = None,
    alpha_max: float | None = None,
    vmic_min: float | None = None,
    vmic_max: float | None = None,
    solver_tail_region: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine seed predictions, with an optional validation-selected region arm.

    The default remains the coordinate-wise median.  A regional arm is only
    enabled explicitly and is intended for a label-defined extrapolation tail
    selected using the unsealed validation rows, never by inspecting the
    sealed development or audit stars.
    """

    if not masses or len(masses) != len(temperatures) or len(masses) != len(seeds):
        raise ValueError("mass, temperature, and seed predictions must align")
    predicted_mass = np.median(np.stack(masses, axis=0), axis=0)
    predicted_temperature = np.median(np.stack(temperatures, axis=0), axis=0)
    regional_mask = np.zeros(len(labels), dtype=bool)
    if regional_seed is not None:
        if regional_seed not in seeds:
            raise ValueError("regional_seed must be one of seeds")
        bounds = (regional_temperature_min, regional_logg_max,
                  regional_metallicity_max, alpha_min, alpha_max, vmic_min, vmic_max)
        if solver_tail_region and any(value is not None for value in bounds):
            raise SystemExit("solver_tail_region cannot be combined with label-box flags")
        if solver_tail_region:
            regional_mask = ((labels[:, 1] < 3.2) | (labels[:, 4] > 3.0)
                             | (labels[:, 3] > 0.4))
        else:
            if all(value is None for value in bounds):
                regional_temperature_min, regional_logg_max, regional_metallicity_max = (
                    6000.0, 2.5, -1.5
                )
            regional_mask = np.ones(len(labels), dtype=bool)
            for column, value, operator in ((0, regional_temperature_min, np.greater),
                                             (1, regional_logg_max, np.less),
                                             (2, regional_metallicity_max, np.less),
                                             (3, alpha_min, np.greater),
                                             (3, alpha_max, np.less),
                                             (4, vmic_min, np.greater),
                                             (4, vmic_max, np.less)):
                if value is not None:
                    regional_mask &= operator(labels[:, column], float(value))
        arm = seeds.index(regional_seed)
        predicted_mass[regional_mask] = masses[arm][regional_mask]
        predicted_temperature[regional_mask] = temperatures[arm][regional_mask]
    return predicted_mass, predicted_temperature, regional_mask


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--indices-from", type=Path, default=DEFAULT_INDICES)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--checkpoint-template",
        default="checkpoint_physical_seed{seed}.pt",
        help="checkpoint filename template relative to --checkpoint-dir",
    )
    parser.add_argument("--seeds", default="20260807,20260808,20260809")
    parser.add_argument(
        "--regional-seed",
        type=int,
        default=None,
        help=(
            "explicitly use this validation-selected seed in the hot, metal-poor, "
            "low-gravity label tail; default is the median everywhere"
        ),
    )
    parser.add_argument("--regional-temperature-min", type=float)
    parser.add_argument("--regional-logg-max", type=float)
    parser.add_argument("--regional-metallicity-max", type=float)
    parser.add_argument("--alpha-min", type=float)
    parser.add_argument("--alpha-max", type=float)
    parser.add_argument("--vmic-min", type=float)
    parser.add_argument("--vmic-max", type=float)
    parser.add_argument("--solver-tail-region", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR / "predicted_physical_ensemble.npz",
    )
    args = parser.parse_args(argv)
    regional_bounds = (args.regional_temperature_min, args.regional_logg_max,
                       args.regional_metallicity_max, args.alpha_min, args.alpha_max,
                       args.vmic_min, args.vmic_max)
    if args.regional_seed is not None and not args.solver_tail_region and all(
        value is None for value in regional_bounds
    ):
        args.regional_temperature_min, args.regional_logg_max, args.regional_metallicity_max = (
            6000.0, 2.5, -1.5
        )

    indices = _indices(args.indices_from)
    corpus = load_corpus(args.corpus)
    labels = corpus["labels"][indices]
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")

    masses = []
    temperatures = []
    checkpoint_provenance = []
    for seed in seeds:
        checkpoint = args.checkpoint_dir / args.checkpoint_template.format(seed=seed)
        model, standardization, meta = load_physical_checkpoint(checkpoint)
        excluded = set(meta.get("development_indices", [])) | set(
            meta.get("sealed_audit_indices", [])
        )
        for manifest in meta.get("additional_excluded_manifests", []):
            if not isinstance(manifest, dict):
                raise ValueError(
                    f"checkpoint {checkpoint} has an invalid additional exclusion"
                )
            excluded.update(int(index) for index in manifest.get("star_indices", []))
        missing = [int(index) for index in indices if int(index) not in excluded]
        if missing:
            raise SystemExit(
                f"checkpoint {checkpoint} did not exclude {len(missing)} requested "
                "evaluation stars"
            )
        mass, temperature = predict_physical_state(model, standardization, labels)
        masses.append(mass)
        temperatures.append(temperature)
        checkpoint_provenance.append(
            {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "seed": seed,
            }
        )

    predicted_mass, predicted_temperature, regional_mask = combine_physical_predictions(
        masses,
        temperatures,
        labels,
        seeds,
        regional_seed=args.regional_seed,
        regional_temperature_min=args.regional_temperature_min,
        regional_logg_max=args.regional_logg_max,
        regional_metallicity_max=args.regional_metallicity_max,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        vmic_min=args.vmic_min,
        vmic_max=args.vmic_max,
        solver_tail_region=args.solver_tail_region,
    )
    violations = int(np.any(np.diff(predicted_mass, axis=1) <= 0.0, axis=1).sum())
    if violations:
        raise SystemExit(f"ensemble produced {violations} non-monotonic profiles")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        star_indices=indices,
        labels=labels,
        column_mass=predicted_mass,
        temperature=predicted_temperature,
        truth_column_mass=corpus["column_mass"][indices],
        truth_temperature=corpus["temperature"][indices],
        seed_column_mass=np.stack(masses, axis=0),
        seed_temperature=np.stack(temperatures, axis=0),
    )
    provenance = {
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "indices_manifest": str(args.indices_from),
        "indices_manifest_sha256": _sha256(args.indices_from),
        "prediction": str(args.out),
        "prediction_sha256": _sha256(args.out),
        "checkpoints": checkpoint_provenance,
        "star_count": int(len(indices)),
        "seeds": list(seeds),
        "ensemble_policy": (
            "median"
            if args.regional_seed is None
            else "median_with_validation_selected_regional_seed"
        ),
        "regional_selection": {
            "seed": args.regional_seed,
            "temperature_min": args.regional_temperature_min,
            "logg_max": args.regional_logg_max,
            "metallicity_max": args.regional_metallicity_max,
            "solver_tail_region": args.solver_tail_region,
            "alpha_min": args.alpha_min,
            "alpha_max": args.alpha_max,
            "vmic_min": args.vmic_min,
            "vmic_max": args.vmic_max,
            "selected_count": int(np.sum(regional_mask)),
        },
    }
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"wrote {args.out} ({len(indices)} stars, seeds={seeds}, "
        f"regional_override={int(np.sum(regional_mask))}, "
        f"monotonicity_violations={violations})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
