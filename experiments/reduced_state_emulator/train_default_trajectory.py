"""Train a standalone reduced-state model calibrated to the default trajectory.

The shipped six-field initializer is used only to create a training target.
At inference time this model is still just a labels -> (m,T) network, so the
solver no longer depends on the six-field checkpoint. The default target keeps
the geometric midpoint only in the low-gravity region; narrower metal-poor or
hot metal-poor low-gravity regions are useful when a teacher correction helps
one tail but hurts another.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from reduced_state.emulator import load_corpus, save_physical_checkpoint
from experiments.reduced_state_emulator.production_teacher import (
    predict_production_mT_batch,
)
from experiments.reduced_state_emulator.train_physical import (
    DEFAULT_AUDIT,
    DEFAULT_CORPUS,
    DEFAULT_DEVELOPMENT,
    HARD_REGION_DEFAULT_WEIGHT,
    LOSS_VERSION,
    TAIL_LOSS_WEIGHT,
    _evaluate,
    _load_indices,
    _sha256,
    _train_one,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = (
    REPO_ROOT / "artifacts" / "reduced_state_emulator" / "physical_default_trajectory_cpu"
)
DEFAULT_TARGET_OUT = REPO_ROOT / "results" / "default_trajectory_target.npz"


def _eligible_indices(
    corpus: dict,
    development: np.ndarray,
    audit: np.ndarray,
    additional_excluded: tuple[np.ndarray, ...] = (),
) -> np.ndarray:
    excluded = set(int(value) for value in development) | set(int(value) for value in audit)
    for values in additional_excluded:
        excluded.update(int(value) for value in values)
    return np.asarray(
        [
            index
            for index, role in enumerate(corpus["roles"])
            if role in {"train", "validation"} and int(index) not in excluded
        ],
        dtype=np.int64,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--development-from", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="additional manifests whose star indices must stay out of fitting",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-out", type=Path, default=DEFAULT_TARGET_OUT)
    parser.add_argument("--seeds", default="20260807,20260808,20260809")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teacher-batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hard-weight", type=float, default=HARD_REGION_DEFAULT_WEIGHT)
    parser.add_argument(
        "--hard-region",
        choices=("hot_metal_poor", "low_gravity", "solver_tail"),
        default="solver_tail",
    )
    parser.add_argument("--tail-weight", type=float, default=TAIL_LOSS_WEIGHT)
    parser.add_argument(
        "--surface-weight",
        type=float,
        default=0.0,
        help="extra mass-profile loss weight on the first eight layers",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--logg-threshold", type=float, default=3.2)
    parser.add_argument(
        "--teacher-region",
        choices=(
            "low_gravity",
            "metal_poor_low_gravity",
            "hot_metal_poor_low_gravity",
        ),
        default="low_gravity",
    )
    parser.add_argument("--metallicity-threshold", type=float, default=-1.2)
    parser.add_argument(
        "--temperature-threshold",
        type=float,
        default=6000.0,
        help="minimum Teff for the hot_metal_poor_low_gravity teacher region",
    )
    args = parser.parse_args(argv)

    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be between 0 and 1")
    if not np.isfinite(args.surface_weight) or args.surface_weight < 0.0:
        raise SystemExit("--surface-weight must be finite and >= 0")
    if args.teacher_batch_size < 1:
        raise SystemExit("--teacher-batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")

    corpus = load_corpus(args.corpus)
    development = _load_indices(args.development_from)
    audit = _load_indices(args.audit_manifest)
    additional_excluded = []
    additional_excluded_arrays = []
    for manifest in args.exclude_manifest:
        values = _load_indices(manifest)
        additional_excluded_arrays.append(values)
        additional_excluded.append(
            {
                "path": str(manifest),
                "sha256": _sha256(manifest),
                "star_count": int(len(values)),
                "star_indices": [int(value) for value in values],
            }
        )
    eligible = _eligible_indices(
        corpus, development, audit, tuple(additional_excluded_arrays)
    )
    excluded_index_set = {
        int(value) for value in development
    } | {int(value) for value in audit}
    for values in additional_excluded_arrays:
        excluded_index_set.update(int(value) for value in values)
    labels = np.asarray(corpus["labels"], dtype=np.float64)
    teacher_mass, teacher_temperature = predict_production_mT_batch(
        labels[eligible], batch_size=args.teacher_batch_size
    )

    target_mass = np.asarray(corpus["column_mass"], dtype=np.float64).copy()
    target_temperature = np.asarray(corpus["temperature"], dtype=np.float64).copy()
    low_gravity = labels[eligible, 1] < float(args.logg_threshold)
    teacher_region = low_gravity.copy()
    if args.teacher_region == "metal_poor_low_gravity":
        teacher_region &= labels[eligible, 2] < float(args.metallicity_threshold)
    elif args.teacher_region == "hot_metal_poor_low_gravity":
        teacher_region &= labels[eligible, 2] < float(args.metallicity_threshold)
        teacher_region &= labels[eligible, 0] > float(args.temperature_threshold)
    eligible_teacher_region = eligible[teacher_region]
    target_mass[eligible_teacher_region] = np.exp(
        (1.0 - args.alpha) * np.log(corpus["column_mass"][eligible_teacher_region])
        + args.alpha * np.log(teacher_mass[teacher_region])
    )
    target_temperature[eligible_teacher_region] = np.exp(
        (1.0 - args.alpha) * np.log(corpus["temperature"][eligible_teacher_region])
        + args.alpha * np.log(teacher_temperature[teacher_region])
    )

    args.target_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.target_out,
        star_indices=eligible,
        labels=labels[eligible],
        truth_column_mass=corpus["column_mass"][eligible],
        truth_temperature=corpus["temperature"][eligible],
        teacher_column_mass=teacher_mass,
        teacher_temperature=teacher_temperature,
        target_column_mass=target_mass[eligible],
        target_temperature=target_temperature[eligible],
    )

    target_corpus = dict(corpus)
    target_corpus["column_mass"] = target_mass
    target_corpus["temperature"] = target_temperature
    train_index = eligible
    validation_index = eligible
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    args.out.mkdir(parents=True, exist_ok=True)
    training_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    summary = {
        "parameterization": "grey_temperature_mass_increment_v1",
        "loss": LOSS_VERSION,
        "target_mode": f"truth_teacher_log_blend_{args.teacher_region}",
        "alpha": float(args.alpha),
        "logg_threshold": float(args.logg_threshold),
        "metallicity_threshold": float(args.metallicity_threshold),
        "temperature_threshold": float(args.temperature_threshold),
        "teacher": "shipped_five_label_initializer_mT_training_only",
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "target_artifact": str(args.target_out),
        "target_artifact_sha256": _sha256(args.target_out),
        "development_manifest": str(args.development_from),
        "development_manifest_sha256": _sha256(args.development_from),
        "sealed_audit_manifest": str(args.audit_manifest),
        "sealed_audit_manifest_sha256": _sha256(args.audit_manifest),
        "development_indices": [int(value) for value in development],
        "sealed_audit_indices": [int(value) for value in audit],
        "additional_excluded_manifests": additional_excluded,
        "excluded_count": int(len(excluded_index_set)),
        "train_count": int(len(train_index)),
        "validation_count": int(len(validation_index)),
        "teacher_region": args.teacher_region,
        "teacher_region_target_count": int(np.sum(teacher_region)),
        "seeds": list(seeds),
        "width": args.width,
        "depth": args.depth,
        "epochs": args.epochs,
        "dtype": args.dtype,
        "device": args.device,
        "hard_weight": args.hard_weight,
        "hard_region": args.hard_region,
        "tail_weight": args.tail_weight,
        "surface_weight": args.surface_weight,
        "arms": {},
    }
    for seed in seeds:
        print(f"=== default-trajectory arm seed={seed} ===", flush=True)
        started = time.perf_counter()
        model, standardization, arm = _train_one(
            target_corpus,
            train_index,
            validation_index,
            width=args.width,
            depth=args.depth,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
            seed=seed,
            dtype=training_dtype,
            device=args.device,
            hard_weight=args.hard_weight,
            tail_weight=args.tail_weight,
            surface_weight=args.surface_weight,
            hard_region=args.hard_region,
        )
        arm["training_seconds"] = time.perf_counter() - started
        arm["validation_target"] = arm["validation"]
        arm["validation_truth"] = _evaluate(
            model,
            standardization,
            labels[eligible],
            corpus["column_mass"][eligible],
            corpus["temperature"][eligible],
            tail_weight=args.tail_weight,
            surface_weight=args.surface_weight,
        )
        checkpoint = args.out / f"checkpoint_physical_seed{seed}.pt"
        save_physical_checkpoint(
            checkpoint,
            model,
            standardization,
            {
                "seed": seed,
                "loss": LOSS_VERSION,
                "target_mode": f"truth_teacher_log_blend_{args.teacher_region}",
                "alpha": float(args.alpha),
                "logg_threshold": float(args.logg_threshold),
                "metallicity_threshold": float(args.metallicity_threshold),
                "temperature_threshold": float(args.temperature_threshold),
                "surface_weight": args.surface_weight,
                "teacher_region": args.teacher_region,
                "teacher": "shipped_five_label_initializer_mT_training_only",
                "corpus": str(args.corpus),
                "corpus_sha256": summary["corpus_sha256"],
                "target_artifact": str(args.target_out),
                "target_artifact_sha256": summary["target_artifact_sha256"],
                "development_manifest": str(args.development_from),
                "development_manifest_sha256": summary[
                    "development_manifest_sha256"
                ],
                "sealed_audit_manifest": str(args.audit_manifest),
                "sealed_audit_manifest_sha256": summary[
                    "sealed_audit_manifest_sha256"
                ],
                "development_indices": [int(value) for value in development],
                "sealed_audit_indices": [int(value) for value in audit],
                "additional_excluded_manifests": additional_excluded,
                "excluded_count": int(len(excluded_index_set)),
                "train_count": int(len(train_index)),
                "validation_count": int(len(validation_index)),
                "hard_region": args.hard_region,
                "hard_weight": args.hard_weight,
                "fit_validation": True,
            },
        )
        arm["checkpoint"] = str(checkpoint)
        summary["arms"][str(seed)] = arm
        print(
            f"target T p95={arm['validation_target']['temperature_relative_p95']:.3e} "
            f"m p95={arm['validation_target']['mass_dex_p95']:.3e} dex; "
            f"truth T p95={arm['validation_truth']['temperature_relative_p95']:.3e} "
            f"m p95={arm['validation_truth']['mass_dex_p95']:.3e} dex",
            flush=True,
        )

    (args.out / "training_default_trajectory_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"wrote {args.out / 'training_default_trajectory_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
