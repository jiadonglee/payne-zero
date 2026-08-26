"""Evaluate the profile gate on the whole held-out validation population.

``evaluate_physical.py`` scores a frozen manifest -- dev-60, audit-200, the
sealed blind-200. Those sets are small because the *solver* runs that accompany
them are expensive. The profile gate itself is not: it compares a predicted
``(m,T)`` against corpus truth, which is one forward pass through an MLP and no
solver at all. There was never a compute reason to estimate it from 60 stars.

Estimating it from 60 stars cost us a real result. The 2026-08-11 blind test
(`solver-in-the-loop-progress.md` §12) selected a checkpoint whose dev-60
column-mass p95 was `1.28e-2` -- inside the `1.52e-2` limit -- and then measured
`2.66e-2` on 200 sealed stars, a 2.75x generalization gap that the selection set
was far too small to reveal. This script exists so that the honest number is
always the cheap one to obtain.

**The `fit_validation` trap.** ``train_physical.py --fit-validation`` trains on
train+validation rows and sets ``validation_index = train_index.copy()``. That is
a legitimate final refit once hyperparameters are chosen, but it means the
validation-role rows are *trained on*, so scoring them measures memorization, not
generalization. A checkpoint carrying ``fit_validation: True`` is therefore
refused here unless ``--allow-fit-validation`` is passed, in which case the
report is labelled as a training-set measurement and must not be quoted as a
held-out result.
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

# Identical to evaluate_physical.py; duplicated deliberately rather than
# imported, so that changing one gate cannot silently move the other.
TEMPERATURE_STAR_MAX_LT = 0.10
MASS_STAR_MAX_LT_DEX = 0.10
TEMPERATURE_POINTWISE_P95_LTE = 3.74e-3
MASS_POINTWISE_P95_LTE_DEX = 1.52e-2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        name: float(np.percentile(values, q))
        for name, q in (("p50", 50.0), ("p90", 90.0), ("p95", 95.0), ("p99", 99.0))
    } | {"max": float(np.max(values))}


def excluded_indices(meta: dict) -> set[int]:
    """Every star index the checkpoint declares it was not trained on."""

    excluded = set(int(v) for v in meta.get("development_indices", []))
    excluded |= set(int(v) for v in meta.get("sealed_audit_indices", []))
    for manifest in meta.get("additional_excluded_manifests", []):
        if not isinstance(manifest, dict):
            raise ValueError("checkpoint has an invalid additional exclusion entry")
        excluded |= set(int(v) for v in manifest.get("star_indices", []))
    return excluded


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="validation_population")
    parser.add_argument(
        "--allow-fit-validation",
        action="store_true",
        help=(
            "score a --fit-validation checkpoint anyway; the result is a "
            "training-set measurement and is labelled as such"
        ),
    )
    args = parser.parse_args(argv)

    model, standardization, meta = load_physical_checkpoint(args.checkpoint)
    trained_on_validation = bool(meta.get("fit_validation", False))
    if trained_on_validation and not args.allow_fit_validation:
        raise SystemExit(
            f"{args.checkpoint} was trained with --fit-validation, so the "
            "validation-role rows were part of its training set. Scoring them "
            "measures memorization, not generalization. Retrain without "
            "--fit-validation, or pass --allow-fit-validation to get a "
            "training-set number that must not be quoted as held out."
        )

    corpus = load_corpus(args.corpus)
    roles = corpus["roles"]
    excluded = excluded_indices(meta)
    indices = np.array(
        [
            i
            for i in range(len(roles))
            if roles[i] == "validation" and int(i) not in excluded
        ],
        dtype=np.int64,
    )
    if indices.size == 0:
        raise SystemExit("no held-out validation rows remain to evaluate")

    labels = corpus["labels"][indices]
    mass, temperature = predict_physical_state(model, standardization, labels)
    truth_mass = corpus["column_mass"][indices]
    truth_temperature = corpus["temperature"][indices]

    temperature_error = np.abs(temperature - truth_temperature) / truth_temperature
    mass_error = np.abs(np.log10(mass) - np.log10(truth_mass))
    temperature_star_max = temperature_error.max(axis=1)
    mass_star_max = mass_error.max(axis=1)

    gate = {
        "temperature_no_blowout": bool(
            np.all(temperature_star_max < TEMPERATURE_STAR_MAX_LT)
        ),
        "mass_no_blowout": bool(np.all(mass_star_max < MASS_STAR_MAX_LT_DEX)),
        "temperature_pointwise_p95": bool(
            np.percentile(temperature_error, 95.0) <= TEMPERATURE_POINTWISE_P95_LTE
        ),
        "mass_pointwise_p95": bool(
            np.percentile(mass_error, 95.0) <= MASS_POINTWISE_P95_LTE_DEX
        ),
    }
    gate["profile_gate_passed"] = all(gate.values())

    result = {
        "name": args.name,
        "measures": "training_set" if trained_on_validation else "held_out",
        "star_count": int(indices.size),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "fit_validation": trained_on_validation,
        "excluded_manifest_star_count": int(len(excluded)),
        "pointwise": {
            "temperature_relative": _percentiles(temperature_error.ravel()),
            "mass_dex": _percentiles(mass_error.ravel()),
        },
        "per_star_maximum": {
            "temperature_relative": _percentiles(temperature_star_max),
            "mass_dex": _percentiles(mass_star_max),
        },
        "blowout_counts": {
            "temperature": int((temperature_star_max >= TEMPERATURE_STAR_MAX_LT).sum()),
            "mass": int((mass_star_max >= MASS_STAR_MAX_LT_DEX).sum()),
        },
        "thresholds": {
            "temperature_star_max_lt": TEMPERATURE_STAR_MAX_LT,
            "mass_star_max_lt_dex": MASS_STAR_MAX_LT_DEX,
            "temperature_pointwise_p95_lte": TEMPERATURE_POINTWISE_P95_LTE,
            "mass_pointwise_p95_lte_dex": MASS_POINTWISE_P95_LTE_DEX,
        },
        "gate": gate,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{args.name} [{result['measures']}] n={result['star_count']}: "
        f"T p95={result['pointwise']['temperature_relative']['p95']:.3e}, "
        f"m p95={result['pointwise']['mass_dex']['p95']:.3e} dex, "
        f"blowouts T/m={result['blowout_counts']['temperature']}/"
        f"{result['blowout_counts']['mass']}, "
        f"gate={gate['profile_gate_passed']}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
