"""Train the first cool-star physical two-field emulator candidate.

The candidate is fitted from scratch on the immutable 52,199-row ATLAS corpus
plus the versioned cool-star truth corpus.  Sampling gives the old and cool
groups equal expected weight without changing either stored corpus.  Model
selection uses only opened validation rows; no sealed M-star track is loaded.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from reduced_state.emulator import (
    PhysicalReducedStateEmulator,
    PhysicalStandardization,
    grey_temperature,
    label_features,
    load_corpus,
    production_tau_grid,
    save_physical_checkpoint,
)

from .train_physical import _coordinate_targets, _evaluate, _load_indices, _loss


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXISTING_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
DEFAULT_COOL_CORPUS = (
    REPO_ROOT / "results" / "m_star_emulator_v1" / "cool_truth_corpus.npz"
)
DEFAULT_DEVELOPMENT = REPO_ROOT / "results" / "reconstruction_metrics.json"
DEFAULT_AUDIT = REPO_ROOT / "results" / "sealed_audit_20260808.json"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "m_star_emulator_v1"

MINIMUM_COOL_TRAIN_ROWS = 20
MINIMUM_COOL_VALIDATION_ROWS = 6
MINIMUM_ROWS_PER_CLASS_TRAIN = 5
MINIMUM_ROWS_PER_CLASS_VALIDATION = 2
GROUP_SAMPLING_WEIGHT = {"existing": 0.5, "cool": 0.5}
LOSS_VERSION = "mstar_balanced_physical_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cool_corpus(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        payload = {
            "labels": np.asarray(data["labels"], dtype=np.float64),
            "column_mass": np.asarray(data["column_mass"], dtype=np.float64),
            "temperature": np.asarray(data["temperature"], dtype=np.float64),
            "roles": np.asarray(data["roles"]).astype(str),
            "track_ids": np.asarray(data["track_ids"]).astype(str),
            "node_ids": np.asarray(data["node_ids"]).astype(str),
            "source_product_paths": np.asarray(data["source_product_paths"]).astype(str),
            "protocol_hash": str(np.asarray(data["protocol_hash"])[0]),
            "flux_gate_hash": str(np.asarray(data["flux_gate_hash"])[0]),
        }
    if "sealed" in set(payload["roles"]):
        raise ValueError("cool training corpus must not contain sealed rows")
    if payload["labels"].shape[1] != 5:
        raise ValueError("cool corpus labels must have five columns")
    if payload["column_mass"].shape != payload["temperature"].shape:
        raise ValueError("cool corpus (m,T) shapes differ")
    if payload["column_mass"].shape[1] != 80:
        raise ValueError("cool corpus must use the 80-layer production grid")
    return payload


def _raw_physical_coordinates(
    labels: np.ndarray,
    column_mass: np.ndarray,
    temperature: np.ndarray,
) -> dict[str, np.ndarray]:
    mass = np.asarray(column_mass, dtype=np.float64)
    increments = np.empty_like(mass)
    increments[:, 0] = mass[:, 0]
    increments[:, 1:] = np.diff(mass, axis=1)
    if np.any(increments <= 0.0):
        raise ValueError("all column-mass increments must be positive")
    tau = production_tau_grid(mass.shape[1])
    return {
        "features": label_features(labels),
        "log_temperature_ratio": np.log10(
            np.asarray(temperature, dtype=np.float64)
            / grey_temperature(labels[:, 0], tau)
        ),
        "log_mass_increment": np.log10(increments),
        "log_column_mass": np.log10(mass),
    }


def _equal_group_moments(
    existing: np.ndarray,
    cool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Moments of a distribution assigning each corpus group weight one half."""

    existing = np.asarray(existing, dtype=np.float64)
    cool = np.asarray(cool, dtype=np.float64)
    mean = 0.5 * existing.mean(axis=0) + 0.5 * cool.mean(axis=0)
    variance = 0.5 * np.mean((existing - mean) ** 2, axis=0) + 0.5 * np.mean(
        (cool - mean) ** 2, axis=0
    )
    return mean, np.sqrt(variance) + 1.0e-12


def fit_balanced_standardization(
    existing_labels: np.ndarray,
    existing_mass: np.ndarray,
    existing_temperature: np.ndarray,
    cool_labels: np.ndarray,
    cool_mass: np.ndarray,
    cool_temperature: np.ndarray,
) -> PhysicalStandardization:
    existing = _raw_physical_coordinates(
        existing_labels, existing_mass, existing_temperature
    )
    cool = _raw_physical_coordinates(cool_labels, cool_mass, cool_temperature)
    feature_mean, feature_std = _equal_group_moments(
        existing["features"], cool["features"]
    )
    ratio_mean, ratio_std = _equal_group_moments(
        existing["log_temperature_ratio"], cool["log_temperature_ratio"]
    )
    increment_mean, increment_std = _equal_group_moments(
        existing["log_mass_increment"], cool["log_mass_increment"]
    )
    mass_mean, mass_std = _equal_group_moments(
        existing["log_column_mass"], cool["log_column_mass"]
    )
    return PhysicalStandardization(
        feature_mean=feature_mean,
        feature_std=feature_std,
        log_temperature_ratio_mean=ratio_mean,
        log_temperature_ratio_std=ratio_std,
        log_mass_increment_mean=increment_mean,
        log_mass_increment_std=increment_std,
        log_column_mass_mean=mass_mean,
        log_column_mass_std=mass_std,
    )


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    labels = np.asarray(labels)
    return {
        "giant": int(np.sum(labels[:, 1] < 3.5)),
        "dwarf": int(np.sum(labels[:, 1] >= 3.5)),
    }


def validate_cool_corpus(cool: dict[str, np.ndarray]) -> dict[str, Any]:
    train = cool["roles"] == "train"
    validation = cool["roles"] == "validation"
    train_counts = _class_counts(cool["labels"][train])
    validation_counts = _class_counts(cool["labels"][validation])
    failures = []
    if int(np.sum(train)) < MINIMUM_COOL_TRAIN_ROWS:
        failures.append("cool_train_row_count")
    if int(np.sum(validation)) < MINIMUM_COOL_VALIDATION_ROWS:
        failures.append("cool_validation_row_count")
    for stellar_class in ("giant", "dwarf"):
        if train_counts[stellar_class] < MINIMUM_ROWS_PER_CLASS_TRAIN:
            failures.append(f"cool_train_{stellar_class}_count")
        if validation_counts[stellar_class] < MINIMUM_ROWS_PER_CLASS_VALIDATION:
            failures.append(f"cool_validation_{stellar_class}_count")
    return {
        "passes": not failures,
        "failures": failures,
        "train_rows": int(np.sum(train)),
        "validation_rows": int(np.sum(validation)),
        "train_class_counts": train_counts,
        "validation_class_counts": validation_counts,
    }


def _standardized_tensors(
    labels: np.ndarray,
    column_mass: np.ndarray,
    temperature: np.ndarray,
    standardization: PhysicalStandardization,
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = label_features(labels)
    x = (features - standardization.feature_mean) / standardization.feature_std
    target_temperature, target_mass = _coordinate_targets(
        labels,
        column_mass,
        temperature,
        standardization,
    )
    return (
        torch.as_tensor(x, dtype=dtype),
        torch.as_tensor(target_temperature, dtype=dtype),
        torch.as_tensor(target_mass, dtype=dtype),
    )


def _train_one(
    *,
    existing: dict[str, np.ndarray],
    cool: dict[str, np.ndarray],
    existing_train_index: np.ndarray,
    existing_validation_index: np.ndarray,
    cool_train_index: np.ndarray,
    cool_validation_index: np.ndarray,
    seed: int,
    width: int,
    depth: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    patience: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[
    PhysicalReducedStateEmulator,
    PhysicalStandardization,
    dict[str, Any],
]:
    torch.manual_seed(seed)
    standardization = fit_balanced_standardization(
        existing["labels"][existing_train_index],
        existing["column_mass"][existing_train_index],
        existing["temperature"][existing_train_index],
        cool["labels"][cool_train_index],
        cool["column_mass"][cool_train_index],
        cool["temperature"][cool_train_index],
    )
    old_train = _standardized_tensors(
        existing["labels"][existing_train_index],
        existing["column_mass"][existing_train_index],
        existing["temperature"][existing_train_index],
        standardization,
        dtype=dtype,
    )
    cool_train = _standardized_tensors(
        cool["labels"][cool_train_index],
        cool["column_mass"][cool_train_index],
        cool["temperature"][cool_train_index],
        standardization,
        dtype=dtype,
    )
    combined = tuple(
        torch.cat([old_values, cool_values], dim=0)
        for old_values, cool_values in zip(old_train, cool_train)
    )
    origin = torch.cat(
        [
            torch.zeros(len(existing_train_index), dtype=torch.int64),
            torch.ones(len(cool_train_index), dtype=torch.int64),
        ]
    )
    weights = torch.empty(len(origin), dtype=torch.float64)
    weights[origin == 0] = (
        GROUP_SAMPLING_WEIGHT["existing"] / len(existing_train_index)
    )
    weights[origin == 1] = GROUP_SAMPLING_WEIGHT["cool"] / len(cool_train_index)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(origin),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        TensorDataset(*combined),
        batch_size=min(batch_size, len(origin)),
        sampler=sampler,
        drop_last=False,
    )

    old_validation = _standardized_tensors(
        existing["labels"][existing_validation_index],
        existing["column_mass"][existing_validation_index],
        existing["temperature"][existing_validation_index],
        standardization,
        dtype=dtype,
    )
    cool_validation = _standardized_tensors(
        cool["labels"][cool_validation_index],
        cool["column_mass"][cool_validation_index],
        cool["temperature"][cool_validation_index],
        standardization,
        dtype=dtype,
    )
    old_validation = tuple(value.to(device) for value in old_validation)
    cool_validation = tuple(value.to(device) for value in cool_validation)

    model = PhysicalReducedStateEmulator(
        width=width,
        depth=depth,
        dtype=dtype,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        seen = 0
        for x_batch, t_batch, m_batch in loader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            m_batch = m_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _terms = _loss(
                model,
                x_batch,
                t_batch,
                m_batch,
                standardization,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total += float(loss.detach()) * len(x_batch)
            seen += len(x_batch)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            old_loss, old_terms = _loss(
                model,
                old_validation[0],
                old_validation[1],
                old_validation[2],
                standardization,
            )
            cool_loss, cool_terms = _loss(
                model,
                cool_validation[0],
                cool_validation[1],
                cool_validation[2],
                standardization,
            )
            validation_loss = 0.5 * old_loss + 0.5 * cool_loss
        value = float(validation_loss)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / max(seen, 1),
                "validation_loss": value,
                "existing_validation_loss": float(old_loss),
                "cool_validation_loss": float(cool_loss),
                "existing_validation_terms": old_terms,
                "cool_validation_terms": cool_terms,
            }
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(
                f"seed={seed} epoch={epoch + 1:03d} "
                f"train={total / max(seen, 1):.3e} "
                f"old={float(old_loss):.3e} cool={float(cool_loss):.3e}",
                flush=True,
            )
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if patience > 0 and stale >= patience:
            break
    model.load_state_dict(best_state)
    model = model.cpu()
    return model, standardization, {
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "history": history,
        "existing_validation": _evaluate(
            model,
            standardization,
            existing["labels"][existing_validation_index],
            existing["column_mass"][existing_validation_index],
            existing["temperature"][existing_validation_index],
        ),
        "cool_validation": _evaluate(
            model,
            standardization,
            cool["labels"][cool_validation_index],
            cool["column_mass"][cool_validation_index],
            cool["temperature"][cool_validation_index],
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-corpus", type=Path, default=DEFAULT_EXISTING_CORPUS)
    parser.add_argument("--cool-corpus", type=Path, default=DEFAULT_COOL_CORPUS)
    parser.add_argument("--development-from", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--audit-manifest", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", default="20260831,20260901,20260902")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--campaign", default="m_star_emulator_v1")
    args = parser.parse_args(argv)

    existing = load_corpus(args.existing_corpus)
    cool = load_cool_corpus(args.cool_corpus)
    cool_gate = validate_cool_corpus(cool)
    if not cool_gate["passes"]:
        raise SystemExit(f"FAIL_STOP: sparse cool corpus: {cool_gate['failures']}")

    development = set(int(value) for value in _load_indices(args.development_from))
    audit = set(int(value) for value in _load_indices(args.audit_manifest))
    excluded = development | audit
    existing_train_index = np.asarray(
        [
            index
            for index, role in enumerate(existing["roles"])
            if role == "train" and index not in excluded
        ],
        dtype=np.int64,
    )
    existing_validation_index = np.asarray(
        [
            index
            for index, role in enumerate(existing["roles"])
            if role == "validation" and index not in excluded
        ],
        dtype=np.int64,
    )
    cool_train_index = np.flatnonzero(cool["roles"] == "train")
    cool_validation_index = np.flatnonzero(cool["roles"] == "validation")
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    args.out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "campaign": str(args.campaign),
        "status": "training",
        "loss_version": LOSS_VERSION,
        "group_sampling_weight": GROUP_SAMPLING_WEIGHT,
        "existing_corpus": str(args.existing_corpus),
        "existing_corpus_sha256": _sha256(args.existing_corpus),
        "cool_corpus": str(args.cool_corpus),
        "cool_corpus_sha256": _sha256(args.cool_corpus),
        "cool_protocol_hash": cool["protocol_hash"],
        "cool_flux_gate_hash": cool["flux_gate_hash"],
        "cool_corpus_gate": cool_gate,
        "sealed_cool_rows_loaded": False,
        "development_manifest": str(args.development_from),
        "development_manifest_sha256": _sha256(args.development_from),
        "development_indices": sorted(development),
        "audit_manifest": str(args.audit_manifest),
        "audit_manifest_sha256": _sha256(args.audit_manifest),
        "sealed_audit_indices": sorted(audit),
        "existing_train_count": len(existing_train_index),
        "existing_validation_count": len(existing_validation_index),
        "cool_train_count": len(cool_train_index),
        "cool_validation_count": len(cool_validation_index),
        "width": args.width,
        "depth": args.depth,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "patience": args.patience,
        "dtype": args.dtype,
        "device": args.device,
        "seeds": list(seeds),
        "arms": {},
        "production_routing_changed": False,
        "existing_sealed_holdout_opened": False,
    }
    for seed in seeds:
        started = time.perf_counter()
        model, standardization, arm = _train_one(
            existing=existing,
            cool=cool,
            existing_train_index=existing_train_index,
            existing_validation_index=existing_validation_index,
            cool_train_index=cool_train_index,
            cool_validation_index=cool_validation_index,
            seed=seed,
            width=args.width,
            depth=args.depth,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
            dtype=dtype,
            device=args.device,
        )
        checkpoint = args.out / f"checkpoint_mstar_seed{seed}.pt"
        arm["training_seconds"] = time.perf_counter() - started
        save_physical_checkpoint(
            checkpoint,
            model,
            standardization,
            {
                "campaign": str(args.campaign),
                "seed": seed,
                "loss_version": LOSS_VERSION,
                "existing_corpus_sha256": summary["existing_corpus_sha256"],
                "cool_corpus_sha256": summary["cool_corpus_sha256"],
                "cool_protocol_hash": cool["protocol_hash"],
                "cool_flux_gate_hash": cool["flux_gate_hash"],
                "development_indices": sorted(development),
                "sealed_audit_indices": sorted(audit),
                "sealed_cool_rows_loaded": False,
                "production_routing_changed": False,
            },
        )
        arm["checkpoint"] = str(checkpoint)
        arm["checkpoint_sha256"] = _sha256(checkpoint)
        summary["arms"][str(seed)] = arm
        temporary = args.out / "training_summary.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        temporary.replace(args.out / "training_summary.json")
    summary["status"] = "complete"
    summary["completed_arms"] = len(summary["arms"])
    temporary = args.out / "training_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out / "training_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
