"""Train the physically conditioned reduced-state emulator.

The target coordinates match the production decoder's stable conventions:

* ``log10(T / T_grey)`` instead of absolute log temperature;
* ``log10(delta_m)`` instead of cumulative log column mass.

Positive mass increments make the decoded column-mass profile strictly
monotonic. The existing 60-star development set and a separately sealed audit
set are excluded from fitting. Multiple seeds are saved so the inference
process can use their coordinate-wise median when a single model is unstable.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from reduced_state.emulator import (
    PhysicalReducedStateEmulator,
    PhysicalStandardization,
    fit_physical_standardization,
    grey_temperature,
    label_features,
    load_corpus,
    production_tau_grid,
    save_physical_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
DEFAULT_DEVELOPMENT = REPO_ROOT / "results" / "reconstruction_metrics.json"
DEFAULT_AUDIT = REPO_ROOT / "results" / "sealed_audit_20260808.json"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "reduced_state_emulator" / "physical"
LOSS_VERSION = "smooth_l1_increment_cumulative_mass_profile_shape_tail_v2"
TAIL_LOSS_WEIGHT = 0.1
HARD_REGION_DEFAULT_WEIGHT = 3.0
SURFACE_LOSS_LAYERS = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coordinate_targets(
    labels: np.ndarray,
    column_mass: np.ndarray,
    temperature: np.ndarray,
    standardization: PhysicalStandardization,
) -> tuple[np.ndarray, np.ndarray]:
    tau = production_tau_grid(column_mass.shape[1])
    increments = np.empty_like(column_mass)
    increments[:, 0] = column_mass[:, 0]
    increments[:, 1:] = np.diff(column_mass, axis=1)
    log_ratio = np.log10(temperature / grey_temperature(labels[:, 0], tau))
    log_increment = np.log10(increments)
    return (
        (log_ratio - standardization.log_temperature_ratio_mean)
        / standardization.log_temperature_ratio_std,
        (log_increment - standardization.log_mass_increment_mean)
        / standardization.log_mass_increment_std,
    )


def _load_indices(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if "star_indices" not in payload:
        raise ValueError(f"{path} does not contain star_indices")
    return np.asarray(payload["star_indices"], dtype=np.int64)


def _load_manifest_indices(path: Path) -> np.ndarray:
    values = _load_indices(path)
    if values.ndim != 1:
        raise ValueError(f"{path} star_indices must be one-dimensional")
    return values


def _loss(
    model: PhysicalReducedStateEmulator,
    x: torch.Tensor,
    target_temperature: torch.Tensor,
    target_mass: torch.Tensor,
    standardization: PhysicalStandardization,
    sample_weight: torch.Tensor | None = None,
    tail_weight: float = TAIL_LOSS_WEIGHT,
    surface_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    def weighted_mean(values: torch.Tensor) -> torch.Tensor:
        per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
        if sample_weight is None:
            return per_sample.mean()
        return (per_sample * sample_weight).sum() / sample_weight.sum()

    raw_temperature, raw_mass = model(x)
    temperature_loss = weighted_mean(
        F.smooth_l1_loss(raw_temperature, target_temperature, reduction="none")
    )
    increment_loss = weighted_mean(
        F.smooth_l1_loss(raw_mass, target_mass, reduction="none")
    )

    increment_mean = torch.as_tensor(
        standardization.log_mass_increment_mean,
        dtype=raw_mass.dtype,
        device=raw_mass.device,
    )
    increment_std = torch.as_tensor(
        standardization.log_mass_increment_std,
        dtype=raw_mass.dtype,
        device=raw_mass.device,
    )
    predicted_log_increment = raw_mass * increment_std + increment_mean
    target_log_increment = target_mass * increment_std + increment_mean
    predicted_log_mass = torch.log10(
        torch.cumsum(torch.pow(10.0, torch.clamp(predicted_log_increment, -30.0, 30.0)), dim=1)
    )
    target_log_mass = torch.log10(
        torch.cumsum(torch.pow(10.0, torch.clamp(target_log_increment, -30.0, 30.0)), dim=1)
    )
    column_mass_mean = torch.as_tensor(
        standardization.log_column_mass_mean,
        dtype=raw_mass.dtype,
        device=raw_mass.device,
    )
    column_mass_std = torch.as_tensor(
        standardization.log_column_mass_std,
        dtype=raw_mass.dtype,
        device=raw_mass.device,
    )
    predicted_standardized_log_mass = (
        predicted_log_mass - column_mass_mean
    ) / column_mass_std
    target_standardized_log_mass = (
        target_log_mass - column_mass_mean
    ) / column_mass_std
    mass_profile_loss = weighted_mean(
        F.smooth_l1_loss(
            predicted_standardized_log_mass,
            target_standardized_log_mass,
            reduction="none",
        )
    )
    mass_profile_tail_loss = weighted_mean(
        F.mse_loss(
            predicted_standardized_log_mass,
            target_standardized_log_mass,
            reduction="none",
        )
    )
    surface_layers = min(SURFACE_LOSS_LAYERS, predicted_standardized_log_mass.shape[1])
    mass_surface_loss = weighted_mean(
        F.smooth_l1_loss(
            predicted_standardized_log_mass[:, :surface_layers],
            target_standardized_log_mass[:, :surface_layers],
            reduction="none",
        )
    )
    mass_loss = (
        increment_loss
        + 0.5 * mass_profile_loss
        + float(surface_weight) * mass_surface_loss
    )

    temperature_mean = torch.as_tensor(
        standardization.log_temperature_ratio_mean,
        dtype=raw_temperature.dtype,
        device=raw_temperature.device,
    )
    temperature_std = torch.as_tensor(
        standardization.log_temperature_ratio_std,
        dtype=raw_temperature.dtype,
        device=raw_temperature.device,
    )
    predicted_log_ratio = raw_temperature * temperature_std + temperature_mean
    target_log_ratio = target_temperature * temperature_std + temperature_mean
    temperature_tail_loss = weighted_mean(
        F.mse_loss(raw_temperature, target_temperature, reduction="none")
    )
    temperature_shape_loss = weighted_mean(
        F.smooth_l1_loss(
            predicted_log_ratio[:, 1:] - predicted_log_ratio[:, :-1],
            target_log_ratio[:, 1:] - target_log_ratio[:, :-1],
            reduction="none",
        )
    )
    mass_shape_loss = weighted_mean(
        F.smooth_l1_loss(
            predicted_log_increment[:, 1:] - predicted_log_increment[:, :-1],
            target_log_increment[:, 1:] - target_log_increment[:, :-1],
            reduction="none",
        )
    )
    shape_loss = temperature_shape_loss + mass_shape_loss
    total = (
        temperature_loss
        + mass_loss
        + 0.1 * shape_loss
        + float(tail_weight) * (temperature_tail_loss + mass_profile_tail_loss)
    )
    return total, {
        "temperature": float(temperature_loss.detach()),
        "mass": float(mass_loss.detach()),
        "mass_increment": float(increment_loss.detach()),
        "mass_profile": float(mass_profile_loss.detach()),
        "temperature_tail": float(temperature_tail_loss.detach()),
        "mass_profile_tail": float(mass_profile_tail_loss.detach()),
        "mass_surface": float(mass_surface_loss.detach()),
        "shape": float(shape_loss.detach()),
    }


def _evaluate(
    model: PhysicalReducedStateEmulator,
    standardization: PhysicalStandardization,
    labels: np.ndarray,
    truth_column_mass: np.ndarray,
    truth_temperature: np.ndarray,
    tail_weight: float = TAIL_LOSS_WEIGHT,
    surface_weight: float = 0.0,
) -> dict:
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device
    features = label_features(labels)
    standardized = (
        features - standardization.feature_mean
    ) / standardization.feature_std
    target_temperature, target_mass = _coordinate_targets(
        labels, truth_column_mass, truth_temperature, standardization
    )
    with torch.no_grad():
        raw_temperature, raw_mass = model(
            torch.as_tensor(standardized, dtype=model_dtype, device=model_device)
        )
        loss, terms = _loss(
            model,
            torch.as_tensor(standardized, dtype=model_dtype, device=model_device),
            torch.as_tensor(target_temperature, dtype=model_dtype, device=model_device),
            torch.as_tensor(target_mass, dtype=model_dtype, device=model_device),
            standardization,
            tail_weight=tail_weight,
            surface_weight=surface_weight,
        )
    log_temperature = (
        raw_temperature.numpy() * standardization.log_temperature_ratio_std
        + standardization.log_temperature_ratio_mean
    )
    log_mass_increment = (
        raw_mass.numpy() * standardization.log_mass_increment_std
        + standardization.log_mass_increment_mean
    )
    tau = production_tau_grid(truth_column_mass.shape[1])
    predicted_temperature = grey_temperature(labels[:, 0], tau) * 10.0**log_temperature
    predicted_mass = np.cumsum(10.0**log_mass_increment, axis=1)
    temperature_relative = np.abs(predicted_temperature - truth_temperature) / truth_temperature
    mass_dex = np.abs(np.log10(predicted_mass) - np.log10(truth_column_mass))
    return {
        "loss": float(loss),
        "loss_terms": terms,
        "temperature_relative_p95": float(np.percentile(temperature_relative, 95.0)),
        "temperature_relative_max": float(np.max(temperature_relative)),
        "mass_dex_p95": float(np.percentile(mass_dex, 95.0)),
        "mass_dex_max": float(np.max(mass_dex)),
        "monotonicity_violations": int(
            np.any(np.diff(predicted_mass, axis=1) <= 0.0, axis=1).sum()
        ),
    }


def _train_one(
    corpus: dict,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    width: int,
    depth: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    patience: int,
    seed: int,
    dtype: torch.dtype,
    device: str,
    hard_weight: float,
    tail_weight: float,
    surface_weight: float,
    hard_region: str,
) -> tuple[PhysicalReducedStateEmulator, PhysicalStandardization, dict]:
    torch.manual_seed(seed)
    labels = corpus["labels"]
    features = label_features(labels)
    standardization = fit_physical_standardization(
        features[train_index],
        corpus["column_mass"][train_index],
        corpus["temperature"][train_index],
        labels[train_index, 0],
    )
    train_x = (features[train_index] - standardization.feature_mean) / standardization.feature_std
    validation_x = (
        features[validation_index] - standardization.feature_mean
    ) / standardization.feature_std
    train_t, train_m = _coordinate_targets(
        labels[train_index],
        corpus["column_mass"][train_index],
        corpus["temperature"][train_index],
        standardization,
    )
    validation_t, validation_m = _coordinate_targets(
        labels[validation_index],
        corpus["column_mass"][validation_index],
        corpus["temperature"][validation_index],
        standardization,
    )
    def hard_region_weight(values: np.ndarray) -> np.ndarray:
        if hard_region == "hot_metal_poor":
            hard = (
                (values[:, 0] > 7000.0)
                & (values[:, 1] < 4.0)
                & (values[:, 2] < -0.3)
            )
        elif hard_region == "low_gravity":
            hard = values[:, 1] < 3.2
        elif hard_region == "solver_tail":
            # The solver failures occupy several label edges rather than one
            # rectangle: low gravity, high microturbulence, and alpha-rich
            # cool giants.  This is a label-defined region, independent of
            # the sealed solver subset.
            hard = (
                (values[:, 1] < 3.2)
                | (values[:, 4] > 3.0)
                | (values[:, 3] > 0.4)
            )
        else:  # pragma: no cover - argparse constrains this in normal use
            raise ValueError(f"unknown hard region: {hard_region}")
        return np.where(hard, float(hard_weight), 1.0).astype(np.float64)

    train_weight = hard_region_weight(labels[train_index])
    validation_weight = hard_region_weight(labels[validation_index])
    dataset = TensorDataset(
        torch.as_tensor(train_x, dtype=dtype),
        torch.as_tensor(train_t, dtype=dtype),
        torch.as_tensor(train_m, dtype=dtype),
        torch.as_tensor(train_weight, dtype=dtype),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=True,
        drop_last=False,
    )
    model = PhysicalReducedStateEmulator(width=width, depth=depth, dtype=dtype).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs))
    )
    validation_x_t = torch.as_tensor(validation_x, dtype=dtype, device=device)
    validation_t_t = torch.as_tensor(validation_t, dtype=dtype, device=device)
    validation_m_t = torch.as_tensor(validation_m, dtype=dtype, device=device)
    validation_weight_t = torch.as_tensor(
        validation_weight, dtype=dtype, device=device
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    history: list[dict] = []
    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        for x_batch, t_batch, m_batch, weight_batch in loader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            m_batch = m_batch.to(device)
            weight_batch = weight_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _terms = _loss(
                model,
                x_batch,
                t_batch,
                m_batch,
                standardization,
                weight_batch,
                tail_weight,
                surface_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            running += float(loss.detach()) * len(x_batch)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            validation_loss, validation_terms = _loss(
                model,
                validation_x_t,
                validation_t_t,
                validation_m_t,
                standardization,
                validation_weight_t,
                tail_weight,
                surface_weight,
            )
        value = float(validation_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(dataset),
                "validation_loss": value,
                "validation_terms": validation_terms,
            }
        )
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == int(epochs) - 1:
            print(
                f"  epoch={epoch + 1:03d} train={running / len(dataset):.3e} "
                f"validation={value:.3e}",
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
        "validation": _evaluate(
            model,
            standardization,
            labels[validation_index],
            corpus["column_mass"][validation_index],
            corpus["temperature"][validation_index],
            tail_weight=tail_weight,
            surface_weight=surface_weight,
        ),
        "hard_region_train_count": int(np.sum(train_weight > 1.0)),
        "hard_region_validation_count": int(np.sum(validation_weight > 1.0)),
    }


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
    parser.add_argument("--seeds", default="20260807,20260808,20260809")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hard-weight", type=float, default=HARD_REGION_DEFAULT_WEIGHT)
    parser.add_argument(
        "--hard-region",
        choices=("hot_metal_poor", "low_gravity", "solver_tail"),
        default="hot_metal_poor",
        help="which label region receives the extra profile loss weight",
    )
    parser.add_argument("--tail-weight", type=float, default=TAIL_LOSS_WEIGHT)
    parser.add_argument(
        "--surface-weight",
        type=float,
        default=0.0,
        help=(
            "extra mass-profile loss weight on the first eight optical-depth "
            "layers; zero preserves the original loss"
        ),
    )
    parser.add_argument(
        "--fit-validation",
        action="store_true",
        help="after model selection, fit on train+validation rows (held-out dev/audit remain excluded)",
    )
    args = parser.parse_args(argv)

    corpus = load_corpus(args.corpus)
    development = _load_indices(args.development_from)
    audit = _load_indices(args.audit_manifest)
    excluded = set(int(value) for value in development) | set(int(value) for value in audit)
    additional_excluded = []
    for manifest in args.exclude_manifest:
        values = _load_manifest_indices(manifest)
        excluded.update(int(value) for value in values)
        additional_excluded.append(
            {
                "path": str(manifest),
                "sha256": _sha256(manifest),
                "star_count": int(len(values)),
                "star_indices": [int(value) for value in values],
            }
        )
    all_indices = np.arange(len(corpus["labels"]), dtype=np.int64)
    if args.fit_validation:
        train_index = np.array(
            [
                i
                for i in all_indices
                if corpus["roles"][i] in {"train", "validation"}
                and int(i) not in excluded
            ],
            dtype=np.int64,
        )
        # This is a final fit after hyperparameters have been selected. The
        # validation loss is only a training diagnostic; development and audit
        # rows are still excluded and remain available for evaluation.
        validation_index = train_index.copy()
    else:
        train_index = np.array(
            [
                i
                for i in all_indices
                if corpus["roles"][i] == "train" and int(i) not in excluded
            ],
            dtype=np.int64,
        )
        validation_index = np.array(
            [
                i
                for i in all_indices
                if corpus["roles"][i] == "validation" and int(i) not in excluded
            ],
            dtype=np.int64,
        )
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if not np.isfinite(args.hard_weight) or args.hard_weight < 1.0:
        raise ValueError("--hard-weight must be finite and >= 1")
    if not np.isfinite(args.tail_weight) or args.tail_weight < 0.0:
        raise ValueError("--tail-weight must be finite and >= 0")
    if not np.isfinite(args.surface_weight) or args.surface_weight < 0.0:
        raise ValueError("--surface-weight must be finite and >= 0")
    args.out.mkdir(parents=True, exist_ok=True)
    training_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    summary = {
        "parameterization": "grey_temperature_mass_increment_v1",
        "loss": LOSS_VERSION,
        "hard_weight": args.hard_weight,
        "hard_region": args.hard_region,
        "tail_weight": args.tail_weight,
        "surface_weight": args.surface_weight,
        "fit_validation": args.fit_validation,
        "corpus": str(args.corpus),
        "corpus_sha256": _sha256(args.corpus),
        "development_manifest": str(args.development_from),
        "development_manifest_sha256": _sha256(args.development_from),
        "sealed_audit_manifest": str(args.audit_manifest),
        "sealed_audit_manifest_sha256": _sha256(args.audit_manifest),
        "development_indices": [int(i) for i in development],
        "sealed_audit_indices": [int(i) for i in audit],
        "additional_excluded_manifests": additional_excluded,
        "excluded_count": int(len(excluded)),
        "train_count": int(len(train_index)),
        "validation_count": int(len(validation_index)),
        "seeds": list(seeds),
        "width": args.width,
        "depth": args.depth,
        "epochs": args.epochs,
        "dtype": args.dtype,
        "device": args.device,
        "arms": {},
    }
    for seed in seeds:
        print(f"=== physical arm seed={seed} ===", flush=True)
        started = time.perf_counter()
        model, standardization, arm = _train_one(
            corpus,
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
        checkpoint = args.out / f"checkpoint_physical_seed{seed}.pt"
        save_physical_checkpoint(
            checkpoint,
            model,
            standardization,
            {
                "seed": seed,
                "loss": LOSS_VERSION,
                "corpus": str(args.corpus),
                "corpus_sha256": summary["corpus_sha256"],
                "development_manifest": str(args.development_from),
                "development_manifest_sha256": summary[
                    "development_manifest_sha256"
                ],
                "sealed_audit_manifest": str(args.audit_manifest),
                "sealed_audit_manifest_sha256": summary[
                    "sealed_audit_manifest_sha256"
                ],
                "development_indices": [int(i) for i in development],
                "sealed_audit_indices": [int(i) for i in audit],
                "additional_excluded_manifests": additional_excluded,
                "excluded_count": int(len(excluded)),
                "fit_validation": args.fit_validation,
                "hard_region": args.hard_region,
                "surface_weight": args.surface_weight,
                "train_count": int(len(train_index)),
                "validation_count": int(len(validation_index)),
            },
        )
        arm["checkpoint"] = str(checkpoint)
        summary["arms"][str(seed)] = arm
        print(
            f"validation T p95={arm['validation']['temperature_relative_p95']:.3e} "
            f"m p95={arm['validation']['mass_dex_p95']:.3e} dex",
            flush=True,
        )
    (args.out / "training_physical_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"wrote {args.out / 'training_physical_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
