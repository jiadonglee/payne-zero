"""Matched two- and six-profile networks for the coauthor validation split."""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from payne_zero_atmosphere.direct_abundance import (
    DIRECT_XH_ATOMIC_NUMBERS,
    _set_encoded_model,
)
from payne_zero_atmosphere.warm_start import INITIALIZER_OUTPUT_FIELDS

LABELS = (
    "effective_temperature", "log_surface_gravity", "metallicity",
    "alpha_enhancement", "microturbulence_km_s",
    "carbon_enhancement", "nitrogen_enhancement", "oxygen_enhancement",
)
SEED = 20261005


def load_family(root: Path, family: str) -> dict:
    if family == "direct_abundance":
        paths = sorted((root / "direct_xh_experimental/training_corpora").rglob("*.npz"))
        pieces = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                pieces.append({key: data[key] for key in (
                    "stellar_labels", "abundance_vectors", "atmosphere_profiles",
                    "parent_group_ids", "identity_sha256",
                )})
                tau = data["standard_rosseland_optical_depth"][0].copy()
        joined = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
        stellar = joined["stellar_labels"]
        abundance = joined["abundance_vectors"]
        iron = abundance[:, 26 - 3]
        features = np.column_stack((
            5040.0 / stellar[:, 0], stellar[:, 1], stellar[:, 2], iron,
            *[abundance[:, z - 3] - iron for z in DIRECT_XH_ATOMIC_NUMBERS if z != 26],
        ))
        return dict(features=features, teff=stellar[:, 0], profiles=joined["atmosphere_profiles"],
                    groups=joined["parent_group_ids"], identities=joined["identity_sha256"],
                    tau=tau, stellar_labels=stellar, abundance_vectors=abundance,
                    sources=[str(path.relative_to(root)) for path in paths])
    name = "strict_truth_52199.npz" if family == "five_label" else "strict_truth_53824.npz"
    path = root / family / name
    with np.load(path, allow_pickle=False) as data:
        rows = [json.loads(str(row)) for row in data["labels_json"]]
        fields = LABELS[:5 if family == "five_label" else 8]
        labels = np.asarray([[row[field] for field in fields] for row in rows])
        features = labels.copy()
        features[:, 0] = 5040.0 / labels[:, 0]
        groups = data["slugs"].copy() if family == "five_label" else data["parent_group_ids"].copy()
        return dict(features=features, labels=labels, teff=labels[:, 0],
                    profiles=data["atmosphere_profiles"].copy(), groups=groups,
                    identities=data["slugs"].copy(), tau=data["standard_rosseland_optical_depth"][0].copy(),
                    sources=[str(path.relative_to(root))])


def make_split(groups: np.ndarray) -> dict:
    unique, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.random.default_rng(SEED).permutation(len(unique))
    roles = np.full(len(unique), "unused", dtype="U10")
    cursor = 0
    for role, target in (("test", 5000), ("validation", 5000), ("train", 42199)):
        count = 0
        while count < target and cursor < len(order):
            group = order[cursor]
            roles[group] = role
            count += int(counts[group])
            cursor += 1
    row_roles = roles[inverse]
    return {role: np.flatnonzero(row_roles == role).tolist()
            for role in ("train", "validation", "test", "unused")}


def encode(profiles: np.ndarray, teff: np.ndarray, tau: np.ndarray, scale: float) -> np.ndarray:
    increments = np.diff(profiles[:, :, 0], axis=1, prepend=0.0)
    if np.any(increments <= 0.0):
        raise ValueError("corpus contains nonpositive column-mass increments")
    coordinates = np.empty_like(profiles, dtype=np.float64)
    coordinates[:, :, 0] = np.log10(increments)
    grey = teff[:, None] * (0.75 * (tau[None, :] + 2.0 / 3.0)) ** 0.25
    coordinates[:, :, 1] = np.log10(profiles[:, :, 1] / grey)
    coordinates[:, :, 2:5] = np.log10(profiles[:, :, 2:5])
    coordinates[:, :, 5] = np.arcsinh(profiles[:, :, 5] / scale)
    return coordinates


def decode(coordinates: np.ndarray, teff: np.ndarray, tau: np.ndarray, scale: float) -> np.ndarray:
    profiles = np.empty_like(coordinates)
    profiles[:, :, 0] = np.cumsum(10.0 ** np.clip(coordinates[:, :, 0], -30.0, 30.0), axis=1)
    grey = teff[:, None] * (0.75 * (tau[None, :] + 2.0 / 3.0)) ** 0.25
    profiles[:, :, 1] = grey * 10.0 ** np.clip(coordinates[:, :, 1], -3.0, 3.0)
    if profiles.shape[-1] == 6:
        profiles[:, :, 2:5] = 10.0 ** np.clip(coordinates[:, :, 2:5], -30.0, 30.0)
        profiles[:, :, 5] = scale * np.sinh(np.clip(coordinates[:, :, 5], -20.0, 20.0))
    return profiles


def build_model(config: dict) -> nn.Module:
    if config.get("architecture") == "set_encoded":
        return _set_encoded_model(config)
    layers = []
    dimension = config["input_dim"]
    for _ in range(config["hidden_layers"]):
        layers.extend((nn.Linear(dimension, config["width"]), nn.SiLU()))
        dimension = config["width"]
    layers.append(nn.Linear(dimension, config["output_dim"]))
    return nn.Sequential(*layers)


def train(data: dict, split: dict, reference: dict, fields: int, args) -> None:
    train_index = np.asarray(split["train"])
    val_index = np.asarray(split["validation"])
    test_index = np.asarray(split["test"])
    scale = max(float(np.median(np.abs(data["profiles"][train_index, :, 5]))), 1.0e-8)
    coordinates = encode(data["profiles"], data["teff"], data["tau"], scale)[:, :, :fields]
    flat = coordinates.reshape(len(coordinates), -1)
    mean, std = flat[train_index].mean(0), flat[train_index].std(0)
    std = np.maximum(std, 1.0e-8)
    standardized = (flat - mean) / std
    # Two profiles already have 160 coordinates; retain them without compression.
    if fields == 2:
        basis = np.eye(160)
    else:
        covariance = standardized[train_index].T @ standardized[train_index] / len(train_index)
        _, eigenvectors = np.linalg.eigh(covariance)
        basis = eigenvectors[:, ::-1][:, :160].T.copy()
    coefficients = standardized @ basis.T
    coefficient_mean = coefficients[train_index].mean(0)
    coefficient_std = np.maximum(coefficients[train_index].std(0), 1.0e-8)
    x_mean, x_std = data["features"][train_index].mean(0), data["features"][train_index].std(0)
    x_std = np.maximum(x_std, 1.0e-8)
    features = (data["features"] - x_mean) / x_std
    config = dict(reference["model"]["config"])
    torch.manual_seed(SEED)
    model = build_model(config).to(args.device)
    x = torch.as_tensor(features, dtype=torch.float32, device=args.device)
    y = torch.as_tensor(standardized, dtype=torch.float32, device=args.device)
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=args.device)
    c_mean_t = torch.as_tensor(coefficient_mean, dtype=torch.float32, device=args.device)
    c_std_t = torch.as_tensor(coefficient_std, dtype=torch.float32, device=args.device)

    def profile_loss(batch):
        predicted = ((model(x[batch]) * c_std_t + c_mean_t) @ basis_t).reshape(-1, 80, fields)
        target = y[batch].reshape(-1, 80, fields)
        values = torch.nn.functional.smooth_l1_loss(predicted, target)
        gradient = torch.nn.functional.smooth_l1_loss(
            predicted[:, 1:] - predicted[:, :-1], target[:, 1:] - target[:, :-1])
        return values + 0.1 * gradient
    train_t = torch.as_tensor(train_index, device=args.device)
    val_t = torch.as_tensor(val_index, device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator(device=args.device).manual_seed(SEED)
    history, best_loss, best_epoch, best_state = [], float("inf"), -1, None
    started = time.monotonic()
    for epoch in range(args.epochs):
        epoch_start = time.monotonic()
        model.train()
        order = train_t[torch.randperm(len(train_t), generator=generator, device=args.device)]
        running = 0.0
        for batch in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = profile_loss(batch)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(batch)
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_loss = sum(float(profile_loss(batch)) * len(batch)
                           for batch in val_t.split(args.batch_size)) / len(val_t)
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(dict(epoch=epoch, train_loss=running / len(train_t),
                            validation_loss=val_loss, seconds=time.monotonic() - epoch_start))
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(json.dumps(dict(family=args.family, profiles=fields, **history[-1])), flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    pca = dict(basis=basis, coordinate_mean=mean, coordinate_std=std,
               coefficient_mean=coefficient_mean, coefficient_std=coefficient_std)
    checkpoint = dict(format="coauthor_matched_profiles_20261005", family=args.family,
                      profile_fields=INITIALIZER_OUTPUT_FIELDS[:fields],
                      model=dict(config=config, state_dict=best_state),
                      labels=dict(mean=x_mean, std=x_std), pca=pca,
                      coordinates=dict(standard_rosseland_optical_depth=data["tau"], acceleration_scale=scale),
                      split=split, training=dict(best_epoch=best_epoch, validation_loss=best_loss))
    destination = args.out / args.family
    torch.save(checkpoint, destination / f"checkpoint_{fields}profile.pt")
    with torch.no_grad():
        predicted = np.concatenate([model(x[batch]).cpu().numpy()
                                    for batch in torch.as_tensor(test_index, device=args.device).split(args.batch_size)])
    predicted_flat = ((predicted * coefficient_std + coefficient_mean) @ basis) * std + mean
    profiles = decode(predicted_flat.reshape(-1, 80, fields), data["teff"][test_index], data["tau"], scale)
    np.savez_compressed(destination / f"predicted_{fields}profile_test.npz",
                        star_indices=test_index, profiles=profiles, features=data["features"][test_index])
    error = np.abs(profiles[:, :, :2] - data["profiles"][test_index, :, :2])
    temperature_error = error[:, :, 1] / data["profiles"][test_index, :, 1]
    mass_error = np.abs(np.log10(profiles[:, :, 0] / data["profiles"][test_index, :, 0]))
    report = dict(family=args.family, profiles=fields, host=socket.gethostname(), device=args.device,
                  config=config, counts={key: len(value) for key, value in split.items()},
                  learning_rate=args.learning_rate, batch_size=args.batch_size,
                  epochs=args.epochs, seed=SEED,
                  loss="decoded standardized profile smooth-L1 + 0.1 adjacent-layer smooth-L1",
                  selection="minimum validation profile loss", best_epoch=best_epoch,
                  best_validation_loss=best_loss, elapsed_seconds=time.monotonic() - started,
                  temperature_relative_p95=float(np.percentile(temperature_error, 95)),
                  column_mass_dex_p95=float(np.percentile(mass_error, 95)), history=history)
    (destination / f"training_{fields}profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "history"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--family", choices=("five_label", "cno8", "direct_abundance"), default="five_label")
    parser.add_argument("--out", type=Path, default=Path("results/coauthor_validation_20261005"))
    parser.add_argument("--reference-root", type=Path, default=Path("source_data_files/atmosphere_emulator"))
    parser.add_argument("--profiles", type=int, choices=(2, 6), default=2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = load_family(args.corpus_root, args.family)
    destination = args.out / args.family
    destination.mkdir(parents=True, exist_ok=True)
    split_path = destination / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text())["indices"]
    else:
        split = make_split(data["groups"])
        split_path.write_text(json.dumps(dict(seed=SEED, grouping="parent group within each family",
                                             sources=data["sources"], counts={key: len(value) for key, value in split.items()},
                                             indices=split), indent=2) + "\n")
    print(json.dumps(dict(family=args.family, counts={key: len(value) for key, value in split.items()})), flush=True)
    reference = torch.load(args.reference_root / args.family / "checkpoint.pt", map_location="cpu", weights_only=False)
    if args.prepare_only:
        scale = float(reference["coordinates"]["acceleration_scale"])
        restored = decode(encode(data["profiles"], data["teff"], data["tau"], scale), data["teff"], data["tau"], scale)
        relative = np.max(np.abs(restored - data["profiles"]) / np.maximum(np.abs(data["profiles"]), 1.0e-20))
        print(json.dumps(dict(coordinate_roundtrip_max_relative=float(relative), input_dim=data["features"].shape[1])), flush=True)
        assert relative < 1.0e-10
        return
    train(data, split, reference, args.profiles, args)


if __name__ == "__main__":
    main()
