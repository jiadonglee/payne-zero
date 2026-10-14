"""Matched two- and six-profile M-giant networks for bootstrapping round 2.

The networks take the eight CNO labels and use the encoding, architecture and
loss of ``coauthor_train_20261005``. Training rows are the cno8 training split
plus the M-giant training rows; every batch draws half its rows from each
group, as for v4. Standardization and the six-profile PCA use the union of the
training rows without weighting. The checkpoint minimizes the M-giant
validation loss.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from payne_zero_atmosphere.warm_start import INITIALIZER_OUTPUT_FIELDS
from coauthor_train_20261005 import LABELS, build_model, decode, encode

SEED = 20261010


def load(corpus_root: Path, split_path: Path, mgiant: Path) -> dict:
    with np.load(corpus_root / "cno8/strict_truth_53824.npz", allow_pickle=False) as d:
        rows = [json.loads(str(row)) for row in d["labels_json"]]
        normal_labels = np.asarray([[row[f] for f in LABELS] for row in rows])
        normal_profiles = d["atmosphere_profiles"].copy()
        tau = d["standard_rosseland_optical_depth"][0].copy()
    split = json.loads(split_path.read_text())["indices"]
    with np.load(mgiant, allow_pickle=False) as d:
        if not np.allclose(d["standard_rosseland_optical_depth"], tau, rtol=1e-12, atol=0):
            raise ValueError("M-giant corpus uses a different optical-depth grid")
        cool_labels, cool_profiles, roles = d["labels"].copy(), d["atmosphere_profiles"].copy(), d["roles"].copy()
    offset = len(normal_labels)
    labels = np.concatenate((normal_labels, cool_labels))
    features = labels.copy()
    features[:, 0] = 5040.0 / labels[:, 0]
    return dict(labels=labels, features=features, teff=labels[:, 0], tau=tau,
                profiles=np.concatenate((normal_profiles, cool_profiles)),
                normal_train=np.asarray(split["train"]), normal_validation=np.asarray(split["validation"]),
                cool_train=offset + np.flatnonzero(roles == "train"),
                cool_validation=offset + np.flatnonzero(roles == "validation"))


def train(data: dict, fields: int, config: dict, args) -> None:
    train_index = np.concatenate((data["normal_train"], data["cool_train"]))
    scale = max(float(np.median(np.abs(data["profiles"][train_index, :, 5]))), 1.0e-8)
    coordinates = encode(data["profiles"], data["teff"], data["tau"], scale)[:, :, :fields]
    flat = coordinates.reshape(len(coordinates), -1)
    mean, std = flat[train_index].mean(0), np.maximum(flat[train_index].std(0), 1.0e-8)
    standardized = (flat - mean) / std
    if fields == 2:
        basis = np.eye(160)
    else:
        covariance = standardized[train_index].T @ standardized[train_index] / len(train_index)
        basis = np.linalg.eigh(covariance)[1][:, ::-1][:, :160].T.copy()
    coefficients = standardized @ basis.T
    coefficient_mean = coefficients[train_index].mean(0)
    coefficient_std = np.maximum(coefficients[train_index].std(0), 1.0e-8)
    x_mean, x_std = data["features"][train_index].mean(0), np.maximum(data["features"][train_index].std(0), 1.0e-8)
    device = args.device
    torch.manual_seed(SEED)
    model = build_model(config).to(device)
    x = torch.as_tensor((data["features"] - x_mean) / x_std, dtype=torch.float32, device=device)
    y = torch.as_tensor(standardized, dtype=torch.float32, device=device)
    basis_t, c_mean_t, c_std_t = (torch.as_tensor(a, dtype=torch.float32, device=device)
                                  for a in (basis, coefficient_mean, coefficient_std))

    def loss(batch):
        predicted = ((model(x[batch]) * c_std_t + c_mean_t) @ basis_t).reshape(-1, 80, fields)
        target = y[batch].reshape(-1, 80, fields)
        return (torch.nn.functional.smooth_l1_loss(predicted, target)
                + 0.1 * torch.nn.functional.smooth_l1_loss(predicted[:, 1:] - predicted[:, :-1],
                                                           target[:, 1:] - target[:, :-1]))

    def mean_loss(index):
        with torch.no_grad():
            t = torch.as_tensor(index, device=device)
            return sum(float(loss(b)) * len(b) for b in t.split(args.batch_size)) / len(t)

    normal_t = torch.as_tensor(data["normal_train"], device=device)
    cool_t = torch.as_tensor(data["cool_train"], device=device)
    half = args.batch_size // 2
    steps = len(normal_t) // half
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator(device=device).manual_seed(SEED)
    history, best, best_epoch, best_state = [], float("inf"), -1, None
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        order = normal_t[torch.randperm(len(normal_t), generator=generator, device=device)]
        for step in range(steps):
            cool = cool_t[torch.randint(len(cool_t), (half,), generator=generator, device=device)]
            batch = torch.cat((order[step * half:(step + 1) * half], cool))
            optimizer.zero_grad(set_to_none=True)
            loss(batch).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        record = dict(epoch=epoch, mgiant_validation_loss=mean_loss(data["cool_validation"]),
                      normal_validation_loss=mean_loss(data["normal_validation"]))
        history.append(record)
        if record["mgiant_validation_loss"] < best:
            best, best_epoch = record["mgiant_validation_loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(json.dumps(dict(profiles=fields, **record)), flush=True)
    model.load_state_dict(best_state)
    checkpoint = dict(format="coauthor_matched_profiles_20261005", family="mgiant_cno8",
                      profile_fields=INITIALIZER_OUTPUT_FIELDS[:fields],
                      model=dict(config=config, state_dict=best_state),
                      labels=dict(mean=x_mean, std=x_std, fields=LABELS),
                      pca=dict(basis=basis, coordinate_mean=mean, coordinate_std=std,
                               coefficient_mean=coefficient_mean, coefficient_std=coefficient_std),
                      coordinates=dict(standard_rosseland_optical_depth=data["tau"], acceleration_scale=scale),
                      training=dict(best_epoch=best_epoch, mgiant_validation_loss=best))
    torch.save(checkpoint, args.out / f"checkpoint_{fields}profile.pt")
    errors = {}
    for name in ("cool_validation", "normal_validation"):
        index = data[name]
        profiles = predict(checkpoint, data["labels"][index], device=device)
        truth = data["profiles"][index]
        errors[name] = dict(
            temperature_relative_p95=float(np.percentile(abs(profiles[:, :, 1] / truth[:, :, 1] - 1), 95)),
            column_mass_dex_p95=float(np.percentile(abs(np.log10(profiles[:, :, 0] / truth[:, :, 0])), 95)))
    report = dict(profiles=fields, host=socket.gethostname(), device=device, config=config, seed=SEED,
                  counts={k: int(len(data[k])) for k in ("normal_train", "normal_validation", "cool_train", "cool_validation")},
                  batch="half cno8 training rows without replacement, half M-giant training rows with replacement",
                  selection="minimum M-giant validation loss", best_epoch=best_epoch, best_mgiant_validation_loss=best,
                  learning_rate=args.learning_rate, batch_size=args.batch_size, epochs=args.epochs,
                  elapsed_seconds=time.monotonic() - started, errors=errors, history=history)
    (args.out / f"training_{fields}profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "history"}), flush=True)


def predict(checkpoint: dict, labels: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Decoded profiles, shape (n, 80, fields), for eight-label rows."""
    fields = len(checkpoint["profile_fields"])
    model = build_model(checkpoint["model"]["config"])
    model.load_state_dict(checkpoint["model"]["state_dict"])
    model.to(device).eval()
    features = np.asarray(labels, dtype=float).copy()
    features[:, 0] = 5040.0 / features[:, 0]
    features = (features - checkpoint["labels"]["mean"]) / checkpoint["labels"]["std"]
    with torch.no_grad():
        output = model(torch.as_tensor(features, dtype=torch.float32, device=device)).cpu().numpy()
    pca = checkpoint["pca"]
    flat = ((output * pca["coefficient_std"] + pca["coefficient_mean"]) @ pca["basis"]) * pca["coordinate_std"] + pca["coordinate_mean"]
    coordinates = checkpoint["coordinates"]
    return decode(flat.reshape(-1, 80, fields), np.asarray(labels)[:, 0],
                  coordinates["standard_rosseland_optical_depth"], coordinates["acceleration_scale"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="cno8 split.json of the coauthor validation")
    parser.add_argument("--mgiant", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="cno8 reference checkpoint (architecture)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profiles", type=int, choices=(2, 6), required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    data = load(args.corpus_root, args.split, args.mgiant)
    config = dict(torch.load(args.reference, map_location="cpu", weights_only=False)["model"]["config"])
    train(data, args.profiles, config, args)


if __name__ == "__main__":
    main()
