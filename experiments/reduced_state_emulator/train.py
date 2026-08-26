"""Train the learned `labels -> (m, T)` initializer and report profile accuracy.

Part 2 answered "is `(m, T)` sufficient" with truth profiles. This trains the
predictor that would have to supply them in production, and measures its error
against the two scales that are known to matter:

* Ting's basin calibration (`solver-in-the-loop-prior-work.md` Sec 2.2) found
  temperature p95 near 3e-3 and column-mass p95 near 7.7e-3 dex convergent
  *marginally*, and the same two errors **coupled** failing at 30 iterations on
  both labels tested. So p95 error is reported in exactly those units, and
  passing them is necessary but explicitly not sufficient.
* The production six-field emulator's own `(m, T)`, evaluated on the same
  held-out stars. If a two-field network cannot beat the six-field one at the
  two fields they share, the reduced state buys nothing.

The 60 stars used by Part 2/3 are excluded from training, so the downstream
solver comparison is on genuinely unseen labels.

Usage::

    export NUMBA_THREADING_LAYER=workqueue
    .venv/bin/python -m experiments.reduced_state_emulator.train \\
        --epochs 150 --out artifacts/reduced_state_emulator
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from reduced_state.emulator import (
    LABEL_FIELDS,
    ReducedStateEmulator,
    Standardization,
    decode_column_mass,
    decode_temperature,
    fit_standardization,
    label_features,
    load_corpus,
    save_checkpoint,
)

DEFAULT_CORPUS = Path(
    "source_data_files/atmosphere_emulator/five_label/strict_truth_52199.npz"
)
DEFAULT_HELD_OUT = Path("results/reconstruction_metrics.json")


def _percentile_report(error: np.ndarray) -> dict:
    """Per-star maxima summarized across stars: the shape gates are stated in."""

    per_star_p95 = np.percentile(error, 95, axis=1)
    per_star_max = error.max(axis=1)
    return {
        "median_overall": float(np.median(error)),
        "p95_overall": float(np.percentile(error, 95)),
        "max_overall": float(error.max()),
        "per_star_p95_median": float(np.median(per_star_p95)),
        "per_star_p95_p90": float(np.percentile(per_star_p95, 90)),
        "per_star_max_median": float(np.median(per_star_max)),
    }


def evaluate(
    model: ReducedStateEmulator,
    standardization: Standardization,
    labels: np.ndarray,
    column_mass: np.ndarray,
    temperature: np.ndarray,
) -> dict:
    features = label_features(labels)
    standardized = (
        features - standardization.feature_mean
    ) / standardization.feature_std
    model.eval()
    with torch.no_grad():
        raw_temperature, raw_column_mass = model(
            torch.as_tensor(standardized, dtype=torch.float64)
        )
        log_temperature = decode_temperature(raw_temperature, standardization).numpy()
        log_column_mass = decode_column_mass(
            raw_column_mass, standardization, monotone=model.monotone
        ).numpy()

    predicted_temperature = np.power(10.0, log_temperature)
    temperature_relative = np.abs(predicted_temperature - temperature) / temperature
    column_mass_dex = np.abs(log_column_mass - np.log10(column_mass))
    monotone_violations = int((np.diff(np.power(10.0, log_column_mass), axis=1) <= 0).any(axis=1).sum())

    return {
        "star_count": int(len(labels)),
        "temperature_relative_error": _percentile_report(temperature_relative),
        "column_mass_dex_error": _percentile_report(column_mass_dex),
        "monotonicity_violations": monotone_violations,
    }


def evaluate_production_emulator(
    labels: np.ndarray, column_mass: np.ndarray, temperature: np.ndarray
) -> dict:
    """The six-field production initializer's own (m, T) on the same stars."""

    from payne_zero_atmosphere.warm_start import emulator_warm_start_model

    predicted_column_mass = np.empty_like(column_mass)
    predicted_temperature = np.empty_like(temperature)
    for index, row in enumerate(labels):
        atmosphere, _deck = emulator_warm_start_model(
            device="cpu", **dict(zip(LABEL_FIELDS, (float(v) for v in row)))
        )
        predicted_column_mass[index] = atmosphere.column_mass
        predicted_temperature[index] = atmosphere.temperature

    temperature_relative = np.abs(predicted_temperature - temperature) / temperature
    column_mass_dex = np.abs(
        np.log10(predicted_column_mass) - np.log10(column_mass)
    )
    return {
        "star_count": int(len(labels)),
        "temperature_relative_error": _percentile_report(temperature_relative),
        "column_mass_dex_error": _percentile_report(column_mass_dex),
    }


def train_one(
    corpus: dict,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    *,
    monotone: bool,
    epochs: int,
    width: int,
    depth: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[ReducedStateEmulator, Standardization, list[dict]]:
    torch.manual_seed(seed)

    features = label_features(corpus["labels"])
    log_column_mass = np.log10(corpus["column_mass"])
    log_temperature = np.log10(corpus["temperature"])

    standardization = fit_standardization(
        features[train_index],
        log_column_mass[train_index],
        log_temperature[train_index],
    )

    def tensors(index):
        standardized = (
            features[index] - standardization.feature_mean
        ) / standardization.feature_std
        target_temperature = (
            log_temperature[index] - standardization.log_temperature_mean
        ) / standardization.log_temperature_std
        target_column_mass = (
            log_column_mass[index] - standardization.log_column_mass_mean
        ) / standardization.log_column_mass_std
        return (
            torch.as_tensor(standardized, dtype=torch.float64),
            torch.as_tensor(target_temperature, dtype=torch.float64),
            torch.as_tensor(target_column_mass, dtype=torch.float64),
            torch.as_tensor(log_column_mass[index], dtype=torch.float64),
        )

    train_x, train_t, train_m, train_logm = tensors(train_index)
    val_x, val_t, val_m, val_logm = tensors(validation_index)

    column_mass_mean = torch.as_tensor(
        standardization.log_column_mass_mean, dtype=torch.float64
    )
    column_mass_std = torch.as_tensor(
        standardization.log_column_mass_std, dtype=torch.float64
    )

    model = ReducedStateEmulator(width=width, depth=depth, monotone=monotone)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    generator = np.random.default_rng(seed)
    history: list[dict] = []

    def losses(x, target_t, target_m, target_logm):
        raw_temperature, raw_column_mass = model(x)
        temperature_loss = torch.mean((raw_temperature - target_t) ** 2)
        if monotone:
            predicted_logm = decode_column_mass(
                raw_column_mass, standardization, monotone=True
            )
            # Compare in standardized space so the two loss terms share a scale.
            column_mass_loss = torch.mean(
                ((predicted_logm - column_mass_mean) / column_mass_std - target_m) ** 2
            )
        else:
            column_mass_loss = torch.mean((raw_column_mass - target_m) ** 2)
        return temperature_loss, column_mass_loss

    n_train = len(train_index)
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(n_train)
        running = 0.0
        for start in range(0, n_train, batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad()
            temperature_loss, column_mass_loss = losses(
                train_x[batch], train_t[batch], train_m[batch], train_logm[batch]
            )
            loss = temperature_loss + column_mass_loss
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(batch)
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_temperature_loss, val_column_mass_loss = losses(
                    val_x, val_t, val_m, val_logm
                )
            entry = {
                "epoch": epoch,
                "train_loss": running / n_train,
                "val_temperature_loss": float(val_temperature_loss),
                "val_column_mass_loss": float(val_column_mass_loss),
            }
            history.append(entry)
            print(
                f"  epoch {epoch:4d}  train {entry['train_loss']:.4e}  "
                f"val_T {entry['val_temperature_loss']:.4e}  "
                f"val_m {entry['val_column_mass_loss']:.4e}",
                flush=True,
            )

    return model, standardization, history


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--held-out-from", type=Path, default=DEFAULT_HELD_OUT)
    parser.add_argument("--out", type=Path, default=Path("artifacts/reduced_state_emulator"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--skip-production-baseline",
        action="store_true",
        help="the six-field emulator comparison loads the release checkpoint per star",
    )
    args = parser.parse_args(argv)

    print(f"loading {args.corpus}", flush=True)
    corpus = load_corpus(args.corpus)
    total = len(corpus["labels"])

    held_out = np.array(
        json.loads(args.held_out_from.read_text())["star_indices"], dtype=np.int64
    )
    print(
        f"{total} stars; holding out {len(held_out)} Part-2/3 evaluation stars",
        flush=True,
    )

    available = np.setdiff1d(np.arange(total), held_out)
    roles = corpus["roles"][available]
    train_index = available[roles == "train"]
    validation_index = available[roles != "train"]
    print(
        f"train {len(train_index)}  validation {len(validation_index)}  "
        f"held-out {len(held_out)}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "corpus": str(args.corpus),
        "star_count": int(total),
        "train_count": int(len(train_index)),
        "validation_count": int(len(validation_index)),
        "held_out_indices": [int(i) for i in held_out],
        "epochs": args.epochs,
        "width": args.width,
        "depth": args.depth,
        "seed": args.seed,
        "arms": {},
    }

    for monotone in (True, False):
        name = "monotone" if monotone else "direct"
        print(f"=== training arm: {name} ===", flush=True)
        start = time.perf_counter()
        model, standardization, history = train_one(
            corpus,
            train_index,
            validation_index,
            monotone=monotone,
            epochs=args.epochs,
            width=args.width,
            depth=args.depth,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        elapsed = time.perf_counter() - start

        arm = {
            "training_seconds": elapsed,
            "history": history,
            "validation": evaluate(
                model,
                standardization,
                corpus["labels"][validation_index],
                corpus["column_mass"][validation_index],
                corpus["temperature"][validation_index],
            ),
            "held_out": evaluate(
                model,
                standardization,
                corpus["labels"][held_out],
                corpus["column_mass"][held_out],
                corpus["temperature"][held_out],
            ),
        }
        summary["arms"][name] = arm
        save_checkpoint(
            args.out / f"checkpoint_{name}.pt",
            model,
            standardization,
            {"seed": args.seed, "epochs": args.epochs, "held_out": summary["held_out_indices"]},
        )
        print(
            f"  held-out  T p95 {arm['held_out']['temperature_relative_error']['p95_overall']:.3e}"
            f"  m p95 {arm['held_out']['column_mass_dex_error']['p95_overall']:.3e} dex"
            f"  monotonicity violations {arm['held_out']['monotonicity_violations']}",
            flush=True,
        )

    if not args.skip_production_baseline:
        print("=== production six-field emulator on the same held-out stars ===", flush=True)
        summary["production_six_field"] = evaluate_production_emulator(
            corpus["labels"][held_out],
            corpus["column_mass"][held_out],
            corpus["temperature"][held_out],
        )
        print(
            f"  held-out  T p95 "
            f"{summary['production_six_field']['temperature_relative_error']['p95_overall']:.3e}"
            f"  m p95 "
            f"{summary['production_six_field']['column_mass_dex_error']['p95_overall']:.3e} dex",
            flush=True,
        )

    out_path = Path("results/reduced_state_emulator_training.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
