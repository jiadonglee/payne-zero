"""Fit a label-local profile rescue adapter on already opened evaluation sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reduced_state.emulator import label_features, load_physical_checkpoint
from reduced_state.solver_adapter import LabelRbfAdapter


COORDINATE_CHECKPOINT = Path(
    "artifacts/reduced_state_emulator/"
    "physical_truth_solver_tail_surface10_hard20_8x1024_full_cpu/"
    "checkpoint_physical_seed20260808.pt"
)


def _increments(column_mass: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [column_mass[:, :1], np.diff(column_mass, axis=1)], axis=1
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rbf-sigma", type=float, default=0.20)
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--offset-limit", type=float, default=1.0)
    parser.add_argument("--temperature-limit", type=float, default=0.10)
    parser.add_argument("--mass-limit-dex", type=float, default=0.10)
    parser.add_argument("--include-star-indices", default="")
    parser.add_argument("--teacher-prediction", type=Path, default=None)
    parser.add_argument("--teacher-star-indices", default="")
    args = parser.parse_args()

    forced = {
        int(value) for value in args.include_star_indices.split(",") if value
    }
    teacher_indices = {
        int(value) for value in args.teacher_star_indices.split(",") if value
    }
    teacher_rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if args.teacher_prediction is not None:
        with np.load(args.teacher_prediction, allow_pickle=False) as teacher:
            indices = np.asarray(teacher["star_indices"], dtype=np.int64)
            for position, star_index in enumerate(indices):
                if int(star_index) in teacher_indices:
                    teacher_rows[int(star_index)] = (
                        np.asarray(teacher["column_mass"][position]),
                        np.asarray(teacher["temperature"][position]),
                    )
        missing = teacher_indices - set(teacher_rows)
        if missing:
            raise SystemExit(f"teacher stars missing from prediction: {sorted(missing)}")
    selected: dict[int, dict[str, np.ndarray]] = {}
    source_failures: dict[str, list[int]] = {}
    for prediction_path in args.prediction:
        with np.load(prediction_path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        temperature_error = np.max(
            np.abs(arrays["temperature"] - arrays["truth_temperature"])
            / arrays["truth_temperature"], axis=1
        )
        mass_error = np.max(
            np.abs(np.log10(arrays["column_mass"])
                   - np.log10(arrays["truth_column_mass"])), axis=1
        )
        indices = np.asarray(arrays["star_indices"], dtype=np.int64)
        mask = (
            (temperature_error >= args.temperature_limit)
            | (mass_error >= args.mass_limit_dex)
            | np.isin(indices, list(forced))
        )
        source_failures[str(prediction_path)] = indices[mask].tolist()
        for position in np.flatnonzero(mask):
            star_index = int(indices[position])
            selected[star_index] = {
                "labels": arrays["labels"][position],
                "column_mass": arrays["column_mass"][position],
                "temperature": arrays["temperature"][position],
                "truth_column_mass": arrays["truth_column_mass"][position],
                "truth_temperature": arrays["truth_temperature"][position],
            }
    if not selected:
        raise SystemExit("no rescue centres selected")

    star_indices = np.asarray(sorted(selected), dtype=np.int64)
    rows = [selected[int(index)] for index in star_indices]
    labels = np.stack([row["labels"] for row in rows])
    base_mass = np.stack([row["column_mass"] for row in rows])
    base_temperature = np.stack([row["temperature"] for row in rows])
    truth_mass = np.stack([row["truth_column_mass"] for row in rows])
    truth_temperature = np.stack([row["truth_temperature"] for row in rows])
    for position, star_index in enumerate(star_indices):
        if int(star_index) in teacher_rows:
            truth_mass[position], truth_temperature[position] = teacher_rows[
                int(star_index)
            ]

    _, standardization, _ = load_physical_checkpoint(COORDINATE_CHECKPOINT)
    features = (
        label_features(labels) - standardization.feature_mean
    ) / standardization.feature_std
    centers = torch.as_tensor(features, dtype=torch.float64)
    adapter = LabelRbfAdapter(centers, sigma=args.rbf_sigma)

    target_temperature_offset = torch.as_tensor(
        np.log10(truth_temperature / base_temperature), dtype=torch.float64
    )
    target_increment_offset = torch.as_tensor(
        np.log10(_increments(truth_mass) / _increments(base_mass)),
        dtype=torch.float64,
    )
    distance_squared = torch.sum(
        (centers[:, None, :] - centers[None, :, :]) ** 2, dim=-1
    )
    weights = torch.exp(-0.5 * distance_squared / args.rbf_sigma**2)
    system = weights + args.ridge * torch.eye(weights.shape[0], dtype=torch.float64)
    with torch.no_grad():
        adapter.temperature_offsets.copy_(
            torch.linalg.solve(system, target_temperature_offset)
        )
        adapter.mass_offsets.copy_(
            torch.linalg.solve(system, target_increment_offset)
        )

    predicted_temperature_offset, predicted_increment_offset = adapter(centers)
    residual = max(
        float(torch.max(torch.abs(
            predicted_temperature_offset - target_temperature_offset
        )).detach()),
        float(torch.max(torch.abs(
            predicted_increment_offset - target_increment_offset
        )).detach()),
    )
    if residual > 1.0e-5:
        raise RuntimeError(f"RBF centre fit residual {residual:.3e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "adapter_kind": "profile_rescue_rbf",
        "adapter_state": adapter.state_dict(),
        "rbf_sigma": args.rbf_sigma,
        "offset_limit": args.offset_limit,
        "feature_mean": standardization.feature_mean,
        "feature_std": standardization.feature_std,
        "base_prediction": str(args.prediction[0]),
        "training_star_indices": star_indices,
        "training_labels": labels,
        "source_failures": source_failures,
        "forced_star_indices": sorted(forced),
        "teacher_prediction": (
            str(args.teacher_prediction) if args.teacher_prediction else None
        ),
        "teacher_star_indices": sorted(teacher_indices),
        "ridge": args.ridge,
        "centre_fit_max_residual": residual,
    }, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({
        "adapter_kind": "profile_rescue_rbf",
        "training_star_indices": star_indices.tolist(),
        "source_failures": source_failures,
        "forced_star_indices": sorted(forced),
        "teacher_prediction": (
            str(args.teacher_prediction) if args.teacher_prediction else None
        ),
        "teacher_star_indices": sorted(teacher_indices),
        "rbf_sigma": args.rbf_sigma,
        "offset_limit": args.offset_limit,
        "centre_fit_max_residual": residual,
    }, indent=2) + "\n")
    print(
        f"saved {args.out}: {len(star_indices)} centres, "
        f"fit residual={residual:.3e}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
