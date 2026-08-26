"""Apply a label-conditioned solver-in-loop adapter to a corpus manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reduced_state.emulator import (
    decode_physical_state,
    label_features,
    load_corpus,
    load_physical_checkpoint,
)
from reduced_state.solver_adapter import LabelRbfAdapter


DEFAULT_ADAPTER = Path(
    "artifacts/reduced_state_emulator/solver_in_loop_k1_hard5_rbf/adapter.pt"
)
DEFAULT_MANIFEST = Path("results/sealed_solver_subset_20260808.json")
DEFAULT_OUT = Path("results/solver_in_loop_k1_hard5_rbf/predicted_dev60.npz")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--indices-from", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=Path(
        "source_data_files/atmosphere_emulator/five_label/strict_truth_52199.npz"
    ))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    model, standardization, _ = load_physical_checkpoint(adapter["base_checkpoint"])
    indices = np.asarray(
        json.loads(args.indices_from.read_text())["star_indices"], dtype=np.int64
    )
    corpus = load_corpus(args.corpus)
    labels = corpus["labels"][indices]
    standardized = (
        label_features(labels) - standardization.feature_mean
    ) / standardization.feature_std
    features = torch.as_tensor(standardized, dtype=torch.float64)
    if adapter.get("adapter_kind") != "rbf_label_local":
        raise ValueError("prediction script expects an rbf_label_local adapter")
    adapter_model = LabelRbfAdapter(
        torch.as_tensor(adapter["labels"]), sigma=adapter["rbf_sigma"]
    )
    # Stored centers are standardized; state loading replaces the constructor buffer.
    adapter_model.load_state_dict(adapter["adapter_state"])
    with torch.no_grad():
        raw_temperature, raw_mass = model(features)
        base_mass, base_temperature = decode_physical_state(
            raw_temperature, raw_mass, standardization,
            torch.as_tensor(labels[:, 0]),
        )
        temperature_offset, mass_offset = adapter_model(features)
        mass, temperature = decode_physical_state(
            raw_temperature + temperature_offset,
            raw_mass + mass_offset, standardization,
            torch.as_tensor(labels[:, 0]),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        star_indices=indices,
        labels=labels,
        column_mass=mass.numpy(),
        temperature=temperature.numpy(),
        truth_column_mass=corpus["column_mass"][indices],
        truth_temperature=corpus["temperature"][indices],
        base_column_mass=base_mass.numpy(),
        base_temperature=base_temperature.numpy(),
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
