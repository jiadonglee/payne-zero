"""Apply a physical RBF solver adapter to its frozen prediction artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from reduced_state.emulator import label_features
from reduced_state.solver_adapter import LabelRbfAdapter


DEFAULT_ADAPTER = Path(
    "artifacts/reduced_state_emulator/solver_in_loop_k1_frozen_rbf/adapter.pt"
)
DEFAULT_OUT = Path(
    "results/solver_in_loop_k1_frozen_rbf/predicted_dev60.npz"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument(
        "--base-prediction",
        type=Path,
        default=None,
        help="compatible prediction artifact; defaults to the adapter's training base",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    payload = torch.load(args.adapter, map_location="cpu", weights_only=False)
    base_prediction = args.base_prediction or Path(payload["base_prediction"])
    with np.load(base_prediction, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    labels = np.asarray(arrays["labels"], dtype=np.float64)
    features = torch.as_tensor(
        (label_features(labels) - payload["feature_mean"]) / payload["feature_std"]
    )
    centers = payload["adapter_state"]["centers"]
    adapter = LabelRbfAdapter(centers, sigma=payload["rbf_sigma"])
    adapter.load_state_dict(payload["adapter_state"])
    base_mass = torch.as_tensor(arrays["column_mass"])
    base_temperature = torch.as_tensor(arrays["temperature"])
    base_increment = torch.cat(
        [base_mass[:, :1], base_mass[:, 1:] - base_mass[:, :-1]], dim=1
    )
    with torch.no_grad():
        temperature_offset, increment_offset = adapter(features)
        offset_limit = float(payload.get("offset_limit", 0.2))
        temperature = base_temperature * 10.0**temperature_offset.clamp(
            -offset_limit, offset_limit
        )
        increment = base_increment * 10.0**increment_offset.clamp(
            -offset_limit, offset_limit
        )
        mass = torch.cumsum(increment, dim=1)
    arrays["base_column_mass"] = arrays["column_mass"]
    arrays["base_temperature"] = arrays["temperature"]
    arrays["column_mass"] = mass.numpy()
    arrays["temperature"] = temperature.numpy()

    # How much of the input did this adapter actually reach?  An RBF whose
    # bandwidth is small relative to the spacing of the stars it is applied to
    # returns weights that decay to nothing, leaving the base prediction
    # untouched while still appearing in the provenance chain as a component.
    # Reporting the reach here makes that visible when the adapter is applied,
    # rather than after a gate has already been run against it.
    mass_move = np.max(
        np.abs(
            np.log10(np.clip(arrays["column_mass"], 1e-300, None))
            - np.log10(np.clip(arrays["base_column_mass"], 1e-300, None))
        ),
        axis=1,
    )
    coverage = {
        "star_count": int(mass_move.size),
        "stars_moved_over_1e-6_dex": int(np.sum(mass_move > 1.0e-6)),
        "stars_moved_over_1e-3_dex": int(np.sum(mass_move > 1.0e-3)),
        "median_move_dex": float(np.median(mass_move)),
        "max_move_dex": float(np.max(mass_move)),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    args.out.with_suffix(".json").write_text(json.dumps({
        "adapter": str(args.adapter),
        "adapter_sha256": _sha256(args.adapter),
        "base_prediction": str(base_prediction),
        "base_prediction_sha256": _sha256(base_prediction),
        "prediction": str(args.out),
        "prediction_sha256": _sha256(args.out),
        "adapter_kind": payload["adapter_kind"],
        "rbf_sigma": payload["rbf_sigma"],
        "offset_limit": offset_limit,
        "hard_case_indices": payload.get("hard_case_indices"),
        "star_count": int(labels.shape[0]),
        "centre_count": int(centers.shape[0]),
        "coverage": coverage,
    }, indent=2) + "\n")
    print(args.out)
    print(
        f"  reach: {coverage['stars_moved_over_1e-6_dex']}/{coverage['star_count']} "
        f"stars moved >1e-6 dex, median {coverage['median_move_dex']:.3e}, "
        f"max {coverage['max_move_dex']:.3e} "
        f"({int(centers.shape[0])} centres, sigma={payload['rbf_sigma']})",
        flush=True,
    )
    if coverage["stars_moved_over_1e-6_dex"] == 0:
        print(
            "  WARNING: this adapter changed nothing on this prediction set. "
            "It will appear in the provenance chain without affecting any "
            "result.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
