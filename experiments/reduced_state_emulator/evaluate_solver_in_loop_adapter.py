"""Real-solver smoke comparison for the three-star K=1 adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reduced_state.emulator import (
    decode_physical_state,
    label_features,
    load_physical_checkpoint,
)
from reduced_state.restart import run_many_restarts
from reduced_state.solver_adapter import LabelRbfAdapter
from bench.labels import StellarLabels
from experiments.reduced_state_emulator.run_learned_restart import (
    reconstruct_predicted_safe,
)
from payne_zero_diffatm.check_twin_molecules import CASES


DEFAULT_ADAPTER = Path(
    "artifacts/reduced_state_emulator/solver_in_loop_k1_hard5_rbf/adapter.pt"
)
DEFAULT_OUT = Path(
    "results/solver_in_loop_k1_hard5_rbf/real_solver_comparison.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _records_summary(records, failures, requested):
    iterations = [
        int(record["converging_trial_iterations"])
        for record in records if record["converged"]
    ]
    return {
        "requested": requested,
        "reconstruction_failures": failures,
        "solver_records": records,
        "converged": int(sum(bool(record["converged"]) for record in records)),
        "mean_iterations_converged": (
            float(np.mean(iterations)) if iterations else None
        ),
    }


def main() -> int:
    args = parse_args()
    adapter = torch.load(args.adapter, map_location="cpu", weights_only=False)
    model, standardization, _ = load_physical_checkpoint(adapter["base_checkpoint"])
    model.eval()
    labels = np.asarray(adapter["labels"], dtype=np.float64)
    cases = tuple(StellarLabels(*row) for row in labels)
    standardized = (
        label_features(labels) - standardization.feature_mean
    ) / standardization.feature_std
    with torch.no_grad():
        feature_tensor = torch.as_tensor(standardized)
        raw_temperature, raw_mass = model(feature_tensor)
        if adapter.get("adapter_kind") == "rbf_label_local":
            adapter_model = LabelRbfAdapter(
                torch.as_tensor(standardized), sigma=adapter["rbf_sigma"]
            )
            adapter_model.load_state_dict(adapter["adapter_state"])
            temperature_offset, mass_offset = adapter_model(feature_tensor)
        elif adapter.get("adapter_kind") == "linear_label_conditioned":
            temperature_adapter = torch.nn.Linear(5, 80, dtype=torch.float64)
            mass_adapter = torch.nn.Linear(5, 80, dtype=torch.float64)
            temperature_adapter.load_state_dict(adapter["temperature_adapter_state"])
            mass_adapter.load_state_dict(adapter["mass_adapter_state"])
            temperature_offset = temperature_adapter(feature_tensor)
            mass_offset = mass_adapter(feature_tensor)
        else:
            temperature_offset = adapter["temperature_bias"][None, :]
            mass_offset = adapter["mass_bias"][None, :]
        base_mass, base_temperature = decode_physical_state(
            raw_temperature, raw_mass, standardization,
            torch.as_tensor(labels[:, 0]),
        )
        adapted_mass, adapted_temperature = decode_physical_state(
            raw_temperature + temperature_offset,
            raw_mass + mass_offset,
            standardization, torch.as_tensor(labels[:, 0]),
        )

    arms = {
        "base": (base_mass.numpy(), base_temperature.numpy()),
        "k1_adapter": (adapted_mass.numpy(), adapted_temperature.numpy()),
    }
    summary = {}
    for name, (mass, temperature) in arms.items():
        print(f"reconstructing {name}", flush=True)
        reconstructed, failures = reconstruct_predicted_safe(
            mass, temperature, labels, workers=1
        )
        items = [(cases[position], atmosphere) for position, atmosphere in reconstructed]
        print(f"running real solver for {name}", flush=True)
        records = run_many_restarts(
            items, workers=1, source=f"solver_in_loop_{name}",
            iterations_per_trial=30,
        )
        summary[name] = _records_summary(records, failures, len(cases))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    base = summary["base"]
    adapted = summary["k1_adapter"]
    print(
        "RESULT: base "
        f"{base['converged']}/{len(cases)} mean={base['mean_iterations_converged']}; "
        "adapter "
        f"{adapted['converged']}/{len(cases)} mean={adapted['mean_iterations_converged']}; "
        f"saved {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
