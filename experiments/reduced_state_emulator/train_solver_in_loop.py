"""K=1 solver-in-loop adapter training on certified one-step templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bench.labels import StellarLabels
from reduced_state.emulator import (
    decode_physical_state,
    label_features,
    load_physical_checkpoint,
)
from reduced_state.solver_adapter import LabelRbfAdapter
from payne_zero_diffatm.check_twin_correction import build_iteration
from payne_zero_diffatm.check_twin_molecules import CASES
from payne_zero_diffatm.twin_loop import (
    reference_solver_step_template_from_iteration,
    solver_step,
)


DEFAULT_CHECKPOINT = Path(
    "artifacts/reduced_state_emulator/"
    "physical_truth_solver_tail_surface10_hard20_8x1024_full_cpu/"
    "checkpoint_physical_seed20260808.pt"
)
DEFAULT_OUT = Path(
    "artifacts/reduced_state_emulator/solver_in_loop_k1_hard5_rbf/adapter.pt"
)
HARD_CASES = (
    StellarLabels(7650.8, 1.79, -1.93, 0.36, 2.06),
    StellarLabels(4995.8, 2.97, 0.19, -0.04, 3.48),
    StellarLabels(5203.4, 2.84, 0.40, 0.05, 3.38),
    StellarLabels(4058.7, 3.10, -1.71, -0.04, 1.64),
    StellarLabels(5339.4, 1.99, 0.45, -0.08, 1.78),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--frequency-stride", type=int, default=431)
    parser.add_argument("--sweeps", type=int, default=12)
    parser.add_argument("--cohort", choices=("hard5", "canonical3"), default="hard5")
    parser.add_argument("--rbf-sigma", type=float, default=0.35)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(20260811)
    model, standardization, meta = load_physical_checkpoint(args.checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cases = HARD_CASES if args.cohort == "hard5" else CASES
    labels = np.asarray(
        [[
            case.effective_temperature,
            case.log_surface_gravity,
            case.metallicity,
            case.alpha_enhancement,
            case.microturbulence_km_s,
        ] for case in cases],
        dtype=np.float64,
    )
    features = label_features(labels)
    standardized = (
        features - standardization.feature_mean
    ) / standardization.feature_std
    with torch.no_grad():
        base_temperature, base_mass = model(torch.as_tensor(standardized))

    print(f"building {len(cases)} certified iteration-1 templates", flush=True)
    templates = [
        reference_solver_step_template_from_iteration(build_iteration(case))
        for case in cases
    ]
    adapter_model = LabelRbfAdapter(
        torch.as_tensor(standardized), sigma=args.rbf_sigma
    )
    adapter_parameters = list(adapter_model.parameters())
    optimizer = torch.optim.Adam(adapter_parameters, lr=args.learning_rate)
    history = []

    frequency_indices = torch.arange(
        0, templates[0].frequency_hz.numel(), args.frequency_stride
    )
    effective_temperature = torch.as_tensor(labels[:, 0], dtype=torch.float64)

    for epoch in range(args.epochs + 1):
        feature_tensor = torch.as_tensor(standardized, dtype=torch.float64)
        temperature_offset, mass_offset = adapter_model(feature_tensor)
        raw_temperature = base_temperature + temperature_offset
        raw_mass = base_mass + mass_offset
        predicted_mass, predicted_temperature = decode_physical_state(
            raw_temperature, raw_mass, standardization, effective_temperature
        )
        losses = []
        for star_index, template in enumerate(templates):
            step = solver_step(
                predicted_mass[star_index], predicted_temperature[star_index],
                template, frequency_indices=frequency_indices, sweeps=args.sweeps,
            )
            target_temperature = template.correction.output_temperature
            target_mass = template.correction.output_column_mass
            temperature_loss = torch.mean(
                (torch.log10(step.correction.temperature[0])
                 - torch.log10(target_temperature)) ** 2
            )
            mass_loss = torch.mean(
                (torch.log10(step.correction.column_mass[0])
                 - torch.log10(target_mass)) ** 2
            )
            losses.append(temperature_loss + mass_loss)
        loss = torch.stack(losses).mean()
        history.append(float(loss.detach()))
        if epoch == args.epochs:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in adapter_parameters
        ):
            raise RuntimeError("non-finite solver-in-loop adapter gradient")
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"epoch={epoch + 1:03d} loss={history[-1]:.6e}", flush=True)

    if not history[-1] < history[0]:
        raise RuntimeError(
            f"solver-in-loop smoke loss did not decrease: {history[0]} -> {history[-1]}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state": adapter_model.state_dict(),
            "adapter_kind": "rbf_label_local",
            "rbf_sigma": args.rbf_sigma,
            "base_checkpoint": str(args.checkpoint),
            "base_meta": meta,
            "labels": labels,
            "history": history,
            "frequency_stride": args.frequency_stride,
            "sweeps": args.sweeps,
            "training_kind": f"solver_in_loop_k1_{args.cohort}_adapter",
        },
        args.out,
    )
    summary = args.out.with_suffix(".json")
    summary.write_text(json.dumps({
        "initial_loss": history[0],
        "final_loss": history[-1],
        "improvement_factor": history[0] / history[-1],
        "epochs": args.epochs,
        "checkpoint": str(args.out),
    }, indent=2))
    print(
        f"PASS: K=1 loss {history[0]:.6e} -> {history[-1]:.6e} "
        f"({history[0] / history[-1]:.2f}x), saved {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
