"""Profile-constrained K=1 training on the frozen dev-60 candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reduced_state.emulator import label_features, load_physical_checkpoint
from reduced_state.solver_adapter import LabelRbfAdapter
from payne_zero_diffatm.check_twin_correction import build_iteration
from payne_zero_diffatm.twin_loop import (
    reference_solver_step_template_from_iteration,
    solver_step,
)
from experiments.reduced_state_emulator.train_solver_in_loop import HARD_CASES


BASE_PREDICTION = Path(
    "results/goal_spectral_20260808/"
    "predicted_lowg_alpha050_refined_hotpoor_alpha100_probe60.npz"
)
COORDINATE_CHECKPOINT = Path(
    "artifacts/reduced_state_emulator/"
    "physical_truth_solver_tail_surface10_hard20_8x1024_full_cpu/"
    "checkpoint_physical_seed20260808.pt"
)
DEFAULT_OUT = Path(
    "artifacts/reduced_state_emulator/solver_in_loop_k1_frozen_rbf/adapter.pt"
)
DEFAULT_SOLVER_TARGET_PRODUCTS = Path(
    "runs/reduced_state_emulator/goal_spectral_20260808/"
    "products/production_six_field"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-prediction", type=Path, default=BASE_PREDICTION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3.0e-3)
    parser.add_argument("--profile-weight", type=float, default=0.5)
    parser.add_argument(
        "--hard-case-indices",
        default="0,1,2,3,4",
        help="comma-separated HARD_CASES positions used as RBF centers",
    )
    parser.add_argument("--rbf-sigma", type=float, default=0.30)
    parser.add_argument("--frequency-stride", type=int, default=431)
    parser.add_argument("--sweeps", type=int, default=12)
    parser.add_argument(
        "--solver-target-products-dir",
        type=Path,
        default=DEFAULT_SOLVER_TARGET_PRODUCTS,
        help="converged production atmospheres that define the accepted fixed point",
    )
    args = parser.parse_args()

    with np.load(args.base_prediction, allow_pickle=False) as data:
        all_labels = np.asarray(data["labels"], dtype=np.float64)
        all_mass = np.asarray(data["column_mass"], dtype=np.float64)
        all_temperature = np.asarray(data["temperature"], dtype=np.float64)
        all_truth_mass = np.asarray(data["truth_column_mass"], dtype=np.float64)
        all_truth_temperature = np.asarray(data["truth_temperature"], dtype=np.float64)
        all_indices = np.asarray(data["star_indices"], dtype=np.int64)

    hard_case_indices = [int(value) for value in args.hard_case_indices.split(",")]
    selected_hard_cases = tuple(HARD_CASES[index] for index in hard_case_indices)
    requested = np.asarray([[case.effective_temperature, case.log_surface_gravity,
                             case.metallicity, case.alpha_enhancement,
                             case.microturbulence_km_s] for case in selected_hard_cases])
    scale = np.asarray([3000.0, 2.0, 1.0, 0.3, 2.0])
    positions = np.asarray([
        np.argmin(np.sum(((all_labels - row) / scale) ** 2, axis=1))
        for row in requested
    ], dtype=np.int64)
    labels = all_labels[positions]
    cases = tuple(type(HARD_CASES[0])(*row) for row in labels)
    base_mass = torch.as_tensor(all_mass[positions])
    base_temperature = torch.as_tensor(all_temperature[positions])
    truth_mass = torch.as_tensor(all_truth_mass[positions])
    truth_temperature = torch.as_tensor(all_truth_temperature[positions])
    target_mass_rows = []
    target_temperature_rows = []
    for case in cases:
        product_path = args.solver_target_products_dir / f"{case.slug}.npz"
        with np.load(product_path, allow_pickle=False) as product:
            target_mass_rows.append(np.asarray(product["column_mass"], dtype=np.float64))
            target_temperature_rows.append(
                np.asarray(product["temperature"], dtype=np.float64)
            )
    solver_target_mass = torch.as_tensor(np.stack(target_mass_rows))
    solver_target_temperature = torch.as_tensor(np.stack(target_temperature_rows))

    _, coordinate_standardization, _ = load_physical_checkpoint(COORDINATE_CHECKPOINT)
    features_np = (
        label_features(labels) - coordinate_standardization.feature_mean
    ) / coordinate_standardization.feature_std
    features = torch.as_tensor(features_np, dtype=torch.float64)
    adapter = LabelRbfAdapter(features, sigma=args.rbf_sigma)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.learning_rate)
    print(f"building {len(cases)} certified iteration-1 templates", flush=True)
    templates = [
        reference_solver_step_template_from_iteration(build_iteration(case))
        for case in cases
    ]
    frequency_indices = torch.arange(
        0, templates[0].frequency_hz.numel(), args.frequency_stride
    )
    base_increment = torch.cat(
        [base_mass[:, :1], base_mass[:, 1:] - base_mass[:, :-1]], dim=1
    )
    history = []

    for epoch in range(args.epochs + 1):
        temperature_offset, increment_offset = adapter(features)
        initial_temperature = base_temperature * 10.0**temperature_offset.clamp(-0.2, 0.2)
        initial_increment = base_increment * 10.0**increment_offset.clamp(-0.2, 0.2)
        initial_mass = torch.cumsum(initial_increment, dim=1)
        solver_losses = []
        for index, template in enumerate(templates):
            step = solver_step(
                initial_mass[index], initial_temperature[index], template,
                frequency_indices=frequency_indices, sweeps=args.sweeps,
            )
            solver_losses.append(
                torch.mean((torch.log10(step.correction.temperature[0])
                            - torch.log10(solver_target_temperature[index])) ** 2)
                + torch.mean((torch.log10(step.correction.column_mass[0])
                              - torch.log10(solver_target_mass[index])) ** 2)
            )
        solver_loss = torch.stack(solver_losses).mean()
        profile_loss = (
            torch.mean((torch.log10(initial_temperature)
                        - torch.log10(truth_temperature)) ** 2)
            + torch.mean((torch.log10(initial_mass) - torch.log10(truth_mass)) ** 2)
        )
        loss = solver_loss + args.profile_weight * profile_loss
        history.append({
            "total": float(loss.detach()),
            "solver": float(solver_loss.detach()),
            "profile": float(profile_loss.detach()),
        })
        if epoch == args.epochs:
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in adapter.parameters()
        ):
            raise RuntimeError("non-finite physical adapter gradient")
        optimizer.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"epoch={epoch + 1:03d} total={history[-1]['total']:.6e} "
                f"solver={history[-1]['solver']:.6e} "
                f"profile={history[-1]['profile']:.6e}", flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "adapter_kind": "physical_rbf_solver_k1",
        "adapter_state": adapter.state_dict(),
        "rbf_sigma": args.rbf_sigma,
        "feature_mean": coordinate_standardization.feature_mean,
        "feature_std": coordinate_standardization.feature_std,
        "base_prediction": str(args.base_prediction),
        "solver_target_products_dir": str(args.solver_target_products_dir),
        "training_positions": positions,
        "training_star_indices": all_indices[positions],
        "training_labels": labels,
        "hard_case_indices": hard_case_indices,
        "profile_weight": args.profile_weight,
        "history": history,
    }, args.out)
    args.out.with_suffix(".json").write_text(json.dumps({
        "initial": history[0], "final": history[-1],
        "improvement_factor": history[0]["total"] / history[-1]["total"],
        "training_star_indices": all_indices[positions].tolist(),
    }, indent=2))
    print(
        f"PASS: total {history[0]['total']:.6e} -> {history[-1]['total']:.6e} "
        f"({history[0]['total'] / history[-1]['total']:.2f}x), saved {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
