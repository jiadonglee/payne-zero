"""Adaptive temperature continuation for cool-star solver starts.

The direct 500 K attempt is useful as a diagnostic, but it should not be the
only way to reach a cool target.  This runner starts with the requested step
and, after a failed solver attempt, halves the step and retries from the last
converged atmosphere.  For ``reduced_rematerialized`` the retry carries only
``(m,T)`` and rebuilds the other four fields through the exact physical
reconstruction path; no target-temperature emulator prediction is used.

Example::

    PYTHONPATH=. python -m experiments.reduced_state_emulator.cool_star_adaptive \
        --mode reduced_rematerialized --initial-step 500 --minimum-step 50 \
        --out-root runs/cool_star_adaptive_g5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.labels import StellarLabels

from .cool_star_step_test import (
    ANCHOR_TEMPERATURE,
    TrackSpec,
    _atmosphere_quality,
    _production_atmosphere,
    _reconstruct_from_mt,
    _retarget_full_state,
    _set_single_thread_environment,
    _solve_attempt,
)
from .cool_star_targeted_3700 import _model_from_product


def proposed_temperature(
    current_temperature: float,
    target_temperature: float,
    step_temperature: float,
) -> float:
    """Return the next temperature without overshooting the target."""

    current = float(current_temperature)
    target = float(target_temperature)
    step = float(step_temperature)
    if step <= 0.0:
        raise ValueError("step_temperature must be positive")
    if target == current:
        return target
    direction = 1.0 if target > current else -1.0
    return target if direction * (target - current) <= step else current + direction * step


def backtrack_step(step_temperature: float, minimum_step: float) -> float:
    """Halve a failed step, respecting the configured minimum."""

    step = float(step_temperature)
    minimum = float(minimum_step)
    if step <= 0.0 or minimum <= 0.0:
        raise ValueError("step_temperature and minimum_step must be positive")
    if minimum > step:
        raise ValueError("minimum_step cannot exceed step_temperature")
    return max(minimum, 0.5 * step)


def _run_adaptive(
    *,
    track: TrackSpec,
    mode: str,
    anchor_temperature: float,
    target_temperature: float,
    initial_step: float,
    minimum_step: float,
    iteration_cap: int,
    out_root: Path,
    anchor_product: Path | None = None,
) -> dict:
    if mode not in {"full_carry", "reduced_rematerialized"}:
        raise ValueError(f"unsupported mode: {mode}")
    if initial_step <= 0.0 or minimum_step <= 0.0:
        raise ValueError("temperature steps must be positive")
    if minimum_step > initial_step:
        raise ValueError("minimum_step cannot exceed initial_step")

    _set_single_thread_environment()
    out_root = Path(out_root)
    print(
        f"[adaptive] anchor {anchor_temperature:.0f} K -> target "
        f"{target_temperature:.0f} K, mode={mode}, "
        f"initial_step={initial_step:g} K",
        flush=True,
    )
    anchor_labels = track.labels(anchor_temperature)
    if anchor_product is not None:
        anchor_product = Path(anchor_product)
        if not anchor_product.is_file():
            raise FileNotFoundError(anchor_product)
        current = _model_from_product(anchor_product, anchor_labels)
        quality = _atmosphere_quality(current)
        anchor_record = {
            "method": "adaptive_anchor_reused",
            "schedule": "adaptive_anchor",
            "labels": anchor_labels.as_kwargs(),
            "source_temperature": None,
            "target_temperature": float(anchor_temperature),
            "iterations": 0,
            "seconds": 0.0,
            "converged": bool(quality["valid"]),
            "solver_converged": bool(quality["valid"]),
            "product_path": str(anchor_product),
            "product_written": True,
            "state_quality": quality,
            "survives_solver": bool(quality["valid"]),
            "status": "reused_anchor_product",
        }
        print(f"[adaptive] reused anchor product {anchor_product}", flush=True)
    else:
        anchor_record, current = _solve_attempt(
            track=track,
            method="adaptive_anchor_production",
            schedule="adaptive_anchor",
            source_temperature=None,
            target_labels=anchor_labels,
            initial_atmosphere=_production_atmosphere(anchor_labels),
            product_dir=out_root / "products" / "anchor_production",
            iteration_cap=iteration_cap,
        )
    attempts: list[dict] = []
    if current is None or not anchor_record["survives_solver"]:
        print("[adaptive] anchor failed", flush=True)
        return {
            "track": track.as_json(),
            "mode": mode,
            "anchor_temperature": float(anchor_temperature),
            "target_temperature": float(target_temperature),
            "initial_step": float(initial_step),
            "minimum_step": float(minimum_step),
            "anchor": anchor_record,
            "attempts": attempts,
            "reached_target": False,
            "accepted_steps": [],
            "status": "anchor_failed",
        }

    current_temperature = float(anchor_temperature)
    step_temperature = float(initial_step)
    accepted_steps: list[float] = []
    attempt_index = 0
    tolerance = 1.0e-9
    while abs(current_temperature - float(target_temperature)) > tolerance:
        proposed = proposed_temperature(
            current_temperature, target_temperature, step_temperature
        )
        labels: StellarLabels = track.labels(proposed)
        try:
            if mode == "full_carry":
                seed = _retarget_full_state(
                    current,
                    _production_atmosphere(labels),
                )
            else:
                seed = _reconstruct_from_mt(
                    labels,
                    current.column_mass,
                    current.temperature,
                )
        except Exception as exc:  # noqa: BLE001 - recorded as a failed attempt
            record = {
                "status": "initialization_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "source_temperature": current_temperature,
                "target_temperature": proposed,
                "requested_step": step_temperature,
            }
            next_state = None
        else:
            attempt_index += 1
            record, next_state = _solve_attempt(
                track=track,
                method=f"adaptive_{mode}",
                schedule="adaptive",
                source_temperature=current_temperature,
                target_labels=labels,
                initial_atmosphere=seed,
                product_dir=out_root / "products" / mode,
                iteration_cap=iteration_cap,
            )
            record.update(
                {
                    "attempt_index": attempt_index,
                    "requested_step": step_temperature,
                    "source_temperature": current_temperature,
                    "target_temperature": proposed,
                }
            )

        attempts.append(record)
        print(
            f"[adaptive] attempt {attempt_index or len(attempts)}: "
            f"{current_temperature:.0f}->{proposed:.0f} K, "
            f"step={step_temperature:g} K, "
            f"status={record.get('status')}",
            flush=True,
        )
        if next_state is not None and record.get("survives_solver"):
            accepted_steps.append(abs(current_temperature - proposed))
            current = next_state
            current_temperature = proposed
            continue

        next_step = backtrack_step(step_temperature, minimum_step)
        if next_step >= step_temperature - tolerance:
            break
        print(
            f"[adaptive] backtrack {step_temperature:g} -> {next_step:g} K",
            flush=True,
        )
        step_temperature = next_step

    reached = abs(current_temperature - float(target_temperature)) <= tolerance
    return {
        "track": track.as_json(),
        "mode": mode,
        "anchor_temperature": float(anchor_temperature),
        "target_temperature": float(target_temperature),
        "initial_step": float(initial_step),
        "minimum_step": float(minimum_step),
        "anchor": anchor_record,
        "attempts": attempts,
        "accepted_steps": accepted_steps,
        "reached_target": reached,
        "status": "reached_target" if reached else "minimum_step_failed",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("full_carry", "reduced_rematerialized"), required=True
    )
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--anchor-product",
        type=Path,
        default=None,
        help="reuse an already converged structured anchor instead of solving it again",
    )
    parser.add_argument("--logg", type=float, default=5.0)
    parser.add_argument("--metallicity", type=float, default=0.0)
    parser.add_argument("--anchor-temperature", type=float, default=ANCHOR_TEMPERATURE)
    parser.add_argument("--target-temperature", type=float, default=3500.0)
    parser.add_argument("--initial-step", type=float, default=500.0)
    parser.add_argument("--minimum-step", type=float, default=50.0)
    parser.add_argument("--iteration-cap", type=int, default=30)
    args = parser.parse_args(argv)

    result = _run_adaptive(
        track=TrackSpec(
            log_surface_gravity=args.logg,
            metallicity=args.metallicity,
        ),
        mode=args.mode,
        anchor_temperature=args.anchor_temperature,
        target_temperature=args.target_temperature,
        initial_step=args.initial_step,
        minimum_step=args.minimum_step,
        iteration_cap=args.iteration_cap,
        out_root=args.out_root,
        anchor_product=args.anchor_product,
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["reached_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
