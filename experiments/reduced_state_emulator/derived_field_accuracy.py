"""How accurate are P, n_e, kappa_R and g_rad when (m,T) is *predicted*, not truth?

The sufficiency figure compares two ways of obtaining the four dependent fields:
the shipped six-field network predicting them directly, and deriving them from
**truth** (m,T) through the certified physics. The second is an oracle -- truth
(m,T) is not available at prediction time -- so it settles whether (m,T) carries
the information, not whether the reduced state is the better product.

This measures the deployable path: the trained network's (m,T), through the same
reconstruction, against the same truth. It is the third curve the figure needs.

There is a concrete reason to expect it to be worse rather than better, and the
measurement exists to find out. Hydrostatic equilibrium makes P essentially an
algebraic function of m (`hydrostatic.py:25`), so P inherits the predicted m's
error directly: 1.52e-2 dex p95 is about 3.6% relative, against the six-field
network's 1.8% median error on P itself. If the derived fields do come out
worse while the solver still converges faster, that is not a contradiction --
it is the sharpest form of Ting's Sec 2.2 result, that profile accuracy and
basin placement are different quantities.

No torch in this process: every atmosphere is built inside a worker. See
``truth_arms.py`` for the deadlock this avoids.

Usage::

    export NUMBA_THREADING_LAYER=workqueue
    PYTHONPATH=. .venv/bin/python -m experiments.reduced_state_emulator.derived_field_accuracy \\
        --workers 28
"""

from __future__ import annotations

# Must precede any Numba import, matching bench/run_reference.py.
from bench import environment as _environment  # noqa: F401,E402

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "source_data_files"
    / "atmosphere_emulator"
    / "five_label"
    / "strict_truth_52199.npz"
)
LABEL_FIELDS = (
    "effective_temperature",
    "log_surface_gravity",
    "metallicity",
    "alpha_enhancement",
    "microturbulence_km_s",
)
PROFILE_FIELDS = (
    "column_mass",
    "temperature",
    "gas_pressure",
    "electron_density",
    "rosseland_opacity",
    "radiative_acceleration",
)
DERIVED_FIELDS = PROFILE_FIELDS[2:]
N_SYNCHRONIZATIONS = 3
G_RAD_SCALE = 2.0577175465785027


def _worker(payload):
    from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere

    position, star_index, column_mass, temperature, label_dict = payload
    try:
        result = reconstruct_full_atmosphere(
            ReducedAtmosphere(
                column_mass=column_mass, temperature=temperature, labels=label_dict
            ),
            n_synchronizations=None,
            max_synchronizations=8,
            pressure_tolerance_dex=1.0e-3,
        )
    except Exception as exc:
        return {
            "position": int(position),
            "star_index": int(star_index),
            "fields": None,
            "failure": {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "pressure_change_dex_by_pass": [
                    float(value)
                    for value in getattr(exc, "pressure_change_dex_by_pass", ())
                ],
            },
        }
    atmosphere = result.atmosphere
    return {
        "position": int(position),
        "star_index": int(star_index),
        "fields": {
            field: np.asarray(getattr(atmosphere, field))
            for field in DERIVED_FIELDS
        },
        "failure": None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--prediction",
        type=Path,
        default=REPO_ROOT
        / "artifacts"
        / "reduced_state_emulator"
        / "predicted_monotone.npz",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "learned_reduced_state_derived_errors.npz",
    )
    args = parser.parse_args(argv)

    with np.load(args.prediction, allow_pickle=False) as data:
        held_out = data["star_indices"]
        predicted_column_mass = data["column_mass"]
        predicted_temperature = data["temperature"]
    with np.load(args.corpus, allow_pickle=False) as data:
        labels_json = [json.loads(str(e)) for e in data["labels_json"][held_out]]
        profiles = np.asarray(data["atmosphere_profiles"][held_out], dtype=np.float64)
    label_dicts = [
        {field: float(entry[field]) for field in LABEL_FIELDS} for entry in labels_json
    ]

    if args.count is not None:
        held_out = held_out[: args.count]
        predicted_column_mass = predicted_column_mass[: args.count]
        predicted_temperature = predicted_temperature[: args.count]
        profiles = profiles[: args.count]
        label_dicts = label_dicts[: args.count]
    print(f"{len(label_dicts)} stars, reconstructing from *predicted* (m,T)", flush=True)

    payloads = [
        (
            i,
            held_out[i],
            predicted_column_mass[i],
            predicted_temperature[i],
            label_dicts[i],
        )
        for i in range(len(label_dicts))
    ]
    if args.workers <= 1:
        results = [_worker(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(_worker, payloads))

    successes = [row for row in results if row["failure"] is None]
    failures = [
        {
            "position": row["position"],
            "star_index": row["star_index"],
            **row["failure"],
        }
        for row in results
        if row["failure"] is not None
    ]
    if not successes:
        raise SystemExit("no predicted atmosphere could be rematerialized")
    if failures:
        print(
            f"{len(failures)}/{len(results)} stars failed reconstruction and "
            "are excluded from the field statistics:",
            flush=True,
        )
        for failure in failures:
            print(
                f"  star {failure['star_index']}: {failure['error_type']}: "
                f"{failure['error']}",
                flush=True,
            )
    success_positions = np.asarray(
        [row["position"] for row in successes], dtype=np.int64
    )

    errors = {}
    for index, field in enumerate(DERIVED_FIELDS, start=2):
        predicted = np.array([row["fields"][field] for row in successes])
        truth = profiles[success_positions, :, index]
        difference = np.abs(predicted - truth)
        errors[f"{field}_relative_error"] = difference / np.maximum(
            np.abs(truth), 1.0e-300
        )
        if field == "radiative_acceleration":
            errors[f"{field}_normalized_error"] = difference / np.maximum(
                np.abs(truth), G_RAD_SCALE
            )
        else:
            errors[f"{field}_log10_error"] = np.abs(
                np.log10(np.maximum(predicted, 1.0e-300))
                - np.log10(np.maximum(truth, 1.0e-300))
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        requested_star_indices=held_out,
        star_indices=held_out[success_positions],
        failed_star_indices=np.asarray(
            [row["star_index"] for row in failures], dtype=np.int64
        ),
        success_positions=success_positions,
        **errors,
    )
    print(f"wrote {args.out}", flush=True)

    metrics = {}
    print(f"\n{'field':>26s} {'median':>10s} {'p90':>10s}")
    for field in DERIVED_FIELDS:
        metric_name = f"{field}_relative_error"
        values = errors[metric_name]
        metrics[field] = {
            "metric": metric_name,
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
        }
        if field == "radiative_acceleration":
            normalized = errors[f"{field}_normalized_error"]
            metrics[field]["normalized_median"] = float(np.median(normalized))
            metrics[field]["normalized_p90"] = float(
                np.percentile(normalized, 90)
            )
        print(
            f"{field:>26s} {metrics[field]['median']:>10.2e} "
            f"{metrics[field]['p90']:>10.2e}"
        )
    summary = {
        "requested_count": int(len(held_out)),
        "successful_count": int(len(successes)),
        "failure_count": int(len(failures)),
        "failures": failures,
        "synchronization_mode": "adaptive",
        "max_synchronizations": 8,
        "pressure_tolerance_dex": 1.0e-3,
        "metrics": metrics,
        "npz": str(args.out),
    }
    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
