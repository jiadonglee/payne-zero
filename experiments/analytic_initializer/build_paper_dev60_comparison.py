"""Build the same-sample paper comparison of learned and analytic initializers.

This script does not run the atmosphere solver.  It joins the frozen
development-60 learned-emulator products to a parity-formula solver run on the
same corpus indices, evaluates both initial profiles against the same truth
rows, and writes one JSON summary plus one NPZ used by the manuscript figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.analytic_initializer.compact_initializer import (
    PARITY_CONFIGURATION,
    fit_compact_profile_parameters,
    load_compact_profile_parameters,
    predict_compact_reduced_state,
)
from experiments.analytic_initializer.discovery import (
    DEFAULT_CORPUS,
    collect_excluded_indices,
    load_strict_truth,
    make_split,
)
from experiments.analytic_initializer.run_h2_solver_funnel import (
    ARM_CANDIDATES,
    ITERATIONS_PER_TRIAL,
    MANIFESTS,
)


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LEARNED_SUMMARY = (
    REPO
    / "results"
    / "paper_physical_seed_20260820"
    / "learned"
    / "convergence_metrics_learned_monotone.json"
)
DEFAULT_LEARNED_PREDICTION = (
    REPO / "artifacts" / "reduced_state_emulator" / "predicted_monotone.npz"
)
DEFAULT_LEARNED_RECORDS = (
    REPO
    / "runs"
    / "paper_physical_seed_20260820"
    / "learned"
    / "records"
    / "learned_reduced_state"
    / "records.jsonl"
)
DEFAULT_LEARNED_TRAINING = (
    REPO / "results" / "reduced_state_emulator_training.json"
)
DEFAULT_LEARNED_CHECKPOINT = (
    REPO / "artifacts" / "reduced_state_emulator" / "checkpoint_monotone.pt"
)
DEFAULT_ANALYTIC_PARAMETERS = (
    REPO
    / "results"
    / "analytic_initializer"
    / "compact_profile_parameters_parity.npz"
)
DEFAULT_ANALYTIC_FRONTIER = (
    REPO / "results" / "analytic_initializer" / "compact_frontier.json"
)
DEFAULT_ANALYTIC_SOLVER = (
    REPO / "results" / "analytic_initializer" / "paper_dev60_parity.json"
)
DEFAULT_OUTPUT = (
    REPO
    / "results"
    / "paper_physical_seed_20260820"
    / "analytic"
    / "paper_dev60_comparison.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _solver_slug(row: np.ndarray) -> str:
    """Match the public five-label formatting in ``bench.labels``."""

    return (
        f"t{row[0]:07.1f}_g{row[1]:+05.2f}_m{row[2]:+05.2f}"
        f"_a{row[3]:+05.2f}_x{row[4]:04.2f}"
    )


def _profile_errors(
    mass: np.ndarray,
    temperature: np.ndarray,
    truth_mass: np.ndarray,
    truth_temperature: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    temperature_relative = np.abs(temperature / truth_temperature - 1.0)
    mass_dex = np.abs(np.log10(mass) - np.log10(truth_mass))
    summary = {
        "temperature_relative_p50": float(np.percentile(temperature_relative, 50.0)),
        "temperature_relative_p95": float(np.percentile(temperature_relative, 95.0)),
        "column_mass_dex_p50": float(np.percentile(mass_dex, 50.0)),
        "column_mass_dex_p95": float(np.percentile(mass_dex, 95.0)),
    }
    return summary, temperature_relative, mass_dex


def _solver_summary(
    converged: np.ndarray,
    iterations: np.ndarray,
    *,
    timeout_count: int = 0,
    error_count: int = 0,
) -> dict[str, float | int]:
    values = iterations[converged]
    if values.size == 0:
        raise ValueError("solver comparison contains no converged stars")
    return {
        "star_count": int(converged.size),
        "converged_count": int(converged.sum()),
        "converged_fraction": float(converged.mean()),
        "failure_count": int((~converged).sum()),
        "timeout_count": int(timeout_count),
        "error_count": int(error_count),
        "mean_iterations_converged": float(values.mean()),
        "median_iterations_converged": float(np.median(values)),
        "p90_iterations_converged": float(np.percentile(values, 90.0)),
    }


def _network_parameter_count(*, labels: int, layers: int, width: int, depth: int) -> int:
    """Count the fitted weights and biases in the recorded two-head MLP."""

    hidden = (labels + 1) * width + (depth - 1) * (width + 1) * width
    heads = 2 * (width + 1) * layers
    return int(hidden + heads)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=REPO / DEFAULT_CORPUS)
    parser.add_argument("--learned-summary", type=Path, default=DEFAULT_LEARNED_SUMMARY)
    parser.add_argument(
        "--learned-prediction", type=Path, default=DEFAULT_LEARNED_PREDICTION
    )
    parser.add_argument("--learned-records", type=Path, default=DEFAULT_LEARNED_RECORDS)
    parser.add_argument("--learned-training", type=Path, default=DEFAULT_LEARNED_TRAINING)
    parser.add_argument(
        "--learned-checkpoint", type=Path, default=DEFAULT_LEARNED_CHECKPOINT
    )
    parser.add_argument(
        "--analytic-parameters", type=Path, default=DEFAULT_ANALYTIC_PARAMETERS
    )
    parser.add_argument(
        "--analytic-frontier", type=Path, default=DEFAULT_ANALYTIC_FRONTIER
    )
    parser.add_argument("--analytic-solver", type=Path, default=DEFAULT_ANALYTIC_SOLVER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    source_paths = {
        "corpus": args.corpus,
        "learned_summary": args.learned_summary,
        "learned_prediction": args.learned_prediction,
        "learned_records": args.learned_records,
        "learned_training": args.learned_training,
        "learned_checkpoint": args.learned_checkpoint,
        "analytic_parameters": args.analytic_parameters,
        "analytic_frontier": args.analytic_frontier,
        "analytic_solver": args.analytic_solver,
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("missing comparison source(s): " + ", ".join(missing))

    corpus = load_strict_truth(args.corpus)
    learned_summary = _read_json(args.learned_summary)
    if learned_summary.get("arm") != "monotone":
        raise ValueError("learned summary is not the monotone two-field arm")
    fixed_three = int(learned_summary.get("n_synchronizations", -1)) == 3
    adaptive_physical = (
        learned_summary.get("synchronization_mode") == "adaptive"
        and int(learned_summary.get("max_synchronizations", -1)) == 8
        and np.isclose(
            float(learned_summary.get("pressure_tolerance_dex", np.nan)),
            1.0e-3,
            rtol=0.0,
            atol=0.0,
        )
    )
    if not (fixed_three or adaptive_physical):
        raise ValueError(
            "learned summary uses neither the recorded three-pass protocol "
            "nor the adaptive physical-seed protocol"
        )
    if Path(learned_summary.get("prediction_artifact", "")).name != (
        args.learned_prediction.name
    ):
        raise ValueError("learned summary names a different prediction artifact")
    indices = np.asarray(learned_summary["star_indices"], dtype=np.int64)
    if indices.ndim != 1 or indices.size != 60 or np.unique(indices).size != indices.size:
        raise ValueError("learned summary must name 60 unique development indices")
    if int(learned_summary.get("star_count", -1)) != indices.size:
        raise ValueError("learned summary star count disagrees with its index list")

    with np.load(args.learned_prediction, allow_pickle=False) as data:
        prediction_indices = np.asarray(data["star_indices"], dtype=np.int64)
        if not np.array_equal(prediction_indices, indices):
            raise ValueError("learned prediction uses a different star order")
        learned_mass = np.asarray(data["column_mass"], dtype=np.float64)
        learned_temperature = np.asarray(data["temperature"], dtype=np.float64)
        prediction_labels = np.asarray(data["labels"], dtype=np.float64)
        learned_truth_mass = np.asarray(data["truth_column_mass"], dtype=np.float64)
        learned_truth_temperature = np.asarray(data["truth_temperature"], dtype=np.float64)

    labels = corpus.labels[indices]
    truth_mass = corpus.column_mass[indices]
    truth_temperature = corpus.temperature[indices]
    for name, left, right in (
        ("labels", prediction_labels, labels),
        ("truth column mass", learned_truth_mass, truth_mass),
        ("truth temperature", learned_truth_temperature, truth_temperature),
    ):
        if not np.allclose(left, right, rtol=0.0, atol=0.0):
            raise ValueError(f"learned prediction {name} disagrees with the corpus")

    parameters = load_compact_profile_parameters(args.analytic_parameters)
    if not np.array_equal(
        parameters.temperature.feature_center, parameters.opacity.feature_center
    ) or not np.array_equal(
        parameters.temperature.feature_scale, parameters.opacity.feature_scale
    ):
        raise ValueError("analytic closures do not share the recorded label scaling")
    shared_label_scaling = 2 * int(parameters.temperature.feature_center.size)
    with np.load(args.analytic_parameters, allow_pickle=False) as data:
        serialized_float_entries = int(
            sum(
                np.asarray(data[key]).size
                for key in data.files
                if np.issubdtype(np.asarray(data[key]).dtype, np.floating)
            )
        )
        serialized_integer_entries = int(
            sum(
                np.asarray(data[key]).size
                for key in data.files
                if np.issubdtype(np.asarray(data[key]).dtype, np.integer)
            )
        )
    if (
        serialized_float_entries
        != parameters.stored_float_count + shared_label_scaling
    ):
        raise ValueError(
            "analytic NPZ float count is not the distinct count plus its "
            "duplicated shared normalization"
        )
    frontier = _read_json(args.analytic_frontier)
    analytic_asset = frontier["assets"]["parity"]
    if int(analytic_asset["stored_floats"]) != parameters.stored_float_count:
        raise ValueError("analytic stored-float count disagrees with frontier record")
    excluded, _ = collect_excluded_indices(MANIFESTS, corpus_size=corpus.size)
    analytic_split = make_split(corpus.size, excluded=excluded, seed=20260816)
    refitted_parameters = fit_compact_profile_parameters(
        corpus, analytic_split, configuration=PARITY_CONFIGURATION
    )
    closure_pairs = (
        (parameters.temperature, refitted_parameters.temperature),
        (parameters.opacity, refitted_parameters.opacity),
    )
    for frozen, refitted in closure_pairs:
        for field in (
            "center_by_regime",
            "modes_by_regime",
            "coefficients_by_regime",
            "feature_center",
            "feature_scale",
        ):
            if not np.array_equal(getattr(frozen, field), getattr(refitted, field)):
                raise ValueError(
                    f"deterministic parity refit differs from the frozen asset in {field}"
                )
    analytic_mass, analytic_temperature, _ = predict_compact_reduced_state(
        labels, corpus.tau, parameters
    )

    learned_profile, learned_temperature_error, learned_mass_error = _profile_errors(
        learned_mass, learned_temperature, truth_mass, truth_temperature
    )
    analytic_profile, analytic_temperature_error, analytic_mass_error = _profile_errors(
        analytic_mass, analytic_temperature, truth_mass, truth_temperature
    )

    raw_learned_records = _read_jsonl(args.learned_records)
    learned_by_slug = {
        str(record["slug"]): record for record in raw_learned_records
    }
    expected_slugs = [_solver_slug(row) for row in labels]
    if len(set(expected_slugs)) != indices.size:
        raise ValueError("development-60 labels do not have unique solver slugs")
    unexpected_slugs = set(learned_by_slug) - set(expected_slugs)
    if unexpected_slugs:
        raise ValueError(
            f"learned solver records contain unexpected stars: "
            f"{sorted(unexpected_slugs)}"
        )
    learned_records = [learned_by_slug.get(slug) for slug in expected_slugs]
    missing_rows = [
        row for row, record in enumerate(learned_records) if record is None
    ]
    reconstruction = learned_summary.get(
        "learned_reduced_state_reconstruction", {}
    )
    reconstruction_failures = reconstruction.get("failures", [])
    if int(reconstruction.get("requested_count", indices.size)) != indices.size:
        raise ValueError("learned reconstruction requested a different sample")
    if int(reconstruction.get("synchronized_count", len(learned_by_slug))) != len(
        learned_by_slug
    ):
        raise ValueError("learned reconstruction count disagrees with solver records")
    if int(reconstruction.get("failure_count", len(missing_rows))) != len(
        missing_rows
    ):
        raise ValueError("learned reconstruction failures do not match missing records")
    if reconstruction and len(reconstruction_failures) != len(missing_rows):
        raise ValueError("learned reconstruction failure table is incomplete")
    expected_missing_indices = {
        int(failure["star_index"]) for failure in reconstruction_failures
    }
    actual_missing_indices = {int(indices[row]) for row in missing_rows}
    if expected_missing_indices != actual_missing_indices:
        raise ValueError("learned reconstruction failure indices disagree with records")
    learned_converged = np.asarray(
        [
            False if record is None else bool(record["converged"])
            for record in learned_records
        ],
        dtype=bool,
    )
    learned_iterations = np.asarray(
        [
            np.nan
            if record is None
            or record.get("converging_trial_iterations") is None
            else float(record["converging_trial_iterations"])
            for record in learned_records
        ],
        dtype=np.float64,
    )
    for row, record in enumerate(learned_records):
        if record is None:
            continue
        if int(record.get("trials_used", 0)) != 1:
            raise ValueError(f"learned solver record {row} did not use one trial")
        if len(record.get("trials", [])) != 1:
            raise ValueError(f"learned solver record {row} has the wrong trial count")
        trial = record["trials"][0]
        if (
            trial.get("trial_index") != 0
            or trial.get("initializer_label", {}).get("source")
            != "learned_reduced_state"
        ):
            raise ValueError(f"learned solver record {row} has wrong initializer source")
        completed = trial.get("iterations_completed")
        if not isinstance(completed, int) or not 1 <= completed <= ITERATIONS_PER_TRIAL:
            raise ValueError(f"learned solver record {row} has invalid iterations")
        if bool(record["converged"]) != bool(trial.get("converged")):
            raise ValueError(f"learned solver record {row} has inconsistent convergence")
        if int(record.get("total_iterations", -1)) != completed:
            raise ValueError(f"learned solver record {row} has inconsistent iterations")
        if record["converged"]:
            if record.get("converging_trial_iterations") != completed:
                raise ValueError(
                    f"learned solver record {row} has wrong converging iteration"
                )
        elif record.get("converging_trial_iterations") is not None:
            raise ValueError(
                f"learned solver record {row} reports convergence inconsistently"
            )
        record_labels = np.asarray(
            [
                record["labels"]["effective_temperature"],
                record["labels"]["log_surface_gravity"],
                record["labels"]["metallicity"],
                record["labels"]["alpha_enhancement"],
                record["labels"]["microturbulence_km_s"],
            ],
            dtype=np.float64,
        )
        if not np.allclose(record_labels, labels[row], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"learned solver record {row} is not in prediction order")
        if not np.isfinite(float(record["seconds"])) or float(record["seconds"]) < 0.0:
            raise ValueError(f"learned solver record {row} has invalid wall time")

    analytic_solver = _read_json(args.analytic_solver)
    if analytic_solver.get("arm") != "parity":
        raise ValueError("analytic solver artifact is not the parity arm")
    if analytic_solver.get("candidate") != ARM_CANDIDATES["parity"]:
        raise ValueError("analytic solver artifact names a different candidate")
    if analytic_solver.get("status") != "funnel_not_production":
        raise ValueError("analytic solver artifact has an unexpected status")
    if int(analytic_solver.get("requested_count", -1)) != indices.size:
        raise ValueError("analytic solver artifact requested a different sample size")
    if int(analytic_solver["iterations_per_trial"]) != ITERATIONS_PER_TRIAL:
        raise ValueError("analytic solver artifact uses a different iteration limit")
    if int(analytic_solver["split_seed"]) != analytic_split.seed:
        raise ValueError("analytic solver run used a different fitting split")
    if analytic_solver["initializer_provenance"]["configuration"] != analytic_asset[
        "configuration"
    ]:
        raise ValueError("analytic solver run used a different parity configuration")
    if int(
        analytic_solver["initializer_provenance"]["stored_float_count"]
    ) != parameters.stored_float_count:
        raise ValueError("analytic solver run reports a different constant count")
    solver_asset = analytic_solver["initializer_provenance"].get("parameter_asset")
    if solver_asset is not None and solver_asset["sha256"] != _sha256(
        args.analytic_parameters
    ):
        raise ValueError("analytic solver run loaded a different parameter asset")
    analytic_records = analytic_solver["records"]
    if len(analytic_records) != indices.size:
        raise ValueError("analytic solver record count does not match development-60")
    analytic_by_index = {int(record["corpus_index"]): record for record in analytic_records}
    if set(analytic_by_index) != set(int(index) for index in indices):
        raise ValueError("analytic solver run uses a different development sample")
    if len(analytic_by_index) != len(analytic_records):
        raise ValueError("analytic solver run contains duplicate corpus indices")
    ordered_analytic = [analytic_by_index[int(index)] for index in indices]
    for row, record in enumerate(ordered_analytic):
        if record.get("arm") != "parity":
            raise ValueError(f"analytic solver record {row} is not the parity arm")
        if int(record.get("trials_used", 1)) != 1:
            raise ValueError(f"analytic solver record {row} used more than one trial")
        record_labels = np.asarray(
            [
                record["effective_temperature"],
                record["log_surface_gravity"],
                record["metallicity"],
                record["alpha_enhancement"],
                record["microturbulence_km_s"],
            ],
            dtype=np.float64,
        )
        if not np.allclose(record_labels, labels[row], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"analytic solver record {row} has different labels")
        outcome = record.get("solver_outcome")
        if outcome not in {"converged", "not_converged", "timeout", "error"}:
            raise ValueError(f"analytic solver record {row} has invalid outcome")
        iterations = record.get("iterations_completed")
        if outcome in {"converged", "not_converged"}:
            if not isinstance(iterations, int) or not 1 <= iterations <= ITERATIONS_PER_TRIAL:
                raise ValueError(f"analytic solver record {row} has invalid iterations")
        if bool(record["converged"]) != (outcome == "converged"):
            raise ValueError(f"analytic solver record {row} has inconsistent convergence")
        if (
            outcome in {"converged", "not_converged"}
            and record.get("finite_final_state") is not True
        ):
            raise ValueError(f"analytic solver record {row} ended non-finitely")
        if not np.isfinite(float(record["seconds"])) or float(record["seconds"]) < 0.0:
            raise ValueError(f"analytic solver record {row} has invalid wall time")
    analytic_converged = np.asarray(
        [bool(record["converged"]) for record in ordered_analytic], dtype=bool
    )
    analytic_iterations = np.asarray(
        [
            np.nan
            if record.get("iterations_completed") is None
            else float(record["iterations_completed"])
            for record in ordered_analytic
        ],
        dtype=np.float64,
    )
    analytic_timeout_count = sum(
        record.get("solver_outcome") == "timeout" for record in ordered_analytic
    )
    analytic_error_count = sum(
        record.get("solver_outcome") == "error" for record in ordered_analytic
    )
    if int(analytic_solver["star_count"]) != indices.size:
        raise ValueError("analytic solver summary has a different star count")
    if int(analytic_solver["converged_count"]) != int(analytic_converged.sum()):
        raise ValueError("analytic solver convergence summary disagrees with records")
    if int(analytic_solver["timeout_count"]) != analytic_timeout_count:
        raise ValueError("analytic solver timeout summary disagrees with records")
    if int(analytic_solver["error_count"]) != analytic_error_count:
        raise ValueError("analytic solver error summary disagrees with records")

    common = learned_converged & analytic_converged
    if not np.any(common):
        raise ValueError("learned and analytic arms have no joint convergence")
    learned_common = learned_iterations[common]
    analytic_common = analytic_iterations[common]
    paired_difference = analytic_common - learned_common

    learned_only = learned_converged & ~analytic_converged
    analytic_only = analytic_converged & ~learned_converged
    paired = {
        "conditioning": "iteration contrasts include only jointly converged stars",
        "common_converged_count": int(common.sum()),
        "learned_only_converged_count": int(learned_only.sum()),
        "analytic_only_converged_count": int(analytic_only.sum()),
        "neither_converged_count": int((~learned_converged & ~analytic_converged).sum()),
        "learned_fewer_iterations_count": int((paired_difference > 0.0).sum()),
        "analytic_fewer_iterations_count": int((paired_difference < 0.0).sum()),
        "tied_count": int((paired_difference == 0.0).sum()),
        "mean_analytic_minus_learned_iterations": float(paired_difference.mean()),
        "median_analytic_minus_learned_iterations": float(
            np.median(paired_difference)
        ),
    }

    training = _read_json(args.learned_training)
    learned_parameters = _network_parameter_count(
        labels=labels.shape[1],
        layers=truth_mass.shape[1],
        width=int(training["width"]),
        depth=int(training["depth"]),
    )
    import torch

    checkpoint = torch.load(
        args.learned_checkpoint, map_location="cpu", weights_only=False
    )
    checkpoint_parameters = int(
        sum(value.numel() for value in checkpoint["state_dict"].values())
    )
    if checkpoint_parameters != learned_parameters:
        raise ValueError(
            "learned parameter count disagrees with the frozen checkpoint: "
            f"{learned_parameters} != {checkpoint_parameters}"
        )
    checkpoint_standardization = int(
        sum(np.asarray(value).size for value in checkpoint["standardization"].values())
    )
    from reduced_state.emulator import load_checkpoint, predict_reduced_state

    model, standardization, checkpoint_meta = load_checkpoint(args.learned_checkpoint)
    if set(int(index) for index in checkpoint_meta.get("held_out", [])) != set(
        int(index) for index in indices
    ):
        raise ValueError("learned checkpoint was not trained with development-60 held out")
    checkpoint_mass, checkpoint_temperature = predict_reduced_state(
        model, standardization, labels
    )
    if not np.array_equal(checkpoint_mass, learned_mass):
        raise ValueError("learned prediction does not match the frozen checkpoint")
    if not np.array_equal(checkpoint_temperature, learned_temperature):
        raise ValueError("learned temperature does not match the frozen checkpoint")
    guard_floats = int(
        parameters.stored_float_count
        - parameters.temperature.stored_float_count
        - parameters.opacity.stored_float_count
        + shared_label_scaling
    )
    learned_solver_summary = _solver_summary(learned_converged, learned_iterations)
    analytic_solver_summary = _solver_summary(
        analytic_converged,
        analytic_iterations,
        timeout_count=analytic_timeout_count,
        error_count=analytic_error_count,
    )
    analytic_solver_summary["iterations_per_trial"] = int(
        analytic_solver["iterations_per_trial"]
    )
    analytic_solver_summary["per_star_timeout_seconds"] = float(
        analytic_solver["per_star_timeout_seconds"]
    )

    payload = {
        "format": "payne_zero_paper_dev60_analytic_comparison_v1",
        "comparison_scope": (
            "same 60 development stars; profile errors use the same truth rows; "
            "successful handoffs use one 15-iteration trial per initializer, "
            "while rematerialization failures count as non-convergences; "
            "the learned and analytic solver records come from separate campaigns"
        ),
        "sample": {
            "name": "development-60",
            "star_count": int(indices.size),
            "star_indices": [int(index) for index in indices],
        },
        "learned_two_field": {
            "predicted_profiles": ["column_mass", "temperature"],
            "trainable_parameter_count": learned_parameters,
            "runtime_float_count": learned_parameters + checkpoint_standardization,
            "standardization_float_count": checkpoint_standardization,
            "training_rows": int(training["train_count"]),
            "validation_rows": int(training["validation_count"]),
            "requires_neural_checkpoint_at_runtime": True,
            "rematerialization_seed": (
                "physical" if adaptive_physical else "historical"
            ),
            "rematerialization_synchronization_mode": (
                "adaptive" if adaptive_physical else "fixed"
            ),
            "rematerialization_synchronizations": (
                None
                if adaptive_physical
                else int(learned_summary["n_synchronizations"])
            ),
            "rematerialization_max_synchronizations": (
                int(learned_summary["max_synchronizations"])
                if adaptive_physical
                else None
            ),
            "rematerialization_pressure_tolerance_dex": (
                float(learned_summary["pressure_tolerance_dex"])
                if adaptive_physical
                else None
            ),
            "rematerialization_failure_count": len(missing_rows),
            "checkpoint_reproduces_prediction_exactly": True,
            "profile_errors": learned_profile,
            "solver": learned_solver_summary,
        },
        "analytic_parity": {
            "fitted_profiles": ["temperature_over_grey", "rosseland_opacity"],
            "derived_profiles": ["column_mass"],
            "stored_float_count": int(parameters.stored_float_count),
            "logical_float_count": int(parameters.stored_float_count),
            "serialized_float_entry_count": serialized_float_entries,
            "duplicated_shared_normalization_float_entries": shared_label_scaling,
            "serialized_structural_integer_entry_count": serialized_integer_entries,
            "training_rows": int(frontier["split"]["train"]),
            "validation_rows": int(frontier["split"]["validation"]),
            "excluded_evaluation_rows": int(frontier["split"]["excluded"]),
            "requires_neural_checkpoint_at_runtime": False,
            "temperature_regimes_K": [5500.0, 7500.0],
            "configuration": analytic_asset["configuration"],
            "split_seed": int(analytic_split.seed),
            "deterministic_refit_matches_frozen_asset": True,
            "parameter_breakdown": {
                "temperature_closure_raw": int(
                    parameters.temperature.stored_float_count
                ),
                "opacity_closure_raw": int(parameters.opacity.stored_float_count),
                "shared_label_scaling_deduplicated": -shared_label_scaling,
                "support_and_monotonicity_guards": guard_floats,
                "total": int(parameters.stored_float_count),
                "label_polynomial_terms_per_regime": int(
                    parameters.temperature.exponents.shape[0]
                ),
                "temperature_regime_count": int(parameters.temperature.regimes),
            },
            "profile_errors": analytic_profile,
            "solver": analytic_solver_summary,
        },
        "paired_solver": paired,
        "sources": {
            name: {"path": str(path.relative_to(REPO)), "sha256": _sha256(path)}
            for name, path in source_paths.items()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        star_indices=indices,
        labels=labels,
        tau=corpus.tau,
        learned_temperature_relative_error=learned_temperature_error,
        analytic_temperature_relative_error=analytic_temperature_error,
        learned_column_mass_dex_error=learned_mass_error,
        analytic_column_mass_dex_error=analytic_mass_error,
        learned_converged=learned_converged,
        analytic_converged=analytic_converged,
        learned_iterations=learned_iterations,
        analytic_iterations=analytic_iterations,
    )
    print(f"wrote {args.output.relative_to(REPO)}")
    print(f"wrote {args.output.with_suffix('.npz').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
