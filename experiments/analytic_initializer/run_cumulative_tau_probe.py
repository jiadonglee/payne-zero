"""Screen the frozen cumulative-``tau`` analytic initializer offline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .cumulative_tau_initializer import (
    anchor_layer,
    fit_cumulative_tau_parameters,
    fit_oracle_targets,
    integrated_partition_windows,
    predict_cumulative_tau_state,
)
from .discovery import (
    DEFAULT_CORPUS,
    Split,
    collect_excluded_indices,
    file_sha256,
    load_strict_truth,
    make_split,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results/analytic_initializer/cumulative_tau_probe.json"
MANIFESTS = (
    REPO_ROOT / "results/reconstruction_metrics.json",
    REPO_ROOT / "results/sealed_solver_subset_20260808.json",
    REPO_ROOT / "results/sealed_audit_20260808.json",
    REPO_ROOT / "results/sealed_audit_20260811.json",
    REPO_ROOT / "results/sealed_initializer_holdout_20260812.json",
    REPO_ROOT / "results/initializer_calibration_20260812.json",
    REPO_ROOT
    / "results/four_initializer_benchmark_expanded_20260814/expanded200_manifest.json",
)
DEEP_START, DEEP_TRIM = 39, 5


def _restricted_split(corpus_size: int, excluded: np.ndarray, max_rows: int | None, seed: int):
    available = np.setdiff1d(
        np.arange(corpus_size, dtype=np.int64), np.asarray(excluded, dtype=np.int64)
    )
    if max_rows is not None:
        if max_rows < 10:
            raise ValueError("--max-rows must be at least 10")
        if max_rows < available.size:
            generator = np.random.default_rng(int(seed) + 1)
            selected = np.sort(
                generator.choice(available, size=int(max_rows), replace=False)
            )
        else:
            selected = available
    else:
        selected = available
    not_selected = np.setdiff1d(
        np.arange(corpus_size, dtype=np.int64), selected, assume_unique=True
    )
    global_split = make_split(corpus_size, excluded=not_selected, seed=seed)
    train = np.searchsorted(selected, global_split.train)
    validation = np.searchsorted(selected, global_split.validation)
    return selected, Split(
        train=train,
        validation=validation,
        excluded=np.empty(0, dtype=np.int64),
        seed=seed,
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50.0)),
        "p95": float(np.percentile(values, 95.0)),
    }


def _invariants(prediction) -> dict[str, object]:
    mass = prediction.column_mass
    temperature = prediction.temperature
    opacity = prediction.opacity
    return {
        "temperature_finite_positive": bool(
            np.all(np.isfinite(temperature)) and np.all(temperature > 0.0)
        ),
        "temperature_strictly_monotone": bool(
            np.all(np.diff(temperature, axis=1) > 0.0)
        ),
        "mass_finite_positive_strictly_monotone": bool(
            np.all(np.isfinite(mass))
            and np.all(mass > 0.0)
            and np.all(np.diff(mass, axis=1) > 0.0)
        ),
        "opacity_finite_positive_and_identity": bool(
            np.all(np.isfinite(opacity)) and np.all(opacity > 0.0)
        ),
    }


def _oracle_representation(
    tau: np.ndarray,
    temperature: np.ndarray,
    mass: np.ndarray,
    targets: np.ndarray,
    held: np.ndarray,
    *,
    width: float,
) -> dict[str, object]:
    """Measure the four-window shape floor before fitting the label map."""

    index = anchor_layer(tau)
    integrated = integrated_partition_windows(
        np.log(tau), np.log(tau[index]), width=width
    )
    reconstructed_temperature = np.exp(
        np.log(temperature[:, index])[:, None]
        + np.exp(targets[:, 2:6]) @ integrated.T
    )
    reconstructed_mass = np.exp(
        np.log(mass[:, index])[:, None]
        + np.exp(targets[:, 6:10]) @ integrated.T
    )
    t_error = np.abs(reconstructed_temperature[held] / temperature[held] - 1.0)
    m_error = np.abs(
        np.log10(reconstructed_mass[held]) - np.log10(mass[held])
    )
    deep = slice(DEEP_START, temperature.shape[1] - DEEP_TRIM)
    return {
        "meaning": "per-star NNLS shape floor before the label-to-parameter map",
        "temperature_relative": _quantiles(t_error),
        "mass_absolute_dex": _quantiles(m_error),
        "deep": {
            "temperature_relative": _quantiles(t_error[:, deep]),
            "mass_absolute_dex": _quantiles(m_error[:, deep]),
        },
    }


def _seam_checks(parameters, tau: np.ndarray, reference_label: np.ndarray) -> list[dict]:
    records = []
    for boundary in (5500.0, 7500.0):
        labels = np.repeat(np.asarray(reference_label, dtype=np.float64)[None, :], 2, axis=0)
        labels[0, 0] = np.nextafter(boundary, -np.inf)
        labels[1, 0] = np.nextafter(boundary, np.inf)
        prediction = predict_cumulative_tau_state(labels, tau, parameters)
        records.append(
            {
                "boundary_K": boundary,
                "left_K": float(labels[0, 0]),
                "right_K": float(labels[1, 0]),
                "temperature_max_relative_jump": float(
                    np.max(np.abs(prediction.temperature[1] / prediction.temperature[0] - 1.0))
                ),
                "mass_max_dex_jump": float(
                    np.max(
                        np.abs(
                            np.log10(prediction.column_mass[1])
                            - np.log10(prediction.column_mass[0])
                        )
                    )
                ),
            }
        )
    return records


def run_probe(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    corpus = load_strict_truth(args.corpus)
    excluded, manifests = collect_excluded_indices(MANIFESTS, corpus_size=corpus.size)
    selected, split = _restricted_split(
        corpus.size, excluded, args.max_rows, args.split_seed
    )
    labels = corpus.labels[selected]
    temperature = corpus.temperature[selected]
    mass = corpus.column_mass[selected]
    support_rows = np.arange(selected.size, dtype=np.int64)
    held = split.validation
    reference_label = np.median(labels, axis=0)
    candidates: list[dict[str, object]] = []

    for width in args.widths:
        width_started = time.perf_counter()
        targets = fit_oracle_targets(
            corpus.tau,
            temperature,
            mass,
            labels[:, 0],
            width=width,
        )
        oracle_representation = _oracle_representation(
            corpus.tau,
            temperature,
            mass,
            targets,
            held,
            width=width,
        )
        oracle_seconds = time.perf_counter() - width_started
        for label_feature_map in args.label_feature_maps:
            for degree in args.degrees:
                candidate_started = time.perf_counter()
                parameters = fit_cumulative_tau_parameters(
                    labels,
                    corpus.tau,
                    targets,
                    split.train,
                    degree=degree,
                    width=width,
                    label_features_name=label_feature_map,
                    support_indices=support_rows,
                )
                prediction = predict_cumulative_tau_state(
                    labels[held], corpus.tau, parameters
                )
                t_error = np.abs(prediction.temperature / temperature[held] - 1.0)
                m_error = np.abs(
                    np.log10(prediction.column_mass) - np.log10(mass[held])
                )
                deep = slice(DEEP_START, corpus.layers - DEEP_TRIM)
                identity = (
                    prediction.opacity
                    * prediction.column_mass
                    * prediction.mass_log_slope
                    / corpus.tau[None, :]
                )
                seams = _seam_checks(parameters, corpus.tau, reference_label)
                invariants = _invariants(prediction)
                identity_error = float(np.max(np.abs(identity - 1.0)))
                invariants["opacity_finite_positive_and_identity"] = bool(
                    invariants["opacity_finite_positive_and_identity"]
                    and identity_error <= 32.0 * np.finfo(np.float64).eps
                )
                score = max(
                    float(np.percentile(t_error, 95.0)) / 0.05,
                    float(np.percentile(m_error, 95.0)) / 0.20,
                )
                gate = {
                    "all_invariants": bool(all(invariants.values())),
                    "temperature_relative_p95_le_0p05": bool(
                        np.percentile(t_error, 95.0) <= 0.05
                    ),
                    "mass_dex_p95_le_0p20": bool(
                        np.percentile(m_error, 95.0) <= 0.20
                    ),
                    "seams_below_1e-3": bool(
                        all(
                            seam["temperature_max_relative_jump"] < 1.0e-3
                            and seam["mass_max_dex_jump"] < 1.0e-3
                            for seam in seams
                        )
                    ),
                    "fitted_parameters_le_200": bool(
                        parameters.fitted_parameter_count <= 200
                    ),
                }
                candidates.append(
                    {
                        "label_feature_map": label_feature_map,
                        "degree": int(degree),
                        "width_ln_tau": float(width),
                        "term_count": parameters.term_count,
                        "fitted_parameter_count": parameters.fitted_parameter_count,
                        "stored_float_count": parameters.stored_float_count,
                        "oracle_representation": oracle_representation,
                        "temperature_relative": _quantiles(t_error),
                        "mass_absolute_dex": _quantiles(m_error),
                        "deep": {
                            "definition": f"layers [{DEEP_START}:{corpus.layers - DEEP_TRIM}]",
                            "temperature_relative": _quantiles(t_error[:, deep]),
                            "mass_absolute_dex": _quantiles(m_error[:, deep]),
                        },
                        "invariants": invariants,
                        "opacity_identity_max_absolute_error": identity_error,
                        "seam_checks": seams,
                        "selection_score": score,
                        "selection_score_definition": "max(T_p95/0.05, m_p95_dex/0.20)",
                        "offline_gate": {**gate, "all_pass": bool(all(gate.values()))},
                        "timing_seconds": {
                            "shared_width_oracle": oracle_seconds,
                            "fit_predict_score": time.perf_counter() - candidate_started,
                        },
                    },
                )

    best = min(
        candidates,
        key=lambda item: (
            item["selection_score"],
            item["fitted_parameter_count"],
        ),
    )
    payload: dict[str, object] = {
        "format": "payne_zero_cumulative_tau_probe_v1",
        "status": "offline_smoke" if args.max_rows is not None else "offline_full_open",
        "sealed_holdout_status": "not opened; manifest indices used only for exclusion",
        "corpus": {
            "path": str(args.corpus),
            "sha256": file_sha256(args.corpus),
            "rows_total": corpus.size,
            "layers": corpus.layers,
        },
        "selection": {
            "max_rows": args.max_rows,
            "selected_open_rows": int(selected.size),
            "selected_original_index_sha256": __import__("hashlib").sha256(
                selected.tobytes()
            ).hexdigest(),
        },
        "split": {
            "seed": args.split_seed,
            "train": int(split.train.size),
            "validation": int(split.validation.size),
            "source_excluded": int(excluded.size),
            "manifests": manifests,
        },
        "screen": {
            "label_feature_maps": list(args.label_feature_maps),
            "degrees": [int(value) for value in args.degrees],
            "widths_ln_tau": [float(value) for value in args.widths],
            "oracle_targets_reused_across_degrees": True,
        },
        "candidates": candidates,
        "best_candidate": best,
        "elapsed_seconds": time.perf_counter() - started,
        "reproducer": (
            "PYTHONPATH=. /Users/jdli/anaconda3/bin/python -m "
            "experiments.analytic_initializer.run_cumulative_tau_probe"
            + (f" --max-rows {args.max_rows}" if args.max_rows is not None else "")
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=REPO_ROOT / DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=20260816)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--degrees", type=int, nargs="+", default=(1, 2))
    parser.add_argument(
        "--label-feature-maps",
        nargs="+",
        choices=("standard", "physical"),
        default=("standard", "physical"),
    )
    parser.add_argument("--widths", type=float, nargs="+", default=(0.35, 0.7, 1.2))
    args = parser.parse_args(argv)
    payload = run_probe(args)
    print(json.dumps(payload["best_candidate"], indent=2))
    print(f"elapsed_seconds={payload['elapsed_seconds']:.3f}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
