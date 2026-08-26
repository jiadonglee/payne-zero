"""Part 2: oracle (m,T) -> full-atmosphere reconstruction, quantified.

For a stratified sample of converged truth atmospheres from
``strict_truth_52199.npz``:

1. Discard P, n_e, kappa_R, g_rad; keep only m(tau), T(tau), and labels.
2. Reconstruct the discarded fields via ``reduced_state.reconstruct``
   (certified physics only, m/T pinned exactly -- see that module's
   docstring for why this is not one call to the production iteration).
3. Compare the reconstruction against the stored truth, per field per depth
   layer -> ``results/reconstruction_metrics.json`` + per-star error arrays
   in ``results/reconstruction_metrics.npz`` + ``figures/reconstruction_parity.png``.
4. Restart the real solver from (a) the reconstructed atmosphere and (b) the
   full six-field truth (the "decoder oracle" -- an upper bound on what any
   initializer can achieve), and compare iteration counts / contraction via
   the reused ``bench.report`` aggregation ->
   ``results/convergence_metrics_reduced_state_parity.json``.

Usage::

    PYTHONPATH=. python -m experiments.reduced_state_parity.run_oracle_parity \
        --count 60 --workers 40 --out results
"""

from __future__ import annotations

# Must precede any Numba import, matching bench/run_reference.py.
from bench import environment as _environment  # noqa: F401,E402

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench.labels import StellarLabels
from bench.report import summarize
from continuity.closure import TARGET_FIELDS, load_corpus
from payne_zero_atmosphere.warm_start import emulator_warm_start_model
from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere
from reduced_state.restart import run_many_restarts
from reduced_state.sampling import sample_star_indices

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
FIELD_INDEX = {name: i for i, name in enumerate(TARGET_FIELDS)}
RECONSTRUCTED_FIELDS = ("gas_pressure", "electron_density", "rosseland_opacity", "radiative_acceleration")
N_SYNCHRONIZATIONS = 3


def _label_subset(entry: dict) -> dict:
    return {key: entry[key] for key in LABEL_FIELDS}


def _reconstruct_worker(payload):
    idx, column_mass, temperature, label_subset = payload
    reduced = ReducedAtmosphere(
        column_mass=column_mass, temperature=temperature, labels=label_subset
    )
    result = reconstruct_full_atmosphere(reduced, n_synchronizations=N_SYNCHRONIZATIONS)
    return idx, result.atmosphere


def reconstruct_sample(
    profiles: np.ndarray, labels: list[dict], indices: np.ndarray, *, workers: int
) -> dict[int, "ModelAtmosphere"]:
    payloads = [
        (
            int(idx),
            profiles[idx, :, FIELD_INDEX["column_mass"]],
            profiles[idx, :, FIELD_INDEX["temperature"]],
            _label_subset(labels[idx]),
        )
        for idx in indices
    ]
    reconstructed: dict[int, object] = {}
    if workers <= 1:
        for payload in payloads:
            idx, atmosphere = _reconstruct_worker(payload)
            reconstructed[idx] = atmosphere
            print(f"reconstructed {idx}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, atmosphere in executor.map(_reconstruct_worker, payloads):
                reconstructed[idx] = atmosphere
                print(f"reconstructed {idx}", flush=True)
    return reconstructed


def compute_reconstruction_errors(
    profiles: np.ndarray, indices: np.ndarray, reconstructed: dict
) -> dict[str, np.ndarray]:
    """Per-star, per-layer relative error for each reconstructed field."""

    errors = {field: np.empty((len(indices), 80), dtype=np.float64) for field in RECONSTRUCTED_FIELDS}
    truth_field_name = {
        "gas_pressure": "gas_pressure",
        "electron_density": "electron_density",
        "rosseland_opacity": "rosseland_opacity",
        "radiative_acceleration": "radiative_acceleration",
    }
    for row, idx in enumerate(indices):
        atmosphere = reconstructed[int(idx)]
        truth = profiles[idx]
        for field in RECONSTRUCTED_FIELDS:
            recon_values = np.asarray(getattr(atmosphere, field), dtype=np.float64)
            truth_values = truth[:, FIELD_INDEX[truth_field_name[field]]]
            errors[field][row] = np.abs(recon_values - truth_values) / np.abs(truth_values)
    return errors


def summarize_errors_by_layer(errors: dict[str, np.ndarray], tau_std: np.ndarray) -> dict:
    summary = {"tau_std": tau_std.tolist()}
    for field, matrix in errors.items():
        summary[field] = {
            "median_by_layer": np.median(matrix, axis=0).tolist(),
            "p90_by_layer": np.percentile(matrix, 90, axis=0).tolist(),
            "max_by_layer": np.max(matrix, axis=0).tolist(),
            "median_overall": float(np.median(matrix)),
            "p90_overall": float(np.percentile(matrix, 90)),
            "max_overall": float(np.max(matrix)),
        }
    return summary


def build_restart_pairs(
    profiles: np.ndarray, labels: list[dict], indices: np.ndarray, reconstructed: dict
) -> tuple[list, list]:
    reduced_state_items = []
    full_truth_items = []
    for idx in indices:
        idx = int(idx)
        label_subset = _label_subset(labels[idx])
        star_labels = StellarLabels(**label_subset)

        reduced_state_items.append((star_labels, reconstructed[idx]))

        warm_start_atmosphere, _deck = emulator_warm_start_model(
            device="cpu", **label_subset
        )
        truth = profiles[idx]
        full_truth_atmosphere = dataclasses.replace(
            warm_start_atmosphere,
            column_mass=truth[:, FIELD_INDEX["column_mass"]],
            temperature=truth[:, FIELD_INDEX["temperature"]],
            gas_pressure=truth[:, FIELD_INDEX["gas_pressure"]],
            electron_density=truth[:, FIELD_INDEX["electron_density"]],
            rosseland_opacity=truth[:, FIELD_INDEX["rosseland_opacity"]],
            radiative_acceleration=truth[:, FIELD_INDEX["radiative_acceleration"]],
        )
        full_truth_items.append((star_labels, full_truth_atmosphere))
    return reduced_state_items, full_truth_items


def make_parity_figure(errors: dict[str, np.ndarray], tau_std: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field_labels = {
        "gas_pressure": "P (dyn cm$^{-2}$)",
        "electron_density": "$n_e$ (cm$^{-3}$)",
        "rosseland_opacity": r"$\kappa_R$ (cm$^2$ g$^{-1}$)",
        "radiative_acceleration": "$g_{\\rm rad}$ (cm s$^{-2}$)",
    }
    color = "#3b6fa0"

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    for ax, field in zip(axes, RECONSTRUCTED_FIELDS):
        matrix = errors[field]
        median_by_layer = np.median(matrix, axis=0)
        p90_by_layer = np.percentile(matrix, 90, axis=0)
        ax.plot(tau_std, median_by_layer, color=color, linewidth=2, label="median")
        ax.plot(tau_std, p90_by_layer, color=color, linewidth=1.2, linestyle="--", label="p90")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\tau_{\rm Ross}$")
        ax.set_ylabel("relative error")
        ax.set_title(field_labels[field])
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Reconstruction error vs depth: (m,T,labels) -> (P, n_e, kappa_R, g_rad)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--hard-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--products-dir",
        type=Path,
        default=None,
        help=(
            "write each converged structured atmosphere here, per arm. Needed by "
            "the spectral gate, which synthesizes from the converged product; the "
            "original Part 2 run did not need it and left it off."
        ),
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "reduced_state_parity",
    )
    args = parser.parse_args()

    print(f"loading corpus {args.corpus}", flush=True)
    profiles, tau_std_per_star, iterations_to_convergence, labels = load_corpus(args.corpus)
    tau_std = tau_std_per_star[0]

    indices = sample_star_indices(labels, args.count, args.seed, args.hard_fraction)
    print(f"sampled {len(indices)} stars (seed={args.seed})", flush=True)

    reconstructed = reconstruct_sample(profiles, labels, indices, workers=args.workers)

    errors = compute_reconstruction_errors(profiles, indices, reconstructed)
    error_summary = summarize_errors_by_layer(errors, tau_std)
    error_summary["star_count"] = len(indices)
    error_summary["star_indices"] = [int(i) for i in indices]
    error_summary["n_synchronizations"] = N_SYNCHRONIZATIONS
    error_summary["seed"] = args.seed
    error_summary["hard_fraction"] = args.hard_fraction

    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "reconstruction_metrics.json").write_text(
        json.dumps(error_summary, indent=2)
    )
    np.savez(
        args.results_dir / "reconstruction_metrics.npz",
        star_indices=indices,
        tau_std=tau_std,
        **{f"{field}_relative_error": matrix for field, matrix in errors.items()},
    )
    print(f"wrote {args.results_dir / 'reconstruction_metrics.json'}", flush=True)

    make_parity_figure(errors, tau_std, args.figures_dir / "reconstruction_parity.png")
    print(f"wrote {args.figures_dir / 'reconstruction_parity.png'}", flush=True)

    reduced_state_items, full_truth_items = build_restart_pairs(
        profiles, labels, indices, reconstructed
    )

    print("restarting from reduced-state reconstructions...", flush=True)
    reduced_state_records = run_many_restarts(
        reduced_state_items,
        workers=args.workers,
        out_path=args.records_dir / "reduced_state_reconstruction" / "records.jsonl",
        source="reduced_state_reconstruction",
        product_dir=(
            args.products_dir / "reduced_state_reconstruction"
            if args.products_dir
            else None
        ),
    )
    print("restarting from full six-field truth (decoder-oracle upper bound)...", flush=True)
    full_truth_records = run_many_restarts(
        full_truth_items,
        workers=args.workers,
        out_path=args.records_dir / "full_truth_oracle" / "records.jsonl",
        source="full_truth_oracle",
        product_dir=(
            args.products_dir / "full_truth_oracle" if args.products_dir else None
        ),
    )

    convergence_summary = {
        "reduced_state_reconstruction": summarize(reduced_state_records),
        "full_truth_oracle": summarize(full_truth_records),
        "star_count": len(indices),
        "seed": args.seed,
        "hard_fraction": args.hard_fraction,
        "n_synchronizations": N_SYNCHRONIZATIONS,
    }
    out_path = args.results_dir / "convergence_metrics_reduced_state_parity.json"
    out_path.write_text(json.dumps(convergence_summary, indent=2))
    print(f"wrote {out_path}", flush=True)

    for tag, summary in (
        ("reduced_state_reconstruction", convergence_summary["reduced_state_reconstruction"]),
        ("full_truth_oracle", convergence_summary["full_truth_oracle"]),
    ):
        print(
            f"  {tag:>28s}  converged={summary['converged_fraction']:.3f}  "
            f"iters(mean/p90)={summary['converging_trial_iterations']['mean']:.2f}/"
            f"{summary['converging_trial_iterations']['p90']:.0f}  "
            f"geomean_q={summary['contraction']['q_ratio']['geometric_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
