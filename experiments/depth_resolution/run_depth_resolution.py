"""Part 3: does the (m,T) representation's resolution affect restart quality?

Reframing agreed with the user (see notes/reduced_state_existing_work.md
Sec 3): the solver hard-codes ``layer_count == 80`` in certified physics
(``line_opacity.py``), and the repo's own ``continuity/`` harness already
found that densifying the closure quadrature 16x changes nothing (the error
lives in the fixed top-boundary seed, not the spacing). Rather than perform
solver surgery to support N != 80 end to end, this measures what a future
continuous emulator's *internal* representation resolution does to restart
behaviour once its output is materialized on the grid the solver actually
consumes -- because that materialization always happens, at N=80, no matter
how fine the emulator's internal resolution is.

For N in {40, 80, 160, 320, 640}: round-trip the true (m,T) curve through an
N-point intermediate log-tau grid (``reduced_state.resample``, cubic spline
in log-log, the same method ``continuity/closure.py`` uses), reconstruct the
full state from the round-tripped (m,T) (``reduced_state.reconstruct``,
identical code path to Part 2 with N=80 reducing to that experiment), and
restart the real solver from it.

Usage::

    PYTHONPATH=. python -m experiments.depth_resolution.run_depth_resolution \
        --count 60 --workers 40 --out results
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench.labels import StellarLabels
from bench.report import summarize
from continuity.closure import TARGET_FIELDS, load_corpus
from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere
from reduced_state.resample import representation_error, resample_reduced_state
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
RESOLUTIONS = (40, 80, 160, 320, 640)


def _label_subset(entry: dict) -> dict:
    return {key: entry[key] for key in LABEL_FIELDS}


def _reconstruct_worker(payload):
    idx, resolution, tau_std, column_mass, temperature, label_subset = payload
    resampled_m, resampled_t = resample_reduced_state(
        tau_std, column_mass, temperature, resolution
    )
    reduced = ReducedAtmosphere(
        column_mass=resampled_m, temperature=resampled_t, labels=label_subset
    )
    result = reconstruct_full_atmosphere(reduced, n_synchronizations=N_SYNCHRONIZATIONS)
    return idx, resolution, resampled_m, resampled_t, result.atmosphere


def load_star_indices(args, labels: list[dict]) -> np.ndarray:
    if args.reuse_indices_from is not None:
        payload = json.loads(Path(args.reuse_indices_from).read_text())
        indices = np.array(payload["star_indices"], dtype=int)
        print(f"reusing {len(indices)} star indices from {args.reuse_indices_from}", flush=True)
        return indices
    return sample_star_indices(labels, args.count, args.seed, args.hard_fraction)


def run_resolution(
    resolution: int,
    profiles: np.ndarray,
    tau_std: np.ndarray,
    labels: list[dict],
    indices: np.ndarray,
    *,
    workers: int,
    records_dir: Path,
) -> dict:
    payloads = [
        (
            int(idx),
            resolution,
            tau_std,
            profiles[idx, :, FIELD_INDEX["column_mass"]],
            profiles[idx, :, FIELD_INDEX["temperature"]],
            _label_subset(labels[idx]),
        )
        for idx in indices
    ]

    reconstructed = {}
    representation_errors = {"column_mass_relative_error": [], "temperature_relative_error": []}
    reconstruction_errors = {field: [] for field in RECONSTRUCTED_FIELDS}

    if workers <= 1:
        results = [_reconstruct_worker(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_reconstruct_worker, payloads))

    for idx, _resolution, resampled_m, resampled_t, atmosphere in results:
        reconstructed[idx] = atmosphere
        truth = profiles[idx]
        column_mass_truth = truth[:, FIELD_INDEX["column_mass"]]
        temperature_truth = truth[:, FIELD_INDEX["temperature"]]
        representation_errors["column_mass_relative_error"].append(
            np.abs(resampled_m - column_mass_truth) / column_mass_truth
        )
        representation_errors["temperature_relative_error"].append(
            np.abs(resampled_t - temperature_truth) / temperature_truth
        )
        for field in RECONSTRUCTED_FIELDS:
            recon_values = np.asarray(getattr(atmosphere, field), dtype=np.float64)
            truth_values = truth[:, FIELD_INDEX[field]]
            reconstruction_errors[field].append(
                np.abs(recon_values - truth_values) / np.abs(truth_values)
            )
        print(f"  N={resolution}: reconstructed star {idx}", flush=True)

    items = [(StellarLabels(**_label_subset(labels[int(idx)])), reconstructed[int(idx)]) for idx in indices]
    print(f"  N={resolution}: restarting {len(items)} stars...", flush=True)
    records = run_many_restarts(
        items,
        workers=workers,
        out_path=records_dir / f"n{resolution:04d}" / "records.jsonl",
        source=f"depth_resolution_n{resolution}",
    )
    convergence = summarize(records)

    representation_summary = {
        key: {
            "median": float(np.median(np.stack(values))),
            "p90": float(np.percentile(np.stack(values), 90)),
            "max": float(np.max(np.stack(values))),
        }
        for key, values in representation_errors.items()
    }
    reconstruction_summary = {
        field: {
            "median": float(np.median(np.stack(values))),
            "p90": float(np.percentile(np.stack(values), 90)),
            "max": float(np.max(np.stack(values))),
        }
        for field, values in reconstruction_errors.items()
    }

    return {
        "resolution": resolution,
        "representation_error": representation_summary,
        "reconstruction_error": reconstruction_summary,
        "convergence": convergence,
    }


def make_resolution_figures(by_resolution: dict[int, dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolutions = sorted(by_resolution)
    color = "#3b6fa0"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _plot(values, ylabel, title, filename, *, logy=True):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(resolutions, values, color=color, marker="o", linewidth=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(resolutions)
        ax.set_xticklabels([str(r) for r in resolutions])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("representation resolution N")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)

    _plot(
        [by_resolution[n]["convergence"]["converging_trial_iterations"]["mean"] for n in resolutions],
        "mean iterations to converge",
        "Depth resolution vs. iteration count",
        "resolution_vs_iterations.png",
        logy=False,
    )
    _plot(
        [by_resolution[n]["convergence"]["contraction"]["q_ratio"]["geometric_mean"] for n in resolutions],
        "geometric-mean contraction q",
        "Depth resolution vs. contraction",
        "resolution_vs_contraction.png",
        logy=False,
    )
    _plot(
        [by_resolution[n]["convergence"]["contraction"]["non_monotonic_fraction"] for n in resolutions],
        "non-monotonic fraction",
        "Depth resolution vs. non-monotonic trajectories",
        "resolution_vs_nonmonotonic.png",
        logy=False,
    )
    _plot(
        [by_resolution[n]["representation_error"]["column_mass_relative_error"]["median"] for n in resolutions],
        "median relative error in m(tau)",
        "Depth resolution vs. representation error",
        "resolution_vs_reconstruction_error.png",
        logy=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--hard-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--figures-dir", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument(
        "--records-dir", type=Path, default=REPO_ROOT / "runs" / "depth_resolution"
    )
    parser.add_argument(
        "--reuse-indices-from",
        type=Path,
        default=None,
        help="reconstruction_metrics.json from Part 2, to use the identical star sample",
    )
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=list(RESOLUTIONS)
    )
    args = parser.parse_args()

    print(f"loading corpus {args.corpus}", flush=True)
    profiles, tau_std_per_star, _iters, labels = load_corpus(args.corpus)
    tau_std = tau_std_per_star[0]

    indices = load_star_indices(args, labels)
    print(f"using {len(indices)} stars, resolutions={args.resolutions}", flush=True)

    by_resolution = {}
    for resolution in args.resolutions:
        print(f"=== N={resolution} ===", flush=True)
        by_resolution[resolution] = run_resolution(
            resolution,
            profiles,
            tau_std,
            labels,
            indices,
            workers=args.workers,
            records_dir=args.records_dir,
        )
        conv = by_resolution[resolution]["convergence"]
        print(
            f"  N={resolution}: converged={conv['converged_fraction']:.3f} "
            f"iters(mean/p90)={conv['converging_trial_iterations']['mean']:.2f}/"
            f"{conv['converging_trial_iterations']['p90']:.0f} "
            f"geomean_q={conv['contraction']['q_ratio']['geometric_mean']:.3f} "
            f"nonmono={conv['contraction']['non_monotonic_fraction']:.3f}",
            flush=True,
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "star_count": len(indices),
        "star_indices": [int(i) for i in indices],
        "seed": args.seed,
        "resolutions": args.resolutions,
        "n_synchronizations": N_SYNCHRONIZATIONS,
        "by_resolution": by_resolution,
    }
    out_path = args.results_dir / "convergence_metrics_depth_resolution.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}", flush=True)

    make_resolution_figures(by_resolution, args.figures_dir)
    print(f"wrote figures to {args.figures_dir}", flush=True)


if __name__ == "__main__":
    main()
