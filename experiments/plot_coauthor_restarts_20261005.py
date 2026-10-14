"""Reconstruction and iteration diagnostics from the matched test restarts."""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

STYLES = {
    "two_profile": ("#1565c0", "-", "Two-profile neural network"),
    "six_profile": ("#d55e00", "--", "Six-profile neural network"),
    "nearest_0": ("#009e73", "-", "Nearest node: negative offset"),
    "nearest_1": ("#009e73", ":", "Nearest node: positive offset"),
    "grey_convective": ("#777777", "-", "Grey–convective"),
}


def save(fig, root, name):
    fig.savefig(root / f"{name}.pdf")
    fig.savefig(root / f"{name}.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--family", choices=("five_label","cno8","direct_abundance"),default="five_label")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in sorted(args.runs.glob("star_*/record.json"))]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.corpus_root:
        from coauthor_benchmark_20261005 import load_data
        data=load_data(args.corpus_root,args.family,Path("results/coauthor_validation_20261005"))
        truth,tau=data["profiles"],data["tau"]
    else:
        with np.load(args.truth, allow_pickle=False) as data:
            truth = data["atmosphere_profiles"]
            tau = data["standard_rosseland_optical_depth"][0]
    report = dict(completed=len(records), reconstruction={}, first_step_growth={}, arms={})
    errors = []
    for record in records:
        path = args.runs / f"star_{record['index']:06d}/exact_rebuild.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                reference = truth[record["index"], :, 2:6]
                errors.append(np.abs(data["start"][:, 2:6] - reference) / np.maximum(np.abs(reference), 1e-30))
    if errors:
        errors = np.asarray(errors)
        fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), constrained_layout=True)
        for j, (ax, label) in enumerate(zip(axes.flat, (r"$P_{\rm gas}$", r"$n_{\rm e}$", r"$\kappa_{\rm R}$", r"$g_{\rm rad}$"))):
            q25, median, q90 = np.percentile(errors[:, :, j], (25, 50, 90), axis=0)
            ax.fill_between(tau, np.maximum(q25, 1e-12), np.maximum(q90, 1e-12), color="#1565c0", alpha=.18)
            ax.plot(tau, np.maximum(median, 1e-12), color="#1565c0")
            ax.set(xscale="log", yscale="log", xlabel=r"$\tau_{\rm R}$", ylabel="Relative error", title=label)
            ax.grid(alpha=.2)
            report["reconstruction"][str(j)] = dict(count=len(errors), median_by_depth=median.tolist(), p90_by_depth=q90.tolist())
        save(fig, args.out, "fig_reconstruction")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), constrained_layout=True)
    for arm, (color, style, label) in STYLES.items():
        entries = [r["arms"][arm] for r in records if arm in r["arms"]]
        if not entries:
            continue
        trajectories = [e.get("trajectory", []) for e in entries]
        nmax = max((len(t) for t in trajectories), default=0)
        for trajectory in trajectories[::max(1, len(trajectories)//20)][:20]:
            y = [step["deep_layer_relative_temperature_change"] for step in trajectory]
            axes[0].plot(np.arange(1, len(y)+1), y, color=color, ls=style, alpha=.08, lw=.6)
        median = [np.median([t[n]["deep_layer_relative_temperature_change"] for t in trajectories if len(t)>n]) for n in range(nmax)]
        axes[0].plot(np.arange(1,nmax+1), median, color=color, ls=style, label=label, lw=1.8)
        counts = np.asarray([e["iterations"] if e["converged"] else np.inf for e in entries])
        n = np.arange(1,16)
        axes[1].plot(n, [(counts<=k).mean() for k in n], color=color, ls=style, label=label)
        pairs = [t for t in trajectories if len(t)>=2]
        growth = [t[1]["deep_layer_relative_temperature_change"] > t[0]["deep_layer_relative_temperature_change"] for t in pairs]
        report["first_step_growth"][arm] = dict(count=len(pairs), increased=sum(growth), fraction=float(np.mean(growth)) if growth else None)
        report["arms"][arm] = dict(total=len(entries), converged=int(np.isfinite(counts).sum()), failed=int(np.isinf(counts).sum()),
            median_iterations=float(np.median(counts[np.isfinite(counts)])) if np.isfinite(counts).any() else None,
            final_p95_flux_error_percent=[t[-1]["p95_absolute_flux_error_percent"] for t in trajectories if t])
    axes[0].axhline(5e-4, color="black", lw=.7, ls=":")
    axes[0].set(yscale="log", xlabel="Completed iteration n", ylabel=r"$\epsilon_T$", xlim=(1,15))
    axes[1].set(xlabel="Completed iteration n", ylabel="Fraction converged", ylim=(0,1.01), xlim=(1,15))
    axes[1].legend(fontsize=7, loc="lower right")
    for ax in axes:
        ax.grid(alpha=.2)
    save(fig, args.out, "fig_convergence")
    candidates = [r for r in records if all(r["arms"].get(a, {}).get("trajectory") for a in ("two_profile", "six_profile"))]
    if candidates:
        candidates.sort(key=lambda r:r["arms"]["two_profile"]["trajectory"][0]["deep_layer_relative_temperature_change"])
        typical = candidates[len(candidates)//2]
        fig, axes = plt.subplots(1, 2, figsize=(9.0,3.3), constrained_layout=True)
        for arm in ("two_profile", "six_profile"):
            color, style, label = STYLES[arm]
            trajectories = [r["arms"][arm]["trajectory"] for r in candidates]
            for t in trajectories[::max(1,len(trajectories)//30)][:30]:
                axes[0].plot(np.arange(1,len(t)+1), [s["deep_layer_relative_temperature_change"] for s in t], color=color, ls=style, alpha=.12, lw=.6)
            nmax=max(map(len,trajectories))
            axes[0].plot(np.arange(1,nmax+1), [np.median([t[n]["deep_layer_relative_temperature_change"] for t in trajectories if len(t)>n]) for n in range(nmax)], color=color, ls=style, label=label)
            path=args.runs/f"star_{typical['index']:06d}/{arm}.npz"
            with np.load(path,allow_pickle=False) as data:
                for n, correction in enumerate(data.get("relative_corrections", [])):
                    axes[1].plot(np.arange(1,81), correction, color=("#009e73","#d55e00","#0072b2")[n], ls=style, label=f"n={n+1}, {arm.replace('_',' ')}")
        axes[0].set(xlabel="Completed iteration n", ylabel=r"$\epsilon_T$", yscale="log", xlim=(1,15))
        axes[1].set(xlabel="Layer j", ylabel=r"$|\Delta T_j/T_j|$", yscale="log")
        for ax in axes:
            ax.grid(alpha=.2); ax.legend(fontsize=7)
        report["typical_index"]=typical["index"]
        report["typical_labels"]=typical["labels"]
        save(fig,args.out,"fig_mismatch")
    report["trajectory_median_population"]="stars still iterating at each n; first-step comparison uses paired n=1 and n=2"
    paired=[r for r in records if all(r["arms"].get(a,{}).get("converged",False)
                                    for a in ("two_profile","nearest_0","nearest_1"))]
    if paired:
        two=np.asarray([r["arms"]["two_profile"]["iterations"] for r in paired])
        nearest=np.asarray([np.median([r["arms"][a]["iterations"] for a in ("nearest_0","nearest_1")]) for r in paired])
        report["nearest_vs_two_paired"]=dict(count=len(paired),two_mean_iterations=float(two.mean()),
            nearest_mean_iterations=float(nearest.mean()),ratio_of_means=float(nearest.mean()/two.mean()),
            selection="two-profile start and both nearest nodes converge; failed counts reported separately")
    (args.out/"diagnostics.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
