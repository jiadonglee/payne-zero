"""Summarize complete matched restart records for the manuscript."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def main():
    root = Path("results/coauthor_validation_20261005")
    report = {"families": {}}
    for family in ("five_label", "cno8", "direct_abundance"):
        run = Path("runs") / f"full_{family}"
        summary = json.loads((run / "summary.json").read_text())
        by_index = {}
        for line in Path(f"runs/full_{family}.log").read_text().splitlines():
            if not line.startswith("{") or '"completed"' not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "completed" in event and "results" in event:
                by_index[event["index"]] = {"index": event["index"], "arms": event["results"]}
        if len(by_index) != summary["expected"]:
            indices = json.loads((root / family / "split.json").read_text())["indices"]["test"]
            for index in indices:
                if index not in by_index:
                    by_index[index] = json.loads((run / f"star_{index:06d}/record.json").read_text())
        records = list(by_index.values())
        if len(records) != summary["expected"]:
            raise ValueError(f"incomplete {family}: {len(records)} records")
        pairs = [r for r in records if all(r["arms"][a]["converged"]
                 for a in ("two_profile", "six_profile"))]
        paired = {a: float(np.mean([r["arms"][a]["iterations"] for r in pairs]))
                  for a in ("two_profile", "six_profile")}
        paired["count"] = len(pairs)
        paired["iteration_saving_fraction"] = 1 - paired["two_profile"] / paired["six_profile"]
        def cpu_cost(record):
            arms = json.loads((run / f"star_{record['index']:06d}/record.json").read_text())["arms"]
            return [arms[a]["start_cpu_seconds"] + arms[a]["solve_cpu_seconds"]
                    for a in ("two_profile", "six_profile")]
        with ThreadPoolExecutor(max_workers=8) as pool:
            costs = np.asarray(list(pool.map(cpu_cost, pairs)))
        paired["cpu_total_seconds"] = dict(zip(("two_profile", "six_profile"),
                                                np.mean(costs, axis=0).tolist()))
        failed = [r for r in records if not r["arms"]["two_profile"]["converged"]]
        summary["two_profile_failures_rescued_by_exact"] = sum(
            r["arms"]["exact_rebuild"]["converged"] for r in failed)
        summary["paired_neural_successes"] = paired
        diagnostic = json.loads((root / family / "restarts/diagnostics.json").read_text())
        summary["first_step_growth"] = diagnostic["first_step_growth"]
        if family == "five_label":
            summary["nearest_vs_two_paired"] = diagnostic["nearest_vs_two_paired"]
            with np.load("data/payne-zero-v1.3-atmosphere-training-corpora/five_label/strict_truth_52199.npz") as data:
                tau = data["standard_rosseland_optical_depth"][0]
            summary["reconstruction"] = {}
            for j, name in enumerate(("gas_pressure", "electron_density", "rosseland_opacity", "radiative_acceleration")):
                median = diagnostic["reconstruction"][str(j)]["median_by_depth"]
                summary["reconstruction"][name] = {
                    "count": diagnostic["reconstruction"][str(j)]["count"],
                    "samples": [{"tau_R": float(tau[k]), "median_relative_error": median[k]}
                                for k in (0, int(np.argmin(abs(np.log10(tau) + 1))),
                                          int(np.argmin(abs(np.log10(tau) - 1))), len(tau)-1)]}
        report["families"][family] = summary
    report["spectra"] = json.loads((root / "five_label/spectral_summary.json").read_text())
    (root / "manuscript_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for family, result in report["families"].items():
        print(family, json.dumps(result["paired_neural_successes"]))


if __name__ == "__main__":
    main()
