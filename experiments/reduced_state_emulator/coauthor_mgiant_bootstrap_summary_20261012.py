"""Numbers of the three M-giant bootstrapping rounds for the manuscript."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results/coauthor_validation_20261005"
ROUNDS = {1: "mgiant_cno_multiparent", 2: "mgiant_cno_round2", 3: "mgiant_cno_round3"}
COOL = 3300.0


def paired_iterations(directory, rows):
    learned, reference = [], []
    for row in rows:
        stability = json.loads((directory / f"case_{row['index']:02d}" / "spectral_stability.json").read_text())
        if stability.get("spectral_consistency"):
            learned.append(stability["arms"]["physical"]["frozen_iteration"])
            reference.append(stability["arms"]["reference"]["frozen_iteration"])
    learned, reference = np.asarray(learned), np.asarray(reference)
    return dict(paired=len(learned), learned_median=float(np.median(learned)),
                reference_median=float(np.median(reference)), learned_faster=int((learned < reference).sum()))


def unstable_last_segments(directory, rows, temperature):
    """Last-segment changes of cool paths that stop at the cap without meeting the stability rule."""
    spectral, deep = [], []
    for row in rows:
        if row["effective_temperature"] >= temperature:
            continue
        stability = json.loads((directory / f"case_{row['index']:02d}" / "spectral_stability.json").read_text())
        if stability.get("arm_errors"):
            continue
        for arm in ("physical", "reference"):
            entry = stability["arms"][arm]
            if entry["frozen_iteration"] is None and entry["segments"]:
                last = entry["segments"][-1]
                spectral.append(max(last["spectral_changes"]))
                deep.append(last["temperature_p95"])
    return spectral, deep


def main():
    setup = json.loads((ROOT / "mgiant_cno_multiparent/setup.json").read_text())
    roles = [p["role"] for p in setup["parents"]]
    report = dict(holdout_parents=roles.count("holdout"), development_parents=roles.count("development"),
                  holdout_box=dict(effective_temperature=75.0, log_surface_gravity=0.25, metallicity=0.25),
                  rounds={})
    spectral, deep = [], []
    for number, name in ROUNDS.items():
        directory = ROOT / name
        summary = json.loads((directory / "summary.json").read_text())
        rows = summary["rows"]
        cool = [r for r in rows if r["effective_temperature"] < COOL]
        warm = [r for r in rows if r["effective_temperature"] >= COOL]
        entry = dict(labels=summary["expected"], admitted=summary["counts"]["admitted"],
                     admitted_fraction=summary["admitted_fraction"],
                     with_reference=summary["expected"] - summary["counts"]["no_reference"],
                     admitted_fraction_with_reference=summary["admitted_fraction_with_reference"],
                     counts=summary["counts"],
                     cool_labels=len(cool), cool_admitted=sum(r["outcome"] == "admitted" for r in cool),
                     warm_labels=len(warm), warm_admitted=sum(r["outcome"] == "admitted" for r in warm),
                     **paired_iterations(directory, rows))
        report["rounds"][str(number)] = entry
        if number > 1:
            s, d = unstable_last_segments(directory, rows, COOL)
            spectral += s
            deep += d
    report["cool_unstable_paths"] = dict(paths=len(deep),
        spectral_change_median=float(np.median(spectral)), deep_temperature_change_median=float(np.median(deep)),
        deep_temperature_change_p10=float(np.percentile(deep, 10)),
        deep_temperature_change_p90=float(np.percentile(deep, 90)),
        stability_temperature_limit=1e-4, stability_spectral_limit=5e-4)
    corpus = json.loads((ROOT / "mgiant_round3/corpus/summary.json").read_text())
    report["corpus"] = dict(before_round3=corpus["rows"],
                            total=corpus["rows"] + report["rounds"]["3"]["admitted"],
                            cool_regenerated=corpus["cool_regeneration"]["passed"],
                            round1_kept=next(v["rows"] - v["excluded_by_holdout_box"]
                                             for k, v in corpus["cno_sources"].items() if "multiparent" in k),
                            excluded_by_holdout_box=corpus["cool_regeneration"]["excluded_by_holdout_box"]
                            + sum(v["excluded_by_holdout_box"] for v in corpus["cno_sources"].values()))
    stable, capped, capped_cool = [], [], 0
    directories = {"mgiant_cno_multiparent_20261008": "mgiant_cno_multiparent",
                   "mgiant_cno_round2_20261009": "mgiant_cno_round2", "mgiant_cno_round3_20261010": "mgiant_cno_round3"}
    for row in json.loads((ROOT / "contraction_rows.json").read_text()):
        stability = json.loads((ROOT / directories[row["run"]] / f"case_{row['case']:02d}" / "spectral_stability.json").read_text())
        arm = stability.get("arms", {}).get(row["arm"], {})
        if "frozen_iteration" in arm:
            (stable if arm["frozen_iteration"] else capped).append(row["ratio"])
            capped_cool += not arm["frozen_iteration"] and row["teff"] < COOL and row["logg"] > 1.5
    report["contraction"] = dict(stable_paths=len(stable), capped_paths=len(capped),
        stable_median=float(np.median(stable)), capped_median=float(np.median(capped)),
        capped_p90=float(np.percentile(capped, 90)), capped_cool_high_gravity=int(capped_cool), capped_at_least_one=int(np.sum(np.asarray(capped) >= 1)),
        definition="median ratio of successive five-iteration 95th-percentile temperature changes, "
                   "from the first segment below 1e-3, over up to six segments")
    order = []
    for directory in ROUNDS.values():
        for path in sorted((ROOT / directory).glob("case_*/spectral_stability.json")):
            for arm in json.loads(path.read_text()).get("arms", {}).values():
                segments = arm.get("segments") if isinstance(arm, dict) else None
                if not segments:
                    continue
                tio = next((s["iteration"] for s in segments if max(s["spectral_changes"]) < 5e-4), None)
                temperature = next((s["iteration"] for s in segments if s["temperature_p95"] < 1e-4), None)
                if tio and temperature:
                    order.append((tio, temperature))
    order = np.asarray(order)
    report["settling_order"] = dict(paths=len(order), tio_first=int((order[:, 0] < order[:, 1]).sum()),
        same_segment=int((order[:, 0] == order[:, 1]).sum()), temperature_first=int((order[:, 0] > order[:, 1]).sum()),
        definition="first five-iteration segment with all three TiO changes below 5e-4 against the first with the "
                   "temperature-change 95th percentile below 1e-4, over paths where both occur")
    (ROOT / "mgiant_bootstrap_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
