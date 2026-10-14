"""Collect accepted local CNO atmospheres, retaining their common parent group."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "results/coauthor_validation_20261005"
FIELDS = ("effective_temperature", "log_surface_gravity", "metallicity",
          "alpha_enhancement", "microturbulence_km_s", "carbon_enhancement",
          "nitrogen_enhancement", "oxygen_enhancement")


def main():
    pilot = ROOT / "mgiant_cno_pilot/cno_development_corpus.npz"
    local = ROOT / "mgiant_cno_local/cno_development_corpus.npz"
    keys = ("labels", "atmosphere_profiles", "standard_rosseland_optical_depth",
            "labels_json", "parent_groups", "roles")
    parts = {key: [] for key in keys}
    for path in (pilot, local):
        with np.load(path) as data:
            for key in keys:
                parts[key].append(data[key].copy())
    with np.load(pilot) as data:
        tau = data["standard_rosseland_optical_depth"][0].copy()
        group = str(data["parent_groups"][0])
    walk = json.loads((ROOT / "mgiant_cno_walk/diagnostic.json").read_text())
    for record in walk["rows"]:
        if not record["passes"]:
            continue
        row = record["labels"]
        path = ROOT / "mgiant_cno_walk" / f"case_{record['index']:02d}" / "strict_restart_profiles.npz"
        with np.load(path) as data:
            profile = data["profiles"].copy()
        for key, value in zip(keys, (
            np.asarray([[row[f] for f in FIELDS]]), profile[None], tau[None],
            np.asarray([json.dumps(row)]), np.asarray([group]), np.asarray(["development_training"]))):
            parts[key].append(value)
    corpus = {key: np.concatenate(values) for key, values in parts.items()}
    if len(np.unique(corpus["labels"], axis=0)) != len(corpus["labels"]):
        raise ValueError("accepted sources contain duplicate label rows")
    output = ROOT / "mgiant_cno_development"
    output.mkdir(exist_ok=True)
    np.savez(output / "corpus.npz", **corpus)
    report = dict(accepted=len(corpus["labels"]), parent_groups=np.unique(corpus["parent_groups"]).tolist(),
        role="development training; no independent validation", label_fields=FIELDS,
        sources=[str(p.relative_to(ROOT)) for p in (pilot, local)] + ["mgiant_cno_walk/diagnostic.json"],
        recovered_walk_cases=[r["index"] for r in walk["rows"] if r["passes"]])
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
