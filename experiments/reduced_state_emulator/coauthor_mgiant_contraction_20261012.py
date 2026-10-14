"""Per-segment contraction of the deep-temperature change along solver paths."""
import json, glob, numpy as np
from pathlib import Path
R = Path("/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-coauthor-20261005/runs")
rows = []
for run in ("mgiant_cno_multiparent_20261008", "mgiant_cno_round2_20261009", "mgiant_cno_round3_20261010"):
    setup = json.loads((R / run / "setup.json").read_text())
    for case in sorted((R / run).glob("case_*")):
        i = int(case.name.split("_")[1]); lab = setup["labels"][i]
        for arm in ("physical", "reference"):
            files = sorted((case / arm).glob("iter_*.npz"))
            if len(files) < 6: continue
            T = []
            for f in files:
                with np.load(f) as d: T.append(d["temperature"].copy())
            T = np.asarray(T)
            if not np.isfinite(T).all(): continue
            d = np.array([np.percentile(abs(T[k] / T[k - 1] - 1), 95) for k in range(1, len(T))])
            # late part of the path: segments after the change first falls below 1e-3
            late = np.flatnonzero(d < 1e-3)
            if len(late) < 4: continue
            seg = d[late[0]:]
            ratios = seg[1:] / seg[:-1]
            rows.append(dict(run=run, case=i, arm=arm, teff=lab["effective_temperature"], logg=lab["log_surface_gravity"],
                             mh=lab["metallicity"], ratio=float(np.median(ratios[:6])), final=float(seg[-1]), segments=len(seg)))
json.dump(rows, open(R.parent / "results/coauthor_validation_20261005/contraction_rows.json", "w"))
r = np.array([x["ratio"] for x in rows]); t = np.array([x["teff"] for x in rows]); g = np.array([x["logg"] for x in rows])
print(len(rows))
for lo, hi in ((3000, 3150), (3150, 3300), (3300, 3600), (3600, 4001)):
    for glo, ghi in ((0.5, 1.5), (1.5, 2.51)):
        k = (t >= lo) & (t < hi) & (g >= glo) & (g < ghi)
        if k.sum(): print(f"T {lo}-{hi} logg {glo}-{ghi}: n={k.sum()} median ratio {np.median(r[k]):.3f} p90 {np.percentile(r[k],90):.3f}")
