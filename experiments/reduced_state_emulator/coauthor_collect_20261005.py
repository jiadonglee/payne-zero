"""Collect complete test runs and draw the reconstruction and spectral figures."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path("results/coauthor_validation_20261005")
CORPUS=Path("data/payne-zero-v1.3-atmosphere-training-corpora")


def main():
    for family in ("five_label","cno8","direct_abundance"):
        run=Path("runs")/f"full_{family}"
        while True:
            try:
                summary=json.loads((run/"summary.json").read_text())
                if summary["completed"]==summary["expected"]: break
            except (FileNotFoundError,json.JSONDecodeError): pass
            time.sleep(60)
        out=ROOT/family/"restarts"
        subprocess.run([sys.executable,"plot_coauthor_restarts_20261005.py","--runs",str(run),
            "--corpus-root",str(CORPUS),"--family",family,"--out",str(out)],check=True)
        shutil.copy2(run/"summary.json",out/"solver_summary.json")
        print(json.dumps(dict(family=family,completed=summary["completed"],figures=str(out))),flush=True)
    expected=json.loads((ROOT/"five_label/split.json").read_text())["counts"]["test"]
    while len(list((ROOT/"spectra").glob("star_*.json")))<expected:
        time.sleep(60)
    records=[json.loads(p.read_text()) for p in sorted((ROOT/"spectra").glob("star_*.json"))]
    arms=("exact_rebuild","two_profile","six_profile","nearest_0","nearest_1","grey_convective")
    colors=("black","#1565c0","#d55e00","#009e73","#009e73","#777777")
    styles=(":","-","--","-",":","-")
    report=dict(expected=expected,completed=len(records),wavelength_nm=[400,900],resolution=20000,arms={})
    fig,ax=plt.subplots(figsize=(6.5,3.8),constrained_layout=True)
    for arm,color,style in zip(arms,colors,styles):
        values=np.asarray([r.get("arms",{}).get(arm,{}).get("maximum_normalized_flux_difference",np.inf) for r in records])
        finite=np.isfinite(values)
        sorted_values=np.sort(values[finite])
        if len(sorted_values):
            ax.plot(np.maximum(sorted_values,1e-12),np.arange(1,len(sorted_values)+1)/expected,
                    color=color,ls=style,label=arm.replace("_"," "))
        by_temperature=[]
        for lo,hi in ((4000,4500),(4500,5500),(5500,7500),(7500,10501)):
            mask=np.asarray([lo<=r["labels"]["effective_temperature"]<hi for r in records]) & finite
            by_temperature.append(dict(temperature_range_K=[lo,hi],count=int(mask.sum()),
                median=float(np.median(values[mask])) if mask.any() else None))
        report["arms"][arm]=dict(measured=int(finite.sum()),missing_or_failed=int((~finite).sum()),
            conditional_median=float(np.median(values[finite])) if finite.any() else None,
            by_temperature=by_temperature)
    pairs=[r for r in records if "two_vs_six_final_maximum_normalized_flux_difference" in r
           and all(r["numerically_converged"][a] for a in ("two_profile","six_profile"))]
    differences=[r["two_vs_six_final_maximum_normalized_flux_difference"] for r in pairs]
    report["final_normalized_flux_parity"]=dict(paired_numerical_successes=len(pairs),threshold=.005,
        exceeding_threshold=sum(x>.005 for x in differences),
        maximum=float(max(differences)) if differences else None)
    report["cdf_denominator"]=expected
    ax.set(xscale="log",xlabel="Maximum normalized-flux difference",ylabel="Fraction of all test stars",ylim=(0,1.01))
    ax.grid(alpha=.2); ax.legend(fontsize=8,loc="lower right")
    fig.savefig(ROOT/"five_label/fig_spectral.pdf")
    fig.savefig(ROOT/"five_label/fig_spectral.png",dpi=180)
    (ROOT/"five_label/spectral_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(spectra_completed=len(records))),flush=True)


if __name__=="__main__":
    main()
