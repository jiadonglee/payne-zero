"""Batch-one CPU timings for neural evaluation, rebuilding, and one iteration."""
from bench import environment as _environment
import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from coauthor_benchmark_20261005 import load_data, atmosphere
from coauthor_train_20261005 import build_model, decode
from bench.run_reference import _solver_config
from payne_zero_atmosphere.runner import run_atmosphere_model
from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, default=Path("results/coauthor_validation_20261005"))
    parser.add_argument("--out", type=Path, required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    data=load_data(args.corpus_root,"five_label",args.result_root)
    root=args.result_root/"five_label"
    ckpt=torch.load(root/"checkpoint_6profile.pt",map_location="cpu",weights_only=False)
    model=build_model(ckpt["model"]["config"])
    model.load_state_dict(ckpt["model"]["state_dict"]); model.eval()
    with np.load(root/"predicted_6profile_test.npz",allow_pickle=False) as d:
        features={int(i): x for i,x in zip(d["star_indices"],d["features"])}
    def network(index):
        row=data["rows"][index]
        x=(features[index]-ckpt["labels"]["mean"])/ckpt["labels"]["std"]
        with torch.no_grad():
            y=model(torch.as_tensor(x[None,:],dtype=torch.float32)).numpy()
        p=ckpt["pca"]
        coordinates=(((y*p["coefficient_std"]+p["coefficient_mean"])@p["basis"])*p["coordinate_std"]+p["coordinate_mean"]).reshape(1,80,6)
        return decode(coordinates,np.array([row["effective_temperature"]]),data["tau"],ckpt["coordinates"]["acceleration_scale"])[0]
    def rebuild(index):
        row=data["rows"][index]; profile=data["prediction"][2][index]
        return reconstruct_full_atmosphere(ReducedAtmosphere(profile[:,0],profile[:,1],row),
            seed=atmosphere(profile,row),n_synchronizations=None,max_synchronizations=8,pressure_tolerance_dex=1e-3)
    def iteration(index):
        seed=atmosphere(data["prediction"][6][index],data["rows"][index])
        return run_atmosphere_model(_solver_config(seed,iterations_per_trial=1,
            structured_atmosphere_path=None,debug_state_path=None))
    warm_index=data["split"]["test"][0]
    network(warm_index); rebuild(warm_index); iteration(warm_index)
    indices=data["split"]["test"]
    by_temperature=sorted(indices,key=lambda i:data["rows"][i]["effective_temperature"])
    selected=[by_temperature[j] for j in np.linspace(0,len(indices)-1,8,dtype=int)]
    rows=[]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for index in selected:
        entry=dict(index=index,labels=data["rows"][index])
        for name,operation,repeats in (("network",network,100),("rebuild",rebuild,1),("iteration",iteration,1)):
            start=time.perf_counter()
            try:
                for _ in range(repeats): result=operation(index)
                entry[name+"_seconds"]=(time.perf_counter()-start)/repeats
                if name=="rebuild": entry["rebuild_evaluations"]=result.n_evaluations
            except Exception as exc:
                entry[name+"_error"]=f"{type(exc).__name__}: {exc}"
        rows.append(entry)
        report=dict(host=socket.gethostname(),batch_size=1,threads=8,device="cpu",dtype="network float32; physics float64",
            warmup_excluded=True,selection="eight evenly spaced temperature ranks in the fixed test set",rows=rows)
        for name in ("network","rebuild","iteration"):
            values=[r[name+"_seconds"] for r in rows if name+"_seconds" in r]
            report[name+"_median_seconds"]=float(np.median(values)) if values else None
        paired=[r["rebuild_seconds"]/r["iteration_seconds"] for r in rows
                if "rebuild_seconds" in r and "iteration_seconds" in r]
        report["paired_rebuild_iteration_ratio_median"]=float(np.median(paired)) if paired else None
        report["completed"]=len(rows)
        report["expected"]=len(selected)
        args.out.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(entry),flush=True)


if __name__=="__main__":
    main()
