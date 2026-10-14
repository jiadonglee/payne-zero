"""Eight development CNO corners from learned and MARCS atmosphere starts."""
from bench import environment as _environment

import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from bench.labels import StellarLabels
from payne_zero_atmosphere.warm_start import INITIALIZER_OUTPUT_FIELDS
from reduced_state.emulator import load_physical_checkpoint, predict_physical_state
from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere
from experiments.reduced_state_emulator.marcs_h5 import load_marcs_node
from coauthor_benchmark_20261005 import atmosphere, FIELDS
from coauthor_mgiant_stability_20261007 import run, evaluate


def worker(payload):
    index,row,profiles,root = payload
    directory = root / f"case_{index:02d}"
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/"solve.json").exists():
        return
    report=dict(labels=row,development_group=row.get("development_group", "t3800_g1.00_m-0.75_CNO_corners"),arms={},arm_errors={})
    for arm,profile,cap in zip(("physical","reference"),profiles,(90,120)):
        try:
            seed=atmosphere(profile,row)
            rebuilt=reconstruct_full_atmosphere(ReducedAtmosphere(profile[:,0],profile[:,1],
                {f:row[f] for f in FIELDS}),seed=seed,n_synchronizations=None,
                max_synchronizations=8,pressure_tolerance_dex=1e-3)
            seed_path=directory/f"{arm}_seed.npz"
            np.savez(seed_path,start=np.column_stack([getattr(rebuilt.atmosphere,f) for f in INITIALIZER_OUTPUT_FIELDS]))
            report["arms"][arm]=run(directory,seed_path,row=row,cap=cap,arm=arm)
            report["arms"][arm]["rebuild_evaluations"]=rebuilt.n_evaluations
        except Exception as exc:
            report["arm_errors"][arm]=f"{type(exc).__name__}: {exc}"
        print(json.dumps(dict(case=index,arm=arm,error=report["arm_errors"].get(arm))),flush=True)
    (directory/"solve.json").write_text(json.dumps(report,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("run","evaluate"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--checkpoints",type=Path)
    parser.add_argument("--marcs-grid",type=Path)
    parser.add_argument("--gate",type=Path)
    args=parser.parse_args()
    args.root.mkdir(parents=True,exist_ok=True)
    if args.stage=="evaluate":
        reports=[]
        for index in range(8):
            directory=args.root/f"case_{index:02d}"
            evaluate(directory,args.gate)
            reports.append(json.loads((directory/"spectral_stability.json").read_text()))
        summary=dict(expected=8,completed=len(reports),population="development corners at one fixed Teff/logg/metallicity",
            consistent_pairs=sum(r.get("spectral_consistency",False) for r in reports),rows=reports,
            training_admission="strict restart remains required")
        (args.root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        return
    base=dict(effective_temperature=3800.,log_surface_gravity=1.,metallicity=-.75,
              alpha_enhancement=0.,microturbulence_km_s=2.)
    labels=np.asarray([[base[f] for f in FIELDS]])
    torch.set_num_threads(1)
    checkpoints=sorted(args.checkpoints.glob("checkpoint_mstar_seed*.pt"))
    if len(checkpoints)!=3:raise ValueError("the frozen v4 predictor requires three checkpoints")
    predictions=[]
    for path in checkpoints:
        model,scaler=load_physical_checkpoint(path)[:2]
        predictions.append(predict_physical_state(model,scaler,labels))
    mass,temperature=np.median(np.asarray(predictions),axis=0)
    learned=np.column_stack((mass[0],temperature[0]))
    marcs=load_marcs_node(args.marcs_grid,StellarLabels(**base),verify_sha256=False,
                         expected_sha256=None,depth_coordinate="log_mass")
    native=np.column_stack((marcs.reduced_column_mass,marcs.reduced_temperature))
    payloads=[]
    for index,(c,n,o) in enumerate(itertools.product((-.1,.1),repeat=3)):
        row=dict(base,carbon_enhancement=c,nitrogen_enhancement=n,oxygen_enhancement=o,
                 absolute_abundance_offsets={6:base["metallicity"]+c,7:base["metallicity"]+n,8:base["metallicity"]+o})
        payloads.append((index,row,(learned,native),args.root))
    (args.root/"setup.json").write_text(json.dumps(dict(labels=[p[1] for p in payloads],
        starts=["frozen v4 (m,T)","native MARCS (m,T) at base composition"],
        roles="development; no sealed holdout",workers=8,threads_per_worker=8,
        numerical_stops_released=True,physical_seed="Pgas=g*m",production_defaults_unchanged=True),indent=2)+"\n")
    with ProcessPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker,payloads))


if __name__=="__main__":
    main()
