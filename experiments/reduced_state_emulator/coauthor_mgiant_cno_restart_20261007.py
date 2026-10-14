"""Strict self-restarts and corpus admission for the eight CNO development points."""
from bench import environment as _environment

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from bench.run_reference import _solver_config
from payne_zero_atmosphere.runner import run_atmosphere_model
from payne_zero_atmosphere.warm_start import INITIALIZER_OUTPUT_FIELDS, INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH
from reduced_state.reconstruct import ReducedAtmosphere, reconstruct_full_atmosphere
from coauthor_benchmark_20261005 import atmosphere, FIELDS


def wait_for(path):
    while not path.exists():
        time.sleep(30)


def worker(payload):
    index,root,gate=payload
    directory=root/f"case_{index:02d}"
    previous=json.loads((directory/"spectral_stability.json").read_text())
    report=dict(index=index,passes=False)
    if previous.get("spectral_consistency"):
        try:
            row=previous["labels"]
            row["absolute_abundance_offsets"]={int(z):v for z,v in row["absolute_abundance_offsets"].items()}
            n=previous["arms"]["physical"]["frozen_iteration"]
            product=directory/"physical"/f"iter_{n:04d}.npz"
            with np.load(product) as data:
                mass,temperature=data["column_mass"].copy(),data["temperature"].copy()
            profile=np.column_stack((mass,temperature))
            seed=atmosphere(profile,row)
            rebuilt=reconstruct_full_atmosphere(ReducedAtmosphere(mass,temperature,{f:row[f] for f in FIELDS}),
                seed=seed,n_synchronizations=None,max_synchronizations=8,pressure_tolerance_dex=1e-3)
            config=_solver_config(rebuilt.atmosphere,iterations_per_trial=120,
                structured_atmosphere_path=directory/"strict_restart.npz",debug_state_path=None)
            result=run_atmosphere_model(config)
            finite=all(np.isfinite(getattr(result.atmosphere,f)).all() for f in INITIALIZER_OUTPUT_FIELDS)
            positive=all((getattr(result.atmosphere,f)>0).all() for f in INITIALIZER_OUTPUT_FIELDS)
            monotone=bool((np.diff(result.atmosphere.column_mass)>0).all())
            last=result.diagnostics["iteration_timings"][-1]
            flux_pass=all(last[k]<=v for k,v in gate.items())
            dT=float(np.percentile(abs(result.atmosphere.temperature/temperature-1),95))
            dm=float(np.percentile(abs(np.log10(result.atmosphere.column_mass/mass)),95))
            report.update(labels=row,numerically_converged=bool(result.converged),finite=finite,
                positive=bool(positive),monotone_column_mass=monotone,
                iterations=result.iterations_completed,flux_pass=flux_pass,final_diagnostics=last,
                temperature_p95=dT,mass_dex_p95=dm,reference_product=str(product),
                structure_limits=dict(temperature_p95=3e-3,mass_dex_p95=7.7e-3),
                passes=bool(result.converged and finite and positive and monotone and flux_pass and dT<=3e-3 and dm<=7.7e-3))
            np.savez(directory/"strict_restart_profiles.npz",
                     profiles=np.column_stack([getattr(result.atmosphere,f) for f in INITIALIZER_OUTPUT_FIELDS]))
        except Exception as exc:
            report["error"]=f"{type(exc).__name__}: {exc}"
    else:
        report["reason"]="independent paths did not pass the spectral criterion"
    (directory/"strict_restart.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(index=index,passes=report["passes"],error=report.get("error"))),flush=True)


def spectrum(path):
    from payne_zero_synthesis import build_structured_atmosphere,synthesize
    with np.load(path) as data:
        columns=build_structured_atmosphere(temperature=data["temperature"],column_mass=data["column_mass"],
            gas_pressure=data["gas_pressure"],electron_density=data["electron_density"],
            elemental_abundances=data["elemental_abundances"],microturbulence=data["microturbulence"],
            molecular_lines=True,device="cuda",dtype="float64")
    result=synthesize(columns,wavelength_start_nm=665,wavelength_end_nm=667,resolution=20000,
                      molecular_lines=True,device="cuda",dtype="float64")
    arrays=tuple(np.asarray(a) for a in (result.normalized_flux,result.flux_total,result.flux_continuum))
    if not all(np.isfinite(a).all() for a in arrays):raise ValueError("nonfinite restart spectrum")
    return arrays


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("run","evaluate"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--gate",type=Path,default=Path("flux_gate.json"))
    args=parser.parse_args()
    wait_for(args.root/"summary.json")
    if args.stage=="run":
        gate=json.loads(args.gate.read_text())["thresholds"]
        with ProcessPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker,[(i,args.root,gate) for i in range(8)]))
        return
    import torch
    torch.set_num_threads(8)
    records=[];labels=[];profiles=[];label_rows=[]
    for index in range(8):
        directory=args.root/f"case_{index:02d}"
        wait_for(directory/"strict_restart.json")
        report=json.loads((directory/"strict_restart.json").read_text())
        if report["passes"]:
            try:
                a=spectrum(directory/"strict_restart.npz")
                b=spectrum(Path(report["reference_product"]))
                metrics=dict(normalized_flux=float(np.max(abs(a[0]-b[0]))),
                    flux_total=float(np.max(abs(a[1]-b[1])/b[2])),
                    flux_continuum=float(np.max(abs(a[2]/b[2]-1))))
                report["spectral_metrics"]=metrics
                report["passes"]=max(metrics.values())<=5e-3
            except Exception as exc:
                report.update(passes=False,error=f"{type(exc).__name__}: {exc}")
        if report["passes"]:
            row=report["labels"]
            labels.append([row[f] for f in (*FIELDS,"carbon_enhancement","nitrogen_enhancement","oxygen_enhancement")])
            label_rows.append(json.dumps(row))
            with np.load(directory/"strict_restart_profiles.npz") as data:profiles.append(data["profiles"].copy())
        records.append(report)
    np.savez(args.root/"cno_development_corpus.npz",labels=np.asarray(labels).reshape(-1,8),
        atmosphere_profiles=np.asarray(profiles).reshape(-1,80,6),
        standard_rosseland_optical_depth=np.broadcast_to(INITIALIZER_STANDARD_ROSSELAND_OPTICAL_DEPTH,(len(labels),80)),
        labels_json=np.asarray(label_rows,dtype=str),
        parent_groups=np.asarray(["t3800_g1.00_m-0.75"]*len(labels)),
        roles=np.asarray(["development_training"]*len(labels)))
    (args.root/"admission.json").write_text(json.dumps(dict(expected=8,completed=8,
        admitted=len(labels),population="one development parent group; no holdout validation",
        wavelength_nm=[665,667],resolution=20000,spectral_threshold=5e-3,
        flux_thresholds=json.loads(args.gate.read_text())["thresholds"],rows=records),indent=2)+"\n")
    print(json.dumps(dict(admitted=len(labels),expected=8)),flush=True)


if __name__=="__main__":
    main()
