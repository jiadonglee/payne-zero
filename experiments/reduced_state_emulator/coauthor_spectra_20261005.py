"""Spectral comparisons of starts and converged test atmospheres on Garching."""
from bench import environment as _environment
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from coauthor_benchmark_20261005 import load_data, atmosphere
from payne_zero_atmosphere.atmosphere_io import linear_elemental_abundances
from payne_zero_synthesis import build_structured_atmosphere, synthesize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, default=Path("results/coauthor_validation_20261005"))
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--wait-for-restarts", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args=parser.parse_args()
    torch.set_num_threads(8)
    data=load_data(args.corpus_root,"five_label",args.result_root)
    indices=data["split"]["test"][:args.count][args.shard::args.shards]
    args.out.mkdir(parents=True,exist_ok=True)
    def spectrum(profile,row):
        model=atmosphere(profile,row)
        columns=build_structured_atmosphere(temperature=model.temperature,column_mass=model.column_mass,
            gas_pressure=model.gas_pressure,electron_density=model.electron_density,
            elemental_abundances=linear_elemental_abundances(model),microturbulence=model.microturbulence,
            molecular_lines=True,device="cuda",dtype="float64")
        result=synthesize(columns,wavelength_start_nm=400,wavelength_end_nm=900,resolution=20000,
            molecular_lines=True,device="cuda",dtype="float64")
        if not all(np.isfinite(x).all() for x in (result.normalized_flux,result.flux_total,result.flux_continuum)):
            raise ValueError("nonfinite synthetic flux")
        return result
    arms=("exact_rebuild","two_profile","six_profile","nearest_0","nearest_1","grey_convective")
    for index in indices:
        destination=args.out/f"star_{index:06d}.json"
        if destination.exists(): continue
        directory=args.runs/f"star_{index:06d}"
        while not (directory/"record.json").exists():
            if not args.wait_for_restarts: break
            time.sleep(30)
        if not (directory/"record.json").exists(): continue
        record=json.loads((directory/"record.json").read_text())
        row=data["rows"][index]
        report=dict(index=index,labels=row,wavelength_nm=[400,900],resolution=20000,
                    numerically_converged={a:e["converged"] for a,e in record["arms"].items()},arms={})
        try:
            started=time.perf_counter()
            reference=spectrum(data["profiles"][index],row)
            report["reference_seconds"]=time.perf_counter()-started
        except Exception as exc:
            report["reference_error"]=f"{type(exc).__name__}: {exc}"
            destination.write_text(json.dumps(report,indent=2)+"\n")
            print(json.dumps(report),flush=True)
            continue
        final_spectra={}
        for arm in arms:
            entry={}; path=directory/f"{arm}.npz"
            try:
                with np.load(path,allow_pickle=False) as d:
                    start=d["start"].copy()
                    final=d["final"].copy() if "final" in d else None
                tick=time.perf_counter()
                predicted=spectrum(start,row)
                entry.update(start_seconds=time.perf_counter()-tick,
                    maximum_normalized_flux_difference=float(np.max(np.abs(predicted.normalized_flux-reference.normalized_flux))),
                    maximum_continuum_scaled_total_difference=float(np.max(np.abs(predicted.flux_total-reference.flux_total)/reference.flux_continuum)),
                    maximum_relative_continuum_difference=float(np.max(np.abs(predicted.flux_continuum/reference.flux_continuum-1))))
                if arm in ("two_profile","six_profile") and final is not None:
                    final_spectra[arm]=spectrum(final,row)
                    entry["final_maximum_normalized_flux_difference"]=float(np.max(np.abs(final_spectra[arm].normalized_flux-reference.normalized_flux)))
            except Exception as exc:
                entry["error"]=f"{type(exc).__name__}: {exc}"
            report["arms"][arm]=entry
            print(json.dumps(dict(index=index,arm=arm,**entry)),flush=True)
        if len(final_spectra)==2:
            report["two_vs_six_final_maximum_normalized_flux_difference"]=float(np.max(np.abs(
                final_spectra["two_profile"].normalized_flux-final_spectra["six_profile"].normalized_flux)))
        destination.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":
    main()
