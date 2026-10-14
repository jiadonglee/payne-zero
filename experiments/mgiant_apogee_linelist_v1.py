#!/usr/bin/env python3
"""Payne Zero synthesis with the APOGEE DR17 line list for the DR20 M giants.

The APOGEE line list (Smith et al. 2021; ``turbospec.20180901t20.{atoms,molec}``
from the DR17 SAS, air wavelengths) is converted to Payne Zero's inputs:

- atoms: Payne Zero's parsed-source schema, through its own catalog builder,
  replacing the native metal lines; hydrogen and helium lines stay native;
- molecules: compiled molecular arrays for CN, CO, OH, C2 and FeH, replacing the
  native lines of these five species; isotopologue lines carry solar isotopic
  fractions in log gf; the other native molecules (H2O, TiO, ...) stay.

Continuum opacity, transfer tables, equation of state and partition functions
stay Payne Zero's. Arms: ``pz`` (native lines + Sun/Arcturus calibration),
``apogee_atoms``, ``apogee_molec``, ``apogee_all``.

Stage ``fixed``: at the ten-parameter best-fit labels of each star
(``results/mgiant_apogee_pzfit_v1/garching/free_stars_sun_arcturus_fts_hband_shared``),
synthesize every arm and score it on the common interior pixels used for the
Astra comparison. Labels are held fixed, so this measures the line list only.
``--labels astra`` holds instead the raw FERRE labels of the Astra ASPCAP fit
(Teff, logg, [M/H], [alpha/M], [C/M], [N/M], vmic), with the radial velocity
and Gaussian broadening of the ten-parameter fit and no rotation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from experiments import mgiant_apogee_pzfit_v1 as fit
from experiments import mgiant_eso_bestfit_v1 as eso

ROOT = fit.PROJECT_ROOT
LINELIST_DIR = ROOT / "data" / "apogee_linelist" / "dr17"
ATOMS = LINELIST_DIR / "turbospec.20180901t20.atoms"
MOLEC = LINELIST_DIR / "turbospec.20180901t20.molec"
CALIBRATION = ROOT / "linelist_calibration" / "data" / "sun_arcturus_fts_hband_shared.npz"
RESULT_DIR = fit.RESULT_ROOT / "apogee_linelist_v1"
FREE_DIR = fit.RESULT_ROOT / "garching" / "free_stars_sun_arcturus_fts_hband_shared"
ARMS = ("pz", "apogee_atoms", "apogee_molec", "apogee_all")

# Turbospectrum molecule code -> (Payne Zero species id, isotopologue log fraction).
# Solar isotopic fractions (Asplund et al. 2009): 12C 0.9893, 13C 0.0107,
# 14N 0.99636, 16O 0.99757, 17O 0.00038, 18O 0.00205, 56Fe 0.91754.
MOLECULES = {
    "0607.012014": ("CN", 270, np.log10(0.9893 * 0.99636)),
    "0607.013014": ("CN", 270, np.log10(0.0107 * 0.99636)),
    "0608.012016": ("CO", 276, np.log10(0.9893 * 0.99757)),
    "0608.012017": ("CO", 276, np.log10(0.9893 * 0.00038)),
    "0608.012018": ("CO", 276, np.log10(0.9893 * 0.00205)),
    "0608.013016": ("CO", 276, np.log10(0.0107 * 0.99757)),
    "0108.001016": ("OH", 258, np.log10(0.99757)),
    "0606.012012": ("C2", 264, np.log10(0.9893 ** 2)),
    "0126.001056": ("FeH", 444, np.log10(0.91754)),
}
REPLACED_SPECIES = sorted({v[1] for v in MOLECULES.values()})


def read_turbospec(path: Path) -> list[tuple[str, int, np.ndarray]]:
    """Blocks of (header code, ionization stage, rows[wave_air_A, chi_eV, loggf, fdamp, gu, raddmp, stark])."""

    blocks, lines, i = [], path.read_text().splitlines(), 0
    while i < len(lines):
        if lines[i].startswith("'"):
            code = lines[i].split("'")[1].strip()
            ion = int(lines[i].split("'")[2].split()[0])
            i += 2
            rows = []
            while i < len(lines) and not lines[i].startswith("'"):
                p = lines[i].split()
                rows.append([float(x) for x in p[:7]])
                i += 1
            blocks.append((code, ion, np.asarray(rows, np.float64)))
        else:
            i += 1
    return blocks


def apogee_atomic_parsed(window_nm) -> dict[str, np.ndarray]:
    """APOGEE metal lines in Payne Zero's parsed-source schema (vacuum wavelengths)."""

    wave, chi, loggf, vdw, rad, stark, species = [], [], [], [], [], [], []
    for code, ion, rows in read_turbospec(ATOMS):
        z = int(float(code))
        if z <= 2 or rows.size == 0:
            continue
        wave.append(eso.air_to_vacuum_A(rows[:, 0]) / 10.0)
        chi.append(rows[:, 1]); loggf.append(rows[:, 2]); vdw.append(rows[:, 3])
        rad.append(rows[:, 5]); stark.append(rows[:, 6])
        species.append(np.full(rows.shape[0], z + (ion - 1) / 100.0))
    wave, chi, loggf, vdw, rad, stark, species = map(np.concatenate, (wave, chi, loggf, vdw, rad, stark, species))
    keep = (wave > window_nm[0] - 1.0) & (wave < window_nm[1] + 1.0)
    wave, chi, loggf, vdw, rad, stark, species = (a[keep] for a in (wave, chi, loggf, vdw, rad, stark, species))
    n = wave.size
    lower_cm = chi * eso.CM_PER_EV
    return {
        "stored_wavelength_nm": wave,
        "raw_log_oscillator_strength": loggf,
        "species_code": species,
        "first_energy_column_cm": lower_cm,
        "second_energy_column_cm": lower_cm + 1.0e7 / wave,
        "radiative_damping_log": np.where(rad > 0.0, np.log10(np.where(rad > 0.0, rad, 1.0)), 0.0),
        "stark_damping_log": np.where(stark < 0.0, stark, 0.0),
        "van_der_waals_damping_log": np.where(vdw < 0.0, vdw, 0.0),
        "lower_principal_quantum_number": np.zeros(n, np.int64),
        "upper_principal_quantum_number": np.zeros(n, np.int64),
        "primary_isotope_number": np.zeros(n, np.int64),
        "primary_isotope_log_correction": np.zeros(n),
        "secondary_isotope_log_correction": np.zeros(n),
        "energy_shift_field": np.full(n, b" " * 10, dtype="S10"),
        "isotope_shift_units": np.zeros(n),
        "line_size": np.zeros(n, np.int64),
        "line_category_tag": np.full(n, b"", dtype="S3"),
    }


def apogee_molecular_compiled(native: dict, molecules: dict = MOLECULES) -> tuple[dict, dict]:
    """APOGEE CN, CO, OH, C2 and FeH lines as compiled arrays on the native grid."""

    ratio, origin = float(native["log_grid_ratio"]), int(native["grid_origin_index"])
    grid_wave = lambda index: np.exp((np.asarray(index, np.float64) - 1 + origin) * ratio)
    parts, counts = [], {}
    for code, _, rows in read_turbospec(MOLEC):
        if code not in molecules:
            continue
        name, species_id, log_fraction = molecules[code]
        wave = eso.air_to_vacuum_A(rows[:, 0]) / 10.0
        keep = (wave > grid_wave(1)) & (wave < grid_wave(native["center_index_1based"].max()))
        wave, chi, loggf = wave[keep], rows[keep, 1], rows[keep, 2] + log_fraction
        freq = eso.LIGHT_SPEED_NM_S / wave
        ref = native["species_code"] == species_id
        ref_freq = np.median(eso.LIGHT_SPEED_NM_S / grid_wave(native["center_index_1based"][ref]))
        damping = {key: (np.median(native[key][ref]) * ref_freq / freq).astype(np.float32)
                   for key in ("radiative_damping", "stark_damping", "van_der_waals_damping")}
        parts.append({
            "center_index_1based": (np.rint(np.log(wave) / ratio) - origin + 1).astype(np.int32),
            "classical_line_strength": (10.0 ** loggf * 0.01502 / freq).astype(np.float32),
            "species_code": np.full(wave.size, species_id, np.int16),
            "lower_excitation_cm": (chi * eso.CM_PER_EV).astype(np.float32),
            **damping,
            "margin_class": np.full(wave.size, 7, np.int16),
        })
        counts[code] = int(wave.size)
    extra = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    kept = ~np.isin(native["species_code"], REPLACED_SPECIES)
    merged = {key: (np.concatenate([native[key][kept], extra[key]]) if np.ndim(native[key]) else native[key])
              for key in native}
    inventory = {"apogee_lines_by_block": counts,
                 "native_lines_removed": {int(s): int((native["species_code"] == s).sum()) for s in REPLACED_SPECIES},
                 "native_lines_kept": int(kept.sum())}
    return merged, inventory


def _molecular_invariants(base, compiled, device):
    import torch

    from payne_zero_synthesis import line_opacity as lo
    from payne_zero_synthesis import molecular_lines as ml
    from payne_zero_synthesis import pipeline as pz

    tables = np.load(pz._SYNTHESIS_TABLE_DIR / "line_profile_tables.npz", allow_pickle=False)
    catalog = ml.build_catalog_from_arrays(compiled)
    catalog.center_index = lo.nearest_grid_indices(base.synthesis_wavelength_nm, catalog.wavelength_nm).astype(np.int32)
    inv = ml.precompute_invariants(catalog=catalog, wavelength_grid_nm=base.synthesis_wavelength_nm,
                                   harris_profile_h0_table=tables["harris_profile_h0_table"],
                                   harris_profile_h1_table=tables["harris_profile_h1_table"],
                                   harris_profile_h2_table=tables["harris_profile_h2_table"],
                                   runtime_device=torch.device(device))
    return inv, len(catalog)


def _atomic_invariants(base, parsed, scratch: Path, device):
    import torch

    from payne_zero_synthesis import atomic_lines
    from payne_zero_synthesis import line_opacity as lo
    from payne_zero_synthesis import pipeline as pz

    scratch.mkdir(parents=True, exist_ok=True)
    path = scratch / "apogee_atomic_parsed.npz"
    np.savez(path, **parsed)
    tables = np.load(pz._SYNTHESIS_TABLE_DIR / "line_profile_tables.npz", allow_pickle=False)
    catalog = atomic_lines.parse_catalog(base.grid_obj, catalog_path=path, sort="catalog", apply_iso_corr=False)
    kernel = pz._atomic_catalog_for_kernels(catalog, tables)
    line_type = kernel["line_type"]
    kernel["helium_line_type"] = line_type[np.isin(line_type, [-3, -4, -6])].astype(np.int64)
    kernel["helium_line_center_cutoff_ratio"] = np.float64(lo.LINE_CENTER_CUTOFF_RATIO)
    metal = np.flatnonzero((line_type == 0) | (line_type == 1) | (line_type == 3))
    chunk = pz.SynthesisPipeline.METAL_CHUNK
    chunks = [lo.precompute_invariants(pz.SynthesisPipeline._slice_atomic_catalog(kernel, metal[s:s + chunk]),
                                       base.synthesis_wavelength_nm, runtime_device=torch.device(device))
              for s in range(0, metal.size, chunk)]
    return chunks, len(catalog), int(metal.size)


def build_bundles(device: str, dtype: str, scratch: Path) -> tuple[dict, dict]:
    import torch

    from payne_zero_synthesis import pipeline as pz

    base = pz.window_invariants_for(wl_start_nm=fit.SYNTH_VAC_NM[0], wl_end_nm=fit.SYNTH_VAC_NM[1],
                                    resolution=fit.R_GRID, molecular_lines=True,
                                    runtime_device=torch.device(device),
                                    work_dtype=torch.float64 if dtype == "float64" else torch.float32)
    calibrated, _ = fit.calibrated_bundle(CALIBRATION, device=device, dtype=dtype)
    native = pz.SynthesisPipeline._compile_molecular(fit.SYNTH_VAC_NM[0], fit.SYNTH_VAC_NM[1], fit.R_GRID)
    merged, molecular_inventory = apogee_molecular_compiled(native)
    mol_inv, n_mol = _molecular_invariants(base, merged, device)
    parsed = apogee_atomic_parsed(fit.SYNTH_VAC_NM)
    metal_chunks, n_atomic, n_metal = _atomic_invariants(base, parsed, scratch, device)
    bundles = {
        "pz": calibrated,
        "apogee_atoms": dataclasses.replace(base, n_atomic=n_atomic, metal_invariant_chunks=metal_chunks),
        "apogee_molec": dataclasses.replace(calibrated, molecular_invariants=mol_inv, n_molecular=n_mol),
        "apogee_all": dataclasses.replace(base, n_atomic=n_atomic, metal_invariant_chunks=metal_chunks,
                                          molecular_invariants=mol_inv, n_molecular=n_mol),
    }
    inventory = {"atomic_exported": int(parsed["stored_wavelength_nm"].size), "atomic_metal_used": n_metal,
                 "molecular": molecular_inventory, "molecular_total": n_mol,
                 "note": "apogee_atoms and apogee_all do not carry the Sun/Arcturus overlay, which belongs to the "
                         "native atomic catalog"}
    return bundles, inventory


def stage_fixed(args) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp())
    started = time.time()
    bundles, inventory = build_bundles(args.device, args.dtype, scratch)
    print(f"bundles built in {time.time() - started:.0f} s: {json.dumps(inventory)}", flush=True)
    rows = json.loads((fit.RESULT_ROOT / "astra_comparison.json").read_text())
    if args.labels == "astra":
        rows = [r for r in rows if r["astra"] is not None]
    predictions = np.load(fit.RESULT_ROOT / "astra_predictions.npz", allow_pickle=False)
    records, arrays = [], {}
    for row in rows:
        sid = row["sdss_id"]
        q = dict(json.loads((FREE_DIR / str(sid) / "summary.json").read_text())["physical"]["parameters"])
        if args.labels == "astra":
            a = row["astra"]
            q.update(teff=a["teff"], logg=a["logg"], m_h=a["m_h_atm"], alpha_m=a["alpha_m_atm"], c_m=a["c_m_atm"],
                     n_m=a["n_m_atm"], vmic_km_s=a["v_micro"], vsini_km_s=0.0)
        obs = fit.ApogeeObservation(sid)
        pixels = predictions[f"{sid}_pixels"]
        record = {"sdss_id": sid, "labels": q, "chi2_per_pixel": {}, "chips": {},
                  "astra_chi2_per_pixel": row["metrics"].get("astra", {}).get("chi2_per_pixel")}
        for arm in args.arms:
            model = fit.FreeAbundanceFastModel(obs, device=args.device, dtype=args.dtype,
                                               cache_dir=scratch / f"{sid}_{arm}", bundle=bundles[arm])
            path = model.base_atmosphere(q["teff"], q["logg"], q["m_h"])
            native = model.synthesize_perturbed(path, q["alpha_m"], q["c_m"], q["n_m"], q["vmic_km_s"])
            projected = model.project_free(native, q["rv_km_s"], q["sigma_extra_km_s"], q["vsini_km_s"])
            prediction = fit._profiled_fit(obs, projected, pixels)
            residual = obs.flux - prediction
            record["chi2_per_pixel"][arm] = float(np.mean(obs.weight[pixels] * residual[pixels] ** 2))
            record["chips"][arm] = {f"{lo:.0f}-{hi:.0f}": float(np.mean(
                obs.weight[pixels & (obs.wave_nm >= lo) & (obs.wave_nm <= hi)]
                * residual[pixels & (obs.wave_nm >= lo) & (obs.wave_nm <= hi)] ** 2)) for lo, hi in fit.CHIPS_NM}
            arrays[f"{sid}_model_{arm}"] = prediction.astype(np.float32)
        records.append(record)
        print(sid, {k: round(v, 1) for k, v in record["chi2_per_pixel"].items()},
              "astra", record["astra_chi2_per_pixel"] and round(record["astra_chi2_per_pixel"], 1), flush=True)
    tag = "" if args.labels == "free" else f"_{args.labels}"
    (RESULT_DIR / f"fixed_label_chi2{tag}.json").write_text(json.dumps(
        {"labels": args.labels, "inventory": inventory, "stars": records}, indent=2))
    np.savez_compressed(RESULT_DIR / f"fixed_label_predictions{tag}.npz", **arrays)


def stage_diagnose(args) -> None:
    """Where the ten-parameter fit loses to the Astra model, and one feature without a Payne Zero line."""

    from payne_zero_synthesis import synthesize

    arm = "free_stars_sun_arcturus_fts_hband_shared"
    rows = json.loads((fit.RESULT_ROOT / "astra_comparison.json").read_text())
    predictions = np.load(fit.RESULT_ROOT / "astra_predictions.npz", allow_pickle=False)
    stars = []
    for row in rows:
        sid = row["sdss_id"]
        if row["astra"] is None:
            continue
        obs = fit.ApogeeObservation(sid)
        pixels = predictions[f"{sid}_pixels"]
        flux = predictions[f"{sid}_flux"].astype(np.float64)
        excess = (obs.weight * (flux - predictions[f"{sid}_model_{arm}"]) ** 2
                  - obs.weight * (flux - predictions[f"{sid}_model_astra"]) ** 2)[pixels]
        ordered = np.cumsum(np.sort(excess)[::-1]) / excess.sum()
        stars.append({"sdss_id": sid, "pixels": int(pixels.sum()),
                      "pixel_fraction_for_half": float((np.searchsorted(ordered, 0.5) + 1) / excess.size),
                      "pixel_fraction_for_80pc": float((np.searchsorted(ordered, 0.8) + 1) / excess.size)})
    sid, feature_nm = args.star, args.feature_nm
    q = json.loads((FREE_DIR / str(sid) / "summary.json").read_text())["physical"]["parameters"]
    obs = fit.ApogeeObservation(sid)
    model = fit.FreeAbundanceFastModel(obs, device=args.device, dtype=args.dtype,
                                       cache_dir=Path(tempfile.mkdtemp()))
    path = model.base_atmosphere(q["teff"], q["logg"], q["m_h"])
    spectrum = synthesize(path, wavelength_start_nm=fit.SYNTH_VAC_NM[0], wavelength_end_nm=fit.SYNTH_VAC_NM[1],
                          resolution=fit.R_GRID, molecular_lines=False, device=args.device, dtype=args.dtype)
    atomic = model.project((np.asarray(spectrum.wavelength_nm, np.float64), np.asarray(spectrum.flux_total, np.float64),
                            np.asarray(spectrum.flux_continuum, np.float64)), q["rv_km_s"], q["sigma_extra_km_s"])
    flux = predictions[f"{sid}_flux"].astype(np.float64)
    window = np.flatnonzero(np.abs(obs.wave_nm - feature_nm) < 0.03)
    j = window[np.argmin(flux[window])]
    feature = {"sdss_id": sid, "wavelength_nm": float(obs.wave_nm[j]), "data": float(flux[j]),
               "astra": float(predictions[f"{sid}_model_astra"][j]),
               "payne_zero_ten_parameter": float(predictions[f"{sid}_model_{arm}"][j]),
               "payne_zero_atomic_only": float(atomic[j])}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "excess_diagnosis.json").write_text(json.dumps({"arm": arm, "stars": stars, "feature": feature},
                                                                 indent=2))
    print(json.dumps({"stars": stars, "feature": feature}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    fixed = sub.add_parser("fixed")
    fixed.add_argument("--device", default="mps")
    fixed.add_argument("--dtype", default="float32")
    fixed.add_argument("--arms", nargs="*", default=list(ARMS))
    fixed.add_argument("--labels", choices=("free", "astra"), default="free")
    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--device", default="mps")
    diagnose.add_argument("--dtype", default="float32")
    diagnose.add_argument("--star", type=int, default=96257659)
    diagnose.add_argument("--feature-nm", type=float, default=1672.746)
    args = parser.parse_args()
    {"fixed": stage_fixed, "diagnose": stage_diagnose}[args.stage](args)


if __name__ == "__main__":
    main()
