#!/usr/bin/env python3
"""Best-fit comparison of Payne-Zero and Korg.jl + MARCS on ESO UVES M giants.

Stages (run in order; each is resumable):

    fetch     ESO ObsCore query, Phase 3 UVES download, window extraction
    pz        native Payne-Zero spectra for every converged M-giant node
    export    Payne-Zero node atmospheres and label tables for Julia
    (julia)   experiments/mgiant_eso_bestfit_v1_korg.jl
    fit       one shared fitter over every model arm
    report    metrics table and figures
    linelists per-species line counts of both synthesis windows
    pz-fit    Payne Zero's own fitter: continuous fast fit, then converged-
              atmosphere refinement (heavy; run on Garching)
    pz-fit-collect  gather the per-star pz-fit summaries into one record
    pz-galah  native Payne Zero synthesis of the 78 saved atmospheres with the
              GALAH DR3 line list that Korg uses, then the shared node fit
    zro-table Payne Zero synthesis molecule table extended by Zr and ZrO,
              with ZrO constants taken from Korg (shadow source-catalog root);
              reads zro_equilibrium/korg_molecular_constants.tsv, written by
              experiments/mgiant_eso_bestfit_v1_korg_constants.jl
    pz-galah-zro  rebuild the saved atmospheres' EOS state with that table and
              synthesize with the full GALAH list, ZrO included

Model arms share the observation, pixel mask, wavelength conversion,
instrumental kernel, broadening/RV nuisance grid and continuum polynomial;
only the intrinsic normalized spectrum differs:

    pz_nodes      native Payne-Zero atmosphere + synthesis, 78 corpus nodes
    marcs_nodes   Korg + MARCS (interpolate_marcs), the same 78 labels
    korg_pzatm    Korg synthesis on the 78 Payne-Zero atmospheres
    marcs_dense   Korg + MARCS on a regular grid, continuous trilinear fit
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-mgiant-eso-bestfit-v1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault(
    "PAYNE_ZERO_SYNTHESIS_CACHE_DIR", str(PROJECT_ROOT / ".cache/payne-zero/synthesis")
)

RESULT_ROOT = PROJECT_ROOT / "results" / "mgiant_eso_bestfit_v1"
CORPUS = PROJECT_ROOT / "results/m_star_cool_corpus_mgiant_v2/cool_truth_corpus.npz"
BORISOV_TABLE = PROJECT_ROOT / "results/m_star_eso_highres_comparison_v1/sources/borisov2023_table6.dat"
GBS_CATALOG = Path("/Users/jdli/Project/jorg/gaia_fgk_benchmark_stars/metadata/catalog_with_local_paths.csv")

ESO_TAP = "https://archive.eso.org/tap_obs/sync"
ESO_FILE = "https://dataportal.eso.org/dataPortal/file/"

# Fit window: GALAH DR3 band 3, the only optical range where Korg's bundled
# line list carries TiO. Air wavelengths, catalogue rest frame.
WINDOW_AIR_A = (6480.0, 6735.0)
SYNTH_VAC_NM = (647.5, 674.7)
PZ_RESOLUTION = 600_000.0
# Chromospheric H-alpha is outside every LTE model here; the Li I 6708
# doublet measures a depleted, star-specific Li abundance, not the atmosphere.
MASKS_AIR_A = ((6555.0, 6572.0), (6707.0, 6708.8))
INSTRUMENT_R = 74_450.0
VMIC_KM_S = 2.0

# (key, Borisov UVES-POP name, spectral type, GBS id or None)
TARGETS = (
    ("alfCet", "HD018884", "alf Cet", "alfCet"),
    ("alfTau", "Aldebaran", "alf Tau", "alfTau"),
    ("psiPhe", "HD011695", "psi Phe", None),
    ("phiAqr", "HD219215", "phi Aqr", None),
    ("mVir", "HD119149", "m Vir", None),
    ("87Vir", "HD120052", "87 Vir", None),
    ("ERVir", "HD123214", "ER Vir", None),
)


def _read_borisov() -> dict[str, dict]:
    rows = {}
    for line in BORISOV_TABLE.read_text().splitlines():

        def field(a: int, b: int) -> str:
            return line[a - 1 : b].strip()

        def number(a: int, b: int) -> float | None:
            text = field(a, b)
            return float(text) if text else None

        rows[field(1, 12)] = {
            "ra_deg": float(field(267, 276)),
            "dec_deg": float(field(278, 286)),
            "spectral_class": field(320, 339),
            "variable": field(396, 396) == "1",
            "gcvs_type": field(412, 417),
            "rv_km_s": number(437, 444),
            "vsini_km_s": number(452, 458),
            "teff_K": number(467, 473),
            "logg": number(482, 488),
            "feh": number(496, 502),
            "alpha_fe": number(510, 516),
        }
    return rows


def _read_gbs() -> dict[str, dict]:
    labels = {}
    with GBS_CATALOG.open() as handle:
        for row in csv.DictReader(handle):
            star = row["star"]
            if star in labels or not row["teff"]:
                continue
            fields = {
                "teff_K": "teff", "e_teff_K": "eteff", "logg": "logg", "e_logg": "elogg",
                "feh": "feh", "e_feh": "efeh", "vmic_km_s": "vmic", "vsini_km_s": "vsini",
            }
            labels[star] = {key: float(row[column]) if row[column] else None for key, column in fields.items()}
            labels[star]["source"] = "Gaia FGK benchmark stars (SVO GBS catalogue)"
    return labels


def _eso_tap(query: str) -> list[dict]:
    url = ESO_TAP + "?" + urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    )
    for attempt in range(4):
        try:
            text = urllib.request.urlopen(url, timeout=240).read().decode()
            return list(csv.DictReader(io.StringIO(text)))
        except OSError:
            if attempt == 3:
                raise
            time.sleep(10)
    raise AssertionError


def _download(url: str, path: Path) -> None:
    if path.is_file() and path.stat().st_size > 0:
        return
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=600) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    partial.rename(path)


def _read_uves_phase3(path: Path) -> dict[str, np.ndarray]:
    from astropy.io import fits

    with fits.open(path) as hdul:
        table = hdul[1].data
        names = [name.upper() for name in table.columns.names]
        flux_name = "FLUX" if "FLUX" in names else "FLUX_REDUCED"
        error_name = "ERR" if "ERR" in names else "ERR_REDUCED"
        wave = np.asarray(table["WAVE"][0], dtype=np.float64)
        flux = np.asarray(table[flux_name][0], dtype=np.float64)
        error = np.asarray(table[error_name][0], dtype=np.float64)
        quality = (
            np.asarray(table["QUAL"][0], dtype=np.int64) if "QUAL" in names else np.zeros(wave.size, np.int64)
        )
        header = dict(hdul[0].header)
    unit = str(header.get("CUNIT1", "")) or "Angstrom"
    return {
        "wave_air_A": wave,
        "flux": flux,
        "error": error,
        "quality": quality,
        "flux_column": flux_name,
        "header": {key: header[key] for key in ("OBJECT", "SPEC_RES", "SNR", "DATE-OBS", "PROG_ID", "ARCFILE", "SPECSYS", "BUNIT", "TELESCOP", "INSTRUME", "ESO QC VRAD BARYCOR") if key in header},
        "unit_hint": unit,
    }


def stage_fetch(args) -> None:
    out = RESULT_ROOT / "observations"
    out.mkdir(parents=True, exist_ok=True)
    borisov = _read_borisov()
    gbs = _read_gbs()
    manifest = []
    for key, borisov_name, common, gbs_id in TARGETS:
        b = borisov[borisov_name]
        query = (
            "SELECT obs_collection, instrument_name, target_name, obs_publisher_did, "
            "em_min, em_max, em_res_power, snr, t_min, s_ra, s_dec, proposal_id, access_estsize "
            "FROM ivoa.ObsCore WHERE instrument_name='UVES' AND "
            f"CONTAINS(POINT('ICRS',s_ra,s_dec),CIRCLE('ICRS',{b['ra_deg']},{b['dec_deg']},0.01))=1"
        )
        rows = _eso_tap(query)
        usable = [
            row for row in rows
            if float(row["em_min"]) * 1e10 < WINDOW_AIR_A[0] - 30
            and float(row["em_max"]) * 1e10 > WINDOW_AIR_A[1] + 30
            and abs(float(row["em_res_power"]) - INSTRUMENT_R) < 1.0
        ]
        usable.sort(key=lambda row: -float(row["snr"] or 0.0))
        if not usable:
            raise RuntimeError(f"no UVES R={INSTRUMENT_R:.0f} product covers the window for {key}")
        chosen = usable[0]
        dp_id = chosen["obs_publisher_did"].split("?", 1)[1]
        fits_path = out / f"{key}_{dp_id.replace(':', '')}.fits"
        _download(ESO_FILE + dp_id, fits_path)
        spectrum = _read_uves_phase3(fits_path)
        manifest.append({
            "key": key,
            "common_name": common,
            "uvespop_name": borisov_name,
            "spectral_class": b["spectral_class"],
            "gcvs_variable": b["variable"],
            "gcvs_type": b["gcvs_type"],
            "eso_obscore": chosen,
            "eso_candidates_in_setting": len(usable),
            "eso_rows_near_target": len(rows),
            "fits_path": str(fits_path.relative_to(PROJECT_ROOT)),
            "fits_header": spectrum["header"],
            "flux_column": spectrum["flux_column"],
            "reference_labels": gbs.get(gbs_id) if gbs_id else None,
            "borisov2023_labels": {k: b[k] for k in ("teff_K", "logg", "feh", "alpha_fe", "rv_km_s", "vsini_km_s")},
        })
        print(f"{key:7s} {dp_id} SNR={chosen['snr']} R={chosen['em_res_power']} -> {fits_path.name}")
    (RESULT_ROOT / "targets.json").write_text(json.dumps(manifest, indent=2, default=str))


DENSE_TEFF = np.arange(3200.0, 4200.0 + 1, 50.0)
DENSE_LOGG = np.arange(0.0, 2.5 + 1e-9, 0.25)
DENSE_MH = np.arange(-1.5, 0.5 + 1e-9, 0.25)


def _nodes() -> list[dict]:
    """Converged M-giant corpus nodes whose products are on disk."""

    corpus = np.load(CORPUS, allow_pickle=False)
    nodes = []
    for labels, path, node_id in zip(corpus["labels"], corpus["source_product_paths"], corpus["node_ids"]):
        product = PROJECT_ROOT / str(path)
        if not product.is_file():
            continue
        teff, logg, m_h, alpha, vmic = (float(v) for v in labels)
        nodes.append({
            "model_id": f"t{teff:.0f}_g{logg:+.2f}_m{m_h:+.2f}",
            "teff": teff, "logg": logg, "m_h": m_h, "alpha": alpha, "vmic": vmic,
            "product_path": str(product), "node_id": str(node_id),
        })
    nodes.sort(key=lambda n: (n["m_h"], n["logg"], n["teff"]))
    return nodes


def stage_export(args) -> None:
    atm_dir = RESULT_ROOT / "pz_atmospheres"
    atm_dir.mkdir(parents=True, exist_ok=True)
    nodes = _nodes()
    columns = ("temperature", "electron_density", "gas_pressure", "mass_density", "column_mass")
    with (RESULT_ROOT / "node_labels.tsv").open("w") as table:
        table.write("model_id\tteff\tlogg\tm_h\tatmosphere_tsv\n")
        for node in nodes:
            with np.load(node["product_path"], allow_pickle=False) as data:
                matrix = np.column_stack([np.asarray(data[key], dtype=np.float64) for key in columns])
            tsv = atm_dir / f"{node['model_id']}.tsv"
            np.savetxt(tsv, matrix, delimiter="\t", comments="", header="\t".join(columns))
            table.write(f"{node['model_id']}\t{node['teff']}\t{node['logg']}\t{node['m_h']}\t{tsv}\n")
    with (RESULT_ROOT / "dense_labels.tsv").open("w") as table:
        table.write("model_id\tteff\tlogg\tm_h\n")
        for m_h in DENSE_MH:
            for logg in DENSE_LOGG:
                for teff in DENSE_TEFF:
                    table.write(f"t{teff:.0f}_g{logg:+.2f}_m{m_h:+.2f}\t{teff}\t{logg}\t{m_h}\n")
    (RESULT_ROOT / "nodes.json").write_text(json.dumps(nodes, indent=2))
    print(f"{len(nodes)} nodes, {DENSE_TEFF.size * DENSE_LOGG.size * DENSE_MH.size} dense labels")


def stage_pz(args) -> None:
    from payne_zero_synthesis.library import synthesize_library

    nodes = _nodes()
    build = synthesize_library(
        [n["product_path"] for n in nodes],
        wavelength_start_nm=SYNTH_VAC_NM[0],
        wavelength_end_nm=SYNTH_VAC_NM[1],
        resolution=PZ_RESOLUTION,
        molecular_lines=True,
        device=args.device,
        dtype="float32",
        workers=args.workers,
        cache_dir=RESULT_ROOT / "pz_cache",
        molecular_chunk_lines=65536,
        on_nonfinite="keep",
    )
    out = RESULT_ROOT / "pz_nodes.npz"
    np.savez_compressed(
        out,
        wavelength_vacuum_A=build.wavelength_nm * 10.0,
        normalized_flux=build.normalized_flux.astype(np.float32),
        model_id=np.array([n["model_id"] for n in nodes]),
        seconds=build.seconds,
        nonfinite=np.array(build.nonfinite),
        config_json=json.dumps(build.config),
    )
    print(f"wrote {out}: {build.normalized_flux.shape}, computed {len(build.computed)}, "
          f"cached {len(build.cache_hits)}, nonfinite {len(build.nonfinite)}")


C_KM_S = 299_792.458
GRID_DV_KM_S = 0.25
SIGMA_LSF_KM_S = C_KM_S / INSTRUMENT_R / (2.0 * np.sqrt(2.0 * np.log(2.0)))
SIGMA_EXTRA_GRID = np.arange(0.0, 8.0 + 1e-9, 0.5)
KNOT_SPACING_A = 25.0
ARMS = ("pz_nodes", "marcs_nodes", "korg_pzatm", "marcs_dense")
PZ_BOX = {"teff": (3000.0, 4000.0), "logg": (0.5, 2.5), "m_h": (-1.0, 0.5)}


def vacuum_to_air_A(vacuum_A):
    """Optical vacuum to air, the native Payne-Zero catalogue law."""

    vacuum_nm = np.asarray(vacuum_A, dtype=np.float64) / 10.0
    sigma2 = (1.0e7 / vacuum_nm) ** 2
    n = 1.0000834213 + 2406030.0 / (1.30e10 - sigma2) + 15997.0 / (3.89e9 - sigma2)
    return 10.0 * vacuum_nm / n


def air_to_vacuum_A(air_A):
    air = np.asarray(air_A, dtype=np.float64)
    vacuum = air * 1.00028
    for _ in range(5):
        vacuum *= air / vacuum_to_air_A(vacuum)
    return vacuum


def common_log_grid() -> np.ndarray:
    lo, hi = np.log(SYNTH_VAC_NM[0] * 10.0 + 1.0), np.log(SYNTH_VAC_NM[1] * 10.0 - 1.0)
    step = GRID_DV_KM_S / C_KM_S
    return np.exp(np.arange(lo, hi, step))


def load_arm(arm: str, grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    """Intrinsic normalized spectra of one arm, resampled to the common grid."""

    if arm in ("pz_nodes", "pz_galah", "pz_galah_zro"):
        data = np.load(RESULT_ROOT / {"pz_nodes": "pz_nodes.npz", "pz_galah": "pz_galah_nodes.npz",
                                      "pz_galah_zro": "pz_galah_zro_nodes.npz"}[arm], allow_pickle=False)
        wave, flux, ids = data["wavelength_vacuum_A"], data["normalized_flux"], data["model_id"]
        status = np.array(["ok" if np.all(np.isfinite(row)) else "nonfinite" for row in flux])
    else:
        import h5py

        parts = sorted((RESULT_ROOT / "korg" / arm).glob("part*_of*.h5"))
        if not parts:
            raise FileNotFoundError(f"no Korg output for {arm}")
        waves, fluxes, id_list, status_list = [], [], [], []
        for part in parts:
            with h5py.File(part, "r") as h:
                waves.append(h["wavelength_vacuum_A"][()])
                # Julia writes column-major (wavelength, model); h5py sees (model, wavelength).
                fluxes.append(h["normalized_flux"][()])
                id_list += [x.decode() if isinstance(x, bytes) else str(x) for x in h["model_id"][()]]
                status_list += [x.decode() if isinstance(x, bytes) else str(x) for x in h["status"][()]]
        for w in waves[1:]:
            if not np.array_equal(w, waves[0]):
                raise ValueError(f"{arm}: parts use different wavelength grids")
        wave, flux = waves[0], np.concatenate(fluxes, axis=0)
        ids, status = np.array(id_list), np.array(status_list)
    table = "dense_labels.tsv" if arm == "marcs_dense" else "node_labels.tsv"
    labels = {}
    with (RESULT_ROOT / table).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            labels[row["model_id"]] = {k: float(row[k]) for k in ("teff", "logg", "m_h")}
    keep = [i for i, st in enumerate(status) if st == "ok"]
    models = [{"model_id": str(ids[i]), **labels[str(ids[i])]} for i in keep]
    resampled = np.empty((len(keep), grid.size), dtype=np.float32)
    for row, i in enumerate(keep):
        resampled[row] = np.interp(grid, wave, flux[i])
    return models, resampled


def broaden(flux: np.ndarray, sigma_extra_km_s: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter1d

    sigma = np.hypot(SIGMA_LSF_KM_S, sigma_extra_km_s) / GRID_DV_KM_S
    return gaussian_filter1d(flux, sigma, axis=-1, mode="nearest", truncate=5.0)


class Observation:
    """One star's fit pixels, errors and continuum basis in the observed frame."""

    def __init__(self, target: dict):
        spectrum = _read_uves_phase3(PROJECT_ROOT / target["fits_path"])
        wave_air = spectrum["wave_air_A"]
        flux, error, quality = spectrum["flux"], spectrum["error"], spectrum["quality"]
        inside = (wave_air >= WINDOW_AIR_A[0]) & (wave_air <= WINDOW_AIR_A[1])
        good = inside & np.isfinite(flux) & np.isfinite(error) & (error > 0) & (flux > 0) & (quality == 0)
        scale = float(np.median(flux[good]))
        self.key = target["key"]
        self.barycor_km_s = spectrum["header"].get("ESO QC VRAD BARYCOR")
        self.wave_air = wave_air[good]
        self.wave_vac = air_to_vacuum_A(self.wave_air)
        self.flux = flux[good] / scale
        self.error = error[good] / scale
        self.weight = 1.0 / self.error**2
        from scipy.interpolate import BSpline

        lo, hi = self.wave_air[0], self.wave_air[-1]
        n_int = max(1, int(round((hi - lo) / KNOT_SPACING_A)))
        interior = np.linspace(lo, hi, n_int + 1)
        knots = np.concatenate([[lo] * 3, interior, [hi] * 3])
        self.basis = BSpline.design_matrix(np.clip(self.wave_air, lo, hi), knots, 3).toarray()
        self.mask = np.ones(self.wave_air.size, dtype=bool)

    def set_rest_frame_mask(self, rv_km_s: float) -> None:
        rest = self.wave_air / (1.0 + rv_km_s / C_KM_S)
        self.mask = np.ones(rest.size, dtype=bool)
        for lo, hi in MASKS_AIR_A:
            self.mask &= ~((rest >= lo) & (rest <= hi))

    def sample(self, grid: np.ndarray, spectra: np.ndarray, rv_km_s: float) -> np.ndarray:
        """Linear interpolation of rest-frame model spectra at Doppler-shifted pixels."""

        rest = self.wave_vac / (1.0 + rv_km_s / C_KM_S)
        idx = np.clip(np.searchsorted(grid, rest) - 1, 0, grid.size - 2)
        w = (rest - grid[idx]) / (grid[idx + 1] - grid[idx])
        return spectra[..., idx] * (1.0 - w) + spectra[..., idx + 1] * w

    def chi2(self, models: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Continuum-marginalized chi2 for a batch of sampled models (n, npix)."""

        m = self.mask
        y, wgt, basis = self.flux[m], self.weight[m], self.basis[m]
        models = np.atleast_2d(models)[:, m].astype(np.float64)
        chi2 = np.empty(models.shape[0])
        coef = np.empty((models.shape[0], basis.shape[1]))
        for start in range(0, models.shape[0], 128):
            block = models[start : start + 128]
            design = block[:, :, None] * basis[None, :, :]
            ata = np.einsum("npk,p,npl->nkl", design, wgt, design)
            aty = np.einsum("npk,p,p->nk", design, wgt, y)
            c = np.linalg.solve(ata, aty[..., None])[..., 0]
            pred = np.einsum("npk,nk->np", design, c)
            chi2[start : start + 128] = np.sum(wgt * (y - pred) ** 2, axis=1)
            coef[start : start + 128] = c
        return chi2, coef

    def prediction(self, sampled: np.ndarray, coef: np.ndarray) -> np.ndarray:
        return sampled * (self.basis @ coef)

    @property
    def dof_pixels(self) -> int:
        return int(self.mask.sum())


def _rv_scan(obs: Observation, grid: np.ndarray, template: np.ndarray) -> float:
    broadened = broaden(template[None, :], 3.0)
    coarse = np.arange(-150.0, 150.0 + 1e-9, 1.0)
    chi = [obs.chi2(obs.sample(grid, broadened, v))[0][0] for v in coarse]
    v0 = coarse[int(np.argmin(chi))]
    fine = np.arange(v0 - 2.0, v0 + 2.0 + 1e-9, 0.05)
    chi = [obs.chi2(obs.sample(grid, broadened, v))[0][0] for v in fine]
    return float(fine[int(np.argmin(chi))])


def _refine_nuisance(obs, grid, spectrum, rv0, sigma0):
    from scipy.optimize import minimize

    def objective(p):
        rv, sigma = p
        sampled = obs.sample(grid, broaden(spectrum[None, :], abs(sigma)), rv)
        return obs.chi2(sampled)[0][0]

    res = minimize(objective, [rv0, sigma0], method="Nelder-Mead",
                   options={"xatol": 0.01, "fatol": 1e-3 * max(1.0, objective([rv0, sigma0])) * 1e-3, "maxiter": 400})
    return float(res.x[0]), abs(float(res.x[1])), float(res.fun)


def _trilinear(labels_axes, cube, teff, logg, m_h):
    """Trilinear interpolation of spectra on the regular dense grid."""

    point = []
    for axis, value in zip(labels_axes, (teff, logg, m_h)):
        value = float(np.clip(value, axis[0], axis[-1]))
        i = int(np.clip(np.searchsorted(axis, value) - 1, 0, axis.size - 2))
        point.append((i, (value - axis[i]) / (axis[i + 1] - axis[i])))
    (i, a), (j, b), (k, c) = point
    out = 0.0
    for di, wi in ((0, 1 - a), (1, a)):
        for dj, wj in ((0, 1 - b), (1, b)):
            for dk, wk in ((0, 1 - c), (1, c)):
                w = wi * wj * wk
                if w:
                    out = out + w * cube[i + di, j + dj, k + dk]
    return out


def _fit_nodes(obs, grid, models, spectra, rv0, sigma_grid=SIGMA_EXTRA_GRID):
    """Grid over nodes x extra broadening at rv0, then refine RV/broadening for the top five."""

    sigma_grid = np.asarray(sigma_grid, dtype=np.float64)
    table = np.empty((len(models), sigma_grid.size))
    for s_index, sigma in enumerate(sigma_grid):
        sampled = obs.sample(grid, broaden(spectra, sigma), rv0)
        table[:, s_index] = obs.chi2(sampled)[0]
    best_per_model = table.min(axis=1)
    order = np.argsort(best_per_model)
    refined = []
    for i in order[:5]:
        sigma0 = sigma_grid[int(np.argmin(table[i]))]
        rv, sigma, chi = _refine_nuisance(obs, grid, spectra[i], rv0, sigma0)
        refined.append({**models[i], "rv_km_s": rv, "sigma_extra_km_s": sigma, "chi2": chi, "index": int(i)})
    refined.sort(key=lambda r: r["chi2"])
    return refined, best_per_model


def _fit_dense(obs, grid, models, spectra, rv0, sigma_start, box=None):
    from scipy.optimize import minimize

    axes = (DENSE_TEFF, DENSE_LOGG, DENSE_MH)
    cube = np.full((DENSE_TEFF.size, DENSE_LOGG.size, DENSE_MH.size, grid.size), np.nan, dtype=np.float32)
    for model, spectrum in zip(models, spectra):
        cube[np.searchsorted(DENSE_TEFF, model["teff"]), np.searchsorted(DENSE_LOGG, model["logg"]),
             np.searchsorted(DENSE_MH, model["m_h"])] = spectrum
    if np.isnan(cube).any():
        raise ValueError("marcs_dense grid has missing models; trilinear fit needs the full cube")
    bounds = box or {"teff": (DENSE_TEFF[0], DENSE_TEFF[-1]), "logg": (DENSE_LOGG[0], DENSE_LOGG[-1]),
                     "m_h": (DENSE_MH[0], DENSE_MH[-1])}
    inside = [i for i, m in enumerate(models)
              if all(bounds[k][0] <= m[k] <= bounds[k][1] for k in ("teff", "logg", "m_h"))]
    start_nodes, _ = _fit_nodes(obs, grid, [models[i] for i in inside], spectra[inside], rv0, [sigma_start])
    start = start_nodes[0]
    scale = np.array([100.0, 0.25, 0.25, 1.0, 1.0])
    lo = np.array([bounds["teff"][0], bounds["logg"][0], bounds["m_h"][0], -np.inf, 0.0])
    hi = np.array([bounds["teff"][1], bounds["logg"][1], bounds["m_h"][1], np.inf, 20.0])

    def unpack(z):
        return np.clip(z * scale, lo, hi)

    def objective(z):
        teff, logg, m_h, rv, sigma = unpack(z)
        spectrum = _trilinear(axes, cube, teff, logg, m_h)
        sampled = obs.sample(grid, broaden(spectrum[None, :], sigma), rv)
        return obs.chi2(sampled)[0][0]

    z0 = np.array([start["teff"], start["logg"], start["m_h"], start["rv_km_s"], start["sigma_extra_km_s"]]) / scale
    res = minimize(objective, z0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-6 * objective(z0), "maxiter": 3000, "maxfev": 3000})
    teff, logg, m_h, rv, sigma = unpack(res.x)
    at_bound = {k: bool(np.isclose(v, bounds[k][0]) or np.isclose(v, bounds[k][1]))
                for k, v in (("teff", teff), ("logg", logg), ("m_h", m_h))}
    return {"teff": float(teff), "logg": float(logg), "m_h": float(m_h), "rv_km_s": float(rv),
            "sigma_extra_km_s": float(sigma), "chi2": float(res.fun), "at_bound": at_bound,
            "optimizer_success": bool(res.success), "nfev": int(res.nfev),
            "start_node": {k: start[k] for k in ("model_id", "teff", "logg", "m_h", "chi2")},
            "bounds": {k: list(v) for k, v in bounds.items()}}, cube, axes


SUBWINDOWS_AIR_A = {
    "6480-6555": (6480.0, 6555.0),
    "6572-6640": (6572.0, 6640.0),
    "6640-6700 TiO bandheads": (6640.0, 6700.0),
    "6700-6735": (6700.0, 6735.0),
}


def _same_atmosphere_diagnostic(obs, grid, arms, best_pz):
    """Native PZ vs Korg synthesis on the identical PZ atmosphere, by sub-window.

    Both spectra use the PZ best node and its RV/broadening; each gets its own
    continuum fit. Reports model-model RMS and each model's residual RMS.
    """

    rest = obs.wave_air / (1.0 + best_pz["rv_km_s"] / C_KM_S)
    preds = {}
    for arm in ("pz_nodes", "korg_pzatm"):
        models, spectra = arms[arm]
        i = next(k for k, m in enumerate(models) if m["model_id"] == best_pz["model_id"])
        sampled = obs.sample(grid, broaden(spectra[i][None, :], best_pz["sigma_extra_km_s"]), best_pz["rv_km_s"])
        _, coef = obs.chi2(sampled)
        preds[arm] = obs.prediction(sampled[0], coef[0])
    out = {"node": best_pz["model_id"]}
    for name, (lo, hi) in SUBWINDOWS_AIR_A.items():
        sel = obs.mask & (rest >= lo) & (rest <= hi)
        out[name] = {
            "n_pixels": int(sel.sum()),
            "model_model_rms": float(np.sqrt(np.mean((preds["pz_nodes"][sel] - preds["korg_pzatm"][sel]) ** 2))),
            "residual_rms_pz": float(np.sqrt(np.mean((obs.flux[sel] - preds["pz_nodes"][sel]) ** 2))),
            "residual_rms_korg": float(np.sqrt(np.mean((obs.flux[sel] - preds["korg_pzatm"][sel]) ** 2))),
            "median_depth_pz": float(np.median(1.0 - preds["pz_nodes"][sel] / np.percentile(preds["pz_nodes"][sel], 95))),
            "median_depth_korg": float(np.median(1.0 - preds["korg_pzatm"][sel] / np.percentile(preds["korg_pzatm"][sel], 95))),
            "median_depth_obs": float(np.median(1.0 - obs.flux[sel] / np.percentile(obs.flux[sel], 95))),
        }
    return out


def stage_continuum_sensitivity(args) -> None:
    """Node-arm best fits at alternative continuum knot spacings."""

    global KNOT_SPACING_A
    grid = common_log_grid()
    rv0 = {r["key"]: r["rv0_km_s"] for r in json.loads((RESULT_ROOT / "fit_results.json").read_text())}
    arms = {arm: load_arm(arm, grid) for arm in ("pz_nodes", "marcs_nodes", "korg_pzatm")}
    rows = []
    default = KNOT_SPACING_A
    for knot in (25.0, 50.0, 100.0, 300.0):
        KNOT_SPACING_A = knot
        for target in json.loads((RESULT_ROOT / "targets.json").read_text()):
            obs = Observation(target)
            obs.set_rest_frame_mask(rv0[target["key"]])
            for arm, (models, spectra) in arms.items():
                best = _fit_nodes(obs, grid, models, spectra, rv0[target["key"]])[0][0]
                rows.append({"knot_spacing_A": knot, "n_coefficients": obs.basis.shape[1], "star": target["key"],
                             "arm": arm, "model_id": best["model_id"],
                             "chi2_per_pixel": best["chi2"] / obs.dof_pixels})
                print(rows[-1], flush=True)
    KNOT_SPACING_A = default
    (RESULT_ROOT / "continuum_sensitivity.json").write_text(json.dumps(rows, indent=2))


def stage_fit(args) -> None:
    grid = common_log_grid()
    targets = json.loads((RESULT_ROOT / "targets.json").read_text())
    arms = {arm: load_arm(arm, grid) for arm in ARMS}
    for arm, (models, _) in arms.items():
        print(f"{arm}: {len(models)} usable models")
    reference_label = {"teff": 3800.0, "logg": 1.5, "m_h": 0.0}
    results, best_spectra = [], {}
    for target in targets:
        if args.only and target["key"] not in args.only:
            continue
        obs = Observation(target)
        rv_templates = {}
        for arm in ("pz_nodes", "marcs_nodes"):
            models, spectra = arms[arm]
            i = next(k for k, m in enumerate(models) if all(m[x] == reference_label[x] for x in reference_label))
            rv_templates[arm] = _rv_scan(obs, grid, spectra[i])
        rv0 = float(np.mean(list(rv_templates.values())))
        obs.set_rest_frame_mask(rv0)
        barycor = obs.barycor_km_s
        lit_rv = target["borisov2023_labels"]["rv_km_s"]
        record = {"key": target["key"], "rv0_km_s": rv0, "rv0_by_template": rv_templates,
                  "rv_frame_check": {
                      "borisov2023_rv_km_s": lit_rv, "eso_qc_vrad_barycor_km_s": barycor,
                      "expected_topocentric_km_s": (lit_rv - barycor) if (lit_rv is not None and barycor is not None) else None},
                  "n_pixels": obs.dof_pixels, "median_snr": float(np.median(obs.flux / obs.error)),
                  "arms": {}}
        n_coef = obs.basis.shape[1]
        for arm in ("pz_nodes", "marcs_nodes", "korg_pzatm"):
            models, spectra = arms[arm]
            refined, node_chi2 = _fit_nodes(obs, grid, models, spectra, rv0)
            best = refined[0]
            sampled = obs.sample(grid, broaden(spectra[best["index"]][None, :], best["sigma_extra_km_s"]), best["rv_km_s"])
            chi, coef = obs.chi2(sampled)
            record["arms"][arm] = {
                "best": {k: v for k, v in best.items() if k != "index"},
                "top5": [{k: r[k] for k in ("model_id", "teff", "logg", "m_h", "chi2", "rv_km_s", "sigma_extra_km_s")} for r in refined],
                "node_chi2_grid_min": {models[i]["model_id"]: float(node_chi2[i]) for i in range(len(models))},
            }
            best_spectra[(target["key"], arm)] = (obs.prediction(sampled[0], coef[0]))
        record["same_atmosphere_diagnostic"] = _same_atmosphere_diagnostic(
            obs, grid, arms, record["arms"]["pz_nodes"]["best"])
        record["same_atmosphere_diagnostic_korg_node"] = _same_atmosphere_diagnostic(
            obs, grid, arms, record["arms"]["korg_pzatm"]["best"])
        models, spectra = arms["marcs_dense"]
        for tag, box in (("marcs_dense", None), ("marcs_dense_pzbox", PZ_BOX)):
            sigma_start = record["arms"]["marcs_nodes"]["best"]["sigma_extra_km_s"]
            fit, cube, axes = _fit_dense(obs, grid, models, spectra, rv0, sigma_start, box)
            spectrum = _trilinear(axes, cube, fit["teff"], fit["logg"], fit["m_h"])
            sampled = obs.sample(grid, broaden(spectrum[None, :], fit["sigma_extra_km_s"]), fit["rv_km_s"])
            chi, coef = obs.chi2(sampled)
            record["arms"][tag] = {"best": fit}
            best_spectra[(target["key"], tag)] = obs.prediction(sampled[0], coef[0])
        for arm_record in record["arms"].values():
            b = arm_record["best"]
            dof = obs.dof_pixels - n_coef - 2 - (3 if "at_bound" in b else 0)
            b["chi2_reduced"] = b["chi2"] / dof
            b["rms_fraction"] = None
        m = obs.mask
        for tag in record["arms"]:
            pred = best_spectra[(target["key"], tag)]
            record["arms"][tag]["best"]["rms_fraction"] = float(np.sqrt(np.mean((obs.flux[m] - pred[m]) ** 2)))
        np.savez_compressed(
            RESULT_ROOT / "fits" / f"{target['key']}.npz",
            wave_air_A=obs.wave_air, flux=obs.flux, error=obs.error, mask=obs.mask,
            **{f"model_{tag}": best_spectra[(target["key"], tag)] for tag in record["arms"]},
        )
        results.append(record)
        summary = "  ".join(
            f"{tag}={r['best']['teff']:.0f}/{r['best']['logg']:.2f}/{r['best']['m_h']:+.2f} chi2r={r['best']['chi2_reduced']:.1f}"
            for tag, r in record["arms"].items())
        print(f"{target['key']:7s} rv0={rv0:+.2f}  {summary}", flush=True)
    (RESULT_ROOT / "fit_results.json").write_text(json.dumps(results, indent=2))


ARM_LABELS = {
    "pz_nodes": "Payne-Zero (78 nodes)",
    "marcs_nodes": "Korg+MARCS (same 78 labels)",
    "korg_pzatm": "Korg on PZ atmospheres",
    "marcs_dense": "Korg+MARCS continuous",
    "marcs_dense_pzbox": "Korg+MARCS continuous, PZ box",
}


def stage_report(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = json.loads((RESULT_ROOT / "fit_results.json").read_text())
    targets = {t["key"]: t for t in json.loads((RESULT_ROOT / "targets.json").read_text())}
    rows = []
    for record in results:
        ref = targets[record["key"]]["reference_labels"]
        for tag, arm in record["arms"].items():
            b = arm["best"]
            rows.append({
                "star": record["key"], "arm": tag, "teff": round(b["teff"], 1), "logg": round(b["logg"], 3),
                "m_h": round(b["m_h"], 3), "rv_km_s": round(b["rv_km_s"], 3),
                "sigma_extra_km_s": round(b["sigma_extra_km_s"], 3), "chi2": round(b["chi2"], 1),
                "chi2_reduced": round(b["chi2_reduced"], 3), "rms_fraction": round(b["rms_fraction"], 5),
                "at_bound": ",".join(k for k, v in b.get("at_bound", {}).items() if v),
                "ref_teff": ref["teff_K"] if ref else "", "ref_logg": ref["logg"] if ref else "",
                "ref_feh": ref["feh"] if ref else "",
            })
    with (RESULT_ROOT / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Spectra: observation, PZ best node, Korg+MARCS continuous best, residuals.
    for record in results:
        key = record["key"]
        data = np.load(RESULT_ROOT / "fits" / f"{key}.npz")
        wave, flux, mask = data["wave_air_A"], data["flux"], data["mask"]
        rest = wave / (1.0 + record["rv0_km_s"] / C_KM_S)
        edges = np.linspace(WINDOW_AIR_A[0], WINDOW_AIR_A[1], 4)
        fig, axes = plt.subplots(6, 1, figsize=(13, 12), gridspec_kw={"height_ratios": [3, 1] * 3})
        for row, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            sel = (rest >= lo) & (rest <= hi)
            ax, rax = axes[2 * row], axes[2 * row + 1]
            ax.plot(rest[sel], flux[sel], color="k", lw=0.6, label="UVES")
            for tag, color in (("pz_nodes", "C3"), ("marcs_dense", "C0")):
                model = data[f"model_{tag}"]
                b = record["arms"][tag]["best"]
                ax.plot(rest[sel], model[sel], color=color, lw=0.6,
                        label=f"{ARM_LABELS[tag]}: {b['teff']:.0f}/{b['logg']:.2f}/{b['m_h']:+.2f}, "
                              f"RMS {b['rms_fraction']:.3f}")
                rax.plot(rest[sel], np.where(mask[sel], flux[sel] - model[sel], np.nan), color=color, lw=0.5)
            rax.axhline(0, color="0.5", lw=0.5)
            rax.set_ylim(-0.25, 0.25)
            ax.set_xlim(lo, hi)
            rax.set_xlim(lo, hi)
            if row == 0:
                ax.legend(fontsize=7, loc="lower left", ncol=3)
        axes[-1].set_xlabel("stellar-rest air wavelength (A)")
        ref = targets[key]["reference_labels"]
        title = f"{targets[key]['common_name']} ({targets[key]['spectral_class']}), {targets[key]['eso_obscore']['obs_publisher_did'].split('?')[1]}"
        if ref:
            title += f"; GBS {ref['teff_K']:.0f}/{ref['logg']:.2f}/{ref['feh']:+.2f}"
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(RESULT_ROOT / "figures" / f"{key}_bestfit.png", dpi=110)
        plt.close(fig)

    # Labels by arm, and chi2 relative to the Payne-Zero best node.
    tags = list(ARM_LABELS)
    keys = [r["key"] for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    x = np.arange(len(keys))
    for j, tag in enumerate(tags):
        offset = (j - 2) * 0.14
        for ax, label in zip(axes[:3], ("teff", "logg", "m_h")):
            ax.plot(x + offset, [r["arms"][tag]["best"][label] for r in results], "o", ms=5,
                    color=f"C{j}", label=ARM_LABELS[tag])
        axes[3].plot(x + offset, [r["arms"][tag]["best"]["chi2"] / r["arms"]["pz_nodes"]["best"]["chi2"] for r in results],
                     "o", ms=5, color=f"C{j}")
    for ax, label in zip(axes[:3], ("teff_K", "logg", "feh")):
        for i, key in enumerate(keys):
            ref = targets[key]["reference_labels"]
            if ref:
                ax.errorbar(i, ref[label], yerr=ref["e_" + label],
                            fmt="k*", ms=10, label="GBS reference" if i == 0 else None)
    for ax, title in zip(axes, ("Teff (K)", "log g", "[M/H]", "chi2 / chi2(PZ best node)")):
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, fontsize=8)
        ax.set_title(title, fontsize=10)
    axes[3].axhline(1.0, color="0.5", lw=0.6)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "figures" / "labels_by_arm.png", dpi=120)
    plt.close(fig)
    print(f"wrote metrics.csv and {len(results) + 1} figures")


# Kurucz molecule codes of the Payne-Zero species ids present in optical windows.
KURUCZ_MOLECULE_NAMES = {
    101.0: "H2", 106.0: "CH", 107.0: "NH", 108.0: "OH", 111.0: "NaH", 112.0: "MgH",
    114.0: "SiH", 120.0: "CaH", 606.0: "C2", 607.0: "CN", 608.0: "CO", 813.0: "AlO",
    814.0: "SiO", 822.0: "TiO", 823.0: "VO", 840.0: "ZrO",
}
KORG_LINELIST_COUNT_JL = r"""
using Korg
lo, hi = parse(Float64, ARGS[1]), parse(Float64, ARGS[2])
ll = filter(l -> lo <= l.wl * 1e8 <= hi, Korg.get_GALAH_DR3_linelist())
counts = Dict{String,Int}()
for l in ll
    key = Korg.ismolecule(l.species) ? string(l.species) : "atomic"
    counts[key] = get(counts, key, 0) + 1
end
for (key, count) in counts
    println("COUNT ", key, " ", count)
end
"""
KORG_SPECIES_NAMES = {"OTi": "TiO", "OZr": "ZrO", "HMg": "MgH", "HCa": "CaH", "HSi": "SiH",
                      "HO": "OH", "HC": "CH", "CN": "CN", "C2": "C2", "atomic": "atomic"}


def stage_linelists(args) -> None:
    """Count the lines each synthesis actually loads over the shared window."""

    import subprocess

    from payne_zero_synthesis import atomic_lines, pipeline
    from payne_zero_synthesis.molecular_equilibrium import _SPECIES_CODE_TO_MOLECULE_CODES
    from payne_zero_synthesis.paths import source_catalog_path

    contract = pipeline._window_grid_contract(SYNTH_VAC_NM[0], SYNTH_VAC_NM[1], PZ_RESOLUTION)
    grid_obj = contract[1]
    lo_nm, hi_nm = grid_obj.start_wavelength_nm, grid_obj.end_wavelength_nm
    atomic = atomic_lines.load_catalog(
        (lo_nm, hi_nm), grid_obj,
        catalog_path=source_catalog_path("lines", "atomic_source_lines_parsed.npz"), sort="catalog")
    compiled = pipeline.SynthesisPipeline._compile_molecular(lo_nm, hi_nm, PZ_RESOLUTION)
    species = np.asarray(compiled["species_code"]).astype(int)
    pz = {"atomic": int(len(atomic))}
    for code, count in zip(*np.unique(species, return_counts=True)):
        kurucz = _SPECIES_CODE_TO_MOLECULE_CODES[int(code)][0]
        pz[KURUCZ_MOLECULE_NAMES.get(kurucz, f"kurucz_{kurucz:.0f}")] = int(count)

    korg_lo, korg_hi = 6475.0 - 10.0, 6747.0 + 10.0   # the Julia arms' filter, vacuum A
    korg_project = os.environ.get("KORG_PROJECT", "/Users/jdli/Project/jorg/Korg.jl-1.0.1")
    raw = subprocess.run(["julia", f"--project={korg_project}", "-e", KORG_LINELIST_COUNT_JL,
                          str(korg_lo), str(korg_hi)], check=True, capture_output=True, text=True).stdout
    korg = {}
    for line in raw.splitlines():
        if line.startswith("COUNT "):
            _, key, count = line.split()
            korg[KORG_SPECIES_NAMES.get(key, key)] = int(count)
    record = {
        "payne_zero": {"window_vacuum_nm": [lo_nm, hi_nm], "counts": pz, "total": int(sum(pz.values())),
                       "sources": {"atomic": "atomic_source_lines_parsed.npz",
                                   "molecular": "molecular_band_lines.npz + titanium_oxide_lines.npy (Schwenke TiO)"}},
        "korg": {"window_vacuum_A": [korg_lo, korg_hi], "counts": korg, "total": int(sum(korg.values())),
                 "source": "GALAH DR3 line list bundled with Korg 1.0.1"},
    }
    (RESULT_ROOT / "linelists.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


PZ_FIT_NAMES = ("teff", "logg", "m_h", "rv_km_s", "sigma_extra_km_s")
V4_BOX = {"teff": (3000.0, 4000.0), "logg": (0.5, 2.5), "m_h": (-1.0, 0.5)}


class V4FastModel:
    """M-giant fast forward model for ``fitter.fit_normalized_spectrum``.

    Labels -> v4 (m,T) ensemble -> physical reconstruction -> native Payne
    Zero synthesis -> ``ObservedSpectrumOperator`` (residual velocity, extra
    Gaussian broadening, R=74,450 LSF, resampling to the observed pixels).
    The production ``synthesize_from_labels`` initializers are trained above
    4000 K, so this route replaces them for the giant slice. Native spectra are
    cached per label triple, so velocity and broadening derivatives only
    re-apply the operator.

    Labels are evaluated at the precision the solver deck carries (whole
    kelvin in Teff, 1e-4 in log g and [M/H]), so the fast and converged
    callbacks see the same labels and a converged product reports exactly the
    labels it was solved at.
    """

    def __init__(self, obs, *, device: str, dtype: str, cache_dir: Path, bundle=None):
        self.obs = obs
        self.device = device
        self.dtype = dtype
        # With a line-list bundle, every atmosphere's EOS and molecular state is
        # recomputed with this process's molecule table before synthesis.
        self.bundle = bundle
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._native: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._operator = None
        self.evaluations = 0
        self.syntheses = 0

    @staticmethod
    def deck_labels(teff, logg, m_h) -> tuple[float, float, float]:
        return float(round(float(teff))), round(float(logg), 4), round(float(m_h), 4)

    @classmethod
    def labels(cls, teff, logg, m_h):
        from bench.labels import StellarLabels

        return StellarLabels(*cls.deck_labels(teff, logg, m_h), 0.0, VMIC_KM_S)

    def _synthesize_path(self, path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from payne_zero_synthesis import synthesize

        if self.bundle is not None:
            import torch

            from payne_zero_synthesis.api import _surface_flux_per_wavelength_nm
            from payne_zero_synthesis.synthesis import synthesize_structured_atmosphere

            dtype = torch.float64 if self.dtype == "float64" else torch.float32
            atmosphere = rebuilt_atmosphere(str(path), device=self.device, dtype=dtype)
            result, _ = synthesize_structured_atmosphere(
                atmosphere, wavelength_start_nm=SYNTH_VAC_NM[0], wavelength_end_nm=SYNTH_VAC_NM[1],
                resolution=PZ_RESOLUTION, molecular_lines=True, device=self.device, dtype=dtype,
                window_invariants=self.bundle)
            wavelength = np.asarray(result.wavelength_nm, np.float64)
            return (wavelength,
                    np.asarray(_surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_total_per_frequency),
                               np.float64),
                    np.asarray(_surface_flux_per_wavelength_nm(wavelength,
                                                               result.eddington_flux_continuum_per_frequency),
                               np.float64))
        spectrum = synthesize(path, wavelength_start_nm=SYNTH_VAC_NM[0], wavelength_end_nm=SYNTH_VAC_NM[1],
                              resolution=PZ_RESOLUTION, molecular_lines=True, device=self.device, dtype=self.dtype)
        return (np.asarray(spectrum.wavelength_nm, np.float64), np.asarray(spectrum.flux_total, np.float64),
                np.asarray(spectrum.flux_continuum, np.float64))

    def native(self, teff, logg, m_h):
        key = self.deck_labels(teff, logg, m_h)
        if key in self._native:
            return self._native[key]
        cache = self.cache_dir / f"t{key[0]:.0f}_g{key[1]:+.4f}_m{key[2]:+.4f}.npz"
        if cache.is_file():
            with np.load(cache) as data:
                self._native[key] = (data["wavelength_nm"], data["flux_total"], data["flux_continuum"])
            return self._native[key]
        import tempfile

        from experiments import mgiant_experimental_interface_v1 as interface
        from payne_zero_atmosphere.synthesis_bridge import save_product_structured_atmosphere

        atmosphere, _ = interface.build_v4_initializer(self.labels(*key), interface.DEFAULT_CHECKPOINT_DIR)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v4_start.npz"
            save_product_structured_atmosphere(atmosphere, path, device="cpu", dtype="float64")
            native = self._synthesize_path(path)
        np.savez(cache, wavelength_nm=native[0], flux_total=native[1], flux_continuum=native[2])
        self._native[key] = native
        self.syntheses += 1
        return native

    def project(self, native, rv_km_s, sigma_km_s) -> np.ndarray:
        import torch

        from fitter import ObservedSpectrumOperator

        wavelength, total, continuum = native
        if self._operator is None or not np.array_equal(self._operator.input_wavelength_nm, wavelength):
            self._operator = ObservedSpectrumOperator(wavelength, self.obs.wave_vac / 10.0,
                                                      resolving_power=INSTRUMENT_R, device="cpu",
                                                      dtype=torch.float64)
        self._operator.set_parameters(radial_velocity_km_s=float(rv_km_s),
                                      broadening_sigma_km_s=float(abs(sigma_km_s)))
        _, _, normalized = self._operator.convolve_fluxes(torch.as_tensor(total), torch.as_tensor(continuum))
        return normalized.detach().cpu().numpy()

    def __call__(self, parameters) -> np.ndarray:
        self.evaluations += 1
        teff, logg, m_h, rv, sigma = parameters
        return self.project(self.native(teff, logg, m_h), rv, sigma)


class ConvergedModel:
    """Converged-atmosphere callback for ``refine_with_physical_atmosphere``.

    Each call runs the unchanged solver from the v4 start through
    ``mgiant_experimental_interface_v1.solve_experimental_point`` and accepts
    the atmosphere only when its independent quality flag is valid.
    """

    def __init__(self, fast: V4FastModel, solve_dir: Path):
        self.fast = fast
        self.solve_dir = solve_dir
        self.records: list[dict] = []

    def __call__(self, parameters) -> np.ndarray:
        from experiments import mgiant_experimental_interface_v1 as interface

        teff, logg, m_h, rv, sigma = parameters
        started = time.time()
        summary = interface.solve_experimental_point(self.fast.labels(teff, logg, m_h), output_dir=self.solve_dir)
        record = {"parameters": dict(zip(PZ_FIT_NAMES, map(float, parameters))),
                  "solver": summary["solver"], "quality_flag": summary["quality_flag"],
                  "final_product": summary["final_product"], "seconds": time.time() - started}
        self.records.append(record)
        if not summary["quality_flag"]["valid"] or summary["final_product"] is None:
            raise RuntimeError(f"converged atmosphere failed its quality flag at {record['parameters']}")
        return self.fast.project(self.fast._synthesize_path(summary["final_product"]), rv, sigma)


def stage_pz_fit(args) -> None:
    from fitter import (FitConfiguration, NormalizedSpectrum, PhysicalAtmosphereConfiguration,
                        fit_normalized_spectrum, refine_with_physical_atmosphere)

    targets = {t["key"]: t for t in json.loads((RESULT_ROOT / "targets.json").read_text())}
    fits = {f["key"]: f for f in json.loads((RESULT_ROOT / "fit_results.json").read_text())}
    record, target = fits[args.star], targets[args.star]
    galah = args.linelist == "galah_zro"
    out = RESULT_ROOT / ("pz_fit_galah_zro" if galah else "pz_fit") / args.star
    out.mkdir(parents=True, exist_ok=True)
    bundle = None
    if galah:
        import torch

        register_zro_species()
        bundle, _ = build_galah_bundle(_read_galah_lines(RESULT_ROOT / "galah_lines.tsv"), device=args.device,
                                       dtype=torch.float64 if args.dtype == "float64" else torch.float32,
                                       scratch=out / "galah_inputs", include_zro=True)

    obs = Observation(target)
    obs.set_rest_frame_mask(record["rv0_km_s"])
    spectrum = NormalizedSpectrum(wavelength=obs.wave_vac / 10.0, flux=obs.flux,
                                  inverse_variance=obs.weight, mask=obs.mask)
    start = (next(r for r in json.loads((RESULT_ROOT / "fit_pz_galah_zro.json").read_text())
                  if r["key"] == args.star)["best"] if galah else record["arms"]["pz_nodes"]["best"])
    rv0 = record["rv0_km_s"]
    configuration = FitConfiguration(
        names=PZ_FIT_NAMES,
        initial=np.array([start["teff"], start["logg"], start["m_h"], start["rv_km_s"],
                          max(start["sigma_extra_km_s"], 0.5)]),
        lower=np.array([V4_BOX["teff"][0], V4_BOX["logg"][0], V4_BOX["m_h"][0], rv0 - 5.0, 0.0]),
        upper=np.array([V4_BOX["teff"][1], V4_BOX["logg"][1], V4_BOX["m_h"][1], rv0 + 5.0, 10.0]),
        derivative_steps=np.array([20.0, 0.05, 0.05, 0.1, 0.1]),
        trust_half_width=np.array([150.0, 0.4, 0.3, 1.0, 1.5]),
        maximum_iterations=args.max_iterations,
    )
    fast = V4FastModel(obs, device=args.device, dtype=args.dtype, cache_dir=out / "native_cache", bundle=bundle)
    started = time.time()
    fast_result = fit_normalized_spectrum(spectrum, configuration, fast, continuum_basis=obs.basis)
    fast_result.save(out / "fast")
    n_good = int(obs.mask.sum())
    summary = {
        "star": args.star,
        "start_node": {k: start[k] for k in ("model_id", "teff", "logg", "m_h", "rv_km_s", "sigma_extra_km_s", "chi2")},
        "fast": {"parameters": dict(zip(PZ_FIT_NAMES, map(float, fast_result.parameters))),
                 "chi2": float(fast_result.mean_weighted_squared_residual * n_good),
                 "converged": fast_result.converged, "stop_reason": fast_result.stop_reason,
                 "evaluations": fast.evaluations, "syntheses": fast.syntheses,
                 "seconds": time.time() - started},
        "n_pixels": n_good,
        "config": {"device": args.device, "dtype": args.dtype, "max_iterations": args.max_iterations,
                   "linelist": ("GALAH DR3 with ZrO; Zr and ZrO added to the synthesis molecular equilibrium"
                                if galah else "native Payne Zero"),
                   "initializer": "m_star_emulator_mgiant_v4 (three-seed median) + physical reconstruction",
                   "r_grid": PZ_RESOLUTION, "instrument_r": INSTRUMENT_R, "knot_spacing_A": KNOT_SPACING_A},
    }
    np.save(out / "fast_model_flux.npy", fast_result.model_flux)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.star} fast: {summary['fast']}", flush=True)
    if args.fast_only:
        return

    converged = ConvergedModel(fast, out / "solves")
    physical_configuration = PhysicalAtmosphereConfiguration(
        maximum_discrepancy_rms=2.0e-3,
        maximum_objective_degradation=0.1,
        minimum_predicted_objective_improvement=1.0e-3,
        maximum_physical_evaluations=args.max_physical,
    )
    started = time.time()
    try:
        physical = refine_with_physical_atmosphere(spectrum, configuration, fast_result, fast, converged,
                                                   physical_configuration, continuum_basis=obs.basis)
    except RuntimeError as error:
        summary["physical"] = {"error": str(error), "solves": converged.records, "seconds": time.time() - started}
    else:
        physical.save(out / "physical")
        np.save(out / "physical_model_flux.npy", physical.model_flux)
        summary["physical"] = {
            "parameters": dict(zip(PZ_FIT_NAMES, map(float, physical.parameters))),
            "chi2": float(physical.mean_weighted_squared_residual * n_good),
            "successful": physical.successful, "physical_fit_stationary": physical.physical_fit_stationary,
            "fast_physical_gates_passed": physical.fast_physical_gates_passed,
            "stop_reason": physical.stop_reason,
            "checks": [{"discrepancy_rms": c.discrepancy_rms, "objective_degradation": c.objective_degradation,
                        "accepted": c.accepted} for c in physical.physical_checks],
            "solves": converged.records, "seconds": time.time() - started,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"{args.star} physical: {summary['physical']}", flush=True)


def stage_pz_fit_collect(args) -> None:
    """One record of the Payne Zero fitter results, beside the shared-fitter arms."""

    fits = {f["key"]: f for f in json.loads((RESULT_ROOT / "fit_results.json").read_text())}
    targets = {t["key"]: t for t in json.loads((RESULT_ROOT / "targets.json").read_text())}
    galah = args.linelist == "galah_zro"
    galah_nodes = ({r["key"]: r["best"] for r in json.loads((RESULT_ROOT / "fit_pz_galah_zro.json").read_text())}
                   if galah else {})
    rows = []
    for key in fits:
        star_dir = RESULT_ROOT / ("pz_fit_galah_zro" if galah else "pz_fit") / key
        summary = json.loads((star_dir / "summary.json").read_text())
        obs = Observation(targets[key])
        obs.set_rest_frame_mask(fits[key]["rv0_km_s"])
        m = obs.mask
        row = {"key": key, "n_pixels": summary["n_pixels"], "start_node": summary["start_node"],
               "fast": summary["fast"], "physical": summary.get("physical")}
        for stage, flux_file in (("fast", "fast_model_flux.npy"), ("physical", "physical_model_flux.npy")):
            if row[stage] is None or not (star_dir / flux_file).is_file():
                continue
            model = np.load(star_dir / flux_file)
            row[stage]["rms_fraction"] = float(np.sqrt(np.mean((obs.flux[m] - model[m]) ** 2)))
            row[stage]["chi2_reduced"] = row[stage]["chi2"] / row["n_pixels"]
            for arm in ("pz_nodes", "marcs_nodes", "korg_pzatm", "marcs_dense", "marcs_dense_pzbox"):
                row[stage].setdefault("chi2_ratio_to", {})[arm] = (
                    fits[key]["arms"][arm]["best"]["chi2"] / row[stage]["chi2"])
            if galah:
                row[stage]["chi2_ratio_to"]["pz_galah_zro"] = galah_nodes[key]["chi2"] / row[stage]["chi2"]
        rows.append(row)
    (RESULT_ROOT / ("pz_fit_galah_zro_results.json" if galah else "pz_fit_results.json")).write_text(
        json.dumps(rows, indent=2, default=str))
    for row in rows:
        for stage in ("fast", "physical"):
            r = row[stage]
            if r and "parameters" in r:
                p = r["parameters"]
                print(f"{row['key']:7s} {stage:8s} {p['teff']:.0f}/{p['logg']:.2f}/{p['m_h']:+.2f} "
                      f"chi2r={r['chi2_reduced']:.1f} rms={r.get('rms_fraction', float('nan')):.4f}")
            elif r:
                print(f"{row['key']:7s} {stage:8s} {r.get('error')}")


# GALAH lines as Korg holds them, for Payne Zero synthesis with Korg's line list.
# Payne Zero scales van der Waals damping as gamma_1e4 * (T / 1e4)^0.3, as Korg
# does for plain gamma values; lines that Korg treats with ABO theory are
# given the gamma that reproduces Korg's value at GALAH_ABO_REFERENCE_K.
GALAH_ABO_REFERENCE_K = 3500.0
GALAH_EXPORT_JL = r"""
using Korg, Printf
lo, hi, tref = parse(Float64, ARGS[1]), parse(Float64, ARGS[2]), parse(Float64, ARGS[3])
ll = filter(l -> lo <= l.wl * 1e8 <= hi, Korg.get_GALAH_DR3_linelist())
open(ARGS[4], "w") do io
    println(io, "wl_vac_A\tlog_gf\tmolecule\tspecies\tatomic_number\tcharge\tE_lower_eV\tgamma_rad\tgamma_stark\tgamma_vdw_1e4\tabo")
    for l in ll
        mol = Korg.ismolecule(l.species)
        atoms = Korg.get_atoms(l.species.formula)
        abo = l.vdW[2] != -1
        gvdw = abo ? Korg.scaled_vdW(l.vdW, Korg.get_mass(l.species), tref) * (1e4 / tref)^0.3 : l.vdW[1]
        @printf(io, "%.10f\t%.6f\t%d\t%s\t%d\t%d\t%.8f\t%.8e\t%.8e\t%.8e\t%d\n", l.wl * 1e8, l.log_gf,
                mol, string(l.species), mol ? 0 : atoms[1], l.species.charge, l.E_lower,
                l.gamma_rad, l.gamma_stark, gvdw, abo)
    end
end
println("COUNT ", length(ll))
"""
# GALAH molecule -> Kurucz code -> Payne Zero molecular species id.
GALAH_MOLECULE_KURUCZ = {"OTi": 822.0, "CN": 607.0, "C2": 606.0, "HMg": 112.0, "HCa": 120.0,
                         "HSi": 114.0, "HO": 108.0, "HC": 106.0}
CM_PER_EV = 8065.543937
LIGHT_SPEED_NM_S = 2.99792458e17


def _read_galah_lines(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in handle if line.strip()]
    columns = {name: [row[i] for row in rows] for i, name in enumerate(header)}
    out = {name: np.asarray(columns[name], dtype=np.float64)
           for name in ("wl_vac_A", "log_gf", "E_lower_eV", "gamma_rad", "gamma_stark", "gamma_vdw_1e4")}
    for name in ("molecule", "atomic_number", "charge", "abo"):
        out[name] = np.asarray(columns[name], dtype=np.int64)
    out["species"] = np.asarray(columns["species"])
    return out


def _log_or_zero(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, np.log10(np.where(values > 0.0, values, 1.0)), 0.0)


def build_galah_bundle(lines: dict[str, np.ndarray], *, device, dtype, scratch: Path,
                       include_zro: bool = False):
    """Payne Zero window invariants whose metal and molecular lines are GALAH's.

    Hydrogen and helium lines, continuum opacity and transfer tables stay those
    of the native Payne Zero window. Atomic GALAH lines pass through Payne
    Zero's own catalog builder in its parsed-source schema; molecular lines
    become compiled molecular arrays. ZrO has no species in the Payne Zero
    molecular equilibrium and is left out.
    """

    import dataclasses

    import torch

    from payne_zero_synthesis import atomic_lines
    from payne_zero_synthesis import line_opacity as line_opacity_engine
    from payne_zero_synthesis import molecular_lines as molecular_lines_engine
    from payne_zero_synthesis import pipeline as pz_pipeline
    from payne_zero_synthesis.molecular_equilibrium import _SPECIES_CODE_TO_MOLECULE_CODES

    base = pz_pipeline.window_invariants_for(
        wl_start_nm=SYNTH_VAC_NM[0], wl_end_nm=SYNTH_VAC_NM[1], resolution=PZ_RESOLUTION,
        molecular_lines=True, runtime_device=torch.device(device), work_dtype=dtype)
    tables = np.load(pz_pipeline._SYNTHESIS_TABLE_DIR / "line_profile_tables.npz", allow_pickle=False)

    atomic = (lines["molecule"] == 0) & (lines["atomic_number"] > 1)
    n = int(atomic.sum())
    wavelength_nm = lines["wl_vac_A"][atomic] / 10.0
    lower_cm = lines["E_lower_eV"][atomic] * CM_PER_EV
    parsed = {
        "stored_wavelength_nm": wavelength_nm,
        "raw_log_oscillator_strength": lines["log_gf"][atomic],
        "species_code": lines["atomic_number"][atomic] + lines["charge"][atomic] / 100.0,
        "first_energy_column_cm": lower_cm,
        "second_energy_column_cm": lower_cm + 1.0e7 / wavelength_nm,
        "radiative_damping_log": _log_or_zero(lines["gamma_rad"][atomic]),
        "stark_damping_log": _log_or_zero(lines["gamma_stark"][atomic]),
        "van_der_waals_damping_log": _log_or_zero(lines["gamma_vdw_1e4"][atomic]),
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
    scratch.mkdir(parents=True, exist_ok=True)
    parsed_path = scratch / "galah_atomic_parsed.npz"
    np.savez(parsed_path, **parsed)
    catalog = atomic_lines.parse_catalog(base.grid_obj, catalog_path=parsed_path, sort="catalog",
                                         apply_iso_corr=False)
    kernel = pz_pipeline._atomic_catalog_for_kernels(catalog, tables)
    line_type = kernel["line_type"]
    kernel["helium_line_type"] = line_type[np.isin(line_type, [-3, -4, -6])].astype(np.int64)
    kernel["helium_line_center_cutoff_ratio"] = np.float64(line_opacity_engine.LINE_CENTER_CUTOFF_RATIO)
    metal = np.flatnonzero((line_type == 0) | (line_type == 1) | (line_type == 3))
    chunk = pz_pipeline.SynthesisPipeline.METAL_CHUNK
    metal_invariants = [
        line_opacity_engine.precompute_invariants(
            pz_pipeline.SynthesisPipeline._slice_atomic_catalog(kernel, metal[start:start + chunk]),
            base.synthesis_wavelength_nm, runtime_device=torch.device(device))
        for start in range(0, metal.size, chunk)
    ]

    kurucz_to_id = {codes[0]: species_id for species_id, codes in _SPECIES_CODE_TO_MOLECULE_CODES.items()}
    molecular = lines["molecule"] == 1
    molecule_kurucz = dict(GALAH_MOLECULE_KURUCZ, **({"OZr": ZRO_KURUCZ_CODE} if include_zro else {}))
    kept = molecular & np.isin(lines["species"], list(molecule_kurucz))
    mol_wavelength = lines["wl_vac_A"][kept] / 10.0
    frequency = LIGHT_SPEED_NM_S / mol_wavelength
    ratio = 1.0 / PZ_RESOLUTION
    origin = int(np.floor(np.log(base.synthesis_wavelength_nm[0]) / ratio))
    compiled = {
        "center_index_1based": (np.rint(np.log(mol_wavelength) / ratio) - origin + 1).astype(np.int32),
        "classical_line_strength": (10.0 ** lines["log_gf"][kept] * 0.01502 / frequency).astype(np.float32),
        "species_code": np.array([kurucz_to_id[molecule_kurucz[sp]] for sp in lines["species"][kept]],
                                 dtype=np.int16),
        "lower_excitation_cm": (lines["E_lower_eV"][kept] * CM_PER_EV).astype(np.float32),
        "radiative_damping": (lines["gamma_rad"][kept] / (12.5664 * frequency)).astype(np.float32),
        "stark_damping": (lines["gamma_stark"][kept] / (12.5664 * frequency)).astype(np.float32),
        "van_der_waals_damping": (lines["gamma_vdw_1e4"][kept] / (12.5664 * frequency)).astype(np.float32),
        "margin_class": np.full(int(kept.sum()), 7, dtype=np.int16),
        "log_grid_ratio": ratio,
        "grid_origin_index": origin,
    }
    molecular_catalog = molecular_lines_engine.build_catalog_from_arrays(compiled)
    molecular_catalog.center_index = line_opacity_engine.nearest_grid_indices(
        base.synthesis_wavelength_nm, molecular_catalog.wavelength_nm).astype(np.int32)
    molecular_invariants = molecular_lines_engine.precompute_invariants(
        catalog=molecular_catalog, wavelength_grid_nm=base.synthesis_wavelength_nm,
        harris_profile_h0_table=tables["harris_profile_h0_table"],
        harris_profile_h1_table=tables["harris_profile_h1_table"],
        harris_profile_h2_table=tables["harris_profile_h2_table"],
        runtime_device=torch.device(device))
    bundle = dataclasses.replace(base, n_atomic=len(catalog), metal_invariant_chunks=metal_invariants,
                                 molecular_invariants=molecular_invariants, n_molecular=len(molecular_catalog))
    inventory = {
        "atomic_used": int(metal.size), "atomic_exported": n,
        "atomic_abo_converted": int((lines["abo"][atomic] == 1).sum()),
        "abo_reference_temperature_K": GALAH_ABO_REFERENCE_K,
        "molecular_used": int(len(molecular_catalog)),
        "molecular_by_species": {sp: int((lines["species"][kept] == sp).sum()) for sp in molecule_kurucz},
        "dropped_without_payne_zero_species": {sp: int((lines["species"][molecular] == sp).sum())
                                               for sp in np.unique(lines["species"][molecular & ~kept])},
        "hydrogen_and_helium_lines": "native Payne Zero window",
    }
    return bundle, inventory


def stage_pz_galah(args) -> None:
    import subprocess

    import torch

    from payne_zero_synthesis.api import _surface_flux_per_wavelength_nm
    from payne_zero_synthesis.synthesis import synthesize_structured_atmosphere

    line_path = RESULT_ROOT / "galah_lines.tsv"
    if not line_path.is_file():
        korg_project = os.environ.get("KORG_PROJECT", "/Users/jdli/Project/jorg/Korg.jl-1.0.1")
        subprocess.run(["julia", f"--project={korg_project}", "-e", GALAH_EXPORT_JL, "6465.0", "6757.0",
                        str(GALAH_ABO_REFERENCE_K), str(line_path)], check=True)
    lines = _read_galah_lines(line_path)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    bundle, inventory = build_galah_bundle(lines, device=args.device, dtype=dtype,
                                           scratch=RESULT_ROOT / "pz_galah_inputs")
    print(json.dumps(inventory, indent=2), flush=True)
    nodes = _nodes()
    if args.limit:
        nodes = [n for n in nodes if n["model_id"] in args.limit]
    fluxes, seconds = [], []
    for node in nodes:
        started = time.time()
        result, _ = synthesize_structured_atmosphere(
            node["product_path"], wavelength_start_nm=SYNTH_VAC_NM[0], wavelength_end_nm=SYNTH_VAC_NM[1],
            resolution=PZ_RESOLUTION, molecular_lines=True, device=args.device, dtype=dtype,
            window_invariants=bundle)
        wavelength = np.asarray(result.wavelength_nm, np.float64)
        total = _surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_total_per_frequency)
        continuum = _surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_continuum_per_frequency)
        fluxes.append((np.asarray(total) / np.asarray(continuum)).astype(np.float32))
        seconds.append(time.time() - started)
        print(f"{node['model_id']} {seconds[-1]:.1f}s", flush=True)
    out = RESULT_ROOT / ("pz_galah_nodes.npz" if not args.limit else "pz_galah_probe.npz")
    np.savez_compressed(out, wavelength_vacuum_A=wavelength * 10.0, normalized_flux=np.stack(fluxes),
                        model_id=np.array([n["model_id"] for n in nodes]), seconds=np.array(seconds),
                        inventory_json=json.dumps(inventory))
    (RESULT_ROOT / "pz_galah_inventory.json").write_text(json.dumps(inventory, indent=2))
    print(f"wrote {out}")


def _pair_diagnostic(obs, grid, arms, first, second, node):
    """Residual RMS of two arms at one saved atmosphere, by sub-window."""

    rest = obs.wave_air / (1.0 + node["rv_km_s"] / C_KM_S)
    preds = {}
    for arm in (first, second):
        models, spectra = arms[arm]
        i = next(k for k, m in enumerate(models) if m["model_id"] == node["model_id"])
        sampled = obs.sample(grid, broaden(spectra[i][None, :], node["sigma_extra_km_s"]), node["rv_km_s"])
        _, coef = obs.chi2(sampled)
        preds[arm] = obs.prediction(sampled[0], coef[0])
    out = {"node": node["model_id"]}
    for name, (lo, hi) in SUBWINDOWS_AIR_A.items():
        sel = obs.mask & (rest >= lo) & (rest <= hi)
        out[name] = {"model_model_rms": float(np.sqrt(np.mean((preds[first][sel] - preds[second][sel]) ** 2))),
                     f"residual_rms_{first}": float(np.sqrt(np.mean((obs.flux[sel] - preds[first][sel]) ** 2))),
                     f"residual_rms_{second}": float(np.sqrt(np.mean((obs.flux[sel] - preds[second][sel]) ** 2)))}
    return out


def stage_fit_extra(args) -> None:
    """Shared node fit for an added arm, against the stored star-level RV and masks."""

    grid = common_log_grid()
    fits = {f["key"]: f for f in json.loads((RESULT_ROOT / "fit_results.json").read_text())}
    targets = {t["key"]: t for t in json.loads((RESULT_ROOT / "targets.json").read_text())}
    compare = args.compare
    arms = {arm: load_arm(arm, grid) for arm in {args.arm, "pz_nodes", "korg_pzatm", compare}}
    compare_fit = ({r["key"]: r for r in json.loads((RESULT_ROOT / f"fit_{compare}.json").read_text())}
                   if compare not in ("pz_nodes", "korg_pzatm", "marcs_nodes", "marcs_dense") else None)
    rows = []
    for key, record in fits.items():
        obs = Observation(targets[key])
        obs.set_rest_frame_mask(record["rv0_km_s"])
        models, spectra = arms[args.arm]
        refined, _ = _fit_nodes(obs, grid, models, spectra, record["rv0_km_s"])
        best = {k: v for k, v in refined[0].items() if k != "index"}
        sampled = obs.sample(grid, broaden(spectra[refined[0]["index"]][None, :], best["sigma_extra_km_s"]),
                             best["rv_km_s"])
        _, coef = obs.chi2(sampled)
        prediction = obs.prediction(sampled[0], coef[0])
        best["rms_fraction"] = float(np.sqrt(np.mean((obs.flux[obs.mask] - prediction[obs.mask]) ** 2)))
        best["chi2_reduced"] = best["chi2"] / obs.dof_pixels
        ratios = {arm: record["arms"][arm]["best"]["chi2"] / best["chi2"]
                  for arm in ("pz_nodes", "korg_pzatm", "marcs_nodes", "marcs_dense")}
        compare_best = (compare_fit[key]["best"] if compare_fit else record["arms"][compare]["best"])
        ratios[compare] = compare_best["chi2"] / best["chi2"]
        rows.append({
            "key": key, "best": best, "compare": compare,
            "top5": [{k: r[k] for k in ("model_id", "teff", "logg", "m_h", "chi2")} for r in refined],
            "chi2_ratio_to": ratios,
            "pair_diagnostic": {
                "at_pz_node": _pair_diagnostic(obs, grid, arms, args.arm, compare,
                                               record["arms"]["pz_nodes"]["best"]),
                "at_compare_node": _pair_diagnostic(obs, grid, arms, args.arm, compare, compare_best),
            },
        })
        np.save(RESULT_ROOT / "fits" / f"{key}_{args.arm}.npy", prediction)
        r = rows[-1]
        print(f"{key:7s} {best['model_id']} chi2r={best['chi2_reduced']:.1f} "
              f"pz_nodes/this={r['chi2_ratio_to']['pz_nodes']:.3f} {compare}/this={r['chi2_ratio_to'][compare]:.3f}",
              flush=True)
    (RESULT_ROOT / f"fit_{args.arm}.json").write_text(json.dumps(rows, indent=2))


ZRO_DIR = RESULT_ROOT / "zro_equilibrium"
ZRO_SPECIES_ID = 570           # free Payne Zero molecular species id (population column 94)
ZRO_KURUCZ_CODE = 840.0
ZRO_MASS_AMU = 91.224 + 15.999
BOLTZMANN_CGS = 1.380649e-16
K_PER_EV = 11604.5


def _korg_constants() -> dict[str, np.ndarray]:
    table = np.genfromtxt(ZRO_DIR / "korg_molecular_constants.tsv", names=True)
    return {name: np.asarray(table[name]) for name in table.dtype.names}


def _saha_line_population_factor(d0_ev, mass_a, mass_b, temperature):
    """Payne Zero molecular line population per (n_A/U_A)(n_B/U_B)."""

    return (np.exp(d0_ev * K_PER_EV / temperature)
            * ((mass_a + mass_b) * temperature) ** 1.5
            / (1.8786e20 * (mass_a * temperature) ** 1.5 * (mass_b * temperature) ** 1.5))


def stage_zro_table(args) -> None:
    """Extend the synthesis molecule table by Zr (element equation) and ZrO.

    Korg's constants define ZrO: D0 is chosen so the Payne Zero Saha-form line
    population reproduces Korg's n/U over 2500-4500 K, and the remaining
    polynomial reproduces Korg's formation constant kT/K_p for the network.
    The same construction applied to TiO checks the conventions against the
    Payne Zero Kurucz entry. The extended table lives in a shadow source-catalog
    root; the packaged table is not modified.
    """

    from payne_zero_synthesis import molecular_equilibrium as me
    from payne_zero_synthesis.paths import source_catalog_root

    k = _korg_constants()
    temperature = k["T"]
    band = (temperature >= 2500.0) & (temperature <= 4500.0)
    korg_tio = BOLTZMANN_CGS * temperature / 10 ** k["logKp_TiO"] / k["U_TiO"] * k["U_Ti"] * k["U_O"]
    korg_zro = BOLTZMANN_CGS * temperature / 10 ** k["logKp_ZrO"] / k["U_ZrO"] * k["U_Zr"] * k["U_O"]

    def fitted_d0(korg, mass_a, mass_b):
        base = _saha_line_population_factor(0.0, mass_a, mass_b, temperature[band])
        return float(np.mean(np.log(korg[band] / base) * temperature[band] / K_PER_EV))

    d0_tio = fitted_d0(korg_tio, 47.867, 15.999)
    d0_zro = fitted_d0(korg_zro, 91.224, 15.999)
    zro_population_residual = np.log10(
        _saha_line_population_factor(d0_zro, 91.224, 15.999, temperature[band]) / korg_zro[band])
    tio_population_residual = np.log10(
        _saha_line_population_factor(6.87, 47.867, 15.999, temperature[band]) / korg_tio[band])

    fit = (temperature >= 1500.0) & (temperature <= 8000.0)
    ln_formation = np.log(BOLTZMANN_CGS * temperature) - np.log(10.0) * k["logKp_ZrO"]
    target = ln_formation - d0_zro * K_PER_EV / temperature + 1.5 * np.log(temperature)
    # Fit in T/1000 K for conditioning, then return to Kurucz's per-kelvin powers.
    x = temperature / 1000.0
    design = np.column_stack([-np.ones_like(x)] + [(-1.0) ** (n + 1) * x ** n for n in range(1, 6)])
    scaled, *_ = np.linalg.lstsq(design[fit], target[fit], rcond=None)
    polynomial = scaled / 1000.0 ** np.arange(6)
    coefficients = np.concatenate([[d0_zro], polynomial])
    formation_residual = (design[fit] @ scaled - target[fit]) / np.log(10.0)

    base = me.read_molecule_table(me._default_molecule_table())
    n_mol, n_eq = base.molecule_count, base.equation_count
    if base.equation_species_codes[n_eq - 1] != 100 or np.any(np.abs(base.molecule_codes[:n_mol] - 40.0) < 0.5):
        raise ValueError("unexpected base molecule table layout")
    zr_equation = n_eq - 1                                  # inserted before the electron equation
    starts = list(base.component_start_indices[: n_mol + 1])
    components = [int(c) + 1 if c >= zr_equation else int(c)
                  for c in base.component_equation_indices[: starts[n_mol]]]
    codes = list(base.molecule_codes[:n_mol])
    coeff = [base.equilibrium_coefficients[:, i].copy() for i in range(n_mol)]
    inverse_electron = n_eq + 1                             # == new equation count
    for code, comps, c in (
        (40.0, [zr_equation], np.zeros(7)),
        (40.01, [zr_equation, inverse_electron], np.zeros(7)),
        (40.02, [zr_equation, inverse_electron, inverse_electron], np.zeros(7)),
        (ZRO_KURUCZ_CODE, [8, zr_equation], coefficients),
    ):
        codes.append(code)
        coeff.append(c)
        components.extend(comps)
        starts.append(len(components))
    count = len(codes)
    molecule_codes = np.zeros(me.MAX_MOLECULES)
    molecule_codes[:count] = codes
    equilibrium = np.zeros((7, me.MAX_MOLECULES))
    equilibrium[:, :count] = np.stack(coeff, axis=1)
    start_indices = np.zeros(me.MAX_MOLECULES + 1, np.int32)
    start_indices[: count + 1] = starts
    component_indices = np.zeros(me.MAX_MOLECULAR_COMPONENTS, np.int32)
    component_indices[: len(components)] = components
    equation_species = np.zeros_like(base.equation_species_codes)
    equation_species[:zr_equation] = base.equation_species_codes[:zr_equation]
    equation_species[zr_equation] = 40
    equation_species[zr_equation + 1] = 100

    shadow = ZRO_DIR / "source_catalogs"
    for sub in ("lines", "molecules"):
        (shadow / sub).mkdir(parents=True, exist_ok=True)
        for item in (source_catalog_root() / sub).iterdir():
            link = shadow / sub / item.name
            if item.name != "molecular_equilibrium_synthesis.npz" and not link.exists():
                link.symlink_to(item.resolve())
    for item in source_catalog_root().iterdir():
        if item.is_file() and not (shadow / item.name).exists():
            (shadow / item.name).symlink_to(item.resolve())
    np.savez(shadow / "lines" / "molecular_equilibrium_synthesis.npz", molecule_count=np.int64(count),
             equation_count=np.int64(n_eq + 1), molecule_codes=molecule_codes,
             equilibrium_coefficients=equilibrium, component_start_indices=start_indices,
             component_equation_indices=component_indices, equation_species_codes=equation_species)
    record = {
        "source": "Korg 1.0.1 default_log_equilibrium_constants and default_partition_funcs",
        "zro_species_id": ZRO_SPECIES_ID, "zro_kurucz_code": ZRO_KURUCZ_CODE,
        "zro_d0_ev": d0_zro, "zro_coefficients": coefficients.tolist(),
        "zro_line_population_residual_dex_2500_4500": [float(zro_population_residual.min()),
                                                       float(zro_population_residual.max())],
        "zro_formation_fit_residual_dex_1500_8000": [float(formation_residual.min()),
                                                     float(formation_residual.max())],
        "tio_check": {"d0_ev_implied_by_korg": d0_tio, "payne_zero_d0_ev": 6.87,
                      "line_population_residual_dex_2500_4500": [float(tio_population_residual.min()),
                                                                 float(tio_population_residual.max())]},
        "table": {"molecule_count": count, "equation_count": n_eq + 1, "zr_equation_index": zr_equation},
        "shadow_root": str(shadow),
    }
    (ZRO_DIR / "zro_table.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


def register_zro_species() -> None:
    """Make this process's Payne Zero synthesis carry Zr and ZrO.

    Points the source-catalog root at the shadow tree whose synthesis molecule
    table holds Zr and ZrO, and registers the ZrO line species id and mass.
    Nothing outside this process changes.
    """

    from payne_zero_synthesis import molecular_equilibrium as me
    from payne_zero_synthesis import molecular_lines as ml
    from payne_zero_synthesis.paths import SOURCE_CATALOG_ENV

    shadow = json.loads((ZRO_DIR / "zro_table.json").read_text())["shadow_root"]
    os.environ[SOURCE_CATALOG_ENV] = shadow
    me._SPECIES_CODE_TO_MOLECULE_CODES[ZRO_SPECIES_ID] = (ZRO_KURUCZ_CODE,)
    ml.SPECIES_MASS_AMU[ZRO_SPECIES_ID] = ZRO_MASS_AMU


def rebuilt_atmosphere(product_path: str, *, device, dtype) -> dict:
    """Recompute a saved atmosphere's EOS and molecular state from its solver columns."""

    from payne_zero_synthesis.synthesis import build_structured_atmosphere_from_columns

    with np.load(product_path, allow_pickle=False) as data:
        columns = {key: np.asarray(data[key], np.float64) for key in
                   ("temperature", "column_mass", "gas_pressure", "electron_density", "mass_density",
                    "microturbulence", "elemental_abundances")}
    return build_structured_atmosphere_from_columns(
        temperature=columns["temperature"], column_mass=columns["column_mass"],
        gas_pressure=columns["gas_pressure"], electron_density=columns["electron_density"],
        elemental_abundances=columns["elemental_abundances"], microturbulence=columns["microturbulence"],
        mass_density=columns["mass_density"], device=device, dtype=dtype, molecular_lines=True)


def stage_pz_galah_zro(args) -> None:
    import torch

    from payne_zero_synthesis.api import _surface_flux_per_wavelength_nm
    from payne_zero_synthesis.synthesis import synthesize_structured_atmosphere

    include_zro = not args.control
    if include_zro:
        register_zro_species()
    lines = _read_galah_lines(RESULT_ROOT / "galah_lines.tsv")
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    bundle, inventory = build_galah_bundle(lines, device=args.device, dtype=dtype,
                                           scratch=RESULT_ROOT / "pz_galah_inputs", include_zro=include_zro)
    inventory["rebuilt_eos_state"] = True
    inventory["zro_in_molecular_equilibrium"] = include_zro
    print(json.dumps(inventory, indent=2), flush=True)
    nodes = _nodes()
    if args.limit:
        nodes = [n for n in nodes if n["model_id"] in args.limit]
    fluxes, seconds, zro_fraction = [], [], []
    for node in nodes:
        started = time.time()
        atmosphere = rebuilt_atmosphere(node["product_path"], device=args.device, dtype=dtype)
        population = np.asarray(atmosphere["partition_normalized_populations"])[:, 5, ZRO_SPECIES_ID // 6 - 1]
        zro_fraction.append(float(np.max(population)))
        result, _ = synthesize_structured_atmosphere(
            atmosphere, wavelength_start_nm=SYNTH_VAC_NM[0], wavelength_end_nm=SYNTH_VAC_NM[1],
            resolution=PZ_RESOLUTION, molecular_lines=True, device=args.device, dtype=dtype,
            window_invariants=bundle)
        wavelength = np.asarray(result.wavelength_nm, np.float64)
        total = _surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_total_per_frequency)
        continuum = _surface_flux_per_wavelength_nm(wavelength, result.eddington_flux_continuum_per_frequency)
        fluxes.append((np.asarray(total) / np.asarray(continuum)).astype(np.float32))
        seconds.append(time.time() - started)
        print(f"{node['model_id']} {seconds[-1]:.1f}s max ZrO line population {zro_fraction[-1]:.3e}", flush=True)
    name = ("pz_galah_zro" if include_zro else "pz_galah_rebuilt") + ("_probe" if args.limit else "_nodes")
    np.savez_compressed(RESULT_ROOT / f"{name}.npz", wavelength_vacuum_A=wavelength * 10.0,
                        normalized_flux=np.stack(fluxes), model_id=np.array([n["model_id"] for n in nodes]),
                        seconds=np.array(seconds), max_zro_line_population=np.array(zro_fraction),
                        inventory_json=json.dumps(inventory))
    if include_zro and not args.limit:
        (RESULT_ROOT / "pz_galah_zro_inventory.json").write_text(json.dumps(inventory, indent=2))
    print(f"wrote {name}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("fetch")
    sub.add_parser("export")
    pz = sub.add_parser("pz")
    pz.add_argument("--device", default="mps")
    pz.add_argument("--workers", type=int, default=2)
    fit = sub.add_parser("fit")
    fit.add_argument("--only", nargs="*", default=None)
    sub.add_parser("report")
    sub.add_parser("continuum-sensitivity")
    sub.add_parser("linelists")
    pzfit = sub.add_parser("pz-fit")
    pzfit.add_argument("--star", required=True)
    pzfit.add_argument("--device", default="mps")
    pzfit.add_argument("--dtype", default="float32")
    pzfit.add_argument("--max-iterations", type=int, default=8)
    pzfit.add_argument("--max-physical", type=int, default=4)
    pzfit.add_argument("--fast-only", action="store_true")
    pzfit.add_argument("--linelist", choices=("native", "galah_zro"), default="native")
    collect = sub.add_parser("pz-fit-collect")
    collect.add_argument("--linelist", choices=("native", "galah_zro"), default="native")
    sub.add_parser("zro-table")
    gzro = sub.add_parser("pz-galah-zro")
    gzro.add_argument("--device", default="cpu")
    gzro.add_argument("--dtype", default="float64")
    gzro.add_argument("--limit", nargs="*", default=None)
    gzro.add_argument("--control", action="store_true")
    extra = sub.add_parser("fit-extra")
    extra.add_argument("--arm", default="pz_galah")
    extra.add_argument("--compare", default="korg_pzatm")
    galah = sub.add_parser("pz-galah")
    galah.add_argument("--device", default="cpu")
    galah.add_argument("--dtype", default="float64")
    galah.add_argument("--limit", nargs="*", default=None)
    args = parser.parse_args()
    (RESULT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    if args.stage == "fit":
        (RESULT_ROOT / "fits").mkdir(parents=True, exist_ok=True)
    {"fetch": stage_fetch, "export": stage_export, "pz": stage_pz, "fit": stage_fit, "report": stage_report,
     "continuum-sensitivity": stage_continuum_sensitivity,
     "linelists": stage_linelists, "pz-fit": stage_pz_fit,
     "pz-fit-collect": stage_pz_fit_collect, "pz-galah": stage_pz_galah,
     "fit-extra": stage_fit_extra, "zro-table": stage_zro_table,
     "pz-galah-zro": stage_pz_galah_zro}[args.stage](args)


if __name__ == "__main__":
    main()
