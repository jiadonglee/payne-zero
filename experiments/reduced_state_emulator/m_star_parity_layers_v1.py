"""Layer-by-layer parity: Payne-Zero products against native MARCS nodes.

The final diagnostic the wall provenance question needs: do the
Payne-Zero converged products on the [M/H]-1.0 track track the MARCS
reference per layer -- electron density, total number density, and
temperature as functions of column mass? Depths are matched on column
mass (the coordinate both formats share). A close match means the EOS
data behind the wall is consistent with the reference and the wall is a
solution-space boundary; a systematic departure growing with depth and
cooling locates the first-unstable physical data for any physics
campaign.
"""

from __future__ import annotations

from bench import environment as _environment  # noqa: F401,E402

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import m_star_pipeline as pipeline
from .marcs_h5 import load_marcs_node
from .m_star_bootstrap_v1 import _load_mt, _write_json

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "m_star_parity_layers_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "results" / CAMPAIGN
DEFAULT_MARCS_GRID = REPO_ROOT / "SDSS_MARCS_atmospheres.h5"
DEFAULT_SCALEOUT_ROOT = REPO_ROOT / "results" / "m_star_pipeline_scaleout_v1"
DEFAULT_DONOR_WALK_ROOT = REPO_ROOT / "results" / "m_star_donor_walk_3600_v1"
MICROTURBULENCE_KM_S = 1.0

NODES = [
    {"temperature_K": 3800.0,
     "product_dir": DEFAULT_SCALEOUT_ROOT
     / "cases"
     / "dwarf_g+4.50_m-1.00_t3800"
     / "cap60"
     / "products"
     / "primary"},
    {"temperature_K": 3700.0,
     "product_dir": DEFAULT_DONOR_WALK_ROOT
     / "cases"
     / "dwarf_g+4.50_m-1.00_t3700"
     / "products"},
]
REPORT_DEPTHS = (1.0, 2.0, 3.0, 4.0, 5.0)


def _log_column_mass(column_mass: np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(column_mass, dtype=np.float64))


def run_parity(args: argparse.Namespace) -> dict[str, Any]:
    result_root = Path(args.result_root)
    marcs_grid = Path(args.marcs_grid)
    schema = inspect_schema(marcs_grid)
    nodes: list[dict[str, Any]] = []
    for node in NODES:
        products = sorted(Path(node["product_dir"]).glob("*.npz"))
        if not products:
            raise FileNotFoundError(f"no product in {node['product_dir']}")
        product = products[0]
        with np.load(product, allow_pickle=False) as data:
            pz = {
                "temperature": np.asarray(data["temperature"], dtype=np.float64),
                "column_mass": np.asarray(data["column_mass"], dtype=np.float64),
                "electron_density": np.asarray(
                    data["electron_density"], dtype=np.float64
                ),
                "gas_pressure": np.asarray(data["gas_pressure"], dtype=np.float64),
            }
        from .marcs_h5 import BOLTZMANN_CGS

        pz["total_number_density"] = pz["gas_pressure"] / (
            BOLTZMANN_CGS * pz["temperature"]
        )
        labels = pipeline.labels_for(
            pipeline.track_payload(
                stellar_class="dwarf",
                log_surface_gravity=4.5,
                metallicity=-1.0,
                microturbulence_km_s=MICROTURBULENCE_KM_S,
            ),
            float(node["temperature_K"]),
        )
        node_marcs = load_marcs_node(
            marcs_grid,
            labels,
            verify_sha256=False,
            expected_sha256=None,
            schema=schema,
            depth_coordinate="log_mass",
        )
        pz_log_m = _log_column_mass(pz["column_mass"])
        native_log_m = _log_column_mass(node_marcs.native_column_mass)
        order = np.argsort(pz_log_m)
        interp = lambda values: np.interp(
            native_log_m, pz_log_m[order], np.asarray(values)[order]
        )
        dz = {
            "log_m": native_log_m.tolist(),
            "marcs_log_T": np.log10(
                np.maximum(node_marcs.native_temperature, 1.0)
            ).tolist(),
            "pz_log_T": np.log10(np.maximum(interp(pz["temperature"]), 1.0)).tolist(),
            "marcs_log_ne": np.log10(
                np.maximum(node_marcs.native_electron_density, 1.0)
            ).tolist(),
            "pz_log_ne": np.log10(np.maximum(interp(pz["electron_density"]), 1.0)).tolist(),
            "marcs_log_ntot": np.log10(
                np.maximum(node_marcs.native_total_number_density, 1.0)
            ).tolist(),
            "pz_log_ntot": np.log10(
                np.maximum(interp(pz["total_number_density"]), 1.0)
            ).tolist(),
        }
        dz["d_log_T"] = [
            pz - marcs for pz, marcs in zip(dz["pz_log_T"], dz["marcs_log_T"])
        ]
        dz["d_log_ne"] = [
            pz - marcs for pz, marcs in zip(dz["pz_log_ne"], dz["marcs_log_ne"])
        ]
        dz["d_log_ntot"] = [
            pz - marcs
            for pz, marcs in zip(dz["pz_log_ntot"], dz["marcs_log_ntot"])
        ]
        nodes.append({"temperature_K": node["temperature_K"], "depth": dz})

    report: dict[str, Any] = {"nodes": nodes, "report_depths": list(REPORT_DEPTHS)}
    for node in nodes:
        log_m = np.array(node["depth"]["log_m"])
        print(f"== T={node['temperature_K']} K ([M/H] -1.0, logg 4.5)")
        for depth in REPORT_DEPTHS:
            index = int(np.argmin(np.abs(log_m - depth)))
            print(
                "   log m {lm:+.2f}: ΔlogT {dt:+.4f}  "
                "Δlog ne {dne:+.4f}  Δlog ntot {dnt:+.4f}".format(
                    lm=depth,
                    dt=node["depth"]["d_log_T"][index],
                    dne=node["depth"]["d_log_ne"][index],
                    dnt=node["depth"]["d_log_ntot"][index],
                )
            )
    output = {"campaign": CAMPAIGN, "nodes": nodes, "status": "complete"}
    protocol = {"campaign": CAMPAIGN, "marcs_grid": str(marcs_grid)}
    protocol["protocol_hash"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    output["protocol_hash"] = protocol["protocol_hash"]
    _write_json(result_root / "parity_layers.json", output)
    _write_json(result_root / "protocol.json", protocol)
    print(f"protocol hash {protocol['protocol_hash']}")
    return output


def inspect_schema(marcs_grid: Path):
    from .marcs_h5 import inspect_marcs_grid

    return inspect_marcs_grid(marcs_grid, verify_sha256=False, expected_sha256=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--marcs-grid", default=str(DEFAULT_MARCS_GRID))
    args = parser.parse_args(argv)
    run_parity(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
