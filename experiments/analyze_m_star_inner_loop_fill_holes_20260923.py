"""Readout of the S3wrc-λ-h single-round check (interior hole filling).

Reads ``results/m_star_inner_loop_fill_holes_20260923/single/{d,a}`` and
applies K0–K4 of ``notes/m_star_inner_loop_fill_holes_preregistration_20260923.md``.
Writes ``results/m_star_inner_loop_fill_holes_20260923/single_summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analyze_m_star_inner_loop_written_gradient_20260923 import DEEP, _stages

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from payne_zero_atmosphere.convection_inner_loop import (  # noqa: E402
    convective_mask_from_gradients,
    interior_layers_radiation_cannot_carry,
)

ROOT = REPO / 'results/m_star_inner_loop_fill_holes_20260923'
A_REFERENCE = REPO / 'results/m_star_inner_loop_relaxation_20260923/a_pchip_s3wrc_l050/arrays/a_pchip_s3wrc_l050_it00.npz'
D_REFERENCE = REPO / 'results/m_star_inner_loop_relaxation_20260923/diag_d_it00/summary.json'
HOLE = (72, 73)
D_INPUT = 0.4349
K3_LIMIT = 0.2
K0_LIMIT = 1.0e-9


def _filled(stages: dict) -> np.ndarray:
    seed = stages['inner_pass_00']
    thermo = seed['thermodynamics']
    mismatch = seed['local_mismatch']
    gradient = np.asarray(thermo['log_temperature_pressure_gradient'], dtype=np.float64)
    adiabatic = np.asarray(thermo['adiabatic_gradient'], dtype=np.float64)
    mask = convective_mask_from_gradients(gradient, adiabatic)
    return interior_layers_radiation_cannot_carry(
        mask=mask,
        logarithmic_gradient=gradient,
        adiabatic_gradient=adiabatic,
        radiative_eddington_flux=np.asarray(mismatch['Hrad'], dtype=np.float64),
        target_eddington_flux=float(mismatch['target_integrated_eddington_flux']),
    )


def _exit_mask(stages: dict) -> np.ndarray:
    thermo = stages['inner_loop_exit']['thermodynamics']
    return convective_mask_from_gradients(
        np.asarray(thermo['log_temperature_pressure_gradient'], dtype=np.float64),
        np.asarray(thermo['adiabatic_gradient'], dtype=np.float64),
    )


def main() -> int:
    a_root = ROOT / 'single/a'
    d_root = ROOT / 'single/d'
    a_stages = _stages(a_root)
    a_summary = json.loads((a_root / 'stages_summary.json').read_text())
    a_n_filled = a_summary['convection_inner_loop_timing'].get('convection_inner_loop_n_filled')
    a_out = np.asarray(a_stages['standard_grid_remap']['structure']['temperature_K'], dtype=np.float64)
    with np.load(A_REFERENCE, allow_pickle=False) as data:
        a_reference = np.asarray(data['output_temperature'], dtype=np.float64)
    a_difference = float(np.max(np.abs(a_out / a_reference - 1.0)))

    d_stages = _stages(d_root)
    d_summary = json.loads((d_root / 'stages_summary.json').read_text())
    filled = _filled(d_stages)
    exit_mask = _exit_mask(d_stages)
    evaluations = json.loads((d_root / 'reeval.json').read_text())['evaluations']
    full = {
        row['stage']: row['full_physics_deep_mean_abs_R_raw']
        for row in evaluations if 'full_physics_deep_mean_abs_R_raw' in row
    }
    with np.load(d_root / 'arrays' / 'd_reeval_standard_grid_remap.npz', allow_pickle=False) as data:
        remap_residual = np.asarray(data['R_raw'], dtype=np.float64)
    correction = np.asarray(
        d_stages['correction_native_grid']['temperature_correction']['delta_T_K'], dtype=np.float64,
    )
    reference = json.loads(D_REFERENCE.read_text())
    criteria = {
        'K0_a_n_filled': a_n_filled,
        'K0_a_max_rel_difference_vs_l050_it00': a_difference,
        'K0': bool(a_n_filled == 0 and a_difference <= K0_LIMIT),
        'K1_filled_layers': np.flatnonzero(filled).tolist(),
        'K1': bool(all(filled[layer] for layer in HOLE)),
        'K2_exit_convective_hole': [bool(exit_mask[layer]) for layer in HOLE],
        'K2': bool(all(exit_mask[layer] for layer in HOLE)),
        'K3_remap_R_raw_hole': [float(remap_residual[layer]) for layer in HOLE],
        'K3': bool(all(abs(remap_residual[layer]) <= K3_LIMIT for layer in HOLE)),
        'K4_remap_full_physics': full.get('standard_grid_remap'),
        'K4': bool(full.get('standard_grid_remap', np.inf) < D_INPUT),
        'n_filled_runner': d_summary['convection_inner_loop_timing'].get('convection_inner_loop_n_filled'),
    }
    payload = {
        'criteria': criteria,
        'full_physics': full,
        'reference_without_filling': {
            'full_physics': reference.get('full_physics'),
        },
        'd_correction_delta_T_K_deep': [float(v) for v in correction[DEEP]],
        'd_remap_signed_R_raw_deep': [float(v) for v in remap_residual[DEEP]],
    }
    (ROOT / 'single_summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
