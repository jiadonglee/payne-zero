"""Readout of the S3wrc preregistration (correction hold on inner-loop layers).

Reads the three single-round captures with their re-evaluations and the D/A
trajectories synced from Garching, with S3w and S3wr as references.  Writes
``results/m_star_inner_loop_hold_correction_20260923/summary.json``.
Definitions follow
``notes/m_star_inner_loop_hold_correction_preregistration_20260923.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analyze_m_star_inner_loop_written_gradient_20260923 import (
    A_ELIGIBILITY,
    D_CEILING,
    D_START,
    _stages,
    _trajectory,
)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from payne_zero_atmosphere.radiative_transfer import differentiate_on_depth_grid  # noqa: E402

ROOT = REPO / 'results/m_star_inner_loop_hold_correction_20260923'
S3W = REPO / 'results/m_star_inner_loop_written_gradient_20260923'
S3WR = REPO / 'results/m_star_inner_loop_state_refresh_20260923'
GRADIENT_WINDOW = slice(67, 79)
OSC_INPUT = 0.0583
G2_LIMIT = 0.10
M_LIMIT = 0.03
A2_LIMIT = 0.0038


def _structure(stage: dict, key: str) -> np.ndarray | None:
    value = stage['structure'][key]
    if isinstance(value, dict):
        value = value.get('value')
    return None if value is None else np.asarray(value, dtype=np.float64)


def _single_round(root: Path) -> dict:
    stages = _stages(root)
    seed = stages['inner_pass_00']['thermodynamics']
    initial = (
        np.asarray(seed['log_temperature_pressure_gradient'], dtype=np.float64)
        - np.asarray(seed['adiabatic_gradient'], dtype=np.float64)
    ) > 0.0
    initial[:3] = False
    removed = np.asarray(
        stages['inner_loop_exit']['mask_release']['removed_layers'], dtype=np.float64,
    ) > 0.0
    held = initial & ~removed
    correction = np.asarray(
        stages['correction_native_grid']['temperature_correction']['delta_T_K'],
        dtype=np.float64,
    )
    m0 = _structure(stages['iteration_input'], 'column_mass_g_cm2')
    t_exit = _structure(stages['inner_loop_exit'], 'temperature_K')
    t_corr = _structure(stages['correction_native_grid'], 'temperature_K')
    m_corr = _structure(stages['correction_native_grid'], 'column_mass_g_cm2')
    exit_thermo = stages['inner_loop_exit']['thermodynamics']
    delta = (
        np.asarray(exit_thermo['log_temperature_pressure_gradient'], dtype=np.float64)
        - np.asarray(exit_thermo['adiabatic_gradient'], dtype=np.float64)
    )

    def nabla(column_mass, temperature):
        return column_mass / temperature * differentiate_on_depth_grid(column_mass, temperature)

    perturbation = (nabla(m_corr, t_corr) - nabla(m0, t_exit))[GRADIENT_WINDOW]
    rows = {
        row['stage']: row['full_physics_deep_mean_abs_R_raw']
        for row in json.loads((root / 'reeval.json').read_text())['evaluations']
        if 'full_physics_deep_mean_abs_R_raw' in row
    }
    return {
        'n_held': int(np.count_nonzero(held)),
        'H0_max_abs_held_correction_K': float(np.max(np.abs(correction[held]))) if np.any(held) else 0.0,
        'H0_holds': bool(np.all(correction[held] == 0.0)),
        'M_max_abs_dnabla_over_delta': float(
            np.max(np.abs(perturbation) / np.abs(delta[GRADIENT_WINDOW]))
        ),
        'max_abs_dlogm_correction_deep': float(np.max(np.abs(np.log10(m_corr / m0))[67:80])),
        'full_physics': rows,
    }


def main() -> int:
    rounds = {
        'a_osc': _single_round(ROOT / 'osc/a'),
        'd_plat': _single_round(ROOT / 'plat/d'),
        'a_plat': _single_round(ROOT / 'plat/a'),
    }
    trajectories = {
        'A_S3wrc': _trajectory(ROOT / 'a_pchip_s3wrc'),
        'D_S3wrc': _trajectory(ROOT / 'd_pchip_s3wrc'),
        'A_S3w': _trajectory(S3W / 'a_pchip_s3w'),
        'A_S3wr': _trajectory(S3WR / 'a_pchip_s3wr'),
        'D_S3wr': _trajectory(S3WR / 'd_pchip_s3wr'),
    }

    def metric(rows, iteration):
        for row in rows:
            if row.get('iteration') == iteration and 'input_state_deep_mean_abs_R_raw' in row:
                return row['input_state_deep_mean_abs_R_raw']
        return None

    a_it09 = metric(trajectories['A_S3wrc'], 9)
    d_it09 = metric(trajectories['D_S3wrc'], 9)
    d_rows = [r for r in trajectories['D_S3wrc'] if 'input_state_deep_mean_abs_R_raw' in r]
    criteria = {
        'H0_holds': bool(all(r['H0_holds'] for r in rounds.values())),
        'M': rounds['a_osc']['M_max_abs_dnabla_over_delta'],
        'M_holds': bool(rounds['a_osc']['M_max_abs_dnabla_over_delta'] <= M_LIMIT),
        'G1_osc_remap': rounds['a_osc']['full_physics'].get('standard_grid_remap'),
        'G1_holds': bool(rounds['a_osc']['full_physics'].get('standard_grid_remap', np.inf) < OSC_INPUT),
        'G2_d_plat_remap': rounds['d_plat']['full_physics'].get('standard_grid_remap'),
        'G2_a_plat_remap': rounds['a_plat']['full_physics'].get('standard_grid_remap'),
        'G2_holds': bool(
            rounds['d_plat']['full_physics'].get('standard_grid_remap', np.inf) < G2_LIMIT
            and rounds['a_plat']['full_physics'].get('standard_grid_remap', np.inf) < G2_LIMIT
        ),
        'A_it09': a_it09,
        'A1_eligible': None if a_it09 is None else bool(a_it09 <= A_ELIGIBILITY),
        'A2_not_worse_than_S3w': None if a_it09 is None else bool(a_it09 <= A2_LIMIT),
        'D_it09': d_it09,
        'D_passes': (
            None if d_it09 is None
            else bool(d_it09 < D_START and all(
                r['input_state_deep_mean_abs_R_raw'] < D_CEILING for r in d_rows
            ))
        ),
    }
    (ROOT / 'summary.json').write_text(json.dumps(
        {'criteria': criteria, 'rounds': rounds, 'trajectories': trajectories}, indent=2,
    ) + '\n')
    print(json.dumps(criteria, indent=2))
    for label, entry in rounds.items():
        print(label, entry)
    for label, rows in trajectories.items():
        values = [
            f"{r['input_state_deep_mean_abs_R_raw']:.4f}" if 'input_state_deep_mean_abs_R_raw' in r
            else 'err' for r in rows
        ]
        print(f'{label:8s}', ' '.join(values))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
