"""Readout of the S3wr preregistration (trial-temperature state refresh).

Reads the S3wr D-plat/A-plat captures and re-evaluations and the D/A
trajectories synced from Garching, with the S3w and S3 records as references.
Writes ``results/m_star_inner_loop_state_refresh_20260923/summary.json``.
Definitions follow
``notes/m_star_inner_loop_state_refresh_preregistration_20260923.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_m_star_inner_loop_written_gradient_20260923 import (
    A_ELIGIBILITY,
    D_CEILING,
    D_START,
    DEEP,
    _arm,
    _stages,
    _trajectory,
)

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_state_refresh_20260923'
S3W = REPO / 'results/m_star_inner_loop_written_gradient_20260923'
C0_REFERENCE = {
    'd': REPO / 'results/m_star_s3_stage_capture_20260922/d',
    'a': S3W / 's3w/a',
}
S3W_GAP = {'d': 0.5798 - 0.0137, 'a': 0.6099 - 0.0227}
G1_LIMIT = 0.6167
G2_LIMIT = 0.4803
C0_LIMIT = 1.0e-3


def _pass0_flux(root: Path) -> np.ndarray:
    stage = _stages(root)['inner_pass_00']
    return np.asarray(stage['local_mismatch']['Hconv_smoothed'], dtype=np.float64)


def _plat(case: str) -> dict:
    root = ROOT / 's3wr' / case
    flux = _pass0_flux(root)[DEEP]
    reference = _pass0_flux(C0_REFERENCE[case])[DEEP]
    c0 = float(np.max(np.abs(flux - reference) / np.maximum(np.abs(reference), 1.0e-300)))
    rows = {row['stage']: row for row in json.loads((root / 'reeval.json').read_text())['evaluations']}
    frozen = _arm(root)['P2_exit_frozen_deep_mean_abs_R_raw']
    full_pass8 = rows['inner_pass_08']['full_physics_deep_mean_abs_R_raw']
    gap = full_pass8 - frozen
    out = {
        'C0_max_rel_pass0_Hconv_difference_deep': c0,
        'C0_holds': bool(c0 <= C0_LIMIT),
        'frozen_exit_deep_mean_abs_R_raw': frozen,
        'full_inner_pass_08': full_pass8,
        'full_remap': rows['standard_grid_remap']['full_physics_deep_mean_abs_R_raw'],
        'full_input': rows['iteration_input']['full_physics_deep_mean_abs_R_raw'],
        'gap_pass_08': gap,
        'gap_S3w': S3W_GAP[case],
        'gap_halved': bool(gap <= 0.5 * S3W_GAP[case]),
        'stages': _arm(root),
        'reeval': list(rows.values()),
    }
    if 'inner_pass_01' in rows:
        out['full_inner_pass_01'] = rows['inner_pass_01']['full_physics_deep_mean_abs_R_raw']
    return out


def main() -> int:
    plats = {case: _plat(case) for case in ('d', 'a')}
    trajectories = {
        'D_S3wr': _trajectory(ROOT / 'd_pchip_s3wr'),
        'A_S3wr': _trajectory(ROOT / 'a_pchip_s3wr'),
        'D_S3w': _trajectory(S3W / 'd_pchip_s3w'),
        'A_S3w': _trajectory(S3W / 'a_pchip_s3w'),
    }

    def metric(rows, iteration):
        for row in rows:
            if row.get('iteration') == iteration and 'input_state_deep_mean_abs_R_raw' in row:
                return row['input_state_deep_mean_abs_R_raw']
        return None

    d_rows = [r for r in trajectories['D_S3wr'] if 'input_state_deep_mean_abs_R_raw' in r]
    criteria = {
        'C0_holds': bool(plats['d']['C0_holds'] and plats['a']['C0_holds']),
        'G1_D_plat_remap': plats['d']['full_remap'],
        'G1_holds': bool(plats['d']['full_remap'] < G1_LIMIT),
        'G2_A_plat_remap': plats['a']['full_remap'],
        'G2_holds': bool(plats['a']['full_remap'] < G2_LIMIT),
        'gap_halved_both': bool(plats['d']['gap_halved'] and plats['a']['gap_halved']),
        'A_it09': metric(trajectories['A_S3wr'], 9),
        'A_eligible': (
            None if metric(trajectories['A_S3wr'], 9) is None
            else bool(metric(trajectories['A_S3wr'], 9) <= A_ELIGIBILITY)
        ),
        'A_it09_S3w': metric(trajectories['A_S3w'], 9),
        'D_it09': metric(trajectories['D_S3wr'], 9),
        'D_below_start': (
            None if metric(trajectories['D_S3wr'], 9) is None
            else bool(metric(trajectories['D_S3wr'], 9) < D_START)
        ),
        'D_all_rounds_below_ceiling': bool(all(
            r['input_state_deep_mean_abs_R_raw'] < D_CEILING for r in d_rows
        )),
    }
    (ROOT / 'summary.json').write_text(json.dumps(
        {'criteria': criteria, 'plats': plats, 'trajectories': trajectories}, indent=2,
    ) + '\n')
    print(json.dumps(criteria, indent=2))
    for case, plat in plats.items():
        print(case, {key: plat[key] for key in plat if key not in ('stages', 'reeval')})
    for label, rows in trajectories.items():
        values = [
            f"{r['input_state_deep_mean_abs_R_raw']:.4f}" if 'input_state_deep_mean_abs_R_raw' in r
            else 'err' for r in rows
        ]
        print(f'{label:7s}', ' '.join(values))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
