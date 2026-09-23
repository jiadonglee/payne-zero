"""Readout of the S3wr oscillation diagnosis (A trajectory, round-5 input).

Reads the stage capture and re-evaluations in
``results/m_star_inner_loop_state_refresh_20260923/diag_a_it05/a`` and applies
the outcome order of ``notes/m_star_s3wr_oscillation_diagnosis_20260923.md``.
Writes ``diag_a_it05/summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_m_star_inner_loop_written_gradient_20260923 import DEEP, _arm, _stages

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_state_refresh_20260923/diag_a_it05'
TRAJECTORY_INPUT_IT05 = 0.0583
TRAJECTORY_INPUT_IT06 = 0.1367
O1_GAP = 0.05


def main() -> int:
    capture = ROOT / 'a'
    stages = _stages(capture)
    arm = _arm(capture)
    first = stages['inner_pass_00']
    mismatch = first['local_mismatch']
    target = float(mismatch['target_integrated_eddington_flux'])
    required = np.maximum(target - np.asarray(mismatch['Hrad'], dtype=np.float64), 0.0)
    thermo = first['thermodynamics']
    mask = (
        np.asarray(thermo['log_temperature_pressure_gradient'], dtype=np.float64)
        - np.asarray(thermo['adiabatic_gradient'], dtype=np.float64)
    ) > 0.0
    mask[:3] = False
    fluxes = [
        np.asarray(stages[f'inner_pass_{k:02d}']['local_mismatch']['Hconv_smoothed'], dtype=np.float64)
        for k in range(9)
    ]
    direction = []
    for k in (1, 2):
        intended = np.sign(required - fluxes[k - 1])
        actual = np.sign(fluxes[k] - fluxes[k - 1])
        active = mask & (intended != 0.0) & (actual != 0.0)
        direction.append({
            'pass': k,
            'n_layers': int(np.count_nonzero(active)),
            'fraction_opposite_to_intent': float(np.mean(intended[active] != actual[active])),
            'fraction_opposite_to_intent_deep': float(np.mean(
                (intended != actual)[DEEP][active[DEEP]]
            )) if np.any(active[DEEP]) else None,
        })
    frozen = arm['per_pass_frozen_deep_mean_abs_R_raw']
    peak = max(frozen[1:])
    rows = {
        row['stage']: row
        for row in json.loads((capture / 'reeval.json').read_text())['evaluations']
    }
    full = {name: row['full_physics_deep_mean_abs_R_raw'] for name, row in rows.items()}
    gap = full['inner_pass_08'] - frozen[8]
    inner_dt = np.max(np.abs(arm['inner_delta_T_K_deep']))
    post_dt = np.max(np.abs(np.asarray(arm['remap_minus_input_delta_T_K_deep_same_index'])
                            - np.asarray(arm['inner_delta_T_K_deep'])))
    outcome = {
        'O1_gap_pass08': gap,
        'O1': bool(gap > O1_GAP),
        'O2_rise_then_fall': bool(peak > frozen[0] and frozen[8] < peak),
        'O2_peak_frozen': peak,
        'direction_test': direction,
        'O2a': None,
        'O2b': None,
        'O3_monotone': bool(all(b <= a for a, b in zip(frozen, frozen[1:]))),
        'inner_dT_max_deep_K': float(inner_dt),
        'post_loop_dT_max_deep_K_same_index': float(post_dt),
    }
    if outcome['O2_rise_then_fall']:
        opposite = np.mean([d['fraction_opposite_to_intent'] for d in direction])
        outcome['O2a'] = bool(opposite > 0.5)
        outcome['O2b'] = bool(opposite <= 0.5)
    for key in ('O1', 'O2a', 'O2b'):
        if outcome.get(key):
            outcome['selected'] = key
            break
    else:
        outcome['selected'] = 'O3' if outcome['O3_monotone'] else 'none'
    payload = {
        'consistency': {
            'input_full_physics': full.get('iteration_input'),
            'trajectory_it05_input': TRAJECTORY_INPUT_IT05,
            'input_matches': bool(abs(full.get('iteration_input', np.nan) - TRAJECTORY_INPUT_IT05) <= 1.0e-3),
            'remap_full_physics': full.get('standard_grid_remap'),
            'trajectory_it06_input': TRAJECTORY_INPUT_IT06,
        },
        'per_pass_frozen_deep_mean_abs_R_raw': frozen,
        'full_physics': full,
        'outcome': outcome,
        'inner_delta_T_K_deep': arm['inner_delta_T_K_deep'],
        'correction_delta_T_K_deep': arm['correction_delta_T_K_deep'],
        'pass1_delta_T_K_deep': arm['pass1_delta_T_K_deep'],
    }
    (ROOT / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: payload[k] for k in ('consistency', 'full_physics', 'outcome')}, indent=2))
    print('per-pass frozen:', np.round(frozen, 4))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
