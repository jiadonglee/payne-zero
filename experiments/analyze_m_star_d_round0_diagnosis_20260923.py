"""Readout of the D round-0 diagnosis (S3wrc, λ = 0.5, from the D control state).

Reads ``results/m_star_inner_loop_relaxation_20260923/diag_d_it00/d`` and
applies the outcome order of ``notes/m_star_d_round0_diagnosis_20260923.md``.
Writes ``diag_d_it00/summary.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from analyze_m_star_inner_loop_written_gradient_20260923 import DEEP, _arm, _stages

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from payne_zero_atmosphere.radiative_transfer import differentiate_on_depth_grid  # noqa: E402

ROOT = REPO / 'results/m_star_inner_loop_relaxation_20260923/diag_d_it00'
INPUT_REFERENCE = 0.4349
ROUND1_REFERENCE = 3.9575
F1_GAP = 0.1


def _structure(stage: dict, key: str):
    value = stage['structure'][key]
    if isinstance(value, dict):
        value = value.get('value')
    return None if value is None else np.asarray(value, dtype=np.float64)


def _nabla(column_mass, temperature):
    return column_mass / temperature * differentiate_on_depth_grid(column_mass, temperature)


def main() -> int:
    capture = ROOT / 'd'
    stages = _stages(capture)
    arm = _arm(capture)
    frozen = arm['per_pass_frozen_deep_mean_abs_R_raw']
    evaluations = json.loads((capture / 'reeval.json').read_text())['evaluations']
    full = {
        row['stage']: row['full_physics_deep_mean_abs_R_raw']
        for row in evaluations if 'full_physics_deep_mean_abs_R_raw' in row
    }
    anchor = [row for row in evaluations if row.get('stage') == 'control_anchor']

    m0 = _structure(stages['iteration_input'], 'column_mass_g_cm2')
    t_in = _structure(stages['iteration_input'], 'temperature_K')
    t_exit = _structure(stages['inner_loop_exit'], 'temperature_K')
    t_out = _structure(stages['post_inner_recompute'], 'temperature_K')
    t_corr = _structure(stages['correction_native_grid'], 'temperature_K')
    m_corr = _structure(stages['correction_native_grid'], 'column_mass_g_cm2')
    t_remap = _structure(stages['standard_grid_remap'], 'temperature_K')
    m_remap = _structure(stages['standard_grid_remap'], 'column_mass_g_cm2')
    thermo = stages['inner_loop_exit']['thermodynamics']
    delta = (
        np.asarray(thermo['log_temperature_pressure_gradient'], dtype=np.float64)
        - np.asarray(thermo['adiabatic_gradient'], dtype=np.float64)
    )
    window = slice(67, 79)
    n_exit = _nabla(m0, t_exit)
    n_out = _nabla(m0, t_out)
    n_corr = _nabla(m_corr, t_corr)
    n_remap = _nabla(m_remap, t_remap)

    def rel(a, b):
        return float(np.max(np.abs((a - b)[window]) / np.abs(delta[window])))

    def on_m0(m, t):
        return np.exp(np.interp(np.log(m0), np.log(m), np.log(t)))

    correction = np.asarray(
        stages['correction_native_grid']['temperature_correction']['delta_T_K'], dtype=np.float64,
    )
    remap_arrays = capture / 'arrays' / 'd_reeval_standard_grid_remap.npz'
    remap_residual = None
    if remap_arrays.exists():
        with np.load(remap_arrays, allow_pickle=False) as data:
            remap_residual = [float(v) for v in np.asarray(data['R_raw'], dtype=np.float64)[DEEP]]

    gap = full.get('inner_pass_08', np.nan) - frozen[8]
    outcome = {
        'gap_pass08': float(gap),
        'F1': bool(gap > F1_GAP),
        'F2': bool(gap <= F1_GAP and full.get('standard_grid_remap', np.nan)
                   > 2.0 * full.get('inner_pass_08', np.nan)),
        'F3': bool(gap <= F1_GAP and full.get('standard_grid_remap', np.nan)
                   <= 2.0 * full.get('inner_pass_08', np.nan)
                   and full.get('inner_pass_08', np.nan) > full.get('iteration_input', np.nan)),
    }
    outcome['selected'] = next((key for key in ('F1', 'F2', 'F3') if outcome[key]), 'none')
    payload = {
        'consistency': {
            'control_anchor': anchor,
            'input_full_physics': full.get('iteration_input'),
            'input_matches': bool(abs(full.get('iteration_input', np.nan) - INPUT_REFERENCE) <= 1.0e-3),
            'remap_full_physics': full.get('standard_grid_remap'),
            'remap_matches_round1': bool(abs(full.get('standard_grid_remap', np.nan) - ROUND1_REFERENCE) <= 1.0e-2),
        },
        'per_pass_frozen_deep_mean_abs_R_raw': frozen,
        'full_physics': full,
        'outcome': outcome,
        'gradient_perturbation_over_delta_L67_78': {
            'relaxation_step': rel(n_out, n_exit),
            'correction_step': rel(n_corr, n_out),
            'remap_step': rel(n_remap, n_corr),
            'post_loop_total': rel(n_remap, n_exit),
        },
        'delta_T_K': {
            'inner_exit_minus_input_deep': [float(v) for v in (t_exit - t_in)[DEEP]],
            'relaxed_minus_input_deep': [float(v) for v in (t_out - t_in)[DEEP]],
            'correction_recorded_deep': [float(v) for v in correction[DEEP]],
            'correction_recorded_max_abs_all_layers': float(np.max(np.abs(correction))),
            'correction_recorded_argmax_layer': int(np.argmax(np.abs(correction))),
            'remap_on_m0_minus_relaxed_deep': [
                float(v) for v in (on_m0(m_remap, t_remap) - t_out)[DEEP]
            ],
        },
        'remap_state_signed_R_raw_deep': remap_residual,
    }
    (ROOT / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({key: payload[key] for key in (
        'consistency', 'full_physics', 'outcome', 'gradient_perturbation_over_delta_L67_78',
    )}, indent=2))
    print('per-pass frozen:', np.round(frozen, 4))
    for key, values in payload['delta_T_K'].items():
        print(key, np.round(values, 1) if isinstance(values, list) else values)
    if remap_residual is not None:
        print('remap-state signed R_raw L67-79:', np.round(remap_residual, 3))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
