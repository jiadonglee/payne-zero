"""Stage-1 readout of the S3w preregistration (D-plat, one round per arm).

Reads the two local captures (S3 baseline, S3w), the Garching S3 capture for
the platform check V0, and the four-state full-physics re-evaluations when
present.  Writes ``results/m_star_inner_loop_written_gradient_20260923/
stage1_summary.json``; with ``--stage2`` it reads the Garching S3w
trajectories and the A-plat round instead and writes ``stage2_summary.json``.
Definitions follow
``notes/m_star_inner_loop_written_gradient_preregistration_20260923.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_written_gradient_20260923'
GARCHING = REPO / 'results/m_star_s3_stage_capture_20260922/d'
ARMS = {'S3': ROOT / 's3_local/d', 'S3w': ROOT / 's3w/d'}
DEEP = slice(67, 80)
P2_LIMIT = 0.10
P3_LIMIT = 0.3
S3_TRAJECTORIES = REPO / 'results/m_star_h2_inner_loop_s3_20260922'
S0_TRAJECTORIES = REPO / 'results/m_star_h2_paired_iteration_20260921'
D_START = 0.4349
D_CEILING = 5.112
A_ELIGIBILITY = 0.1698


def _stages(root: Path) -> dict:
    return json.loads((root / 'stages.json').read_text())['stages']


def _temperature(stage: dict) -> np.ndarray:
    return np.asarray(stage['structure']['temperature_K'], dtype=np.float64)


def _deep_mean(stage: dict) -> float:
    return float(stage['raw_total_flux_residual']['deep_mean_abs'])


def _arm(root: Path) -> dict:
    stages = _stages(root)
    t_input = _temperature(stages['iteration_input'])
    t_pass1 = _temperature(stages['inner_pass_01'])
    t_exit = _temperature(stages['inner_loop_exit'])
    inner_dt = t_exit - t_input
    correction_dt = np.asarray(
        stages['correction_native_grid']['temperature_correction']['delta_T_K'],
        dtype=np.float64,
    )
    ratio = np.abs(correction_dt[DEEP]) / np.maximum(np.abs(inner_dt[DEEP]), 1.0e-12)
    passes = [
        stages[f'inner_pass_{index:02d}'] for index in range(9)
        if f'inner_pass_{index:02d}' in stages
    ]
    summary = {
        'frozen_deep_mean_abs_R_raw': {name: _deep_mean(stage) for name, stage in stages.items()},
        'pass1_delta_T_K_deep': [float(v) for v in (t_pass1 - t_input)[DEEP]],
        'P1_pass1_cools_all_deep_layers': bool(np.all((t_pass1 - t_input)[DEEP] < 0.0)),
        'inner_delta_T_K_deep': [float(v) for v in inner_dt[DEEP]],
        'correction_delta_T_K_deep': [float(v) for v in correction_dt[DEEP]],
        'P2_exit_frozen_deep_mean_abs_R_raw': _deep_mean(stages['inner_loop_exit']),
        'P2_holds': bool(_deep_mean(stages['inner_loop_exit']) <= P2_LIMIT),
        'P3_max_abs_correction_over_inner_deep': float(np.max(ratio)),
        'P3_holds': bool(np.max(ratio) < P3_LIMIT),
        'remap_minus_input_delta_T_K_deep_same_index': [
            float(v) for v in (_temperature(stages['standard_grid_remap']) - t_input)[DEEP]
        ],
        'per_pass_frozen_deep_mean_abs_R_raw': [_deep_mean(stage) for stage in passes],
        'rederivation_matches_all_passes': all(
            stage['update_action'].get('rederivation_matches_observed_temperature', True)
            for stage in passes
        ),
    }
    reeval = root / 'reeval.json'
    if reeval.exists():
        summary['reeval'] = json.loads(reeval.read_text())['evaluations']
    return summary


def _input_state_metric(root: Path, row: dict) -> float:
    """Input-state deep mean |R_raw| from the round arrays and pre-inner MLT."""

    if 'R_raw_input_state_deep_mean_abs' in row:
        return float(row['R_raw_input_state_deep_mean_abs'])
    with np.load(root / 'arrays' / f"{row['run_id']}.npz", allow_pickle=False) as data:
        h_rad = np.asarray(data['Hrad'], dtype=np.float64)[DEEP]
    h_conv = np.array([
        row['layers'][str(index)]['Hconv_raw_over_target'] for index in range(67, 80)
    ])
    return float(np.mean(np.abs(
        h_rad / float(row['target_integrated_eddington_flux']) + h_conv - 1.0
    )))


def _trajectory(root: Path) -> list[dict]:
    rows = json.loads((root / 'trajectory.json').read_text())['iterations']
    out = []
    for row in rows:
        if 'error' in row:
            out.append({'iteration': row['iteration'], 'error': row['error'].splitlines()[0]})
            continue
        entry = {
            'iteration': int(row['iteration']),
            'input_state_deep_mean_abs_R_raw': _input_state_metric(root, row),
            'dT_max_abs_K': float(row['dT_max_abs_K']),
        }
        record_path = root / 'records' / f"{row['run_id']}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            timing = record.get('timing', {})
            if 'convection_inner_loop_mean_temperature_change' in timing:
                with np.load(root / 'arrays' / f"{row['run_id']}.npz", allow_pickle=False) as data:
                    net = np.asarray(data['output_temperature'], dtype=np.float64)[DEEP] - np.asarray(
                        data['temperature'], dtype=np.float64
                    )[DEEP]
                entry['inner_mean_dT_K_masked'] = float(
                    timing['convection_inner_loop_mean_temperature_change']
                )
                entry['net_mean_dT_K_deep_same_index'] = float(np.mean(net))
        out.append(entry)
    return out


def stage2() -> int:
    arms = {
        'D_S3w': _trajectory(ROOT / 'd_pchip_s3w'),
        'A_S3w': _trajectory(ROOT / 'a_pchip_s3w'),
        'D_S3': _trajectory(S3_TRAJECTORIES / 'd_pchip_s3'),
        'A_S3': _trajectory(S3_TRAJECTORIES / 'a_pchip_s3'),
        'D_S0': _trajectory(S0_TRAJECTORIES / 'd_pchip'),
        'A_S0': _trajectory(S0_TRAJECTORIES / 'a_pchip'),
    }

    def metric(rows, iteration):
        for row in rows:
            if row.get('iteration') == iteration and 'input_state_deep_mean_abs_R_raw' in row:
                return row['input_state_deep_mean_abs_R_raw']
        return None

    d_rows = [r for r in arms['D_S3w'] if 'input_state_deep_mean_abs_R_raw' in r]
    d_it09 = metric(arms['D_S3w'], 9)
    a_it09 = metric(arms['A_S3w'], 9)
    criteria = {
        'D_it09': d_it09,
        'D_below_start': None if d_it09 is None else bool(d_it09 < D_START),
        'D_max_round': max(r['input_state_deep_mean_abs_R_raw'] for r in d_rows) if d_rows else None,
        'D_all_rounds_below_ceiling': bool(all(
            r['input_state_deep_mean_abs_R_raw'] < D_CEILING for r in d_rows
        )),
        'A_it09': a_it09,
        'A_eligible': None if a_it09 is None else bool(a_it09 <= A_ELIGIBILITY),
        'A_it09_S0': metric(arms['A_S0'], 9),
        'A_S3w_below_S0_it09': None if a_it09 is None else bool(a_it09 < metric(arms['A_S0'], 9)),
    }
    a_plat = ROOT / 's3w/a/reeval.json'
    payload = {
        'criteria': criteria,
        'trajectories': arms,
        'a_plat_reeval': json.loads(a_plat.read_text())['evaluations'] if a_plat.exists() else None,
    }
    if (ROOT / 's3w/a/stages.json').exists():
        payload['a_plat_stage1_style'] = _arm(ROOT / 's3w/a')
    (ROOT / 'stage2_summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(criteria, indent=2))
    for label, rows in arms.items():
        values = [
            f"{r['input_state_deep_mean_abs_R_raw']:.4f}" if 'input_state_deep_mean_abs_R_raw' in r
            else 'err' for r in rows
        ]
        print(f'{label:6s}', ' '.join(values))
    for label in ('D_S3w', 'A_S3w'):
        print(label, 'inner mean dT / net deep mean dT:', [
            (round(r.get('inner_mean_dT_K_masked', float('nan')), 2),
             round(r.get('net_mean_dT_K_deep_same_index', float('nan')), 2))
            for r in arms[label]
        ])
    return 0


def main() -> int:
    if '--stage2' in sys.argv[1:]:
        return stage2()
    garching = _stages(GARCHING)
    local = _stages(ARMS['S3'])
    differences = {
        name: abs(_deep_mean(local[name]) - _deep_mean(garching[name]))
        for name in garching if name in local
    }
    temperature_difference = max(
        float(np.max(np.abs(_temperature(local[name]) - _temperature(garching[name]))))
        for name in garching if name in local
    )
    payload = {
        'V0_max_abs_stage_deep_mean_R_raw_difference': float(max(differences.values())),
        'V0_max_abs_stage_temperature_difference_K': temperature_difference,
        'V0_holds': bool(max(differences.values()) <= 1.0e-6),
        'arms': {label: _arm(root) for label, root in ARMS.items()},
    }
    for label, arm in payload['arms'].items():
        rows = {row['stage']: row for row in arm.get('reeval', [])}
        if 'standard_grid_remap' in rows and 'iteration_input' in rows:
            remap = rows['standard_grid_remap']['full_physics_deep_mean_abs_R_raw']
            start = rows['iteration_input']['full_physics_deep_mean_abs_R_raw']
            arm['G1_remap_full_physics_deep_mean_abs_R_raw'] = remap
            arm['G1_input_full_physics_deep_mean_abs_R_raw'] = start
            arm['G1_holds'] = bool(remap < start)
    out = ROOT / 'stage1_summary.json'
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({
        key: value for key, value in payload.items() if key != 'arms'
    }, indent=2))
    for label, arm in payload['arms'].items():
        print(label, {
            key: arm[key] for key in arm
            if key.startswith(('P1', 'P2', 'P3', 'G1', 'rederivation'))
        })
        print('   per-pass frozen deep mean:', np.round(arm['per_pass_frozen_deep_mean_abs_R_raw'], 4))
        print('   inner dT deep:', np.round(arm['inner_delta_T_K_deep'], 2))
        print('   correction dT deep:', np.round(arm['correction_delta_T_K_deep'], 2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
