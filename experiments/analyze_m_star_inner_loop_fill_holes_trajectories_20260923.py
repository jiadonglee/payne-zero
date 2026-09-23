"""Readout of the S3wrc-λ-h trajectories (second stage of the fill-holes arm).

Reads the D/A trajectories in ``results/m_star_inner_loop_fill_holes_20260923``
and applies D1, DS, DH, AH of
``notes/m_star_inner_loop_fill_holes_preregistration_20260923.md``.  Writes
``trajectories_summary.json`` next to them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze_m_star_inner_loop_written_gradient_20260923 import D_CEILING, D_START, _trajectory

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_fill_holes_20260923'
RELAXATION = REPO / 'results/m_star_inner_loop_relaxation_20260923'
HOLE = (72, 73)
DH_LIMIT = 0.2
SHAPE_FLOOR_K = 1.0


def _rows(root: Path) -> list[dict]:
    return json.loads((root / 'trajectory.json').read_text())['iterations']


def _layer_residual(root: Path, row: dict, layer: int) -> float:
    with np.load(root / 'arrays' / f"{row['run_id']}.npz", allow_pickle=False) as data:
        h_rad = float(np.asarray(data['Hrad'], dtype=np.float64)[layer])
    target = float(row['target_integrated_eddington_flux'])
    return h_rad / target + float(row['layers'][str(layer)]['Hconv_raw_over_target']) - 1.0


def _n_filled(root: Path, row: dict) -> float | None:
    record = json.loads((root / 'records' / f"{row['run_id']}.json").read_text())
    return record.get('timing', {}).get('convection_inner_loop_n_filled')


def main() -> int:
    d_root = ROOT / 'd_pchip_s3wrc_l050_h'
    a_root = ROOT / 'a_pchip_s3wrc_l050_h'
    d = _trajectory(d_root)
    a = _trajectory(a_root)
    d_rows = _rows(d_root)
    a_rows = _rows(a_root)
    d_values = [r['input_state_deep_mean_abs_R_raw'] for r in d if 'input_state_deep_mean_abs_R_raw' in r]
    d_inner = [r.get('inner_mean_dT_K_masked') for r in d]
    shape = [
        bool(abs(d_inner[k]) <= max(abs(d_inner[k - 1]), SHAPE_FLOOR_K))
        for k in (8, 9) if len(d_inner) > k and d_inner[k] is not None
    ]
    hole = {
        str(row['iteration']): [_layer_residual(d_root, row, layer) for layer in HOLE]
        for row in d_rows if 'layers' in row and row['layers']
    }
    a_filled = [_n_filled(a_root, row) for row in a_rows]
    d_filled = [_n_filled(d_root, row) for row in d_rows]
    reference_a = _trajectory(RELAXATION / 'a_pchip_s3wrc_l050')
    criteria = {
        'D_it09': d_values[9] if len(d_values) > 9 else None,
        'D1': bool(len(d_values) > 9 and d_values[9] < D_START and all(v < D_CEILING for v in d_values)),
        'D_max_round': max(d_values) if d_values else None,
        'DS_rounds_8_9': shape,
        'DS': bool(len(shape) == 2 and all(shape)),
        'DH_it09_R_raw_hole': hole.get('9'),
        'DH': bool('9' in hole and all(abs(v) <= DH_LIMIT for v in hole['9'])),
        'AH_n_filled_per_round': a_filled,
        'AH': bool(a_filled and all(v == 0 for v in a_filled)),
        'A_it09': a[9]['input_state_deep_mean_abs_R_raw'] if len(a) > 9 else None,
        'A_it09_without_filling': reference_a[9]['input_state_deep_mean_abs_R_raw'] if len(reference_a) > 9 else None,
        'D_n_filled_per_round': d_filled,
    }
    payload = {
        'criteria': criteria,
        'd_trajectory': d,
        'a_trajectory': a,
        'd_hole_input_state_R_raw_per_round': hole,
    }
    (ROOT / 'trajectories_summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(criteria, indent=2))
    print('D', ' '.join(f"{v:.4f}" for v in d_values))
    print('A', ' '.join(f"{r['input_state_deep_mean_abs_R_raw']:.4f}" for r in a if 'input_state_deep_mean_abs_R_raw' in r))
    print('D inner mean dT:', [None if v is None else round(v, 2) for v in d_inner])
    print('D hole R per round:', {k: [round(x, 3) for x in v] for k, v in hole.items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
