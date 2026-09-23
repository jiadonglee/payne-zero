"""Readout of the S3wrc-λ preregistration (inner-loop under-relaxation).

Reads the A/D trajectories and the A oscillation-round single rounds for
λ = 0.5 and 0.33 synced from Garching, with S3w and S3wrc as references.
Writes ``results/m_star_inner_loop_relaxation_20260923/summary.json``.
Definitions follow
``notes/m_star_inner_loop_relaxation_preregistration_20260923.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

from analyze_m_star_inner_loop_written_gradient_20260923 import (
    A_ELIGIBILITY,
    D_CEILING,
    D_START,
    _trajectory,
)

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_relaxation_20260923'
REFERENCES = {
    'S3w': REPO / 'results/m_star_inner_loop_written_gradient_20260923',
    'S3wrc': REPO / 'results/m_star_inner_loop_hold_correction_20260923',
}
TAGS = {'l050': 0.5, 'l033': 0.33}
A2_LIMIT = 0.0038
SHAPE_FLOOR_K = 1.0


def _metric(rows, iteration):
    for row in rows:
        if row.get('iteration') == iteration and 'input_state_deep_mean_abs_R_raw' in row:
            return row['input_state_deep_mean_abs_R_raw']
    return None


def _inner(rows, iteration):
    for row in rows:
        if row.get('iteration') == iteration:
            return row.get('inner_mean_dT_K_masked')
    return None


def _arm(tag: str) -> dict:
    a_rows = _trajectory(ROOT / f'a_pchip_s3wrc_{tag}')
    d_rows = _trajectory(ROOT / f'd_pchip_s3wrc_{tag}')
    a_it09 = _metric(a_rows, 9)
    shape = []
    for k in (8, 9):
        current, previous = _inner(a_rows, k), _inner(a_rows, k - 1)
        shape.append(
            None if current is None or previous is None
            else bool(abs(current) <= max(abs(previous), SHAPE_FLOOR_K))
        )
    d_values = [r['input_state_deep_mean_abs_R_raw'] for r in d_rows if 'input_state_deep_mean_abs_R_raw' in r]
    d_it09 = _metric(d_rows, 9)
    out = {
        'relaxation': TAGS[tag],
        'A_it09': a_it09,
        'A1': None if a_it09 is None else bool(a_it09 <= A_ELIGIBILITY),
        'A2': None if a_it09 is None else bool(a_it09 <= A2_LIMIT),
        'S_rounds_8_9': shape,
        'S': None if None in shape else bool(all(shape)),
        'A_inner_mean_dT_K': [_inner(a_rows, k) for k in range(10)],
        'D_it01': _metric(d_rows, 1),
        'D_it09': d_it09,
        'D_passes': (
            None if d_it09 is None
            else bool(d_it09 < D_START and all(v < D_CEILING for v in d_values))
        ),
        'trajectories': {'A': a_rows, 'D': d_rows},
    }
    reeval = ROOT / f'osc_{tag}/a/reeval.json'
    if reeval.exists():
        out['osc_round_full_physics'] = {
            row['stage']: row['full_physics_deep_mean_abs_R_raw']
            for row in json.loads(reeval.read_text())['evaluations']
            if 'full_physics_deep_mean_abs_R_raw' in row
        }
    return out


def main() -> int:
    arms = {tag: _arm(tag) for tag in TAGS}
    references = {
        'A_S3w': _trajectory(REFERENCES['S3w'] / 'a_pchip_s3w'),
        'A_S3wrc': _trajectory(REFERENCES['S3wrc'] / 'a_pchip_s3wrc'),
        'D_S3wrc': _trajectory(REFERENCES['S3wrc'] / 'd_pchip_s3wrc'),
    }
    (ROOT / 'summary.json').write_text(json.dumps(
        {'arms': arms, 'references': references}, indent=2,
    ) + '\n')
    for tag, arm in arms.items():
        print(tag, {key: arm[key] for key in arm if key != 'trajectories'})

    def row(values):
        return ' '.join(
            f"{r['input_state_deep_mean_abs_R_raw']:.4f}" if 'input_state_deep_mean_abs_R_raw' in r
            else 'err' for r in values
        )

    for tag, arm in arms.items():
        print(f'A_{tag}  ', row(arm['trajectories']['A']))
        print(f'D_{tag}  ', row(arm['trajectories']['D']))
    for label, rows in references.items():
        print(f'{label:8s}', row(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
