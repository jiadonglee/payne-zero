"""Surrogate test of mask-interior filters of the written gradient (W5, W121i).

Protocol: ``notes/m_star_inner_loop_interior_filtered_written_gradient_surrogate_preregistration_20260924.md``.
The surrogates are those of
``m_star_inner_loop_filtered_written_gradient_surrogate_20260924`` (S-3850 rounds 4
and 8, S-D).  Candidates filter the written gradient only where the whole
stencil lies inside the working mask: W5 with (-1, 4, 10, 4, -1)/16, W121i with
(1, 2, 1)/4.  Writes ``surrogate.json`` to the result directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .m_star_inner_loop_filtered_written_gradient_surrogate_20260924 import (
    D_DEEP,
    S3850_INPUTS,
    WINDOWS,
    alternating_amplitude,
    build_s3850,
    build_sd,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/m_star_inner_loop_interior_filtered_written_gradient_surrogate_20260924'
WEIGHTS = {
    'W5': np.array([-1.0, 4.0, 10.0, 4.0, -1.0]) / 16.0,
    'W121i': np.array([1.0, 2.0, 1.0]) / 4.0,
}
CANDIDATES = ('W5', 'W121i')
DEEP_SETTLING = slice(60, 80)


def interior_filter(weights: np.ndarray):
    half = weights.size // 2

    def apply(written: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = written.copy()
        for index in range(half, written.size - half):
            if mask[index - half:index + half + 1].all():
                out[index] = float(np.dot(weights, written[index - half:index + half + 1]))
        return out

    return apply


FILTERS = {name: interior_filter(weights) for name, weights in WEIGHTS.items()}


def _runs(surrogate) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Final and one-before-final temperature for W, D and each candidate."""

    out = {}
    for arm in ('W', 'D', *CANDIDATES):
        rule = 'D' if arm == 'D' else 'W'
        written_filter = FILTERS.get(arm)
        before = surrogate.run(rule, passes=7, written_filter=written_filter)
        final = surrogate.one_pass(before, rule, written_filter)
        out[arm] = (final, before)
    return out


def _settling(runs, arm) -> float:
    final, before = runs[arm]
    return float(np.max(np.abs(final[DEEP_SETTLING] - before[DEEP_SETTLING])))


def main() -> int:
    result: dict = {'s3850': {}, 'sd': {}, 'criteria': {}}
    status = {name: {'M1': True, 'M2': True, 'M3': True, 'M4': True} for name in CANDIDATES}
    validity = True
    for capture in S3850_INPUTS:
        surrogate, extra = build_s3850(capture)
        mask = surrogate.mask
        runs = _runs(surrogate)
        recorded_dt = extra['exit_temperature'] - surrogate.t_input
        dt_w = runs['W'][0] - surrogate.t_input
        v1_error = float(np.max(np.abs(dt_w[mask] - recorded_dt[mask])) / np.max(np.abs(recorded_dt[mask])))
        log_input = np.log(surrogate.t_input)
        amplitudes = {
            name: {
                'input': alternating_amplitude(log_input, lo, hi),
                'recorded_exit': alternating_amplitude(np.log(extra['exit_temperature']), lo, hi),
                **{arm: alternating_amplitude(np.log(runs[arm][0]), lo, hi) for arm in runs},
            }
            for name, (lo, hi) in WINDOWS.items()
        }
        v1 = v1_error <= 0.20
        v2 = all(abs(a['W'] - a['recorded_exit']) <= 0.25 * abs(a['input']) for a in amplitudes.values())
        validity &= v1 and v2
        flux_error = {arm: surrogate.flux_ratio_error(runs[arm][0], mask) for arm in runs}
        settling = {arm: _settling(runs, arm) for arm in runs}
        entry = {
            'V1_relative_max_error': v1_error, 'V1': v1, 'V2': v2,
            'alternating_amplitude': amplitudes,
            'mask_max_abs_flux_over_required_minus_1': flux_error,
            'pass8_deep_max_abs_change_K': settling,
            'delta_T_max_abs_K': {arm: float(np.max(np.abs(runs[arm][0][mask] - surrogate.t_input[mask])))
                                  for arm in runs},
        }
        for name in CANDIDATES:
            m1 = all(abs(a[name]) <= 0.2 * abs(a['input']) for a in amplitudes.values())
            m2 = flux_error[name] <= flux_error['W'] + 0.05
            m4 = settling[name] <= settling['W'] + 1.0
            entry[name] = {'M1': m1, 'M2': m2, 'M4': m4}
            status[name]['M1'] &= m1
            status[name]['M2'] &= m2
            status[name]['M4'] &= m4
        result['s3850'][capture] = entry

    sd = build_sd()
    runs = _runs(sd)
    deep = np.arange(80)[D_DEEP]
    flux_error = {arm: sd.flux_ratio_error(runs[arm][0], deep) for arm in runs}
    settling = {arm: _settling(runs, arm) for arm in runs}
    dt_w_max = float(np.max(np.abs(runs['W'][0][D_DEEP] - sd.t_input[D_DEEP])))
    entry = {
        'deep_max_abs_flux_over_required_minus_1': flux_error,
        'pass8_deep_max_abs_change_K': settling,
        'delta_T_K_L71': {arm: float(runs[arm][0][71] - sd.t_input[71]) for arm in runs},
        'delta_T_K_L79': {arm: float(runs[arm][0][79] - sd.t_input[79]) for arm in runs},
    }
    for name in CANDIDATES:
        m2 = flux_error[name] <= flux_error['W'] + 0.05
        m3_error = float(np.max(np.abs(runs[name][0][D_DEEP] - runs['W'][0][D_DEEP])) / dt_w_max)
        m3 = m3_error <= 0.05
        m4 = settling[name] <= settling['W'] + 1.0
        entry[name] = {'M2': m2, 'M3_relative_max_temperature_difference': m3_error, 'M3': m3, 'M4': m4}
        status[name]['M2'] &= m2
        status[name]['M3'] &= m3
        status[name]['M4'] &= m4
    result['sd'] = entry
    result['criteria'] = {'V': bool(validity), **{name: {k: bool(v) for k, v in s.items()}
                                                   for name, s in status.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'surrogate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
