"""Surrogate stage of the edge-projected interior written-gradient filter (W5e).

Protocol: ``notes/m_star_inner_loop_edge_projected_filter_preregistration_20260924.md``.
Surrogates as in ``m_star_inner_loop_filtered_written_gradient_surrogate_20260924``
(S-3850 rounds 4 and 8, S-D).  W5e removes the ``(-1)^i`` component of the
written gradient inside the working mask: the 5-point formula where the
stencil fits, and at the two layers next to each mask edge a least-squares fit
of quadratic plus alternating terms over the five nearest masked layers.
Writes ``surrogate.json`` to the result directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .m_star_inner_loop_filtered_written_gradient_surrogate_20260924 import (
    D_DEEP,
    S3850_INPUTS,
    alternating_amplitude,
    build_s3850,
    build_sd,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'results/m_star_inner_loop_edge_projected_filter_surrogate_20260924'
WINDOWS = {'L45_51': (45, 52), 'L52_58': (52, 59), 'L59_65': (59, 66), 'L66_72': (66, 73), 'L73_78': (73, 79)}
INTERIOR = np.array([-1.0, 4.0, 10.0, 4.0, -1.0]) / 16.0
SHORT = 8
LONG = 16
SETTLING = slice(60, 80)


def _runs(mask: np.ndarray) -> list[np.ndarray]:
    index = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(index) > 1)
    return np.split(index, breaks + 1)


def w5e(written: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove the alternating component of ``written`` inside each masked run."""

    out = written.copy()
    for run in _runs(np.asarray(mask, dtype=bool)):
        if run.size < 5:
            continue
        for position, layer in enumerate(run):
            if 2 <= position <= run.size - 3:
                out[layer] = float(np.dot(INTERIOR, written[layer - 2:layer + 3]))
                continue
            window = run[:5] if position < 2 else run[-5:]
            u = (window - layer).astype(np.float64)
            design = np.stack([np.ones(5), u, u * u, (-1.0) ** window], axis=1)
            coefficients, *_ = np.linalg.lstsq(design, written[window], rcond=None)
            out[layer] = float(written[layer] - coefficients[3] * (-1.0) ** layer)
    return out


def _series(surrogate, arm: str, passes: int) -> list[np.ndarray]:
    rule = 'D' if arm == 'D' else 'W'
    written_filter = w5e if arm == 'W5e' else None
    states = [surrogate.t_input.copy()]
    for _ in range(passes):
        states.append(surrogate.one_pass(states[-1], rule, written_filter))
    return states


def main() -> int:
    arms = ('W', 'W5e', 'D')
    result: dict = {'s3850': {}, 'sd': {}}
    validity = True
    criteria = {'M1': True, 'M2': True, 'M3': True, 'M4': True, 'M5': True}
    for capture in S3850_INPUTS:
        surrogate, extra = build_s3850(capture)
        mask = surrogate.mask
        series = {arm: _series(surrogate, arm, LONG) for arm in arms}
        recorded_dt = extra['exit_temperature'] - surrogate.t_input
        dt_w = series['W'][SHORT] - surrogate.t_input
        v1_error = float(np.max(np.abs(dt_w[mask] - recorded_dt[mask])) / np.max(np.abs(recorded_dt[mask])))
        log_input = np.log(surrogate.t_input)
        amplitudes = {
            name: {
                'input': alternating_amplitude(log_input, lo, hi),
                'recorded_exit': alternating_amplitude(np.log(extra['exit_temperature']), lo, hi),
                **{arm: alternating_amplitude(np.log(series[arm][SHORT]), lo, hi) for arm in arms},
            }
            for name, (lo, hi) in WINDOWS.items()
        }
        core = {k: v for k, v in amplitudes.items() if k != 'L73_78'}
        v1 = v1_error <= 0.20
        v2 = all(abs(a['W'] - a['recorded_exit']) <= 0.25 * abs(a['input']) for a in core.values())
        validity &= v1 and v2
        flux8 = {arm: surrogate.flux_ratio_error(series[arm][SHORT], mask) for arm in arms}
        flux16 = {arm: surrogate.flux_ratio_error(series[arm][LONG], mask) for arm in arms}
        settle16 = {arm: float(np.max(np.abs(series[arm][LONG][SETTLING] - series[arm][LONG - 1][SETTLING])))
                    for arm in arms}
        m1 = all(abs(a['W5e']) <= 0.2 * abs(a['input']) for a in amplitudes.values())
        m2 = flux8['W5e'] <= flux8['W'] + 0.05
        m4 = settle16['W5e'] <= 1.0
        m5 = flux16['W5e'] <= 0.10
        criteria['M1'] &= m1
        criteria['M2'] &= m2
        criteria['M4'] &= m4
        criteria['M5'] &= m5
        result['s3850'][capture] = {
            'V1_relative_max_error': v1_error, 'V1': v1, 'V2': v2,
            'alternating_amplitude_pass8': amplitudes,
            'mask_max_abs_flux_over_required_minus_1_pass8': flux8,
            'mask_max_abs_flux_over_required_minus_1_pass16': flux16,
            'pass16_deep_max_abs_change_K': settle16,
            'per_pass_deep_max_abs_change_K': {
                arm: [float(np.max(np.abs(series[arm][k][SETTLING] - series[arm][k - 1][SETTLING])))
                      for k in range(1, LONG + 1)] for arm in arms},
            'net_deep_max_abs_change_K_pass16': {
                arm: float(np.max(np.abs(series[arm][LONG][SETTLING] - surrogate.t_input[SETTLING])))
                for arm in arms},
            'M1': m1, 'M2': m2, 'M4': m4, 'M5': m5,
        }

    sd = build_sd()
    series = {arm: _series(sd, arm, LONG) for arm in arms}
    deep = np.arange(80)[D_DEEP]
    flux8 = {arm: sd.flux_ratio_error(series[arm][SHORT], deep) for arm in arms}
    settle16 = {arm: float(np.max(np.abs(series[arm][LONG][SETTLING] - series[arm][LONG - 1][SETTLING])))
                for arm in arms}
    dt_w_max = float(np.max(np.abs(series['W'][SHORT][D_DEEP] - sd.t_input[D_DEEP])))
    m3_error = float(np.max(np.abs(series['W5e'][SHORT][D_DEEP] - series['W'][SHORT][D_DEEP])) / dt_w_max)
    m2_sd = flux8['W5e'] <= flux8['W'] + 0.05
    m3 = m3_error <= 0.05
    m4_sd = settle16['W5e'] <= 1.0
    criteria['M2'] &= m2_sd
    criteria['M3'] &= m3
    criteria['M4'] &= m4_sd
    required = np.where(sd.required > 0.0, sd.required, np.nan)
    result['sd'] = {
        'deep_max_abs_flux_over_required_minus_1_pass8': flux8,
        'deep_flux_over_required_minus_1_pass8_L67_79': {
            arm: (sd.flux(series[arm][SHORT]) / required - 1.0)[D_DEEP].tolist() for arm in arms},
        'M3_relative_max_temperature_difference': m3_error,
        'pass16_deep_max_abs_change_K': settle16,
        'M2': m2_sd, 'M3': m3, 'M4': m4_sd,
    }
    result['criteria'] = {'V': bool(validity), **{k: bool(v) for k, v in criteria.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'surrogate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['criteria']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
