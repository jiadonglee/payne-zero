"""Where the layer-alternating ln T mode inside the inner-loop mask comes from.

Uses the per-pass stage records of
``results/m_star_inner_loop_nonstationary_capture_20260924/<capture>/stages.json``.
Inside a pass with ``correct_written_gradient`` the written gradient becomes
``w_new = w_old + (target - read)`` on correctable layers, and the temperature
is integrated inward on the written stencil, so ``target - read = w_p - w_{p-1}``
exactly (no temperature cap is touched in these captures).  From consecutive
pass states this gives, per pass, ``target - read`` and its split into the
adiabatic gradient and the trial superadiabatic excess.

For every quantity the signed amplitude of the ``(-1)^i`` component is taken
per window (L45-51, L52-58, L59-65, L66-72), which is insensitive to smooth
curvature.  Reported per capture and window, averaged over the 8 passes:

* read blindness: amp(read) / amp(written) for the pass input;
* the adiabatic gradient's response to the mode: amp(nabla_ad) / amp(ln T);
* the source split of amp(target - read): nabla_ad, trial excess, minus read;
* the per-pass change of the mode, amp(Delta ln T) / amp(ln T), and the same
  for the direct-write rule (``w_new = target``) applied to the same pass.

Writes ``alternating_mode.json`` next to the captures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_nonstationary_capture_20260924'
WINDOWS = {'L45_51': (45, 52), 'L52_58': (52, 59), 'L59_65': (59, 66), 'L66_72': (66, 73)}
PASSES = 8


def _amp(values: np.ndarray, lo: int, hi: int) -> float:
    index = np.arange(lo, hi)
    residual = values[index] - 0.5 * (values[index - 1] + values[index + 1])
    return float(np.mean(residual * (-1.0) ** index) / 2.0)


def _written(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    gradient = np.zeros_like(temperature)
    gradient[1:] = np.log(temperature[1:] / temperature[:-1]) / np.log(pressure[1:] / pressure[:-1])
    gradient[0] = gradient[1]
    return gradient


def _integrate(temperature: np.ndarray, pressure: np.ndarray, nabla: np.ndarray, mask: np.ndarray) -> np.ndarray:
    proposed = temperature.copy()
    for index in range(1, temperature.size):
        if not mask[index]:
            continue
        reference = proposed[index - 1] if mask[index - 1] else temperature[index - 1]
        proposed[index] = reference * np.exp(nabla[index] * np.log(pressure[index] / pressure[index - 1]))
    return proposed


def _stage(stages: dict, pass_index: int) -> dict:
    stage = stages[f'inner_pass_{pass_index:02d}']
    thermo = stage['thermodynamics']
    return {
        'temperature': np.asarray(stage['structure']['temperature_K'], dtype=np.float64),
        'pressure': np.asarray(stage['structure']['total_pressure_dyn_cm2'], dtype=np.float64),
        'read': np.asarray(thermo['log_temperature_pressure_gradient'], dtype=np.float64),
        'adiabatic': np.asarray(thermo['adiabatic_gradient'], dtype=np.float64),
    }


def analyse(capture_dir: Path) -> dict:
    stages = json.loads((capture_dir / 'stages.json').read_text())['stages']
    states = [_stage(stages, k) for k in range(PASSES + 1)]
    pressure = states[0]['pressure']
    moved = np.zeros(pressure.size, dtype=bool)
    for k in range(1, PASSES + 1):
        moved |= np.abs(states[k]['temperature'] - states[k - 1]['temperature']) > 0.0
    per_window: dict[str, dict[str, list[float]]] = {name: {} for name in WINDOWS}
    for k in range(1, PASSES + 1):
        before, after = states[k - 1], states[k]
        written_before = _written(before['temperature'], pressure)
        written_after = _written(after['temperature'], pressure)
        change = written_after - written_before          # target - read on correctable layers
        target = change + before['read']
        excess = target - before['adiabatic']
        direct = _integrate(before['temperature'], pressure, target, moved)
        log_t = np.log(before['temperature'])
        for name, (lo, hi) in WINDOWS.items():
            mode = _amp(log_t, lo, hi)
            values = {
                'amp_ln_t': mode,
                'read_over_written': _amp(before['read'], lo, hi) / _amp(written_before, lo, hi),
                'adiabatic_over_ln_t': _amp(before['adiabatic'], lo, hi) / mode,
                'source_adiabatic': _amp(before['adiabatic'], lo, hi),
                'source_excess': _amp(excess, lo, hi),
                'source_minus_read': -_amp(before['read'], lo, hi),
                'source_total': _amp(change, lo, hi),
                'growth_correct_written': _amp(np.log(after['temperature']) - log_t, lo, hi) / mode,
                'growth_direct_write': _amp(np.log(direct) - log_t, lo, hi) / mode,
                'half_dlnp': float(0.5 * np.mean(np.log(pressure[lo:hi] / pressure[lo - 1:hi - 1]))),
            }
            for key, value in values.items():
                per_window[name].setdefault(key, []).append(float(value))
    summary = {
        name: {key: float(np.median(series)) for key, series in values.items()}
        for name, values in per_window.items()
    }
    return {'capture': capture_dir.name, 'moved_layers': [int(i) for i in np.flatnonzero(moved)],
            'windows': summary, 'per_pass': per_window}


def main() -> int:
    rows = [analyse(path) for path in sorted(ROOT.iterdir()) if (path / 'stages.json').is_file()]
    (ROOT / 'alternating_mode.json').write_text(json.dumps({'captures': rows}, indent=2) + '\n')
    keys = ('amp_ln_t', 'read_over_written', 'adiabatic_over_ln_t', 'source_adiabatic', 'source_excess',
            'source_minus_read', 'source_total', 'growth_correct_written', 'growth_direct_write', 'half_dlnp')
    for row in rows:
        print(f"== {row['capture']}  moved L{row['moved_layers'][0]}-{row['moved_layers'][-1]}")
        print('   window  ' + ' '.join(f'{k[:13]:>13s}' for k in keys))
        for name, values in row['windows'].items():
            print(f'   {name}  ' + ' '.join(f'{values[k]:+13.2e}' for k in keys))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
