"""Stage decomposition of the candidate's non-stationary rounds (3600 K and 3850 K).

Reads ``results/m_star_inner_loop_nonstationary_capture_20260924/<capture>/``
(``capture.json``, ``stages.json``) written by
``experiments/reduced_state_emulator/m_star_inner_loop_nonstationary_capture_20260924.py``
and, per capture, splits the round's layer-wise temperature change into

* inner loop (relaxed): ``correction_native_grid`` minus its own correction
  delta, minus ``iteration_input`` (checked against ``post_inner_recompute``);
* global correction: the ``correction_native_grid`` delta;
* remap: ``standard_grid_remap`` minus ``correction_native_grid`` (index-wise,
  so it includes the regridding).

The three add up to the capture's net change.  A capture represents its
recorded round when max over L35-79 of |net_capture - net_recorded| is at most
0.3 of max over L35-79 of |net_recorded| (the rule in the driver).  Writes
``summary.json`` next to the captures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_nonstationary_capture_20260924'
WINDOW = slice(35, 80)
REPRESENTATIVE_FRACTION = 0.3
REPORT_LAYERS = {'L43': 43, 'L44': 44, 'L45': 45, 'L72': 72}


def _local(path: str) -> Path:
    marker = 'results/'
    return REPO / ('results/' + path.split(marker, 1)[1]) if marker in path else Path(path)


def _temperature(stage: dict) -> np.ndarray:
    return np.asarray(stage['structure']['temperature_K'], dtype=np.float64)


def _decompose(capture_dir: Path) -> dict:
    meta = json.loads((capture_dir / 'capture.json').read_text())
    stages = json.loads((capture_dir / 'stages.json').read_text())['stages']
    t_in = _temperature(stages['iteration_input'])
    t_exit = _temperature(stages['inner_loop_exit'])
    t_post_inner = _temperature(stages['post_inner_recompute'])
    t_corr = _temperature(stages['correction_native_grid'])
    t_remap = _temperature(stages['standard_grid_remap'])
    correction_delta = np.asarray(
        stages['correction_native_grid']['update_action']['delta_temperature_K'], dtype=np.float64)
    correction = correction_delta / t_in
    inner = (t_corr - correction_delta) / t_in - 1.0
    remap = (t_remap - t_corr) / t_in
    net = t_remap / t_in - 1.0
    chain_gap = float(np.max(np.abs(t_post_inner + correction_delta - t_corr) / t_in))
    with np.load(_local(meta['recorded_round']), allow_pickle=True) as data:
        recorded = np.asarray(data['temperature_post'], dtype=np.float64) / np.asarray(
            data['temperature_pre'], dtype=np.float64) - 1.0
        recorded_input = np.asarray(data['temperature_pre'], dtype=np.float64)
    scale = float(np.max(np.abs(recorded[WINDOW])))
    mismatch = float(np.max(np.abs(net[WINDOW] - recorded[WINDOW])))
    unrelaxed = t_exit - t_in
    moved = np.abs(unrelaxed) > 1.0e-6 * t_in
    held = (np.abs(t_corr - t_post_inner) == 0.0) & (np.abs(np.asarray(
        stages['correction_native_grid']['update_action']['raw_temperature_correction_K'])) > 0.0)
    peak = int(35 + np.argmax(np.abs(recorded[WINDOW])))

    def at(layer: int) -> dict:
        return {
            'inner': float(inner[layer]), 'correction': float(correction[layer]),
            'remap': float(remap[layer]), 'net_capture': float(net[layer]),
            'net_recorded': float(recorded[layer]),
        }

    return {
        'capture': meta['capture'],
        'run': meta['run'],
        'round': meta['round'],
        'input_matches_recorded_input_max_rel': float(np.max(np.abs(t_in / recorded_input - 1.0))),
        'chain_gap_max_rel': chain_gap,
        'representative': bool(mismatch <= REPRESENTATIVE_FRACTION * scale),
        'net_mismatch_over_recorded_max': mismatch / scale if scale > 0 else None,
        'recorded_max_abs_L35_79': scale,
        'recorded_peak_layer': peak,
        'relaxation_ratio_median': float(np.median((t_post_inner - t_in)[moved] / unrelaxed[moved]))
        if moved.any() else None,
        'inner_moved_layers': [int(i) for i in np.flatnonzero(moved)],
        'held_layers': [int(i) for i in np.flatnonzero(held)],
        'max_abs_by_stage_L35_79': {
            'inner': float(np.max(np.abs(inner[WINDOW]))),
            'correction': float(np.max(np.abs(correction[WINDOW]))),
            'remap': float(np.max(np.abs(remap[WINDOW]))),
        },
        'layers': {**{name: at(layer) for name, layer in REPORT_LAYERS.items()},
                   'recorded_peak': {'layer': peak, **at(peak)}},
        'profiles': {
            'inner': inner.tolist(), 'correction': correction.tolist(), 'remap': remap.tolist(),
            'net_capture': net.tolist(), 'net_recorded': recorded.tolist(),
        },
    }


def _range(layers: list[int]) -> str:
    return f'L{layers[0]}-{layers[-1]} ({len(layers)})' if layers else '-'


def main() -> int:
    rows = [
        _decompose(path) for path in sorted(ROOT.iterdir())
        if (path / 'stages.json').is_file() and (path / 'capture.json').is_file()
    ]
    (ROOT / 'summary.json').write_text(json.dumps({'captures': rows}, indent=2) + '\n')
    for row in rows:
        m = row['max_abs_by_stage_L35_79']
        print(f"{row['capture']:24s} repr={row['representative']!s:5s} "
              f"mismatch/rec={row['net_mismatch_over_recorded_max']:.2f} rec_max={row['recorded_max_abs_L35_79']:.1e} "
              f"peak L{row['recorded_peak_layer']} | max inner {m['inner']:.1e} corr {m['correction']:.1e} "
              f"remap {m['remap']:.1e} | relax {row['relaxation_ratio_median']} "
              f"moved {_range(row['inner_moved_layers'])} held {_range(row['held_layers'])} "
              f"in_vs_rec {row['input_matches_recorded_input_max_rel']:.1e}")
        for name, entry in row['layers'].items():
            print(f"    {name:13s} " + ' '.join(
                f"{key} {entry[key]:+.1e}" for key in ('inner', 'correction', 'remap', 'net_capture', 'net_recorded')
            ) + (f" (L{entry['layer']})" if 'layer' in entry else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
