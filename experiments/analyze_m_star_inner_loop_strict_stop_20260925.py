"""Readout of the candidate rerun with a three-iteration stop (development and validation points).

Applies G1 and G2 of ``notes/m_star_inner_loop_strict_stop_preregistration_20260925.md``
to ``results/m_star_inner_loop_strict_stop_20260925/{dev,validation}/candidate_strict_stop``
against the candidate and S0-pchip records of the development sweep and the
validation.  g4.75 3850 K is reported apart from the criteria.  Diagnostics:
iterations added over the candidate, the last three all-layer changes and p95
of each primary, grid-bottom flux errors and the alternating amplitude of ln T.
Writes ``summary.json`` in the rerun directory.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_strict_stop_20260925'
DEV_CONTROL = REPO / 'results/m_star_inner_loop_dev_sweep_20260923'
VALIDATION_CONTROL = REPO / 'results/m_star_inner_loop_validation_20260924'
ARM = 'candidate_strict_stop'
SEPARATE = 'g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3850'
DEV_POINTS = {
    'A': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3500),
    'B': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3400),
    'C': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3300),
    'D': ('g+4.50_m-0.50_a+0.00_c+0.00_x1.00', 3600),
    'E': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3200),
}
WINDOWS = {'L45_51': (45, 52), 'L52_58': (52, 59), 'L59_65': (59, 66), 'L66_72': (66, 73), 'L73_78': (73, 79)}
RESULT_DIRS = ('m_star_inner_loop_strict_stop_20260925', 'm_star_inner_loop_dev_sweep_20260923',
               'm_star_inner_loop_validation_20260924')


def _local(path: str | None) -> Path | None:
    if not path:
        return None
    for name in RESULT_DIRS:
        marker = name + '/'
        if marker in path:
            return REPO / 'results' / name / path.split(marker, 1)[1]
    return Path(path)


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _alternating(values: np.ndarray, lo: int, hi: int) -> float:
    index = np.arange(lo, hi)
    residual = values[index] - 0.5 * (values[index - 1] + values[index + 1])
    return float(np.mean(residual * (-1.0) ** index) / 2.0)


def _p95(gate: dict | None):
    entry = (gate or {}).get('metrics', {}).get('p95_absolute_flux_error_percent')
    return None if entry is None else entry.get('value')


def _row(case: dict | None, iterations_dir: Path, start_independent=None) -> dict:
    if case is None:
        return {'available': False}
    primary = case.get('primary') or {}
    out = {
        'available': True,
        'eligible': bool(case.get('eligible')),
        'primary_iterations': primary.get('iterations'),
        'primary_survives': primary.get('survives_solver'),
        'restart_iterations': (case.get('restart') or {}).get('iterations'),
        'primary_p95': _p95(case.get('primary_flux_gate')),
        'restart_p95': _p95(case.get('restart_flux_gate')),
        'path_consistency': (case.get('path_consistency') or {}).get('passes'),
        'path_temperature_p95': ((case.get('path_consistency') or {}).get('temperature_relative') or {}).get('p95'),
        'tio_dual_path': (case.get('tio_dual_path') or {}).get('passes'),
        'start_independent': start_independent,
    }
    files = sorted(glob.glob(str(iterations_dir / '*.npz')))
    if files:
        rows = [np.load(path, allow_pickle=True) for path in files[-3:]]
        out['last_all_layer_changes'] = [float(r['timing_all_layer_relative_temperature_change']) for r in rows]
        out['last_p95'] = [float(r['timing_p95_absolute_flux_error_percent']) for r in rows]
        error = np.asarray(rows[-1]['flux_error_percent'], dtype=np.float64)
        out['bottom_flux_error_percent_L74_79'] = [float(v) for v in error[74:80]]
        out['max_abs_flux_error_percent_L0_73'] = float(np.max(np.abs(error[:74])))
    product = _local(primary.get('product_path'))
    if product is not None and product.is_file():
        with np.load(product, allow_pickle=False) as data:
            log_t = np.log(np.asarray(data['temperature'], dtype=np.float64))
        out['alternating_amplitude_ln_t'] = {name: _alternating(log_t, lo, hi) for name, (lo, hi) in WINDOWS.items()}
    return out


def _dev(root: Path, arm: str, point: str) -> dict:
    slug, teff = DEV_POINTS[point]
    case_root = root / arm / 'cases' / 'dwarf' / slug / f't{teff:04d}'
    return _row(_load(case_root / 'case.json'), case_root / 'iterations' / 'primary')


def _validation(root: Path, arm: str, node_id: str) -> dict:
    case_root = root / arm / 'cases' / node_id
    independence = _load(case_root / 'start_independence.json')
    return _row(_load(case_root / 'case.json'), case_root / 'iterations' / 'primary',
                None if independence is None else bool(independence.get('start_independent')))


def main() -> int:
    with np.load(VALIDATION_CONTROL / 'inputs' / 'warm_starts.npz', allow_pickle=False) as data:
        nodes = [(str(n), bool(h)) for n, h in zip(data['node_ids'], data['held_out'])]
    dev = {p: {ARM: _dev(ROOT / 'dev', ARM, p), 'candidate': _dev(DEV_CONTROL, 'candidate', p),
               's0': _dev(DEV_CONTROL, 's0', p)} for p in DEV_POINTS}
    validation = {n: {ARM: _validation(ROOT / 'validation', ARM, n),
                      'candidate': _validation(VALIDATION_CONTROL, 'candidate', n),
                      's0': _validation(VALIDATION_CONTROL, 's0', n)} for n, _ in nodes}
    judged = [n for n, held_out in nodes if held_out and n != SEPARATE]

    def eligible(table, arm, keys):
        return {k for k in keys if table[k][arm].get('eligible')}

    dev_before, dev_after = eligible(dev, 'candidate', DEV_POINTS), eligible(dev, ARM, DEV_POINTS)
    val_before, val_after = eligible(validation, 'candidate', judged), eligible(validation, ARM, judged)
    criteria = {
        'complete': bool(all(dev[p][ARM]['available'] for p in dev)
                         and all(validation[n][ARM]['available'] for n in validation)
                         and all(validation[n][ARM].get('start_independent') is not None for n in val_after)),
        'dev_candidate_eligible': sorted(dev_before), 'dev_strict_eligible': sorted(dev_after),
        'validation_candidate_eligible_judged': sorted(val_before), 'validation_strict_eligible_judged': sorted(val_after),
        'validation_strict_start_independent_judged': sorted(
            n for n in val_after if validation[n][ARM].get('start_independent')),
        'G1': bool(dev_before <= dev_after and val_before <= val_after),
        'G2': bool(all(validation[n][ARM].get('start_independent') for n in val_after)) if val_after else None,
        'separate_3850K': {k: validation[SEPARATE][ARM].get(k) for k in
                           ('eligible', 'primary_iterations', 'primary_p95', 'start_independent')},
    }
    (ROOT / 'summary.json').write_text(json.dumps(
        {'criteria': criteria, 'dev': dev, 'validation': validation}, indent=2, default=str) + '\n')
    print(json.dumps(criteria, indent=2))

    def fmt(value, spec):
        return '-' if value is None else format(value, spec)

    def line(label, arm, r):
        if not r.get('available'):
            return f'{label:20s} {arm:21s} (not available)'
        bottom = r.get('bottom_flux_error_percent_L74_79')
        amp = r.get('alternating_amplitude_ln_t') or {}
        last = r.get('last_all_layer_changes') or []
        return (f"{label:20s} {arm:21s} elig={r['eligible']!s:5s} it={fmt(r['primary_iterations'], 'd'):>3s}"
                f"/{fmt(r['restart_iterations'], 'd'):>2s} p95={fmt(r['primary_p95'], '.3f'):>7s} "
                f"pathT={fmt(r['path_temperature_p95'], '.1e'):>7s} indep={r.get('start_independent')!s:5s} "
                f"last3={' '.join(f'{v:.1e}' for v in last)} "
                f"L79={fmt(bottom[-1] if bottom else None, '+.1f')} "
                f"alt={max(abs(v) for v in amp.values()) if amp else float('nan'):.1e}")

    for point in DEV_POINTS:
        for arm in (ARM, 'candidate'):
            print(line(f'dev {point}', arm, dev[point][arm]))
    for node, held_out in nodes:
        label = node.split('_')[0] + node.split('_')[1] + '_' + node.split('_')[-1] + ('' if held_out else '*')
        for arm in (ARM, 'candidate'):
            print(line(label, arm, validation[node][arm]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
