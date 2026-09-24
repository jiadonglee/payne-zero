"""Readout of the candidate plus written-gradient filter rerun (development and validation points).

Applies F1-F3 of ``notes/m_star_inner_loop_filter_rerun_preregistration_20260924.md``
to ``results/m_star_inner_loop_filter_rerun_20260924/{dev,validation}/candidate_filter``
against the candidate and S0-pchip records of the development sweep and the
validation.  Diagnostics: grid-bottom flux errors of each primary's last
iteration, the alternating amplitude of ln T in each primary product, primary
iterations and flux p95.  Writes ``summary.json`` in the rerun directory.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_filter_rerun_20260924'
DEV_CONTROL = REPO / 'results/m_star_inner_loop_dev_sweep_20260923'
VALIDATION_CONTROL = REPO / 'results/m_star_inner_loop_validation_20260924'
DEV_POINTS = {
    'A': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3500),
    'B': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3400),
    'C': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3300),
    'D': ('g+4.50_m-0.50_a+0.00_c+0.00_x1.00', 3600),
    'E': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3200),
}
WINDOWS = {'L45_51': (45, 52), 'L52_58': (52, 59), 'L59_65': (59, 66), 'L66_72': (66, 73), 'L73_78': (73, 79)}
RESULT_DIRS = ('m_star_inner_loop_filter_rerun_20260924', 'm_star_inner_loop_dev_sweep_20260923',
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


def _diagnostics(case: dict, iterations_dir: Path) -> dict:
    out: dict = {}
    files = sorted(glob.glob(str(iterations_dir / '*.npz')))
    if files:
        with np.load(files[-1], allow_pickle=True) as data:
            error = np.asarray(data['flux_error_percent'], dtype=np.float64)
        out['bottom_flux_error_percent_L74_79'] = [float(v) for v in error[74:80]]
        out['max_abs_flux_error_percent_L0_73'] = float(np.max(np.abs(error[:74])))
    product = _local((case.get('primary') or {}).get('product_path'))
    if product is not None and product.is_file():
        with np.load(product, allow_pickle=False) as data:
            log_t = np.log(np.asarray(data['temperature'], dtype=np.float64))
        out['alternating_amplitude_ln_t'] = {name: _alternating(log_t, lo, hi) for name, (lo, hi) in WINDOWS.items()}
    return out


def _row(case: dict | None, iterations_dir: Path, extra: dict | None = None) -> dict:
    if case is None:
        return {'available': False}
    primary = case.get('primary') or {}
    return {
        'available': True,
        'eligible': bool(case.get('eligible')),
        'primary_iterations': primary.get('iterations'),
        'primary_survives': primary.get('survives_solver'),
        'primary_p95': _p95(case.get('primary_flux_gate')),
        'restart_p95': _p95(case.get('restart_flux_gate')),
        'path_consistency': (case.get('path_consistency') or {}).get('passes'),
        **(extra or {}),
        **_diagnostics(case, iterations_dir),
    }


def _dev(root: Path, arm: str, point: str) -> dict:
    slug, teff = DEV_POINTS[point]
    case_root = root / arm / 'cases' / 'dwarf' / slug / f't{teff:04d}'
    return _row(_load(case_root / 'case.json'), case_root / 'iterations' / 'primary')


def _validation(root: Path, arm: str, node_id: str) -> dict:
    case_root = root / arm / 'cases' / node_id
    independence = _load(case_root / 'start_independence.json')
    case = _load(case_root / 'case.json')
    extra = {
        'start_independent': None if independence is None else bool(independence.get('start_independent')),
        'versus_reference_tio_max': max(((case or {}).get('versus_reference', {}).get('tio') or {})
                                        .get('metrics', {'-': float('nan')}).values()) if case else None,
    }
    return _row(case, case_root / 'iterations' / 'primary', extra)


def main() -> int:
    with np.load(VALIDATION_CONTROL / 'inputs' / 'warm_starts.npz', allow_pickle=False) as data:
        nodes = [(str(n), bool(h)) for n, h in zip(data['node_ids'], data['held_out'])]
    dev = {point: {'candidate_filter': _dev(ROOT / 'dev', 'candidate_filter', point),
                   'candidate': _dev(DEV_CONTROL, 'candidate', point),
                   's0': _dev(DEV_CONTROL, 's0', point)} for point in DEV_POINTS}
    validation = {node: {'candidate_filter': _validation(ROOT / 'validation', 'candidate_filter', node),
                         'candidate': _validation(VALIDATION_CONTROL, 'candidate', node),
                         's0': _validation(VALIDATION_CONTROL, 's0', node)} for node, _ in nodes}
    held_out = [node for node, flag in nodes if flag]

    def eligible(table, arm, keys):
        return {k for k in keys if table[k][arm].get('eligible')}

    dev_c, dev_f, dev_s = (eligible(dev, arm, DEV_POINTS) for arm in ('candidate', 'candidate_filter', 's0'))
    val_c, val_f, val_s = (eligible(validation, arm, held_out) for arm in ('candidate', 'candidate_filter', 's0'))
    complete = (all(dev[p]['candidate_filter']['available'] for p in dev)
                and all(validation[n]['candidate_filter']['available'] for n in validation)
                and all(validation[n]['candidate_filter'].get('start_independent') is not None for n in val_f))
    criteria = {
        'complete': complete,
        'dev_candidate_eligible': sorted(dev_c), 'dev_candidate_filter_eligible': sorted(dev_f),
        'validation_candidate_eligible': sorted(val_c), 'validation_candidate_filter_eligible': sorted(val_f),
        'validation_candidate_filter_start_independent': sorted(
            n for n in val_f if validation[n]['candidate_filter'].get('start_independent')),
        'F1': bool(dev_c <= dev_f),
        'F2': bool(val_c <= val_f),
        'F3': bool(all(validation[n]['candidate_filter'].get('start_independent') for n in val_f)) if val_f else None,
        'dev_E1_vs_s0': bool(dev_s <= dev_f), 'dev_E2_vs_s0': bool(dev_f - dev_s),
        'validation_E1_vs_s0': bool(val_s <= val_f), 'validation_E2_vs_s0': bool(val_f - val_s),
    }
    (ROOT / 'summary.json').write_text(json.dumps(
        {'criteria': criteria, 'dev': dev, 'validation': validation}, indent=2, default=str) + '\n')
    print(json.dumps(criteria, indent=2))

    def fmt(value, spec):
        return '-' if value is None else format(value, spec)

    def line(label, arm, r):
        if not r.get('available'):
            return f'{label:22s} {arm:17s} (not available)'
        bottom = r.get('bottom_flux_error_percent_L74_79')
        amp = r.get('alternating_amplitude_ln_t') or {}
        return (f"{label:22s} {arm:17s} elig={r['eligible']!s:5s} it={fmt(r['primary_iterations'], 'd'):>3s} "
                f"p95={fmt(r['primary_p95'], '.3f'):>7s} indep={r.get('start_independent')!s:5s} "
                f"maxL0-73={fmt(r.get('max_abs_flux_error_percent_L0_73'), '.2f'):>6s} "
                f"L74-79={' '.join(f'{v:+.1f}' for v in bottom) if bottom else '-'} "
                f"maxalt={max(abs(v) for v in amp.values()) if amp else float('nan'):.1e}")

    for point in DEV_POINTS:
        for arm in ('candidate_filter', 'candidate', 's0'):
            print(line(f'dev {point}', arm, dev[point][arm]))
    for node, flag in nodes:
        label = node.split('_')[0] + node.split('_')[1] + '_' + node.split('_')[-1] + ('' if flag else '*')
        for arm in ('candidate_filter', 'candidate', 's0'):
            print(line(label, arm, validation[node][arm]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
