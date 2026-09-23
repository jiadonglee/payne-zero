"""Readout of the inner-loop development-point sweep (candidate versus S0-pchip).

Reads ``results/m_star_inner_loop_dev_sweep_20260923/{candidate,s0}/cases`` and
applies E1/E2 of ``notes/m_star_inner_loop_dev_sweep_preregistration_20260923.md``.
The cross-arm TiO difference on points where both arms are eligible is a
diagnostic; it reuses the dual-path statistics on the stored primary spectra.
Writes ``summary.json`` next to the arms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_dev_sweep_20260923'
POINTS = {
    'A': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3500),
    'B': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3400),
    'C': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3300),
    'D': ('g+4.50_m-0.50_a+0.00_c+0.00_x1.00', 3600),
    'E': ('g+4.50_m+0.00_a+0.00_c+0.00_x1.00', 3200),
}
HISTORICAL_S0_LINEAR = {
    'A': {'eligible': True, 'iterations': 38, 'p95': 7.584},
    'B': {'eligible': False, 'iterations': 30, 'p95': 9.146},
    'C': {'eligible': True, 'iterations': 28, 'p95': 8.609},
    'D': {'eligible': False, 'iterations': 25, 'p95': 11.560},
    'E': {'eligible': False, 'iterations': None, 'p95': None},
}
TIO_GATE = 5.0e-3


def _case(arm: str, point: str) -> dict | None:
    slug, teff = POINTS[point]
    path = ROOT / arm / 'cases' / 'dwarf' / slug / f't{teff:04d}' / 'case.json'
    return json.loads(path.read_text()) if path.is_file() else None


def _metric(gate: dict, name: str):
    entry = (gate or {}).get('metrics', {}).get(name)
    return None if entry is None else entry.get('value')


def _row(case: dict | None) -> dict:
    if case is None:
        return {'available': False}
    primary = case.get('primary') or {}
    restart = case.get('restart') or {}
    return {
        'available': True,
        'eligible': bool(case.get('eligible')),
        'primary_iterations': primary.get('iterations'),
        'primary_survives': primary.get('survives_solver'),
        'primary_flux_gate': case['primary_flux_gate'].get('passes'),
        'primary_p95': _metric(case['primary_flux_gate'], 'p95_absolute_flux_error_percent'),
        'primary_max': _metric(case['primary_flux_gate'], 'maximum_absolute_flux_error_percent'),
        'primary_median': _metric(case['primary_flux_gate'], 'median_absolute_flux_error_percent'),
        'restart_iterations': restart.get('iterations'),
        'restart_flux_gate': case['restart_flux_gate'].get('passes'),
        'restart_p95': _metric(case['restart_flux_gate'], 'p95_absolute_flux_error_percent'),
        'path_consistency': case['path_consistency'].get('passes'),
        'path_temperature_p95': (case['path_consistency'].get('temperature_relative') or {}).get('p95'),
        'tio_passes': case['tio_dual_path'].get('passes'),
        'tio_metrics': case['tio_dual_path'].get('metrics'),
        'primary_spectrum': (case['tio_dual_path'].get('spectra') or {}).get('primary'),
    }


def _local(path: str) -> Path:
    """Map a path recorded on Garching onto the synced local results tree."""

    marker = ROOT.name + '/'
    text = str(path)
    return ROOT / text.split(marker, 1)[1] if marker in text else Path(text)


def _spectrum(path: str) -> dict:
    with np.load(_local(path), allow_pickle=False) as data:
        return {key: np.asarray(data[key], dtype=np.float64) for key in data.files}


def _cross_arm_tio(candidate_path: str, s0_path: str) -> dict:
    candidate = _spectrum(candidate_path)
    s0 = _spectrum(s0_path)
    metrics = {
        'normalized_flux': float(np.max(np.abs(candidate['normalized_flux'] - s0['normalized_flux']))),
        'flux_total': float(np.max(
            np.abs(candidate['flux_total'] - s0['flux_total'])
            / np.maximum(s0['flux_continuum'], 1.0e-300)
        )),
        'flux_continuum': float(np.max(
            np.abs(candidate['flux_continuum'] - s0['flux_continuum'])
            / np.maximum(np.abs(s0['flux_continuum']), 1.0e-300)
        )),
    }
    return {'metrics': metrics, 'within_gate': bool(max(metrics.values()) <= TIO_GATE)}


def main() -> int:
    table = {point: {'candidate': _row(_case('candidate', point)), 's0': _row(_case('s0', point))}
             for point in POINTS}
    complete = all(row['candidate']['available'] and row['s0']['available'] for row in table.values())
    s0_eligible = {p for p, row in table.items() if row['s0'].get('eligible')}
    candidate_eligible = {p for p, row in table.items() if row['candidate'].get('eligible')}
    s0_ineligible = set(POINTS) - s0_eligible
    criteria = {
        'complete': complete,
        's0_pchip_eligible': sorted(s0_eligible),
        'candidate_eligible': sorted(candidate_eligible),
        'E1_applies': bool(s0_eligible),
        'E1': bool(s0_eligible <= candidate_eligible) if s0_eligible else None,
        'E2_applies': bool(s0_ineligible),
        'E2': bool(candidate_eligible & s0_ineligible) if s0_ineligible else None,
    }
    cross = {}
    for point in sorted(s0_eligible & candidate_eligible):
        paths = table[point]['candidate']['primary_spectrum'], table[point]['s0']['primary_spectrum']
        if all(paths) and all(_local(p).is_file() for p in paths):
            cross[point] = _cross_arm_tio(*paths)
    payload = {
        'criteria': criteria,
        'points': table,
        'cross_arm_tio_where_both_eligible': cross,
        'historical_s0_linear': HISTORICAL_S0_LINEAR,
    }
    (ROOT / 'summary.json').write_text(json.dumps(payload, indent=2, default=str) + '\n')
    print(json.dumps(criteria, indent=2))
    header = f"{'pt':3s} {'arm':9s} {'elig':5s} {'iters':>5s} {'p95%':>7s} {'max%':>7s} {'med%':>6s} {'rst_it':>6s} {'rst_p95':>7s} {'path':5s} {'tio':5s} tio_max"
    print(header)
    for point, row in table.items():
        for arm in ('candidate', 's0'):
            r = row[arm]
            if not r.get('available'):
                print(f'{point:3s} {arm:9s} (not available)')
                continue
            tio_max = max(r['tio_metrics'].values()) if r.get('tio_metrics') else None
            fmt = lambda v, f: '-' if v is None else format(v, f)  # noqa: E731
            print(f"{point:3s} {arm:9s} {str(r['eligible']):5s} {fmt(r['primary_iterations'], 'd'):>5s} "
                  f"{fmt(r['primary_p95'], '.3f'):>7s} {fmt(r['primary_max'], '.3f'):>7s} "
                  f"{fmt(r['primary_median'], '.3f'):>6s} {fmt(r['restart_iterations'], 'd'):>6s} "
                  f"{fmt(r['restart_p95'], '.3f'):>7s} {str(r['path_consistency']):5s} "
                  f"{str(r['tio_passes']):5s} {fmt(tio_max, '.2e')}")
    for point, entry in cross.items():
        print('cross-arm TiO', point, entry)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
