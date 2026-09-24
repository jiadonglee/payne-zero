"""Readout of the inner-loop validation on the safezone v2 dwarf points.

Reads ``results/m_star_inner_loop_validation_20260924/{candidate,s0}/cases`` and
applies E1/E2/E3 of ``notes/m_star_inner_loop_validation_preregistration_20260924.md``
on the held-out nodes.  Diagnostics: the comparison with the reference product
(beside the reference's own flux p95), S0 start independence, the cross-arm
TiO difference where both arms are eligible, and convergence within 30
iterations.  Writes ``summary.json`` next to the arms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'results/m_star_inner_loop_validation_20260924'
ARMS = ('candidate', 's0')
TIO_GATE = 5.0e-3
HISTORICAL_S0_SAFEZONE_V2 = {
    'protocol': 'linear EOS, cap 30, deep-layer relative temperature change <= 5e-4',
    'converged': {
        'g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t4000': {'iterations': 5, 'tio_vs_reference_max': 7.32e-3},
        'g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3950': {'iterations': 9, 'tio_vs_reference_max': 1.106e-2},
        'g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3900': {'iterations': 7, 'tio_vs_reference_max': 7.73e-3},
    },
}


def _nodes() -> list[dict]:
    with np.load(ROOT / 'inputs' / 'warm_starts.npz', allow_pickle=False) as data:
        return [
            {
                'node_id': str(node),
                'held_out': bool(data['held_out'][index]),
                'reference_flux_p95_percent': float(data['reference_flux_p95_percent'][index]),
            }
            for index, node in enumerate(data['node_ids'])
        ]


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _metric(gate: dict | None, name: str):
    entry = (gate or {}).get('metrics', {}).get(name)
    return None if entry is None else entry.get('value')


def _tio_max(tio: dict | None):
    metrics = (tio or {}).get('metrics')
    return max(metrics.values()) if metrics else None


def _local(path: str | None) -> Path | None:
    """Map a path recorded on Garching onto the synced local results tree."""

    if not path:
        return None
    marker = ROOT.name + '/'
    text = str(path)
    return ROOT / text.split(marker, 1)[1] if marker in text else Path(text)


def _row(arm: str, node_id: str) -> dict:
    case_root = ROOT / arm / 'cases' / node_id
    case = _load(case_root / 'case.json')
    if case is None:
        return {'available': False}
    reference_start = _load(case_root / 'reference_start.json') or {}
    independence = _load(case_root / 'start_independence.json')
    primary = case.get('primary') or {}
    restart = case.get('restart') or {}
    versus_reference = case.get('versus_reference') or {}
    iterations = primary.get('iterations')
    return {
        'available': True,
        'eligible': bool(case.get('eligible')),
        'primary_survives': primary.get('survives_solver'),
        'primary_iterations': iterations,
        'converged_within_30': bool(primary.get('survives_solver') and iterations is not None and iterations <= 30),
        'primary_flux_gate': case['primary_flux_gate'].get('passes'),
        'primary_p95': _metric(case['primary_flux_gate'], 'p95_absolute_flux_error_percent'),
        'primary_max': _metric(case['primary_flux_gate'], 'maximum_absolute_flux_error_percent'),
        'restart_iterations': restart.get('iterations'),
        'restart_flux_gate': case['restart_flux_gate'].get('passes'),
        'restart_p95': _metric(case['restart_flux_gate'], 'p95_absolute_flux_error_percent'),
        'path_consistency': case['path_consistency'].get('passes'),
        'tio_dual_path_max': _tio_max(case['tio_dual_path']),
        'tio_dual_path': case['tio_dual_path'].get('passes'),
        'reference_start_iterations': (reference_start.get('solve') or {}).get('iterations'),
        'reference_start_p95': _metric(reference_start.get('flux_gate'), 'p95_absolute_flux_error_percent'),
        'start_independence': independence,
        'start_independent': None if independence is None else bool(independence.get('start_independent')),
        'versus_reference_temperature_p95': ((versus_reference.get('path_consistency') or {})
                                             .get('temperature_relative') or {}).get('p95'),
        'versus_reference_mass_p95_dex': ((versus_reference.get('path_consistency') or {})
                                          .get('column_mass_dex') or {}).get('p95'),
        'versus_reference_tio_max': _tio_max(versus_reference.get('tio')),
        'primary_spectrum': (case['tio_dual_path'].get('spectra') or {}).get('base'),
    }


def _cross_arm_tio(candidate_path: str, s0_path: str) -> dict:
    with np.load(_local(candidate_path), allow_pickle=False) as data:
        candidate = {key: np.asarray(data[key], dtype=np.float64) for key in data.files}
    with np.load(_local(s0_path), allow_pickle=False) as data:
        s0 = {key: np.asarray(data[key], dtype=np.float64) for key in data.files}
    metrics = {
        'normalized_flux': float(np.max(np.abs(candidate['normalized_flux'] - s0['normalized_flux']))),
        'flux_total': float(np.max(
            np.abs(candidate['flux_total'] - s0['flux_total']) / np.maximum(s0['flux_continuum'], 1.0e-300)
        )),
        'flux_continuum': float(np.max(
            np.abs(candidate['flux_continuum'] - s0['flux_continuum'])
            / np.maximum(np.abs(s0['flux_continuum']), 1.0e-300)
        )),
    }
    return {'metrics': metrics, 'within_gate': bool(max(metrics.values()) <= TIO_GATE)}


def main() -> int:
    nodes = _nodes()
    table = {node['node_id']: {arm: _row(arm, node['node_id']) for arm in ARMS} for node in nodes}
    held_out = [node['node_id'] for node in nodes if node['held_out']]
    complete = all(table[n][arm]['available'] and table[n][arm]['start_independence'] is not None
                   for n in table for arm in ARMS)
    s0_eligible = {n for n in held_out if table[n]['s0'].get('eligible')}
    candidate_eligible = {n for n in held_out if table[n]['candidate'].get('eligible')}
    s0_ineligible = set(held_out) - s0_eligible
    criteria = {
        'complete': complete,
        'held_out': held_out,
        's0_pchip_eligible': sorted(s0_eligible),
        'candidate_eligible': sorted(candidate_eligible),
        'candidate_start_independent': sorted(n for n in candidate_eligible
                                              if table[n]['candidate'].get('start_independent')),
        'E1_applies': bool(s0_eligible),
        'E1': bool(s0_eligible <= candidate_eligible) if s0_eligible else None,
        'E2_applies': bool(s0_ineligible),
        'E2': bool(candidate_eligible & s0_ineligible) if s0_ineligible else None,
        'E3_applies': bool(candidate_eligible),
        'E3': (all(table[n]['candidate'].get('start_independent') for n in candidate_eligible)
               if candidate_eligible else None),
    }
    cross = {}
    for node_id in table:
        rows = table[node_id]
        if rows['candidate'].get('eligible') and rows['s0'].get('eligible'):
            paths = rows['candidate']['primary_spectrum'], rows['s0']['primary_spectrum']
            if all(paths) and all(_local(p).is_file() for p in paths):
                cross[node_id] = _cross_arm_tio(*paths)
    payload = {
        'criteria': criteria,
        'reference_flux_p95_percent': {node['node_id']: node['reference_flux_p95_percent'] for node in nodes},
        'points': table,
        'cross_arm_tio_where_both_eligible': cross,
        'historical_s0_safezone_v2': HISTORICAL_S0_SAFEZONE_V2,
    }
    (ROOT / 'summary.json').write_text(json.dumps(payload, indent=2, default=str) + '\n')
    print(json.dumps(criteria, indent=2))

    def fmt(value, spec):
        return '-' if value is None else format(value, spec)

    print(f"{'node':14s} {'arm':9s} {'elig':5s} {'it':>3s} {'p95%':>7s} {'rs_p95':>7s} {'path':5s} "
          f"{'tio2':>8s} {'ref_it':>6s} {'ref_p95':>7s} {'indep':5s} {'vsRefT':>8s} {'vsRefM':>8s} "
          f"{'vsRefTiO':>8s} refp95")
    for node in nodes:
        node_id = node['node_id']
        short = node_id.split('_')[0] + node_id.split('_')[1] + '_' + node_id.split('_')[-1]
        for arm in ARMS:
            r = table[node_id][arm]
            if not r.get('available'):
                print(f'{short:14s} {arm:9s} (not available)')
                continue
            print(f"{short:14s} {arm:9s} {str(r['eligible']):5s} {fmt(r['primary_iterations'], 'd'):>3s} "
                  f"{fmt(r['primary_p95'], '.3f'):>7s} {fmt(r['restart_p95'], '.3f'):>7s} "
                  f"{str(r['path_consistency']):5s} {fmt(r['tio_dual_path_max'], '.2e'):>8s} "
                  f"{fmt(r['reference_start_iterations'], 'd'):>6s} {fmt(r['reference_start_p95'], '.3f'):>7s} "
                  f"{str(r['start_independent']):5s} {fmt(r['versus_reference_temperature_p95'], '.2e'):>8s} "
                  f"{fmt(r['versus_reference_mass_p95_dex'], '.2e'):>8s} "
                  f"{fmt(r['versus_reference_tio_max'], '.2e'):>8s} {node['reference_flux_p95_percent']:.2f}")
    for node_id, entry in cross.items():
        print('cross-arm TiO', node_id, entry)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
