"""Replay stricter stop settings on the recorded candidate solves.

Motivation for ``notes/m_star_inner_loop_strict_stop_preregistration_20260925.md``.
Every recorded candidate primary (development sweep, validation) and
reference-start solve has per-iteration ``all_layer_relative_temperature_change``
and ``p95_absolute_flux_error_percent``.  The stop policy of
``payne_zero_atmosphere.convergence.evaluate_convergence_stop`` is replayed on
each sequence with ``required_consecutive_converged_iterations`` K = 1, 2, 3 and
``require_improving_flux_residual`` off and on (all-layer limit 5e-4, minimum 3
iterations).  A recorded solve ends at its original stop, so the replay tells
whether a setting would have stopped at the same iteration, earlier (never, by
construction) or not yet; the continuation records past the stop
(``results/m_star_inner_loop_validation_20260924/continuation``) show where the
stricter settings would fire on three nodes.  Writes ``stop_rule_replay.json``.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'results/m_star_inner_loop_validation_20260924/stop_rule_replay.json'
LIMIT = 5.0e-4
MINIMUM = 3
SETTINGS = [(k, improving) for improving in (False, True) for k in (1, 2, 3)]


def _sequence(iterations_dir: Path) -> tuple[list[float], list[float]]:
    changes, p95 = [], []
    for path in sorted(glob.glob(str(iterations_dir / '*.npz'))):
        with np.load(path, allow_pickle=True) as data:
            changes.append(float(data['timing_all_layer_relative_temperature_change']))
            p95.append(float(data['timing_p95_absolute_flux_error_percent']))
    return changes, p95


def first_stop(changes: list[float], p95: list[float], k: int, improving: bool) -> int | None:
    """1-based iteration at which the replayed policy stops, or None within the record."""

    consecutive = 0
    previous = None
    for index, (change, flux) in enumerate(zip(changes, p95), start=1):
        within = np.isfinite(change) and change < LIMIT
        if improving:
            within = within and (previous is None or not np.isfinite(previous) or flux <= previous)
        previous = flux
        consecutive = consecutive + 1 if (index >= MINIMUM and within) else 0
        if consecutive >= k:
            return index
    return None


def _runs() -> dict[str, Path]:
    runs = {}
    dev = REPO / 'results/m_star_inner_loop_dev_sweep_20260923/candidate/cases/dwarf'
    for case in sorted(dev.glob('*/t*')):
        runs[f'dev {case.parent.name[:10]} {case.name}'] = case / 'iterations' / 'primary'
    validation = REPO / 'results/m_star_inner_loop_validation_20260924/candidate/cases'
    for case in sorted(validation.iterdir()):
        for start in ('primary', 'reference_start'):
            runs[f'val {case.name[:10]}{case.name[-6:]} {start}'] = case / 'iterations' / start
    continuation = REPO / 'results/m_star_inner_loop_validation_20260924/continuation'
    for case in sorted(p for p in continuation.iterdir() if p.name.startswith('g')):
        for start in ('primary', 'reference_start'):
            runs[f'cont {case.name[:10]}{case.name[-6:]} {start}'] = case / start / 'iterations'
    return runs


def main() -> int:
    out = {}
    header = ' '.join(f"K{k}{'+imp' if imp else '    '}" for k, imp in SETTINGS)
    print(f"{'run':44s} {'n':>3s}  {header}")
    for name, directory in _runs().items():
        changes, p95 = _sequence(directory)
        if not changes:
            continue
        stops = {f"K{k}{'_improving' if imp else ''}": first_stop(changes, p95, k, imp) for k, imp in SETTINGS}
        out[name] = {'recorded_iterations': len(changes), 'stop_iteration': stops,
                     'last_all_layer_changes': changes[-4:], 'last_p95': p95[-4:]}
        cells = ' '.join(f"{'-' if v is None else v:>7}" for v in stops.values())
        print(f'{name:44s} {len(changes):>3d}  {cells}')
    OUT.write_text(json.dumps(out, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
