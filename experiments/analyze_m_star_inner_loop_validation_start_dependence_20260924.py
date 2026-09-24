"""Layer-by-layer comparison of the warm-start and reference-start candidate products.

Diagnostic for row 2 of the decision table in
``notes/m_star_inner_loop_validation_preregistration_20260924.md`` (E1, E2 hold,
E3 does not).  For every candidate-eligible node with a reference-start product:
where the two products differ (upper L0-19, middle L20-49, deep L50-79 of the
80-layer standard grid), and the solver's own stop metrics over the last four
iterations of each solve.  Writes ``start_dependence.json`` next to the arms.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments.reduced_state_emulator.m_star_bootstrap_v1 import _load_mt  # noqa: E402

ROOT = REPO / 'results/m_star_inner_loop_validation_20260924'
REGIONS = {'upper_L0_19': slice(0, 20), 'middle_L20_49': slice(20, 50), 'deep_L50_79': slice(50, 80)}


def _local(path: str) -> Path:
    marker = ROOT.name + '/'
    return ROOT / path.split(marker, 1)[1] if marker in path else Path(path)


def _stop_history(iterations_dir: Path, count: int = 4) -> list[dict]:
    history = []
    for path in sorted(glob.glob(str(iterations_dir / '*.npz')))[-count:]:
        with np.load(path, allow_pickle=True) as data:
            history.append({
                'iteration': int(data['iteration']),
                'all_layer_relative_temperature_change': float(data['timing_all_layer_relative_temperature_change']),
                'p95_absolute_flux_error_percent': float(data['timing_p95_absolute_flux_error_percent']),
            })
    return history


def main() -> int:
    out = {}
    for case_dir in sorted((ROOT / 'candidate' / 'cases').iterdir()):
        case = json.loads((case_dir / 'case.json').read_text())
        independence_path = case_dir / 'start_independence.json'
        if not case.get('eligible') or not independence_path.is_file():
            continue
        reference_start = json.loads((case_dir / 'reference_start.json').read_text())
        primary_m, primary_t = _load_mt(_local(case['primary']['product_path']))
        start_m, start_t = _load_mt(_local(reference_start['solve']['product_path']))
        with np.load(sorted(glob.glob(str(case_dir / 'iterations' / 'primary' / '*.npz')))[-1]) as data:
            log_tau = np.asarray(data['log_tau_standard'], dtype=np.float64)
        temperature = np.abs(start_t / primary_t - 1.0)
        mass = np.abs(np.log10(start_m) - np.log10(primary_m))
        out[case_dir.name] = {
            'held_out': case['held_out'],
            'start_independent': json.loads(independence_path.read_text())['start_independent'],
            'temperature_relative_max_by_region': {k: float(temperature[s].max()) for k, s in REGIONS.items()},
            'column_mass_dex_max_by_region': {k: float(mass[s].max()) for k, s in REGIONS.items()},
            'temperature_max_layer': int(np.argmax(temperature)),
            'temperature_max_log_tau': float(log_tau[np.argmax(temperature)]),
            'column_mass_max_layer': int(np.argmax(mass)),
            'primary_last_iterations': _stop_history(case_dir / 'iterations' / 'primary'),
            'reference_start_last_iterations': _stop_history(case_dir / 'iterations' / 'reference_start'),
        }
    (ROOT / 'start_dependence.json').write_text(json.dumps(out, indent=2) + '\n')
    for node, row in out.items():
        t = row['temperature_relative_max_by_region']
        m = row['column_mass_dex_max_by_region']
        print(f"{node[:10]}{node[-6:]} indep={row['start_independent']!s:5s} "
              f"T upper/mid/deep {t['upper_L0_19']:.1e}/{t['middle_L20_49']:.1e}/{t['deep_L50_79']:.1e} "
              f"(max L{row['temperature_max_layer']}, log tau {row['temperature_max_log_tau']:.2f}) "
              f"m upper/mid/deep {m['upper_L0_19']:.1e}/{m['middle_L20_49']:.1e}/{m['deep_L50_79']:.1e}")
        for tag in ('primary', 'reference_start'):
            steps = ' '.join(f"{h['all_layer_relative_temperature_change']:.1e}/{h['p95_absolute_flux_error_percent']:.2f}"
                             for h in row[f'{tag}_last_iterations'])
            print(f'    {tag:15s} last dT/p95: {steps}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
