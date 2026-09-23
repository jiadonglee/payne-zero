"""Full-physics re-evaluation of saved stage states (measure only, no chaining)."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = REPO / 'results/m_star_s3_stage_capture_20260922'
STAGES_BY_CASE = {
    'D': ('iteration_input', 'inner_pass_01', 'inner_pass_08', 'standard_grid_remap'),
    'A': ('iteration_input', 'inner_pass_08', 'standard_grid_remap'),
}

# Measurement protocol: every saved stage state is re-evaluated with the
# evaluate-state-only entry (inner loop off, opacity lagging off, one round,
# no correction accepted).  The first evaluation is the D control state under
# the pchip scheme and must reproduce the recorded control residual
# (deep-window max |R_smoothed| = 0.8081); a mismatch stops the run before
# any stage is judged.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'D'), required=True)
    parser.add_argument('--stages', nargs='+', default=None)
    parser.add_argument('--with-control-anchor', action='store_true')
    parser.add_argument('--capture-root', type=Path, default=CAPTURE_ROOT)
    args = parser.parse_args()

    for key, value in (
        ('NUMBA_THREADING_LAYER', 'workqueue'), ('NUMBA_NUM_THREADS', '1'),
        ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'),
        ('OPENBLAS_NUM_THREADS', '1'), ('VECLIB_MAXIMUM_THREADS', '1'),
        ('CUDA_VISIBLE_DEVICES', ''),
    ):
        os.environ[key] = value
    os.environ['PAYNE_ZERO_DATA_ROOT'] = str(REPO / 'source_data_files')

    root = args.capture_root / args.case.lower()
    sys.path.insert(0, str(REPO))
    from experiments.reduced_state_emulator.m_star_h2_paired_iteration_20260921 import (
        build_overlay,
    )
    overlay_root = build_overlay('pchip', root)
    os.environ['NUMBA_CACHE_DIR'] = str(root / 'numba_cache')
    sys.path.insert(0, str(overlay_root))

    from experiments.reduced_state_emulator.m_star_s3_stage_capture_20260922 import (
        evaluate_state_only,
    )
    from experiments.reduced_state_emulator.m_star_local_iteration_response_20260919 import (
        _load_case, _state_from_mt,
    )

    case = _load_case(args.case)
    labels = case['labels']
    teff = case['target_temperature_K']
    stages_path = root / 'stages.json'
    stages = json.loads(stages_path.read_text())['stages']

    def stage_state(name):
        structure = stages[name]['structure']

        def values(spec):
            if isinstance(spec, dict):
                value = spec.get('value')
                if isinstance(value, dict) and 'full_depth' in value:
                    return np.asarray(value['full_depth'], dtype=np.float64)
                if value is None:
                    return None
                return np.asarray(value, dtype=np.float64)
            return np.asarray(spec, dtype=np.float64)

        temperature = values(structure['temperature_K'])
        column_mass = values(structure['column_mass_g_cm2'])
        if column_mass is None:
            # inner passes hold the round-input column mass (documented in
            # the capture's gap_note); the snapshot does not carry it
            column_mass = values(
                stages['iteration_input']['structure']['column_mass_g_cm2'])
        return _state_from_mt(labels, column_mass, temperature)

    evaluations = []
    if args.with_control_anchor:
        control = (REPO / 'results/m_star_trial_comparison_20260920'
                   / f'controls_{args.case}/arrays/{args.case.lower()}_alpha_0p0.npz')
        with np.load(control, allow_pickle=False) as data:
            anchor_state = _state_from_mt(labels, data['column_mass'],
                                          data['temperature'])
        record = evaluate_state_only(anchor_state, labels, teff, root,
                                     f'{args.case.lower()}_reeval_control_anchor')
        with np.load(record['arrays_path'], allow_pickle=False) as data:
            anchor_smooth = float(np.max(np.abs(np.asarray(
                data['R_smoothed'], dtype=np.float64)[67:80])))
        ok = abs(anchor_smooth - 0.8081) < 0.001
        evaluations.append({'stage': 'control_anchor', 'deep_max_abs_R_smoothed':
                            anchor_smooth, 'matches_recorded_control': ok})
        print(f'control anchor deep_max_smooth={anchor_smooth:.4f} '
              f'matches={ok}', flush=True)
        if not ok:
            (root / 'reeval.json').write_text(json.dumps(
                {'evaluations': evaluations, 'aborted': 'control anchor mismatch'},
                indent=2) + '\n')
            return 1

    names = tuple(args.stages) if args.stages else STAGES_BY_CASE[args.case]
    for name in names:
        state = stage_state(name)
        record = evaluate_state_only(state, labels, teff, root,
                                     f'{args.case.lower()}_reeval_{name}')
        with np.load(record['arrays_path'], allow_pickle=False) as data:
            r_raw = np.abs(np.asarray(data['R_raw'], dtype=np.float64))
            r_smoothed = np.abs(np.asarray(data['R_smoothed'], dtype=np.float64))
            deep = slice(67, 80)
            frozen_raw = stages[name]['raw_total_flux_residual']['deep_mean_abs']
            row = {
                'stage': name,
                'full_physics_deep_mean_abs_R_raw': float(r_raw[deep].mean()),
                'full_physics_deep_max_abs_R_raw': float(r_raw[deep].max()),
                'full_physics_deep_max_abs_R_smoothed': float(r_smoothed[deep].max()),
                'frozen_field_deep_mean_abs_R_raw_from_capture': float(frozen_raw),
            }
        evaluations.append(row)
        print(f'{name}: full-physics mean|R_raw|='
              f'{row["full_physics_deep_mean_abs_R_raw"]:.4f} '
              f'frozen-field={frozen_raw:.4f}', flush=True)

    (root / 'reeval.json').write_text(json.dumps(
        {'case': args.case, 'evaluations': evaluations}, indent=2) + '\n')
    print('reeval complete', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
