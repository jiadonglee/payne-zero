"""Stage captures of the candidate's non-stationary rounds at g4.50 m+0.5 3600 K and g4.75 3850 K.

Diagnostic for ``notes/m_star_inner_loop_validation_closeout_20260924.md`` and
``notes/m_star_inner_loop_start_dependence_continuation_closeout_20260924.md``.
Each capture re-runs one recorded candidate round from that round's input
state with ``m_star_s3_stage_capture_20260922.run_capture_round`` (candidate
flags, pchip overlay) and records every stage: input, inner passes, loop exit,
post-inner recompute, global correction, remap.

Rounds (input of round k = post state of round k - 1 in the recorded run; round
1 of the 3600 K reference start is the reference product itself):

* 3600 K reference start, rounds 1-4: the candidate leaves S0's converged state.
* 3600 K warm start, rounds 59-60: the two phases of the period-2 cycle.
* 3850 K warm-start continuation, rounds 2, 4, 6, 8: the upward-moving change.
* 3850 K reference-start continuation, rounds 4, 8: near-stationary control.

A capture is a fresh first round: the correction's cross-round damping (upper
radiative layers L0-25 only) is absent.  A capture represents its recorded
round when its net layer-wise temperature change (remapped minus input) agrees
with the recorded round's to within 30% of the recorded change's maximum over
L35-79; only representative captures are used to attribute a change to a stage.

    python -m experiments.reduced_state_emulator.m_star_inner_loop_nonstationary_capture_20260924 \\
        --capture t3850_cont_primary_r04
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .m_star_inner_loop_dev_sweep_20260923 import ARMS, _write_json
from .m_star_inner_loop_validation_20260924 import DEFAULT_ROOT, _nodes, _track

REPO = Path(__file__).resolve().parents[2]
VALIDATION = DEFAULT_ROOT
OUT = REPO / 'results/m_star_inner_loop_nonstationary_capture_20260924'
N3600 = 'g+4.50_m+0.50_a+0.00_c+0.00_x1.00_t3600'
N3850 = 'g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3850'
RUNS = {
    't3600_ref': (N3600, VALIDATION / 'candidate/cases' / N3600 / 'iterations/reference_start'),
    't3600_warm': (N3600, VALIDATION / 'candidate/cases' / N3600 / 'iterations/primary'),
    't3850_cont_primary': (N3850, VALIDATION / 'continuation' / N3850 / 'primary/iterations'),
    't3850_cont_ref': (N3850, VALIDATION / 'continuation' / N3850 / 'reference_start/iterations'),
}
ROUNDS = {
    't3600_ref': (1, 2, 3, 4),
    't3600_warm': (59, 60),
    't3850_cont_primary': (2, 4, 6, 8),
    't3850_cont_ref': (4, 8),
}
CAPTURES = {f'{run}_r{k:02d}': (run, k) for run, rounds in ROUNDS.items() for k in rounds}
CANDIDATE_CAPTURE_FLAGS = {
    'correct_written_gradient': True,
    'refresh_state': True,
    'hold_correction': True,
    'relaxation': 0.5,
    'fill_holes': True,
}


def _iteration_record(run: str, k: int) -> Path:
    return RUNS[run][1] / f'iter_{k:04d}.npz'


def _round_input(run: str, k: int) -> tuple[np.ndarray, np.ndarray, str]:
    """(column_mass, temperature, source) entering round k of the recorded run."""

    if k == 1:
        node_id = RUNS[run][0]
        record = json.loads((VALIDATION / 'candidate/cases' / node_id / 'reference_start.json').read_text())
        source = record['reference_product']
        keys = ('column_mass', 'temperature')
    else:
        source = str(_iteration_record(run, k - 1))
        keys = ('column_mass_post', 'temperature_post')
    with np.load(source, allow_pickle=True) as data:
        return (np.asarray(data[keys[0]], dtype=np.float64),
                np.asarray(data[keys[1]], dtype=np.float64), source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--capture', choices=sorted(CAPTURES), required=True)
    parser.add_argument('--inputs', type=Path, default=VALIDATION / 'inputs' / 'warm_starts.npz')
    args = parser.parse_args(argv)

    expected = {
        'convection_zone_inner_loop_correct_written_gradient': CANDIDATE_CAPTURE_FLAGS['correct_written_gradient'],
        'convection_zone_inner_loop_refresh_state': CANDIDATE_CAPTURE_FLAGS['refresh_state'],
        'convection_zone_inner_loop_hold_correction': CANDIDATE_CAPTURE_FLAGS['hold_correction'],
        'convection_zone_inner_loop_relaxation': CANDIDATE_CAPTURE_FLAGS['relaxation'],
        'convection_zone_inner_loop_fill_holes': CANDIDATE_CAPTURE_FLAGS['fill_holes'],
        'convection_zone_inner_loop_passes': 8,
        'convection_zone_inner_loop_freeze_mask': True,
    }
    assert expected == ARMS['candidate'], (expected, ARMS['candidate'])

    run, k = CAPTURES[args.capture]
    run_root = OUT / args.capture
    run_root.mkdir(parents=True, exist_ok=True)

    # The pchip overlay must precede every physics import (``_track`` pulls in
    # the solver); run_capture_round reuses the same overlay and cache.
    for key, value in (
        ('NUMBA_THREADING_LAYER', 'workqueue'), ('NUMBA_NUM_THREADS', '1'),
        ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'),
        ('OPENBLAS_NUM_THREADS', '1'), ('VECLIB_MAXIMUM_THREADS', '1'),
        ('CUDA_VISIBLE_DEVICES', ''),
    ):
        os.environ[key] = value
    os.environ['PAYNE_ZERO_DATA_ROOT'] = str(REPO / 'source_data_files')
    sys.path.insert(0, str(REPO))
    from experiments.reduced_state_emulator.m_star_h2_paired_iteration_20260921 import build_overlay
    overlay_root = build_overlay('pchip', run_root)
    os.environ['NUMBA_CACHE_DIR'] = str(run_root / 'numba_cache')
    sys.path.insert(0, str(overlay_root))

    node_id = RUNS[run][0]
    node = _nodes(args.inputs)[node_id]
    labels = _track(node['labels']).labels(float(node['labels'][0]))
    mass, temperature, source = _round_input(run, k)
    start = run_root / 'start.npz'
    np.savez(start, column_mass=mass, temperature=temperature)
    _write_json(run_root / 'capture.json', {
        'capture': args.capture,
        'node_id': node_id,
        'run': run,
        'round': k,
        'input_source': source,
        'recorded_round': str(_iteration_record(run, k)),
        'flags': CANDIDATE_CAPTURE_FLAGS,
    })

    from . import m_star_s3_stage_capture_20260922 as capture

    return capture.run_capture_round(
        case=args.capture, start_npz=start, run_root=run_root,
        labels=labels, effective_temperature=float(node['labels'][0]),
        **CANDIDATE_CAPTURE_FLAGS,
    )


if __name__ == '__main__':
    raise SystemExit(main())
