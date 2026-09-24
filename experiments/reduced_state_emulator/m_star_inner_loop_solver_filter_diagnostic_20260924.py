"""Solver diagnostic: candidate plus the mask-interior written-gradient filter at g4.75 3850 K.

Plan: ``notes/m_star_inner_loop_solver_filter_diagnostic_plan_20260924.md``.

``--start {primary,reference_start}``
    continue one of the two candidate products for a fixed ten iterations,
    stop disabled, with ``convection_zone_inner_loop_filter_written_gradient``
    on; every iteration is recorded and the terminal product written.
``--pair``
    the per-iteration gap between the two continuations, per-iteration edge
    flux errors, and after ten iterations path consistency, TiO and the
    alternating amplitude of ln T in both terminal products.

    python -m experiments.reduced_state_emulator.m_star_inner_loop_solver_filter_diagnostic_20260924 \\
        --start primary --synthesis-root ...
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .m_star_inner_loop_dev_sweep_20260923 import ARMS, _file_sha256, _import_physics, _write_json
from .m_star_inner_loop_start_dependence_continuation_20260924 import (
    EXTRA_ITERATIONS,
    STARTS,
    _gap,
    _source_product,
    _trajectory,
)
from .m_star_inner_loop_validation_20260924 import DEFAULT_ROOT as VALIDATION, _guarded_tio, _nodes, _track

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / 'notes/m_star_inner_loop_solver_filter_diagnostic_plan_20260924.md'
OUT = REPO / 'results/m_star_inner_loop_solver_filter_diagnostic_20260924'
NODE = 'g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3850'
OVERRIDES = {**ARMS['candidate'], 'convection_zone_inner_loop_filter_written_gradient': True}
WINDOWS = {'L45_51': (45, 52), 'L52_58': (52, 59), 'L59_65': (59, 66), 'L66_72': (66, 73), 'L73_78': (73, 79)}
EDGE_LAYERS = (45, 77, 78, 79)
MASK = slice(45, 80)


def _alternating(values: np.ndarray, lo: int, hi: int) -> float:
    index = np.arange(lo, hi)
    residual = values[index] - 0.5 * (values[index - 1] + values[index + 1])
    return float(np.mean(residual * (-1.0) ** index) / 2.0)


def run_continue(args, node: dict, identity: dict) -> dict:
    from bench.run_reference import _solver_config
    from payne_zero_atmosphere.runner import run_atmosphere_model
    from payne_zero_atmosphere.synthesis_bridge import (
        infer_synthesis_source_catalog_root,
        save_product_structured_atmosphere,
    )

    from . import m_star_iteration_tomography_v1 as tomography
    from .cool_star_step_test import _clone_atmosphere, _reconstruct_from_mt
    from .m_star_bootstrap_v1 import _load_mt

    source = _source_product(VALIDATION / 'candidate' / 'cases' / NODE, args.start)
    out_root = OUT / args.start
    track = _track(node['labels'])
    labels = track.labels(float(node['labels'][0]))
    seed_m, seed_t = _load_mt(source)
    config = _solver_config(
        _clone_atmosphere(_reconstruct_from_mt(labels, seed_m, seed_t)),
        iterations_per_trial=EXTRA_ITERATIONS,
        structured_atmosphere_path=None,
        debug_state_path=None,
    )
    config = dataclasses.replace(config, enable_convergence_stop=False, **OVERRIDES)
    started = time.perf_counter()
    result = run_atmosphere_model(
        config, after_iteration_hook=tomography.make_tomography_hook(out_root / 'iterations'),
    )
    product = out_root / 'products' / f'{labels.slug}.npz'
    product.parent.mkdir(parents=True, exist_ok=True)
    save_product_structured_atmosphere(
        result.atmosphere, product,
        source_catalog_root=infer_synthesis_source_catalog_root(config.inputs.molecules_path),
        molecular_lines=bool(config.enable_molecules), device='cpu', dtype='float64',
    )
    row = {
        'node_id': NODE, 'start': args.start, 'source_product': source,
        'method': 'continuation_from_terminal_mt_stop_disabled',
        'extra_iterations': EXTRA_ITERATIONS,
        'iterations_completed': int(result.iterations_completed),
        'seconds': float(time.perf_counter() - started),
        'product_path': str(product),
        'identity': identity,
    }
    _write_json(out_root / 'continuation.json', row)
    print(f"[continue {args.start}] iterations={row['iterations_completed']} seconds={row['seconds']:.0f}", flush=True)
    return row


def _edge_history(iterations_dir: Path) -> list[dict]:
    import glob

    rows = []
    for path in sorted(glob.glob(str(iterations_dir / '*.npz'))):
        with np.load(path, allow_pickle=True) as data:
            error = np.abs(np.asarray(data['flux_error_percent'], dtype=np.float64))
            rows.append({
                'iteration': int(data['iteration']),
                'all_layer_relative_temperature_change': float(data['timing_all_layer_relative_temperature_change']),
                'p95_absolute_flux_error_percent': float(data['timing_p95_absolute_flux_error_percent']),
                'mask_median_abs_flux_error_percent': float(np.median(error[MASK])),
                **{f'abs_flux_error_percent_L{layer}': float(error[layer]) for layer in EDGE_LAYERS},
            })
    return rows


def run_pair(compare_spectra) -> dict:
    from .m_star_bootstrap_v1 import _load_mt, _product_consistency

    runs = {start: json.loads((OUT / start / 'continuation.json').read_text()) for start in STARTS}
    sources = {start: _load_mt(runs[start]['source_product']) for start in STARTS}
    trajectories = {start: _trajectory(OUT / start / 'iterations') for start in STARTS}
    gaps = [_gap(*sources['primary'], *sources['reference_start'])]
    for first, second in zip(trajectories['primary'], trajectories['reference_start']):
        gaps.append(_gap(first['column_mass'], first['temperature'], second['column_mass'], second['temperature']))
    products = {start: runs[start]['product_path'] for start in STARTS}
    consistency = _product_consistency(products['primary'], products['reference_start'])
    tio = _guarded_tio(
        compare_spectra,
        (products['primary'], OUT / 'spectra' / 'primary.npz'),
        (products['reference_start'], OUT / 'spectra' / 'reference_start.npz'),
    )
    history = {start: _edge_history(OUT / start / 'iterations') for start in STARTS}
    amplitudes = {}
    for start in STARTS:
        with np.load(products[start], allow_pickle=False) as data:
            log_t = np.log(np.asarray(data['temperature'], dtype=np.float64))
        amplitudes[start] = {name: _alternating(log_t, lo, hi) for name, (lo, hi) in WINDOWS.items()}
    reference = {
        'closes': bool(consistency.get('passes') and tio.get('passes')),
        'stationary': all(
            row['all_layer_relative_temperature_change'] <= 5.0e-4
            for start in STARTS for row in history[start][-3:]
        ),
        'no_alternating_mode': all(
            abs(value) <= 3.0e-4
            for start in STARTS for name, value in amplitudes[start].items() if name != 'L73_78'
        ),
    }
    payload = {
        'node_id': NODE,
        'overrides': OVERRIDES,
        'gap_by_iteration': gaps,
        'path_consistency': consistency,
        'tio': tio,
        'iteration_history': history,
        'terminal_alternating_amplitude_ln_t': amplitudes,
        'preregistered_stage2_lines_for_reference': reference,
    }
    _write_json(OUT / 'pair.json', payload)
    print(json.dumps({'reference': reference, 'gap0': gaps[0], 'gap10': gaps[-1]}, indent=2), flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--start', choices=STARTS)
    parser.add_argument('--pair', action='store_true')
    parser.add_argument('--inputs', type=Path, default=VALIDATION / 'inputs' / 'warm_starts.npz')
    parser.add_argument('--synthesis-root', type=Path, required=True,
                        help='directory containing the emulator_v1_2 package')
    args = parser.parse_args(argv)
    if args.pair == bool(args.start):
        parser.error('give exactly one of --start or --pair')

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
    job_root = OUT / 'jobs' / ('pair' if args.pair else args.start)
    overlay_root = build_overlay('pchip', job_root)
    os.environ['NUMBA_CACHE_DIR'] = str(job_root / 'numba_cache')
    compare_spectra, modules = _import_physics(overlay_root, Path(args.synthesis_root))

    if args.pair:
        run_pair(compare_spectra)
        return 0
    identity = {
        'plan': str(PLAN),
        'plan_sha256': _file_sha256(PLAN),
        'driver_sha256': _file_sha256(Path(__file__)),
        'modules': modules,
        'overlay': str(overlay_root),
        'config_overrides': OVERRIDES,
    }
    run_continue(args, _nodes(args.inputs)[NODE], identity)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
