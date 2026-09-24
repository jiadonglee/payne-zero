"""Start-dependence continuation of the inner-loop validation (three nodes, candidate arm).

Protocol: ``notes/m_star_inner_loop_start_dependence_continuation_preregistration_20260924.md``.

``--start {primary,reference_start}``
    continue one of the two candidate products of one node: reconstruct from
    its (m, T), iterate a fixed ten iterations with the stop disabled, record
    every iteration, and write the terminal product with the runner's writer.
``--pair``
    per node, the gap between the two continuations after every iteration and,
    after ten, path consistency and TiO between the two terminal products.

    python -m experiments.reduced_state_emulator.m_star_inner_loop_start_dependence_continuation_20260924 \\
        --start primary --only g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t4000 --synthesis-root ...
"""

from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .m_star_inner_loop_dev_sweep_20260923 import ARMS, _file_sha256, _import_physics, _write_json
from .m_star_inner_loop_validation_20260924 import DEFAULT_ROOT, _guarded_tio, _nodes, _track

REPO = Path(__file__).resolve().parents[2]
PREREGISTRATION = REPO / 'notes/m_star_inner_loop_start_dependence_continuation_preregistration_20260924.md'
NODES = (
    'g+4.50_m-1.00_a+0.00_c+0.00_x1.00_t3800',
    'g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3850',
    'g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t4000',
)
STARTS = ('primary', 'reference_start')
EXTRA_ITERATIONS = 10


def _source_product(case_root: Path, start: str) -> str:
    if start == 'primary':
        return json.loads((case_root / 'case.json').read_text())['primary']['product_path']
    return json.loads((case_root / 'reference_start.json').read_text())['solve']['product_path']


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

    case_root = Path(args.result_root) / 'candidate' / 'cases' / node['node_id']
    out_root = Path(args.result_root) / 'continuation' / node['node_id'] / args.start
    source = _source_product(case_root, args.start)
    track = _track(node['labels'])
    labels = track.labels(float(node['labels'][0]))
    seed_m, seed_t = _load_mt(source)
    seed = _reconstruct_from_mt(labels, seed_m, seed_t)
    config = _solver_config(
        _clone_atmosphere(seed),
        iterations_per_trial=EXTRA_ITERATIONS,
        structured_atmosphere_path=None,
        debug_state_path=None,
    )
    config = dataclasses.replace(config, enable_convergence_stop=False, **ARMS['candidate'])
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
        'node_id': node['node_id'],
        'start': args.start,
        'source_product': source,
        'method': 'continuation_from_terminal_mt_stop_disabled',
        'extra_iterations': EXTRA_ITERATIONS,
        'iterations_completed': int(result.iterations_completed),
        'seconds': float(time.perf_counter() - started),
        'product_path': str(product),
        'identity': identity,
    }
    _write_json(out_root / 'continuation.json', row)
    print(f"[continue {args.start}] {node['node_id']} iterations={row['iterations_completed']} "
          f"seconds={row['seconds']:.0f}", flush=True)
    return row


def _trajectory(iterations_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(iterations_dir / '*.npz'))):
        with np.load(path, allow_pickle=True) as data:
            rows.append({
                'temperature': np.asarray(data['temperature_post'], dtype=np.float64),
                'column_mass': np.asarray(data['column_mass_post'], dtype=np.float64),
                'all_layer_relative_temperature_change': float(data['timing_all_layer_relative_temperature_change']),
                'p95_absolute_flux_error_percent': float(data['timing_p95_absolute_flux_error_percent']),
            })
    return rows


def _gap(first_m, first_t, second_m, second_t) -> dict:
    return {
        'temperature_relative_p95': float(np.percentile(np.abs(second_t / first_t - 1.0), 95.0)),
        'column_mass_dex_p95': float(np.percentile(np.abs(np.log10(second_m) - np.log10(first_m)), 95.0)),
    }


def run_pair(args, compare_spectra) -> None:
    from .m_star_bootstrap_v1 import _load_mt, _product_consistency

    for node_id in NODES:
        root = Path(args.result_root) / 'continuation' / node_id
        records = {start: root / start / 'continuation.json' for start in STARTS}
        if not all(path.is_file() for path in records.values()):
            print(f'[pair] {node_id} missing continuation record', flush=True)
            continue
        runs = {start: json.loads(path.read_text()) for start, path in records.items()}
        sources = {start: _load_mt(runs[start]['source_product']) for start in STARTS}
        trajectories = {start: _trajectory(root / start / 'iterations') for start in STARTS}
        gaps = [_gap(*sources['primary'], *sources['reference_start'])]
        for first, second in zip(trajectories['primary'], trajectories['reference_start']):
            gaps.append(_gap(first['column_mass'], first['temperature'], second['column_mass'], second['temperature']))
        products = {start: runs[start]['product_path'] for start in STARTS}
        consistency = _product_consistency(products['primary'], products['reference_start'])
        tio = _guarded_tio(
            compare_spectra,
            (products['primary'], root / 'spectra' / 'primary.npz'),
            (products['reference_start'], root / 'spectra' / 'reference_start.npz'),
        )
        closes = bool(consistency.get('passes') and tio.get('passes'))
        ratio = {
            key: gaps[-1][key] / gaps[0][key] if gaps[0][key] > 0 else None
            for key in ('temperature_relative_p95', 'column_mass_dex_p95')
        }
        drift = {start: _product_consistency(runs[start]['source_product'], products[start]) for start in STARTS}
        _write_json(root / 'pair.json', {
            'node_id': node_id,
            'gap_by_iteration': gaps,
            'gap_ratio_final_over_initial': ratio,
            'path_consistency': consistency,
            'tio': tio,
            'closes': closes,
            'drift_from_source': drift,
            'iteration_history': {
                start: [{k: v for k, v in row.items() if k not in ('temperature', 'column_mass')}
                        for row in trajectories[start]]
                for start in STARTS
            },
        })
        print(f"[pair] {node_id} closes={closes} consistency={consistency.get('passes')} "
              f"tio={tio.get('passes')} gap0 T {gaps[0]['temperature_relative_p95']:.2e} "
              f"m {gaps[0]['column_mass_dex_p95']:.2e} gap10 T {gaps[-1]['temperature_relative_p95']:.2e} "
              f"m {gaps[-1]['column_mass_dex_p95']:.2e}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--start', choices=STARTS)
    parser.add_argument('--pair', action='store_true')
    parser.add_argument('--only', choices=NODES)
    parser.add_argument('--result-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--inputs', type=Path, default=DEFAULT_ROOT / 'inputs' / 'warm_starts.npz')
    parser.add_argument('--synthesis-root', type=Path, required=True,
                        help='directory containing the emulator_v1_2 package')
    args = parser.parse_args(argv)
    if args.pair == bool(args.start):
        parser.error('give exactly one of --start or --pair')
    if args.start and not args.only:
        parser.error('--start needs --only')

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
    tag = 'pair' if args.pair else f'{args.start}_{args.only}'
    job_root = Path(args.result_root) / 'continuation' / 'jobs' / tag
    overlay_root = build_overlay('pchip', job_root)
    os.environ['NUMBA_CACHE_DIR'] = str(job_root / 'numba_cache')
    compare_spectra, modules = _import_physics(overlay_root, Path(args.synthesis_root))

    if args.pair:
        run_pair(args, compare_spectra)
        return 0
    identity = {
        'preregistration': str(PREREGISTRATION),
        'preregistration_sha256': _file_sha256(PREREGISTRATION),
        'driver_sha256': _file_sha256(Path(__file__)),
        'modules': modules,
        'overlay': str(overlay_root),
        'config_overrides': dict(ARMS['candidate']),
    }
    run_continue(args, _nodes(args.inputs)[args.only], identity)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
