"""Inner-loop candidate versus S0 (pchip overlay) on the safezone v2 dwarf validation points.

Protocol: ``notes/m_star_inner_loop_validation_preregistration_20260924.md``.
Three modes, each for one arm on the nodes selected with ``--only``:

``--start warm``
    strict primary solve from the frozen safezone v2 warm start, independent
    self-restart, frozen flux gate on both, path consistency and TiO dual path
    (the development-sweep eligibility), plus the diagnostic comparison of the
    primary product with the reference product.
``--start reference``
    strict solve from the reference product's (m, T).
``--pair``
    after both have finished: start independence (reference-start survival,
    flux gate, path consistency and TiO against the primary product) for every
    arm and node.

Arms, the physics import and the TiO settings come from the development-sweep
driver; reconstruction, solves, the flux gate and path consistency from the S3
v1 functions.

    python -m experiments.reduced_state_emulator.m_star_inner_loop_validation_20260924 \\
        --arm candidate --start warm --only g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3800 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .m_star_inner_loop_dev_sweep_20260923 import (
    ARMS,
    GATE_HASH_PREFIX,
    ITERATION_CAP,
    STRICT_ALL_LAYER_LIMIT,
    TIO_GATE,
    TIO_RESOLUTION,
    TIO_WINDOW_NM,
    _file_sha256,
    _import_physics,
    _write_json,
)

REPO = Path(__file__).resolve().parents[2]
PREREGISTRATION = REPO / 'notes/m_star_inner_loop_validation_preregistration_20260924.md'
DEFAULT_ROOT = REPO / 'results/m_star_inner_loop_validation_20260924'


def _nodes(inputs: Path) -> dict[str, dict[str, Any]]:
    with np.load(inputs, allow_pickle=False) as data:
        return {
            str(node): {
                'node_id': str(node),
                'labels': np.asarray(data['labels'][index], dtype=np.float64),
                'warm_column_mass': np.asarray(data['column_mass'][index], dtype=np.float64),
                'warm_temperature': np.asarray(data['temperature'][index], dtype=np.float64),
                'reference_product': str(data['reference_product_paths'][index]),
                'reference_campaign': str(data['reference_campaigns'][index]),
                'reference_flux_p95_percent': float(data['reference_flux_p95_percent'][index]),
                'held_out': bool(data['held_out'][index]),
            }
            for index, node in enumerate(data['node_ids'])
        }


def _track(labels: np.ndarray):
    from .cool_star_step_test import TrackSpec

    return TrackSpec(
        log_surface_gravity=float(labels[1]),
        metallicity=float(labels[2]),
        alpha_enhancement=float(labels[3]),
        carbon_enhancement=0.0,
        microturbulence_km_s=float(labels[4]),
    )


def _spectrum(compare_spectra, product: str | Path, path: Path) -> dict:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        compare_spectra._synthesize_one(
            Path(product), path,
            wavelength_start_nm=TIO_WINDOW_NM[0], wavelength_end_nm=TIO_WINDOW_NM[1],
            resolution=TIO_RESOLUTION, molecular_lines=True, device=None, dtype='float64',
        )
    return compare_spectra._load_spectrum_npz(path)


def _tio(compare_spectra, base: tuple[str | Path, Path], other: tuple[str | Path, Path]) -> dict[str, Any]:
    """TiO statistics of ``other`` against ``base``, as in the development-sweep dual path."""

    base_spectrum = _spectrum(compare_spectra, *base)
    other_spectrum = _spectrum(compare_spectra, *other)
    metrics = {
        'normalized_flux': compare_spectra._absolute_stats(
            other_spectrum['normalized_flux'], base_spectrum['normalized_flux'])['max'],
        'flux_total': compare_spectra._continuum_scaled_stats(
            other_spectrum['flux_total'], base_spectrum['flux_total'], base_spectrum['flux_continuum'])['max'],
        'flux_continuum': compare_spectra._relative_stats(
            other_spectrum['flux_continuum'], base_spectrum['flux_continuum'])['max'],
    }
    return {
        'available': True,
        'metrics': metrics,
        'passes': bool(max(metrics.values()) <= TIO_GATE),
        'spectra': {'base': str(base[1]), 'other': str(other[1])},
    }


def _guarded_tio(compare_spectra, base, other) -> dict[str, Any]:
    try:
        return _tio(compare_spectra, base, other)
    except Exception as exc:  # noqa: BLE001 - a failed synthesis is recorded, not raised
        return {'available': False, 'passes': False, 'error': f'{type(exc).__name__}: {exc}'}


def _case_root(result_root: Path, arm: str, node_id: str) -> Path:
    return Path(result_root) / arm / 'cases' / node_id


def _flux_gate(path: Path) -> dict:
    gate = json.loads(Path(path).read_text())
    if not str(gate.get('gate_hash', '')).startswith(GATE_HASH_PREFIX):
        raise ValueError(f'flux gate hash {gate.get("gate_hash")} is not the frozen gate')
    return gate if 'thresholds' in gate else {'thresholds': gate}


def run_warm(args, node: dict, compare_spectra, identity: dict) -> dict:
    from . import m_star_iteration_tomography_v1 as tomography
    from .cool_star_step_test import _reconstruct_from_mt, _solve_attempt
    from .m_star_bootstrap_v1 import _load_mt, _passes_flux_gate, _product_consistency

    flux_gate = _flux_gate(args.flux_gate)
    overrides = dict(ARMS[args.arm])
    track = _track(node['labels'])
    labels = track.labels(float(node['labels'][0]))
    case_root = _case_root(args.result_root, args.arm, node['node_id'])
    reference = Path(args.reference_root) / node['reference_product']

    primary, _ = _solve_attempt(
        track=track, method=f'{args.arm}_warm_primary', schedule='validation_warm_start',
        source_temperature=None, target_labels=labels,
        initial_atmosphere=_reconstruct_from_mt(labels, node['warm_column_mass'], node['warm_temperature']),
        product_dir=case_root / 'products' / 'primary', iteration_cap=ITERATION_CAP,
        maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
        config_overrides=overrides,
        after_iteration_hook=tomography.make_tomography_hook(case_root / 'iterations' / 'primary'),
    )
    restart = None
    if primary.get('survives_solver') and primary.get('product_path'):
        solved_m, solved_t = _load_mt(primary['product_path'])
        restart, _ = _solve_attempt(
            track=track, method=f'{args.arm}_strict_self_restart', schedule='independent_self_restart',
            source_temperature=float(node['labels'][0]), target_labels=labels,
            initial_atmosphere=_reconstruct_from_mt(labels, solved_m, solved_t),
            product_dir=case_root / 'products' / 'restart', iteration_cap=ITERATION_CAP,
            maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
            config_overrides=overrides,
        )
    primary_flux = _passes_flux_gate(primary, flux_gate)
    restart_flux = {'passes': False, 'metrics': {}} if restart is None else _passes_flux_gate(restart, flux_gate)
    consistency = _product_consistency(
        primary.get('product_path'), None if restart is None else restart.get('product_path'),
    )
    spectra = case_root / 'spectra'
    tio: dict[str, Any] = {'available': False, 'passes': False}
    if restart is not None and primary.get('product_path') and restart.get('product_path'):
        tio = _guarded_tio(
            compare_spectra,
            (primary['product_path'], spectra / 'primary.npz'),
            (restart['product_path'], spectra / 'restart.npz'),
        )
    versus_reference: dict[str, Any] = {'available': False}
    if primary.get('product_path'):
        versus_reference = {
            'available': True,
            'path_consistency': _product_consistency(reference, primary['product_path']),
            'tio': _guarded_tio(
                compare_spectra,
                (reference, spectra / 'reference.npz'),
                (primary['product_path'], spectra / 'primary.npz'),
            ),
        }
    eligible = bool(
        primary.get('survives_solver') and primary_flux.get('passes')
        and restart_flux.get('passes') and consistency.get('passes') and tio.get('passes')
    )
    row = {
        'node_id': node['node_id'],
        'held_out': node['held_out'],
        'labels': labels.as_kwargs(),
        'arm': args.arm,
        'config_overrides': overrides,
        'reference': {
            'product': str(reference),
            'campaign': node['reference_campaign'],
            'flux_p95_percent': node['reference_flux_p95_percent'],
        },
        'primary': primary,
        'restart': restart,
        'primary_flux_gate': primary_flux,
        'restart_flux_gate': restart_flux,
        'path_consistency': consistency,
        'tio_dual_path': tio,
        'eligible': eligible,
        'versus_reference': versus_reference,
        'identity': identity,
    }
    _write_json(case_root / 'case.json', row)
    print(
        f"[{args.arm} warm] {node['node_id']} primary_iters={primary.get('iterations')} "
        f"survives={primary.get('survives_solver')} primary_flux={primary_flux.get('passes')} "
        f"restart_flux={restart_flux.get('passes')} consistency={consistency.get('passes')} "
        f"tio={tio.get('passes')} eligible={eligible}",
        flush=True,
    )
    return row


def run_reference(args, node: dict, identity: dict) -> dict:
    from . import m_star_iteration_tomography_v1 as tomography
    from .cool_star_step_test import _reconstruct_from_mt, _solve_attempt
    from .m_star_bootstrap_v1 import _load_mt, _passes_flux_gate

    flux_gate = _flux_gate(args.flux_gate)
    overrides = dict(ARMS[args.arm])
    track = _track(node['labels'])
    labels = track.labels(float(node['labels'][0]))
    case_root = _case_root(args.result_root, args.arm, node['node_id'])
    reference = Path(args.reference_root) / node['reference_product']
    reference_m, reference_t = _load_mt(reference)
    solve, _ = _solve_attempt(
        track=track, method=f'{args.arm}_reference_start', schedule='validation_reference_start',
        source_temperature=None, target_labels=labels,
        initial_atmosphere=_reconstruct_from_mt(labels, reference_m, reference_t),
        product_dir=case_root / 'products' / 'reference_start', iteration_cap=ITERATION_CAP,
        maximum_all_layer_relative_temperature_change=STRICT_ALL_LAYER_LIMIT,
        config_overrides=overrides,
        after_iteration_hook=tomography.make_tomography_hook(case_root / 'iterations' / 'reference_start'),
    )
    flux = _passes_flux_gate(solve, flux_gate)
    row = {
        'node_id': node['node_id'],
        'arm': args.arm,
        'reference_product': str(reference),
        'solve': solve,
        'flux_gate': flux,
        'identity': identity,
    }
    _write_json(case_root / 'reference_start.json', row)
    print(
        f"[{args.arm} reference] {node['node_id']} iters={solve.get('iterations')} "
        f"survives={solve.get('survives_solver')} flux={flux.get('passes')}",
        flush=True,
    )
    return row


def run_pair(args, nodes: dict, compare_spectra) -> None:
    from .m_star_bootstrap_v1 import _product_consistency

    for arm in sorted(ARMS):
        for node_id in nodes:
            case_root = _case_root(args.result_root, arm, node_id)
            case_path, reference_path = case_root / 'case.json', case_root / 'reference_start.json'
            if not (case_path.is_file() and reference_path.is_file()):
                print(f'[{arm} pair] {node_id} missing case or reference-start record', flush=True)
                continue
            case = json.loads(case_path.read_text())
            reference_start = json.loads(reference_path.read_text())
            primary_product = (case.get('primary') or {}).get('product_path')
            start_product = (reference_start.get('solve') or {}).get('product_path')
            survives = bool((reference_start.get('solve') or {}).get('survives_solver'))
            flux = bool((reference_start.get('flux_gate') or {}).get('passes'))
            consistency = _product_consistency(primary_product, start_product)
            tio: dict[str, Any] = {'available': False, 'passes': False}
            if primary_product and start_product:
                tio = _guarded_tio(
                    compare_spectra,
                    (primary_product, case_root / 'spectra' / 'primary.npz'),
                    (start_product, case_root / 'spectra' / 'reference_start.npz'),
                )
            start_independent = bool(survives and flux and consistency.get('passes') and tio.get('passes'))
            _write_json(case_root / 'start_independence.json', {
                'node_id': node_id,
                'arm': arm,
                'reference_start_survives': survives,
                'reference_start_flux_gate': flux,
                'path_consistency': consistency,
                'tio': tio,
                'start_independent': start_independent,
            })
            print(
                f'[{arm} pair] {node_id} survives={survives} flux={flux} '
                f"consistency={consistency.get('passes')} tio={tio.get('passes')} "
                f'start_independent={start_independent}',
                flush=True,
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arm', choices=sorted(ARMS))
    parser.add_argument('--start', choices=('warm', 'reference'))
    parser.add_argument('--pair', action='store_true')
    parser.add_argument('--only', nargs='+', help='node ids; all nodes when omitted')
    parser.add_argument('--result-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--inputs', type=Path, default=DEFAULT_ROOT / 'inputs' / 'warm_starts.npz')
    parser.add_argument('--reference-root', type=Path, required=True,
                        help='campaign tree the reference product paths are relative to')
    parser.add_argument('--flux-gate', type=Path, required=True)
    parser.add_argument('--synthesis-root', type=Path, required=True,
                        help='directory containing the emulator_v1_2 package')
    args = parser.parse_args(argv)
    if args.pair == bool(args.start):
        parser.error('give exactly one of --start or --pair')
    if args.start and not args.arm:
        parser.error('--start needs --arm')

    for key, value in (
        ('NUMBA_THREADING_LAYER', 'workqueue'), ('NUMBA_NUM_THREADS', '1'),
        ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'),
        ('OPENBLAS_NUM_THREADS', '1'), ('VECLIB_MAXIMUM_THREADS', '1'),
        ('CUDA_VISIBLE_DEVICES', ''),
    ):
        os.environ[key] = value
    os.environ['PAYNE_ZERO_DATA_ROOT'] = str(REPO / 'source_data_files')

    nodes = _nodes(args.inputs)
    if args.only:
        unknown = sorted(set(args.only) - set(nodes))
        if unknown:
            raise ValueError(f'unknown node ids {unknown}')
        selected = [nodes[node_id] for node_id in args.only]
    else:
        selected = list(nodes.values())

    sys.path.insert(0, str(REPO))
    from experiments.reduced_state_emulator.m_star_h2_paired_iteration_20260921 import build_overlay
    if args.pair:
        job_root = Path(args.result_root) / 'pair' / 'job'
    else:
        tag = selected[0]['node_id'] if len(selected) == 1 else f'{len(selected)}_nodes'
        job_root = Path(args.result_root) / args.arm / 'jobs' / f'{args.start}_{tag}'
    overlay_root = build_overlay('pchip', job_root)
    os.environ['NUMBA_CACHE_DIR'] = str(job_root / 'numba_cache')
    compare_spectra, modules = _import_physics(overlay_root, Path(args.synthesis_root))

    if args.pair:
        run_pair(args, {node['node_id']: node for node in selected}, compare_spectra)
        return 0
    identity = {
        'preregistration': str(PREREGISTRATION),
        'preregistration_sha256': _file_sha256(PREREGISTRATION),
        'driver_sha256': _file_sha256(Path(__file__)),
        'inputs': str(args.inputs),
        'inputs_sha256': _file_sha256(args.inputs),
        'flux_gate': str(args.flux_gate),
        'modules': modules,
        'overlay': str(overlay_root),
        'iteration_cap': ITERATION_CAP,
        'strict_all_layer_limit': STRICT_ALL_LAYER_LIMIT,
        'tio': {'window_nm': TIO_WINDOW_NM, 'resolution': TIO_RESOLUTION, 'gate': TIO_GATE},
    }
    for node in selected:
        if args.start == 'warm':
            run_warm(args, node, compare_spectra, identity)
        else:
            run_reference(args, node, identity)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
