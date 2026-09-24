"""Development-point sweep of the inner-loop candidate against S0 (pchip overlay).

Protocol: ``notes/m_star_inner_loop_dev_sweep_preregistration_20260923.md``.
One invocation runs one arm on the points selected with ``--only``: a strict
primary solve, an independent self-restart from its (m, T), the frozen flux
gate on both, path consistency, and the TiO dual-path check (primary versus
restart spectra, 665-667 nm, R = 20000, molecular lines).  Seeds, solves, the
flux gate and path consistency reuse the S3 v1 driver's functions.

    python -m experiments.reduced_state_emulator.m_star_inner_loop_dev_sweep_20260923 \\
        --arm candidate --only t3600 --synthesis-root /path/with/emulator_v1_2 ...
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PREREGISTRATION = REPO / 'notes/m_star_inner_loop_dev_sweep_preregistration_20260923.md'
GATE_HASH_PREFIX = 'ae0d384e'
ITERATION_CAP = 60
STRICT_ALL_LAYER_LIMIT = 5.0e-4
TIO_WINDOW_NM = (665.0, 667.0)
TIO_RESOLUTION = 20000.0
TIO_GATE = 5.0e-3
ARMS: dict[str, dict[str, Any]] = {
    'candidate': {
        'convection_zone_inner_loop_passes': 8,
        'convection_zone_inner_loop_freeze_mask': True,
        'convection_zone_inner_loop_correct_written_gradient': True,
        'convection_zone_inner_loop_refresh_state': True,
        'convection_zone_inner_loop_hold_correction': True,
        'convection_zone_inner_loop_relaxation': 0.5,
        'convection_zone_inner_loop_fill_holes': True,
    },
    's0': {},
}
ARMS['candidate_filter'] = {
    **ARMS['candidate'], 'convection_zone_inner_loop_filter_written_gradient': True,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + '\n')


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _import_physics(overlay_root: Path, synthesis_root: Path):
    """Load the overlay atmosphere and this tree's synthesis before emulator_v1_2.

    ``emulator_v1_2.lib.harness`` prepends its own workspace to ``sys.path``;
    importing the two packages first pins them to the intended sources.
    """

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(overlay_root))
    import payne_zero_atmosphere
    import payne_zero_synthesis
    from payne_zero_atmosphere import molecular_equilibrium

    sys.path.append(str(synthesis_root))
    from emulator_v1_2.gates import compare_spectra

    atmosphere_path = Path(payne_zero_atmosphere.__file__).resolve()
    synthesis_path = Path(payne_zero_synthesis.__file__).resolve()
    assert str(overlay_root.resolve()) in str(atmosphere_path), atmosphere_path
    assert str(overlay_root.resolve()) in str(Path(molecular_equilibrium.__file__).resolve())
    assert str(REPO.resolve()) in str(synthesis_path), synthesis_path
    assert sys.modules['payne_zero_atmosphere'] is payne_zero_atmosphere
    assert sys.modules['payne_zero_synthesis'] is payne_zero_synthesis
    return compare_spectra, {
        'payne_zero_atmosphere': str(atmosphere_path),
        'payne_zero_synthesis': str(synthesis_path),
        'compare_spectra': str(Path(compare_spectra.__file__).resolve()),
    }


def _tio(compare_spectra, primary_product: str, restart_product: str, spectra_dir: Path) -> dict[str, Any]:
    paths = {}
    for tag, product in (('primary', primary_product), ('restart', restart_product)):
        path = spectra_dir / f'{tag}.npz'
        if not path.is_file():
            compare_spectra._synthesize_one(
                Path(product), path,
                wavelength_start_nm=TIO_WINDOW_NM[0], wavelength_end_nm=TIO_WINDOW_NM[1],
                resolution=TIO_RESOLUTION, molecular_lines=True, device=None, dtype='float64',
            )
        paths[tag] = path
    primary = compare_spectra._load_spectrum_npz(paths['primary'])
    restart = compare_spectra._load_spectrum_npz(paths['restart'])
    metrics = {
        'normalized_flux': compare_spectra._absolute_stats(
            restart['normalized_flux'], primary['normalized_flux'])['max'],
        'flux_total': compare_spectra._continuum_scaled_stats(
            restart['flux_total'], primary['flux_total'], primary['flux_continuum'])['max'],
        'flux_continuum': compare_spectra._relative_stats(
            restart['flux_continuum'], primary['flux_continuum'])['max'],
    }
    return {
        'metrics': metrics,
        'passes': bool(max(metrics.values()) <= TIO_GATE),
        'spectra': {tag: str(path) for tag, path in paths.items()},
    }


def run_point(args: argparse.Namespace, candidate: dict[str, Any], compare_spectra, identity: dict) -> dict:
    from . import m_star_iteration_tomography_v1 as tomography
    from . import m_star_solver_policy_arms_v2 as policy
    from .cool_star_step_test import _reconstruct_from_mt, _solve_attempt
    from .m_star_bootstrap_v1 import _load_mt, _passes_flux_gate, _product_consistency
    from . import m_star_bootstrap_v1r2_marcs100 as base

    gate = json.loads(Path(args.flux_gate).read_text())
    if not str(gate.get('gate_hash', '')).startswith(GATE_HASH_PREFIX):
        raise ValueError(f'flux gate hash {gate.get("gate_hash")} is not the frozen gate')
    flux_gate = gate if 'thresholds' in gate else {'thresholds': gate}
    overrides = dict(ARMS[args.arm])

    seed, provenance = policy._case_seed(
        candidate,
        interp_root=Path(args.interp_root),
        v1r2_root=Path(args.v1r2_root),
        tomography_root=Path(args.tomography_root),
    )
    track = base._track_from_payload(provenance['track'])
    labels = track.labels(float(candidate['temperature_K']))
    case_root = (
        Path(args.result_root) / args.arm / 'cases' / 'dwarf'
        / candidate['track_slug'] / f"t{int(candidate['temperature_K']):04d}"
    )
    primary, _ = _solve_attempt(
        track=track, method=f'{args.arm}_strict_primary', schedule='dev_sweep',
        source_temperature=None, target_labels=labels, initial_atmosphere=seed,
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
            source_temperature=float(candidate['temperature_K']), target_labels=labels,
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
    tio: dict[str, Any] = {'available': False, 'passes': False}
    if restart is not None and primary.get('product_path') and restart.get('product_path'):
        try:
            tio = {'available': True, **_tio(
                compare_spectra, primary['product_path'], restart['product_path'], case_root / 'spectra',
            )}
        except Exception as exc:  # noqa: BLE001 - a failed synthesis is recorded, not raised
            tio = {'available': False, 'passes': False, 'error': f'{type(exc).__name__}: {exc}'}
    eligible = bool(
        primary.get('survives_solver') and primary_flux.get('passes')
        and restart_flux.get('passes') and consistency.get('passes') and tio.get('passes')
    )
    row = {
        **candidate,
        'arm': args.arm,
        'config_overrides': overrides,
        'seed_provenance': provenance,
        'primary': primary,
        'restart': restart,
        'primary_flux_gate': primary_flux,
        'restart_flux_gate': restart_flux,
        'path_consistency': consistency,
        'tio_dual_path': tio,
        'eligible': eligible,
        'identity': identity,
    }
    _write_json(case_root / 'case.json', row)
    print(
        f"[{args.arm}] {candidate['candidate_id']} primary_iters={primary.get('iterations')} "
        f"survives={primary.get('survives_solver')} primary_flux={primary_flux.get('passes')} "
        f"restart_flux={restart_flux.get('passes')} consistency={consistency.get('passes')} "
        f"tio={tio.get('passes')} eligible={eligible}",
        flush=True,
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--arm', choices=sorted(ARMS), required=True)
    parser.add_argument('--only', nargs='+', required=True, help='candidate_id substrings, e.g. t3600')
    parser.add_argument('--result-root', type=Path, default=REPO / 'results/m_star_inner_loop_dev_sweep_20260923')
    parser.add_argument('--tomography-root', type=Path, required=True)
    parser.add_argument('--interp-root', type=Path, required=True)
    parser.add_argument('--v1r2-root', type=Path, required=True)
    parser.add_argument('--flux-gate', type=Path, required=True)
    parser.add_argument('--synthesis-root', type=Path, required=True,
                        help='directory containing the emulator_v1_2 package')
    parser.add_argument('--preregistration', type=Path, default=PREREGISTRATION)
    args = parser.parse_args(argv)

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
    tag = '_'.join(args.only)
    job_root = Path(args.result_root) / args.arm / 'jobs' / tag
    overlay_root = build_overlay('pchip', job_root)
    os.environ['NUMBA_CACHE_DIR'] = str(job_root / 'numba_cache')
    compare_spectra, modules = _import_physics(overlay_root, Path(args.synthesis_root))

    from . import m_star_iteration_tomography_v1 as tomography
    from . import m_star_solver_policy_arms_v2 as policy

    candidates = [
        {
            'candidate_id': f'{slug}_t{int(teff)}', 'track_slug': slug,
            'temperature_K': float(teff), 'seed_source': 'tomography_seed',
        }
        for slug, teff in tomography.CASES
    ] + [dict(policy.CONTINUATION_CASE)]
    selected = [c for c in candidates if any(token in c['candidate_id'] for token in args.only)]
    if not selected:
        raise ValueError(f'no development point matched --only {args.only}')
    identity = {
        'preregistration': str(args.preregistration),
        'preregistration_sha256': (
            _file_sha256(args.preregistration) if args.preregistration.is_file() else None
        ),
        'driver_sha256': _file_sha256(Path(__file__)),
        'flux_gate': str(args.flux_gate),
        'modules': modules,
        'overlay': str(overlay_root),
        'iteration_cap': ITERATION_CAP,
        'strict_all_layer_limit': STRICT_ALL_LAYER_LIMIT,
        'tio': {'window_nm': TIO_WINDOW_NM, 'resolution': TIO_RESOLUTION, 'gate': TIO_GATE},
    }
    for candidate in selected:
        run_point(args, candidate, compare_spectra, identity)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
