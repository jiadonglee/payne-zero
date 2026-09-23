"""S3 convective inner-loop arm (8 passes, frozen mask) on pchip thermodynamics."""

import argparse
import copy
import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO / 'results/m_star_h2_inner_loop_s3_20260922'
DEEP_LAYERS = list(range(67, 80))
INNER_PASSES = 8

# Preregistered before the first run (adapted from the 2026-09-18 S3 decision
# table to the pchip context; the S0-style references are the 20260921 arms,
# not re-run): D arm succeeds iff its deep-window mean |R_raw| trajectory
# declines without the alpha=1 explosion (below 5.112 at it01-equivalent);
# A must stay eligible: A it09 mean |R_raw| <= its S0-pchip it0 value 0.1698.
# Systematic opposition of successive corrections on convective layers near
# the end = Hubeny cancellation, stop the arm and record it.  No material
# change versus the S0 references means solver order is not the missing
# deep-coupling ingredient.


def _run_exact_state_s3(
    *,
    atmosphere,
    labels,
    effective_temperature: float,
    run_id: str,
    result_root: Path,
    resume: bool = False,
    correct_written_gradient: bool = False,
    refresh_state: bool = False,
    hold_correction: bool = False,
    relaxation: float = 1.0,
    fill_holes: bool = False,
) -> dict[str, Any]:
    from bench.run_reference import _solver_config
    from payne_zero_atmosphere.runner import run_atmosphere_model
    from experiments.reduced_state_emulator.m_star_local_iteration_response_20260919 import (
        _clone_atmosphere, _physics_rows, _selected_state, _write_json,
    )

    if resume:
        array_path = result_root / 'arrays' / f'{run_id}.npz'
        if array_path.exists():
            from experiments.reduced_state_emulator.m_star_local_iteration_response_20260919 import (
                _record_from_saved_arrays,
            )
            record = _record_from_saved_arrays(
                array_path=array_path, run_id=run_id,
                effective_temperature=effective_temperature, indices=DEEP_LAYERS,
            )
            _write_json(result_root / 'records' / f'{run_id}.json', record)
            print(f'[{run_id}] resumed from {array_path}', flush=True)
            return record

    capture: dict[str, Any] = {}

    def hook(iteration_index, setup, step):
        physics, physics_arrays = _physics_rows(
            step=step, effective_temperature=effective_temperature,
            indices=DEEP_LAYERS,
        )
        capture['iteration_index'] = int(iteration_index)
        capture['input_arrays'] = {
            'temperature': np.asarray(setup.atmosphere.temperature, dtype=np.float64).copy(),
            'column_mass': np.asarray(setup.atmosphere.column_mass, dtype=np.float64).copy(),
        }
        capture['output_arrays'] = {
            'temperature': np.asarray(step.remapped.atmosphere.temperature, dtype=np.float64).copy(),
            'column_mass': np.asarray(step.remapped.atmosphere.column_mass, dtype=np.float64).copy(),
        }
        capture['physics_arrays'] = physics_arrays
        capture['timing'] = dict(step.timing)
        return {'captured_input_residual': True}

    started = time.perf_counter()
    error = None
    result = None
    try:
        config = _solver_config(
            _clone_atmosphere(atmosphere),
            iterations_per_trial=1,
            structured_atmosphere_path=None,
            debug_state_path=None,
        )
        config = dataclasses.replace(
            config,
            enable_opacity_lagging=False,
            opacity_recompute_interval=1,
            flux_residual_guided_damping=False,
            require_improving_flux_residual=False,
            convection_zone_inner_loop_passes=INNER_PASSES,
            convection_zone_inner_loop_freeze_mask=True,
            convection_zone_inner_loop_correct_written_gradient=bool(
                correct_written_gradient
            ),
            convection_zone_inner_loop_refresh_state=bool(refresh_state),
            convection_zone_inner_loop_hold_correction=bool(hold_correction),
            convection_zone_inner_loop_relaxation=float(relaxation),
            convection_zone_inner_loop_fill_holes=bool(fill_holes),
        )
        result = run_atmosphere_model(config, after_iteration_hook=hook)
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'
    seconds = time.perf_counter() - started
    record: dict[str, Any] = {
        'run_id': run_id, 'seconds': float(seconds),
        'opacity_lagging': False, 'iteration_cap': 1,
        'inner_loop_passes': INNER_PASSES, 'error': error,
    }
    if result is not None and capture:
        record.update({
            'iterations_completed': int(result.iterations_completed),
            'diagnostics': {
                key: value for key, value in result.diagnostics.items()
                if key.startswith('convection_inner_loop_')
                or key in ('p95_absolute_flux_error_percent',
                           'maximum_absolute_flux_error_percent', 'total_seconds')
            },
            'timing': capture['timing'],
        })
        arrays = {
            **capture['input_arrays'],
            **{f'output_{key}': value for key, value in capture['output_arrays'].items()},
            **capture['physics_arrays'],
        }
        array_path = result_root / 'arrays' / f'{run_id}.npz'
        array_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(array_path, **arrays)
        record['arrays_path'] = str(array_path.resolve())
    elif error is None:
        record['error'] = 'runner returned without an iteration hook capture'
    _write_json(result_root / 'records' / f'{run_id}.json', record)
    print(
        f'[{run_id}] {seconds:.1f}s '
        f'status={"ok" if record.get("error") is None else "failed"}', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'D'), required=True)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--correct-written-gradient', action='store_true',
        help='S3w: inner passes shift the written gradient by the change the '
             'centred MLT read-back gradient needs',
    )
    parser.add_argument(
        '--refresh-state', action='store_true',
        help='inner-loop physics re-solves the pressure-iteration state at each '
             'trial temperature (opacity stays at the round input)',
    )
    parser.add_argument(
        '--hold-correction', action='store_true',
        help='the global correction leaves the layers the inner loop set unchanged',
    )
    parser.add_argument(
        '--relaxation', type=float, default=1.0,
        help='fraction of the inner-loop temperature change kept at loop exit',
    )
    parser.add_argument(
        '--fill-holes', action='store_true',
        help='subadiabatic interior layers that radiation cannot carry join the '
             'inner-loop working mask',
    )
    parser.add_argument('--result-root', type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    arm = 's3w' if args.correct_written_gradient else 's3'
    if args.refresh_state:
        arm += 'r'
    if args.hold_correction:
        arm += 'c'
    if args.relaxation != 1.0:
        arm += f'_l{int(round(100 * args.relaxation)):03d}'
    if args.fill_holes:
        arm += '_h'

    for key, value in (
        ('NUMBA_THREADING_LAYER', 'workqueue'), ('NUMBA_NUM_THREADS', '1'),
        ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'),
        ('OPENBLAS_NUM_THREADS', '1'), ('VECLIB_MAXIMUM_THREADS', '1'),
        ('CUDA_VISIBLE_DEVICES', ''),
    ):
        os.environ[key] = value
    os.environ['PAYNE_ZERO_DATA_ROOT'] = str(REPO / 'source_data_files')

    root = args.result_root / f'{args.case.lower()}_pchip_{arm}'
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO))
    from experiments.reduced_state_emulator.m_star_h2_paired_iteration_20260921 import (
        build_overlay,
    )
    overlay_root = build_overlay('pchip', root)
    os.environ['NUMBA_CACHE_DIR'] = str(root / 'numba_cache')
    sys.path.insert(0, str(overlay_root))

    import payne_zero_atmosphere.runner as runner_module
    from experiments.reduced_state_emulator.m_star_local_iteration_response_20260919 import (
        _load_case, _state_for_trial, _state_from_mt, _write_json,
    )
    from payne_zero_atmosphere import molecular_equilibrium as me
    from scipy.interpolate import PchipInterpolator

    table = me._HYDROGEN_MOLECULE_PARTITION_TABLE
    temperatures = np.linspace(100., 19900., 2001)
    nodes = np.arange(1, len(table) + 1) * 100.
    actual = np.array([me._interp_hydrogen_molecule_partition_compiled(t, table)
                       for t in temperatures])
    error = float(np.max(np.abs(actual - PchipInterpolator(nodes, table)(temperatures))))
    assert error < 1e-10, error
    assert str(overlay_root) in me.__file__, me.__file__
    _write_json(root / 'implementation.json', {
        'case': args.case, 'scheme': 'pchip', 'inner_loop_passes': INNER_PASSES,
        'correct_written_gradient': bool(args.correct_written_gradient),
        'refresh_state': bool(args.refresh_state),
        'hold_correction': bool(args.hold_correction),
        'relaxation': float(args.relaxation),
        'fill_holes': bool(args.fill_holes),
        'module': me.__file__, 'scipy_max_abs_error': error,
    })

    case = _load_case(args.case)
    labels = case['labels']
    effective_temperature = case['target_temperature_K']
    control = (REPO / 'results/m_star_trial_comparison_20260920'
               / f'controls_{args.case}/arrays/{args.case.lower()}_alpha_0p0.npz')
    with np.load(control, allow_pickle=False) as data:
        state = _state_from_mt(labels, data['column_mass'], data['temperature'])

    trajectory = {
        'case': args.case, 'scheme': 'pchip',
        'inner_loop': f'{arm.upper()} ({INNER_PASSES} passes, frozen mask)',
        'effective_temperature_K': effective_temperature,
        'initial_control': str(control), 'deep_layers': DEEP_LAYERS,
        'metrics_definition': (
            'primary = mean|R_raw| over layers 67-79; reference arms are the '
            '20260921 S0-style pchip trajectories for the same origins; '
            'R_raw_input_state_deep_mean_abs uses this round transfer H_rad and '
            'the pre-inner-loop MLT flux of the input state'),
        'iterations': [],
    }
    trajectory_path = root / 'trajectory.json'

    for iteration in range(args.iterations):
        run_id = f'{args.case.lower()}_pchip_{arm}_it{iteration:02d}'
        capture = []
        original = runner_module.compute_convection

        def wrapped(**kwargs):
            result = original(**kwargs)
            capture.append((copy.deepcopy(kwargs), copy.deepcopy(result)))
            return result

        runner_module.compute_convection = wrapped
        try:
            record = _run_exact_state_s3(
                atmosphere=state, labels=labels,
                effective_temperature=effective_temperature, run_id=run_id,
                result_root=root, resume=args.resume,
                correct_written_gradient=args.correct_written_gradient,
                refresh_state=args.refresh_state,
                hold_correction=args.hold_correction,
                relaxation=args.relaxation,
                fill_holes=args.fill_holes,
            )
        finally:
            runner_module.compute_convection = original
        if record.get('error') is not None:
            trajectory['iterations'].append({'iteration': iteration, 'run_id': run_id,
                                             'error': record['error']})
            _write_json(trajectory_path, trajectory)
            print(f'[{run_id}] failed: {record["error"].splitlines()[0]}', flush=True)
            return 1
        with np.load(record['arrays_path'], allow_pickle=False) as data:
            r_raw = np.abs(np.asarray(data['R_raw'], dtype=np.float64))
            r_smoothed = np.abs(np.asarray(data['R_smoothed'], dtype=np.float64))
            temperature_in = np.asarray(data['temperature'], dtype=np.float64)
            temperature_out = np.asarray(data['output_temperature'], dtype=np.float64)
            deep = slice(DEEP_LAYERS[0], DEEP_LAYERS[-1] + 1)
            row = {
                'iteration': iteration, 'run_id': run_id,
                'R_raw_deep_mean_abs': float(np.mean(r_raw[deep])),
                'R_raw_deep_max': float(np.max(r_raw[deep])),
                'R_raw_full_max': float(np.max(r_raw)),
                'R_smoothed_deep_max': float(np.max(r_smoothed[deep])),
                'dT_max_abs_K': float(np.max(np.abs(temperature_out - temperature_in))),
            }
        if capture:
            kwargs, result = capture[0]
            target = float(kwargs['target_integrated_eddington_flux'])
            row['target_integrated_eddington_flux'] = target
            with np.load(record['arrays_path'], allow_pickle=False) as data:
                h_rad = np.asarray(data['Hrad'], dtype=np.float64)
            input_state_r_raw = np.abs(
                (h_rad + np.asarray(result.raw_convective_flux, dtype=np.float64))
                / target - 1.0
            )
            row['R_raw_input_state_deep_mean_abs'] = float(
                np.mean(input_state_r_raw[deep])
            )
            row['layers'] = {}
            for index in DEEP_LAYERS:
                row['layers'][str(index)] = {
                    'heat_capacity': float(result.heat_capacity[index]),
                    'adiabatic_gradient': float(result.adiabatic_gradient[index]),
                    'log_temperature_pressure_gradient': float(
                        result.logarithmic_temperature_pressure_gradient[index]),
                    'Hconv_raw_over_target': float(
                        result.raw_convective_flux[index] / target),
                    'convection_on': bool(result.raw_convective_flux[index] > 0.0),
                }
        else:
            row['layers'] = None
        row['inner_loop_diagnostics'] = {
            key: value for key, value in record.get('diagnostics', {}).items()
            if key.startswith('convection_inner_loop_')
        }
        trajectory['iterations'].append(row)
        _write_json(trajectory_path, trajectory)
        print(
            f'[{run_id}] input_state_deep_mean_raw='
            f'{row.get("R_raw_input_state_deep_mean_abs", float("nan")):.4f} '
            f'deep_mean_raw={row["R_raw_deep_mean_abs"]:.4f} '
            f'full_max_raw={row["R_raw_full_max"]:.4f} '
            f'deep_max_smooth={row["R_smoothed_deep_max"]:.4f} '
            f'dTmax={row["dT_max_abs_K"]:.3f}', flush=True)
        state = _state_for_trial(labels, record)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
