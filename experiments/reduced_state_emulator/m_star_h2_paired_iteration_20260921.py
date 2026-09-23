"""Continuous paired iterations under alternative H2 partition interpolation."""

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO / 'results/m_star_h2_paired_iteration_20260921'
DEEP_LAYERS = list(range(67, 80))

# Preregistered before the first run of this experiment:
# primary metric = mean |R_raw| over the fixed deep window 67-79 (equal layer
# weights); secondary = max |R_raw| full depth, max |R_raw| deep, max |R_smoothed|
# deep.  An arm that reaches iteration 10 with the primary metric below its
# iteration-0 value and still declining over iterations 8-10 is extended to 20;
# otherwise it stops at 10 and the trajectory decides the branch: sustained
# candidate decline (continue), improved local response with periodic residual
# oscillation (solver step control), or heat-capacity/adiabatic-gradient jumps
# persisting near table nodes (return to the EOS).

PCHIP_PYTHON = '''def _interp_hydrogen_molecule_partition(temperature_k: float) -> float:
    temperature = float(temperature_k)
    if not np.isfinite(temperature) or temperature <= 100.0:
        temperature = 100.0
    elif temperature >= 19900.0:
        temperature = 19900.0
    i = min(198, max(0, int(temperature / 100.0) - 1))
    y = _HYDROGEN_MOLECULE_PARTITION_TABLE
    slopes = [0.0, 0.0]
    for k in (0, 1):
        j = i + k
        if j == 0:
            d0 = (y[1] - y[0]) / 100.0
            d1 = (y[2] - y[1]) / 100.0
            s = (3.0*d0 - d1) / 2.0
            if s*d0 <= 0.0:
                s = 0.0
            elif d0*d1 < 0.0 and abs(s) > 3.0*abs(d0):
                s = 3.0*d0
        elif j == len(y) - 1:
            d0 = (y[-1] - y[-2]) / 100.0
            d1 = (y[-2] - y[-3]) / 100.0
            s = (3.0*d0 - d1) / 2.0
            if s*d0 <= 0.0:
                s = 0.0
            elif d0*d1 < 0.0 and abs(s) > 3.0*abs(d0):
                s = 3.0*d0
        else:
            dl = (y[j] - y[j-1]) / 100.0
            dr = (y[j+1] - y[j]) / 100.0
            s = 2.0*dl*dr/(dl+dr) if dl*dr > 0.0 else 0.0
        slopes[k] = s
    t = (temperature - (i+1)*100.0)/100.0
    return float((2*t**3-3*t**2+1)*y[i] + (t**3-2*t**2+t)*100.0*slopes[0]
                 + (-2*t**3+3*t**2)*y[i+1] + (t**3-t**2)*100.0*slopes[1])'''

PCHIP_COMPILED = '''    def _interp_hydrogen_molecule_partition_compiled(temperature_k, h2_partition_table):
        temperature = temperature_k
        if not math.isfinite(temperature) or temperature <= 100.0:
            temperature = 100.0
        elif temperature >= 19900.0:
            temperature = 19900.0
        i = min(198, max(0, int(temperature / 100.0) - 1))
        y = h2_partition_table
        slopes = np.empty(2)
        for k in range(2):
            j = i + k
            if j == 0:
                d0 = (y[1] - y[0]) / 100.0
                d1 = (y[2] - y[1]) / 100.0
                s = (3.0*d0 - d1) / 2.0
                if s*d0 <= 0.0:
                    s = 0.0
                elif d0*d1 < 0.0 and abs(s) > 3.0*abs(d0):
                    s = 3.0*d0
            elif j == len(y) - 1:
                d0 = (y[-1] - y[-2]) / 100.0
                d1 = (y[-2] - y[-3]) / 100.0
                s = (3.0*d0 - d1) / 2.0
                if s*d0 <= 0.0:
                    s = 0.0
                elif d0*d1 < 0.0 and abs(s) > 3.0*abs(d0):
                    s = 3.0*d0
            else:
                dl = (y[j] - y[j-1]) / 100.0
                dr = (y[j+1] - y[j]) / 100.0
                s = 2.0*dl*dr/(dl+dr) if dl*dr > 0.0 else 0.0
            slopes[k] = s
        t = (temperature - (i+1)*100.0)/100.0
        return ((2*t**3-3*t**2+1)*y[i] + (t**3-2*t**2+t)*100*slopes[0]
                + (-2*t**3+3*t**2)*y[i+1] + (t**3-t**2)*100*slopes[1])
'''

CUBIC_PYTHON = '''def _interp_hydrogen_molecule_partition(temperature_k: float) -> float:
    temperature = float(temperature_k)
    if not np.isfinite(temperature) or temperature <= 100.0:
        temperature = 100.0
    elif temperature >= 19900.0:
        temperature = 19900.0
    y = _HYDROGEN_MOLECULE_PARTITION_TABLE
    n = len(y)
    h = 100.0
    hh = h * h
    second = np.zeros(n)
    cp = np.zeros(n)
    dp = np.zeros(n)
    cp[1] = 0.25
    dp[1] = 1.5 * (y[2] - 2.0 * y[1] + y[0]) / hh
    for j in range(2, n - 1):
        denom = 4.0 - cp[j - 1]
        cp[j] = 1.0 / denom
        dp[j] = (6.0 * (y[j + 1] - 2.0 * y[j] + y[j - 1]) / hh - dp[j - 1]) / denom
    for j in range(n - 2, 0, -1):
        second[j] = dp[j] - cp[j] * second[j + 1]
    i = min(n - 2, max(0, int(temperature / 100.0) - 1))
    t = (temperature - (i + 1) * 100.0) / 100.0
    a = (1.0 - t) ** 3 / 6.0
    b = t ** 3 / 6.0
    return float((a * second[i] + b * second[i + 1]) * hh
                 + (y[i] - second[i] * hh / 6.0) * (1.0 - t)
                 + (y[i + 1] - second[i + 1] * hh / 6.0) * t)'''

CUBIC_COMPILED = '''    def _interp_hydrogen_molecule_partition_compiled(temperature_k, h2_partition_table):
        temperature = temperature_k
        if not math.isfinite(temperature) or temperature <= 100.0:
            temperature = 100.0
        elif temperature >= 19900.0:
            temperature = 19900.0
        y = h2_partition_table
        n = len(y)
        h = 100.0
        hh = h * h
        second = np.zeros(n)
        cp = np.zeros(n)
        dp = np.zeros(n)
        cp[1] = 0.25
        dp[1] = 1.5 * (y[2] - 2.0 * y[1] + y[0]) / hh
        for j in range(2, n - 1):
            denom = 4.0 - cp[j - 1]
            cp[j] = 1.0 / denom
            dp[j] = (6.0 * (y[j + 1] - 2.0 * y[j] + y[j - 1]) / hh - dp[j - 1]) / denom
        for j in range(n - 2, 0, -1):
            second[j] = dp[j] - cp[j] * second[j + 1]
        i = min(n - 2, max(0, int(temperature / 100.0) - 1))
        t = (temperature - (i + 1) * 100.0) / 100.0
        a = (1.0 - t) ** 3 / 6.0
        b = t ** 3 / 6.0
        return ((a * second[i] + b * second[i + 1]) * hh
                + (y[i] - second[i] * hh / 6.0) * (1.0 - t)
                + (y[i + 1] - second[i + 1] * hh / 6.0) * t)
'''

PYTHON_SOURCES = {'pchip': PCHIP_PYTHON, 'cubic': CUBIC_PYTHON}
COMPILED_SOURCES = {'pchip': PCHIP_COMPILED, 'cubic': CUBIC_COMPILED}


def _replace_module_level_function(source: str, name: str, replacement: str) -> str:
    start = source.index(f'def {name}(')
    following = source.find('\n\n\ndef ', start)
    if following < 0:
        raise RuntimeError(f'no module-level boundary after {name!r}')
    return source[:start] + replacement + '\n\n\n' + source[following + 3:]


def build_overlay(scheme: str, root: Path) -> Path:
    overlay_root = root / f'code_override_{scheme}'
    marker = overlay_root / '.built'
    if marker.exists():
        return overlay_root
    overlay = overlay_root / 'payne_zero_atmosphere'
    shutil.copytree(REPO / 'payne_zero_atmosphere', overlay,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.nbc', '*.nbi'))
    path = overlay / 'molecular_equilibrium.py'
    source = path.read_text()
    source = _replace_module_level_function(
        source, '_interp_hydrogen_molecule_partition', PYTHON_SOURCES[scheme])
    start = source.index('    def _interp_hydrogen_molecule_partition_compiled(')
    end = source.index('    @_njit_inline', start)
    source = source[:start] + COMPILED_SOURCES[scheme] + '\n' + source[end:]
    path.write_text(source)
    marker.write_text('ok\n')
    return overlay_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=('A', 'D'), default='D')
    parser.add_argument('--scheme', choices=('linear', 'pchip', 'cubic'), default='linear')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    for key, value in (
        ('NUMBA_THREADING_LAYER', 'workqueue'), ('NUMBA_NUM_THREADS', '1'),
        ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'),
        ('OPENBLAS_NUM_THREADS', '1'), ('VECLIB_MAXIMUM_THREADS', '1'),
        ('CUDA_VISIBLE_DEVICES', ''),
    ):
        os.environ[key] = value
    os.environ['PAYNE_ZERO_DATA_ROOT'] = str(REPO / 'source_data_files')

    root = RESULT_ROOT / f'{args.case.lower()}_{args.scheme}'
    root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO))
    if args.scheme != 'linear':
        overlay_root = build_overlay(args.scheme, root)
        os.environ['NUMBA_CACHE_DIR'] = str(root / 'numba_cache')
        sys.path.insert(0, str(overlay_root))

    from experiments.reduced_state_emulator.m_star_local_iteration_response_20260919 import (
        _load_case, _run_exact_state, _state_for_trial, _state_from_mt, _write_json,
    )
    import payne_zero_atmosphere.runner as runner_module
    from payne_zero_atmosphere import molecular_equilibrium as me
    from scipy.interpolate import CubicSpline, PchipInterpolator

    table = me._HYDROGEN_MOLECULE_PARTITION_TABLE
    node_count = len(table)
    temperatures = np.r_[np.arange(100., 19901., 100.), np.linspace(100., 19900., 2001)]
    nodes = np.arange(1, node_count + 1) * 100.
    if args.scheme == 'pchip':
        reference = PchipInterpolator(nodes, table)(temperatures)
    elif args.scheme == 'cubic':
        reference = CubicSpline(nodes, table, bc_type='natural')(temperatures)
    else:
        reference = np.interp(temperatures, nodes, table)
    actual = np.array([me._interp_hydrogen_molecule_partition_compiled(t, table)
                       for t in temperatures])
    error = float(np.max(np.abs(actual - reference)))
    assert error < 1e-10, error
    assert actual.min() > 0.0, float(actual.min())
    if args.scheme != 'linear':
        assert str(root / f'code_override_{args.scheme}') in me.__file__, me.__file__
    _write_json(root / 'implementation.json', {
        'case': args.case, 'scheme': args.scheme, 'module': me.__file__,
        'cache': os.environ.get('NUMBA_CACHE_DIR'),
        'scipy_max_abs_error': error, 'min_partition_value': float(actual.min()),
        'table_nodes': node_count, 'runtime_clamp_K': [100.0, 19900.0],
    })

    case = _load_case(args.case)
    labels = case['labels']
    effective_temperature = case['target_temperature_K']
    control = (REPO / 'results/m_star_trial_comparison_20260920'
               / f'controls_{args.case}/arrays/{args.case.lower()}_alpha_0p0.npz')
    with np.load(control, allow_pickle=False) as data:
        state = _state_from_mt(labels, data['column_mass'], data['temperature'])

    trajectory = {
        'case': args.case, 'scheme': args.scheme,
        'effective_temperature_K': effective_temperature,
        'initial_control': str(control), 'deep_layers': DEEP_LAYERS,
        'metrics_definition': (
            'primary = mean|R_raw| over layers 67-79; secondary = max|R_raw| full, '
            'max|R_raw| deep, max|R_smoothed| deep; extension rule preregistered in '
            'the module docstring'),
        'iterations': [],
    }
    trajectory_path = root / 'trajectory.json'

    for iteration in range(args.iterations):
        run_id = f'{args.case.lower()}_{args.scheme}_it{iteration:02d}'
        capture = []
        original = runner_module.compute_convection

        def wrapped(**kwargs):
            result = original(**kwargs)
            capture.append((copy.deepcopy(kwargs), copy.deepcopy(result)))
            return result

        runner_module.compute_convection = wrapped
        try:
            record = _run_exact_state(
                atmosphere=state, labels=labels,
                effective_temperature=effective_temperature, run_id=run_id,
                indices=DEEP_LAYERS, result_root=root, resume=args.resume,
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
                'R_raw_full_max': float(np.max(r_raw)),
                'R_raw_full_l2': float(np.sqrt(np.mean(r_raw ** 2))),
                'R_raw_deep_max': float(np.max(r_raw[deep])),
                'R_raw_deep_mean_abs': float(np.mean(r_raw[deep])),
                'R_smoothed_deep_max': float(np.max(r_smoothed[deep])),
                'dT_max_abs_K': float(np.max(np.abs(temperature_out - temperature_in))),
                'temperature_deep_in_K': [float(v) for v in temperature_in[deep]],
                'temperature_deep_out_K': [float(v) for v in temperature_out[deep]],
            }
        if capture:
            kwargs, result = capture[0]
            target = float(kwargs['target_integrated_eddington_flux'])
            row['target_integrated_eddington_flux'] = target
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
        trajectory['iterations'].append(row)
        _write_json(trajectory_path, trajectory)
        print(
            f'[{run_id}] deep_mean_raw={row["R_raw_deep_mean_abs"]:.4f} '
            f'full_max_raw={row["R_raw_full_max"]:.4f} '
            f'deep_max_smooth={row["R_smoothed_deep_max"]:.4f} '
            f'dTmax={row["dT_max_abs_K"]:.3f}', flush=True)
        state = _state_for_trial(labels, record)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
