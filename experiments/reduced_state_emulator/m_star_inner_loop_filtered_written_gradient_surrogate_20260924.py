"""Surrogate test of a 1-2-1 filtered written gradient in the convective inner loop.

Protocol: ``notes/m_star_inner_loop_filtered_written_gradient_surrogate_preregistration_20260924.md``.

Two surrogates of eight inner passes on the real grid and stencils:

* S-3850: the g4.75 3850 K warm-start continuation captures (rounds 4 and 8).
  Column mass, total pressure, frozen mask, H_rad and the target flux come from
  the capture; the adiabatic gradient and the smoothed convective flux respond
  to temperature through per-layer fits to the capture's nine pass states.
* S-D: the 2026-09-23 D-plat surrogate (adiabatic gradient, EOS and opacity
  fixed; flux law calibrated on the input state).

Three update rules on correctable layers: W (current written-gradient
correction), W121 (the written gradient passed through a 1-2-1 filter first)
and D (direct write).  Writes ``surrogate.json`` to the result directory.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from payne_zero_atmosphere.radiative_transfer import differentiate_on_depth_grid  # noqa: E402

CAPTURES = REPO / 'results/m_star_inner_loop_nonstationary_capture_20260924'
S3850_INPUTS = ('t3850_cont_primary_r04', 't3850_cont_primary_r08')
D_CAPTURE = REPO / 'results/m_star_s3_stage_capture_20260922/d/stages.json'
D_START_NPZ = REPO / 'results/m_star_h2_inner_loop_s3_20260922/d_pchip_s3/arrays/d_pchip_s3_it09.npz'
OUT = REPO / 'results/m_star_inner_loop_filtered_written_gradient_surrogate_20260924'
WINDOWS = {'L45_51': (45, 52), 'L52_58': (52, 59), 'L59_65': (59, 66), 'L66_72': (66, 73)}
D_DEEP = slice(67, 80)
PASSES = 8
SUPERADIABATIC_CAP = 2.0
TEMPERATURE_RELATIVE_CAP = 0.15
RULES = ('W', 'W121', 'D')


def alternating_amplitude(values: np.ndarray, lo: int, hi: int) -> float:
    index = np.arange(lo, hi)
    residual = values[index] - 0.5 * (values[index - 1] + values[index + 1])
    return float(np.mean(residual * (-1.0) ** index) / 2.0)


def filter_121(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    out[1:-1] = 0.25 * values[:-2] + 0.5 * values[1:-1] + 0.25 * values[2:]
    return out


class Surrogate:
    """Eight inner passes with the production stencils and a local MLT law."""

    def __init__(self, *, column_mass, pressure, gravity, t_input, mask, required,
                 adiabatic_input, adiabatic_slope, flux_coefficient, flux_temperature_exponent):
        self.column_mass = column_mass
        self.pressure = pressure
        self.gravity = gravity
        self.t_input = t_input
        self.mask = mask
        self.correctable = mask & np.append(mask[1:], True)
        self.required = required
        self.adiabatic_input = adiabatic_input
        self.adiabatic_slope = adiabatic_slope
        self.flux_coefficient = flux_coefficient
        self.flux_temperature_exponent = flux_temperature_exponent

    def read(self, temperature):
        return self.pressure / (temperature * self.gravity) * differentiate_on_depth_grid(
            self.column_mass, temperature)

    def written(self, temperature):
        gradient = np.zeros_like(temperature)
        gradient[1:] = np.log(temperature[1:] / temperature[:-1]) / np.log(
            self.pressure[1:] / self.pressure[:-1])
        gradient[0] = gradient[1]
        return gradient

    def adiabatic(self, temperature):
        return self.adiabatic_input + self.adiabatic_slope * np.log(temperature / self.t_input)

    def flux(self, temperature):
        delta = np.maximum(self.read(temperature) - self.adiabatic(temperature), 0.0)
        scale = np.exp(self.flux_temperature_exponent * np.log(temperature / self.t_input))
        return self.flux_coefficient * scale * delta ** 1.5

    def integrate(self, temperature, nabla):
        out = temperature.copy()
        for index in range(1, temperature.size):
            if not self.mask[index]:
                continue
            reference = out[index - 1] if self.mask[index - 1] else temperature[index - 1]
            trial = reference * np.exp(nabla[index] * np.log(self.pressure[index] / self.pressure[index - 1]))
            lo = temperature[index] * (1.0 - TEMPERATURE_RELATIVE_CAP)
            hi = temperature[index] * (1.0 + TEMPERATURE_RELATIVE_CAP)
            out[index] = min(max(trial, lo), hi)
        out[0] = temperature[0]
        return out

    def one_pass(self, temperature, rule, written_filter=None):
        read = self.read(temperature)
        adiabatic = self.adiabatic(temperature)
        flux = self.flux(temperature)
        delta = read - adiabatic
        trial = np.zeros_like(temperature)
        for layer in np.flatnonzero(self.mask):
            required = max(float(self.required[layer]), 0.0)
            current_flux = max(float(flux[layer]), 0.0)
            current_delta = max(float(delta[layer]), 0.0)
            if required <= 0.0:
                trial[layer] = 0.0
            elif current_flux <= 0.0 or current_delta <= 0.0:
                trial[layer] = float(np.clip(max(current_delta, 1.0e-6), 0.0, SUPERADIABATIC_CAP))
            else:
                trial[layer] = float(np.clip(
                    current_delta * (required / current_flux) ** (1.0 / 1.5), 0.0, SUPERADIABATIC_CAP))
        target = adiabatic + trial
        nabla = read.copy()
        if rule == 'D':
            nabla[self.mask] = target[self.mask]
        else:
            written = self.written(temperature)
            if rule == 'W121':
                written = filter_121(written)
            if written_filter is not None:
                written = written_filter(written, self.mask)
            update = np.where(self.correctable, written + (target - read), target)
            nabla[self.mask] = update[self.mask]
        return self.integrate(temperature, nabla)

    def run(self, rule, passes=PASSES, written_filter=None):
        temperature = self.t_input.copy()
        for _ in range(passes):
            temperature = self.one_pass(temperature, rule, written_filter)
        return temperature

    def flux_ratio_error(self, temperature, layers):
        required = self.required[layers]
        valid = required > 0.0
        ratio = self.flux(temperature)[layers][valid] / required[valid]
        return float(np.max(np.abs(ratio - 1.0)))


def _pass_state(stage):
    return {
        'temperature': np.asarray(stage['structure']['temperature_K'], dtype=np.float64),
        'adiabatic': np.asarray(stage['thermodynamics']['adiabatic_gradient'], dtype=np.float64),
        'delta': np.asarray(stage['thermodynamics']['superadiabatic_difference'], dtype=np.float64),
        'flux': np.asarray(stage['local_mismatch']['Hconv_smoothed'], dtype=np.float64),
        'read': np.asarray(stage['thermodynamics']['log_temperature_pressure_gradient'], dtype=np.float64),
    }


def build_s3850(capture: str) -> tuple[Surrogate, dict]:
    stages = json.loads((CAPTURES / capture / 'stages.json').read_text())['stages']
    first = stages['inner_pass_00']
    states = [_pass_state(stages[f'inner_pass_{k:02d}']) for k in range(PASSES + 1)]
    t_input = states[0]['temperature']
    moved = np.zeros(t_input.size, dtype=bool)
    for state in states[1:]:
        moved |= state['temperature'] != t_input
    x = np.stack([np.log(s['temperature'] / t_input) for s in states])
    adiabatic_change = np.stack([s['adiabatic'] - states[0]['adiabatic'] for s in states])
    denominator = np.maximum(np.sum(x * x, axis=0), 1.0e-300)
    slope = np.sum(x * adiabatic_change, axis=0) / denominator
    fit_residual = adiabatic_change - slope * x
    flux_coefficient = np.zeros(t_input.size)
    flux_exponent = np.zeros(t_input.size)
    for layer in np.flatnonzero(moved):
        flux = np.array([s['flux'][layer] for s in states])
        delta = np.array([s['delta'][layer] for s in states])
        ok = (flux > 0.0) & (delta > 0.0)
        if ok.sum() < 3:
            continue
        y = np.log(flux[ok]) - 1.5 * np.log(delta[ok])
        design = np.stack([np.ones(ok.sum()), x[ok, layer]], axis=1)
        (log_c, b), *_ = np.linalg.lstsq(design, y, rcond=None)
        flux_coefficient[layer] = np.exp(log_c)
        flux_exponent[layer] = b
    mismatch = first['local_mismatch']
    required = np.maximum(float(mismatch['target_integrated_eddington_flux'])
                          - np.asarray(mismatch['Hrad'], dtype=np.float64), 0.0)
    input_stage = stages['iteration_input']
    surrogate = Surrogate(
        column_mass=np.asarray(input_stage['structure']['column_mass_g_cm2'], dtype=np.float64),
        pressure=np.asarray(first['structure']['total_pressure_dyn_cm2'], dtype=np.float64),
        gravity=10.0 ** 4.75,
        t_input=t_input,
        mask=moved,
        required=required,
        adiabatic_input=states[0]['adiabatic'],
        adiabatic_slope=slope,
        flux_coefficient=flux_coefficient,
        flux_temperature_exponent=flux_exponent,
    )
    exit_temperature = np.asarray(stages['inner_loop_exit']['structure']['temperature_K'], dtype=np.float64)
    read_check = float(np.max(np.abs(surrogate.read(t_input)[moved] - states[0]['read'][moved])))
    calibration = {
        'read_reproduction_max_abs': read_check,
        'adiabatic_fit_max_abs_residual': float(np.max(np.abs(fit_residual[:, moved]))),
        'adiabatic_change_max_abs': float(np.max(np.abs(adiabatic_change[:, moved]))),
        'adiabatic_slope_range': [float(slope[moved].min()), float(slope[moved].max())],
        'flux_temperature_exponent_range': [float(flux_exponent[moved].min()), float(flux_exponent[moved].max())],
        'mask': [int(moved.nonzero()[0][0]), int(moved.nonzero()[0][-1])],
    }
    return surrogate, {'exit_temperature': exit_temperature, 'calibration': calibration}


def build_sd() -> Surrogate:
    stages = json.loads(D_CAPTURE.read_text())['stages']
    with np.load(D_START_NPZ, allow_pickle=False) as data:
        column_mass = np.asarray(data['column_mass'], dtype=np.float64)
    stage = stages['iteration_input']
    t_input = np.asarray(stage['structure']['temperature_K'], dtype=np.float64)
    pressure = np.asarray(stage['structure']['total_pressure_dyn_cm2'], dtype=np.float64)
    adiabatic = np.asarray(stage['thermodynamics']['adiabatic_gradient'], dtype=np.float64)
    mismatch = stage['local_mismatch']
    required = np.maximum(float(mismatch['target_integrated_eddington_flux'])
                          - np.asarray(mismatch['Hrad'], dtype=np.float64), 0.0)
    flux_input = np.asarray(mismatch['Hconv_smoothed'], dtype=np.float64)
    gravity = 10.0 ** 4.5
    read_input = pressure / (t_input * gravity) * differentiate_on_depth_grid(column_mass, t_input)
    delta_input = read_input - adiabatic
    mask = delta_input > 0.0
    mask[:3] = False
    coefficient = np.where(mask, flux_input / np.maximum(delta_input, 1.0e-12) ** 1.5, 0.0)
    return Surrogate(
        column_mass=column_mass, pressure=pressure, gravity=gravity, t_input=t_input, mask=mask,
        required=required, adiabatic_input=adiabatic, adiabatic_slope=np.zeros_like(adiabatic),
        flux_coefficient=coefficient, flux_temperature_exponent=np.zeros_like(adiabatic),
    )


def main() -> int:
    out: dict = {'s3850': {}, 'sd': {}}
    validity = True
    m1 = True
    m2 = True
    for capture in S3850_INPUTS:
        surrogate, extra = build_s3850(capture)
        mask = surrogate.mask
        log_input = np.log(surrogate.t_input)
        recorded_dt = extra['exit_temperature'] - surrogate.t_input
        finals = {rule: surrogate.run(rule) for rule in RULES}
        dt_w = finals['W'] - surrogate.t_input
        v1_error = float(np.max(np.abs(dt_w[mask] - recorded_dt[mask])) / np.max(np.abs(recorded_dt[mask])))
        amplitudes = {
            name: {
                'input': alternating_amplitude(log_input, lo, hi),
                'recorded_exit': alternating_amplitude(np.log(extra['exit_temperature']), lo, hi),
                **{rule: alternating_amplitude(np.log(finals[rule]), lo, hi) for rule in RULES},
            }
            for name, (lo, hi) in WINDOWS.items()
        }
        v2_ok = all(abs(a['W'] - a['recorded_exit']) <= 0.25 * abs(a['input']) for a in amplitudes.values())
        v1_ok = v1_error <= 0.20
        m1_ok = all(abs(a['W121']) <= 0.2 * abs(a['input']) for a in amplitudes.values())
        flux_error = {rule: surrogate.flux_ratio_error(finals[rule], mask) for rule in RULES}
        m2_ok = flux_error['W121'] <= flux_error['W'] + 0.05
        validity &= v1_ok and v2_ok
        m1 &= m1_ok
        m2 &= m2_ok
        out['s3850'][capture] = {
            'calibration': extra['calibration'],
            'V1_relative_max_error': v1_error, 'V1': v1_ok, 'V2': v2_ok,
            'alternating_amplitude': amplitudes,
            'M1': m1_ok,
            'mask_max_abs_flux_over_required_minus_1': flux_error,
            'M2': m2_ok,
            'delta_T_max_abs': {rule: float(np.max(np.abs(finals[rule][mask] - surrogate.t_input[mask]))) for rule in RULES},
        }
    sd = build_sd()
    finals = {rule: sd.run(rule) for rule in RULES}
    flux_error = {rule: sd.flux_ratio_error(finals[rule], np.arange(80)[D_DEEP]) for rule in RULES}
    dt_w = finals['W'][D_DEEP] - sd.t_input[D_DEEP]
    m3_error = float(np.max(np.abs(finals['W121'][D_DEEP] - finals['W'][D_DEEP])) / np.max(np.abs(dt_w)))
    m2_sd = flux_error['W121'] <= flux_error['W'] + 0.05
    m3 = m3_error <= 0.05
    m2 &= m2_sd
    out['sd'] = {
        'deep_max_abs_flux_over_required_minus_1': flux_error,
        'M2': m2_sd,
        'M3_relative_max_temperature_difference': m3_error,
        'M3': m3,
        'delta_T_K_L71': {rule: float(finals[rule][71] - sd.t_input[71]) for rule in RULES},
        'delta_T_K_L79': {rule: float(finals[rule][79] - sd.t_input[79]) for rule in RULES},
    }
    out['criteria'] = {'V': bool(validity), 'M1': bool(m1), 'M2': bool(m2), 'M3': bool(m3)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'surrogate.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
