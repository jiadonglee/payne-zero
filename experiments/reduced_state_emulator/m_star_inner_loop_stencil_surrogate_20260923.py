"""Surrogate of one D-plat inner loop: real grid and read stencil, local MLT law.

The surrogate keeps the round-input column mass, total pressure, temperature,
∇_ad and frozen H_rad of the 2026-09-22 D stage capture, and reads ∇ exactly
as ``compute_convection`` does (``P/(T g) · dT/dm`` with
``differentiate_on_depth_grid``).  The convective flux follows the local law
``F_i = F_in_i (δ_i/δ_in_i)^1.5`` calibrated on the input state; ∇_ad, the
EOS and the opacity do not respond to temperature.  It isolates the gradient
stencil: the S3 row must reproduce the recorded capture, the other rows
predict the written-gradient correction before any solver run.

Output: ``results/m_star_inner_loop_written_gradient_20260923/surrogate.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from payne_zero_atmosphere.radiative_transfer import differentiate_on_depth_grid  # noqa: E402

CAPTURE = REPO / 'results/m_star_s3_stage_capture_20260922/d/stages.json'
START_NPZ = REPO / 'results/m_star_h2_inner_loop_s3_20260922/d_pchip_s3/arrays/d_pchip_s3_it09.npz'
OUTPUT = REPO / 'results/m_star_inner_loop_written_gradient_20260923/surrogate.json'
SURFACE_GRAVITY = 10.0 ** 4.5
DEEP = slice(67, 80)
RECORDED_S3_PASS8_RATIO_L71_73 = (1.730, 1.794, 1.952)
RECORDED_S3_PASS1_DT_L71 = 8.55


def main() -> int:
    stages = json.loads(CAPTURE.read_text())['stages']
    with np.load(START_NPZ, allow_pickle=False) as data:
        column_mass = np.asarray(data['column_mass'], dtype=np.float64)
    stage = stages['iteration_input']
    t_input = np.asarray(stage['structure']['temperature_K'], dtype=np.float64)
    pressure = np.asarray(stage['structure']['total_pressure_dyn_cm2'], dtype=np.float64)
    nabla_ad = np.asarray(stage['thermodynamics']['adiabatic_gradient'], dtype=np.float64)
    mismatch = stage['local_mismatch']
    target = float(mismatch['target_integrated_eddington_flux'])
    required = np.maximum(target - np.asarray(mismatch['Hrad'], dtype=np.float64), 0.0)
    flux_input = np.asarray(mismatch['Hconv_smoothed'], dtype=np.float64)

    def read(temperature):
        return pressure / (temperature * SURFACE_GRAVITY) * differentiate_on_depth_grid(
            column_mass, temperature
        )

    def written(temperature):
        gradient = np.zeros_like(temperature)
        gradient[1:] = np.log(temperature[1:] / temperature[:-1]) / np.log(
            pressure[1:] / pressure[:-1]
        )
        gradient[0] = gradient[1]
        return gradient

    delta_input = read(t_input) - nabla_ad
    mask = delta_input > 0.0
    mask[:3] = False
    correctable = mask & np.append(mask[1:], True)
    coefficient = np.where(mask, flux_input / np.maximum(delta_input, 1.0e-12) ** 1.5, 0.0)

    def flux(temperature):
        return coefficient * np.maximum(read(temperature) - nabla_ad, 0.0) ** 1.5

    def integrate(temperature, nabla):
        out = temperature.copy()
        for index in range(1, temperature.size):
            if not mask[index]:
                continue
            reference = out[index - 1] if mask[index - 1] else temperature[index - 1]
            out[index] = reference * np.exp(
                nabla[index] * np.log(pressure[index] / pressure[index - 1])
            )
        return out

    def run(correct: bool, passes: int):
        temperature = t_input.copy()
        for _ in range(passes):
            read_nabla = read(temperature)
            current_flux = flux(temperature)
            delta = read_nabla - nabla_ad
            active = mask & (current_flux > 0.0) & (delta > 0.0)
            trial = np.where(
                active,
                delta * (required / np.where(current_flux > 0.0, current_flux, 1.0)) ** (2.0 / 3.0),
                0.0,
            )
            target_nabla = nabla_ad + trial
            nabla = read_nabla.copy()
            if correct:
                nabla[mask] = np.where(
                    correctable, written(temperature) + (target_nabla - read_nabla), target_nabla
                )[mask]
            else:
                nabla[mask] = target_nabla[mask]
            temperature = integrate(temperature, nabla)
        ratio = flux(temperature) / np.where(required > 0.0, required, np.nan)
        return temperature, ratio

    rows = []
    for label, correct in (('S3', False), ('S3w', True)):
        for passes in (1, 2, 4, 8, 16):
            temperature, ratio = run(correct, passes)
            rows.append({
                'arm': label,
                'passes': passes,
                'deep_max_abs_flux_over_required_minus_1': float(np.nanmax(np.abs(ratio[DEEP] - 1.0))),
                'deep_mean_abs_flux_over_required_minus_1': float(np.nanmean(np.abs(ratio[DEEP] - 1.0))),
                'delta_T_K_L71': float(temperature[71] - t_input[71]),
                'delta_T_K_L79': float(temperature[79] - t_input[79]),
                'delta_T_K_deep_max_abs': float(np.max(np.abs(temperature[DEEP] - t_input[DEEP]))),
            })
    s3_temperature_1, _ = run(False, 1)
    _, s3_ratio_8 = run(False, 8)
    check = {
        'surrogate_S3_pass8_flux_over_required_L71_73': [float(v) for v in s3_ratio_8[71:74]],
        'recorded_S3_pass8_flux_over_required_L71_73': list(RECORDED_S3_PASS8_RATIO_L71_73),
        'surrogate_S3_pass1_delta_T_K_L71': float(s3_temperature_1[71] - t_input[71]),
        'recorded_S3_pass1_delta_T_K_L71': RECORDED_S3_PASS1_DT_L71,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        'capture': str(CAPTURE.relative_to(REPO)),
        'model': 'local F ∝ δ^1.5 calibrated on the input state; ∇_ad, EOS, opacity fixed',
        'n_masked': int(np.count_nonzero(mask)),
        'reproduction_check': check,
        'rows': rows,
    }, indent=2) + '\n')
    for row in rows:
        print(row)
    print(check)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
