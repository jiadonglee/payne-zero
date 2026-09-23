"""S3 inner-loop stage capture for one solver round, plus a state-only evaluator.

One invocation runs exactly one S3 round (8 inner passes, frozen mask, pchip
overlay) on a start state and writes one JSON block per stage of that round.
It never launches remote work and never touches ``payne_zero_atmosphere/``:
every capture point is an experiment-side wrapper around a module-level name
in ``payne_zero_atmosphere.runner``, installed and restored per round.

Stage list (names are the code facts; labels map them to the analysis list)
--------------------------------------------------------------------------
The actual solver order for one S3 iteration is:

    prepare/population/opacity/transfer (input state)
      -> finalize_transfer_state:
           convection on input state            runner.py:1607
           convection inner loop                runner.py:1750
             seed physics call                  convection_inner_loop.py:250
             passes 1..8, physics after each    convection_inner_loop.py:262-302
             mask release, final mismatch       convection_inner_loop.py:304-315
           convection recompute at inner T      runner.py:1763-1833
           global temperature correction        runner.py:1877-1924
      -> remap to standard grid                 runner.py:607 -> :1940-2087
      -> after_iteration_hook                   runner.py:2406-2409

Recorded stages, in that execution order:

  stage_id                        analysis label
  ------------------------------  ------------------------------------------------
  iteration_input                 stage1: before the inner loop (input state)
  inner_pass_00                   stage2: physics seed at the input temperature
  inner_pass_01 .. inner_pass_08  stage2: after inner pass 1..8
  inner_loop_exit                 stage3: loop end, mask re-judged/released
  post_inner_recompute            stage3b: runner convection re-run at inner T
  correction_native_grid          stage4: after global correction, before remap
  standard_grid_remap             stage5: after remap (the hook's step.remapped)

Points where the code differs from a naive stage list:

* The physics callback fires 9 times, not 8: ``run_convection_zone_inner_loop``
  evaluates physics once on the input temperature before pass 1
  (convection_inner_loop.py:250) and once after every pass
  (convection_inner_loop.py:287).  The seed call is recorded as
  ``inner_pass_00``; it defines ``convective_mask_initial``, which is the
  frozen mask for all 8 passes.
* Stage 4 is distinguishable from stage 5: the correction runs inside
  ``finalize_transfer_state`` (runner.py:1877) and the remap is the next
  function call (``remap_finalized_iteration_state``, runner.py:1940, invoked
  at run_single_iteration:607).  No production reordering was done.
* Between the inner loop and the correction the runner recomputes convection
  at the inner-loop temperature with a fresh finite-difference sample
  (runner.py:1763-1833, compute_convection at :1782, seed ``inner_seed_base +
  9000``).  This is the MLT state the correction actually consumes; it is
  recorded separately as ``post_inner_recompute``.

Attachment-point map (file:line at the time of writing)
-------------------------------------------------------
stage1  runner.py:1607 -- the input-state ``compute_convection`` call inside
        ``finalize_transfer_state`` is intercepted by wrapping
        ``runner_module.compute_convection`` (kwargs carry temperature_k,
        column_mass, gas_pressure, total_pressure, rosseland_optical_depth;
        the result carries all ConvectionResult fields).  H_rad for its
        residual is ``temperature_correction.integrated_eddington_flux``
        (this iteration's transfer accumulator), captured from
        ``finalization.transfer_accumulation``.
stage2  The ``PhysicsCallback`` ``_physics_at_temperature`` is a closure built
        at runner.py:1672-1747 and passed to the loop at runner.py:1750.  It
        cannot be reached directly, so ``runner_module.run_convection_zone_inner_loop``
        (imported into the runner namespace at runner.py:41-45) is wrapped and
        its ``physics`` argument is wrapped: every ``ConvectivePhysicsSnapshot``
        is deep-copied together with the trial temperature that produced it.
        Snapshot fields: convection_inner_loop.py:39-50; runner adds
        ``heat_capacity``, ``pressure_scale_height``, ``raw_convective_flux``
        and the density derivative as ``extras`` (runner.py:1739-1746).
        Gaps: the snapshot carries no temperature (taken from the callback
        argument), no column_mass and no gas_pressure (both are held fixed
        during the loop; column_mass is recorded once in stage1, gas_pressure
        is null in pass blocks and flagged as a gap).
stage3  The return value ``ConvectiveInnerLoopResult``
        (convection_inner_loop.py:316-329) is captured by the same wrapper:
        released mask (``:309-315``), final mismatch, pass_records.  The
        following post-inner recompute (runner.py:1782) is tagged by the
        ``compute_convection`` wrapper as ``post_inner_recompute``.
stage4  ``runner_module.remap_finalized_iteration_state`` is wrapped; on entry
        the ``finalization`` argument holds ``temperature_correction_result``
        (temperature_correction.py:40-55: corrected temperature and column
        mass on the native grid, raw/damped correction, smoothed convective
        flux, flux_error_percent) plus the native Rosseland depth grid and
        ``convection_inner_loop_diagnostics``.
stage5  The wrapper's return value is the ``IterationRemap``; equivalently the
        production ``after_iteration_hook`` (runner.py:2406-2409) sees it as
        ``step.remapped``.  ``_physics_rows`` semantics
        (m_star_local_iteration_response_20260919.py:162-215) are applied to
        the lightweight pair (remapped, transfer) -- the same objects the
        production hook receives.
Timing  ``convection_inner_loop_*`` keys enter ``step.timing`` at
        runner.py:664-666 (merged from
        ``remapped.finalization.convection_inner_loop_diagnostics``, built at
        runner.py:1837-1875); the summary copies them verbatim.

Residual semantics (verified against recorded anchors)
------------------------------------------------------
Within one round there is exactly one transfer evaluation, run on the round's
input state.  Every residual below therefore uses the frozen radiative flux
of that transfer: ``radiation_field`` is ``"frozen"`` on every residual field.
A ``"current"`` field (full transfer re-run on a given structure) exists only
through ``evaluate_state_only`` on an explicit state.

Every residual block additionally carries ``state_provenance``: which
structure the convective term was evaluated on.  This is required because the
S3 trajectory's R labels are not pure input-state residuals -- d_pchip_s3_it00
and d_pchip_it00 share a bit-identical input state (the D control state) yet
report deep-window max|R_smoothed| 1.4844 vs 0.8081, because with the inner
loop on, ``finalization.convection_result`` is the post-inner recompute at the
inner-loop temperature (runner.py:1782-1833 overwrites the convection_result
stored at runner.py:1934).  The per-stage blocks here name exactly which
convection evaluation they use:

* ``iteration_input`` / ``inner_pass_*``: H_conv at that stage's own
  temperature (frozen H_rad from the input transfer).
* ``inner_loop_exit`` / ``post_inner_recompute`` / ``correction_native_grid`` /
  ``standard_grid_remap``: H_conv_raw from the post-inner recompute at the
  inner-loop output temperature; H_rad still the input transfer's.

``_physics_rows`` semantic check (asked explicitly): with the inner loop off,
``finalization.convection_result`` is the runner.py:1607 evaluation of the
input temperature, and ``correction.convective_flux`` is the 1-2-1 smoothing
of that same field (temperature_correction.py:525-546, returned at :947).
R_raw/R_smoothed are then the input state's residual, never the next state's:
the transfer is never re-run on the remapped output inside the iteration.
Anchor evidence: d_pchip it00 (input = D control state) reports deep mean
|R_raw| 0.4349 and deep max |R_smoothed| 0.8081, the values all downstream
analysis attributes to the control input state; the same input state through
S3 (inner loop on) reports 0.9141/1.4844, so the change tracks the convection
evaluation temperature of the same round, and it01 (fed the it00 output)
reports yet different values -- the rows are not the successor state's.
Consequence honoured here: ``evaluate_state_only`` defaults to
``inner_loop_passes=0`` so its rows are the pure input-state residual.

State-only evaluation
---------------------
``evaluate_state_only(state, labels, teff, result_root, run_id)`` runs one
``iterations_per_trial=1`` round of full physics (EOS + opacity + transfer +
convection, opacity lagging off, inner loop off by default) on an explicit
state and returns that state's R_raw/R_smoothed.  The returned correction is
never fed into any trajectory.  Structure fields come from the state objects
(reconstructed atmosphere, runtime_state, convection result), never from the
trajectory npz files, which lack gas_pressure (all 20 S3 arrays) and
total_pressure (all 68 arrays).

Verification procedure (documented; intentionally not executed today):
  1. python experiments/reduced_state_emulator/m_star_s3_stage_capture_20260922.py \
         --evaluate-only --start-npz \
         results/m_star_trial_comparison_20260920/controls_D/arrays/d_alpha_0p0.npz
  2. Read ``records/evaluate_d_control_state.json``.
  3. Expected: ``R_smoothed_deep_max == 0.8081`` and
     ``R_raw_deep_mean_abs == 0.4349`` within float roundoff (the d_pchip it00
     anchors produced from the same control state with the same pchip overlay
     and inner loop off).  A mismatch means the evaluator's physics flags do
     not reproduce the paired-iteration arm and must be reconciled before any
     use.

Field naming follows the scaffold reserved in
results/m_star_trajectory_index_20260922/index.md: each stage block carries
``temperature_correction``, ``local_mismatch``, ``raw_total_flux_residual``
and ``radiative_field_annotation``, alongside the richer structure /
thermodynamics / energy_transport / update_action blocks.

Resume: ``--resume`` skips the round only when a completed capture
(``standard_grid_remap`` present and ``stages_summary.json`` written) exists.
A partial stages.json comes from an interrupted process; the solver cannot
execute part of a round, so the round is re-run and stages are rewritten in
order as they complete.  One invocation = one round = one interruptible unit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO / 'results/m_star_s3_stage_capture_20260922'
DEEP_LAYERS = list(range(67, 80))
INNER_PASSES = 8
CONVECTION_RESULT_FIELDS = (
    'geometric_depth_below_surface_km',
    'logarithmic_temperature_pressure_gradient',
    'heat_capacity',
    'log_density_temperature_derivative_at_constant_total_pressure',
    'sound_speed',
    'adiabatic_gradient',
    'pressure_scale_height',
    'convective_flux',
    'convective_velocity',
    'raw_convective_flux',
    'overshoot_convective_flux',
)

DEFAULT_START_NPZ = {
    'D': REPO / 'results/m_star_h2_inner_loop_s3_20260922/d_pchip_s3/arrays/d_pchip_s3_it09.npz',
    'A': REPO / 'results/m_star_h2_inner_loop_s3_20260922/a_pchip_s3/arrays/a_pchip_s3_it09.npz',
}
CONTROL_STATE_NPZ = {
    'D': REPO / 'results/m_star_trial_comparison_20260920/controls_D/arrays/d_alpha_0p0.npz',
    'A': REPO / 'results/m_star_trial_comparison_20260920/controls_A/arrays/a_alpha_0p0.npz',
}

STAGE_ID_INPUT = 'iteration_input'
STAGE_ID_EXIT = 'inner_loop_exit'
STAGE_ID_POST_INNER = 'post_inner_recompute'
STAGE_ID_CORRECTION = 'correction_native_grid'
STAGE_ID_REMAP = 'standard_grid_remap'
TERMINAL_STAGE_ID = STAGE_ID_REMAP

FROZEN = 'frozen'
CURRENT = 'current'


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + '\n')


def _read_stages(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {'stages': {}}
    return json.loads(path.read_text())


def _append_stage(stage_path: Path, record: dict[str, Any]) -> None:
    """Persist one stage block immediately; one array element per line."""

    payload = _read_stages(stage_path)
    payload.setdefault('stages', {})[record['stage_id']] = record
    _write_json(stage_path, payload)


def _full(values) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64)]


def _deep_window(fields: dict[str, list[float] | None]) -> dict[str, dict[str, float]]:
    block: dict[str, dict[str, float]] = {}
    for layer in DEEP_LAYERS:
        row: dict[str, float] = {}
        for name, values in fields.items():
            if values is None:
                row[name] = None
            else:
                row[name] = float(values[layer])
        block[str(layer)] = row
    return block


def _residual_field(values, radiation_field: str, state_provenance: str, definition: str):
    return {
        'radiation_field': radiation_field,
        'state_provenance': state_provenance,
        'definition': definition,
        'full_depth': _full(values),
        'deep_window': {
            str(layer): float(np.asarray(values)[layer]) for layer in DEEP_LAYERS
        },
    }


def _null_field(note: str):
    return {'value': None, 'gap_note': note}


def _convection_result_dict(result) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(result, name), dtype=np.float64).copy()
        for name in CONVECTION_RESULT_FIELDS
    }


def _snapshot_dict(snapshot) -> dict[str, Any]:
    return {
        'convective_flux': np.asarray(snapshot.convective_flux, dtype=np.float64).copy(),
        'logarithmic_gradient': np.asarray(
            snapshot.logarithmic_gradient, dtype=np.float64
        ).copy(),
        'adiabatic_gradient': np.asarray(
            snapshot.adiabatic_gradient, dtype=np.float64
        ).copy(),
        'total_pressure': np.asarray(snapshot.total_pressure, dtype=np.float64).copy(),
        'extras': {
            key: np.asarray(value, dtype=np.float64).copy()
            for key, value in dict(snapshot.extras).items()
        },
    }


def _transport_block(
    *,
    h_rad: np.ndarray,
    h_conv_raw: np.ndarray | None,
    h_conv_smoothed: np.ndarray | None,
    target_h: float,
    provenance_raw: str,
    provenance_smoothed: str,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        'target_integrated_eddington_flux_H': float(target_h),
        'H_rad': _residual_field(
            h_rad, FROZEN, 'iteration_input (transfer of this round input state)',
            'integrated Eddington flux H of this round transfer; frozen for every '
            'residual inside the round',
        ),
    }
    if h_conv_raw is not None:
        r_raw = (h_rad + h_conv_raw - target_h) / max(target_h, 1.0e-300)
        block['H_conv_raw'] = _residual_field(
            h_conv_raw, FROZEN, provenance_raw,
            'raw MLT convective flux H (pre-smoothing), evaluated at the '
            'provenance state temperature',
        )
        block['R_raw'] = _residual_field(
            r_raw, FROZEN, provenance_raw,
            '(H_rad + H_conv_raw - H_target) / H_target',
        )
    else:
        block['H_conv_raw'] = _null_field('no convection evaluation recorded')
        block['R_raw'] = _null_field('requires H_conv_raw')
    if h_conv_smoothed is not None:
        r_smooth = (h_rad + h_conv_smoothed - target_h) / max(target_h, 1.0e-300)
        block['H_conv_smoothed'] = _residual_field(
            h_conv_smoothed, FROZEN, provenance_smoothed,
            'smoothed convective flux H (the flux the loop and correction consume)',
        )
        block['R_smoothed'] = _residual_field(
            r_smooth, FROZEN, provenance_smoothed,
            '(H_rad + H_conv_smoothed - H_target) / H_target',
        )
    else:
        block['H_conv_smoothed'] = _null_field('no smoothed convective flux recorded')
        block['R_smoothed'] = _null_field('requires H_conv_smoothed')
    return block


def _radiative_annotation(
    *,
    convection_on: np.ndarray | None,
    n_masked: int | None,
    mask_frozen: bool | None,
    opacity_lagging: bool,
    runner_r_smoothed: np.ndarray | None,
) -> dict[str, Any]:
    return {
        'convection_on': None if convection_on is None else _full(convection_on.astype(float)),
        'convection_on_definition': 'raw MLT convective flux > 0',
        'n_masked': n_masked,
        'mask_frozen': mask_frozen,
        'opacity_lagging': bool(opacity_lagging),
        'runner_R_smoothed': (
            None if runner_r_smoothed is None else _full(runner_r_smoothed)
        ),
    }


class _RoundRecorder:
    """Collects the wrapped solver calls of one round and writes stage blocks.

    The public ``observe_*``/``enter_*``/``exit_*`` methods take plain dicts
    and arrays, so the recording path can be exercised without the solver.
    """

    def __init__(
        self,
        *,
        case: str,
        run_root: Path,
        effective_temperature: float,
        opacity_lagging: bool = False,
        input_npz: str | None = None,
    ):
        self.case = case
        self.run_root = run_root
        self.stage_path = run_root / 'stages.json'
        self.effective_temperature = float(effective_temperature)
        self.target_h = 5.6697e-5 / 12.5664 * float(effective_temperature) ** 4
        self.opacity_lagging = bool(opacity_lagging)
        self.input_npz = input_npz
        self.pre_inner = None            # (kwargs, result dict) from runner:1607
        self.pass_stash: list[dict[str, Any]] = []   # per compute_convection call
        self.inner_passes: list[dict[str, Any]] = []  # per inner physics callback
        self.inner_result = None         # ConvectiveInnerLoopResult fields
        self.inner_h_rad = None
        self.inner_config = None         # ConvectiveInnerLoopConfig fields
        self.post_inner = None           # (kwargs, result dict) from runner:1782
        self.finalization = None         # IterationFinalization capture
        self.remap_result = None         # IterationRemap capture
        self.timing: dict[str, Any] = {}
        self.input_crosscheck = None
        self.unexpected_convection_calls: list[str] = []
        self._inside_inner = False
        self._inner_done = False
        self._remap_entered = False
        self.surface_gravity_cgs: float | None = None
        self.started = time.perf_counter()

    # -- observation API (wired to runner module names in wire_round) -------

    def observe_convection_call(self, kwargs: dict, result) -> None:
        if self._inside_inner:
            self.pass_stash.append({'kwargs': _full_kwargs(kwargs),
                                    'result': _convection_result_dict(result)})
        elif self._inner_done and not self._remap_entered and self.post_inner is None:
            self.post_inner = {'kwargs': _full_kwargs(kwargs),
                               'result': _convection_result_dict(result)}
        elif not self._inner_done and self.pre_inner is None:
            self.pre_inner = {'kwargs': _full_kwargs(kwargs),
                              'result': _convection_result_dict(result)}
        else:
            self.unexpected_convection_calls.append(
                f'inside_inner={self._inside_inner} inner_done={self._inner_done} '
                f'remap_entered={self._remap_entered}'
            )

    def enter_inner_loop(self, *, radiative_eddington_flux, config) -> None:
        self.inner_h_rad = np.asarray(radiative_eddington_flux, dtype=np.float64).copy()
        self.inner_config = {
            'passes': int(config.passes),
            'freeze_mask_during_passes': bool(config.freeze_mask_during_passes),
            'release_mask_after': bool(config.release_mask_after),
            'temperature_relative_cap': float(config.temperature_relative_cap),
            'superadiabatic_cap': float(config.superadiabatic_cap),
            'zero_top_layer_count': int(config.zero_top_layer_count),
            'correct_written_gradient': bool(
                getattr(config, 'correct_written_gradient', False)
            ),
            'fill_interior_holes': bool(getattr(config, 'fill_interior_holes', False)),
            'constants_source': (
                'payne_zero_atmosphere/convection_inner_loop.py '
                'DEFAULT_TEMPERATURE_RELATIVE_CAP / DEFAULT_SUPERADIABATIC_CAP; '
                'the runner sets passes/freeze/release/correct_written_gradient'
            ),
        }
        self._inside_inner = True
        self._write_stage1()

    def observe_inner_snapshot(self, trial_temperature, snapshot) -> None:
        self.inner_passes.append({
            'trial_temperature': np.asarray(trial_temperature, dtype=np.float64).copy(),
            'snapshot': _snapshot_dict(snapshot),
        })
        # the update that produced this pass is fully determined by the
        # previous stored snapshot, so the stage block is written the moment
        # the loop's physics callback for this pass returns
        self._write_pass_stage(len(self.inner_passes) - 1)

    def exit_inner_loop(self, result) -> None:
        self._inside_inner = False
        self._inner_done = True
        self.inner_result = {
            'temperature': np.asarray(result.temperature, dtype=np.float64).copy(),
            'convective_flux': np.asarray(result.convective_flux, dtype=np.float64).copy(),
            'required_convective_flux': np.asarray(
                result.required_convective_flux, dtype=np.float64
            ).copy(),
            'local_energy_mismatch': np.asarray(
                result.local_energy_mismatch, dtype=np.float64
            ).copy(),
            'convective_mask_initial': np.asarray(
                result.convective_mask_initial, dtype=bool
            ).copy(),
            'convective_mask_final': np.asarray(
                result.convective_mask_final, dtype=bool
            ).copy(),
            'mask_was_frozen': bool(result.mask_was_frozen),
            'mask_was_released': bool(result.mask_was_released),
            'assigned_required_flux': bool(result.assigned_required_flux),
            'physics_evaluations': int(result.physics_evaluations),
            'pass_records': [dict(record) for record in result.pass_records],
            'extras': {
                key: np.asarray(value, dtype=np.float64).copy()
                for key, value in dict(result.extras).items()
            },
        }
        self._write_stage3()

    def enter_remap(self, finalization) -> None:
        self._remap_entered = True
        self.finalization = finalization
        self._write_stage_post_inner()
        self._write_stage4()

    def exit_remap(self, remap_result) -> None:
        self.remap_result = remap_result
        self._write_stage5()

    def finalize(self, *, timing: dict, setup_atmosphere=None) -> None:
        self.timing = dict(timing)
        if setup_atmosphere is not None and self.pre_inner is not None:
            self.input_crosscheck = {
                'setup_atmosphere_matches_pre_inner_temperature': bool(np.array_equal(
                    np.asarray(setup_atmosphere.temperature, dtype=np.float64),
                    self.pre_inner['kwargs']['temperature_k'],
                )),
                'setup_atmosphere_matches_pre_inner_column_mass': bool(np.array_equal(
                    np.asarray(setup_atmosphere.column_mass, dtype=np.float64),
                    self.pre_inner['kwargs']['column_mass'],
                )),
            }
        self._write_summary()

    # -- stage builders ------------------------------------------------------

    def _h_rad(self) -> np.ndarray:
        if self.inner_h_rad is not None:
            return self.inner_h_rad
        if self.finalization is not None:
            return np.asarray(
                self.finalization.transfer_accumulation.temperature_correction_state
                .integrated_eddington_flux,
                dtype=np.float64,
            ).copy()
        raise RuntimeError('no radiative flux captured for residuals')

    def _write_stage1(self) -> None:
        kwargs = self.pre_inner['kwargs']
        result = self.pre_inner['result']
        temperature = kwargs['temperature_k']
        nabla = result['logarithmic_temperature_pressure_gradient']
        nabla_ad = result['adiabatic_gradient']
        h_rad = self._h_rad()
        provenance = 'iteration_input (native grid, before the inner loop)'
        record = {
            'stage_id': STAGE_ID_INPUT,
            'label': 'stage1_before_inner_loop',
            'grid': 'native',
            'depth_coordinate': {
                'name': 'log10_rosseland_optical_depth',
                'values': _full(np.log10(kwargs['rosseland_optical_depth'])),
                'gradient_coordinate_note': (
                    'the inner loop integrates d ln T along d ln total_pressure; '
                    'total_pressure is recorded under structure'
                ),
            },
            'structure': {
                'temperature_K': _full(temperature),
                'column_mass_g_cm2': _full(kwargs['column_mass']),
                'gas_pressure_dyn_cm2': _full(kwargs['gas_pressure']),
                'total_pressure_dyn_cm2': _full(kwargs['total_pressure']),
                'mass_density_g_cm3': _full(kwargs['mass_density']),
            },
            'thermodynamics': {
                'heat_capacity': _full(result['heat_capacity']),
                'adiabatic_gradient': _full(nabla_ad),
                'log_temperature_pressure_gradient': _full(nabla),
                'superadiabatic_difference': _full(np.asarray(nabla) - np.asarray(nabla_ad)),
                'log_density_temperature_derivative_at_constant_total_pressure': _full(
                    result['log_density_temperature_derivative_at_constant_total_pressure']
                ),
            },
            'energy_transport': _transport_block(
                h_rad=h_rad,
                h_conv_raw=result['raw_convective_flux'],
                h_conv_smoothed=result['convective_flux'],
                target_h=self.target_h,
                provenance_raw=provenance,
                provenance_smoothed=provenance,
            ),
            'update_action': self._update_action_zero(temperature),
            'radiative_field_annotation': _radiative_annotation(
                convection_on=result['raw_convective_flux'] > 0.0,
                n_masked=int(np.count_nonzero(self._mask_from(
                    nabla, nabla_ad,
                ))),
                mask_frozen=None,
                opacity_lagging=self.opacity_lagging,
                runner_r_smoothed=None,
            ),
            'local_mismatch': self._local_mismatch_block(
                h_rad, result['raw_convective_flux'], result['convective_flux'],
            ),
            'temperature_correction': self._temperature_correction_block(
                np.zeros_like(temperature), None,
            ),
            'raw_total_flux_residual': self._raw_residual_block(
                h_rad, result['raw_convective_flux'], result['convective_flux'],
            ),
        }
        record['radiative_field_annotation']['mask_source'] = (
            're-derived with convective_mask_from_gradients from this stage '
            'gradients; cross-checked against the loop mask in inner_loop_exit'
        )
        _append_stage(self.stage_path, record)

    def _mask_from(self, nabla, nabla_ad) -> np.ndarray:
        from payne_zero_atmosphere.convection_inner_loop import (
            convective_mask_from_gradients,
        )

        zero_top = (
            self.inner_config['zero_top_layer_count']
            if self.inner_config is not None else 3
        )
        return convective_mask_from_gradients(
            np.asarray(nabla), np.asarray(nabla_ad), zero_top_layer_count=zero_top,
        )

    def _frozen_mask(self) -> np.ndarray:
        first = self.inner_passes[0]['snapshot']
        return self._mask_from(
            first['logarithmic_gradient'], first['adiabatic_gradient'],
        )

    def _rederive_pass_update(
        self, pass_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Replicate the pass update from the stored snapshots (no solver code).

        Returns (trial temperature, temperature-cap-touched, trial-delta
        cap-touched, rederivation_matches).  Uses the same imported pure
        functions as the loop, so an exact match validates the capture.
        """

        from payne_zero_atmosphere.convection_inner_loop import (
            _neighbour_superadiabatic_seed,
            apply_logarithmic_gradient_to_temperature,
            interior_layers_radiation_cannot_carry,
            superadiabatic_for_required_flux,
            written_logarithmic_gradient,
        )

        previous = self.inner_passes[pass_index - 1]
        current = self.inner_passes[pass_index]
        snapshot = previous['snapshot']
        schwarzschild = self._frozen_mask()
        h_rad = self.inner_h_rad
        filled = np.zeros_like(schwarzschild)
        if self.inner_config.get('fill_interior_holes'):
            seed_snapshot = self.inner_passes[0]['snapshot']
            filled = interior_layers_radiation_cannot_carry(
                mask=schwarzschild,
                logarithmic_gradient=seed_snapshot['logarithmic_gradient'],
                adiabatic_gradient=seed_snapshot['adiabatic_gradient'],
                radiative_eddington_flux=h_rad,
                target_eddington_flux=self.target_h,
            )
        mask = schwarzschild | filled
        required = np.maximum(self.target_h - h_rad, 0.0)
        cap = self.inner_config['superadiabatic_cap']
        nabla_ad = snapshot['adiabatic_gradient']
        superadiabatic = snapshot['logarithmic_gradient'] - nabla_ad
        trial_delta = np.zeros_like(nabla_ad)
        for layer in np.flatnonzero(mask):
            if filled[layer] and float(snapshot['convective_flux'][layer]) <= 0.0:
                trial_delta[layer] = min(
                    _neighbour_superadiabatic_seed(superadiabatic, schwarzschild, layer),
                    float(cap),
                )
                continue
            trial_delta[layer] = superadiabatic_for_required_flux(
                float(superadiabatic[layer]),
                float(snapshot['convective_flux'][layer]),
                float(required[layer]),
                cap=cap,
            )
        previous_temperature = previous['trial_temperature']
        read_nabla = snapshot['logarithmic_gradient']
        target_nabla = nabla_ad + trial_delta
        nabla = read_nabla.copy()
        if self.inner_config.get('correct_written_gradient'):
            written = written_logarithmic_gradient(
                previous_temperature, snapshot['total_pressure'],
            )
            correctable = mask & np.append(mask[1:], True)
            corrected = np.where(
                correctable, written + (target_nabla - read_nabla), target_nabla,
            )
            nabla[mask] = corrected[mask]
        else:
            nabla[mask] = target_nabla[mask]
        proposed = apply_logarithmic_gradient_to_temperature(
            temperature=previous_temperature,
            total_pressure=snapshot['total_pressure'],
            nabla=nabla,
            mask=mask,
            relative_cap=self.inner_config['temperature_relative_cap'],
        )
        observed = current['trial_temperature']
        relative = proposed / np.maximum(previous_temperature, 1.0) - 1.0
        temp_cap = np.zeros_like(proposed, dtype=bool)
        temp_cap[mask] = (
            np.abs(relative[mask])
            >= self.inner_config['temperature_relative_cap'] - 1.0e-12
        )
        delta_cap = np.zeros_like(proposed, dtype=bool)
        delta_cap[mask] = trial_delta[mask] >= cap - 1.0e-12
        matches = bool(np.allclose(proposed, observed, rtol=0.0, atol=1.0e-9))
        return proposed, temp_cap, delta_cap, matches

    def _write_pass_stage(self, pass_index: int) -> None:
        entry = self.inner_passes[pass_index]
        snapshot = entry['snapshot']
        temperature = entry['trial_temperature']
        extras = snapshot['extras']
        nabla = snapshot['logarithmic_gradient']
        nabla_ad = snapshot['adiabatic_gradient']
        h_conv_raw = extras['raw_convective_flux']
        h_conv_used = snapshot['convective_flux']
        h_rad = self.inner_h_rad
        provenance = (
            f'inner_pass_{pass_index:02d} temperature '
            '(opacity, gas pressure and column mass held at the round input '
            'state; radiative field frozen at the input transfer)'
        )
        if pass_index == 0:
            delta_t = np.zeros_like(temperature)
            temp_cap = np.zeros_like(temperature, dtype=bool)
            delta_cap = np.zeros_like(temperature, dtype=bool)
            matches = True
            relative_to = STAGE_ID_INPUT
            action_note = (
                'physics seed evaluated at the input temperature before pass 1; '
                'no gradient update applied yet'
            )
        else:
            proposed, temp_cap, delta_cap, matches = self._rederive_pass_update(pass_index)
            delta_t = temperature - self.inner_passes[pass_index - 1]['trial_temperature']
            relative_to = f'inner_pass_{pass_index - 1:02d}'
            action_note = (
                'per-layer delta vs the previous stage; cap flags re-derived with '
                'the imported pure functions from the stored snapshots'
            )
        mask = self._frozen_mask()
        record = {
            'stage_id': f'inner_pass_{pass_index:02d}',
            'label': (
                'stage2_physics_seed_at_input_temperature' if pass_index == 0
                else f'stage2_after_inner_pass_{pass_index}'
            ),
            'grid': 'native (inner loop holds the depth grid fixed)',
            'depth_coordinate': {
                'name': None,
                'values': None,
                'gradient_coordinate_note': (
                    'the pass update integrates along d ln total_pressure, which '
                    'is recorded under structure; the Rosseland grid is the '
                    'stage1 native grid and does not move during the loop'
                ),
            },
            'structure': {
                'temperature_K': _full(temperature),
                'column_mass_g_cm2': _null_field(
                    'not in ConvectivePhysicsSnapshot; held fixed at the stage1 '
                    'value for the whole loop'
                ),
                'gas_pressure_dyn_cm2': _null_field(
                    'not in ConvectivePhysicsSnapshot (runtime_state.gas_pressure '
                    'is held fixed at the round input value during the loop)'
                ),
                'total_pressure_dyn_cm2': _full(snapshot['total_pressure']),
            },
            'thermodynamics': {
                'heat_capacity': _full(extras['heat_capacity']),
                'adiabatic_gradient': _full(nabla_ad),
                'log_temperature_pressure_gradient': _full(nabla),
                'superadiabatic_difference': _full(np.asarray(nabla) - np.asarray(nabla_ad)),
                'log_density_temperature_derivative_at_constant_total_pressure': _full(
                    extras['log_density_temperature_derivative_at_constant_total_pressure']
                ),
            },
            'energy_transport': _transport_block(
                h_rad=h_rad,
                h_conv_raw=h_conv_raw,
                h_conv_smoothed=h_conv_used,
                target_h=self.target_h,
                provenance_raw=provenance,
                provenance_smoothed=provenance,
            ),
            'update_action': {
                'relative_to': relative_to,
                'note': action_note,
                'delta_temperature_K': _full(delta_t),
                'convection_on': _full(h_conv_raw > 0.0),
                'temperature_relative_cap': self.inner_config['temperature_relative_cap'],
                'temperature_cap_touched_layers': _full(temp_cap.astype(float)),
                'n_temperature_cap_touched': int(np.count_nonzero(temp_cap)),
                'n_temperature_cap_touched_deep': int(np.count_nonzero(
                    temp_cap[DEEP_LAYERS[0]:DEEP_LAYERS[-1] + 1]
                )),
                'superadiabatic_cap': self.inner_config['superadiabatic_cap'],
                'superadiabatic_cap_touched_layers': _full(delta_cap.astype(float)),
                'n_superadiabatic_cap_touched': int(np.count_nonzero(delta_cap)),
                'n_superadiabatic_cap_touched_deep': int(np.count_nonzero(
                    delta_cap[DEEP_LAYERS[0]:DEEP_LAYERS[-1] + 1]
                )),
                'rederivation_matches_observed_temperature': matches,
            },
            'radiative_field_annotation': _radiative_annotation(
                convection_on=h_conv_raw > 0.0,
                n_masked=int(np.count_nonzero(mask)),
                mask_frozen=bool(self.inner_config['freeze_mask_during_passes']),
                opacity_lagging=self.opacity_lagging,
                runner_r_smoothed=None,
            ),
            'local_mismatch': self._local_mismatch_block(
                h_rad, h_conv_raw, h_conv_used,
            ),
            'temperature_correction': self._temperature_correction_block(None, None),
            'raw_total_flux_residual': self._raw_residual_block(
                h_rad, h_conv_raw, h_conv_used,
            ),
        }
        _append_stage(self.stage_path, record)

    def _write_stage3(self) -> None:
        result = self.inner_result
        last = self.inner_passes[-1]['snapshot']
        extras = last['extras']
        nabla = last['logarithmic_gradient']
        nabla_ad = last['adiabatic_gradient']
        h_rad = self.inner_h_rad
        h_conv_raw = extras['raw_convective_flux']
        h_conv_used = result['convective_flux']
        input_temperature = self.inner_passes[0]['trial_temperature']
        provenance = (
            'inner_loop_exit (convection recomputed by the last physics callback '
            'at the inner-loop temperature; radiative field still the input transfer)'
        )
        released = result['convective_mask_final'] & ~result['convective_mask_initial']
        removed = result['convective_mask_initial'] & ~result['convective_mask_final']
        mask_stage1_matches = bool(np.array_equal(
            result['convective_mask_initial'], self._frozen_mask(),
        ))
        aggregate_delta = result['temperature'] - input_temperature
        any_temp_cap = any(
            record['n_temperature_cap_touched'] > 0
            for record in self._pass_cap_flags()
        )
        record = {
            'stage_id': STAGE_ID_EXIT,
            'label': 'stage3_inner_loop_end_mask_released',
            'grid': 'native',
            'depth_coordinate': {
                'name': 'log10_rosseland_optical_depth',
                'values': _full(np.log10(self.pre_inner['kwargs']['rosseland_optical_depth'])),
                'gradient_coordinate_note': 'unchanged native grid of the round',
            },
            'structure': {
                'temperature_K': _full(result['temperature']),
                'column_mass_g_cm2': _null_field(
                    'held fixed by the loop (pressure held fixed, '
                    'convection_inner_loop.py apply_logarithmic_gradient_to_temperature); '
                    'identical to the stage1 column_mass'
                ),
                'gas_pressure_dyn_cm2': _null_field(
                    'held fixed at the round input value during the loop'
                ),
                'total_pressure_dyn_cm2': _full(last['total_pressure']),
            },
            'thermodynamics': {
                'heat_capacity': _full(extras['heat_capacity']),
                'adiabatic_gradient': _full(nabla_ad),
                'log_temperature_pressure_gradient': _full(nabla),
                'superadiabatic_difference': _full(np.asarray(nabla) - np.asarray(nabla_ad)),
            },
            'energy_transport': _transport_block(
                h_rad=h_rad,
                h_conv_raw=h_conv_raw,
                h_conv_smoothed=h_conv_used,
                target_h=self.target_h,
                provenance_raw=provenance,
                provenance_smoothed=provenance,
            ),
            'update_action': {
                'relative_to': STAGE_ID_INPUT,
                'note': (
                    'aggregate delta over all 8 passes; per-pass actions are in '
                    'the inner_pass_* blocks'
                ),
                'delta_temperature_K': _full(aggregate_delta),
                'convection_on': _full(h_conv_raw > 0.0),
                'any_pass_touched_temperature_cap': any_temp_cap,
            },
            'mask_release': {
                'mask_was_frozen': result['mask_was_frozen'],
                'mask_was_released': result['mask_was_released'],
                'n_masked_initial': int(np.count_nonzero(result['convective_mask_initial'])),
                'n_masked_final': int(np.count_nonzero(result['convective_mask_final'])),
                'released_layers': _full(released.astype(float)),
                'removed_layers': _full(removed.astype(float)),
                'assigned_required_flux': result['assigned_required_flux'],
                'physics_evaluations': result['physics_evaluations'],
                'mask_initial_matches_stage1_derivation': mask_stage1_matches,
                'pass_records': result['pass_records'],
            },
            'radiative_field_annotation': _radiative_annotation(
                convection_on=h_conv_raw > 0.0,
                n_masked=int(np.count_nonzero(result['convective_mask_final'])),
                mask_frozen=result['mask_was_frozen'],
                opacity_lagging=self.opacity_lagging,
                runner_r_smoothed=None,
            ),
            'local_mismatch': self._local_mismatch_block(
                h_rad, h_conv_raw, h_conv_used,
            ),
            'temperature_correction': self._temperature_correction_block(None, None),
            'raw_total_flux_residual': self._raw_residual_block(
                h_rad, h_conv_raw, h_conv_used,
            ),
        }
        _append_stage(self.stage_path, record)

    def _pass_cap_flags(self) -> list[dict[str, int]]:
        flags = []
        stage_payload = _read_stages(self.stage_path)['stages']
        for index in range(0, self.inner_config['passes'] + 1):
            record = stage_payload.get(f'inner_pass_{index:02d}')
            if record is None:
                continue
            action = record['update_action']
            flags.append({
                'n_temperature_cap_touched': action['n_temperature_cap_touched'],
                'n_superadiabatic_cap_touched': action['n_superadiabatic_cap_touched'],
            })
        return flags

    def _write_stage_post_inner(self) -> None:
        kwargs = self.post_inner['kwargs']
        result = self.post_inner['result']
        nabla = result['logarithmic_temperature_pressure_gradient']
        nabla_ad = result['adiabatic_gradient']
        h_rad = self._h_rad()
        provenance = (
            'post_inner_recompute (runner.py:1782 re-runs convection at the '
            'inner-loop output temperature with a fresh finite-difference sample, '
            'seed inner_seed_base+9000; this is the MLT state the correction '
            'consumes)'
        )
        temperature_equal = bool(np.allclose(
            kwargs['temperature_k'], self.inner_result['temperature'],
            rtol=0.0, atol=1.0e-9,
        ))
        record = {
            'stage_id': STAGE_ID_POST_INNER,
            'label': 'stage3b_post_inner_convection_recompute',
            'grid': 'native',
            'depth_coordinate': {
                'name': 'log10_rosseland_optical_depth',
                'values': _full(np.log10(kwargs['rosseland_optical_depth'])),
                'gradient_coordinate_note': 'unchanged native grid of the round',
            },
            'structure': {
                'temperature_K': _full(kwargs['temperature_k']),
                'column_mass_g_cm2': _full(kwargs['column_mass']),
                'gas_pressure_dyn_cm2': _full(kwargs['gas_pressure']),
                'total_pressure_dyn_cm2': _full(kwargs['total_pressure']),
                'mass_density_g_cm3': _full(kwargs['mass_density']),
                'temperature_matches_inner_loop_exit': temperature_equal,
            },
            'thermodynamics': {
                'heat_capacity': _full(result['heat_capacity']),
                'adiabatic_gradient': _full(nabla_ad),
                'log_temperature_pressure_gradient': _full(nabla),
                'superadiabatic_difference': _full(np.asarray(nabla) - np.asarray(nabla_ad)),
            },
            'energy_transport': _transport_block(
                h_rad=h_rad,
                h_conv_raw=result['raw_convective_flux'],
                h_conv_smoothed=result['convective_flux'],
                target_h=self.target_h,
                provenance_raw=provenance,
                provenance_smoothed=provenance,
            ),
            'update_action': self._update_action_zero(kwargs['temperature_k']),
            'radiative_field_annotation': _radiative_annotation(
                convection_on=result['raw_convective_flux'] > 0.0,
                n_masked=None,
                mask_frozen=None,
                opacity_lagging=self.opacity_lagging,
                runner_r_smoothed=None,
            ),
            'local_mismatch': self._local_mismatch_block(
                h_rad, result['raw_convective_flux'], result['convective_flux'],
            ),
            'temperature_correction': self._temperature_correction_block(None, None),
            'raw_total_flux_residual': self._raw_residual_block(
                h_rad, result['raw_convective_flux'], result['convective_flux'],
            ),
        }
        _append_stage(self.stage_path, record)

    def _write_stage4(self) -> None:
        correction = self.finalization.temperature_correction_result
        native_tau = np.asarray(self.finalization.rosseland_optical_depth, dtype=np.float64)
        h_rad = self._h_rad()
        previous_temperature = self.post_inner['kwargs']['temperature_k']
        delta_t = np.asarray(correction.temperature, dtype=np.float64) - previous_temperature
        raw_correction = np.asarray(correction.raw_temperature_correction, dtype=np.float64)
        relative_correction = raw_correction / np.maximum(previous_temperature, 1.0)
        provenance_smoothed = (
            'correction_native_grid: smoothed convective flux evaluated at the '
            'inner-loop output temperature (pre-correction), 1-2-1 smoothed in '
            'temperature_correction.py; the correction is derived from it but '
            'the flux is not re-evaluated at the corrected temperature'
        )
        provenance_raw = (
            'correction_native_grid: post-inner recompute at the inner-loop '
            'output temperature'
        )
        record = {
            'stage_id': STAGE_ID_CORRECTION,
            'label': 'stage4_after_global_correction_before_remap',
            'grid': 'native (correction grid, before remap_to_grid)',
            'depth_coordinate': {
                'name': 'log10_rosseland_optical_depth',
                'values': _full(np.log10(native_tau)),
                'gradient_coordinate_note': (
                    'correction and remap are separate calls (runner.py:1877 vs '
                    'runner.py:607), so this stage is on the native grid'
                ),
            },
            'structure': {
                'temperature_K': _full(correction.temperature),
                'column_mass_g_cm2': _full(correction.column_mass),
                'gas_pressure_dyn_cm2': _null_field(
                    'gas pressure is not re-derived between correction and remap; '
                    'the next iteration re-integrates hydrostatically at its start '
                    '(runner.py:466-478)'
                ),
                'total_pressure_dyn_cm2': _null_field(
                    'not formed at this stage; the correction only derives a '
                    'column_mass_correction from a trial pressure change'
                ),
            },
            'thermodynamics': {
                'heat_capacity': _null_field(
                    'EOS/MLT are not re-evaluated between correction and remap'
                ),
                'adiabatic_gradient': _null_field(
                    'EOS/MLT are not re-evaluated between correction and remap'
                ),
                'log_temperature_pressure_gradient': _null_field(
                    'EOS/MLT are not re-evaluated between correction and remap'
                ),
                'superadiabatic_difference': _null_field(
                    'EOS/MLT are not re-evaluated between correction and remap'
                ),
            },
            'energy_transport': {
                'target_integrated_eddington_flux_H': float(self.target_h),
                'H_rad': _residual_field(
                    h_rad, FROZEN,
                    'iteration_input (transfer of this round input state)',
                    'frozen transfer flux; no re-transfer happens before remap',
                ),
                'H_conv_smoothed': _residual_field(
                    correction.convective_flux, FROZEN, provenance_smoothed,
                    'TemperatureCorrectionResult.convective_flux '
                    '(temperature_correction.py:947)',
                ),
                'R_smoothed': _residual_field(
                    (h_rad + np.asarray(correction.convective_flux) - self.target_h)
                    / max(self.target_h, 1.0e-300),
                    FROZEN, provenance_smoothed,
                    '(H_rad + H_conv_smoothed - H_target) / H_target',
                ),
                'H_conv_raw': _residual_field(
                    self.post_inner['result']['raw_convective_flux'], FROZEN,
                    provenance_raw, 'post-inner raw MLT flux',
                ),
                'R_raw': _residual_field(
                    (h_rad + self.post_inner['result']['raw_convective_flux'] - self.target_h)
                    / max(self.target_h, 1.0e-300),
                    FROZEN, provenance_raw,
                    '(H_rad + H_conv_raw - H_target) / H_target',
                ),
                'runner_flux_error_percent': _full(correction.flux_error_percent),
                'runner_flux_error_note': (
                    'the correction own flux error, same frozen H_rad '
                    '(temperature_correction.py:704-712)'
                ),
            },
            'update_action': {
                'relative_to': STAGE_ID_POST_INNER,
                'note': (
                    'global Avrett-style correction on the native grid; the '
                    'relative caps recorded here are the inner-loop constants '
                    'reported for information only -- the global correction is '
                    'damped by temperature_correction_damping, not by them'
                ),
                'delta_temperature_K': _full(delta_t),
                'raw_temperature_correction_K': _full(raw_correction),
                'relative_correction_vs_inner_loop_cap': _full(relative_correction),
                'n_layers_relative_correction_over_inner_cap': int(np.count_nonzero(
                    np.abs(relative_correction) >= 0.15
                )),
                'convection_on': _full(
                    self.post_inner['result']['raw_convective_flux'] > 0.0
                ),
                'convection_on_note': (
                    'evaluated at the pre-correction (inner-loop output) '
                    'temperature; no mask re-derivation happens between '
                    'correction and remap'
                ),
            },
            'radiative_field_annotation': _radiative_annotation(
                convection_on=self.post_inner['result']['raw_convective_flux'] > 0.0,
                n_masked=None,
                mask_frozen=None,
                opacity_lagging=self.opacity_lagging,
                runner_r_smoothed=(
                    np.asarray(correction.flux_error_percent, dtype=np.float64) / 100.0
                ),
            ),
            'local_mismatch': self._local_mismatch_block(
                h_rad,
                self.post_inner['result']['raw_convective_flux'],
                correction.convective_flux,
            ),
            'temperature_correction': self._temperature_correction_block(
                delta_t, None,
            ),
            'raw_total_flux_residual': self._raw_residual_block(
                h_rad,
                self.post_inner['result']['raw_convective_flux'],
                correction.convective_flux,
            ),
        }
        _append_stage(self.stage_path, record)

    def _write_stage5(self) -> None:
        from experiments.reduced_state_emulator.m_star_local_iteration_response_20260919 import (
            _physics_rows,
        )

        atmosphere = self.remap_result.atmosphere
        standard_tau = np.asarray(
            self.remap_result.standard_rosseland_optical_depth, dtype=np.float64
        )
        rows, arrays = _physics_rows(
            step=SimpleNamespace(remapped=self.remap_result, transfer=self.finalization.transfer_accumulation),
            effective_temperature=self.effective_temperature,
            indices=DEEP_LAYERS,
        )
        previous_temperature = np.asarray(
            self.finalization.temperature_correction_result.temperature,
            dtype=np.float64,
        )
        delta_t = np.asarray(atmosphere.temperature, dtype=np.float64) - previous_temperature
        total_pressure = (
            self.surface_gravity_cgs * np.asarray(atmosphere.column_mass, dtype=np.float64)
            + float(self.finalization.radiative_pressure_state.surface_radiation_pressure_constant)
            + np.asarray(self.remap_result.turbulent_pressure, dtype=np.float64)
        )
        record = {
            'stage_id': STAGE_ID_REMAP,
            'label': 'stage5_after_remap_step_remapped',
            'grid': 'standard Rosseland optical-depth grid (log step 0.125)',
            'depth_coordinate': {
                'name': 'log10_rosseland_optical_depth',
                'values': _full(np.log10(standard_tau)),
                'gradient_coordinate_note': (
                    'remap_finalized_iteration_state interpolates every column '
                    'onto this grid; layer count is unchanged (80) but the depth '
                    'coordinates differ from the native grid'
                ),
            },
            'structure': {
                'temperature_K': _full(atmosphere.temperature),
                'column_mass_g_cm2': _full(atmosphere.column_mass),
                'gas_pressure_dyn_cm2': _full(atmosphere.gas_pressure),
                'total_pressure_dyn_cm2': {
                    'full_depth': _full(total_pressure),
                    'note': (
                        'recomputed as g*column_mass + surface radiation-pressure '
                        'constant + turbulent pressure (the runner.py:569-572 '
                        'formula); the remapped atmosphere itself does not carry '
                        'total pressure'
                    ),
                },
            },
            'thermodynamics': {
                'heat_capacity': _null_field(
                    'not re-evaluated by the remap; the next evaluation is the '
                    'next round population stage'
                ),
                'adiabatic_gradient': _null_field(
                    'not re-evaluated by the remap'
                ),
                'log_temperature_pressure_gradient': _null_field(
                    'not re-evaluated by the remap'
                ),
                'superadiabatic_difference': _null_field(
                    'not re-evaluated by the remap'
                ),
            },
            'energy_transport': {
                'target_integrated_eddington_flux_H': float(self.target_h),
                'H_rad': _residual_field(
                    arrays['Hrad'], FROZEN,
                    'iteration_input (transfer of this round input state)',
                    'frozen transfer flux; the remap runs no transfer',
                ),
                'H_conv_raw': _residual_field(
                    arrays['Hconv_raw'], FROZEN,
                    'inner_loop_exit / post_inner_recompute (inner-loop output '
                    'temperature, runner.py:1782 recompute stored as '
                    'finalization.convection_result)',
                    'raw MLT flux; NOT evaluated at the remapped output state',
                ),
                'H_conv_smoothed': _residual_field(
                    arrays['Hconv_smoothed'], FROZEN,
                    'correction_native_grid (smoothed pre-correction flux, '
                    'remapped to the standard grid at runner.py:2032-2050)',
                    'remapped atmosphere convective_flux',
                ),
                'R_raw': _residual_field(
                    arrays['R_raw'], FROZEN,
                    'hybrid: frozen input-transfer H_rad with inner-loop-output '
                    'H_conv_raw; this is the quantity the S3 trajectory labels R_raw',
                    '(H_rad + H_conv_raw - H_target) / H_target via _physics_rows',
                ),
                'R_smoothed': _residual_field(
                    arrays['R_smoothed'], FROZEN,
                    'hybrid: frozen input-transfer H_rad with pre-correction '
                    'smoothed H_conv',
                    '(H_rad + H_conv_smoothed - H_target) / H_target via _physics_rows',
                ),
                'runner_flux_error_percent': _full(
                    self.finalization.temperature_correction_result.flux_error_percent
                ),
            },
            'update_action': {
                'relative_to': STAGE_ID_CORRECTION,
                'note': (
                    'cross-grid difference: the native correction grid and the '
                    'standard grid share 80 layers but different depth '
                    'coordinates, so per-layer deltas include the remap '
                    'interpolation, not only a physical update'
                ),
                'delta_temperature_K': _full(delta_t),
                'remapped_delta_temperature_K': _full(delta_t),
                'convection_on': None,
                'convection_on_note': (
                    'no convection evaluation happens at this stage; the next '
                    'evaluation belongs to the next round'
                ),
            },
            'radiative_field_annotation': _radiative_annotation(
                convection_on=None,
                n_masked=None,
                mask_frozen=None,
                opacity_lagging=self.opacity_lagging,
                runner_r_smoothed=arrays['runner_R_smoothed'],
            ),
            'local_mismatch': self._local_mismatch_block(
                arrays['Hrad'], arrays['Hconv_raw'], arrays['Hconv_smoothed'],
            ),
            'temperature_correction': self._temperature_correction_block(
                delta_t, delta_t,
            ),
            'raw_total_flux_residual': {
                'layer': [str(i) for i in range(len(arrays['R_raw']))],
                'R_raw': _full(arrays['R_raw']),
                'R_smoothed': _full(arrays['R_smoothed']),
                'deep_mean_abs': float(np.mean(np.abs(arrays['R_raw'][DEEP_LAYERS]))),
                'deep_max': float(np.max(np.abs(arrays['R_raw'][DEEP_LAYERS]))),
            },
            'physics_rows_summary': rows,
        }
        _append_stage(self.stage_path, record)

    # -- shared block builders -----------------------------------------------

    def _local_mismatch_block(
        self, h_rad, h_conv_raw, h_conv_smoothed,
    ) -> dict[str, Any]:
        target = self.target_h
        return {
            'layer': [str(i) for i in range(len(h_rad))],
            'Hconv_raw': _full(h_conv_raw),
            'Hconv_smoothed': _full(h_conv_smoothed),
            'Hrad': _full(h_rad),
            'target_integrated_eddington_flux': float(target),
            'relative_mismatch': _full((h_rad + h_conv_raw - target) / max(target, 1.0e-300)),
            'radiation_field': FROZEN,
        }

    def _raw_residual_block(self, h_rad, h_conv_raw, h_conv_smoothed) -> dict[str, Any]:
        target = self.target_h
        r_raw = (h_rad + h_conv_raw - target) / max(target, 1.0e-300)
        r_smooth = (h_rad + h_conv_smoothed - target) / max(target, 1.0e-300)
        return {
            'layer': [str(i) for i in range(len(r_raw))],
            'R_raw': _full(r_raw),
            'R_smoothed': _full(r_smooth),
            'deep_mean_abs': float(np.mean(np.abs(r_raw[DEEP_LAYERS]))),
            'deep_max': float(np.max(np.abs(r_raw[DEEP_LAYERS]))),
            'radiation_field': FROZEN,
            'state_provenance': 'see energy_transport blocks of this stage',
        }

    def _temperature_correction_block(self, delta_t, remapped_delta_t) -> dict[str, Any]:
        return {
            'layer': None if delta_t is None else [str(i) for i in range(len(delta_t))],
            'delta_T_K': None if delta_t is None else _full(delta_t),
            'remapped_delta_T_K': None if remapped_delta_t is None else _full(remapped_delta_t),
            'max_abs_K': None if delta_t is None else float(np.max(np.abs(delta_t))),
        }

    def _update_action_zero(self, temperature: np.ndarray) -> dict[str, Any]:
        zeros = np.zeros_like(temperature)
        return {
            'relative_to': 'none (stage boundary of this round)',
            'note': 'no temperature update is applied at this stage',
            'delta_temperature_K': _full(zeros),
            'convection_on': None,
            'convection_on_note': 'see energy_transport H_conv_raw of this stage',
        }

    # -- summary --------------------------------------------------------------

    def _stage_scalar_summary(self, record: dict[str, Any]) -> dict[str, Any]:
        transport = record.get('energy_transport', {})
        r_raw = transport.get('R_raw')
        r_smooth = transport.get('R_smoothed')
        action = record.get('update_action', {})

        def _deep(values):
            if values is None or values.get('full_depth') is None:
                return None
            array = np.abs(np.asarray(values['full_depth'], dtype=np.float64)[DEEP_LAYERS])
            return {'deep_mean_abs': float(np.mean(array)), 'deep_max': float(np.max(array))}

        delta = action.get('delta_temperature_K')
        return {
            'stage_id': record['stage_id'],
            'label': record['label'],
            'radiation_field_of_residuals': (
                r_raw['radiation_field'] if isinstance(r_raw, dict) else FROZEN
            ),
            'R_raw': None if not isinstance(r_raw, dict) else _deep(r_raw),
            'R_smoothed': None if not isinstance(r_smooth, dict) else _deep(r_smooth),
            'raw_total_flux_residual': {
                'deep_mean_abs': record['raw_total_flux_residual']['deep_mean_abs'],
                'deep_max': record['raw_total_flux_residual']['deep_max'],
            },
            'delta_T_max_abs_K_vs_previous_stage': (
                None if delta is None else float(np.max(np.abs(delta)))
            ),
            'n_temperature_cap_touched': action.get('n_temperature_cap_touched'),
            'n_superadiabatic_cap_touched': action.get('n_superadiabatic_cap_touched'),
            'n_masked': record.get('radiative_field_annotation', {}).get('n_masked'),
        }

    def _write_summary(self) -> None:
        payload = _read_stages(self.stage_path)
        stages = payload.get('stages', {})
        convection_timing = {
            key: value for key, value in self.timing.items()
            if key.startswith('convection_inner_loop_')
        }
        summary = {
            'case': self.case,
            'effective_temperature_K': self.effective_temperature,
            'target_integrated_eddington_flux_H': float(self.target_h),
            'input_npz': self.input_npz,
            'deep_layers': DEEP_LAYERS,
            'inner_loop_config': self.inner_config,
            'stage_order': [
                key for key in (
                    [STAGE_ID_INPUT]
                    + [f'inner_pass_{index:02d}' for index in range(
                        (self.inner_config or {'passes': INNER_PASSES})['passes'] + 1)]
                    + [STAGE_ID_EXIT, STAGE_ID_POST_INNER, STAGE_ID_CORRECTION, STAGE_ID_REMAP]
                ) if key in stages
            ],
            'stages': {
                stage_id: self._stage_scalar_summary(record)
                for stage_id, record in stages.items()
            },
            'convection_inner_loop_timing': convection_timing,
            'other_iteration_timing': {
                key: value for key, value in self.timing.items()
                if not key.startswith('convection_inner_loop_')
            },
            'input_crosscheck': self.input_crosscheck,
            'unexpected_convection_calls': self.unexpected_convection_calls,
            'expected_convection_call_pattern': (
                '1 pre_inner + 9 inner physics (seed + 8 passes) + 1 post_inner '
                'per round; anything else is listed above'
            ),
            'seconds': float(time.perf_counter() - self.started),
            'radiation_field_convention': (
                'every residual inside one round uses the round input transfer '
                'flux, hence frozen; a current residual requires a fresh full '
                'transfer, which only evaluate_state_only performs'
            ),
        }
        _write_json(self.run_root / 'stages_summary.json', summary)


def _full_kwargs(kwargs: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            out[key] = np.asarray(value, dtype=np.float64).copy()
        elif isinstance(value, (int, float, bool, str)) or value is None:
            out[key] = value
        # non-scalar objects (tables, samples) are intentionally not copied
    return out


def _s3_replace_flags(
    inner_loop_passes: int,
    correct_written_gradient: bool = False,
    refresh_state: bool = False,
    hold_correction: bool = False,
    relaxation: float = 1.0,
    fill_holes: bool = False,
) -> dict[str, Any]:
    """The S3 solver flags, factored so the evaluator can reuse them."""

    return {
        'enable_opacity_lagging': False,
        'opacity_recompute_interval': 1,
        'flux_residual_guided_damping': False,
        'require_improving_flux_residual': False,
        'convection_zone_inner_loop_passes': int(inner_loop_passes),
        'convection_zone_inner_loop_freeze_mask': True,
        'convection_zone_inner_loop_correct_written_gradient': bool(
            correct_written_gradient
        ),
        'convection_zone_inner_loop_refresh_state': bool(refresh_state),
        'convection_zone_inner_loop_hold_correction': bool(hold_correction),
        'convection_zone_inner_loop_relaxation': float(relaxation),
        'convection_zone_inner_loop_fill_holes': bool(fill_holes),
    }


def _build_s3_config(
    atmosphere,
    *,
    inner_loop_passes: int,
    correct_written_gradient: bool = False,
    refresh_state: bool = False,
    hold_correction: bool = False,
    relaxation: float = 1.0,
    fill_holes: bool = False,
):
    from bench.run_reference import _solver_config

    config = _solver_config(
        atmosphere,
        iterations_per_trial=1,
        structured_atmosphere_path=None,
        debug_state_path=None,
    )
    return dataclasses.replace(
        config,
        **_s3_replace_flags(
            inner_loop_passes, correct_written_gradient, refresh_state, hold_correction,
            relaxation, fill_holes,
        ),
    )


def wire_round(recorder: _RoundRecorder):
    """Install the experiment-side wrappers; returns a restore callable.

    Only module-level names in ``payne_zero_atmosphere.runner`` are rebound,
    and only for the duration of the round; no production file is modified.
    """

    import payne_zero_atmosphere.runner as runner_module

    original_convection = runner_module.compute_convection
    original_inner = runner_module.run_convection_zone_inner_loop
    original_remap = runner_module.remap_finalized_iteration_state

    def wrapped_convection(**kwargs):
        result = original_convection(**kwargs)
        recorder.observe_convection_call(kwargs, result)
        return result

    def wrapped_inner(*, temperature, radiative_eddington_flux,
                      target_eddington_flux, physics, config):
        recorder.enter_inner_loop(
            radiative_eddington_flux=radiative_eddington_flux, config=config,
        )

        def wrapped_physics(trial_temperature):
            snapshot = physics(trial_temperature)
            # called once as the loop seed, then after every pass; the
            # recorder indexes the stash by call order (seed = pass 00)
            recorder.observe_inner_snapshot(trial_temperature, snapshot)
            return snapshot

        try:
            result = original_inner(
                temperature=temperature,
                radiative_eddington_flux=radiative_eddington_flux,
                target_eddington_flux=target_eddington_flux,
                physics=wrapped_physics,
                config=config,
            )
        finally:
            recorder._inside_inner = False
            recorder._inner_done = True
        recorder.exit_inner_loop(result)
        return result

    def wrapped_remap(finalization, **kwargs):
        recorder.enter_remap(finalization)
        result = original_remap(finalization, **kwargs)
        recorder.exit_remap(result)
        return result

    runner_module.compute_convection = wrapped_convection
    runner_module.run_convection_zone_inner_loop = wrapped_inner
    runner_module.remap_finalized_iteration_state = wrapped_remap

    def restore():
        runner_module.compute_convection = original_convection
        runner_module.run_convection_zone_inner_loop = original_inner
        runner_module.remap_finalized_iteration_state = original_remap

    return restore


def run_capture_round(
    *,
    case: str,
    start_npz: Path,
    run_root: Path,
    resume: bool = False,
    correct_written_gradient: bool = False,
    refresh_state: bool = False,
    hold_correction: bool = False,
    relaxation: float = 1.0,
    fill_holes: bool = False,
) -> int:
    """Run exactly one S3 round from an explicit start state, capturing stages."""

    for key, value in (
        ('NUMBA_THREADING_LAYER', 'workqueue'), ('NUMBA_NUM_THREADS', '1'),
        ('OMP_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1'),
        ('OPENBLAS_NUM_THREADS', '1'), ('VECLIB_MAXIMUM_THREADS', '1'),
        ('CUDA_VISIBLE_DEVICES', ''),
    ):
        os.environ[key] = value
    os.environ['PAYNE_ZERO_DATA_ROOT'] = str(REPO / 'source_data_files')

    run_root.mkdir(parents=True, exist_ok=True)
    stage_path = run_root / 'stages.json'
    if resume:
        payload = _read_stages(stage_path)
        if TERMINAL_STAGE_ID in payload.get('stages', {}) and (
            run_root / 'stages_summary.json'
        ).exists():
            print(f'[{case}] complete capture already present at {stage_path}', flush=True)
            return 0

    sys.path.insert(0, str(REPO))
    from experiments.reduced_state_emulator.m_star_h2_paired_iteration_20260921 import (
        build_overlay,
    )
    overlay_root = build_overlay('pchip', run_root)
    os.environ['NUMBA_CACHE_DIR'] = str(run_root / 'numba_cache')
    sys.path.insert(0, str(overlay_root))

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
    _write_json(run_root / 'implementation.json', {
        'case': case, 'scheme': 'pchip', 'inner_loop_passes': INNER_PASSES,
        'correct_written_gradient': bool(correct_written_gradient),
        'refresh_state': bool(refresh_state),
        'hold_correction': bool(hold_correction),
        'relaxation': float(relaxation),
        'fill_holes': bool(fill_holes),
        'module': me.__file__, 'scipy_max_abs_error': error,
    })

    from experiments.reduced_state_emulator.m_star_local_iteration_response_20260919 import (
        _load_case, _state_from_mt,
    )
    from payne_zero_atmosphere.run_setup import surface_gravity_from_atmosphere
    from payne_zero_atmosphere.runner import run_atmosphere_model

    loaded = _load_case(case)
    labels = loaded['labels']
    effective_temperature = loaded['target_temperature_K']
    with np.load(start_npz, allow_pickle=False) as data:
        state = _state_from_mt(labels, data['column_mass'], data['temperature'])
    recorder = _RoundRecorder(
        case=case, run_root=run_root, effective_temperature=effective_temperature,
        opacity_lagging=False, input_npz=str(start_npz.resolve()),
    )
    recorder.surface_gravity_cgs = float(surface_gravity_from_atmosphere(state))

    def hook(iteration_index, setup, step):
        recorder.finalize(
            timing=dict(step.timing), setup_atmosphere=setup.atmosphere,
        )
        return {'stage_capture': True, 'stages_written': len(
            _read_stages(stage_path).get('stages', {})
        )}

    restore = wire_round(recorder)
    try:
        config = _build_s3_config(
            state, inner_loop_passes=INNER_PASSES,
            correct_written_gradient=correct_written_gradient,
            refresh_state=refresh_state,
            hold_correction=hold_correction,
            relaxation=relaxation,
            fill_holes=fill_holes,
        )
        result = run_atmosphere_model(config, after_iteration_hook=hook)
    finally:
        restore()

    completed = int(getattr(result, 'iterations_completed', 0))
    stage_count = len(_read_stages(recorder.stage_path).get('stages', {}))
    print(
        f'[{case}] capture complete: stages={stage_count} '
        f'iterations_completed={completed} '
        f'status={"ok" if completed else "failed"}', flush=True,
    )
    return 0 if completed else 1


def evaluate_state_only(
    state,
    labels,
    teff: float,
    result_root: Path,
    run_id: str,
    *,
    inner_loop_passes: int = 0,
) -> dict[str, Any]:
    """Evaluate one explicit state: full physics, one round, no trajectory.

    Runs ``iterations_per_trial=1`` with the S3 flags (opacity lagging off,
    interval 1, no residual-guided damping, no require-improving) and the
    inner loop OFF by default, so ``_physics_rows`` sees
    ``finalization.convection_result`` evaluated at the input temperature
    (runner.py:1607 path) and its R_raw/R_smoothed are the input state's own
    residual (see the module docstring for the anchor-based verification).
    The returned correction is never fed into any trajectory: the caller's
    state object is not mutated (the round runs on a clone) and nothing is
    chained forward.
    """

    from experiments.reduced_state_emulator.m_star_local_iteration_response_20260919 import (
        _clone_atmosphere, _physics_rows, _write_json,
    )
    from payne_zero_atmosphere.runner import run_atmosphere_model

    capture: dict[str, Any] = {}
    effective_temperature = float(teff)

    def hook(iteration_index, setup, step):
        physics, physics_arrays = _physics_rows(
            step=step, effective_temperature=effective_temperature,
            indices=DEEP_LAYERS,
        )
        capture['physics_rows'] = physics
        capture['physics_arrays'] = physics_arrays
        capture['input_arrays'] = {
            'temperature': np.asarray(setup.atmosphere.temperature, dtype=np.float64).copy(),
            'column_mass': np.asarray(setup.atmosphere.column_mass, dtype=np.float64).copy(),
            'gas_pressure': np.asarray(setup.atmosphere.gas_pressure, dtype=np.float64).copy(),
        }
        capture['output_arrays'] = {
            'temperature': np.asarray(
                step.remapped.atmosphere.temperature, dtype=np.float64
            ).copy(),
            'column_mass': np.asarray(
                step.remapped.atmosphere.column_mass, dtype=np.float64
            ).copy(),
        }
        capture['timing'] = dict(step.timing)
        capture['radiation_field'] = FROZEN
        capture['state_provenance'] = (
            'input state of this evaluation round' if inner_loop_passes == 0 else (
                'hybrid: frozen input-transfer H_rad with convection evaluated '
                f'after {inner_loop_passes} inner passes at the inner-loop '
                'output temperature'
            )
        )
        return {'evaluated_input_state': True}

    started = time.perf_counter()
    error = None
    result = None
    try:
        config = _build_s3_config(
            _clone_atmosphere(state), inner_loop_passes=inner_loop_passes,
        )
        result = run_atmosphere_model(config, after_iteration_hook=hook)
    except Exception as exc:  # noqa: BLE001 - recorded, not raised, like the S3 reference
        error = f'{type(exc).__name__}: {exc}'
    seconds = time.perf_counter() - started
    record: dict[str, Any] = {
        'run_id': run_id,
        'seconds': float(seconds),
        'opacity_lagging': False,
        'iteration_cap': 1,
        'inner_loop_passes': int(inner_loop_passes),
        'radiation_field': FROZEN,
        'state_provenance': capture.get('state_provenance'),
        'log_surface_gravity': float(getattr(labels, 'log_surface_gravity', float('nan'))),
        'error': error,
    }
    if result is not None and 'physics_arrays' in capture:
        r_raw = np.abs(np.asarray(capture['physics_arrays']['R_raw'], dtype=np.float64))
        r_smoothed = np.abs(np.asarray(
            capture['physics_arrays']['R_smoothed'], dtype=np.float64
        ))
        deep = slice(DEEP_LAYERS[0], DEEP_LAYERS[-1] + 1)
        record.update({
            'iterations_completed': int(result.iterations_completed),
            'R_raw_deep_mean_abs': float(np.mean(r_raw[deep])),
            'R_raw_deep_max': float(np.max(r_raw[deep])),
            'R_raw_full_max': float(np.max(r_raw)),
            'R_smoothed_deep_max': float(np.max(r_smoothed[deep])),
            'physics_rows': capture['physics_rows'],
            'timing': {
                key: value for key, value in capture['timing'].items()
                if key.startswith('convection_inner_loop_')
                or key in ('p95_absolute_flux_error_percent',
                           'maximum_absolute_flux_error_percent', 'total_seconds')
            },
        })
        arrays_path = Path(result_root) / 'arrays' / f'{run_id}.npz'
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(arrays_path, **capture['physics_arrays'], **capture['input_arrays'])
        record['arrays_path'] = str(arrays_path.resolve())
    elif error is None:
        record['error'] = 'runner returned without an evaluation hook capture'
    _write_json(Path(result_root) / 'records' / f'{run_id}.json', record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Capture one S3 inner-loop round stage by stage, or '
                    'evaluate one explicit state without a trajectory.',
    )
    parser.add_argument('--case', choices=('A', 'D'), default='D')
    parser.add_argument(
        '--start-npz', type=Path, default=None,
        help='start state npz with temperature and column_mass columns '
             '(default: the case S3 it09 input state)',
    )
    parser.add_argument('--resume', action='store_true')
    parser.add_argument(
        '--correct-written-gradient', action='store_true',
        help='inner passes shift the written one-sided gradient by the change '
             'the centred MLT read-back gradient needs',
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
    parser.add_argument(
        '--run-root', type=Path, default=None,
        help='capture directory (default: the 2026-09-22 capture root / case)',
    )
    parser.add_argument(
        '--evaluate-only', action='store_true',
        help='do not capture stages; evaluate the given start state once and '
             'return its own R_raw/R_smoothed',
    )
    args = parser.parse_args()

    case = args.case
    root = args.run_root or RESULT_ROOT / case.lower()
    start_npz = args.start_npz or DEFAULT_START_NPZ[case]
    if not start_npz.exists():
        raise FileNotFoundError(f'start state npz missing: {start_npz}')

    if args.evaluate_only:
        run_id = f'{case.lower()}_state_only_eval'
        record = _evaluate_only_entry(case, start_npz, root, run_id)
        error = record.get('error')
        if error:
            print(f'[{run_id}] failed: {error}', flush=True)
            return 1
        print(
            f'[{run_id}] R_raw_deep_mean_abs='
            f'{record["R_raw_deep_mean_abs"]:.4f} '
            f'R_smoothed_deep_max={record["R_smoothed_deep_max"]:.4f}',
            flush=True,
        )
        return 0

    return run_capture_round(
        case=case, start_npz=start_npz, run_root=root, resume=args.resume,
        correct_written_gradient=args.correct_written_gradient,
        refresh_state=args.refresh_state,
        hold_correction=args.hold_correction,
        relaxation=args.relaxation,
        fill_holes=args.fill_holes,
    )


def _evaluate_only_entry(case, start_npz, root, run_id):
    sys.path.insert(0, str(REPO))
    from experiments.reduced_state_emulator.m_star_h2_paired_iteration_20260921 import (
        build_overlay,
    )
    overlay_root = build_overlay('pchip', root)
    os.environ['NUMBA_CACHE_DIR'] = str(root / 'numba_cache')
    sys.path.insert(0, str(overlay_root))

    from experiments.reduced_state_emulator.m_star_local_iteration_response_20260919 import (
        _load_case, _state_from_mt,
    )

    loaded = _load_case(case)
    with np.load(start_npz, allow_pickle=False) as data:
        state = _state_from_mt(loaded['labels'], data['column_mass'], data['temperature'])
    return evaluate_state_only(
        state, loaded['labels'], loaded['target_temperature_K'], root, run_id,
        inner_loop_passes=0,
    )


if __name__ == '__main__':
    raise SystemExit(main())
