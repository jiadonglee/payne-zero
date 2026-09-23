#!/usr/bin/env bash
# D round-0 diagnosis on Garching: one S3wrc (λ = 0.5) stage capture from the D
# control state, then full-physics re-evaluation of five of its states.
# Measurement only; production defaults stay off.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
root="$PWD/results/m_star_inner_loop_relaxation_20260923/diag_d_it00"
start="$PWD/results/m_star_trial_comparison_20260920/controls_D/arrays/d_alpha_0p0.npz"
mkdir -p "$root/d"

"$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_capture_20260922 \
  --case D --start-npz "$start" --run-root "$root/d" \
  --correct-written-gradient --refresh-state --hold-correction --relaxation 0.5 \
  >> "$root/d/capture.log" 2>&1 \
&& "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_reeval_20260922 \
  --case D --capture-root "$root" --with-control-anchor \
  --stages iteration_input inner_pass_02 inner_pass_04 inner_pass_08 standard_grid_remap \
  >> "$root/d/reeval.log" 2>&1
status=$?
echo "D round-0 diagnosis finished status=$status $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$status"
