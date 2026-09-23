#!/usr/bin/env bash
# S3wr oscillation diagnosis on Garching: one S3wr stage capture from the A
# trajectory's round-5 input state, then full-physics re-evaluation of five of
# its states.  Measurement only; production defaults stay off.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
base="$PWD/results/m_star_inner_loop_state_refresh_20260923"
root="$base/diag_a_it05"
mkdir -p "$root/a"

"$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_capture_20260922 \
  --case A --start-npz "$base/a_pchip_s3wr/arrays/a_pchip_s3wr_it05.npz" \
  --run-root "$root/a" --correct-written-gradient --refresh-state \
  >> "$root/a/capture.log" 2>&1 \
&& "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_reeval_20260922 \
  --case A --capture-root "$root" \
  --stages iteration_input inner_pass_02 inner_pass_04 inner_pass_08 standard_grid_remap \
  >> "$root/a/reeval.log" 2>&1
status=$?
echo "S3wr oscillation diagnosis finished status=$status $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$status"
