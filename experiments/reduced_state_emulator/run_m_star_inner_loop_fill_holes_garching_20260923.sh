#!/usr/bin/env bash
# S3wrc-λ-h single-round check on Garching: one round from the D control state
# (stage capture, then full-physics re-evaluation) and one round from the A
# control state (stage capture only), with the written-gradient correction, the
# trial-temperature state refresh, the correction hold, λ = 0.5, and interior
# hole filling.  One single-threaded background process per job.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
root="$PWD/results/m_star_inner_loop_fill_holes_20260923"
controls="$PWD/results/m_star_trial_comparison_20260920"
arm=(--correct-written-gradient --refresh-state --hold-correction --relaxation 0.5 --fill-holes)
mkdir -p "$root/single/d" "$root/single/a"

(
  "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_capture_20260922 \
    --case D --start-npz "$controls/controls_D/arrays/d_alpha_0p0.npz" \
    --run-root "$root/single/d" "${arm[@]}" \
    >> "$root/single/d/capture.log" 2>&1 \
  && "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_reeval_20260922 \
    --case D --capture-root "$root/single" --with-control-anchor \
    --stages iteration_input inner_pass_08 standard_grid_remap \
    >> "$root/single/d/reeval.log" 2>&1
) &
d_pid=$!
"$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_capture_20260922 \
  --case A --start-npz "$controls/controls_A/arrays/a_alpha_0p0.npz" \
  --run-root "$root/single/a" "${arm[@]}" \
  >> "$root/single/a/capture.log" 2>&1 &
a_pid=$!
printf 'd_single %s\na_single %s\n' "$d_pid" "$a_pid" | tee "$root/garching_pids.txt"
fail=0
wait "$d_pid" || fail=1
wait "$a_pid" || fail=1
echo "all fill-holes single-round jobs finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
