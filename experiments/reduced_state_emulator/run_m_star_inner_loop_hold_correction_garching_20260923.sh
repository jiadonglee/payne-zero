#!/usr/bin/env bash
# S3wrc on Garching: single rounds (stage capture, then full-physics
# re-evaluation) from the A S3wr round-5 input and the D/A plateau states, and
# D/A 10-round trajectories, all with the written-gradient correction, the
# trial-temperature state refresh, and the correction hold on inner-loop layers.
# One single-threaded background process per job; production defaults stay off.
# Pass --resume to continue interrupted trajectories from their saved rounds.
# Only the D re-evaluation runs with --with-control-anchor (the anchor is the D
# control value 0.8081).
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
root="$PWD/results/m_star_inner_loop_hold_correction_20260923"
osc_start="$PWD/results/m_star_inner_loop_state_refresh_20260923/a_pchip_s3wr/arrays/a_pchip_s3wr_it05.npz"
mkdir -p "$root/osc/a" "$root/plat/d" "$root/plat/a" "$root/d_pchip_s3wrc" "$root/a_pchip_s3wrc"
resume=()
[ "${1:-}" = "--resume" ] && resume=(--resume)
arm=(--correct-written-gradient --refresh-state --hold-correction)
stages=(iteration_input inner_pass_08 standard_grid_remap)

single_round() {
  local case="$1" group="$2" start="$3" anchor="$4"
  local lower
  lower=$(echo "$case" | tr '[:upper:]' '[:lower:]')
  local start_args=()
  [ -n "$start" ] && start_args=(--start-npz "$start")
  local anchor_args=()
  [ "$anchor" = "anchor" ] && anchor_args=(--with-control-anchor)
  "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_capture_20260922 \
    --case "$case" --run-root "$root/$group/$lower" "${start_args[@]}" "${arm[@]}" \
    >> "$root/$group/$lower/capture.log" 2>&1 \
  && "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_reeval_20260922 \
    --case "$case" --capture-root "$root/$group" --stages "${stages[@]}" "${anchor_args[@]}" \
    >> "$root/$group/$lower/reeval.log" 2>&1
}

pids=()
names=()
single_round A osc "$osc_start" no & pids+=("$!"); names+=("a_osc")
single_round D plat "" anchor & pids+=("$!"); names+=("d_plat")
single_round A plat "" no & pids+=("$!"); names+=("a_plat")
for case in D A; do
  lower=$(echo "$case" | tr '[:upper:]' '[:lower:]')
  "$PY" -u -m experiments.reduced_state_emulator.m_star_h2_inner_loop_s3_20260922 \
    --case "$case" "${arm[@]}" --result-root "$root" "${resume[@]}" \
    >> "$root/${lower}_pchip_s3wrc/trajectory.log" 2>&1 &
  pids+=("$!"); names+=("${lower}_trajectory")
done

for index in "${!pids[@]}"; do
  echo "${names[$index]} ${pids[$index]}"
done | tee "$root/garching_pids.txt"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "all S3wrc jobs finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
