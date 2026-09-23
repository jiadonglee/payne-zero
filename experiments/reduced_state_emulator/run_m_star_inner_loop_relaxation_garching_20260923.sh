#!/usr/bin/env bash
# S3wrc-λ on Garching: for λ = 0.5 and 0.33, one single round from the A S3wr
# round-5 input (stage capture, then full-physics re-evaluation of its input and
# remapped states) and D/A 10-round trajectories, all with the written-gradient
# correction, the trial-temperature state refresh, and the correction hold.
# One single-threaded background process per job; production defaults stay off.
# Pass --resume to continue interrupted trajectories from their saved rounds.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
root="$PWD/results/m_star_inner_loop_relaxation_20260923"
osc_start="$PWD/results/m_star_inner_loop_state_refresh_20260923/a_pchip_s3wr/arrays/a_pchip_s3wr_it05.npz"
resume=()
[ "${1:-}" = "--resume" ] && resume=(--resume)
arm=(--correct-written-gradient --refresh-state --hold-correction)

pids=()
names=()
for spec in 0.5:l050 0.33:l033; do
  lam=${spec%%:*}
  tag=${spec##*:}
  mkdir -p "$root/osc_$tag/a"
  (
    "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_capture_20260922 \
      --case A --start-npz "$osc_start" --run-root "$root/osc_$tag/a" \
      "${arm[@]}" --relaxation "$lam" \
      >> "$root/osc_$tag/a/capture.log" 2>&1 \
    && "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_reeval_20260922 \
      --case A --capture-root "$root/osc_$tag" \
      --stages iteration_input standard_grid_remap \
      >> "$root/osc_$tag/a/reeval.log" 2>&1
  ) &
  pids+=("$!"); names+=("a_osc_$tag")
  for case in D A; do
    lower=$(echo "$case" | tr '[:upper:]' '[:lower:]')
    mkdir -p "$root/${lower}_pchip_s3wrc_$tag"
    "$PY" -u -m experiments.reduced_state_emulator.m_star_h2_inner_loop_s3_20260922 \
      --case "$case" "${arm[@]}" --relaxation "$lam" --result-root "$root" "${resume[@]}" \
      >> "$root/${lower}_pchip_s3wrc_$tag/trajectory.log" 2>&1 &
    pids+=("$!"); names+=("${lower}_trajectory_$tag")
  done
done

for index in "${!pids[@]}"; do
  echo "${names[$index]} ${pids[$index]}"
done | tee "$root/garching_pids.txt"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "all S3wrc-relaxation jobs finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
