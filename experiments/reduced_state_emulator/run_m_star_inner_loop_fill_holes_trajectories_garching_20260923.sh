#!/usr/bin/env bash
# S3wrc-λ-h trajectories on Garching: D and A 10-round trajectories with the
# written-gradient correction, the trial-temperature state refresh, the
# correction hold, λ = 0.5, and interior hole filling.  One single-threaded
# background process per job.  Pass --resume to continue from saved rounds.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
root="$PWD/results/m_star_inner_loop_fill_holes_20260923"
resume=()
[ "${1:-}" = "--resume" ] && resume=(--resume)
arm=(--correct-written-gradient --refresh-state --hold-correction --relaxation 0.5 --fill-holes)

pids=()
names=()
for case in D A; do
  lower=$(echo "$case" | tr '[:upper:]' '[:lower:]')
  mkdir -p "$root/${lower}_pchip_s3wrc_l050_h"
  "$PY" -u -m experiments.reduced_state_emulator.m_star_h2_inner_loop_s3_20260922 \
    --case "$case" "${arm[@]}" --result-root "$root" "${resume[@]}" \
    >> "$root/${lower}_pchip_s3wrc_l050_h/trajectory.log" 2>&1 &
  pids+=("$!"); names+=("${lower}_trajectory")
done
for index in "${!pids[@]}"; do
  echo "${names[$index]} ${pids[$index]}"
done | tee "$root/garching_trajectory_pids.txt"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "all fill-holes trajectory jobs finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
