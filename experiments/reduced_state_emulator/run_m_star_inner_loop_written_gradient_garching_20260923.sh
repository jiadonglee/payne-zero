#!/usr/bin/env bash
# S3w stage 2 on Garching: D and A 10-round trajectories, and the A-plat
# single round (stage capture, then full-physics re-evaluation of its states).
# One single-threaded background process per job; production defaults stay off.
# Pass --resume to continue interrupted trajectories from their saved rounds.
# The A re-evaluation runs without --with-control-anchor: the reeval script's
# anchor check is the D control value (0.8081).
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
root="$PWD/results/m_star_inner_loop_written_gradient_20260923"
mkdir -p "$root/d_pchip_s3w" "$root/a_pchip_s3w" "$root/s3w/a"
resume=()
[ "${1:-}" = "--resume" ] && resume=(--resume)

pids=()
for case in D A; do
  lower=$(echo "$case" | tr '[:upper:]' '[:lower:]')
  "$PY" -u -m experiments.reduced_state_emulator.m_star_h2_inner_loop_s3_20260922 \
    --case "$case" --correct-written-gradient --result-root "$root" "${resume[@]}" \
    >> "$root/${lower}_pchip_s3w/trajectory.log" 2>&1 &
  pids+=("$!")
  echo "launched ${case} S3w trajectory pid=$!"
done

(
  "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_capture_20260922 \
    --case A --run-root "$root/s3w/a" --correct-written-gradient "${resume[@]}" \
    >> "$root/s3w/a/capture.log" 2>&1 \
  && "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_reeval_20260922 \
    --case A --capture-root "$root/s3w" \
    >> "$root/s3w/a/reeval.log" 2>&1
) &
pids+=("$!")
echo "launched A-plat S3w capture+reeval pid=$!"

printf 'd_trajectory %s\na_trajectory %s\na_plat %s\n' "${pids[@]}" > "$root/garching_pids.txt"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "all S3w stage-2 jobs finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
