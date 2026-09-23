#!/usr/bin/env bash
# S3wr on Garching: D-plat and A-plat single rounds (stage capture, then
# full-physics re-evaluation) and D/A 10-round trajectories, all with the
# written-gradient correction and the trial-temperature state refresh.
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
root="$PWD/results/m_star_inner_loop_state_refresh_20260923"
mkdir -p "$root/d_pchip_s3wr" "$root/a_pchip_s3wr" "$root/s3wr/d" "$root/s3wr/a"
resume=()
[ "${1:-}" = "--resume" ] && resume=(--resume)
arm=(--correct-written-gradient --refresh-state)

pids=()
names=()
for case in D A; do
  lower=$(echo "$case" | tr '[:upper:]' '[:lower:]')
  anchor=()
  [ "$case" = "D" ] && anchor=(--with-control-anchor)
  (
    "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_capture_20260922 \
      --case "$case" --run-root "$root/s3wr/$lower" "${arm[@]}" "${resume[@]}" \
      >> "$root/s3wr/$lower/capture.log" 2>&1 \
    && "$PY" -u -m experiments.reduced_state_emulator.m_star_s3_stage_reeval_20260922 \
      --case "$case" --capture-root "$root/s3wr" "${anchor[@]}" \
      >> "$root/s3wr/$lower/reeval.log" 2>&1
  ) &
  pids+=("$!"); names+=("${lower}_plat")
  echo "launched ${case}-plat S3wr capture+reeval pid=$!"

  "$PY" -u -m experiments.reduced_state_emulator.m_star_h2_inner_loop_s3_20260922 \
    --case "$case" "${arm[@]}" --result-root "$root" "${resume[@]}" \
    >> "$root/${lower}_pchip_s3wr/trajectory.log" 2>&1 &
  pids+=("$!"); names+=("${lower}_trajectory")
  echo "launched ${case} S3wr trajectory pid=$!"
done

for index in "${!pids[@]}"; do
  echo "${names[$index]} ${pids[$index]}"
done > "$root/garching_pids.txt"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "all S3wr jobs finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
