#!/usr/bin/env bash
# Stage captures of the candidate's non-stationary rounds at g4.50 m+0.5
# 3600 K and g4.75 3850 K on Garching: twelve single-round captures, one
# single-threaded process each (~20 GB resident).
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
root="$PWD/results/m_star_inner_loop_nonstationary_capture_20260924"
module=experiments.reduced_state_emulator.m_star_inner_loop_nonstationary_capture_20260924
captures=(
  t3600_ref_r01 t3600_ref_r02 t3600_ref_r03 t3600_ref_r04
  t3600_warm_r59 t3600_warm_r60
  t3850_cont_primary_r02 t3850_cont_primary_r04 t3850_cont_primary_r06 t3850_cont_primary_r08
  t3850_cont_ref_r04 t3850_cont_ref_r08
)
mkdir -p "$root/logs"

pids=()
: > "$root/garching_pids.txt"
for capture in "${captures[@]}"; do
  "$PY" -u -m "$module" --capture "$capture" >> "$root/logs/${capture}.log" 2>&1 &
  pids+=("$!")
  echo "$capture $!" >> "$root/garching_pids.txt"
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
for capture in "${captures[@]}"; do
  grep -q "status=ok" "$root/logs/${capture}.log" || fail=1
done
echo "captures finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
