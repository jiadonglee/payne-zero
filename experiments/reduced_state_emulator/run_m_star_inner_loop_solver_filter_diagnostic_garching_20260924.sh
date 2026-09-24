#!/usr/bin/env bash
# Solver diagnostic of the mask-interior written-gradient filter at g4.75
# 3850 K on Garching: the candidate's primary and reference-start products,
# each continued ten iterations with the filter on (one single-threaded
# process each, ~20 GB resident), then the pairing step.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
synthesis_root=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero
root="$PWD/results/m_star_inner_loop_solver_filter_diagnostic_20260924"
module=experiments.reduced_state_emulator.m_star_inner_loop_solver_filter_diagnostic_20260924
mkdir -p "$root/logs"

pids=()
: > "$root/garching_pids.txt"
for start in primary reference_start; do
  "$PY" -u -m "$module" --start "$start" --synthesis-root "$synthesis_root" \
    >> "$root/logs/${start}.log" 2>&1 &
  pids+=("$!")
  echo "$start $!" >> "$root/garching_pids.txt"
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "filter diagnostic solves finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -u -m "$module" --pair --synthesis-root "$synthesis_root" >> "$root/logs/pair.log" 2>&1 || fail=1
echo "filter diagnostic pairing finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
