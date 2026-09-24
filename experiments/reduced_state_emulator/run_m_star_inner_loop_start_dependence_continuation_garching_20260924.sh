#!/usr/bin/env bash
# Start-dependence continuation on Garching: the candidate's primary and
# reference-start products at three validation nodes, each continued a fixed
# ten iterations with the stop disabled (one single-threaded process each,
# ~20 GB resident), then the pairing step.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
synthesis_root=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero
root="$PWD/results/m_star_inner_loop_validation_20260924"
module=experiments.reduced_state_emulator.m_star_inner_loop_start_dependence_continuation_20260924
nodes=(
  g+4.50_m-1.00_a+0.00_c+0.00_x1.00_t3800
  g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3850
  g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t4000
)
mkdir -p "$root/continuation/logs"

pids=()
names=()
for node in "${nodes[@]}"; do
  for start in primary reference_start; do
    "$PY" -u -m "$module" --start "$start" --only "$node" --result-root "$root" \
      --synthesis-root "$synthesis_root" \
      >> "$root/continuation/logs/${start}_${node}.log" 2>&1 &
    pids+=("$!"); names+=("${start}_${node}")
  done
done
for index in "${!pids[@]}"; do
  echo "${names[$index]} ${pids[$index]}"
done | tee "$root/continuation/garching_pids.txt"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "continuation solves finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -u -m "$module" --pair --result-root "$root" --synthesis-root "$synthesis_root" \
  >> "$root/continuation/logs/pair.log" 2>&1 || fail=1
echo "continuation pairing finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
