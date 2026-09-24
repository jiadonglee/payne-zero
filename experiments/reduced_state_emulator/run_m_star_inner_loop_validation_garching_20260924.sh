#!/usr/bin/env bash
# Inner-loop validation on Garching: candidate and S0 (both on the pchip
# overlay) at the nine safezone v2 dwarf validation points.  Per arm and node,
# one single-threaded process solves from the frozen warm start (primary and
# self-restart) and one from the reference product; the pairing step runs once
# all of them have finished.  Reference products and the frozen flux gate come
# from the emulator-v1 campaign tree; emulator_v1_2 (TiO synthesis) from the
# payne-zero checkout.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
campaign_tree=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-mstar-emulator-v1-20260831
synthesis_root=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero
root="$PWD/results/m_star_inner_loop_validation_20260924"
module=experiments.reduced_state_emulator.m_star_inner_loop_validation_20260924
common=(--result-root "$root" --inputs "$root/inputs/warm_starts.npz"
        --reference-root "$campaign_tree"
        --flux-gate "$campaign_tree/results/m_star_iteration_tomography_v1/flux_gate.json"
        --synthesis-root "$synthesis_root")
nodes=(
  g+4.50_m-1.00_a+0.00_c+0.00_x1.00_t3800
  g+4.50_m+0.00_a+0.00_c+0.00_x1.00_t3400
  g+4.50_m+0.50_a+0.00_c+0.00_x1.00_t3600
  g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t4000
  g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3950
  g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3900
  g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3850
  g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3800
  g+4.75_m+0.00_a+0.00_c+0.00_x1.00_t3750
)
mkdir -p "$root/logs"

pids=()
names=()
for arm in candidate s0; do
  for start in warm reference; do
    for node in "${nodes[@]}"; do
      "$PY" -u -m "$module" --arm "$arm" --start "$start" --only "$node" "${common[@]}" \
        >> "$root/logs/${arm}_${start}_${node}.log" 2>&1 &
      pids+=("$!"); names+=("${arm}_${start}_${node}")
    done
  done
done
for index in "${!pids[@]}"; do
  echo "${names[$index]} ${pids[$index]}"
done | tee "$root/garching_pids.txt"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "all validation solves finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -u -m "$module" --pair "${common[@]}" >> "$root/logs/pair.log" 2>&1 || fail=1
echo "pairing finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
