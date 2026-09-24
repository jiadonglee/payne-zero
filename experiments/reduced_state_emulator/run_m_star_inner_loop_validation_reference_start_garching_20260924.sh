#!/usr/bin/env bash
# Reference-start solves of the inner-loop validation on a second Garching
# node (both arms, nine nodes, one single-threaded process each; ~20 GB
# resident per process), while the warm-start solves of
# run_m_star_inner_loop_validation_garching_20260924.sh run elsewhere.  The
# pairing step runs once these solves and that launcher have both finished.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
campaign_tree=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-mstar-emulator-v1-20260831
synthesis_root=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero
root="$PWD/results/m_star_inner_loop_validation_20260924"
warm_launcher_log="$PWD/logs/validation_launcher.log"
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
  for node in "${nodes[@]}"; do
    "$PY" -u -m "$module" --arm "$arm" --start reference --only "$node" "${common[@]}" \
      >> "$root/logs/${arm}_reference_${node}.log" 2>&1 &
    pids+=("$!"); names+=("${arm}_reference_${node}")
  done
done
for index in "${!pids[@]}"; do
  echo "${names[$index]} ${pids[$index]}"
done | tee "$root/garching_pids_reference_start.txt"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "reference-start solves finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
until grep -q "pairing finished" "$warm_launcher_log" 2>/dev/null; do
  sleep 60
done
"$PY" -u -m "$module" --pair "${common[@]}" >> "$root/logs/pair.log" 2>&1 || fail=1
echo "pairing finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
