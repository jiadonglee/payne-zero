#!/usr/bin/env bash
# Rerun of the validation points with the candidate plus the written-gradient
# filter (arm candidate_filter) on Garching: per node one single-threaded
# process from the frozen warm start (primary and self-restart) and one from
# the reference product (~20 GB resident each), then the pairing step.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
campaign_tree=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-mstar-emulator-v1-20260831
synthesis_root=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero
root="$PWD/results/m_star_inner_loop_filter_rerun_20260924/validation"
module=experiments.reduced_state_emulator.m_star_inner_loop_validation_20260924
common=(--result-root "$root"
        --inputs "$PWD/results/m_star_inner_loop_validation_20260924/inputs/warm_starts.npz"
        --reference-root "$campaign_tree"
        --flux-gate "$campaign_tree/results/m_star_iteration_tomography_v1/flux_gate.json"
        --synthesis-root "$synthesis_root"
        --preregistration "$PWD/notes/m_star_inner_loop_filter_rerun_preregistration_20260924.md")
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
: > "$root/garching_pids.txt"
for start in warm reference; do
  for node in "${nodes[@]}"; do
    "$PY" -u -m "$module" --arm candidate_filter --start "$start" --only "$node" "${common[@]}" \
      >> "$root/logs/candidate_filter_${start}_${node}.log" 2>&1 &
    pids+=("$!")
    echo "candidate_filter_${start}_${node} $!" >> "$root/garching_pids.txt"
  done
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "filter rerun validation solves finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"$PY" -u -m "$module" --pair "${common[@]}" >> "$root/logs/pair.log" 2>&1 || fail=1
echo "filter rerun validation pairing finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
