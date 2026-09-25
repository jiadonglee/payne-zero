#!/usr/bin/env bash
# Rerun of the development points A-E with the candidate and a
# three-iteration stop (arm candidate_strict_stop) on Garching: one
# single-threaded process per point (~20 GB resident each).
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
campaign=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-mstar-emulator-v1-20260831/results
synthesis_root=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero
root="$PWD/results/m_star_inner_loop_strict_stop_20260925/dev"
mkdir -p "$root/logs"

pids=()
: > "$root/garching_pids.txt"
for point in t3500 t3400 t3300 t3600 t3200; do
  "$PY" -u -m experiments.reduced_state_emulator.m_star_inner_loop_dev_sweep_20260923 \
    --arm candidate_strict_stop --only "$point" --result-root "$root" \
    --tomography-root "$campaign/m_star_iteration_tomography_v1" \
    --interp-root "$campaign/m_star_interpolated_mt_seed_v1" \
    --v1r2-root "$campaign/m_star_emulator_v1r2_marcs100" \
    --flux-gate "$campaign/m_star_iteration_tomography_v1/flux_gate.json" \
    --synthesis-root "$synthesis_root" \
    --preregistration "$PWD/notes/m_star_inner_loop_strict_stop_preregistration_20260925.md" \
    >> "$root/logs/candidate_strict_stop_${point}.log" 2>&1 &
  pids+=("$!")
  echo "candidate_strict_stop_${point} $!" >> "$root/garching_pids.txt"
done
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "strict-stop rerun dev solves finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
