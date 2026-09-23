#!/usr/bin/env bash
# Development-point sweep on Garching: candidate and S0 (both on the pchip
# overlay) at A 3500 K, B 3400 K, C 3300 K, D 3600 K and E 3200 K, one
# single-threaded background process per arm and point.  Seeds, the frozen flux
# gate and the tomography products come from the emulator-v1 campaign directory;
# emulator_v1_2 (TiO synthesis) from the payne-zero checkout.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PWD/.matplotlib"
campaign=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-mstar-emulator-v1-20260831/results
synthesis_root=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero
root="$PWD/results/m_star_inner_loop_dev_sweep_20260923"
mkdir -p "$root/logs"

pids=()
names=()
for arm in candidate s0; do
  for point in t3500 t3400 t3300 t3600 t3200; do
    "$PY" -u -m experiments.reduced_state_emulator.m_star_inner_loop_dev_sweep_20260923 \
      --arm "$arm" --only "$point" --result-root "$root" \
      --tomography-root "$campaign/m_star_iteration_tomography_v1" \
      --interp-root "$campaign/m_star_interpolated_mt_seed_v1" \
      --v1r2-root "$campaign/m_star_emulator_v1r2_marcs100" \
      --flux-gate "$campaign/m_star_iteration_tomography_v1/flux_gate.json" \
      --synthesis-root "$synthesis_root" \
      >> "$root/logs/${arm}_${point}.log" 2>&1 &
    pids+=("$!"); names+=("${arm}_${point}")
  done
done
for index in "${!pids[@]}"; do
  echo "${names[$index]} ${pids[$index]}"
done | tee "$root/garching_pids.txt"
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
echo "all dev-sweep jobs finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
