#!/usr/bin/env bash
# Interpolation baseline on development-60 on Garching
# (notes/interpolation_baseline_dev60_preregistration_20260923.md).
#
# Solves the six-field control, the full-state interpolation arm (I6), and the
# (m,T) interpolation arm (I2) on the 60 development stars with the solver code
# already in this tree, then gates the converged I6 and I2 spectra against the
# frozen six-field products. An arm whose convergence JSON exists is skipped,
# so the script can be rerun after an interruption.
set -u
cd "$(dirname "$0")/../.."

PY=${PY:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
WORKERS=${WORKERS:-12}
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export NUMBA_THREADING_LAYER=workqueue
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

manifest=results/reconstruction_metrics.json
run=runs/interpolation_baseline_dev60_20260923
res=results/interpolation_baseline_dev60_20260923
frozen_products=runs/paper_physical_seed_20260820/learned/products/production_six_field
frozen_spectra=runs/paper_physical_seed_20260820/learned/spectra/production_six_field
mkdir -p "$run/control" "$res/control"

solve() {
  local arm=$1 run_root=$2 result_root=$3
  if [ -f "$result_root/convergence_$arm.json" ]; then
    echo "skip $arm: $result_root/convergence_$arm.json exists"
    return 0
  fi
  echo "solve $arm $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PY" -u -m experiments.reduced_state_emulator.grey_start_benchmark \
    --arm "$arm" --manifest "$manifest" --workers "$WORKERS" \
    --run-root "$run_root" --result-root "$result_root" \
    --interpolation-neighbours 8 --interpolation-power 2.0 \
    >> "$result_root/solve_$arm.log" 2>&1
}

fail=0
solve production_six_field "$run/control" "$res/control" || fail=1
solve interpolated_full_state "$run" "$res" || fail=1
solve interpolated_grid "$run" "$res" || fail=1

# Spectral gate: frozen six-field products as the baseline arm, the new arms as
# candidates. Cached six-field spectra are copied, never linked, so the three
# missing ones are written here rather than into the frozen run.
gate=$run/gate
mkdir -p "$gate/products" "$gate/spectra/production_six_field"
ln -sfn "$PWD/$frozen_products" "$gate/products/production_six_field"
cp -n "$frozen_spectra"/*.npz "$gate/spectra/production_six_field/"
for arm in interpolated_full_state interpolated_grid; do
  ln -sfn "$PWD/$run/products/$arm" "$gate/products/$arm"
  out="$res/spectral_${arm}_vs_six.json"
  [ -f "$out" ] && { echo "skip spectra $arm"; continue; }
  echo "spectra $arm $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  "$PY" -u -m experiments.reduced_state_emulator.spectral_gate \
    --products-dir "$gate/products" --spectra-dir "$gate/spectra" \
    --out "$out" --baseline-arm production_six_field --candidate-arm "$arm" \
    --workers "$WORKERS" --dtype float64 \
    --wavelength-start-nm 400 --wavelength-end-nm 900 --resolution 20000 \
    >> "$res/spectra_$arm.log" 2>&1
  # A gate exit status of 1 records a spectral discrepancy, not an execution error.
  status=$?
  [ "$status" -gt 1 ] && fail=1
done

echo "interpolation baseline finished fail=$fail $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$fail"
