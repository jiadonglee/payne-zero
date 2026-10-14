#!/usr/bin/env bash
# Benchmark giants: reference solves -> TiO calibration -> calibrated fits; with "all", the default-list fits run alongside.
set -uo pipefail
cd /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-apogee-fit-20261013
export PYTHONPATH="$PWD:$PWD/experiments/reduced_state_emulator"
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python
SCRIPT=experiments/reduced_state_emulator/coauthor_mgiant_benchmark_fit_20261013.py
OUT=results/coauthor_validation_20261005/mgiant_benchmark_fit
mkdir -p "$OUT/logs"
if [ "${1:-all}" = all ]; then
    for star in alfCet alfTau gamSge; do
        "$PY" -u "$SCRIPT" fit --star "$star" > "$OUT/logs/fit_default_$star.log" 2>&1 &
    done
fi
references=()
for star in alfCet alfTau; do
    "$PY" -u "$SCRIPT" reference --star "$star" > "$OUT/logs/reference_$star.log" 2>&1 &
    references+=($!)
done
wait "${references[@]}"
"$PY" -u "$SCRIPT" calibrate --max-iterations 6 > "$OUT/logs/calibrate.log" 2>&1
for star in alfCet alfTau gamSge; do
    "$PY" -u "$SCRIPT" fit --star "$star" --offsets "$OUT/calibration/offsets.json" \
        > "$OUT/logs/fit_calibrated_$star.log" 2>&1 &
done
wait
