#!/usr/bin/env bash
# Usage: run_mgiant_apogee_linebyline_garching_20261015.sh check STAR | calibrate
# GPU node; the refits use run_coauthor_mgiant_apogee_fit_garching_20261013.sh with
# --offsets results/coauthor_validation_20261005/mgiant_apogee_fit/linebyline/linebyline.json
set -euo pipefail
cd /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-apogee-fit-20261013
export PYTHONPATH="$PWD:$PWD/experiments/reduced_state_emulator"
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python
SCRIPT=experiments/reduced_state_emulator/mgiant_apogee_linebyline_20261015.py
LOGS=results/coauthor_validation_20261005/mgiant_apogee_fit/logs
mkdir -p "$LOGS"
case "$1" in
    check) exec "$PY" -u "$SCRIPT" check --star "$2" --device cuda > "$LOGS/linebyline_check_$2.log" 2>&1 ;;
    calibrate) exec "$PY" -u "$SCRIPT" calibrate --device cuda > "$LOGS/linebyline_calibrate.log" 2>&1 ;;
esac
