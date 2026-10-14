#!/usr/bin/env bash
# Usage: run_coauthor_mgiant_apogee_fit_garching_20261013.sh WORKERS fit|reference [ARGS] < star ids
#        run_coauthor_mgiant_apogee_fit_garching_20261013.sh 1 calibrate [ARGS]
# One process per star from stdin, WORKERS at a time, four threads each.
set -euo pipefail
cd /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-apogee-fit-20261013
export PYTHONPATH="$PWD:$PWD/experiments/reduced_state_emulator"
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python
SCRIPT=experiments/reduced_state_emulator/coauthor_mgiant_apogee_fit_20261013.py
LOGS=results/coauthor_validation_20261005/mgiant_apogee_fit/logs
mkdir -p "$LOGS"
workers=$1 stage=$2
shift 2
tag=$(printf '%s' "$stage $*" | tr -c 'A-Za-z0-9_' '_')
if [ "$stage" = calibrate ]; then
    exec "$PY" -u "$SCRIPT" calibrate "$@" > "$LOGS/$tag.log" 2>&1
fi
xargs -P "$workers" -I{} sh -c "\"$PY\" -u $SCRIPT $stage --star {} $* > $LOGS/${tag}_{}.log 2>&1"
