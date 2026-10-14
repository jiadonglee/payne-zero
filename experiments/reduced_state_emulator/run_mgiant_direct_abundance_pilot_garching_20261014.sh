#!/usr/bin/env bash
# Usage: run_mgiant_direct_abundance_pilot_garching_20261014.sh run WORKERS|evaluate
set -euo pipefail
cd /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-coauthor-20261005
export PYTHONPATH="$PWD:$PWD/experiments/reduced_state_emulator"
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python
SCRIPT=experiments/reduced_state_emulator/mgiant_direct_abundance_pilot_20261014.py
ROOT=runs/mgiant_direct_abundance_pilot_20261014
ARGS=(--root "$ROOT" --parents results/coauthor_validation_20261005/marcs_structure/summary.json)
case "$1" in
    run) exec "$PY" -u "$SCRIPT" run "${ARGS[@]}" --networks results/coauthor_validation_20261005/mgiant_final \
            --workers "$2" ;;
    evaluate) exec "$PY" -u "$SCRIPT" evaluate "${ARGS[@]}" ;;
esac
