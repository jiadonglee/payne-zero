#!/usr/bin/env bash
# Usage: run_mdwarf_perturbation_test_garching_20261015.sh run WORKERS|collect
set -euo pipefail
cd /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-coauthor-20261005
export PYTHONPATH="$PWD:$PWD/experiments/reduced_state_emulator"
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python
SCRIPT=experiments/reduced_state_emulator/mdwarf_perturbation_test_20261015.py
ROOT=runs/mdwarf_perturbation_test_20261015
case "$1" in
    run) exec "$PY" -u "$SCRIPT" run --root "$ROOT" --products "$ROOT/products.json" --workers "$2" ;;
    collect) exec "$PY" -u "$SCRIPT" collect --root "$ROOT" ;;
esac
