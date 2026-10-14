#!/usr/bin/env bash
# Usage: run_coauthor_mgiant_round2_solve_garching_20261009.sh run|evaluate SHARD SHARDS|collect
set -euo pipefail
cd /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-coauthor-20261005
export PYTHONPATH="$PWD"
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python
ROOT=runs/mgiant_cno_round2_20261009
NET=results/coauthor_validation_20261005/mgiant_round2
COMMON=(--root "$ROOT" --holdout runs/mgiant_cno_multiparent_20261008/setup.json --corpus "$NET/corpus/corpus.npz")
DRIVER=coauthor_mgiant_cno_round2_solve_20261009.py
case "$1" in
    run) exec "$PY" -u "$DRIVER" run "${COMMON[@]}" --network "$NET/checkpoint_2profile.pt" --workers 11 ;;
    evaluate) exec "$PY" -u "$DRIVER" evaluate "${COMMON[@]}" --shard "$2" --shards "$3" ;;
    collect) exec "$PY" -u "$DRIVER" collect "${COMMON[@]}" ;;
esac
