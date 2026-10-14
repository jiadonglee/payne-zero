#!/usr/bin/env bash
# Usage: run_coauthor_mgiant_multiparent_garching_20261008.sh run|evaluate SHARD SHARDS|collect
set -euo pipefail
cd /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-coauthor-20261005
export PYTHONPATH="$PWD"
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python
ROOT=runs/mgiant_cno_multiparent_20261008
PARENTS=results/coauthor_validation_20261005/marcs_structure/summary.json
V4=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-mstar-emulator-v1-20260831/artifacts/m_star_emulator_mgiant_v4
DRIVER=coauthor_mgiant_cno_multiparent_20261008.py
case "$1" in
    run) exec "$PY" -u "$DRIVER" run --root "$ROOT" --parents "$PARENTS" --checkpoints "$V4" --workers 12 ;;
    evaluate) exec "$PY" -u "$DRIVER" evaluate --root "$ROOT" --parents "$PARENTS" --shard "$2" --shards "$3" ;;
    collect) exec "$PY" -u "$DRIVER" collect --root "$ROOT" --parents "$PARENTS" ;;
esac
