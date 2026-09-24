#!/usr/bin/env bash
# Payne Zero's own fitter on the seven ESO UVES M giants
# (experiments/mgiant_eso_bestfit_v1.py pz-fit): a continuous fast fit with the
# v4 M-giant start, then converged-atmosphere refinement with the unchanged
# solver. One process per star; a solver process holds about 20 GB.
#
# Run from the run directory, after one process has populated the synthesis
# cache for the fit window:
#   setsid nohup bash experiments/reduced_state_emulator/run_mgiant_eso_pzfit_garching_20260924.sh \
#       > logs/launcher.log 2>&1 < /dev/null &
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PYTHON=${PYTHON:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
THREADS=${THREADS:-8}
STARS=${STARS:-"alfCet alfTau psiPhe phiAqr mVir 87Vir ERVir"}
LINELIST=${LINELIST:-native}   # native, or galah_zro (run the zro-table stage first)

cd "$ROOT"
mkdir -p logs
export NUMBA_THREADING_LAYER=workqueue
export PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES=65536
export PAYNE_ZERO_SYNTHESIS_CACHE_DIR="$ROOT/.cache/payne-zero/synthesis"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

for star in $STARS; do
    setsid nohup "$PYTHON" experiments/mgiant_eso_bestfit_v1.py pz-fit --star "$star" \
        --device cpu --dtype float64 --linelist "$LINELIST" \
        > "logs/pzfit_${LINELIST}_${star}.log" 2>&1 < /dev/null &
    echo "$star $!"
done
