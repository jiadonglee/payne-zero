#!/usr/bin/env bash
# Payne Zero's M-giant fitter on the eight DR20 APOGEE M giants of
# mgiant_aspcap_fit_v1 (experiments/mgiant_apogee_pzfit_v1.py): a continuous
# fast fit with the v4 M-giant start, then converged-atmosphere refinement with
# the unchanged solver. One process per star; a solver process holds about 20 GB.
#
# Run from the run directory:
#   setsid nohup bash experiments/reduced_state_emulator/run_mgiant_apogee_pzfit_garching_20260927.sh \
#       > logs/apogee_launcher.log 2>&1 < /dev/null &
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PYTHON=${PYTHON:-/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python}
THREADS=${THREADS:-4}
STAGE=${STAGE:-fast}   # fast (five parameters) or free (ten parameters, after the matching fast arm)
LINELIST=${LINELIST:-native}   # free stage only: native, apogee_atoms, apogee_molec, apogee_all
CALIBRATION=${CALIBRATION:-}   # optional schema-4 atomic overlay, e.g. linelist_calibration/data/sun_arcturus_fts_hband_shared.npz
STARS=${STARS:-"86810089 65420634 66450191 114992838 54948768 96257659 74858601 77720555"}

cd "$ROOT"
mkdir -p logs
export NUMBA_THREADING_LAYER=workqueue
export PAYNE_ZERO_SYNTHESIS_MOLECULAR_CHUNK_LINES=65536
export PAYNE_ZERO_SYNTHESIS_CACHE_DIR="$ROOT/.cache/payne-zero/synthesis"
export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS

for star in $STARS; do
    if [ "$STAGE" = free ]; then extra="--linelist $LINELIST"; tag="_free"; else extra="--physical"; tag=""; fi
    [ "$STAGE" = free ] && [ "$LINELIST" != native ] && tag="_free_${LINELIST}"
    setsid nohup "$PYTHON" experiments/mgiant_apogee_pzfit_v1.py "$STAGE" --star "$star" \
        --device cpu --dtype float64 $extra ${CALIBRATION:+--atomic-calibration "$CALIBRATION"} \
        > "logs/apogee_pzfit${tag}${CALIBRATION:+_$(basename "$CALIBRATION" .npz)}_${star}.log" 2>&1 < /dev/null &
    echo "$star $!"
done
