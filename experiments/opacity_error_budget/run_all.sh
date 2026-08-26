#!/bin/bash
# Serial 12-star decimation ladder. ONE solver process at a time: peak RSS is
# ~7.8 GB (briefly ~14 GB at startup), and worker sizing is RAM_GB/16, so this
# 16 GB machine holds exactly one. Never parallelise this loop here.
#
# Stratum order is deliberate: cool first. It is the stratum where decimation is
# predicted to hurt most and therefore the one that carries the decision; if the
# run dies partway the most informative rows are already on disk.
set -u
cd /Users/jdli/Project/payne-zero
LABELS=experiments/opacity_error_budget/labels.jsonl
OUT=runs/opacity_error_budget
STRIDES="1 2 4 8"

# cool (rows 10-12), then giant (7-9), solar (4-6), hot (1-3)
ORDER="10 11 12 7 8 9 4 5 6 1 2 3"
STRATUM=([10]=cool [11]=cool [12]=cool [7]=giant [8]=giant [9]=giant \
         [4]=solar [5]=solar [6]=solar [1]=hot [2]=hot [3]=hot)

for i in $ORDER; do
  row=$(sed -n "${i}p" "$LABELS")
  teff=$(echo "$row" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['effective_temperature'])")
  logg=$(echo "$row" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['log_surface_gravity'])")
  mh=$(echo   "$row" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['metallicity'])")
  am=$(echo   "$row" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['alpha_enhancement'])")
  xi=$(echo   "$row" | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['microturbulence_km_s'])")
  slug="row${i}_${STRATUM[$i]}_t${teff}_g${logg}_m${mh}"
  echo "@@@ START $slug"
  NUMBA_THREADING_LAYER=workqueue PYTHONPATH=. .venv/bin/python \
    -m experiments.opacity_decimation.run_decimation \
    --strides $STRIDES \
    --effective-temperature "$teff" --log-surface-gravity "$logg" \
    --metallicity "$mh" --alpha-enhancement "$am" --microturbulence-km-s "$xi" \
    --out "$OUT/$slug" --summary "$OUT/$slug/summary.json" 2>&1 \
    | sed "s/^/[$slug] /"
  echo "@@@ DONE $slug rc=$?"
done
echo "@@@ ALL DONE"
