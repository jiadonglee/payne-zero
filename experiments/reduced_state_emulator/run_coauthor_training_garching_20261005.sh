#!/usr/bin/env bash
set -euo pipefail
cd /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-coauthor-20261005
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python
CORPUS=data/payne-zero-v1.3-atmosphere-training-corpora
OUT=results/coauthor_validation_20261005
mkdir -p "$OUT"
for family in five_label cno8 direct_abundance; do
    learning_rate=0.001
    if [[ "$family" == direct_abundance ]]; then learning_rate=0.0003; fi
    "$PY" -u coauthor_train_20261005.py --corpus-root "$CORPUS" \
        --family "$family" --out "$OUT" --prepare-only
    for profiles in 2 6; do
        "$PY" -u coauthor_train_20261005.py --corpus-root "$CORPUS" \
            --family "$family" --profiles "$profiles" --out "$OUT" \
            --device cuda --threads 2 --epochs 300 --learning-rate "$learning_rate" \
            > "$OUT/${family}/train_${profiles}profile.log" 2>&1
    done
done
