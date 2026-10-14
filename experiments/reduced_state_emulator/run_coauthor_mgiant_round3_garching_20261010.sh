#!/usr/bin/env bash
# Usage: run_coauthor_mgiant_round3_garching_20261010.sh assemble|train|run|evaluate SHARD SHARDS|collect
set -euo pipefail
cd /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-coauthor-20261005
export PYTHONPATH="$PWD"
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python
NET=results/coauthor_validation_20261005/mgiant_round3
HOLDOUT=runs/mgiant_cno_multiparent_20261008/setup.json
COOL=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-mstar-emulator-v1-20260831/results/m_star_cool_corpus_mgiant_v2/cool_truth_corpus.npz
ROOT=runs/mgiant_cno_round3_20261010
SOLVE=(--root "$ROOT" --holdout "$HOLDOUT" --corpus "$NET/corpus/corpus.npz" --round 3)
case "$1" in
    assemble)
        exec "$PY" -u coauthor_mgiant_round2_corpus_20261009.py assemble --holdout "$HOLDOUT" --cool "$COOL" \
            --cool-dir results/coauthor_validation_20261005/mgiant_round2/corpus --out "$NET/corpus" \
            --cno results/coauthor_validation_20261005/mgiant_cno_development/corpus.npz \
                  runs/mgiant_cno_multiparent_20261008/cno_development_corpus.npz \
                  runs/mgiant_cno_round2_20261009/cno_development_corpus.npz ;;
    train)
        export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
        for profiles in 2 6; do
            "$PY" -u coauthor_mgiant_train_20261009.py \
                --corpus-root data/payne-zero-v1.3-atmosphere-training-corpora \
                --split results/coauthor_validation_20261005/cno8/split.json \
                --mgiant "$NET/corpus/corpus.npz" \
                --reference source_data_files/atmosphere_emulator/cno8/checkpoint.pt \
                --out "$NET" --profiles "$profiles" --device cuda --threads 2 \
                > "$NET/train_${profiles}profile.log" 2>&1
        done ;;
    run) exec "$PY" -u coauthor_mgiant_cno_round2_solve_20261009.py run "${SOLVE[@]}" --network "$NET/checkpoint_2profile.pt" --workers 10 ;;
    evaluate) exec "$PY" -u coauthor_mgiant_cno_round2_solve_20261009.py evaluate "${SOLVE[@]}" --shard "$2" --shards "$3" ;;
    collect) exec "$PY" -u coauthor_mgiant_cno_round2_solve_20261009.py collect "${SOLVE[@]}" ;;
esac
