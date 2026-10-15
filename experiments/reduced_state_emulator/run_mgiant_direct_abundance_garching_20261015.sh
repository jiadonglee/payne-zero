#!/usr/bin/env bash
# Usage: run_mgiant_direct_abundance_garching_20261015.sh
#   round N run SHARD SHARDS WORKERS | round N evaluate SHARD SHARDS | round N collect
#   assemble N | train N
#   test run SHARD SHARDS WORKERS | test evaluate SHARD SHARDS | test collect
set -euo pipefail
cd /nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero-coauthor-20261005
export PYTHONPATH="$PWD:$PWD/experiments/reduced_state_emulator"
export NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/nexus/posix0/MIA-astro-env/hxr/jdli/payne-zero/.venv-linux/bin/python
DIR=experiments/reduced_state_emulator
OUT=results/coauthor_validation_20261005/mgiant_direct_abundance
HOLDOUT=runs/mgiant_cno_multiparent_20261008/setup.json
EIGHT=results/coauthor_validation_20261005/mgiant_final
round_root() { echo "runs/mgiant_direct_abundance_round$1_20261015"; }
case "$1" in
    round)
        N=$2
        if [ "$N" = 1 ]; then CORPUS="$EIGHT/corpus/corpus.npz"; NET="$EIGHT/checkpoint_2profile.pt"
        else CORPUS="$OUT/round$((N - 1))/corpus/corpus.npz"; NET="$OUT/round$((N - 1))/checkpoint_2profile.pt"; fi
        ARGS=(--round "$N" --root "$(round_root "$N")" --holdout "$HOLDOUT" --corpus "$CORPUS")
        case "$3" in
            run) exec "$PY" -u "$DIR/mgiant_direct_abundance_bootstrap_20261015.py" run "${ARGS[@]}" --network "$NET" \
                    --shard "$4" --shards "$5" --workers "$6" ;;
            evaluate) exec "$PY" -u "$DIR/mgiant_direct_abundance_bootstrap_20261015.py" evaluate "${ARGS[@]}" --shard "$4" --shards "$5" ;;
            collect) exec "$PY" -u "$DIR/mgiant_direct_abundance_bootstrap_20261015.py" collect "${ARGS[@]}" ;;
        esac ;;
    assemble)
        ROUNDS=()
        for n in $(seq 1 "$2"); do ROUNDS+=("$(round_root "$n")/direct_development_corpus.npz"); done
        exec "$PY" -u "$DIR/mgiant_direct_abundance_bootstrap_20261015.py" assemble --corpus "$EIGHT/corpus/corpus.npz" \
            --rounds "${ROUNDS[@]}" --out "$OUT/round$2/corpus" ;;
    train)
        export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
        for profiles in 2 6; do
            "$PY" -u "$DIR/mgiant_direct_abundance_train_20261015.py" \
                --corpus-root data/payne-zero-v1.3-atmosphere-training-corpora \
                --split results/coauthor_validation_20261005/direct_abundance/split.json \
                --mgiant "$OUT/round$2/corpus/corpus.npz" \
                --reference source_data_files/atmosphere_emulator/direct_abundance/checkpoint.pt \
                --out "$OUT/round$2" --profiles "$profiles" --device cuda --threads 2 \
                > "$OUT/round$2/train_${profiles}profile.log" 2>&1
        done ;;
    test)
        TEST=(--root runs/mgiant_direct_abundance_test_20261015 --holdout "$HOLDOUT")
        case "$2" in
            run) exec "$PY" -u "$DIR/mgiant_direct_abundance_test_20261015.py" run "${TEST[@]}" \
                    --parents results/coauthor_validation_20261005/marcs_structure/summary.json \
                    --networks "$OUT/round2" --eight-label "$EIGHT/checkpoint_2profile.pt" \
                    --marcs-grid SDSS_MARCS_atmospheres.h5 --shard "$3" --shards "$4" --workers "$5" ;;
            evaluate) exec "$PY" -u "$DIR/mgiant_direct_abundance_test_20261015.py" evaluate "${TEST[@]}" --shard "$3" --shards "$4" ;;
            collect) exec "$PY" -u "$DIR/mgiant_direct_abundance_test_20261015.py" collect "${TEST[@]}" ;;
        esac ;;
esac
