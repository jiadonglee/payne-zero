#!/usr/bin/env bash
set -eu

mode="${1:?usage: run_mstar_v1r2_supervisor.sh MODE REPO_ROOT PYTHON}"
repo_root="${2:?missing repository root}"
python_bin="${3:?missing Python executable}"

campaign="m_star_emulator_v1r2_marcs100"
result_root="results/${campaign}"
artifact_root="artifacts/${campaign}"

cd "${repo_root}"

wait_for_corpus() {
    while ! test -f "${result_root}/CORPUS_READY"; do
        if test -f "${result_root}/GIANT_QUOTA_FAILED" \
            || test -f "${result_root}/DWARF_QUOTA_FAILED" \
            || test -f "${result_root}/CORPUS_BUILD_FAILED"; then
            return 20
        fi
        sleep 60
    done
}

case "${mode}" in
    corpus)
        while ! test -f "${result_root}/GIANT_QUOTA_REACHED" \
            || ! test -f "${result_root}/DWARF_QUOTA_REACHED"; do
            if test -f "${result_root}/GIANT_QUOTA_FAILED" \
                || test -f "${result_root}/DWARF_QUOTA_FAILED"; then
                touch "${result_root}/CORPUS_BUILD_FAILED"
                exit 20
            fi
            sleep 60
        done
        if env \
            NUMBA_THREADING_LAYER=workqueue \
            NUMBA_NUM_THREADS=1 \
            OMP_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
            PYTHONPATH=. \
            "${python_bin}" -m \
            experiments.reduced_state_emulator.m_star_bootstrap_v1r2_marcs100 \
            build-corpus; then
            touch "${result_root}/CORPUS_BUILD_FINISHED"
        else
            touch "${result_root}/CORPUS_BUILD_FAILED"
            exit 21
        fi
        ;;
    train)
        if ! wait_for_corpus; then
            mkdir -p "${artifact_root}"
            touch "${artifact_root}/TRAINING_FAILED"
            exit 20
        fi
        while true; do
            gpu_utilization="$(
                nvidia-smi \
                    --query-gpu=utilization.gpu \
                    --format=csv,noheader,nounits \
                | head -n 1 \
                | tr -d ' '
            )"
            gpu_memory_free="$(
                nvidia-smi \
                    --query-gpu=memory.free \
                    --format=csv,noheader,nounits \
                | head -n 1 \
                | tr -d ' '
            )"
            if test "${gpu_utilization}" -le 15 \
                && test "${gpu_memory_free}" -ge 60000; then
                break
            fi
            sleep 60
        done
        mkdir -p "${artifact_root}"
        if env \
            CUDA_VISIBLE_DEVICES=0 \
            NUMBA_THREADING_LAYER=workqueue \
            PYTHONPATH=. \
            "${python_bin}" -m \
            experiments.reduced_state_emulator.train_mstar_physical_v1 \
            --campaign "${campaign}" \
            --existing-corpus \
            source_data_files/atmosphere_emulator/five_label/strict_truth_52199.npz \
            --cool-corpus "${result_root}/cool_truth_corpus.npz" \
            --out "${artifact_root}" \
            --device cuda \
            --dtype float64 \
            --epochs 300 \
            --patience 60 \
            --seeds 20260831,20260901,20260902; then
            touch "${artifact_root}/TRAINING_SUCCESS"
        else
            touch "${artifact_root}/TRAINING_FAILED"
            exit 22
        fi
        ;;
    validate)
        while ! test -f "${artifact_root}/TRAINING_SUCCESS"; do
            if test -f "${artifact_root}/TRAINING_FAILED"; then
                touch "${result_root}/VALIDATION_BLOCKED_BY_TRAINING"
                exit 20
            fi
            sleep 60
        done
        if env \
            NUMBA_THREADING_LAYER=workqueue \
            NUMBA_NUM_THREADS=1 \
            OMP_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            OPENBLAS_NUM_THREADS=1 \
            PYTHONPATH=. \
            "${python_bin}" -m \
            experiments.reduced_state_emulator.evaluate_mstar_candidate_v1 \
            --campaign "${campaign}" \
            --cool-corpus "${result_root}/cool_truth_corpus.npz" \
            --checkpoint-dir "${artifact_root}" \
            --flux-gate "${result_root}/flux_gate.json" \
            --out "${result_root}/candidate_validation" \
            --workers 8 \
            --iteration-cap 30; then
            touch "${result_root}/VALIDATION_FINISHED"
        else
            touch "${result_root}/VALIDATION_FAILED"
            exit 23
        fi
        ;;
    *)
        echo "unknown mode: ${mode}" >&2
        exit 2
        ;;
esac
