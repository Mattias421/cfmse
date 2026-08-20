#!/usr/bin/env bash
# Run augmented-DDP jobs on one or both of Mimas's two GPU pairs.
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "${repo_root}/xps/lib/runtime.sh" mimas

if ! command -v setsid >/dev/null 2>&1; then
    echo "setsid is required so cancellation reaches each DDP process group" >&2
    exit 1
fi

mode=${1:-all}
case ${mode} in
    all)
        indices=(0 1)
        gpu_pairs=(0,1 2,3)
        ;;
    finetune)
        indices=(0)
        gpu_pairs=(0,1)
        ;;
    anneal)
        indices=(1)
        gpu_pairs=(0,1)
        ;;
    *)
        echo "usage: $0 [all|finetune|anneal]" >&2
        exit 2
        ;;
esac

if [[ ${mode} != anneal && ${CFMSE_PHASE:-train} != evaluate ]]; then
    : "${CFMSE_ICFM_FULL_CKPT:?set CFMSE_ICFM_FULL_CKPT for the fine-tune job}"
fi

environment_args=(--require-cuda-build)
if [[ ${CFMSE_ALLOW_VERSION_DRIFT:-0} == 1 ]]; then
    environment_args+=(--allow-version-drift)
fi
"${CFMSE_PYTHON}" xps/check_environment.py "${environment_args[@]}"
"${CFMSE_PYTHON}" xps/validate_data.py --base-dir "${CFMSE_DATA_ROOT}"
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

pids=()
names=()
cleanup() {
    for pid in "${pids[@]:-}"; do
        # Each slot is its own session below; terminate the runner, Lightning
        # ranks, and DataLoader descendants together.
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
}
trap cleanup INT TERM

runner_args=()
if [[ ${CFMSE_RESUME:-0} == 1 ]]; then
    runner_args+=(--resume)
fi

for slot in "${!indices[@]}"; do
    experiment_index=${indices[$slot]}
    (
        export CUDA_VISIBLE_DEVICES=${gpu_pairs[$slot]}
        export MASTER_PORT=$(( ${CFMSE_MASTER_PORT_BASE:-12920} + slot ))
        export TORCH_EXTENSIONS_DIR="${CFMSE_TORCH_EXTENSIONS_BASE}/mimas_slot_${slot}"
        mkdir -p "${TORCH_EXTENSIONS_DIR}"
        exec setsid --wait "${CFMSE_PYTHON}" -u xps/run_experiment.py \
            --server mimas \
            --index "${experiment_index}" \
            --devices 2 \
            --global-batch-size "${CFMSE_GLOBAL_BATCH_SIZE:-16}" \
            --phase "${CFMSE_PHASE:-train}" \
            "${runner_args[@]}"
    ) &
    pids+=("$!")
    names+=("mimas index ${experiment_index}")
done

failed=0
for slot in "${!pids[@]}"; do
    if ! wait "${pids[$slot]}"; then
        echo "${names[$slot]} failed" >&2
        failed=1
    fi
done
exit "${failed}"
