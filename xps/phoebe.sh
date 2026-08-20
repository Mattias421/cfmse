#!/usr/bin/env bash
# Run the two direct baselines sequentially on Phoebe's GPU pair.
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CFMSE_LAUNCHER_NAME=phoebe
# shellcheck disable=SC1091
source "${repo_root}/xps/lib/runtime.sh" phoebe

environment_args=(--require-cuda-build)
if [[ ${CFMSE_ALLOW_VERSION_DRIFT:-0} == 1 ]]; then
    environment_args+=(--allow-version-drift)
fi
"${CFMSE_PYTHON}" xps/check_environment.py "${environment_args[@]}"
"${CFMSE_PYTHON}" xps/validate_data.py --base-dir "${CFMSE_DATA_ROOT}"
export CUDA_VISIBLE_DEVICES=${CFMSE_GPU_PAIR:-0,1}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export TORCH_EXTENSIONS_DIR="${CFMSE_TORCH_EXTENSIONS_BASE}/phoebe"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

runner_args=()
if [[ ${CFMSE_RESUME:-0} == 1 ]]; then
    runner_args+=(--resume)
fi

for experiment_index in 0 1; do
    export MASTER_PORT=$(( ${CFMSE_MASTER_PORT_BASE:-12910} + experiment_index ))
    "${CFMSE_PYTHON}" -u xps/run_experiment.py \
        --server phoebe \
        --index "${experiment_index}" \
        --devices 2 \
        --global-batch-size "${CFMSE_GLOBAL_BATCH_SIZE:-16}" \
        --phase "${CFMSE_PHASE:-train}" \
        "${runner_args[@]}"
done
echo "Phoebe experiments completed successfully."
