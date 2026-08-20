#!/bin/bash
#SBATCH --job-name=cfmse-main
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=80:00:00
#SBATCH --array=0-17%6
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -Eeuo pipefail

# Slurm executes a spooled copy of this file, so BASH_SOURCE does not point at
# the checkout. submit.sh exports the absolute root; direct submissions must be
# made from the repository root.
repo_root=${CFMSE_REPO_ROOT:-${SLURM_SUBMIT_DIR:?}}
if [[ ! -f ${repo_root}/xps/lib/runtime.sh ]]; then
    echo "Repository not found at ${repo_root}; submit with xps/stanage/submit.sh" >&2
    exit 1
fi
# shellcheck disable=SC1091
source "${repo_root}/xps/lib/runtime.sh" stanage

# runtime.sh also runs during submission, where TMPDIR is not Slurm job-local.
# Detect the GPU Slurm selected from the A100/H100/H100-NVL pool, then compile
# the CUDA extensions only for that device's architecture.
: "${TMPDIR:?Stanage did not provide a job-local TMPDIR}"
gpu_info=$("${CFMSE_PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU is visible inside the Stanage allocation")
name = torch.cuda.get_device_name(0).lower()
major, minor = torch.cuda.get_device_capability(0)
if "a100" in name:
    gpu_type = "a100"
elif "h100" in name and "nvl" in name:
    gpu_type = "h100nvl"
elif "h100" in name:
    gpu_type = "h100"
else:
    raise SystemExit(f"Unsupported GPU selected by Slurm: {name}")
print(gpu_type, f"{major}.{minor}")
PY
)
read -r CFMSE_GPU_TYPE CFMSE_CUDA_ARCH_LIST <<<"${gpu_info}"
export CFMSE_GPU_TYPE CFMSE_CUDA_ARCH_LIST
export TORCH_CUDA_ARCH_LIST=${CFMSE_CUDA_ARCH_LIST}
cuda_arch_tag=${CFMSE_CUDA_ARCH_LIST//./}
CFMSE_TORCH_EXTENSIONS_BASE=${TMPDIR}/cfmse_torch_extensions_${CFMSE_GPU_TYPE}_sm${cuda_arch_tag}
export CFMSE_TORCH_EXTENSIONS_BASE
export TORCH_EXTENSIONS_DIR="${CFMSE_TORCH_EXTENSIONS_BASE}/array_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TORCH_EXTENSIONS_DIR}"

# VB+DMD is only about 7.5 GB but contains ~25k small WAVs. Staging once avoids
# reopening those files from Lustre throughout 300 epochs. $TMPDIR is job-local
# and is deleted automatically by Slurm; checkpoints remain in CFMSE_LOG_ROOT.
if [[ ${CFMSE_STAGE_DATA:-1} == 1 ]]; then
    staged_data=${TMPDIR}/VB+DMD
    mkdir -p "${staged_data}"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "${CFMSE_DATA_ROOT}/" "${staged_data}/"
    else
        cp -a "${CFMSE_DATA_ROOT}/." "${staged_data}/"
    fi
    export CFMSE_DATA_ROOT=${staged_data}
fi

runner_args=()
if [[ ${CFMSE_RESUME:-0} == 1 ]]; then
    runner_args+=(--resume)
fi

srun --ntasks=1 "${CFMSE_PYTHON}" -u xps/run_experiment.py \
    --server stanage \
    --index "${SLURM_ARRAY_TASK_ID}" \
    --devices 1 \
    --global-batch-size "${CFMSE_GLOBAL_BATCH_SIZE:-16}" \
    --phase "${CFMSE_PHASE:-train}" \
    "${runner_args[@]}"
