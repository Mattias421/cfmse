#!/usr/bin/env bash
# Shared environment setup. Source as: source xps/lib/runtime.sh <server>.

set -Eeuo pipefail

CFMSE_SERVER_NAME=${1:?server name is required}
CFMSE_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

CFMSE_SERVER_ENV=${CFMSE_SERVER_ENV:-"${CFMSE_REPO_ROOT}/xps/env/${CFMSE_SERVER_NAME}.env"}
if [[ -f ${CFMSE_SERVER_ENV} ]]; then
    # shellcheck disable=SC1090
    source "${CFMSE_SERVER_ENV}"
fi

if [[ -n ${CFMSE_MODULES:-} ]]; then
    if ! command -v module >/dev/null 2>&1; then
        echo "CFMSE_MODULES is set, but the module command is unavailable" >&2
        return 1 2>/dev/null || exit 1
    fi
    read -r -a cfmse_modules <<<"${CFMSE_MODULES}"
    module load "${cfmse_modules[@]}"
fi

if [[ -n ${CFMSE_ENV_ACTIVATE:-} ]]; then
    if [[ ! -f ${CFMSE_ENV_ACTIVATE} ]]; then
        echo "Environment activation script not found: ${CFMSE_ENV_ACTIVATE}" >&2
        return 1 2>/dev/null || exit 1
    fi
    # Activation scripts are not generally nounset-clean.
    set +u
    # shellcheck disable=SC1090
    source "${CFMSE_ENV_ACTIVATE}"
    set -u
fi

if [[ -z ${CFMSE_DATA_ROOT:-} && -n ${DATA:-} ]]; then
    CFMSE_DATA_ROOT=${DATA}/VB+DMD
fi
CFMSE_DATA_ROOT=${CFMSE_DATA_ROOT:?set CFMSE_DATA_ROOT in the server environment file}
CFMSE_LOG_ROOT=${CFMSE_LOG_ROOT:-"${CFMSE_REPO_ROOT}/logs/experiments"}
CFMSE_PYTHON=${CFMSE_PYTHON:-$(command -v python || true)}
if [[ -z ${CFMSE_PYTHON} || ! -x ${CFMSE_PYTHON} ]]; then
    echo "Set CFMSE_PYTHON to an executable Python in the project environment" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ ${CFMSE_SERVER_NAME} == stanage ]]; then
    case ${CFMSE_GPU_TYPE:-any} in
        a100)
            CFMSE_CUDA_ARCH_LIST=${CFMSE_CUDA_ARCH_LIST:-8.0}
            ;;
        h100|h100nvl)
            CFMSE_CUDA_ARCH_LIST=${CFMSE_CUDA_ARCH_LIST:-9.0}
            ;;
        any)
            CFMSE_CUDA_ARCH_LIST=${CFMSE_CUDA_ARCH_LIST:-"8.0;9.0"}
            ;;
        *)
            echo "Unsupported Stanage GPU type: ${CFMSE_GPU_TYPE}" >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
else
    CFMSE_CUDA_ARCH_LIST=${CFMSE_CUDA_ARCH_LIST:-8.6}
fi
CFMSE_TORCH_EXTENSIONS_BASE=${CFMSE_TORCH_EXTENSIONS_BASE:-"${TMPDIR:-/tmp}/cfmse_torch_extensions_sm${CFMSE_CUDA_ARCH_LIST//./}"}

export CFMSE_DATA_ROOT CFMSE_LOG_ROOT CFMSE_PYTHON CFMSE_REPO_ROOT
export CFMSE_CUDA_ARCH_LIST CFMSE_TORCH_EXTENSIONS_BASE
export PYTHONPATH="${CFMSE_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_CUDA_ARCH_LIST=${CFMSE_CUDA_ARCH_LIST}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "${CFMSE_REPO_ROOT}"
