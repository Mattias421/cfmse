#!/usr/bin/env bash
# Shared environment setup. Source as: source xps/lib/runtime.sh <server>.

set -Eeuo pipefail

CFMSE_SERVER_NAME=${1:?server name is required}
cfmse_checkout_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

# All three machines use the same layout.  Keep host env files optional: they
# are now only needed for genuine exceptions to this contract.
CFMSE_REPO_ROOT=${CFMSE_REPO_ROOT:-${EXP:+${EXP}/cfmse}}
CFMSE_REPO_ROOT=${CFMSE_REPO_ROOT:-${cfmse_checkout_root}}
CFMSE_DATA_ROOT=${CFMSE_DATA_ROOT:-${DATA:+${DATA}/VB+DMD}}
CFMSE_LOG_ROOT=${CFMSE_LOG_ROOT:-${CFMSE_REPO_ROOT}/logs}
CFMSE_PYTHON=${CFMSE_PYTHON:-${CFMSE_REPO_ROOT}/xps/uv-python}
CFMSE_UV=${CFMSE_UV:-$(command -v uv || true)}

cfmse_pause_after_error() {
    if [[ -t 0 && ${CFMSE_PAUSE_ON_ERROR:-1} == 1 ]]; then
        read -r -p "Press Enter to close this launcher... " _cfmse_unused || true
    fi
}

cfmse_error_handler() {
    local status=${1:-1}
    local line=${2:-unknown}
    local command=${3:-unknown}
    set +e
    trap - ERR
    printf '\nERROR: %s failed with status %s at line %s\n' \
        "${CFMSE_SERVER_NAME}" "${status}" "${line}" >&2
    printf 'Command: %s\n' "${command}" >&2
    if [[ -n ${CFMSE_LAUNCH_LOG:-} ]]; then
        printf 'Full launcher output: %s\n' "${CFMSE_LAUNCH_LOG}" >&2
    fi
    cfmse_pause_after_error
    exit "${status}"
}

cfmse_install_error_handler() {
    trap 'cfmse_error_handler "$?" "${LINENO}" "${BASH_COMMAND}"' ERR
}

cfmse_enable_launcher_reporting() {
    local launcher_name=${1:?launcher name is required}
    local timestamp
    cfmse_install_error_handler
    timestamp=$(date -u +%Y%m%dT%H%M%SZ)
    mkdir -p "${CFMSE_LOG_ROOT}/launcher"
    CFMSE_LAUNCH_LOG="${CFMSE_LOG_ROOT}/launcher/${launcher_name}-${timestamp}-$$.log"
    # This describes the current wrapper only; do not leak it into Slurm jobs or
    # Lightning child processes as though it were their own error transcript.
    export -n CFMSE_LAUNCH_LOG
    exec > >(tee -a "${CFMSE_LAUNCH_LOG}") 2>&1
    printf 'Launcher output: %s\n' "${CFMSE_LAUNCH_LOG}"
}

cfmse_runtime_fail() {
    local message=${1:?error message is required}
    printf '%s\n' "${message}" >&2
    if [[ -n ${CFMSE_LAUNCHER_NAME:-} ]]; then
        cfmse_error_handler 1 "${BASH_LINENO[0]}" "runtime environment check"
    fi
    return 1
}

# Interactive wrappers set this before sourcing runtime.sh so failures in the
# optional host env, module setup, or uv checks are also retained and visible.
if [[ -n ${CFMSE_LAUNCHER_NAME:-} ]]; then
    cfmse_enable_launcher_reporting "${CFMSE_LAUNCHER_NAME}"
fi

CFMSE_SERVER_ENV=${CFMSE_SERVER_ENV:-"${CFMSE_REPO_ROOT}/xps/env/${CFMSE_SERVER_NAME}.env"}
if [[ -f ${CFMSE_SERVER_ENV} ]]; then
    # shellcheck disable=SC1090
    source "${CFMSE_SERVER_ENV}"
fi

if [[ -n ${CFMSE_MODULES:-} ]]; then
    if ! command -v module >/dev/null 2>&1; then
        cfmse_runtime_fail "CFMSE_MODULES is set, but the module command is unavailable" \
            || return 1 2>/dev/null || exit 1
    fi
    read -r -a cfmse_modules <<<"${CFMSE_MODULES}"
    module load "${cfmse_modules[@]}"
fi

if [[ -n ${CFMSE_ENV_ACTIVATE:-} ]]; then
    if [[ ! -f ${CFMSE_ENV_ACTIVATE} ]]; then
        cfmse_runtime_fail "Environment activation script not found: ${CFMSE_ENV_ACTIVATE}" \
            || return 1 2>/dev/null || exit 1
    fi
    # Activation scripts are not generally nounset-clean.
    set +u
    # shellcheck disable=SC1090
    source "${CFMSE_ENV_ACTIVATE}"
    set -u
fi

if [[ -z ${CFMSE_DATA_ROOT:-} ]]; then
    cfmse_runtime_fail "Set DATA (expected dataset: \$DATA/VB+DMD)" \
        || return 1 2>/dev/null || exit 1
fi
if [[ ! -f ${CFMSE_REPO_ROOT}/xps/experiments.json ]]; then
    cfmse_runtime_fail \
        "Repository not found at ${CFMSE_REPO_ROOT}; set EXP (expected checkout: \$EXP/cfmse)" \
        || return 1 2>/dev/null || exit 1
fi
if [[ -z ${CFMSE_UV} || ! -x ${CFMSE_UV} ]]; then
    cfmse_runtime_fail "uv executable not found; install uv or override CFMSE_UV" \
        || return 1 2>/dev/null || exit 1
fi
if [[ -z ${CFMSE_PYTHON} || ! -x ${CFMSE_PYTHON} ]]; then
    cfmse_runtime_fail \
        "uv Python launcher not found at ${CFMSE_PYTHON}" \
        || return 1 2>/dev/null || exit 1
fi

# uv installs helper executables such as ninja beside Python.  Calling the
# interpreter by absolute path does not otherwise make those helpers visible.
export PATH="${CFMSE_REPO_ROOT}/.venv/bin:${PATH}"

# uv provides the Python environment but the NCSN++ operators also need the
# external CUDA compiler. Prefer an explicit/toolkit-provided CUDA_HOME, then a
# compiler already on PATH, then the newest versioned /usr/local installation.
cfmse_cuda_home=${CUDA_HOME:-${CUDA_PATH:-}}
if [[ -z ${cfmse_cuda_home} || ! -x ${cfmse_cuda_home}/bin/nvcc ]]; then
    cfmse_nvcc=$(command -v nvcc || true)
    if [[ -n ${cfmse_nvcc} ]]; then
        cfmse_cuda_home=$(cd -- "$(dirname -- "${cfmse_nvcc}")/.." && pwd)
    else
        cfmse_cuda_home=
        while IFS= read -r candidate; do
            if [[ -x ${candidate}/bin/nvcc ]]; then
                cfmse_cuda_home=${candidate}
                break
            fi
        done < <(find /usr/local -maxdepth 1 -type d -name 'cuda-*' -print 2>/dev/null | sort -Vr)
    fi
fi
if [[ -n ${cfmse_cuda_home} ]]; then
    export CUDA_HOME=${cfmse_cuda_home}
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
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
            cfmse_runtime_fail "Unsupported Stanage GPU type: ${CFMSE_GPU_TYPE}" \
                || return 1 2>/dev/null || exit 1
            ;;
    esac
else
    CFMSE_CUDA_ARCH_LIST=${CFMSE_CUDA_ARCH_LIST:-8.6}
fi
CFMSE_TORCH_EXTENSIONS_BASE=${CFMSE_TORCH_EXTENSIONS_BASE:-"${TMPDIR:-/tmp}/cfmse_torch_extensions_sm${CFMSE_CUDA_ARCH_LIST//./}"}

export CFMSE_DATA_ROOT CFMSE_LOG_ROOT CFMSE_PYTHON CFMSE_REPO_ROOT CFMSE_UV
export CFMSE_CUDA_ARCH_LIST CFMSE_TORCH_EXTENSIONS_BASE
export PYTHONPATH="${CFMSE_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TORCH_CUDA_ARCH_LIST=${CFMSE_CUDA_ARCH_LIST}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

cd "${CFMSE_REPO_ROOT}"
