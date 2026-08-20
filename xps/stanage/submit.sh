#!/usr/bin/env bash
set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"
export CFMSE_REPO_ROOT=${repo_root}
CFMSE_LAUNCHER_NAME=stanage-submit

# Load exactly the module/environment contract that each array task will use.
# shellcheck disable=SC1091
source "${repo_root}/xps/lib/runtime.sh" stanage

environment_args=(--require-cuda-build)
if [[ ${CFMSE_ALLOW_VERSION_DRIFT:-0} == 1 ]]; then
    environment_args+=(--allow-version-drift)
fi
"${CFMSE_PYTHON}" xps/check_environment.py "${environment_args[@]}"

"${CFMSE_PYTHON}" xps/run_experiment.py --validate
if [[ ${CFMSE_SKIP_DATA_PREFLIGHT:-0} != 1 ]]; then
    "${CFMSE_PYTHON}" xps/validate_data.py --base-dir "${CFMSE_DATA_ROOT}"
fi
mkdir -p logs/slurm

sbatch "$@" xps/stanage/train_array.sbatch
echo "Stanage array submitted successfully."
