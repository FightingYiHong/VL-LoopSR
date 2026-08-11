#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ID="${1:-feynman_highdim_pysr_practical_$(date +%Y%m%d_%H%M%S)}"
BASE_RUN_ROOT="${FEYNMAN_HIGHDIM_RUN_ROOT:-${ROOT_DIR}/runs/paper_experiments/01_standard_recovery/feynman_highdim_confirmatory_20260802}"
REUSE_DIR="${BASE_RUN_ROOT}/reuse"
LOG_ROOT="${FEYNMAN_HIGHDIM_PYSR_LOG_ROOT:-${ROOT_DIR}/logs/paper_experiments/01_standard_recovery/${RUN_ID}}"

mkdir -p "${LOG_ROOT}"
cd "${ROOT_DIR}"

python scripts/build_feynman_highdim_reuse_inventory.py \
  --run-root "${BASE_RUN_ROOT}" \
  --output-dir "${REUSE_DIR}" \
  > "${LOG_ROOT}/reuse_before.log" 2>&1

pysr_cases="$(paste -sd, "${REUSE_DIR}/missing_pysr_cases.txt")"
if [[ -z "${pysr_cases}" ]]; then
  printf 'status\tno_missing_cases\n' > "${LOG_ROOT}/status.tsv"
  exit 0
fi

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CPU_SR_SEARCH_EFFICIENCY_THRESHOLD=0.999
export CPU_SR_CONFIGURED_WALL_BUDGET_SEC=120
export SRBENCH_ROOT="${SRBENCH_ROOT:-${ROOT_DIR}/srbench}"
export SRBENCH_DATASETS_INFO_CSV="${SRBENCH_DATASETS_INFO_CSV:-${SRBENCH_ROOT}/docs/csv/datasets_info.csv}"
export SRBENCH_PMLB_CACHE_DIR="${SRBENCH_PMLB_CACHE_DIR:-${ROOT_DIR}/.cache/pmlb_cache}"
export SRBENCH_LOCAL_CSV_ROOT="${SRBENCH_LOCAL_CSV_ROOT:-${ROOT_DIR}/data/pmlb_regression_csv}"
export SRBENCH_ALLOW_GROUPS=feynman
export SRBENCH_ALLOW_CASE_NAMES="${pysr_cases}"
export SRBENCH_MIN_FEATURES=6
export SRBENCH_MAX_FEATURES=none
export SRBENCH_MAX_FILES=none

cat > "${LOG_ROOT}/run_manifest.tsv" <<EOF
run_id	${RUN_ID}
study	frozen_highdimensional_feynman_pysr_missing_only_practical
base_run_root	${BASE_RUN_ROOT}
missing_cases	$(wc -l < "${REUSE_DIR}/missing_pysr_cases.txt")
repeat_seed	611
case_budget_sec	120
max_train_rows	5000
execution_mode	persistent_python_and_julia_process
started_at	$(date -Is)
EOF

set +e
conda run --no-capture-output -n srbench-pysr \
  python scripts/run_cpu_baseline_benchmarks.py \
    --benchmark srbench \
    --method pysr \
    --config evaluation_suites/cpu_symbolic_regression_fourbench/configs/pysr_practical_2min_1thread.yaml \
    --random-state 611 \
    --results-root "${BASE_RUN_ROOT}/pysr/seed_611/srbench" \
    --resume \
  > "${LOG_ROOT}/pysr.log" 2>&1
status="$?"
set -e

python scripts/build_feynman_highdim_reuse_inventory.py \
  --run-root "${BASE_RUN_ROOT}" \
  --output-dir "${REUSE_DIR}" \
  > "${LOG_ROOT}/reuse_after.log" 2>&1 || status=1

printf 'exit_status\t%s\nfinished_at\t%s\n' "${status}" "$(date -Is)" \
  > "${LOG_ROOT}/status.tsv"
exit "${status}"
