#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ID="${1:-feynman_highdim_gaps_$(date +%Y%m%d_%H%M%S)}"
BASE_RUN_ROOT="${FEYNMAN_HIGHDIM_RUN_ROOT:-${ROOT_DIR}/runs/paper_experiments/01_standard_recovery/feynman_highdim_confirmatory_20260802}"
REUSE_DIR="${BASE_RUN_ROOT}/reuse"
LOG_ROOT="${FEYNMAN_HIGHDIM_GAP_LOG_ROOT:-${ROOT_DIR}/logs/paper_experiments/01_standard_recovery/${RUN_ID}}"
BUDGET_SEC="${FEYNMAN_HIGHDIM_BUDGET_SEC:-600}"
MODEL="${FEYNMAN_HIGHDIM_MODEL:-Qwen3-VL-32B-Instruct}"
API_BASE="${FEYNMAN_HIGHDIM_API_BASE:-http://127.0.0.1:8001/v1}"

mkdir -p "${LOG_ROOT}"
cd "${ROOT_DIR}"

python scripts/build_feynman_highdim_reuse_inventory.py \
  --run-root "${BASE_RUN_ROOT}" \
  --output-dir "${REUSE_DIR}" \
  > "${LOG_ROOT}/reuse_before.log" 2>&1

pysr_cases="$(paste -sd, "${REUSE_DIR}/missing_pysr_cases.txt")"
icsr_cases="$(paste -sd, "${REUSE_DIR}/missing_icsr_cases.txt")"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CPU_SR_SEARCH_EFFICIENCY_THRESHOLD=0.999
export CPU_SR_CONFIGURED_WALL_BUDGET_SEC="${BUDGET_SEC}"
export CPU_SR_CASE_SUBPROCESS_TIMEOUT_SEC="$((BUDGET_SEC + 120))"
export CPU_SR_CASE_SUBPROCESS_GRACE_SEC=120
export SRBENCH_ROOT="${SRBENCH_ROOT:-${ROOT_DIR}/srbench}"
export SRBENCH_DATASETS_INFO_CSV="${SRBENCH_DATASETS_INFO_CSV:-${SRBENCH_ROOT}/docs/csv/datasets_info.csv}"
export SRBENCH_PMLB_CACHE_DIR="${SRBENCH_PMLB_CACHE_DIR:-${ROOT_DIR}/.cache/pmlb_cache}"
export SRBENCH_LOCAL_CSV_ROOT="${SRBENCH_LOCAL_CSV_ROOT:-${ROOT_DIR}/data/pmlb_regression_csv}"
export SRBENCH_ALLOW_GROUPS=feynman
export SRBENCH_MIN_FEATURES=6
export SRBENCH_MAX_FEATURES=none
export SRBENCH_MAX_FILES=none

status_file="${LOG_ROOT}/status.tsv"
printf 'worker\tstatus\tstarted_at\tfinished_at\tlog\n' > "${status_file}"
cat > "${LOG_ROOT}/run_manifest.tsv" <<EOF
run_id	${RUN_ID}
study	frozen_highdimensional_feynman_missing_results_only
base_run_root	${BASE_RUN_ROOT}
methods	PySR,ICSR
pysr_missing_cases	$(wc -l < "${REUSE_DIR}/missing_pysr_cases.txt")
icsr_missing_cases	$(wc -l < "${REUSE_DIR}/missing_icsr_cases.txt")
repeat_seed	611
case_budget_sec	${BUDGET_SEC}
started_at	$(date -Is)
EOF

run_logged() {
  local worker="$1"
  shift
  local log_file="${LOG_ROOT}/${worker}.log"
  local started_at
  local finished_at
  local status
  started_at="$(date -Is)"
  set +e
  "$@" > "${log_file}" 2>&1
  status="$?"
  set -e
  finished_at="$(date -Is)"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${worker}" "${status}" "${started_at}" "${finished_at}" "${log_file}" \
    >> "${status_file}"
  return "${status}"
}

run_pysr() {
  if [[ -z "${pysr_cases}" ]]; then
    echo "[SKIP] PySR has no missing cases"
    return 0
  fi
  SRBENCH_ALLOW_CASE_NAMES="${pysr_cases}" \
    conda run --no-capture-output -n srbench-pysr \
      python scripts/run_cpu_baseline_benchmarks.py \
        --benchmark srbench \
        --method pysr \
        --config evaluation_suites/cpu_symbolic_regression_fourbench/configs/pysr_practical_2min_1thread.yaml \
        --random-state 611 \
        --results-root "${BASE_RUN_ROOT}/pysr/seed_611/srbench" \
        --resume
}

run_icsr() {
  if [[ -z "${icsr_cases}" ]]; then
    echo "[SKIP] ICSR has no missing cases"
    return 0
  fi
  export OFFICIAL_ICSR_ROOT="${OFFICIAL_ICSR_ROOT:-${ROOT_DIR}/external_repos/In-Context-Symbolic-Regression}"
  export OFFICIAL_ICSR_MODEL_NAME="${MODEL}"
  export OFFICIAL_ICSR_OPENAI_BASE_URL="${API_BASE}"
  export OFFICIAL_ICSR_OPENAI_API_KEY=EMPTY
  SRBENCH_ALLOW_CASE_NAMES="${icsr_cases}" PYTHONHASHSEED=611 \
    conda run --no-capture-output -n base \
      python evaluation_suites/official_icsr_fourbench/run_official_icsr_fourbench.py \
        --benchmarks srbench \
        --results-root "${BASE_RUN_ROOT}/icsr/seed_611" \
        --case-timeout-sec "${BUDGET_SEC}" \
        --startup-timeout-sec 600 \
        --iterations 500 \
        --optimizer-timeout-sec 10 \
        --optimizer-threads 5 \
        --max-new-tokens 2048 \
        --resume
}

pids=()
workers=()
run_logged pysr run_pysr &
pids+=("$!")
workers+=("pysr")
run_logged icsr run_icsr &
pids+=("$!")
workers+=("icsr")

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'worker failed: %s\n' "${workers[$index]}" >&2
    failed=1
  fi
done

python scripts/build_feynman_highdim_reuse_inventory.py \
  --run-root "${BASE_RUN_ROOT}" \
  --output-dir "${REUSE_DIR}" \
  > "${LOG_ROOT}/reuse_after.log" 2>&1 || failed=1

printf 'finished_at\t%s\n' "$(date -Is)" >> "${LOG_ROOT}/run_manifest.tsv"
exit "${failed}"
