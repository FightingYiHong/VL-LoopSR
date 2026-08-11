#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${LLMSR_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
RUN_ID="${1:-standard_recovery_$(date +%Y%m%d_%H%M%S)}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT_DIR}/runs/paper_experiments/01_standard_recovery/${RUN_ID}/vega_sr}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/paper_experiments/01_standard_recovery/${RUN_ID}/vega_sr}"
BENCHMARKS="${STANDARD_BENCHMARKS:-llmsrbench srbench srsd}"
CASE_BUDGET_SEC="${CASE_BUDGET_SEC:-300}"
MAX_CASES="${MAX_CASES:-0}"
RANDOM_SEED="${RANDOM_SEED:-42}"

mkdir -p "${RESULTS_BASE}" "${LOG_DIR}"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export LLMSR_V10_PATH="${LLMSR_V10_PATH:-${ROOT_DIR}/scripts/vega_sr.py}"
export SRSD_V10_PATH="${LLMSR_V10_PATH}"
export LLMSRBENCH_V10_PATH="${LLMSR_V10_PATH}"
export SRBENCH_V10_PATH="${LLMSR_V10_PATH}"

export LLMSR_AGENT_API_BASE="${LLMSR_AGENT_API_BASE:-http://127.0.0.1:8001/v1}"
export LLMSR_AGENT_API_KEY="${LLMSR_AGENT_API_KEY:-EMPTY}"
export LLMSR_AGENT_MODEL="${LLMSR_AGENT_MODEL:-Qwen3-VL-32B-Instruct}"
export LLMSR_PLANNER_API_BASE="${LLMSR_PLANNER_API_BASE:-${LLMSR_AGENT_API_BASE}}"
export LLMSR_PLANNER_API_KEY="${LLMSR_PLANNER_API_KEY:-${LLMSR_AGENT_API_KEY}}"
export LLMSR_PLANNER_MODEL="${LLMSR_PLANNER_MODEL:-${LLMSR_AGENT_MODEL}}"

export LLMSR_EVAL_PROFILE="${LLMSR_EVAL_PROFILE:-quality}"
export LLMSR_METHOD_MODE="${LLMSR_METHOD_MODE:-planner_guided}"
export LLMSR_MAX_RUNTIME_PER_TASK_SEC="${CASE_BUDGET_SEC}"
export LLMSR_ALLOW_TRUE_EXPR_DIAGNOSTICS=0
export LLMSR_USE_TEST_FOR_SELECTION=0
export LLMSR_V11_BUDGET_AWARE_MODE=on
export LLMSR_V11_ENABLE_STRUCTURE_EVALUATOR=1
export LLMSR_V11_ENABLE_GENERALIZATION_GUARD=1

DATA_EXTERNAL="${DATA_EXTERNAL:-${ROOT_DIR}/data/external}"
export LLMSRBENCH_ROOT="${LLMSRBENCH_ROOT:-${DATA_EXTERNAL}/llmsrbench}"
export LLMSRBENCH_HDF5="${LLMSRBENCH_HDF5:-${LLMSRBENCH_ROOT}/lsr_bench_data.hdf5}"
export SRBENCH_ROOT="${SRBENCH_ROOT:-${DATA_EXTERNAL}/srbench}"
export SRBENCH_DATASETS_INFO_CSV="${SRBENCH_DATASETS_INFO_CSV:-${SRBENCH_ROOT}/docs/csv/datasets_info.csv}"
export SRBENCH_PMLB_CACHE_DIR="${SRBENCH_PMLB_CACHE_DIR:-${ROOT_DIR}/.cache/pmlb_cache}"
export SRSD_ROOT="${SRSD_ROOT:-${DATA_EXTERNAL}/srsd-benchmark/resource/datasets/srsd}"

export LLMSRBENCH_RANDOM_SEED="${LLMSRBENCH_RANDOM_SEED:-${RANDOM_SEED}}"
export SRBENCH_RANDOM_SEED="${SRBENCH_RANDOM_SEED:-${RANDOM_SEED}}"
export SRSD_RANDOM_SEED="${SRSD_RANDOM_SEED:-${RANDOM_SEED}}"

if [[ "${MAX_CASES}" -gt 0 ]]; then
  export LLMSRBENCH_RANDOM_SAMPLE_K="${MAX_CASES}"
  export SRBENCH_RANDOM_SAMPLE_K="${MAX_CASES}"
  export SRSD_RANDOM_SAMPLE_K="${MAX_CASES}"
else
  unset LLMSRBENCH_RANDOM_SAMPLE_K SRBENCH_RANDOM_SAMPLE_K SRSD_RANDOM_SAMPLE_K || true
fi

run_benchmark() {
  local benchmark="$1"
  local output_dir="${RESULTS_BASE}/${benchmark}"
  local log_file="${LOG_DIR}/${benchmark}.log"
  mkdir -p "${output_dir}"
  printf '[standard-recovery] benchmark=%s output=%s\n' "${benchmark}" "${output_dir}"
  case "${benchmark}" in
    llmsrbench) LLMSRBENCH_V10_RESULTS_ROOT="${output_dir}" python3 scripts/run_llmsrbench.py ;;
    srbench) SRBENCH_V10_RESULTS_ROOT="${output_dir}" python3 scripts/run_SRbench.py ;;
    srsd) SRSD_V10_RESULTS_ROOT="${output_dir}" python3 scripts/run_srds.py ;;
    *) printf 'Unknown benchmark: %s\n' "${benchmark}" >&2; return 2 ;;
  esac >"${log_file}" 2>&1
}

status_file="${RESULTS_BASE}/status.tsv"
printf 'benchmark\tstatus\tstarted_at\tfinished_at\n' >"${status_file}"
failed=0
for benchmark in ${BENCHMARKS}; do
  started_at="$(date -Is)"
  if run_benchmark "${benchmark}"; then
    status=0
  else
    status=$?
    failed=1
  fi
  printf '%s\t%s\t%s\t%s\n' \
    "${benchmark}" "${status}" "${started_at}" "$(date -Is)" >>"${status_file}"
done
exit "${failed}"
