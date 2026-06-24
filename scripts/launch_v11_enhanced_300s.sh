#!/usr/bin/env bash
set -euo pipefail

if [[ -f /etc/profile.d/llmsr_baseline.sh ]]; then
  # shellcheck source=/dev/null
  source /etc/profile.d/llmsr_baseline.sh
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${LLMSR_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
RUN_ID="${1:-v11_enhanced300_$(date +%Y%m%d_%H%M)}"
RESULTS_BASE="${RESULTS_BASE:-${ROOT_DIR}/runs/standard_recovery/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/standard_recovery/${RUN_ID}}"
BENCHMARKS="${V11_300_BENCHMARKS:-sldbench llmsrbench srsd srbench}"
SAMPLE_K="${V11_300_SAMPLE_K:-12}"
SEED="${V11_300_RANDOM_SEED:-20260526}"
WAIT_FOR_CURRENT="${WAIT_FOR_CURRENT:-1}"
QUEUE_AFTER_EQUAL100="${QUEUE_AFTER_EQUAL100:-0}"

mkdir -p "${RESULTS_BASE}" "${LOG_DIR}"
cd "${ROOT_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_DIR}/enhanced300.log"
}

wait_for_current_jobs() {
  if [[ "${WAIT_FOR_CURRENT}" != "1" ]]; then
    return
  fi
  log "waiting for active benchmark runners"
  while pgrep -af 'python3 scripts/(run_sldbench|run_llmsrbench|run_srds|run_SRbench)\.py|run_cpu_baseline_benchmarks\.py.*standard_recovery_fill' >/dev/null; do
    pgrep -af 'python3 scripts/(run_sldbench|run_llmsrbench|run_srds|run_SRbench)\.py|run_cpu_baseline_benchmarks\.py.*standard_recovery_fill' \
      >> "${LOG_DIR}/enhanced300.log" || true
    sleep 300
  done

  if [[ "${QUEUE_AFTER_EQUAL100}" == "1" ]]; then
    log "waiting for equal100 queue to finish"
    while tmux has-session -t llmsr_equal100_wait_20260526 2>/dev/null; do
      sleep 300
    done
  fi
}

configure_enhanced_budget() {
  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

  export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

  export LLMSR_V10_PATH="${LLMSR_V10_PATH:-${ROOT_DIR}/scripts/test_fey_v11_complexity_exact.py}"
  export SRSD_V10_PATH="${SRSD_V10_PATH:-${LLMSR_V10_PATH}}"
  export SLDBENCH_V10_PATH="${SLDBENCH_V10_PATH:-${LLMSR_V10_PATH}}"
  export LLMSRBENCH_V10_PATH="${LLMSRBENCH_V10_PATH:-${LLMSR_V10_PATH}}"
  export SRBENCH_V10_PATH="${SRBENCH_V10_PATH:-${LLMSR_V10_PATH}}"

  export LLMSR_AGENT_API_BASE="${LLMSR_AGENT_API_BASE:-${LLMSR_PLANNER_API_BASE:-http://127.0.0.1:8001/v1}}"
  export LLMSR_AGENT_API_KEY="${LLMSR_AGENT_API_KEY:-${LLMSR_PLANNER_API_KEY:-EMPTY}}"
  export LLMSR_AGENT_MODEL="${LLMSR_AGENT_MODEL:-${LLMSR_PLANNER_MODEL:-Qwen/Qwen2.5-VL-32B-Instruct}}"
  export LLMSR_PLANNER_API_BASE="${LLMSR_AGENT_API_BASE}"
  export LLMSR_PLANNER_API_KEY="${LLMSR_AGENT_API_KEY}"
  export LLMSR_PLANNER_MODEL="${LLMSR_AGENT_MODEL}"

  export LLMSR_EVAL_PROFILE="${LLMSR_EVAL_PROFILE:-quality}"
  export LLMSR_METHOD_MODE="${LLMSR_METHOD_MODE:-planner_guided}"
  export LLMSR_LIGHT_MODE="${LLMSR_LIGHT_MODE:-0}"
  export LLMSR_MAX_RUNTIME_PER_TASK_SEC="${LLMSR_MAX_RUNTIME_PER_TASK_SEC:-300}"

  export LLMSR_V11_DISABLE_RUNTIME_FAST_PATH=1
  export LLMSR_V11_BUDGET_AWARE_MODE="${LLMSR_V11_BUDGET_AWARE_MODE:-on}"
  export LLMSR_V11_FULL_BUDGET=1
  export LLMSR_V11_LOW_DIM_FULL_BUDGET=1
  export LLMSR_V11_ENABLE_STRUCTURE_EVALUATOR="${LLMSR_V11_ENABLE_STRUCTURE_EVALUATOR:-1}"
  export LLMSR_V11_ENABLE_GENERALIZATION_GUARD="${LLMSR_V11_ENABLE_GENERALIZATION_GUARD:-1}"
  export LLMSR_V11_FULL_BUDGET_TEXT_CALLS="${LLMSR_V11_FULL_BUDGET_TEXT_CALLS:-4}"
  export LLMSR_V11_FULL_BUDGET_MM_CALLS="${LLMSR_V11_FULL_BUDGET_MM_CALLS:-1}"
  export LLMSR_V11_FULL_BUDGET_PROPOSAL_K="${LLMSR_V11_FULL_BUDGET_PROPOSAL_K:-16}"
  export LLMSR_V11_FULL_BUDGET_REFINED_K="${LLMSR_V11_FULL_BUDGET_REFINED_K:-8}"
  export LLMSR_V11_FULL_BUDGET_REFINE_ROUNDS="${LLMSR_V11_FULL_BUDGET_REFINE_ROUNDS:-4}"
  export LLMSR_V11_LOW_DIM_FULL_TEXT_CALLS="${LLMSR_V11_LOW_DIM_FULL_TEXT_CALLS:-3}"
  export LLMSR_V11_LOW_DIM_FULL_MM_CALLS="${LLMSR_V11_LOW_DIM_FULL_MM_CALLS:-1}"
  export LLMSR_V11_LOW_DIM_FULL_PROPOSAL_K="${LLMSR_V11_LOW_DIM_FULL_PROPOSAL_K:-14}"
  export LLMSR_V11_LOW_DIM_FULL_REFINED_K="${LLMSR_V11_LOW_DIM_FULL_REFINED_K:-7}"
  export LLMSR_V11_LOW_DIM_FULL_REFINE_ROUNDS="${LLMSR_V11_LOW_DIM_FULL_REFINE_ROUNDS:-4}"
  export LLMSR_V11_IMAGE_LOOP_MAX_ROUNDS="${LLMSR_V11_IMAGE_LOOP_MAX_ROUNDS:-4}"
  export LLMSR_V11_IMAGE_LOOP_MAX_AUGMENTED_CANDIDATES="${LLMSR_V11_IMAGE_LOOP_MAX_AUGMENTED_CANDIDATES:-160}"
  export LLMSR_V11_VLM_OBSERVER_GENERATE_IMAGES="${LLMSR_V11_VLM_OBSERVER_GENERATE_IMAGES:-1}"
  export LLMSR_V11_VLM_OBSERVER_MAX_IMAGES="${LLMSR_V11_VLM_OBSERVER_MAX_IMAGES:-1}"
  export LLMSR_V11_VLM_OBSERVER_MAX_ROWS="${LLMSR_V11_VLM_OBSERVER_MAX_ROWS:-24}"
  export LLMSR_V11_VLM_OBSERVER_MAX_TOKENS="${LLMSR_V11_VLM_OBSERVER_MAX_TOKENS:-600}"

  export LLMSRBENCH_ROOT="${LLMSRBENCH_ROOT:-${ROOT_DIR}/data/llmsrbench}"
  export LLMSRBENCH_HDF5="${LLMSRBENCH_HDF5:-${LLMSRBENCH_ROOT}/lsr_bench_data.hdf5}"
  export SRSD_ROOT="${SRSD_ROOT:-${ROOT_DIR}/srsd-benchmark/resource/datasets/srsd}"
  export SRBENCH_ROOT="${SRBENCH_ROOT:-${ROOT_DIR}/srbench}"
  export SRBENCH_DATASETS_INFO_CSV="${SRBENCH_DATASETS_INFO_CSV:-${SRBENCH_ROOT}/docs/csv/datasets_info.csv}"
  export SRBENCH_PMLB_CACHE_DIR="${SRBENCH_PMLB_CACHE_DIR:-${ROOT_DIR}/.cache/pmlb_cache}"
  export SRBENCH_LOCAL_CSV_ROOT="${SRBENCH_LOCAL_CSV_ROOT:-${ROOT_DIR}/data/pmlb_regression_csv}"

  if [[ "${V11_300_FULL:-0}" != "1" ]]; then
    export SLDBENCH_RANDOM_SAMPLE_K="${SLDBENCH_RANDOM_SAMPLE_K:-${SAMPLE_K}}"
    export LLMSRBENCH_RANDOM_SAMPLE_K="${LLMSRBENCH_RANDOM_SAMPLE_K:-${SAMPLE_K}}"
    export SRSD_RANDOM_SAMPLE_K="${SRSD_RANDOM_SAMPLE_K:-${SAMPLE_K}}"
    export SRBENCH_RANDOM_SAMPLE_K="${SRBENCH_RANDOM_SAMPLE_K:-${SAMPLE_K}}"
  fi
  export SLDBENCH_RANDOM_SEED="${SLDBENCH_RANDOM_SEED:-${SEED}}"
  export LLMSRBENCH_RANDOM_SEED="${LLMSRBENCH_RANDOM_SEED:-${SEED}}"
  export SRSD_RANDOM_SEED="${SRSD_RANDOM_SEED:-${SEED}}"
  export SRBENCH_RANDOM_SEED="${SRBENCH_RANDOM_SEED:-${SEED}}"

  export SLDBENCH_MAX_FEATURES="${SLDBENCH_MAX_FEATURES:-20}"
  export LLMSRBENCH_MAX_FEATURES="${LLMSRBENCH_MAX_FEATURES:-8}"
  export SRSD_MAX_FEATURES="${SRSD_MAX_FEATURES:-8}"
  export SRBENCH_MAX_FEATURES="${SRBENCH_MAX_FEATURES:-none}"
}

run_one() {
  local benchmark="$1"
  local out="${RESULTS_BASE}/${benchmark}"
  local log_file="${LOG_DIR}/${benchmark}.log"
  mkdir -p "${out}"
  log "start ${benchmark} results=${out}"
  case "${benchmark}" in
    sldbench)
      SLDBENCH_V10_RESULTS_ROOT="${out}" python3 scripts/run_sldbench.py > "${log_file}" 2>&1
      ;;
    llmsrbench)
      LLMSRBENCH_V10_RESULTS_ROOT="${out}" python3 scripts/run_llmsrbench.py > "${log_file}" 2>&1
      ;;
    srsd)
      SRSD_V10_RESULTS_ROOT="${out}" python3 scripts/run_srds.py > "${log_file}" 2>&1
      ;;
    srbench)
      SRBENCH_V10_RESULTS_ROOT="${out}" python3 scripts/run_SRbench.py > "${log_file}" 2>&1
      ;;
    *)
      printf 'unknown benchmark: %s\n' "${benchmark}" >&2
      return 2
      ;;
  esac
  log "done ${benchmark}"
}

wait_for_current_jobs
configure_enhanced_budget
llmsr-vllm-status >> "${LOG_DIR}/enhanced300.log" 2>&1 || true

status_file="${RESULTS_BASE}/enhanced300_status.tsv"
printf 'run_id\tbenchmark\tstatus\tstart_time\tend_time\truntime_sec\tresults_root\tlog_file\n' > "${status_file}"

for benchmark in ${BENCHMARKS}; do
  start_time="$(date '+%F %T')"
  start_epoch="$(date +%s)"
  set +e
  run_one "${benchmark}"
  status="$?"
  set -e
  end_time="$(date '+%F %T')"
  end_epoch="$(date +%s)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${RUN_ID}" "${benchmark}" "${status}" "${start_time}" "${end_time}" "$((end_epoch - start_epoch))" \
    "${RESULTS_BASE}/${benchmark}" "${LOG_DIR}/${benchmark}.log" >> "${status_file}"
done

log "finished run_id=${RUN_ID} results=${RESULTS_BASE}"
