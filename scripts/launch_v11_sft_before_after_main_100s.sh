#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-${LLMSR_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
export ROOT_DIR
export LLMSR_ROOT="${ROOT_DIR}"
RUN_STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"

RESULTS_PARENT="${SFT_COMPARE_RESULTS_PARENT:-${ROOT_DIR}/runs/sft_before_after}"
LOG_PARENT="${SFT_COMPARE_LOG_PARENT:-${ROOT_DIR}/logs/sft_before_after}"
REPORT_PARENT="${SFT_COMPARE_REPORT_PARENT:-${ROOT_DIR}/runs/reports}"

TIME_BUDGET="${SFT_COMPARE_TIME_BUDGET:-100}"
PASS_THRESHOLD="${SFT_COMPARE_PASS_THRESHOLD:-100}"
BENCHMARKS="${SFT_COMPARE_BENCHMARKS:-llmsrbench srbench srsd}"
SAMPLE_K="${SFT_COMPARE_SAMPLE_K:-0}"
RANDOM_SEED="${SFT_COMPARE_RANDOM_SEED:-20260526}"
WAIT_FOR_CURRENT="${SFT_COMPARE_WAIT_FOR_CURRENT:-0}"

BEFORE_LABEL="${SFT_COMPARE_BEFORE_LABEL:-before_sft_qwen3vl32b_instruct}"
AFTER_LABEL="${SFT_COMPARE_AFTER_LABEL:-after_sft32b}"
BEFORE_RUN_ID="${SFT_COMPARE_BEFORE_RUN_ID:-v11_before_sft_qwen3vl32b_instruct_full100_${RUN_STAMP}}"
AFTER_RUN_ID="${SFT_COMPARE_AFTER_RUN_ID:-v11_after_sft32b_full100_${RUN_STAMP}}"
BEFORE_ROOT="${SFT_COMPARE_BEFORE_ROOT:-${RESULTS_PARENT}/${BEFORE_RUN_ID}}"
AFTER_ROOT="${SFT_COMPARE_AFTER_ROOT:-${RESULTS_PARENT}/${AFTER_RUN_ID}}"
OUT_DIR="${SFT_COMPARE_OUT_DIR:-${REPORT_PARENT}/v11_sft_before_after_${RUN_STAMP}}"

BEFORE_API_BASE="${SFT_COMPARE_BEFORE_API_BASE:-http://127.0.0.1:9001/v1}"
BEFORE_API_KEY="${SFT_COMPARE_BEFORE_API_KEY:-EMPTY}"
BEFORE_MODEL="${SFT_COMPARE_BEFORE_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"

AFTER_API_BASE="${SFT_COMPARE_AFTER_API_BASE:-http://127.0.0.1:8011/v1}"
AFTER_API_KEY="${SFT_COMPARE_AFTER_API_KEY:-EMPTY}"
AFTER_MODEL="${SFT_COMPARE_AFTER_MODEL:-qwen3-vl-32b-proposer-sft-10000-qlora-merged}"

ONLY_SUMMARIZE="${SFT_COMPARE_ONLY_SUMMARIZE:-0}"
SKIP_BEFORE="${SFT_COMPARE_SKIP_BEFORE:-0}"
SKIP_AFTER="${SFT_COMPARE_SKIP_AFTER:-0}"
SKIP_SUMMARY="${SFT_COMPARE_SKIP_SUMMARY:-0}"
REQUIRE_ENDPOINTS="${SFT_COMPARE_REQUIRE_ENDPOINTS:-1}"
DRY_RUN="${SFT_COMPARE_DRY_RUN:-0}"

mkdir -p "${BEFORE_ROOT}" "${AFTER_ROOT}" "${OUT_DIR}" "${LOG_PARENT}"
cd "${ROOT_DIR}"

unset ALL_PROXY HTTPS_PROXY HTTP_PROXY all_proxy https_proxy http_proxy
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export HF_HOME="${SFT_COMPARE_HF_HOME:-${HF_HOME:-${ROOT_DIR}/.cache/huggingface}}"
export HF_DATASETS_CACHE="${SFT_COMPARE_HF_DATASETS_CACHE:-${HF_DATASETS_CACHE:-${HF_HOME}/datasets}}"
export HF_HUB_CACHE="${SFT_COMPARE_HF_HUB_CACHE:-${HF_HUB_CACHE:-${HF_HOME}/hub}}"
export LLMSRBENCH_ROOT="${SFT_COMPARE_LLMSRBENCH_ROOT:-${LLMSRBENCH_ROOT:-${ROOT_DIR}/data/external/llmsrbench}}"
export LLMSRBENCH_HDF5="${SFT_COMPARE_LLMSRBENCH_HDF5:-${LLMSRBENCH_HDF5:-${LLMSRBENCH_ROOT}/lsr_bench_data.hdf5}}"
export SRSD_ROOT="${SFT_COMPARE_SRSD_ROOT:-${SRSD_ROOT:-${ROOT_DIR}/data/external/srsd-benchmark/resource/datasets/srsd}}"
export SRBENCH_ROOT="${SFT_COMPARE_SRBENCH_ROOT:-${SRBENCH_ROOT:-${ROOT_DIR}/data/external/srbench}}"
export SRBENCH_DATASETS_INFO_CSV="${SFT_COMPARE_SRBENCH_DATASETS_INFO_CSV:-${SRBENCH_DATASETS_INFO_CSV:-${SRBENCH_ROOT}/docs/csv/datasets_info.csv}}"
export SRBENCH_PMLB_CACHE_DIR="${SFT_COMPARE_SRBENCH_PMLB_CACHE_DIR:-${SRBENCH_PMLB_CACHE_DIR:-${ROOT_DIR}/.cache/pmlb_cache}}"
export SRBENCH_LOCAL_CSV_ROOT="${SFT_COMPARE_SRBENCH_LOCAL_CSV_ROOT:-${SRBENCH_LOCAL_CSV_ROOT:-${ROOT_DIR}/data/pmlb_regression_csv}}"

MAIN_LOG="${OUT_DIR}/launch.log"

log() {
  mkdir -p "$(dirname "${MAIN_LOG}")"
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${MAIN_LOG}"
}

die() {
  log "ERROR: $*"
  exit 1
}

endpoint_models_url() {
  local api_base="$1"
  printf '%s/models' "${api_base%/}"
}

wait_for_endpoint() {
  local label="$1"
  local api_base="$2"
  local api_key="$3"
  local url
  url="$(endpoint_models_url "${api_base}")"
  if [[ "${DRY_RUN}" == "1" || "${REQUIRE_ENDPOINTS}" != "1" ]]; then
    log "skip endpoint check for ${label}: ${url}"
    return
  fi
  log "checking ${label} endpoint: ${url}"
  for _ in $(seq 1 60); do
    if curl -fsS --max-time 5 -H "Authorization: Bearer ${api_key}" "${url}" >/dev/null; then
      log "${label} endpoint is ready"
      return
    fi
    sleep 10
  done
  die "${label} endpoint is not ready: ${url}"
}

configure_common_runner_env() {
  export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
  export WAIT_FOR_CURRENT="${WAIT_FOR_CURRENT}"
  export V11_300_BENCHMARKS="${BENCHMARKS}"
  export V11_300_RANDOM_SEED="${RANDOM_SEED}"
  export LLMSR_MAX_RUNTIME_PER_TASK_SEC="${TIME_BUDGET}"

  if [[ "${SAMPLE_K}" =~ ^[0-9]+$ ]] && (( SAMPLE_K > 0 )); then
    export V11_300_FULL=0
    export V11_300_SAMPLE_K="${SAMPLE_K}"
    export LLMSRBENCH_RANDOM_SAMPLE_K="${SAMPLE_K}"
    export SRSD_RANDOM_SAMPLE_K="${SAMPLE_K}"
    export SRBENCH_RANDOM_SAMPLE_K="${SAMPLE_K}"
  else
    export V11_300_FULL=1
    unset V11_300_SAMPLE_K LLMSRBENCH_RANDOM_SAMPLE_K SRSD_RANDOM_SAMPLE_K SRBENCH_RANDOM_SAMPLE_K
  fi

  export LLMSR_V11_ENABLE_VLM_OBSERVER="${LLMSR_V11_ENABLE_VLM_OBSERVER:-1}"
  export LLMSR_V11_VLM_OBSERVER_GENERATE_IMAGES="${LLMSR_V11_VLM_OBSERVER_GENERATE_IMAGES:-1}"
  export LLMSR_V11_VLM_OBSERVER_MAX_IMAGES="${LLMSR_V11_VLM_OBSERVER_MAX_IMAGES:-1}"
  export LLMSR_V11_VLM_OBSERVER_MAX_ROWS="${LLMSR_V11_VLM_OBSERVER_MAX_ROWS:-24}"
  export LLMSR_V11_VLM_OBSERVER_MAX_TOKENS="${LLMSR_V11_VLM_OBSERVER_MAX_TOKENS:-600}"
}

configure_model_env() {
  local api_base="$1"
  local api_key="$2"
  local model="$3"
  export LLMSR_AGENT_API_BASE="${api_base}"
  export LLMSR_AGENT_API_KEY="${api_key}"
  export LLMSR_AGENT_MODEL="${model}"
  export LLMSR_PLANNER_API_BASE="${api_base}"
  export LLMSR_PLANNER_API_KEY="${api_key}"
  export LLMSR_PLANNER_MODEL="${model}"
  export LLMSR_OBSERVER_API_BASE="${api_base}"
  export LLMSR_OBSERVER_API_KEY="${api_key}"
  export LLMSR_OBSERVER_MODEL="${model}"
}

run_v11_phase() {
  local label="$1"
  local run_id="$2"
  local result_root="$3"
  local api_base="$4"
  local api_key="$5"
  local model="$6"
  local phase_log_dir="${LOG_PARENT}/${run_id}"

  configure_common_runner_env
  configure_model_env "${api_base}" "${api_key}" "${model}"
  export RESULTS_BASE="${result_root}"
  export LOG_DIR="${phase_log_dir}"

  log "start ${label}: run_id=${run_id}"
  log "  model=${model}"
  log "  api=${api_base}"
  log "  results=${result_root}"
  log "  logs=${phase_log_dir}"
  log "  benchmarks=${BENCHMARKS}; budget=${TIME_BUDGET}s; sample_k=${SAMPLE_K}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  bash scripts/launch_v11_enhanced_300s.sh "${run_id}"
  log "finished ${label}: ${result_root}"
}

write_manifest() {
  cat > "${OUT_DIR}/experiment_manifest.json" <<EOF
{
  "run_stamp": "${RUN_STAMP}",
  "time_budget_sec": ${TIME_BUDGET},
  "pass_threshold": ${PASS_THRESHOLD},
  "benchmarks": "${BENCHMARKS}",
  "sample_k": "${SAMPLE_K}",
  "before": {
    "label": "${BEFORE_LABEL}",
    "run_id": "${BEFORE_RUN_ID}",
    "root": "${BEFORE_ROOT}",
    "api_base": "${BEFORE_API_BASE}",
    "model": "${BEFORE_MODEL}"
  },
  "after": {
    "label": "${AFTER_LABEL}",
    "run_id": "${AFTER_RUN_ID}",
    "root": "${AFTER_ROOT}",
    "api_base": "${AFTER_API_BASE}",
    "model": "${AFTER_MODEL}"
  },
  "out_dir": "${OUT_DIR}"
}
EOF
}

summarize_results() {
  log "summarizing before/after results"
  if [[ "${DRY_RUN}" == "1" ]]; then
    log "dry run: skip summarizer"
    return
  fi
  python scripts/summarize_v11_sft_before_after.py \
    --before-root "${BEFORE_ROOT}" \
    --after-root "${AFTER_ROOT}" \
    --out-dir "${OUT_DIR}" \
    --before-label "${BEFORE_LABEL}" \
    --after-label "${AFTER_LABEL}" \
    --pass-threshold "${PASS_THRESHOLD}"
  log "summary report: ${OUT_DIR}/comparison_report.md"
}

write_manifest
log "V11 SFT before/after experiment"
log "manifest=${OUT_DIR}/experiment_manifest.json"

if [[ "${ONLY_SUMMARIZE}" != "1" ]]; then
  if [[ "${SKIP_BEFORE}" != "1" ]]; then
    wait_for_endpoint "${BEFORE_LABEL}" "${BEFORE_API_BASE}" "${BEFORE_API_KEY}"
    run_v11_phase "${BEFORE_LABEL}" "${BEFORE_RUN_ID}" "${BEFORE_ROOT}" "${BEFORE_API_BASE}" "${BEFORE_API_KEY}" "${BEFORE_MODEL}"
  else
    log "skip before phase; using existing results at ${BEFORE_ROOT}"
  fi

  if [[ "${SKIP_AFTER}" != "1" ]]; then
    wait_for_endpoint "${AFTER_LABEL}" "${AFTER_API_BASE}" "${AFTER_API_KEY}"
    run_v11_phase "${AFTER_LABEL}" "${AFTER_RUN_ID}" "${AFTER_ROOT}" "${AFTER_API_BASE}" "${AFTER_API_KEY}" "${AFTER_MODEL}"
  else
    log "skip after phase; using existing results at ${AFTER_ROOT}"
  fi
else
  log "only summarize; using existing results"
fi

if [[ "${SKIP_SUMMARY}" != "1" ]]; then
  summarize_results
else
  log "skip summary"
fi
log "done"
