#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-sr}"
SELF="${PROJECT_ROOT}/evaluation_suites/paper_experiments/scripts/launch_01_standard_search_efficiency_full.sh"

run_worker() {
  local run_id="$1"
  local benchmark="$2"
  local results_root="$3"
  local log_root="$4"
  local timeout_sec="$5"
  local api_base="$6"
  local model="$7"

  export ROOT_DIR="${PROJECT_ROOT}"
  export RESULTS_BASE="${results_root}"
  export LOG_DIR="${log_root}/vl_loopsr/${benchmark}"
  export STATUS_FILE="${results_root}/status_${benchmark}.tsv"
  export V11_300_BENCHMARKS="${benchmark}"
  export V11_300_FULL=1
  export WAIT_FOR_CURRENT=0
  export LLMSR_MAX_RUNTIME_PER_TASK_SEC="${timeout_sec}"
  export LLMSR_AGENT_API_BASE="${api_base}"
  export LLMSR_AGENT_API_KEY="${LLMSR_AGENT_API_KEY:-EMPTY}"
  export LLMSR_AGENT_MODEL="${model}"
  export LLMSR_V11_RESTORE_TEXT_PROPOSER=1
  export LLMSR_V11_ENABLE_VLM_OBSERVER=1
  export LLMSR_V11_DISABLE_RUNTIME_FAST_PATH=1
  export LLMSRBENCH_CASES_ROOT="${PROJECT_ROOT}/data/sim-datasets-llmsr/llm-srbench"
  export SLDBENCH_HUB_REPO="${PROJECT_ROOT}/data/sldbench_repo"
  export SRBENCH_ROOT="${PROJECT_ROOT}/srbench"
  export SRBENCH_DATASETS_INFO_CSV="${PROJECT_ROOT}/srbench/docs/csv/datasets_info.csv"
  export SRSD_ROOT="${PROJECT_ROOT}/srsd-benchmark/resource/datasets/srsd"

  if [[ "${benchmark}" == "srbench" ]]; then
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  fi

  cd "${PROJECT_ROOT}"
  exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash scripts/launch_v11_enhanced_300s.sh "${run_id}"
}

run_monitor() {
  local results_root="$1"
  local log_root="$2"
  local session_csv="$3"
  local interval_sec="${SEARCH_EFFICIENCY_SUMMARY_INTERVAL_SEC:-900}"
  local summary_log="${log_root}/search_efficiency_summary.log"
  local -a sessions
  IFS=',' read -r -a sessions <<< "${session_csv}"

  cd "${PROJECT_ROOT}"
  while true; do
    local active=0
    local session
    for session in "${sessions[@]}"; do
      if tmux has-session -t "${session}" 2>/dev/null; then
        active=1
      fi
    done

    {
      printf '[%s] refreshing search-efficiency tables and figure\n' "$(date '+%F %T')"
      "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV_NAME}" \
        python scripts/summarize_standard_search_efficiency.py \
        --results-root "${results_root}" \
        --output-dir "${results_root}/search_efficiency"
    } >> "${summary_log}" 2>&1 || true

    if [[ "${active}" == "0" ]]; then
      printf '[%s] all benchmark workers finished\n' "$(date '+%F %T')" \
        >> "${summary_log}"
      break
    fi
    sleep "${interval_sec}"
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  run_worker "$@"
  exit
fi

if [[ "${1:-}" == "--monitor" ]]; then
  shift
  run_monitor "$@"
  exit
fi

RUN_ID="${1:-standard_search_efficiency_full_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/01_standard_recovery/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/01_standard_recovery/${RUN_ID}}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-600}"
LLMSR_AGENT_API_BASE="${LLMSR_AGENT_API_BASE:-http://127.0.0.1:8001/v1}"
LLMSR_AGENT_MODEL="${LLMSR_AGENT_MODEL:-Qwen3-VL-32B-Instruct}"
SESSION_PREFIX="${SESSION_PREFIX:-sr972_$(date +%H%M%S)}"
BENCHMARKS=(llmsrbench sldbench srbench srsd)

for required in \
  "${PROJECT_ROOT}/data/sim-datasets-llmsr/llm-srbench" \
  "${PROJECT_ROOT}/data/sldbench_repo" \
  "${PROJECT_ROOT}/srbench/docs/csv/datasets_info.csv" \
  "${PROJECT_ROOT}/srsd-benchmark/resource/datasets/srsd"; do
  if [[ ! -e "${required}" ]]; then
    printf 'required benchmark data is missing: %s\n' "${required}" >&2
    exit 2
  fi
done

mapfile -t lfs_pointers < <(
  rg -l '^version https://git-lfs.github.com/spec/v1$' \
    "${PROJECT_ROOT}/data/sim-datasets-llmsr/llm-srbench" -g '*.csv' || true
)
pointer_count="${#lfs_pointers[@]}"
if [[ "${pointer_count}" != "0" ]]; then
  printf 'LLM-SRBench still has %s unresolved Git LFS CSV files\n' "${pointer_count}" >&2
  exit 2
fi

curl -fsS "${LLMSR_AGENT_API_BASE}/models" >/dev/null
mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

{
  printf 'run_id\t%s\n' "${RUN_ID}"
  printf 'suite\tstandard_recovery_972\n'
  printf 'tasks\t972\n'
  printf 'validation_r2_threshold\t0.999\n'
  printf 'first_hit_rule\tstrictly_greater\n'
  printf 'case_timeout_sec\t%s\n' "${CASE_TIMEOUT_SEC}"
  printf 'model\t%s\n' "${LLMSR_AGENT_MODEL}"
  printf 'api_base\t%s\n' "${LLMSR_AGENT_API_BASE}"
  printf 'mode\tfull\n'
  printf 'fast_path\tdisabled\n'
  printf 'started_at\t%s\n' "$(date '+%F %T')"
} > "${RESULTS_ROOT}/search_efficiency_run_manifest.tsv"

sessions=()
for benchmark in "${BENCHMARKS[@]}"; do
  session="${SESSION_PREFIX}_${benchmark}"
  sessions+=("${session}")
  if tmux has-session -t "${session}" 2>/dev/null; then
    printf 'tmux session already exists: %s\n' "${session}" >&2
    exit 2
  fi
done

for index in "${!BENCHMARKS[@]}"; do
  benchmark="${BENCHMARKS[${index}]}"
  session="${sessions[${index}]}"
  printf -v worker_command '%q ' \
    bash "${SELF}" --worker "${RUN_ID}" "${benchmark}" "${RESULTS_ROOT}" \
    "${LOG_ROOT}" "${CASE_TIMEOUT_SEC}" "${LLMSR_AGENT_API_BASE}" "${LLMSR_AGENT_MODEL}"
  tmux new-session -d -s "${session}" "${worker_command}"
done

session_csv="$(IFS=,; printf '%s' "${sessions[*]}")"
monitor_session="${SESSION_PREFIX}_summary"
printf -v monitor_command '%q ' \
  bash "${SELF}" --monitor "${RESULTS_ROOT}" "${LOG_ROOT}" "${session_csv}"
tmux new-session -d -s "${monitor_session}" "${monitor_command}"

printf 'run_id=%s\n' "${RUN_ID}"
printf 'results=%s\n' "${RESULTS_ROOT}"
printf 'logs=%s\n' "${LOG_ROOT}"
printf 'workers=%s\n' "${session_csv}"
printf 'monitor=%s\n' "${monitor_session}"
