#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
SELF="${PROJECT_ROOT}/evaluation_suites/paper_experiments/scripts/launch_01_baseline_search_efficiency_full.sh"
BENCHMARKS=(llmsrbench sldbench srbench srsd)

common_env() {
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
  export SLDBENCH_HUB_REPO="${PROJECT_ROOT}/data/sldbench_repo"
  export LLMSRBENCH_CASES_ROOT="${PROJECT_ROOT}/data/sim-datasets-llmsr/llm-srbench"
  export SRBENCH_ROOT="${PROJECT_ROOT}/srbench"
  export SRBENCH_DATASETS_INFO_CSV="${PROJECT_ROOT}/srbench/docs/csv/datasets_info.csv"
  export SRSD_ROOT="${PROJECT_ROOT}/srsd-benchmark/resource/datasets/srsd"
  export CPU_SR_SEARCH_EFFICIENCY_THRESHOLD="0.999"
  export CPU_SR_CASE_SUBPROCESS_GRACE_SEC="${CPU_SR_CASE_SUBPROCESS_GRACE_SEC:-120}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
}

run_cpu_worker() {
  local method="$1"
  local benchmark="$2"
  local results_root="$3"
  local log_root="$4"
  local budget_sec="${BASELINE_EFFICIENCY_CASE_BUDGET_SEC:-600}"
  local env_name config_path gpu_id cpu_id benchmark_offset
  case "${benchmark}" in
    llmsrbench) benchmark_offset=0 ;;
    sldbench) benchmark_offset=1 ;;
    srbench) benchmark_offset=2 ;;
    srsd) benchmark_offset=3 ;;
  esac
  case "${method}" in
    gplearn)
      cpu_id=$((0 + benchmark_offset))
      env_name="srbench-gplearn"
      if [[ "${budget_sec}" == "100" ]]; then
        config_path="evaluation_suites/cpu_symbolic_regression_fourbench/configs/gplearn_100s_1thread.yaml"
      else
        config_path="evaluation_suites/cpu_symbolic_regression_fourbench/configs/gplearn_10min_1thread.yaml"
      fi
      ;;
    pysr)
      cpu_id=$((4 + benchmark_offset))
      env_name="srbench-pysr"
      if [[ "${budget_sec}" == "100" ]]; then
        config_path="evaluation_suites/cpu_symbolic_regression_fourbench/configs/pysr_100s_1thread.yaml"
      else
        config_path="evaluation_suites/cpu_symbolic_regression_fourbench/configs/pysr_10min_1thread.yaml"
      fi
      ;;
    dso)
      cpu_id=$((8 + benchmark_offset))
      env_name="dso"
      if [[ "${budget_sec}" == "100" ]]; then
        config_path="evaluation_suites/cpu_symbolic_regression_fourbench/configs/dso_100s.yaml"
      else
        config_path="evaluation_suites/cpu_symbolic_regression_fourbench/configs/dso_10min.yaml"
      fi
      ;;
    psrn)
      cpu_id=$((12 + benchmark_offset))
      env_name="PSRN"
      if [[ "${budget_sec}" == "100" ]]; then
        config_path="evaluation_suites/gpu_symbolic_regression_fourbench/configs/psrn_100s_safe.yaml"
      else
        config_path="evaluation_suites/gpu_symbolic_regression_fourbench/configs/psrn_10min_safe.yaml"
      fi
      case "${benchmark}" in
        llmsrbench) gpu_id="${PSRN_GPU_LLMSRBENCH:-0}" ;;
        sldbench) gpu_id="${PSRN_GPU_SLDBENCH:-3}" ;;
        srbench) gpu_id="${PSRN_GPU_SRBENCH:-6}" ;;
        srsd) gpu_id="${PSRN_GPU_SRSD:-8}" ;;
      esac
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      ;;
    *)
      printf 'unknown CPU baseline method: %s\n' "${method}" >&2
      exit 2
      ;;
  esac

  common_env
  export CPU_SR_CONFIGURED_WALL_BUDGET_SEC="${budget_sec}"
  if [[ "${budget_sec}" == "100" ]]; then
    export CPU_SR_CASE_SUBPROCESS_GRACE_SEC="${BASELINE_EFFICIENCY_CASE_SUBPROCESS_GRACE_SEC:-30}"
  fi
  local worker_root="${results_root}/${method}/${benchmark}"
  mkdir -p "${worker_root}" "${log_root}/${method}"
  cd "${PROJECT_ROOT}"
  exec taskset -c "${cpu_id}" "${CONDA_BIN}" run --no-capture-output -n "${env_name}" \
    python scripts/run_cpu_baseline_benchmarks.py \
      --benchmark "${benchmark}" \
      --method "${method}" \
      --config "${config_path}" \
      --results-root "${worker_root}" \
      --random-state 42 \
      --isolate-cases \
      --resume \
    >> "${log_root}/${method}/${benchmark}.log" 2>&1
}

run_icsr_benchmark_worker() {
  local benchmark="$1"
  local results_root="$2"
  local log_root="$3"
  local budget_sec="${BASELINE_EFFICIENCY_CASE_BUDGET_SEC:-600}"
  local cpu_id
  case "${benchmark}" in
    llmsrbench) cpu_id=16 ;;
    srbench) cpu_id=17 ;;
    srsd) cpu_id=18 ;;
    *) printf 'unsupported ICSR benchmark: %s\n' "${benchmark}" >&2; exit 2 ;;
  esac
  common_env
  mkdir -p "${results_root}/icsr" "${log_root}/icsr"
  cd "${PROJECT_ROOT}"
  exec taskset -c "${cpu_id}" env \
    OFFICIAL_ICSR_ROOT="${OFFICIAL_ICSR_ROOT:-${PROJECT_ROOT}/external_repos/In-Context-Symbolic-Regression}" \
    OFFICIAL_ICSR_MODEL_NAME="${OFFICIAL_ICSR_MODEL_NAME:-Qwen3-VL-32B-Instruct}" \
    OFFICIAL_ICSR_OPENAI_BASE_URL="${OFFICIAL_ICSR_OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}" \
    OFFICIAL_ICSR_OPENAI_API_KEY="${OFFICIAL_ICSR_OPENAI_API_KEY:-EMPTY}" \
    OPENAI_BASE_URL="${OFFICIAL_ICSR_OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}" \
    OPENAI_API_KEY="${OFFICIAL_ICSR_OPENAI_API_KEY:-EMPTY}" \
    NO_PROXY="127.0.0.1,localhost" \
    no_proxy="127.0.0.1,localhost" \
    "${CONDA_BIN}" run --no-capture-output -n base \
      python evaluation_suites/official_icsr_fourbench/run_official_icsr_fourbench.py \
        --benchmarks "${benchmark}" \
        --results-root "${results_root}/icsr" \
        --case-timeout-sec "${budget_sec}" \
        --iterations 500 \
        --method-name icsr \
        --resume \
    >> "${log_root}/icsr/resume_${benchmark}.log" 2>&1
}

run_llmsr_worker() {
  local benchmark="$1"
  local results_root="$2"
  local log_root="$3"
  local budget_sec="${BASELINE_EFFICIENCY_CASE_BUDGET_SEC:-600}"
  local cpu_id
  case "${benchmark}" in
    llmsrbench) cpu_id=19 ;;
    srbench) cpu_id=20 ;;
    srsd) cpu_id=21 ;;
    *) printf 'unsupported LLM-SR benchmark: %s\n' "${benchmark}" >&2; exit 2 ;;
  esac
  common_env
  mkdir -p "${results_root}/llmsr" "${log_root}/llmsr"
  cd "${PROJECT_ROOT}"
  exec taskset -c "${cpu_id}" env \
    OFFICIAL_LLMSR_ROOT="${OFFICIAL_LLMSR_ROOT:-${PROJECT_ROOT}/external_repos/LLM-SR}" \
    OFFICIAL_LLMSR_API_BASE="${OFFICIAL_LLMSR_API_BASE:-http://127.0.0.1:8001/v1}" \
    OFFICIAL_LLMSR_API_KEY="${OFFICIAL_LLMSR_API_KEY:-EMPTY}" \
    OFFICIAL_LLMSR_MODEL="${OFFICIAL_LLMSR_MODEL:-Qwen3-VL-32B-Instruct}" \
    NO_PROXY="127.0.0.1,localhost" \
    no_proxy="127.0.0.1,localhost" \
    "${CONDA_BIN}" run --no-capture-output -n llmsr \
      python evaluation_suites/official_llmsr_fourbench/run_official_llmsr_fourbench.py \
        --benchmarks "${benchmark}" \
        --results-root "${results_root}/llmsr" \
        --api-base "${OFFICIAL_LLMSR_API_BASE:-http://127.0.0.1:8001/v1}" \
        --api-key "${OFFICIAL_LLMSR_API_KEY:-EMPTY}" \
        --model "${OFFICIAL_LLMSR_MODEL:-Qwen3-VL-32B-Instruct}" \
        --case-timeout-sec "${budget_sec}" \
        --evaluate-timeout-sec 30 \
        --request-timeout-sec 90 \
        --samples-per-prompt 4 \
        --max-sample-nums 100000 \
        --official-commit 41c212312df6c16d936c9cb395356a62774c47e3 \
        --isolate-cases \
        --resume \
    >> "${log_root}/llmsr/${benchmark}.log" 2>&1
}

run_icsr_worker() {
  local results_root="$1"
  local log_root="$2"
  common_env
  mkdir -p "${results_root}/icsr" "${log_root}/icsr"
  cd "${PROJECT_ROOT}"
  exec taskset -c 16-19 env \
    OFFICIAL_ICSR_ROOT="${OFFICIAL_ICSR_ROOT:-${PROJECT_ROOT}/external_repos/In-Context-Symbolic-Regression}" \
    OFFICIAL_ICSR_MODEL_NAME="${OFFICIAL_ICSR_MODEL_NAME:-Qwen3-VL-32B-Instruct}" \
    OFFICIAL_ICSR_OPENAI_BASE_URL="${OFFICIAL_ICSR_OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}" \
    OFFICIAL_ICSR_OPENAI_API_KEY="${OFFICIAL_ICSR_OPENAI_API_KEY:-EMPTY}" \
    OPENAI_BASE_URL="${OFFICIAL_ICSR_OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}" \
    OPENAI_API_KEY="${OFFICIAL_ICSR_OPENAI_API_KEY:-EMPTY}" \
    NO_PROXY="127.0.0.1,localhost" \
    no_proxy="127.0.0.1,localhost" \
    "${CONDA_BIN}" run --no-capture-output -n base \
      python evaluation_suites/official_icsr_fourbench/run_official_icsr_fourbench.py \
        --benchmarks "llmsrbench,sldbench,srbench,srsd" \
        --results-root "${results_root}/icsr" \
        --case-timeout-sec 600 \
        --iterations 500 \
        --method-name icsr \
        --resume \
    >> "${log_root}/icsr/run.log" 2>&1
}

run_monitor() {
  local results_root="$1"
  local log_root="$2"
  local session_csv="$3"
  local interval="${BASELINE_EFFICIENCY_MONITOR_INTERVAL_SEC:-300}"
  local ours_case_rows="${OURS_SEARCH_EFFICIENCY_CASE_ROWS:-${PROJECT_ROOT}/runs/paper_experiments/01_standard_recovery/standard_search_efficiency_full_20260724_152138/search_efficiency/standard_search_efficiency_case_rows.csv}"
  IFS=',' read -r -a sessions <<< "${session_csv}"
  while true; do
    local active=0
    local session
    for session in "${sessions[@]}"; do
      if tmux has-session -t "${session}" 2>/dev/null; then
        active=1
      fi
    done
    {
      printf 'timestamp\tmethod\tbenchmark\tcompleted\ttarget\n'
      local method benchmark completed target
      local active_benchmarks=(llmsrbench srbench srsd)
      for method in gplearn pysr dso psrn icsr llmsr; do
        for benchmark in "${active_benchmarks[@]}"; do
          completed="$(find "${results_root}/${method}/${benchmark}/case_results" -type f -name '*.json' 2>/dev/null | wc -l)"
          case "${benchmark}" in
            llmsrbench) target=240 ;;
            srbench) target=417 ;;
            srsd) target=238 ;;
          esac
          printf '%s\t%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${method}" "${benchmark}" "${completed}" "${target}"
        done
      done
    } > "${results_root}/progress.tsv"
    "${CONDA_BIN}" run --no-capture-output -n base \
      python "${PROJECT_ROOT}/scripts/summarize_baseline_search_efficiency.py" \
        --baseline-root "${results_root}" \
        --ours-case-rows "${ours_case_rows}" \
        --output-dir "${results_root}/search_efficiency" \
      > "${log_root}/search_efficiency_summary.log" 2>&1 || true
    if [[ "${active}" == "0" ]]; then
      break
    fi
    sleep "${interval}"
  done
}

case "${1:-}" in
  --cpu-worker)
    shift
    run_cpu_worker "$@"
    exit
    ;;
  --icsr-worker)
    shift
    run_icsr_worker "$@"
    exit
    ;;
  --icsr-benchmark-worker)
    shift
    run_icsr_benchmark_worker "$@"
    exit
    ;;
  --llmsr-worker)
    shift
    run_llmsr_worker "$@"
    exit
    ;;
  --monitor)
    shift
    run_monitor "$@"
    exit
    ;;
esac

RUN_ID="${1:-baseline_search_efficiency_full_$(date +%Y%m%d_%H%M%S)}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/01_standard_recovery/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/01_standard_recovery/${RUN_ID}}"
SESSION_PREFIX="${SESSION_PREFIX:-baseff_$(date +%H%M%S)}"
mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

cat > "${RESULTS_ROOT}/search_efficiency_run_manifest.tsv" <<EOF
run_id	${RUN_ID}
suite	standard_recovery_972
tasks_per_method	972
methods	gplearn,pysr,dso,psrn,icsr
validation_r2_threshold	0.999
first_hit_rule	strictly_greater
case_timeout_sec	${BASELINE_EFFICIENCY_CASE_BUDGET_SEC:-600}
test_selection	disabled
random_seed	42
cpu_affinity	one dedicated CPU per gplearn,pysr,dso,psrn worker; CPUs 16-19 for ICSR
mode	full
fast_path	disabled
started_at	$(date '+%F %T')
EOF

sessions=()
for method in gplearn pysr dso psrn; do
  for benchmark in "${BENCHMARKS[@]}"; do
    mkdir -p "${RESULTS_ROOT}/${method}/${benchmark}/case_results"
    session="${SESSION_PREFIX}_${method}_${benchmark}"
    sessions+=("${session}")
    printf -v command '%q ' bash "${SELF}" --cpu-worker \
      "${method}" "${benchmark}" "${RESULTS_ROOT}" "${LOG_ROOT}"
    tmux new-session -d -s "${session}" "${command}"
  done
done

icsr_session="${SESSION_PREFIX}_icsr"
for benchmark in "${BENCHMARKS[@]}"; do
  mkdir -p "${RESULTS_ROOT}/icsr/${benchmark}/case_results"
done
sessions+=("${icsr_session}")
printf -v command '%q ' bash "${SELF}" --icsr-worker "${RESULTS_ROOT}" "${LOG_ROOT}"
tmux new-session -d -s "${icsr_session}" "${command}"

session_csv="$(IFS=,; printf '%s' "${sessions[*]}")"
monitor_session="${SESSION_PREFIX}_monitor"
printf -v command '%q ' bash "${SELF}" --monitor "${RESULTS_ROOT}" "${LOG_ROOT}" "${session_csv}"
tmux new-session -d -s "${monitor_session}" "${command}"

printf 'run_id=%s\n' "${RUN_ID}"
printf 'results=%s\n' "${RESULTS_ROOT}"
printf 'logs=%s\n' "${LOG_ROOT}"
printf 'workers=%s\n' "${session_csv}"
printf 'monitor=%s\n' "${monitor_session}"
