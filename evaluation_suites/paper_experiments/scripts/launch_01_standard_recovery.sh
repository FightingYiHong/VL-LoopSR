#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-standard_recovery_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/01_standard_recovery/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/01_standard_recovery/${RUN_ID}}"
RUN_OURS="${RUN_OURS:-${LAUNCH_VL_LOOPSR:-0}}"
RUN_BASELINES="${RUN_BASELINES:-${LAUNCH_BASELINES:-0}}"
RUN_LLM_SR="${RUN_LLM_SR:-${LAUNCH_OFFICIAL_LLMSR:-0}}"
RUN_ICSR="${RUN_ICSR:-${LAUNCH_OFFICIAL_ICSR:-0}}"
MAX_CASES="${MAX_CASES:-0}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

printf '[Fig. 2] standard recovery: 895 cases across LLMSRBench, SRBench/Feynman and SRSD\n'
printf 'results=%s\n' "${RESULTS_ROOT}"

if [[ "${RUN_OURS}" == "1" ]]; then
  RESULTS_BASE="${RESULTS_ROOT}/vega_sr" \
  LOG_DIR="${LOG_ROOT}/vega_sr" \
  CASE_BUDGET_SEC="${OURS_CASE_BUDGET_SEC:-300}" \
  MAX_CASES="${MAX_CASES}" \
    bash scripts/launch_v11_enhanced_300s.sh "${RUN_ID}"
fi

if [[ "${RUN_BASELINES}" == "1" ]]; then
  methods="${STANDARD_BASELINE_METHODS:-dso,gplearn,pysr,psrn}"
  benchmarks="${STANDARD_BENCHMARKS:-llmsrbench,srbench,srsd}"
  IFS=',' read -r -a method_list <<<"${methods}"
  IFS=',' read -r -a benchmark_list <<<"${benchmarks}"
  for method in "${method_list[@]}"; do
    case "${method}" in
      dso) config="evaluation_suites/gpu_symbolic_regression_fourbench/configs/dso_limited.yaml" ;;
      gplearn) config="evaluation_suites/cpu_symbolic_regression_fourbench/configs/gplearn_10min_1thread.yaml" ;;
      pysr) config="evaluation_suites/cpu_symbolic_regression_fourbench/configs/pysr_10min_1thread.yaml" ;;
      psrn) config="evaluation_suites/gpu_symbolic_regression_fourbench/configs/psrn_10min_safe.yaml" ;;
      *) printf 'Unsupported standard baseline: %s\n' "${method}" >&2; exit 2 ;;
    esac
    for benchmark in "${benchmark_list[@]}"; do
      command=(python3 scripts/run_cpu_baseline_benchmarks.py
        --benchmark "${benchmark}"
        --method "${method}"
        --config "${config}"
        --results-root "${RESULTS_ROOT}/${method}/${benchmark}"
        --random-state 42
        --isolate-cases
        --resume)
      if [[ "${MAX_CASES}" -gt 0 ]]; then
        command+=(--max-cases "${MAX_CASES}")
      fi
      "${command[@]}" 2>&1 | tee -a "${LOG_ROOT}/${method}_${benchmark}.log"
    done
  done
fi

if [[ "${RUN_LLM_SR}" == "1" ]]; then
  CASE_TIMEOUT_SEC="${LLM_SR_CASE_BUDGET_SEC:-600}" \
  OFFICIAL_LLMSR_BENCHMARKS="${STANDARD_BENCHMARKS:-llmsrbench,srbench,srsd}" \
  OFFICIAL_LLMSR_RESULTS_ROOT="${RESULTS_ROOT}/llm_sr" \
  OFFICIAL_LLMSR_LOG_ROOT="${LOG_ROOT}/llm_sr" \
    bash evaluation_suites/official_llmsr_fourbench/launch_official_llmsr_v11_budget.sh "${RUN_ID}"
fi

if [[ "${RUN_ICSR}" == "1" ]]; then
  OFFICIAL_ICSR_BENCHMARKS="${STANDARD_BENCHMARKS:-llmsrbench,srbench,srsd}" \
  OFFICIAL_ICSR_CASE_TIMEOUT_SEC="${ICSR_CASE_BUDGET_SEC:-600}" \
  OFFICIAL_ICSR_RESULTS_ROOT="${RESULTS_ROOT}/icsr" \
  OFFICIAL_ICSR_LOG_ROOT="${LOG_ROOT}/icsr" \
    bash evaluation_suites/official_icsr_fourbench/launch_official_icsr_v11_budget.sh "${RUN_ID}"
fi

if [[ "${RUN_OURS}${RUN_BASELINES}${RUN_LLM_SR}${RUN_ICSR}" == "0000" ]]; then
  printf 'Prepared only. Set RUN_OURS=1, RUN_BASELINES=1, RUN_LLM_SR=1 or RUN_ICSR=1.\n'
fi
