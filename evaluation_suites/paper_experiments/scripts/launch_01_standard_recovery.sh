#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-standard_recovery_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/01_standard_recovery/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/01_standard_recovery/${RUN_ID}}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${ROOT_DIR}"

cat > "${RESULTS_ROOT}/README.txt" <<EOF
Standard formula recovery suite.

This suite corresponds to Fig. 2 in the paper. It aggregates the four
manuscript benchmark groups and launches official LLM baselines with per-case
hard timeout when requested.

Benchmarks: llmsrbench, sldbench, srbench/feynman, srsd.
Primary metrics: numerical complete fit (test R2 > 0.999), strict and SRBench
formula recovery, MSE/RMSE/NMSE/NRMSE, expression-tree nodes, expression
evaluations and runtime. PASS@100 is retained as a legacy compatibility screen.
Search efficiency is measured by the first candidate evaluation whose validation
R2 exceeds 0.999. Non-hits remain in the analysis as right-censored cases.
EOF

echo "[01-standard-recovery] results=${RESULTS_ROOT}"
echo "[01-standard-recovery] logs=${LOG_ROOT}"

if [[ "${LAUNCH_VL_LOOPSR:-0}" == "1" ]]; then
  RESULTS_BASE="${RESULTS_ROOT}" \
  LOG_DIR="${LOG_ROOT}/vl_loopsr" \
  STATUS_FILE="${RESULTS_ROOT}/vl_loopsr_status.tsv" \
  V11_300_FULL=1 \
  WAIT_FOR_CURRENT="${WAIT_FOR_CURRENT:-0}" \
  LLMSR_MAX_RUNTIME_PER_TASK_SEC="${CASE_TIMEOUT_SEC:-600}" \
  LLMSR_AGENT_API_BASE="${LLMSR_AGENT_API_BASE:-http://127.0.0.1:8001/v1}" \
  LLMSR_AGENT_API_KEY="${LLMSR_AGENT_API_KEY:-EMPTY}" \
  LLMSR_AGENT_MODEL="${LLMSR_AGENT_MODEL:-Qwen3-VL-32B-Instruct}" \
  LLMSR_V11_RESTORE_TEXT_PROPOSER=1 \
  LLMSR_V11_ENABLE_VLM_OBSERVER=1 \
  bash scripts/launch_v11_enhanced_300s.sh "${RUN_ID}"

  python scripts/summarize_standard_search_efficiency.py \
    --results-root "${RESULTS_ROOT}" \
    --output-dir "${RESULTS_ROOT}/search_efficiency" \
    > "${LOG_ROOT}/search_efficiency_summary.log" 2>&1
fi

if [[ "${LAUNCH_OFFICIAL_LLMSR:-0}" == "1" ]]; then
  CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-600}" \
  PASS_THRESHOLD="${PASS_THRESHOLD:-100}" \
  OFFICIAL_LLMSR_BENCHMARKS="${OFFICIAL_LLMSR_BENCHMARKS:-sldbench,llmsrbench,srsd,srbench}" \
  bash evaluation_suites/official_llmsr_fourbench/launch_official_llmsr_v11_budget.sh \
    "paper_01_official_llmsr_${RUN_ID}"
fi

if [[ "${LAUNCH_OFFICIAL_ICSR:-0}" == "1" ]]; then
  OFFICIAL_ICSR_CASE_TIMEOUT_SEC="${OFFICIAL_ICSR_CASE_TIMEOUT_SEC:-600}" \
  OFFICIAL_ICSR_BENCHMARKS="${OFFICIAL_ICSR_BENCHMARKS:-sldbench,llmsrbench,srsd,srbench}" \
  bash evaluation_suites/official_icsr_fourbench/launch_official_icsr_v11_budget.sh \
    "paper_01_official_icsr_${RUN_ID}"
fi
