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
Primary acceptance screen: PASS@100.
EOF

echo "[01-standard-recovery] results=${RESULTS_ROOT}"
echo "[01-standard-recovery] logs=${LOG_ROOT}"

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
