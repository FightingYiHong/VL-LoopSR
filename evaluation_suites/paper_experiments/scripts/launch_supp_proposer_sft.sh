#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-supp_proposer_sft_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/supp_proposer_sft/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/supp_proposer_sft/${RUN_ID}}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

printf 'Descriptive 895-pair archived SFT comparison; not a checkpoint-only causal estimate.\n'
if [[ "${RUN_SFT:-0}" == "1" || "${ONLY_SUMMARIZE:-0}" == "1" ]]; then
  SFT_COMPARE_RESULTS_PARENT="${SFT_COMPARE_RESULTS_PARENT:-${RESULTS_ROOT}/runs}" \
  SFT_COMPARE_LOG_PARENT="${SFT_COMPARE_LOG_PARENT:-${LOG_ROOT}}" \
  SFT_COMPARE_REPORT_PARENT="${SFT_COMPARE_REPORT_PARENT:-${RESULTS_ROOT}}" \
  SFT_COMPARE_OUT_DIR="${SFT_COMPARE_OUT_DIR:-${RESULTS_ROOT}/summary}" \
  SFT_COMPARE_BENCHMARKS="${SFT_COMPARE_BENCHMARKS:-llmsrbench srbench srsd}" \
  SFT_COMPARE_ONLY_SUMMARIZE="${ONLY_SUMMARIZE:-0}" \
    bash scripts/launch_v11_sft_before_after_main_100s.sh "${RUN_ID}" \
    2>&1 | tee -a "${LOG_ROOT}/proposer_sft.log"
else
  printf 'Prepared only. Set RUN_SFT=1 or ONLY_SUMMARIZE=1.\n'
fi
