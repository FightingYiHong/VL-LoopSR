#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-proposer_sft_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/04_proposer_sft/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/04_proposer_sft/${RUN_ID}}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${ROOT_DIR}"

cat > "${RESULTS_ROOT}/README.txt" <<EOF
Proposer SFT before/after suite.

This suite corresponds to Fig. 5 in the paper.

Contrast:
- before: base Qwen3-VL-32B-Instruct Proposer
- after: SFT-adapted Proposer
- fixed: benchmark list, wrappers, Evaluator, validation selector

Primary readouts: candidate count, recovered-case MSE, repair utility, runtime,
and timeout behavior.

Controls:
  LAUNCH_SFT_COMPARISON=1
  ONLY_SUMMARIZE=1
  SFT_COMPARE_BEFORE_ROOT=/path/to/before
  SFT_COMPARE_AFTER_ROOT=/path/to/after
EOF

echo "[04-proposer-sft] results=${RESULTS_ROOT}"
echo "[04-proposer-sft] logs=${LOG_ROOT}"

if [[ "${LAUNCH_SFT_COMPARISON:-0}" == "1" || "${ONLY_SUMMARIZE:-0}" == "1" ]]; then
  SFT_COMPARE_RESULTS_PARENT="${SFT_COMPARE_RESULTS_PARENT:-${RESULTS_ROOT}/runs}" \
  SFT_COMPARE_LOG_PARENT="${SFT_COMPARE_LOG_PARENT:-${LOG_ROOT}}" \
  SFT_COMPARE_REPORT_PARENT="${SFT_COMPARE_REPORT_PARENT:-${RESULTS_ROOT}}" \
  SFT_COMPARE_OUT_DIR="${SFT_COMPARE_OUT_DIR:-${RESULTS_ROOT}/summary}" \
  SFT_COMPARE_TIME_BUDGET="${SFT_COMPARE_TIME_BUDGET:-100}" \
  SFT_COMPARE_PASS_THRESHOLD="${SFT_COMPARE_PASS_THRESHOLD:-100}" \
  SFT_COMPARE_ONLY_SUMMARIZE="${ONLY_SUMMARIZE:-0}" \
  bash scripts/launch_v11_sft_before_after_main_100s.sh "${RUN_ID}" \
    2>&1 | tee -a "${LOG_ROOT}/proposer_sft.log"
else
  echo "[04-proposer-sft] scaffold only; set LAUNCH_SFT_COMPARISON=1 or ONLY_SUMMARIZE=1."
fi
