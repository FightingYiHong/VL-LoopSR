#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-qwen3_text_vs_vl_balanced96_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/07_multimodal_model_comparison/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/07_multimodal_model_comparison/${RUN_ID}}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/v11_balanced_component_ablation_96}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/evaluation_suites/paper_experiments/configs/multimodal_model_comparison_balanced96.json}"
REPEAT_SEED="${REPEAT_SEED:-0}"
CASE_BUDGET_SEC="${CASE_BUDGET_SEC:-90}"
PARENT_TIMEOUT_SEC="${PARENT_TIMEOUT_SEC:-120}"
TIMEOUT_GRACE_SEC="${TIMEOUT_GRACE_SEC:-30}"
MAX_CASES="${MAX_CASES:-0}"
METHODS="${METHODS:-qwen3_vl_multimodal,qwen3_vl_numeric_only}"

mkdir -p "${RESULTS_ROOT}/seed_${REPEAT_SEED}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

# All inference endpoints in this experiment are local.  The host may expose a
# SOCKS proxy without the optional httpx SOCKS dependency; clear proxy settings
# so the OpenAI-compatible clients connect directly to 127.0.0.1.
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy

python3 scripts/run_v11_ablation_experiments.py \
  --experiment-config "${CONFIG}" \
  --suite component_ablation \
  --dataset-root "${DATASET_ROOT}" \
  --methods "${METHODS}" \
  --out-dir "${RESULTS_ROOT}/seed_${REPEAT_SEED}" \
  --n-train 128 \
  --n-val 128 \
  --n-test 1024 \
  --repeat-seed "${REPEAT_SEED}" \
  --random-state 20260526 \
  --case-budget-sec "${CASE_BUDGET_SEC}" \
  --timeout-grace-sec "${TIMEOUT_GRACE_SEC}" \
  --parent-timeout-sec "${PARENT_TIMEOUT_SEC}" \
  --max-cases "${MAX_CASES}" \
  --interleave-methods \
  --resume \
  2>&1 | tee -a "${LOG_ROOT}/seed_${REPEAT_SEED}.log"
