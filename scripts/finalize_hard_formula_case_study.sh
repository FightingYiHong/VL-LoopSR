#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ID="${1:-hard_formula_confirmatory_20260802}"
MAIN_SESSION="${2:-hard_formula_confirmatory_20260802}"
LLMSR_SESSION="${3:-hard_formula_llmsr_retry_20260802}"
RUN_ROOT="${ROOT_DIR}/runs/paper_experiments/01_standard_recovery/${RUN_ID}"
LOG_ROOT="${ROOT_DIR}/logs/paper_experiments/01_standard_recovery/${RUN_ID}"
REPEAT_SEEDS="${HARD_FORMULA_REPEAT_SEEDS:-101 202 303 404 505}"
BUDGET_SEC="${HARD_FORMULA_BUDGET_SEC:-600}"
MODEL="${HARD_FORMULA_MODEL:-Qwen3-VL-32B-Instruct}"
API_BASE="${HARD_FORMULA_API_BASE:-http://127.0.0.1:8001/v1}"

cd "${ROOT_DIR}"
mkdir -p "${RUN_ROOT}/summary" "${LOG_ROOT}"

while tmux has-session -t "${MAIN_SESSION}" 2>/dev/null \
    || tmux has-session -t "${LLMSR_SESSION}" 2>/dev/null; do
  sleep 60
done

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export LLMSR_V10_PATH="${ROOT_DIR}/scripts/test_fey_v11_complexity_exact.py"
export SRSD_V10_PATH="${LLMSR_V10_PATH}"
export LLMSR_AGENT_API_BASE="${API_BASE}"
export LLMSR_AGENT_API_KEY=EMPTY
export LLMSR_AGENT_MODEL="${MODEL}"
export LLMSR_PLANNER_API_BASE="${API_BASE}"
export LLMSR_PLANNER_API_KEY=EMPTY
export LLMSR_PLANNER_MODEL="${MODEL}"
export LLMSR_EVAL_PROFILE=quality
export LLMSR_METHOD_MODE=planner_guided
export LLMSR_LIGHT_MODE=0
export LLMSR_MAX_RUNTIME_PER_TASK_SEC="${BUDGET_SEC}"
export LLMSR_V11_DISABLE_RUNTIME_FAST_PATH=1
export LLMSR_V11_BUDGET_AWARE_MODE=on
export LLMSR_V11_FULL_BUDGET=1
export LLMSR_V11_LOW_DIM_FULL_BUDGET=1
export LLMSR_V11_FULL_BUDGET_TEXT_CALLS=4
export LLMSR_V11_FULL_BUDGET_MM_CALLS=3
export LLMSR_V11_FULL_BUDGET_PROPOSAL_K=16
export LLMSR_V11_FULL_BUDGET_REFINED_K=8
export LLMSR_V11_FULL_BUDGET_REFINE_ROUNDS=4
export LLMSR_V11_ENABLE_STRUCTURE_EVALUATOR=1
export LLMSR_V11_ENABLE_GENERALIZATION_GUARD=1
export LLMSR_ALLOW_TRUE_EXPR_DIAGNOSTICS=0
export LLMSR_USE_TEST_FOR_SELECTION=0
export SRSD_ROOT="${ROOT_DIR}/srsd-benchmark/resource/datasets/srsd"
export SRSD_DATASET_DIRS=easy_set
export SRSD_ALLOW_BASENAMES=feynman-ii.2.42
export SRSD_HARD_TIMEOUT_SEC="$((BUDGET_SEC + 360))"
export SRSD_RESUME=0

retry_log="${LOG_ROOT}/ours_srsd_retry.log"
: > "${retry_log}"
for seed in ${REPEAT_SEEDS}; do
  seed_root="${RUN_ROOT}/ours/seed_${seed}/srsd"
  mkdir -p "${seed_root}"
  printf '[%s] seed=%s start\n' "$(date -Is)" "${seed}" | tee -a "${retry_log}"
  LLMSR_REPEAT_SEED="${seed}" \
  SRSD_RANDOM_SEED="${seed}" \
  SRSD_V10_RESULTS_ROOT="${seed_root}" \
    conda run --no-capture-output -n sr python scripts/run_srds.py >> "${retry_log}" 2>&1
  printf '[%s] seed=%s done\n' "$(date -Is)" "${seed}" | tee -a "${retry_log}"
done

conda run --no-capture-output -n sr \
  python scripts/summarize_hard_formula_confirmatory.py \
    --run-root "${RUN_ROOT}" \
    --output-dir "${RUN_ROOT}/summary" \
  > "${LOG_ROOT}/final_summary.log" 2>&1

printf 'finalized_at\t%s\n' "$(date -Is)" >> "${RUN_ROOT}/run_manifest.tsv"
