#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-multimodal_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/05_multimodal/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/05_multimodal/${RUN_ID}}"
DATASET_ROOT="${INTERFERENCE_DATASET_ROOT:-data/generated/balanced_interference_96}"
RUN_MULTIMODAL="${RUN_MULTIMODAL:-0}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

printf '[Fig. 4] matched multimodal comparison: numeric-only vs numeric + one image, 96 tasks\n'

if [[ "${RUN_MULTIMODAL}" == "1" && ! -f "${DATASET_ROOT}/manifest.csv" ]]; then
  python3 scripts/generate_v11_balanced_ablation_dataset.py \
    --out-dir "${DATASET_ROOT}" --num-cases 96 --n-train 128 --n-val 128 --n-test 1024
fi

if [[ "${RUN_MULTIMODAL}" == "1" ]]; then
  LLMSR_V11_FULL_BUDGET=1 \
  LLMSR_V11_FULL_BUDGET_TEXT_CALLS=1 \
  LLMSR_V11_FULL_BUDGET_MM_CALLS=1 \
  LLMSR_V11_FULL_BUDGET_PROPOSAL_K=12 \
  LLMSR_V11_FULL_BUDGET_REFINED_K=4 \
  LLMSR_V11_FULL_BUDGET_REFINE_ROUNDS=1 \
  LLMSR_V11_IMAGE_LOOP_MAX_ROUNDS=1 \
  LLMSR_V11_FORCE_INITIAL_MODALITY_PROPOSAL=1 \
  LLMSR_V11_MATCHED_MODALITY_CALLS=1 \
  LLMSR_V11_MATCHED_INITIAL_CANDIDATES=12 \
  LLMSR_V11_MATCH_REFINEMENT_BUDGET=1 \
  LLMSR_V11_FORCE_REFINEMENT_ROUNDS=1 \
    python3 scripts/run_v11_ablation_experiments.py \
    --suite component_ablation --dataset-root "${DATASET_ROOT}" \
    --methods "full,w_o_observer" --interleave-methods \
    --out-dir "${RESULTS_ROOT}/paired_runs" \
    --case-budget-sec 90 --parent-timeout-sec 120 --timeout-grace-sec 30 \
    --max-cases "${MAX_CASES:-0}" --resume \
    2>&1 | tee -a "${LOG_ROOT}/paired_runs.log"
  python3 scripts/compute_multimodal_ned.py \
    --results-csv "${RESULTS_ROOT}/paired_runs/all_v11_ablation_results.csv" \
    --manifest-csv "${DATASET_ROOT}/manifest.csv" \
    --reference-method full --out-dir "${RESULTS_ROOT}/ned" \
    2>&1 | tee -a "${LOG_ROOT}/ned.log"
else
  printf 'Prepared only. Set RUN_MULTIMODAL=1.\n'
fi
