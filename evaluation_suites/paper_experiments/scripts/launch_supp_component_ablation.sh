#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-supp_component_ablation_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/supp_component_ablation/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/supp_component_ablation/${RUN_ID}}"
DATASET_ROOT="${INTERFERENCE_DATASET_ROOT:-data/generated/balanced_interference_96}"
RUN_ABLATION="${RUN_ABLATION:-0}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

if [[ "${RUN_ABLATION}" == "1" && ! -f "${DATASET_ROOT}/manifest.csv" ]]; then
  python3 scripts/generate_v11_balanced_ablation_dataset.py \
    --out-dir "${DATASET_ROOT}" --num-cases 96 --n-train 128 --n-val 128 --n-test 1024
fi

if [[ "${RUN_ABLATION}" == "1" ]]; then
  python3 scripts/run_v11_ablation_experiments.py \
    --suite component_ablation --dataset-root "${DATASET_ROOT}" \
    --methods "full,w_o_observer_all,w_o_critic,w_o_proposer" --interleave-methods \
    --out-dir "${RESULTS_ROOT}/runs" \
    --case-budget-sec 600 --parent-timeout-sec 630 --timeout-grace-sec 30 \
    --max-cases "${MAX_CASES:-0}" --resume \
    2>&1 | tee -a "${LOG_ROOT}/component_ablation.log"
else
  printf 'Prepared only. Set RUN_ABLATION=1.\n'
fi
