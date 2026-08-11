#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-agentic_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/06_agentic/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/06_agentic/${RUN_ID}}"
DATASET_ROOT="${INTERFERENCE_DATASET_ROOT:-data/generated/balanced_interference_96}"
RUN_AGENTIC="${RUN_AGENTIC:-0}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

printf '[Fig. 5] matched agentic comparison: initial evaluation only vs complete loop, 96 tasks\n'

if [[ "${RUN_AGENTIC}" == "1" && ! -f "${DATASET_ROOT}/manifest.csv" ]]; then
  python3 scripts/generate_v11_balanced_ablation_dataset.py \
    --out-dir "${DATASET_ROOT}" --num-cases 96 --n-train 128 --n-val 128 --n-test 1024
fi

if [[ "${RUN_AGENTIC}" == "1" ]]; then
  python3 scripts/run_v11_ablation_experiments.py \
    --suite component_ablation --dataset-root "${DATASET_ROOT}" \
    --methods "full,w_o_critic" --interleave-methods \
    --out-dir "${RESULTS_ROOT}/paired_runs" \
    --case-budget-sec 600 --parent-timeout-sec 630 --timeout-grace-sec 30 \
    --max-cases "${MAX_CASES:-0}" --resume \
    2>&1 | tee -a "${LOG_ROOT}/paired_runs.log"
else
  printf 'Prepared only. Set RUN_AGENTIC=1.\n'
fi
