#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-component_ablation_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/06_component_ablation/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/06_component_ablation/${RUN_ID}}"
ABLATION_DATASET_ROOT="${ABLATION_DATASET_ROOT:-data/v11_balanced_component_ablation_96}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${ROOT_DIR}"

cat > "${RESULTS_ROOT}/README.txt" <<EOF
Component ablation suite.

This suite corresponds to Fig. 7 in the paper.

Dataset: matched balanced 96-task component-ablation suite.
Default dataset root: ${ABLATION_DATASET_ROOT}

Variants:
- full
- w_o_observer_all
- w_o_critic
- w_o_proposer

Primary readouts: paired test R2/NMSE, strict/SRBench recovery, active-variable
recall, proxy misuse, complexity, runtime, valid-expression count, and loop
trajectory.

Controls:
  LAUNCH_ABLATION=1
  ABLATION_METHODS=full,w_o_observer_all,w_o_critic,w_o_proposer
EOF

echo "[06-ablation] results=${RESULTS_ROOT}"
echo "[06-ablation] logs=${LOG_ROOT}"
echo "[06-ablation] dataset=${ABLATION_DATASET_ROOT}"

if [[ ! -f "${ABLATION_DATASET_ROOT}/manifest.csv" ]]; then
  echo "[06-ablation] dataset missing; generating balanced 96 suite"
  python3 scripts/generate_v11_balanced_ablation_dataset.py \
    --out-dir "${ABLATION_DATASET_ROOT}" \
    --num-cases 96 \
    --n-train 128 \
    --n-val 128 \
    --n-test 1024
fi

if [[ "${LAUNCH_ABLATION:-0}" == "1" ]]; then
  echo "[06-ablation] launching methods=${ABLATION_METHODS:-full,w_o_observer_all,w_o_critic,w_o_proposer}"
  python3 scripts/run_v11_ablation_experiments.py \
    --suite component_ablation \
    --dataset-root "${ABLATION_DATASET_ROOT}" \
    --methods "${ABLATION_METHODS:-full,w_o_observer_all,w_o_critic,w_o_proposer}" \
    --out-dir "${RESULTS_ROOT}/ablation" \
    --case-budget-sec "${ABLATION_CASE_BUDGET_SEC:-600}" \
    --parent-timeout-sec "${ABLATION_PARENT_TIMEOUT_SEC:-630}" \
    --timeout-grace-sec "${ABLATION_TIMEOUT_GRACE_SEC:-30}" \
    --resume \
    2>&1 | tee -a "${LOG_ROOT}/component_ablation.log"
else
  echo "[06-ablation] scaffold only; set LAUNCH_ABLATION=1 to run."
fi
