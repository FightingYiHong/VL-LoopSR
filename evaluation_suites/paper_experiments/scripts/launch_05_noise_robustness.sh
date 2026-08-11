#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-noise_robustness_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/05_noise_robustness/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/05_noise_robustness/${RUN_ID}}"
NOISE_DATASET_MANIFEST="${NOISE_DATASET_MANIFEST:-data/noise_robustness_metric_v2/manifest.csv}"
LAUNCH_VL_LOOPSR="${LAUNCH_VL_LOOPSR:-${LAUNCH_OURS_V11:-0}}"
NOISE_VL_LOOPSR_CASE_BUDGET_SEC="${NOISE_VL_LOOPSR_CASE_BUDGET_SEC:-${NOISE_V11_CASE_BUDGET_SEC:-100}}"
NOISE_VL_LOOPSR_PARENT_TIMEOUT_SEC="${NOISE_VL_LOOPSR_PARENT_TIMEOUT_SEC:-${NOISE_V11_PARENT_TIMEOUT_SEC:-130}}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${ROOT_DIR}"

cat > "${RESULTS_ROOT}/README.txt" <<EOF
Noise robustness suite.

This suite corresponds to Fig. 6 in the paper.

Dataset: NoiseRobust-Metric v2.
Default manifest: ${NOISE_DATASET_MANIFEST}
Cases: 20 formulas x noise levels 0/0.001/0.01/0.1 x 5 seeds = 400 tasks.

Methods see noisy train/validation targets and are scored on clean held-out test
targets.

Controls:
  LAUNCH_VL_LOOPSR=1
  LAUNCH_BASELINES=1
  NOISE_BASELINE_METHODS=psrn_pse,itea,deap,bingo,dso,llm_direct,llm_sr
EOF

echo "[05-noise] results=${RESULTS_ROOT}"
echo "[05-noise] logs=${LOG_ROOT}"
echo "[05-noise] manifest=${NOISE_DATASET_MANIFEST}"

if [[ ! -f "${NOISE_DATASET_MANIFEST}" ]]; then
  echo "[05-noise] manifest missing; generating NoiseRobust-Metric v2"
  python3 scripts/generate_noise_robustness_dataset.py \
    --out-dir "$(dirname "${NOISE_DATASET_MANIFEST}")" \
    --formula-count 20 \
    --noise-levels "0,0.001,0.01,0.1" \
    --repeat-seeds 5 \
    --n-train 512 \
    --n-val 256 \
    --n-test 1024
fi

if [[ "${LAUNCH_VL_LOOPSR:-0}" == "1" ]]; then
  echo "[05-noise] launching VL-LoopSR"
  python3 scripts/run_v11_noise_robustness.py \
    --out-dir "${RESULTS_ROOT}/vl_loopsr" \
    --dataset-manifest "${NOISE_DATASET_MANIFEST}" \
    --case-budget-sec "${NOISE_VL_LOOPSR_CASE_BUDGET_SEC}" \
    --parent-timeout-sec "${NOISE_VL_LOOPSR_PARENT_TIMEOUT_SEC}" \
    --timeout-grace-sec "${NOISE_TIMEOUT_GRACE_SEC:-30}" \
    --resume \
    2>&1 | tee -a "${LOG_ROOT}/vl_loopsr.log"
fi

if [[ "${LAUNCH_BASELINES:-0}" == "1" ]]; then
  echo "[05-noise] launching baselines=${NOISE_BASELINE_METHODS:-psrn_pse,itea,deap,bingo,dso,llm_direct,llm_sr}"
  python3 scripts/run_noise_robustness_baselines.py \
    --methods "${NOISE_BASELINE_METHODS:-psrn_pse,itea,deap,bingo,dso,llm_direct,llm_sr}" \
    --out-dir "${RESULTS_ROOT}/baselines" \
    --dataset-manifest "${NOISE_DATASET_MANIFEST}" \
    --case-budget-sec "${NOISE_BASELINE_CASE_BUDGET_SEC:-100}" \
    --parent-timeout-sec "${NOISE_BASELINE_PARENT_TIMEOUT_SEC:-130}" \
    --timeout-grace-sec "${NOISE_TIMEOUT_GRACE_SEC:-30}" \
    --resume \
    2>&1 | tee -a "${LOG_ROOT}/baselines.log"
fi

if [[ "${LAUNCH_VL_LOOPSR:-0}" != "1" && "${LAUNCH_BASELINES:-0}" != "1" ]]; then
  echo "[05-noise] scaffold only; set LAUNCH_VL_LOOPSR=1 or LAUNCH_BASELINES=1."
fi
