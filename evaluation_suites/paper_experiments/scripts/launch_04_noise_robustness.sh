#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-noise_robustness_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/04_noise_robustness/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/04_noise_robustness/${RUN_ID}}"
DATASET_MANIFEST="${NOISE_DATASET_MANIFEST:-data/generated/noise_sr_20/manifest.csv}"
RUN_OURS="${RUN_OURS:-0}"
RUN_BASELINES="${RUN_BASELINES:-0}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

printf '[Fig. 3d] NoiseSR-20: 20 formulas x 3 noise levels x 3 seeds = 180 tasks\n'

if [[ ( "${RUN_OURS}" == "1" || "${RUN_BASELINES}" == "1" ) && ! -f "${DATASET_MANIFEST}" ]]; then
  python3 scripts/generate_noise_robustness_dataset.py \
    --out-dir "$(dirname "${DATASET_MANIFEST}")" \
    --formula-count 20 --noise-levels "0,0.001,0.01" --repeat-seeds 3 \
    --n-train 512 --n-val 256 --n-test 1024 --random-state 42
fi

if [[ "${RUN_OURS}" == "1" ]]; then
  python3 scripts/run_v11_noise_robustness.py \
    --out-dir "${RESULTS_ROOT}/vega_sr" \
    --dataset-manifest "${DATASET_MANIFEST}" \
    --case-budget-sec "${NOISE_VEGA_SR_CASE_BUDGET_SEC:-100}" \
    --parent-timeout-sec "${NOISE_VEGA_SR_PARENT_TIMEOUT_SEC:-130}" \
    --timeout-grace-sec "${NOISE_TIMEOUT_GRACE_SEC:-30}" \
    --max-cases "${MAX_CASES:-0}" --resume \
    2>&1 | tee -a "${LOG_ROOT}/vega_sr.log"
fi

if [[ "${RUN_BASELINES}" == "1" ]]; then
  python3 scripts/run_noise_robustness_baselines.py \
    --methods "${NOISE_BASELINE_METHODS:-psrn_pse,itea,ffx,gplearn,deap,bingo,dso,official_llm_sr,llm_direct}" \
    --out-dir "${RESULTS_ROOT}/baselines" \
    --dataset-manifest "${DATASET_MANIFEST}" \
    --case-budget-sec "${NOISE_BASELINE_CASE_BUDGET_SEC:-100}" \
    --parent-timeout-sec "${NOISE_BASELINE_PARENT_TIMEOUT_SEC:-130}" \
    --timeout-grace-sec "${NOISE_TIMEOUT_GRACE_SEC:-30}" \
    --max-cases "${MAX_CASES:-0}" --resume \
    2>&1 | tee -a "${LOG_ROOT}/baselines.log"
fi

if [[ "${RUN_OURS}${RUN_BASELINES}" == "00" ]]; then
  printf 'Prepared only. Set RUN_OURS=1 or RUN_BASELINES=1.\n'
fi
