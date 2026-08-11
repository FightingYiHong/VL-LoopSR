#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-extrapolation_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/02_extrapolation/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/02_extrapolation/${RUN_ID}}"
RUN_OURS="${RUN_OURS:-0}"
RUN_BASELINES="${RUN_BASELINES:-0}"
MAX_CASES="${MAX_CASES:-62}"
[[ "${MAX_CASES}" -gt 0 ]] || MAX_CASES=62

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

printf '[Fig. 3a] Constructed62 OOD extrapolation: 43 one-dimensional + 19 two-dimensional tasks\n'
printf 'results=%s\n' "${RESULTS_ROOT}"

if [[ "${RUN_OURS}" == "1" ]]; then
  python3 scripts/run_v11_extrapolation_suites.py \
    --suite constructed \
    --out-dir "${RESULTS_ROOT}/vega_sr" \
    --case-budget-sec "${OOD_VEGA_SR_CASE_BUDGET_SEC:-600}" \
    --parent-timeout-sec "${OOD_VEGA_SR_PARENT_TIMEOUT_SEC:-630}" \
    --timeout-grace-sec "${OOD_TIMEOUT_GRACE_SEC:-30}" \
    --max-cases "${MAX_CASES}" \
    --n-train 256 --n-val 128 --n-test 512 --resume \
    2>&1 | tee -a "${LOG_ROOT}/vega_sr.log"
fi

if [[ "${RUN_BASELINES}" == "1" ]]; then
  methods="${OOD_BASELINE_METHODS:-pysr,pyoperon,dso,gplearn,ffx,deap,bingo,rils_rols}"
  IFS=',' read -r -a method_list <<<"${methods}"
  for method in "${method_list[@]}"; do
    python3 scripts/run_strict_extrapolation_cpu_baselines.py \
      --method "${method}" \
      --out-dir "${RESULTS_ROOT}/${method}" \
      --case-budget-sec "${OOD_BASELINE_CASE_BUDGET_SEC:-100}" \
      --max-cases "${MAX_CASES}" --resume \
      2>&1 | tee -a "${LOG_ROOT}/${method}.log"
  done
fi

if [[ "${RUN_OURS}${RUN_BASELINES}" == "00" ]]; then
  printf 'Prepared only. Set RUN_OURS=1 or RUN_BASELINES=1.\n'
fi
