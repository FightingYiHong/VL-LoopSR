#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-highdim_interference_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/03_high_dimensional_interference/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/03_high_dimensional_interference/${RUN_ID}}"
LAUNCH_VL_LOOPSR="${LAUNCH_VL_LOOPSR:-${LAUNCH_OURS_V11:-0}}"
HIGHDIM_VL_LOOPSR_CASE_BUDGET_SEC="${HIGHDIM_VL_LOOPSR_CASE_BUDGET_SEC:-${HIGHDIM_V11_CASE_BUDGET_SEC:-900}}"
HIGHDIM_VL_LOOPSR_PARENT_TIMEOUT_SEC="${HIGHDIM_VL_LOOPSR_PARENT_TIMEOUT_SEC:-${HIGHDIM_V11_PARENT_TIMEOUT_SEC:-930}}"
HIGHDIM_BASELINE_CHILD_PYTHON="${HIGHDIM_BASELINE_CHILD_PYTHON:-${HOME}/anaconda3/envs/srbench-gplearn/bin/python}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${ROOT_DIR}"

cat > "${RESULTS_ROOT}/README.txt" <<EOF
High-dimensional distractor suite.

This suite corresponds to Fig. 4 in the paper.

Cases: 24 formulas x d=200/500/1000 x three distractor regimes = 216 tasks.
Each formula uses k=2/3/4/5 true variables; remaining variables are independent
irrelevant variables, correlated proxies, or nonlinear decoys.

Metrics: exact support recovery, true-variable recall/precision, false
discovery rate, irrelevant-variable false-positive rate, proxy/nonlinear
misuse, strict/SRBench recovery, MSE/NRMSE, complexity, runtime, timeout rate.

Launch controls:
  LAUNCH_VL_LOOPSR=1
  LAUNCH_BASELINES=1
  HIGHDIM_BASELINE_METHODS=pyoperon,gplearn
  LAUNCH_GPU_BASELINES=1
  HIGHDIM_GPU_BASELINE_METHODS=physo
  LAUNCH_LLM_BASELINES=1
  LLMSR_BASELINE_API_BASE=http://127.0.0.1:8001/v1
EOF

echo "[03-highdim] results=${RESULTS_ROOT}"
echo "[03-highdim] logs=${LOG_ROOT}"

if [[ "${LAUNCH_VL_LOOPSR:-0}" == "1" ]]; then
  echo "[03-highdim] launching VL-LoopSR"
  python3 scripts/run_v11_high_dimensional_interference.py \
    --out-dir "${RESULTS_ROOT}/vl_loopsr" \
    --case-budget-sec "${HIGHDIM_VL_LOOPSR_CASE_BUDGET_SEC}" \
    --parent-timeout-sec "${HIGHDIM_VL_LOOPSR_PARENT_TIMEOUT_SEC}" \
    --timeout-grace-sec "${HIGHDIM_TIMEOUT_GRACE_SEC:-30}" \
    --resume \
    2>&1 | tee -a "${LOG_ROOT}/vl_loopsr.log"
fi

if [[ "${LAUNCH_BASELINES:-0}" == "1" ]]; then
  echo "[03-highdim] launching baselines=${HIGHDIM_BASELINE_METHODS:-pyoperon,gplearn}"
  python3 scripts/run_highdim_interference_baselines.py \
    --methods "${HIGHDIM_BASELINE_METHODS:-pyoperon,gplearn}" \
    --child-python "${HIGHDIM_BASELINE_CHILD_PYTHON}" \
    --out-dir "${RESULTS_ROOT}/baselines" \
    --case-timeout-sec "${HIGHDIM_BASELINE_CASE_TIMEOUT_SEC:-900}" \
    --parent-timeout-sec "${HIGHDIM_BASELINE_PARENT_TIMEOUT_SEC:-930}" \
    --timeout-grace-sec "${HIGHDIM_TIMEOUT_GRACE_SEC:-30}" \
    --resume \
    2>&1 | tee -a "${LOG_ROOT}/baselines.log"
fi

if [[ "${LAUNCH_GPU_BASELINES:-0}" == "1" ]]; then
  echo "[03-highdim] launching gpu baselines=${HIGHDIM_GPU_BASELINE_METHODS:-physo}"
  python3 scripts/run_highdim_interference_baselines.py \
    --methods "${HIGHDIM_GPU_BASELINE_METHODS:-physo}" \
    --out-dir "${RESULTS_ROOT}/gpu_baselines" \
    --case-timeout-sec "${HIGHDIM_GPU_BASELINE_CASE_TIMEOUT_SEC:-900}" \
    --parent-timeout-sec "${HIGHDIM_GPU_BASELINE_PARENT_TIMEOUT_SEC:-930}" \
    --timeout-grace-sec "${HIGHDIM_TIMEOUT_GRACE_SEC:-30}" \
    --resume \
    2>&1 | tee -a "${LOG_ROOT}/gpu_baselines.log"
fi

if [[ "${LAUNCH_LLM_BASELINES:-0}" == "1" ]]; then
  echo "[03-highdim] launching llm baseline=${LLMSR_BASELINE_METHOD_NAME:-llm_sr_baseline}"
  python3 scripts/run_highdim_llm_baselines.py \
    --out-dir "${RESULTS_ROOT}/llm_baselines" \
    --api-base "${LLMSR_BASELINE_API_BASE:-http://127.0.0.1:8001/v1}" \
    --api-key "${LLMSR_BASELINE_API_KEY:-EMPTY}" \
    --model "${LLMSR_BASELINE_MODEL:-}" \
    --method-name "${LLMSR_BASELINE_METHOD_NAME:-llm_sr_baseline}" \
    --case-timeout-sec "${HIGHDIM_LLM_CASE_TIMEOUT_SEC:-300}" \
    --parent-timeout-sec "${HIGHDIM_LLM_PARENT_TIMEOUT_SEC:-330}" \
    --timeout-grace-sec "${HIGHDIM_TIMEOUT_GRACE_SEC:-30}" \
    --request-timeout-sec "${HIGHDIM_LLM_REQUEST_TIMEOUT_SEC:-80}" \
    --max-attempts "${HIGHDIM_LLM_MAX_ATTEMPTS:-4}" \
    --sample-rows "${HIGHDIM_LLM_SAMPLE_ROWS:-6}" \
    --resume \
    2>&1 | tee -a "${LOG_ROOT}/llm_baselines.log"
fi
