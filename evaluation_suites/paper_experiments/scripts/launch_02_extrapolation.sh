#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-extrapolation_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT_DIR="${ROOT_DIR:-${PROJECT_ROOT}}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/02_extrapolation/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/paper_experiments/02_extrapolation/${RUN_ID}}"
LAUNCH_VL_LOOPSR="${LAUNCH_VL_LOOPSR:-${LAUNCH_OURS_V11:-0}}"
EXTRAP_VL_LOOPSR_CASE_BUDGET_SEC="${EXTRAP_VL_LOOPSR_CASE_BUDGET_SEC:-${EXTRAP_V11_CASE_BUDGET_SEC:-600}}"
EXTRAP_VL_LOOPSR_PARENT_TIMEOUT_SEC="${EXTRAP_VL_LOOPSR_PARENT_TIMEOUT_SEC:-${EXTRAP_V11_PARENT_TIMEOUT_SEC:-630}}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"
cd "${ROOT_DIR}"

cat > "${RESULTS_ROOT}/README.txt" <<EOF
OOD extrapolation suite.

This suite corresponds to Fig. 3 in the paper.

Benchmarks:
- Constructed62: 62 analytic ID/OOD extrapolation tasks.
- SurfaceBench40: public 2D surface tasks from pandoradox/symbolic-regression-surfaces.

Primary readouts: ID/OOD log-MSE, ID-to-OOD shift, and OOD/ID degradation.

Controls:
  LAUNCH_VL_LOOPSR=1
  LAUNCH_CONSTRUCTED62=1
  LAUNCH_SURFACEBENCH40=1
EOF

echo "[02-extrapolation] results=${RESULTS_ROOT}"
echo "[02-extrapolation] logs=${LOG_ROOT}"

if [[ "${LAUNCH_VL_LOOPSR:-0}" == "1" ]]; then
  if [[ "${LAUNCH_CONSTRUCTED62:-1}" == "1" ]]; then
    echo "[02-extrapolation] launching Constructed62 VL-LoopSR"
    python3 scripts/run_v11_extrapolation_suites.py \
      --suite constructed \
      --out-dir "${RESULTS_ROOT}/constructed62/vl_loopsr" \
      --case-budget-sec "${EXTRAP_VL_LOOPSR_CASE_BUDGET_SEC}" \
      --parent-timeout-sec "${EXTRAP_VL_LOOPSR_PARENT_TIMEOUT_SEC}" \
      --timeout-grace-sec "${EXTRAP_TIMEOUT_GRACE_SEC:-30}" \
      --n-train "${EXTRAP_CONSTRUCTED_N_TRAIN:-256}" \
      --n-val "${EXTRAP_CONSTRUCTED_N_VAL:-128}" \
      --n-test "${EXTRAP_CONSTRUCTED_N_TEST:-512}" \
      --resume \
      2>&1 | tee -a "${LOG_ROOT}/constructed62_vl_loopsr.log"
  fi

  if [[ "${LAUNCH_SURFACEBENCH40:-1}" == "1" ]]; then
    echo "[02-extrapolation] launching SurfaceBench40 VL-LoopSR"
    python3 scripts/run_v11_extrapolation_suites.py \
      --suite surfacebench \
      --out-dir "${RESULTS_ROOT}/surfacebench40/vl_loopsr" \
      --surfacebench-path "${SURFACEBENCH_PATH:-data/surfacebench_public/dataset.h5}" \
      --max-cases "${SURFACEBENCH_MAX_CASES:-40}" \
      --case-budget-sec "${EXTRAP_VL_LOOPSR_CASE_BUDGET_SEC}" \
      --parent-timeout-sec "${EXTRAP_VL_LOOPSR_PARENT_TIMEOUT_SEC}" \
      --timeout-grace-sec "${EXTRAP_TIMEOUT_GRACE_SEC:-30}" \
      --n-train "${EXTRAP_SURFACE_N_TRAIN:-1000}" \
      --n-val "${EXTRAP_SURFACE_N_VAL:-500}" \
      --n-test "${EXTRAP_SURFACE_N_TEST:-500}" \
      --resume \
      2>&1 | tee -a "${LOG_ROOT}/surfacebench40_vl_loopsr.log"
  fi
else
  echo "[02-extrapolation] scaffold only; set LAUNCH_VL_LOOPSR=1 to run VL-LoopSR."
fi
