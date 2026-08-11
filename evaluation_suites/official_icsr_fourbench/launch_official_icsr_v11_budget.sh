#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-official_icsr_equal_v11_600s_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OFFICIAL_ICSR_ROOT="${OFFICIAL_ICSR_ROOT:-${PROJECT_ROOT}/data/external/methods/In-Context-Symbolic-Regression}"
RESULTS_ROOT="${OFFICIAL_ICSR_RESULTS_ROOT:-${PROJECT_ROOT}/runs/official_icsr/${RUN_ID}}"
LOG_ROOT="${OFFICIAL_ICSR_LOG_ROOT:-${PROJECT_ROOT}/logs/official_icsr/${RUN_ID}}"
BENCHMARKS="${OFFICIAL_ICSR_BENCHMARKS:-llmsrbench,srbench,srsd}"
CASE_TIMEOUT_SEC="${OFFICIAL_ICSR_CASE_TIMEOUT_SEC:-600}"
ITERATIONS="${OFFICIAL_ICSR_ITERATIONS:-500}"
LLMSRBENCH_ROOT="${LLMSRBENCH_ROOT:-${PROJECT_ROOT}/data/external/llmsrbench}"
LLMSRBENCH_HDF5="${LLMSRBENCH_HDF5:-${LLMSRBENCH_ROOT}/lsr_bench_data.hdf5}"

mkdir -p "${RESULTS_ROOT}" "${LOG_ROOT}"

cd "${PROJECT_ROOT}"

PYTHON_CMD=("${OFFICIAL_ICSR_PYTHON:-conda}" )
if [[ "${PYTHON_CMD[0]}" == "conda" ]]; then
  PYTHON_CMD=(conda run --no-capture-output -n "${OFFICIAL_ICSR_CONDA_ENV:-base}" python)
fi

nohup env \
  OFFICIAL_ICSR_ROOT="${OFFICIAL_ICSR_ROOT}" \
  OFFICIAL_ICSR_MODEL_NAME="${OFFICIAL_ICSR_MODEL_NAME:-llm-baseline-qwen2.5-32b}" \
  OFFICIAL_ICSR_OPENAI_BASE_URL="${OFFICIAL_ICSR_OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}" \
  OFFICIAL_ICSR_OPENAI_API_KEY="${OFFICIAL_ICSR_OPENAI_API_KEY:-EMPTY}" \
  LLMSRBENCH_ROOT="${LLMSRBENCH_ROOT}" \
  LLMSRBENCH_HDF5="${LLMSRBENCH_HDF5}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PROJECT_ROOT}/.cache/huggingface/datasets}" \
  HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}" \
  TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
  NO_PROXY="127.0.0.1,localhost" \
  no_proxy="127.0.0.1,localhost" \
  "${PYTHON_CMD[@]}" evaluation_suites/official_icsr_fourbench/run_official_icsr_fourbench.py \
      --benchmarks "${BENCHMARKS}" \
      --results-root "${RESULTS_ROOT}" \
      --case-timeout-sec "${CASE_TIMEOUT_SEC}" \
      --iterations "${ITERATIONS}" \
      --resume \
  > "${LOG_ROOT}/run.log" 2>&1 &

echo "$!" > "${LOG_ROOT}/pid"
cat > "${LOG_ROOT}/launcher.log" <<EOF
run_id=${RUN_ID}
pid=$(cat "${LOG_ROOT}/pid")
project_root=${PROJECT_ROOT}
official_icsr_root=${OFFICIAL_ICSR_ROOT}
results_root=${RESULTS_ROOT}
benchmarks=${BENCHMARKS}
case_timeout_sec=${CASE_TIMEOUT_SEC}
iterations=${ITERATIONS}
llmsrbench_root=${LLMSRBENCH_ROOT}
llmsrbench_hdf5=${LLMSRBENCH_HDF5}
started_at=$(date -Is)
EOF

echo "[official-icsr] started pid=$(cat "${LOG_ROOT}/pid")"
echo "[official-icsr] results=${RESULTS_ROOT}"
echo "[official-icsr] log=${LOG_ROOT}/run.log"
