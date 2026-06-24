#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-official_llm_sr_equal_v11_600s_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
OFFICIAL_ROOT="${OFFICIAL_LLMSR_ROOT:-${ROOT_DIR}/external_repos/LLM-SR}"
RESULTS_ROOT="${OFFICIAL_LLMSR_RESULTS_ROOT:-${ROOT_DIR}/runs/official_llmsr/${RUN_ID}}"
LOG_DIR="${OFFICIAL_LLMSR_LOG_ROOT:-${ROOT_DIR}/logs/official_llmsr/${RUN_ID}}"

mkdir -p "${RESULTS_ROOT}" "${LOG_DIR}"
cd "${ROOT_DIR}"

export OFFICIAL_LLMSR_ROOT="${OFFICIAL_ROOT}"
export OFFICIAL_LLMSR_COMMIT="${OFFICIAL_LLMSR_COMMIT:-$(git -C "${OFFICIAL_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)}"
export OFFICIAL_LLMSR_API_BASE="${OFFICIAL_LLMSR_API_BASE:-http://127.0.0.1:8001/v1}"
export OFFICIAL_LLMSR_API_KEY="${OFFICIAL_LLMSR_API_KEY:-EMPTY}"
export OFFICIAL_LLMSR_MODEL="${OFFICIAL_LLMSR_MODEL:-llm-baseline-qwen2.5-32b}"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"

export LLMSRBENCH_ROOT="${LLMSRBENCH_ROOT:-${ROOT_DIR}/data/llmsrbench}"
export LLMSRBENCH_HDF5="${LLMSRBENCH_HDF5:-${LLMSRBENCH_ROOT}/lsr_bench_data.hdf5}"
export SRSD_ROOT="${SRSD_ROOT:-${ROOT_DIR}/srsd-benchmark/resource/datasets/srsd}"
export SRBENCH_ROOT="${SRBENCH_ROOT:-${ROOT_DIR}/srbench}"
export SRBENCH_DATASETS_INFO_CSV="${SRBENCH_DATASETS_INFO_CSV:-${SRBENCH_ROOT}/docs/csv/datasets_info.csv}"
export SRBENCH_PMLB_CACHE_DIR="${SRBENCH_PMLB_CACHE_DIR:-${ROOT_DIR}/.cache/pmlb_cache}"
export SRBENCH_LOCAL_CSV_ROOT="${SRBENCH_LOCAL_CSV_ROOT:-${ROOT_DIR}/data/pmlb_regression_csv}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

echo "[OFFICIAL-LLMSR] run_id=${RUN_ID}"
echo "[OFFICIAL-LLMSR] official_root=${OFFICIAL_ROOT}"
echo "[OFFICIAL-LLMSR] official_commit=${OFFICIAL_LLMSR_COMMIT}"
echo "[OFFICIAL-LLMSR] results=${RESULTS_ROOT}"
echo "[OFFICIAL-LLMSR] api_base=${OFFICIAL_LLMSR_API_BASE}"
echo "[OFFICIAL-LLMSR] case_timeout_sec=${CASE_TIMEOUT_SEC:-600}"

nohup conda run --no-capture-output -n "${OFFICIAL_LLMSR_CONDA_ENV:-base}" \
  python evaluation_suites/official_llmsr_fourbench/run_official_llmsr_fourbench.py \
    --benchmarks "${OFFICIAL_LLMSR_BENCHMARKS:-sldbench,llmsrbench,srsd,srbench}" \
    --results-root "${RESULTS_ROOT}" \
    --case-timeout-sec "${CASE_TIMEOUT_SEC:-600}" \
    --samples-per-prompt "${OFFICIAL_LLMSR_SAMPLES_PER_PROMPT:-4}" \
    --evaluate-timeout-sec "${OFFICIAL_LLMSR_EVALUATE_TIMEOUT_SEC:-30}" \
    --request-timeout-sec "${OFFICIAL_LLMSR_REQUEST_TIMEOUT_SEC:-180}" \
    --max-sample-nums "${OFFICIAL_LLMSR_MAX_SAMPLE_NUMS:-100000}" \
    --isolate-cases \
    --resume \
  > "${LOG_DIR}/run.log" 2>&1 &

echo "$!" > "${LOG_DIR}/pid"
cat > "${LOG_DIR}/launcher.log" <<EOF
run_id=${RUN_ID}
pid=$(cat "${LOG_DIR}/pid")
root_dir=${ROOT_DIR}
official_root=${OFFICIAL_ROOT}
official_commit=${OFFICIAL_LLMSR_COMMIT}
results_root=${RESULTS_ROOT}
benchmarks=${OFFICIAL_LLMSR_BENCHMARKS:-sldbench,llmsrbench,srsd,srbench}
case_timeout_sec=${CASE_TIMEOUT_SEC:-600}
started_at=$(date -Is)
EOF

echo "[OFFICIAL-LLMSR] started pid=$(cat "${LOG_DIR}/pid")"
echo "[OFFICIAL-LLMSR] log=${LOG_DIR}/run.log"
