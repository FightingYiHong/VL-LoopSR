#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ID="${1:-feynman_highdim_confirmatory_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${FEYNMAN_HIGHDIM_RUN_ROOT:-${ROOT_DIR}/runs/paper_experiments/01_standard_recovery/${RUN_ID}}"
LOG_ROOT="${FEYNMAN_HIGHDIM_LOG_ROOT:-${ROOT_DIR}/logs/paper_experiments/01_standard_recovery/${RUN_ID}}"
BUDGET_SEC="${FEYNMAN_HIGHDIM_BUDGET_SEC:-600}"
REPEAT_SEEDS="${FEYNMAN_HIGHDIM_REPEAT_SEEDS:-611}"
MODEL="${FEYNMAN_HIGHDIM_MODEL:-Qwen3-VL-32B-Instruct}"
API_BASE="${FEYNMAN_HIGHDIM_API_BASE:-http://127.0.0.1:8001/v1}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/scripts:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CPU_SR_SEARCH_EFFICIENCY_THRESHOLD=0.999
export CPU_SR_CONFIGURED_WALL_BUDGET_SEC="${BUDGET_SEC}"
export CPU_SR_CASE_SUBPROCESS_TIMEOUT_SEC="$((BUDGET_SEC + 120))"
export CPU_SR_CASE_SUBPROCESS_GRACE_SEC=120
export SRBENCH_ROOT="${SRBENCH_ROOT:-${ROOT_DIR}/srbench}"
export SRBENCH_DATASETS_INFO_CSV="${SRBENCH_DATASETS_INFO_CSV:-${SRBENCH_ROOT}/docs/csv/datasets_info.csv}"
export SRBENCH_PMLB_CACHE_DIR="${SRBENCH_PMLB_CACHE_DIR:-${ROOT_DIR}/.cache/pmlb_cache}"
export SRBENCH_LOCAL_CSV_ROOT="${SRBENCH_LOCAL_CSV_ROOT:-${ROOT_DIR}/data/pmlb_regression_csv}"
export SRBENCH_ALLOW_GROUPS=feynman
# SRBench metadata counts the target column, so 6 means at least 5 observed inputs.
export SRBENCH_MIN_FEATURES=6
export SRBENCH_MAX_FEATURES=none
export SRBENCH_MAX_FILES=none
export SRBENCH_HARD_TIMEOUT_SEC="$((BUDGET_SEC + 360))"

status_file="${RUN_ROOT}/status.tsv"
printf 'worker\tstatus\tstarted_at\tfinished_at\tlog\n' > "${status_file}"

cat > "${RUN_ROOT}/run_manifest.tsv" <<EOF
run_id	${RUN_ID}
study	frozen_highdimensional_feynman_confirmatory
selection_rule	SRBench feynman group with at least five observed input variables
runner_filter	SRBENCH_ALLOW_GROUPS=feynman; SRBENCH_MIN_FEATURES=6 because metadata includes target
expected_unique_cases	34
methods	Ours,LLM-SR,DSO,gplearn,PSE,PySR,ICSR
repeat_seeds	${REPEAT_SEEDS}
repeats_per_case	$(wc -w <<< "${REPEAT_SEEDS}")
case_budget_sec	${BUDGET_SEC}
validation_hit_rule	R2 strictly greater than 0.999
confirmation_rule	validation hit and held-out test R2 strictly greater than 0.999
test_selection	disabled
model	${MODEL}
api_base	${API_BASE}
started_at	$(date -Is)
EOF

run_logged_worker() {
  local worker="$1"
  shift
  local log_file="${LOG_ROOT}/${worker}.log"
  local started_at
  local finished_at
  local status
  started_at="$(date -Is)"
  set +e
  "$@" > "${log_file}" 2>&1
  status="$?"
  set -e
  finished_at="$(date -Is)"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${worker}" "${status}" "${started_at}" "${finished_at}" "${log_file}" >> "${status_file}"
  return "${status}"
}

run_cpu_worker() {
  local method="$1"
  local conda_env="$2"
  local config="$3"
  local seed
  for seed in ${REPEAT_SEEDS}; do
    local out="${RUN_ROOT}/${method}/seed_${seed}/srbench"
    mkdir -p "${out}"
    conda run --no-capture-output -n "${conda_env}" \
      python scripts/run_cpu_baseline_benchmarks.py \
        --benchmark srbench \
        --method "${method}" \
        --config "${config}" \
        --random-state "${seed}" \
        --results-root "${out}" \
        --isolate-cases || return $?
  done
}

configure_ours_env() {
  export LLMSR_V10_PATH="${ROOT_DIR}/scripts/test_fey_v11_complexity_exact.py"
  export SRBENCH_V10_PATH="${LLMSR_V10_PATH}"
  export LLMSR_AGENT_API_BASE="${API_BASE}"
  export LLMSR_AGENT_API_KEY=EMPTY
  export LLMSR_AGENT_MODEL="${MODEL}"
  export LLMSR_PLANNER_API_BASE="${API_BASE}"
  export LLMSR_PLANNER_API_KEY=EMPTY
  export LLMSR_PLANNER_MODEL="${MODEL}"
  export LLMSR_EVAL_PROFILE=quality
  export LLMSR_METHOD_MODE=planner_guided
  export LLMSR_LIGHT_MODE=0
  export LLMSR_MAX_RUNTIME_PER_TASK_SEC="${BUDGET_SEC}"
  export LLMSR_V11_DISABLE_RUNTIME_FAST_PATH=1
  export LLMSR_V11_BUDGET_AWARE_MODE=on
  export LLMSR_V11_FULL_BUDGET=1
  export LLMSR_V11_LOW_DIM_FULL_BUDGET=1
  export LLMSR_V11_FULL_BUDGET_TEXT_CALLS=4
  export LLMSR_V11_FULL_BUDGET_MM_CALLS=3
  export LLMSR_V11_FULL_BUDGET_PROPOSAL_K=16
  export LLMSR_V11_FULL_BUDGET_REFINED_K=8
  export LLMSR_V11_FULL_BUDGET_REFINE_ROUNDS=4
  export LLMSR_V11_ENABLE_STRUCTURE_EVALUATOR=1
  export LLMSR_V11_ENABLE_GENERALIZATION_GUARD=1
  export LLMSR_ALLOW_TRUE_EXPR_DIAGNOSTICS=0
  export LLMSR_USE_TEST_FOR_SELECTION=0
}

run_ours_worker() {
  configure_ours_env
  local seed
  for seed in ${REPEAT_SEEDS}; do
    local out="${RUN_ROOT}/ours/seed_${seed}/srbench"
    mkdir -p "${out}"
    export LLMSR_REPEAT_SEED="${seed}"
    export SRBENCH_RANDOM_SEED="${seed}"
    SRBENCH_V10_RESULTS_ROOT="${out}" \
      conda run --no-capture-output -n sr python scripts/run_SRbench.py || return $?
  done
}

run_llmsr_worker() {
  local seed
  export OFFICIAL_LLMSR_ROOT="${OFFICIAL_LLMSR_ROOT:-${ROOT_DIR}/external_repos/LLM-SR}"
  export OFFICIAL_LLMSR_API_BASE="${API_BASE}"
  export OFFICIAL_LLMSR_API_KEY=EMPTY
  export OFFICIAL_LLMSR_MODEL="${MODEL}"
  for seed in ${REPEAT_SEEDS}; do
    local out="${RUN_ROOT}/llmsr/seed_${seed}"
    PYTHONHASHSEED="${seed}" conda run --no-capture-output -n llmsr \
      python evaluation_suites/official_llmsr_fourbench/run_official_llmsr_fourbench.py \
        --benchmarks srbench \
        --results-root "${out}" \
        --case-timeout-sec "${BUDGET_SEC}" \
        --request-timeout-sec 180 \
        --samples-per-prompt 4 \
        --isolate-cases || return $?
  done
}

run_icsr_worker() {
  local seed
  export OFFICIAL_ICSR_ROOT="${OFFICIAL_ICSR_ROOT:-${ROOT_DIR}/external_repos/In-Context-Symbolic-Regression}"
  export OFFICIAL_ICSR_MODEL_NAME="${MODEL}"
  export OFFICIAL_ICSR_OPENAI_BASE_URL="${API_BASE}"
  export OFFICIAL_ICSR_OPENAI_API_KEY=EMPTY
  for seed in ${REPEAT_SEEDS}; do
    local out="${RUN_ROOT}/icsr/seed_${seed}"
    PYTHONHASHSEED="${seed}" conda run --no-capture-output -n base \
      python evaluation_suites/official_icsr_fourbench/run_official_icsr_fourbench.py \
        --benchmarks srbench \
        --results-root "${out}" \
        --case-timeout-sec "${BUDGET_SEC}" \
        --startup-timeout-sec 600 \
        --iterations 500 \
        --optimizer-timeout-sec 10 \
        --optimizer-threads 5 \
        --max-new-tokens 2048 || return $?
  done
}

pids=()
workers=()
launch_worker() {
  local worker="$1"
  shift
  run_logged_worker "${worker}" "$@" &
  pids+=("$!")
  workers+=("${worker}")
}

launch_worker gplearn run_cpu_worker gplearn srbench-gplearn \
  evaluation_suites/cpu_symbolic_regression_fourbench/configs/gplearn_10min_1thread.yaml
launch_worker pysr run_cpu_worker pysr srbench-pysr \
  evaluation_suites/cpu_symbolic_regression_fourbench/configs/pysr_10min_1thread.yaml
launch_worker dso run_cpu_worker dso dso \
  evaluation_suites/cpu_symbolic_regression_fourbench/configs/dso_10min.yaml
CUDA_VISIBLE_DEVICES=0 launch_worker psrn run_cpu_worker psrn PSRN \
  evaluation_suites/gpu_symbolic_regression_fourbench/configs/psrn_10min_safe.yaml
launch_worker ours run_ours_worker
launch_worker llmsr run_llmsr_worker
launch_worker icsr run_icsr_worker

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    printf 'worker failed: %s\n' "${workers[$index]}" >&2
    failed=1
  fi
done

conda run --no-capture-output -n sr \
  python scripts/summarize_feynman_highdim_confirmatory.py \
    --run-root "${RUN_ROOT}" \
    --output-dir "${RUN_ROOT}/summary" \
  > "${LOG_ROOT}/summary.log" 2>&1 || failed=1

printf 'finished_at\t%s\n' "$(date -Is)" >> "${RUN_ROOT}/run_manifest.tsv"
exit "${failed}"
