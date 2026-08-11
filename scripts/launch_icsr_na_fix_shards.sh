#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd "${script_dir}/.." && pwd)}"
official_root="${OFFICIAL_ICSR_ROOT:-${project_root}/external_repos/In-Context-Symbolic-Regression}"
shards_root="${project_root}/runs/paper_experiments/01_standard_recovery/pareto_na_fix_180s_full_shards"
merged_root="${project_root}/runs/paper_experiments/01_standard_recovery/pareto_na_fix_180s_full_merged/icsr"
runner="${project_root}/evaluation_suites/official_icsr_fourbench/run_official_icsr_fourbench.py"
merger="${project_root}/scripts/merge_official_icsr_shards.py"

mkdir -p "${shards_root}"

pids=()
labels=()

launch_shard() {
    local label="$1"
    local benchmark="$2"
    local case_start="$3"
    local case_end="$4"
    local result_root="${shards_root}/${label}/icsr"
    mkdir -p "${result_root}"
    (
        cd "${project_root}"
        env \
            OFFICIAL_ICSR_ROOT="${official_root}" \
            OFFICIAL_ICSR_MODEL_NAME="Qwen3-VL-32B-Instruct" \
            python "${runner}" \
                --benchmarks "${benchmark}" \
                --results-root "${result_root}" \
                --case-timeout-sec 180 \
                --startup-timeout-sec 600 \
                --iterations 500 \
                --optimizer-timeout-sec 1 \
                --optimizer-threads 1 \
                --max-new-tokens 2048 \
                --top-p 0.9 \
                --top-k 60 \
                --temperature 1.0 \
                --max-retries 2 \
                --max-points-in-prompt 40 \
                --prompt-size 5 \
                --case-start "${case_start}" \
                --case-end "${case_end}" \
                --resume
    ) >"${result_root}/wrapper.log" 2>&1 &
    pids+=("$!")
    labels+=("${label}")
}

launch_shard "shard_sr01" "srbench" 1 105
launch_shard "shard_sr02" "srbench" 106 210
launch_shard "shard_sr03" "srbench" 211 314
launch_shard "shard_sr04" "srbench" 315 417
launch_shard "shard_srsd01" "srsd" 1 80
launch_shard "shard_srsd02" "srsd" 81 159
launch_shard "shard_srsd03" "srsd" 160 238

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        printf 'Shard failed: %s\n' "${labels[$index]}" >&2
        failed=1
    fi
done
if [[ "${failed}" -ne 0 ]]; then
    exit 1
fi

python "${merger}" \
    --shards-root "${shards_root}" \
    --output-root "${merged_root}" \
    --benchmarks "srbench,srsd"
