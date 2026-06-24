#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-paper_experiments_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/launch_01_standard_recovery.sh" "${RUN_ID}"
bash "${SCRIPT_DIR}/launch_02_extrapolation.sh" "${RUN_ID}"
bash "${SCRIPT_DIR}/launch_03_high_dimensional_interference.sh" "${RUN_ID}"
bash "${SCRIPT_DIR}/launch_04_proposer_sft.sh" "${RUN_ID}"
bash "${SCRIPT_DIR}/launch_05_noise_robustness.sh" "${RUN_ID}"
bash "${SCRIPT_DIR}/launch_06_component_ablation.sh" "${RUN_ID}"

echo "[paper-experiments] scaffolded all suites for run_id=${RUN_ID}"
