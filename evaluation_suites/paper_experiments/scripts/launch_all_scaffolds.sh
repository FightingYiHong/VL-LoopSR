#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-paper_experiments_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for launcher in \
  launch_01_standard_recovery.sh \
  launch_02_extrapolation.sh \
  launch_03_high_dimensional_interference.sh \
  launch_04_noise_robustness.sh \
  launch_05_multimodal_comparison.sh \
  launch_06_agentic_evaluation.sh; do
  bash "${SCRIPT_DIR}/${launcher}" "${RUN_ID}"
done

printf '[paper-experiments] scaffolded executable suites for run_id=%s\n' "${RUN_ID}"
printf '[paper-experiments] Fig. 6 coverage needs stored candidate tables; run command 07 separately.\n'
