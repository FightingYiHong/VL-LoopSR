#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_ROOT="${FEYNMAN_HIGHDIM_RUN_ROOT:-${ROOT_DIR}/runs/paper_experiments/01_standard_recovery/feynman_highdim_confirmatory_20260802}"
LOG_ROOT="${FEYNMAN_HIGHDIM_GAP_LOG_ROOT:-${ROOT_DIR}/logs/paper_experiments/01_standard_recovery/feynman_highdim_gaps_20260802}"

mkdir -p "${RUN_ROOT}/reuse" "${RUN_ROOT}/summary" "${LOG_ROOT}"
cd "${ROOT_DIR}"

python scripts/build_feynman_highdim_reuse_inventory.py \
  --run-root "${RUN_ROOT}" \
  --output-dir "${RUN_ROOT}/reuse" \
  > "${LOG_ROOT}/final_reuse_inventory.log" 2>&1

conda run --no-capture-output -n sr \
  python scripts/summarize_feynman_highdim_confirmatory.py \
    --run-root "${RUN_ROOT}" \
    --output-dir "${RUN_ROOT}/summary" \
  > "${LOG_ROOT}/summary.log" 2>&1

printf 'finalized_at\t%s\n' "$(date -Is)" > "${LOG_ROOT}/finalizer_status.tsv"
