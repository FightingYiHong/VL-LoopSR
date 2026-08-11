#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-candidate_coverage_$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/runs/paper_experiments/07_candidate_coverage/${RUN_ID}}"
CANDIDATES_CSV="${CANDIDATE_COVERAGE_CANDIDATES_CSV:-}"
FINALS_CSV="${CANDIDATE_COVERAGE_FINALS_CSV:-}"

mkdir -p "${RESULTS_ROOT}"
cd "${PROJECT_ROOT}"

if [[ -z "${CANDIDATES_CSV}" || -z "${FINALS_CSV}" ]]; then
  printf '[Fig. 6] candidate coverage requires tidy post-hoc test-score tables.\n'
  printf 'Set CANDIDATE_COVERAGE_CANDIDATES_CSV and CANDIDATE_COVERAGE_FINALS_CSV.\n'
  printf 'Run `python3 scripts/summarize_candidate_coverage.py --help` for the schemas.\n'
  exit 2
fi

python3 scripts/summarize_candidate_coverage.py \
  --candidates-csv "${CANDIDATES_CSV}" --finals-csv "${FINALS_CSV}" \
  --output-dir "${RESULTS_ROOT}" --expected-tasks 895 --max-budget 100
