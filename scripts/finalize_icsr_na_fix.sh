#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd "${script_dir}/.." && pwd)}"
shards_root="${project_root}/runs/paper_experiments/01_standard_recovery/pareto_na_fix_180s_full_shards"
merged_root="${project_root}/runs/paper_experiments/01_standard_recovery/pareto_na_fix_180s_full_merged/icsr"
figure_root="${project_root}/figs/nature_subjournal/main_unified_evaluation"
expected_cases=655

while true; do
    observed_cases="$(
        find "${shards_root}" -path '*/case_results/*.json' -type f | wc -l
    )"
    if [[ "${observed_cases}" -ge "${expected_cases}" ]]; then
        break
    fi
    sleep 60
done

cd "${project_root}"
python scripts/merge_official_icsr_shards.py \
    --shards-root "${shards_root}" \
    --output-root "${merged_root}" \
    --benchmarks "srbench,srsd"

python scripts/plot_unified_evaluation.py \
    --output-dir "${figure_root}" \
    --output-stem "fig_main_unified_evaluation"

if rg -q '>NA<' "${figure_root}/fig_main_unified_evaluation.svg"; then
    printf 'Visible NA remains in the final SVG.\n' >&2
    exit 1
fi

FIGURE_ROOT="${figure_root}" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

source = Path(os.environ["FIGURE_ROOT"]) / "fig_main_unified_evaluation_source_data.csv"
frame = pd.read_csv(source, low_memory=False)
summary = frame[
    frame["record_type"].eq("summary") & frame["panel"].isin(["a", "b", "c", "d"])
]
if summary["value"].isna().any():
    missing = summary.loc[
        summary["value"].isna(), ["panel", "benchmark", "method", "metric"]
    ]
    raise SystemExit(f"Missing primary summary values remain:\n{missing}")
print("ICSR NA repair finalized: all primary summary values are finite.")
PY
