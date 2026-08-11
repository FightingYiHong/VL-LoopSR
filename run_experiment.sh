#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

usage() {
  cat <<'EOF'
Usage:
  bash run_experiment.sh list
  bash run_experiment.sh prepare <standard|noise|interference|sft|all>
  bash run_experiment.sh train-sft
  bash run_experiment.sh <01|02|03|04|05|06|07|all> [run_id]
  bash run_experiment.sh <supp-sft|supp-ablation> [run_id]

Expensive runs are disabled by default. Enable the corresponding RUN_* flag;
for example: RUN_OURS=1 MAX_CASES=2 bash run_experiment.sh 01 smoke
EOF
}

list_experiments() {
  cat <<'EOF'
01  Fig. 2     Standard recovery (895 tasks)
02  Fig. 3a    Constructed62 OOD extrapolation (62 tasks)
03  Fig. 3b-c  High-dimensional distractors (216 tasks)
04  Fig. 3d    NoiseSR-20 robustness (180 tasks)
05  Fig. 4     Matched multimodal input comparison (96 paired tasks)
06  Fig. 5     Matched agentic-evaluation comparison (96 paired tasks)
07  Fig. 6     Post-hoc candidate coverage (895 tasks/method)

supp-sft       Descriptive archived Proposer SFT comparison (895 pairs)
supp-ablation  Full component ablation (96 matched tasks)
train-sft      Train the Proposer LoRA adapter on the bundled 10k corpus
EOF
}

command_name="${1:-list}"
run_id="${2:-paper_${command_name}_$(date +%Y%m%d_%H%M%S)}"
case "${command_name}" in
  list) list_experiments ;;
  prepare) python3 scripts/prepare_data.py "${2:-all}" ;;
  train-sft) bash evaluation_suites/paper_experiments/scripts/launch_04_train_proposer_sft.sh ;;
  01) bash evaluation_suites/paper_experiments/scripts/launch_01_standard_recovery.sh "${run_id}" ;;
  02) bash evaluation_suites/paper_experiments/scripts/launch_02_extrapolation.sh "${run_id}" ;;
  03) bash evaluation_suites/paper_experiments/scripts/launch_03_high_dimensional_interference.sh "${run_id}" ;;
  04) bash evaluation_suites/paper_experiments/scripts/launch_04_noise_robustness.sh "${run_id}" ;;
  05) bash evaluation_suites/paper_experiments/scripts/launch_05_multimodal_comparison.sh "${run_id}" ;;
  06) bash evaluation_suites/paper_experiments/scripts/launch_06_agentic_evaluation.sh "${run_id}" ;;
  07) bash evaluation_suites/paper_experiments/scripts/launch_07_candidate_coverage.sh "${run_id}" ;;
  supp-sft) bash evaluation_suites/paper_experiments/scripts/launch_supp_proposer_sft.sh "${run_id}" ;;
  supp-ablation) bash evaluation_suites/paper_experiments/scripts/launch_supp_component_ablation.sh "${run_id}" ;;
  all) bash evaluation_suites/paper_experiments/scripts/launch_all_scaffolds.sh "${run_id}" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
