#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${PROJECT_ROOT}"

CORPUS_ROOT="${SFT_CORPUS_ROOT:-${PROJECT_ROOT}/data/proposer_sft}"
CONFIG="${SFT_TRAIN_CONFIG:-${PROJECT_ROOT}/evaluation_suites/paper_experiments/configs/proposer_sft_qwen3vl32b_qlora.yaml}"
CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
BASE_MODEL="${SFT_BASE_MODEL:-Qwen/Qwen3-VL-32B-Instruct}"
OUTPUT_DIR="${SFT_OUTPUT_DIR:-${PROJECT_ROOT}/saves/qwen3-vl-32b-proposer-sft-10000-qlora}"

if ! command -v "${CLI}" >/dev/null 2>&1; then
  printf 'LLaMA-Factory CLI not found: %s\n' "${CLI}" >&2
  printf 'Install LLaMA-Factory or set LLAMAFACTORY_CLI to its executable.\n' >&2
  exit 2
fi
python3 scripts/package_sft_corpus.py unpack "${CORPUS_ROOT}"

exec "${CLI}" train "${CONFIG}" \
  model_name_or_path="${BASE_MODEL}" \
  dataset_dir="${CORPUS_ROOT}" \
  output_dir="${OUTPUT_DIR}"
