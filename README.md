# VL-LoopSR Reproducibility Package

This repository contains the public, minimal code and data needed to reproduce
the VL-LoopSR symbolic-regression experiments. It is arranged around six
paper-aligned experiment launchers and uses repository-relative paths by
default, so it can be cloned and run on a new machine without editing source
files.

No API keys, private model paths, local usernames, generated logs, or local run
outputs are committed. Runtime secrets and machine-specific paths should be set
through environment variables or a local `.env` file.

## Quick Start

```bash
git clone <this-repository-url>
cd <repo>

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Recommended when pushing the bundled SFT images to GitHub.
git lfs install

cp .env.example .env
# Edit .env for your OpenAI-compatible endpoint, model name, and optional
# external benchmark paths.

bash evaluation_suites/paper_experiments/scripts/launch_all_scaffolds.sh
```

The scaffold command creates the six experiment result/log directories and
checks that the launch layer is wired correctly. It does not start expensive
runs unless the relevant `LAUNCH_*` environment variables are enabled.

## Repository Layout

- `evaluation_suites/paper_experiments/`: main public experiment entry point.
- `scripts/`: lower-level runners used by the six paper launchers.
- `llms/` and `tools/`: core VL-LoopSR runtime modules.
- `data/`: generated datasets and cached public benchmark assets included with
  this package, including the complete 10,000-example Proposer SFT data under
  `data/proposer_sft/raw_10k/`.
- `pmlb/`, `srsd-benchmark/`, `external_repos/deep-symbolic-optimization/`:
  trimmed public dependencies used by benchmark wrappers and baselines.

## Configuration

Copy `.env.example` to `.env` and edit local values. Important variables:

- `LLMSR_AGENT_API_BASE`: OpenAI-compatible endpoint, for example a local vLLM
  server.
- `LLMSR_AGENT_API_KEY`: API key for that endpoint. Use `EMPTY` for local vLLM
  servers that do not enforce authentication.
- `LLMSR_AGENT_MODEL`: model identifier served by the endpoint.
- `LLMSRBENCH_ROOT` and `LLMSRBENCH_HDF5`: optional local LLMSRBench download
  paths. LLMSRBench numeric arrays are not bundled here.
- `HF_HOME`: local Hugging Face cache directory for SLDBench downloads.

The code defaults to `runs/`, `logs/`, and `.cache/` under the repository root.
These directories are ignored by Git.

## Running Experiments

Main scaffold:

```bash
bash evaluation_suites/paper_experiments/scripts/launch_all_scaffolds.sh my_run
```

Run selected suites by enabling their flags:

```bash
source .env
LAUNCH_VL_LOOPSR=1 \
bash evaluation_suites/paper_experiments/scripts/launch_02_extrapolation.sh ood_run

LAUNCH_ABLATION=1 \
bash evaluation_suites/paper_experiments/scripts/launch_06_component_ablation.sh ablation_run
```

For the full six-suite workflow, see
`evaluation_suites/paper_experiments/README.md`.

## Reproducibility Notes

- Random seeds and per-case budgets are documented in
  `evaluation_suites/paper_experiments/configs/experiment_matrix.yaml`.
- Noise robustness and component-ablation manifests are included under `data/`.
- The complete Proposer SFT training package is included under
  `data/proposer_sft/raw_10k/`.
- SFT images are marked for Git LFS in `.gitattributes`; install Git LFS before
  adding and pushing them.
- SLDBench is loaded through Hugging Face `datasets`.
- LLMSRBench requires a local copy of its parquet metadata and HDF5 arrays.
- API credentials, model weights, private adapters, logs, generated reports, and
  run outputs should remain local and are covered by `.gitignore`.

## Public Release Checklist

Before pushing to GitHub:

```bash
gitleaks detect --no-git --source .  # or your preferred secret scanner
bash -n evaluation_suites/paper_experiments/scripts/*.sh scripts/*.sh
python3 -m py_compile scripts/*.py tools/*.py llms/*.py llms/tasks/*.py
```
