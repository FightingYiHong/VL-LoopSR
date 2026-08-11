# VEGA-SR

Official code and reproducibility package for *Multimodal reasoning and
agentic evaluation enable self-refining symbolic regression*. The experiment
matrix is audited against the current local `main.pdf` manuscript build.

VEGA-SR combines multimodal evidence-guided candidate generation with
validation-grounded agentic evaluation and feedback. The release contains the
method, paper-aligned launchers, deterministic data generators and the complete
10,000-example multimodal Proposer SFT corpus. Downloaded benchmarks, generated
splits, model weights and results remain outside Git.

## Install

```bash
git clone https://github.com/RUCAIBox/VEGA-SR.git
cd VEGA-SR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with the OpenAI-compatible endpoint and model used by the agents.
Some baselines require their own upstream environment; the paper reports those
versions separately rather than forcing incompatible packages into one Python
environment. Verified versions are recorded in
[`software_versions.yaml`](evaluation_suites/paper_experiments/configs/software_versions.yaml).

## Paper experiments

```bash
bash run_experiment.sh list
bash run_experiment.sh prepare all

# Two-case wiring check.
RUN_OURS=1 MAX_CASES=2 bash run_experiment.sh 01 smoke

# Main-paper examples.
RUN_OURS=1 bash run_experiment.sh 02 ood_main
RUN_OURS=1 bash run_experiment.sh 05 multimodal_main
RUN_OURS=1 bash run_experiment.sh 06 agentic_main
```

| ID | Paper result | Analysis unit |
|---|---|---:|
| 01 | Fig. 2 standard recovery | 895 tasks |
| 02 | Fig. 3a Constructed62 OOD extrapolation | 62 tasks |
| 03 | Fig. 3b–c high-dimensional distractors | 216 tasks |
| 04 | Fig. 3d NoiseSR-20 robustness | 180 tasks |
| 05 | Fig. 4 numeric-only versus numeric + image | 96 paired tasks |
| 06 | Fig. 5 initial-only versus agentic evaluation | 96 paired tasks |
| 07 | Fig. 6 post-hoc candidate coverage | 895 tasks per method |

All constants are fitted on fitting data, candidates are selected on validation
data, and test/OOD data are reserved for final reporting. Ground-truth
expressions are never supplied during inference. Exact task counts, metrics and
budgets are recorded in
[`experiment_matrix.yaml`](evaluation_suites/paper_experiments/configs/experiment_matrix.yaml).

## SFT corpus and supplementary analyses

The complete paper corpus is in `data/proposer_sft/`: 5,000 candidate-generation
examples and 5,000 matched refinement examples, with 10,000 images and 5,000
CSV files. The two asset archives use Git LFS.

```bash
git lfs install
git lfs pull
bash run_experiment.sh prepare sft
bash run_experiment.sh train-sft

# Paper supplement analyses.
bash run_experiment.sh supp-sft
bash run_experiment.sh supp-ablation
```

The SFT before–after result in the paper is a descriptive archived comparison:
the intended matched SFT-side run was incomplete, so it must not be interpreted
as a checkpoint-only causal estimate.

## Layout

- `scripts/vega_sr.py`: canonical public VEGA-SR method entry point.
- `scripts/vl_loopsr.py`: compatibility entry point retained for archived runs.
- `evaluation_suites/paper_experiments/`: paper-aligned launchers and protocol.
- `scripts/prepare_data.py`: pinned benchmark downloads and generated suites.
- `data/proposer_sft/`: complete multimodal SFT corpus tracked with Git LFS.
- `runs/` and `logs/`: ignored local outputs.

The existing `LLMSR_*` environment-variable prefix is retained for backward
compatibility with archived experiment manifests.

## Validation before release

```bash
bash -n run_experiment.sh evaluation_suites/paper_experiments/scripts/*.sh scripts/*.sh
python3 -m py_compile scripts/*.py tools/*.py llms/*.py llms/tasks/*.py
pytest -q
```

Do not commit `.env`, model weights, downloaded datasets, generated splits,
logs or results. Add the project license before making the repository public.
