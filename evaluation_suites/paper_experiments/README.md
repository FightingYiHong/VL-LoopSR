# Paper experiments

These launchers implement the protocols in the current `main.pdf`. They use
fitting data for constant estimation, validation data for model selection and
held-out test/OOD data only for reporting.

## Main-text analyses

| Command | Figure | Experiment | Cases | Default budget |
|---|---|---|---:|---:|
| `01` | Fig. 2 | LLMSRBench 240 + SRBench/Feynman 417 + SRSD 238 | 895 | method-specific |
| `02` | Fig. 3a | Constructed62 OOD extrapolation | 62 | method-specific |
| `03` | Fig. 3b–c | 24 formulas × 3 dimensions × 3 distractor regimes | 216 | VEGA-SR 900 s |
| `04` | Fig. 3d | 20 formulas × 3 noise levels × 3 seeds | 180 | 100 s |
| `05` | Fig. 4 | Numeric-only versus numeric + one image | 96 paired | 90 s |
| `06` | Fig. 5 | Initial evaluation versus complete agentic loop | 96 paired | 600 s |
| `07` | Fig. 6 | Stored-candidate coverage at budgets 1–100 | 895/method | post-hoc |

Run from the repository root:

```bash
cp .env.example .env
bash run_experiment.sh prepare all
bash run_experiment.sh list

RUN_OURS=1 MAX_CASES=2 bash run_experiment.sh 01 smoke
RUN_OURS=1 bash run_experiment.sh 02 ood_main
RUN_BASELINES=1 bash run_experiment.sh 04 noise_baselines
RUN_MULTIMODAL=1 bash run_experiment.sh 05 multimodal_main
RUN_AGENTIC=1 bash run_experiment.sh 06 agentic_main
```

Expensive methods default to disabled. `MAX_CASES` is a wiring-check option;
zero selects the complete suite. Results are written to
`runs/paper_experiments/<suite>/<run_id>` and logs to the corresponding
`logs/paper_experiments/` directory.

The Fig. 6 coverage analysis does not use test scores during search. It reads
stored candidates after all runs, evaluates them on held-out test data and
records the first candidate with test R2 > 0.999. Its input schema is documented
by `scripts/summarize_candidate_coverage.py --help`.

## Supplementary analyses

```bash
bash run_experiment.sh supp-sft
bash run_experiment.sh supp-ablation
```

`supp-sft` summarizes the 895-pair base/SFT archive. It is intentionally
labelled descriptive because the intended matched SFT-side run was incomplete.
`supp-ablation` runs the full, no-Observer, no-Critic and no-Proposer variants
on the balanced 96-task suite.

External baseline repositories are downloaded or installed separately and
remain under ignored paths. The complete 10,000-example Proposer corpus is the
one bundled-data exception; `bash run_experiment.sh train-sft` reproduces the
paper's 4-bit QLoRA configuration after LLaMA-Factory is installed.
