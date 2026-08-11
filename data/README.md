# Data

Public benchmarks and deterministic stress-test splits are prepared locally;
the paper's Proposer SFT corpus is the deliberate bundled-data exception and
uses Git LFS.

```bash
bash run_experiment.sh prepare all
```

- `external/`: pinned LLMSRBench, SRBench and SRSD sources for the 895-task
  standard-recovery schedule.
- `generated/noise_sr_20/`: 20 formulas × 3 noise levels × 3 seeds = 180 tasks.
- `generated/balanced_interference_96/`: the shared 96-task suite used for the
  multimodal, agentic-evaluation and component-ablation comparisons.
- `proposer_sft/`: bundled 10,000-example corpus with 10,000 images and 5,000
  task CSV files.

Constructed62 is generated deterministically by its runner. Third-party method
checkouts remain external inputs configured through `.env`. Preparation never
overwrites a non-empty benchmark checkout or an existing generated suite.
