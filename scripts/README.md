# Script Entry Points

Use `evaluation_suites/paper_experiments/` as the paper-facing orchestration
layer. The scripts here are the lower-level runners and generators required by
those launchers.

## Current Paper Pipeline

- VL-LoopSR core method: `test_fey_v11_complexity_exact.py`.
- OOD extrapolation: `run_v11_extrapolation_suites.py`.
- High-dimensional distractors: `run_v11_high_dimensional_interference.py`.
- High-dimensional baselines: `run_highdim_interference_baselines.py`,
  `run_highdim_llm_baselines.py`.
- SFT before/after: `launch_v11_sft_before_after_main_100s.sh`,
  `summarize_v11_sft_before_after.py`.
- Noise robustness: `run_v11_noise_robustness.py`,
  `run_noise_robustness_baselines.py`,
  `generate_noise_robustness_dataset.py`.
- Component ablation: `generate_v11_balanced_ablation_dataset.py`,
  `run_v11_ablation_experiments.py`.

## Minimal Scope

Only the scripts needed by the six paper launchers are kept here. Historical
plotting helpers, exploratory workflow variants, one-off queue scripts, and old
root-level launch wrappers have been removed.
