# Data Directory

This directory contains generated datasets and public benchmark assets used by
the current paper launchers.

## Included

- `noise_robustness_metric_v2/`
  - Current NoiseRobust-Metric manifest and split files.
  - 20 formulas x 4 noise levels x 5 seeds = 400 tasks.

- `noise_robustness_smallrange_v1/`
  - Legacy NoiseRobust-SmallRange manifest retained for comparison.
  - 20 formulas x 3 noise levels x 3 seeds = 180 tasks.

- `noise_robustness_v1_20/`
  - Split files referenced by `noise_robustness_smallrange_v1/manifest.csv`.

- `surfacebench_public/`
  - SurfaceBench40 cache used by the OOD extrapolation suite.

- `v11_balanced_component_ablation_96/`
  - Matched 96-task component-ablation suite.

- `proposer_sft/`
  - Complete synthetic 10,000-example multimodal Proposer SFT package.
  - Includes 5,000 1D, 3,000 2D/surface, and 2,000 high-dimensional examples
    with relative image and CSV paths.

- `vega_sr_claim_validation_v1/`
  - Frozen inputs used by the VEGA-SR claim-validation workflow.

## Not Included

- `data/llmsrbench/`
  - Optional location for a local LLMSRBench download.
  - Set `LLMSRBENCH_ROOT` and `LLMSRBENCH_HDF5` in `.env` if you store the
    benchmark elsewhere.

Generated results, logs, temporary caches, private model weights, and downloaded
large external datasets should stay out of Git.
