# Scripts

Public entry points are intentionally limited:

- `vega_sr.py`: canonical VEGA-SR method module.
- `prepare_data.py`: pinned benchmark downloads and paper-suite generation.
- `generate_noise_robustness_dataset.py`: Fig. 3d, 180 NoiseSR-20 tasks.
- `generate_v11_balanced_ablation_dataset.py`: shared 96-task suite for Fig. 4,
  Fig. 5 and supplementary component ablations.
- `run_v11_extrapolation_suites.py`: Fig. 3a Constructed62 runner.
- `run_v11_high_dimensional_interference.py`: Fig. 3b–c runner.
- `run_v11_noise_robustness.py`: Fig. 3d runner.
- `run_v11_ablation_experiments.py`: matched multimodal, agentic and component
  variants.
- `compute_multimodal_ned.py`: paper NED definition and paired structural audit.
- `summarize_candidate_coverage.py`: Fig. 6 held-out candidate coverage.
- `package_sft_corpus.py`: package, unpack and validate the 10k Proposer corpus.

`vl_loopsr.py`, `vl_loopsr_core.py` and the `V11`/`LLMSR_*` identifiers remain
for compatibility with archived runs. New documentation uses VEGA-SR.
