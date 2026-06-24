# VL-LoopSR Paper Experiment Suite

This directory is the paper-facing entry point for the six VL-LoopSR
experiments. It keeps manuscript experiments separate so that runs, source data,
and baseline configs do not get mixed with older exploratory scripts.

## Experiment Layout

1. `01_standard_recovery`
   - Paper figure: Fig. 2.
   - Question: compact formula recovery across heterogeneous benchmarks.
   - Benchmarks: LLMSRBench (240), SLDBench (77), SRBench/Feynman (417), SRSD
     (238), for 972 tasks.
   - Primary readouts: PASS@100, MSE(PASS), perfect-fit rate, expression
     complexity, runtime.

2. `02_extrapolation`
   - Paper figure: Fig. 3.
   - Question: whether selected equations remain accurate outside the observed
     input range.
   - Benchmarks: Constructed62 and SurfaceBench40.
   - Primary readouts: ID/OOD log-MSE, ID-to-OOD shift, OOD/ID degradation.

3. `03_high_dimensional_distractors`
   - Paper figure: Fig. 4.
   - Question: sparse active-variable identification under irrelevant variables,
     correlated proxies, and nonlinear decoys.
   - Construction: 24 formulas x dimensions 200/500/1000 x three distractor
     regimes = 216 tasks.
   - Primary readouts: mean/median test MSE, true-variable recall, FDR, runtime.

4. `04_proposer_sft`
   - Paper figure: Fig. 5.
   - Question: whether task-specific Proposer SFT improves proposal and repair
     behavior with the Evaluator and selector fixed.
   - Dataset: the same 972-task benchmark family before and after Proposer SFT.
   - Primary readouts: candidate count, recovered-case MSE, repair utility,
     runtime, timeout behavior.

5. `05_noise_robustness`
   - Paper figure: Fig. 6.
   - Question: whether symbolic skeletons remain compact and stable under noisy
     fitting/validation targets.
   - Dataset: NoiseRobust-SmallRange v1, 20 formulas x noise levels
     0/0.001/0.01 x three seeds = 180 tasks.
   - Primary readouts: skeleton recovery, clean-test MSE, complexity, runtime,
     timeout behavior.

6. `06_component_ablation`
   - Paper figure: Fig. 7.
   - Question: which loop functions causally support recovery.
   - Dataset: matched balanced 96-task component-ablation suite.
   - Variants: full, w/o Observer, w/o Critic, w/o Proposer.
   - Primary readouts: mean/median test MSE, MSE < 1, active-variable recall,
     proxy misuse, valid-expression count, loop trajectory.

## Code Map

- VL-LoopSR core method: `scripts/test_fey_v11_complexity_exact.py`.
- Standard recovery launch backbone: `scripts/launch_v11_enhanced_300s.sh` and
  official baseline wrappers under `evaluation_suites/*_fourbench`.
- OOD runner: `scripts/run_v11_extrapolation_suites.py`.
- High-dimensional runner: `scripts/run_v11_high_dimensional_interference.py`.
- High-dimensional baselines: `scripts/run_highdim_interference_baselines.py`
  and `scripts/run_highdim_llm_baselines.py`.
- SFT comparison: `scripts/launch_v11_sft_before_after_main_100s.sh` and
  `scripts/summarize_v11_sft_before_after.py`.
- Noise runner and baselines: `scripts/run_v11_noise_robustness.py`,
  `scripts/run_noise_robustness_baselines.py`.
- Component ablation: `scripts/generate_v11_balanced_ablation_dataset.py` and
  `scripts/run_v11_ablation_experiments.py`.
## Data Map

- Noise default manifest: `data/noise_robustness_smallrange_v1/manifest.csv`.
- Noise split files referenced by the manifest: `data/noise_robustness_v1_20/splits`.
- SurfaceBench public cache: `data/surfacebench_public`.
- Balanced component ablation data is generated on demand at
  `data/v11_balanced_component_ablation_96`.
- Public benchmark dependencies are kept in `pmlb/`, `srsd-benchmark/`, and the
  external baseline repositories.

## Launching

Each launcher accepts an optional run id. By default it resolves the repository
root from the launcher path and writes to:

- `${RESULTS_ROOT:-<repo>/runs/paper_experiments/<suite>/<run_id>}`
- `${LOG_ROOT:-<repo>/logs/paper_experiments/<suite>/<run_id>}`

Set `ROOT_DIR`, `RESULTS_ROOT`, or `LOG_ROOT` to target a custom machine layout.

Run the paper-aligned scaffold:

```bash
bash evaluation_suites/paper_experiments/scripts/launch_all_scaffolds.sh
```

Use environment flags such as `LAUNCH_VL_LOOPSR=0`, `LAUNCH_BASELINES=1`, or
`ONLY_SUMMARIZE=1` to control expensive runs.  The launchers default to the
manuscript protocol and avoid older exploratory PASS@300/noise-level settings.
