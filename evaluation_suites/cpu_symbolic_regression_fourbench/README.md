# CPU Baseline Configs

This directory is kept as configuration support for the paper launchers.

The executable CPU baseline runner is:

```bash
scripts/run_cpu_baseline_benchmarks.py
```

The paper-facing entry point is:

```bash
bash evaluation_suites/paper_experiments/scripts/launch_all_scaffolds.sh
```

Historical queue scripts, result summaries, notes, and old manifests have been
removed. The remaining files are method registry metadata and YAML configs used
by the current runners.
