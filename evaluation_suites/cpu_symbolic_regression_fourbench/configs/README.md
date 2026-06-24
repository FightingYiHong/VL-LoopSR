# Config Policy

这个套件保留原始 `fast/practical` 配置，另提供更适合论文式质量对比的 profile：

- `fast`: 快速 sanity check，沿用 external `cpu_fast.yaml` / `cpu_practical.yaml`。
- `fair`/`medium`: 快速预实验入口，使用本目录的 medium/extended 配置，单 case 预算大致为数分钟级，接近 `our_full` 的 420s 上限。
- `extended`: 更强 CPU baseline，单 case 预算约 20-30 分钟。
- `paper`/`paper_hours`: 长预算 baseline，单 case 最长约 6 小时；这是更接近论文式完整配置的主对比入口。

These configs are consumed by the current runner:

```bash
python3 scripts/run_cpu_baseline_benchmarks.py --help
```

The old tmux/queue helper scripts have been removed from the minimal paper
repository; use the launchers in `evaluation_suites/paper_experiments/scripts`.

`aifeynman` 当前结果有效率很低，建议作为附录或单独诊断项，不作为主表核心 CPU baseline。
