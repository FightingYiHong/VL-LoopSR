#!/usr/bin/env python3
"""Orchestrate VEGA-SR claim-validation experiments from the JSON config."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "vega_sr_claim_validation_v1.json"


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def expand_value(value: str, defaults: dict[str, str]) -> str:
    text = str(value)

    def repl(match: re.Match) -> str:
        name = match.group(1)
        return os.environ.get(name, defaults.get(name, match.group(0)))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, text)


def resolved_paths(config: dict) -> dict[str, Path]:
    defaults = {
        "LLMSR_ROOT": str(ROOT),
        "LLMSR_RESULTS_ROOT": str(ROOT / "reports"),
        "LLMSR_ABLATION_DATASET_ROOT": str(ROOT / "data" / "vega_sr_claim_validation_v1"),
    }
    paths = {}
    for key, value in dict(config.get("paths", {}) or {}).items():
        paths[key] = Path(expand_value(str(value), defaults)).resolve()
    paths.setdefault("root", ROOT)
    paths.setdefault("dataset_root", ROOT / "data" / "vega_sr_claim_validation_v1")
    paths.setdefault("output_root", ROOT / "reports" / "vega_sr_claim_validation_v1")
    return paths


def base_env(config: dict, paths: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("LLMSR_ROOT", str(paths["root"]))
    env.setdefault("LLMSR_ABLATION_DATASET_ROOT", str(paths["dataset_root"]))
    env.setdefault("LLMSR_RESULTS_ROOT", str(paths["output_root"].parent))
    for key, value in dict(config.get("shared_env", {}) or {}).items():
        env[str(key)] = str(value)
    api_base = (
        env.get("LLMSR_AGENT_API_BASE")
        or env.get("LLMSR_DEFAULT_API_BASE")
        or "http://127.0.0.1:8001/v1"
    )
    if any(token in str(api_base) for token in ["127.0.0.1", "localhost", "::1"]):
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)
    return env


def missing_required_env(config: dict, env: dict[str, str]) -> list[str]:
    missing = []
    for name in list(config.get("required_environment", []) or []):
        if not str(env.get(name, "")).strip():
            missing.append(str(name))
    return missing


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("[cmd] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


def ensure_dataset(config: dict, paths: dict[str, Path], env: dict[str, str], overwrite: bool = False) -> None:
    dataset_cfg = dict(config.get("dataset", {}) or {})
    dataset_root = paths["dataset_root"]
    manifest = dataset_root / "manifest.csv"
    if manifest.exists() and not overwrite:
        print(f"[dataset] using existing {manifest}", flush=True)
        return
    if not bool(dataset_cfg.get("generate_if_missing", True)) and not manifest.exists():
        raise FileNotFoundError(f"dataset manifest missing and generate_if_missing=false: {manifest}")
    generator = paths.get("dataset_generator", ROOT / "scripts" / "generate_v11_component_ablation_v4.py")
    cmd = [
        sys.executable,
        str(generator),
        "--out-dir",
        str(dataset_root),
        "--cases-per-target",
        str(dataset_cfg.get("cases_per_target", 24)),
        "--targets",
        ",".join(str(x) for x in list(dataset_cfg.get("targets", ["observer", "critic"]))),
        "--n-train",
        str(dataset_cfg.get("n_train", 256)),
        "--n-val",
        str(dataset_cfg.get("n_val", 128)),
        "--n-test",
        str(dataset_cfg.get("n_test", 1024)),
    ]
    if overwrite:
        cmd.append("--overwrite")
    run_cmd(cmd, env)


def selected_studies(config: dict, study_names: str) -> list[dict]:
    studies = list(config.get("studies", []) or [])
    if not study_names:
        return studies
    wanted = {x.strip() for x in re.split(r"[, ]+", study_names) if x.strip()}
    return [study for study in studies if str(study.get("name")) in wanted]


def selected_seeds(config: dict, seed_text: str, stage: str) -> list[int]:
    seeds = [int(x) for x in list((config.get("execution", {}) or {}).get("seeds", [0]))]
    if seed_text:
        seeds = [int(x) for x in re.split(r"[, ]+", seed_text.strip()) if x]
    if stage == "pilot":
        return seeds[:1]
    return seeds


def max_cases_for_stage(config: dict, stage: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    execution = dict(config.get("execution", {}) or {})
    if stage == "pilot":
        return int(execution.get("pilot_max_cases", 4) or 4)
    return int(execution.get("full_max_cases", 0) or 0)


def run_experiments(args: argparse.Namespace, config: dict, paths: dict[str, Path], env: dict[str, str]) -> None:
    execution = dict(config.get("execution", {}) or {})
    dataset_cfg = dict(config.get("dataset", {}) or {})
    studies = selected_studies(config, args.studies)
    seeds = selected_seeds(config, args.seeds, args.stage)
    max_cases = max_cases_for_stage(config, args.stage, args.max_cases)
    runner = paths.get("ablation_runner", ROOT / "scripts" / "run_v11_ablation_experiments.py")
    v11_path = paths.get("v11_path", ROOT / "scripts" / "test_fey_v11_complexity_exact.py")
    output_root = paths["output_root"]

    for study in studies:
        for seed in seeds:
            run_env = env.copy()
            run_env["PYTHONHASHSEED"] = str(seed)
            run_env["LLMSR_REPEAT_SEED"] = str(seed)
            out_dir = output_root / str(study["name"]) / f"seed_{seed}"
            cmd = [
                sys.executable,
                str(runner),
                "--experiment-config",
                str(args.config),
                "--suite",
                str(study.get("suite", "component_ablation")),
                "--target-component",
                str(study.get("target_component", "")),
                "--methods",
                ",".join(str(x) for x in list(study.get("methods", []) or [])),
                "--out-dir",
                str(out_dir),
                "--v11-path",
                str(v11_path),
                "--dataset-root",
                str(paths["dataset_root"]),
                "--n-train",
                str(dataset_cfg.get("n_train", 256)),
                "--n-val",
                str(dataset_cfg.get("n_val", 128)),
                "--n-test",
                str(dataset_cfg.get("n_test", 1024)),
                "--repeat-seed",
                str(seed),
                "--random-state",
                str(execution.get("random_state", 20260526)),
                "--case-budget-sec",
                str(args.case_budget_sec if args.case_budget_sec is not None else execution.get("case_budget_sec", 300)),
                "--timeout-grace-sec",
                str(execution.get("timeout_grace_sec", 60)),
                "--parent-timeout-sec",
                str(execution.get("parent_timeout_sec", 390)),
                "--max-cases",
                str(max_cases),
                "--interleave-methods",
            ]
            if bool(execution.get("resume", True)) and not args.no_resume:
                cmd.append("--resume")
            if args.rerun_failures:
                cmd.append("--rerun-failures")
            if args.rerun_timeouts:
                cmd.append("--rerun-timeouts")
            run_cmd(cmd, run_env)


def run_summary(args: argparse.Namespace, paths: dict[str, Path], env: dict[str, str]) -> None:
    summarizer = paths.get("claim_summarizer", ROOT / "scripts" / "summarize_vega_sr_claim_validation.py")
    cmd = [
        sys.executable,
        str(summarizer),
        "--config",
        str(args.config),
        "--output-root",
        str(paths["output_root"]),
    ]
    run_cmd(cmd, env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--studies", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--case-budget-sec", type=float, default=None)
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--rerun-failures", action="store_true")
    parser.add_argument("--rerun-timeouts", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.config = args.config.resolve()
    config = load_config(args.config)
    paths = resolved_paths(config)
    env = base_env(config, paths)

    if args.summary_only:
        run_summary(args, paths, env)
        return 0

    ensure_dataset(config, paths, env, overwrite=args.overwrite_dataset)
    if args.prepare_only:
        print(f"[prepare] dataset ready at {paths['dataset_root']}", flush=True)
        return 0

    missing = missing_required_env(config, env)
    if missing and not args.skip_env_check:
        print("[env] missing required environment variables: " + ", ".join(missing), file=sys.stderr, flush=True)
        print("[env] export them or rerun with --skip-env-check to use code defaults.", file=sys.stderr, flush=True)
        return 2

    run_experiments(args, config, paths, env)
    run_summary(args, paths, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
