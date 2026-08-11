#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run non-V11 baselines on the high-dimensional interference suite.

This uses the exact same synthetic cases/splits as
``run_v11_high_dimensional_interference.py`` and scores the same post-hoc
variable-selection metrics. It is intentionally a small adapter around the
main experiment baseline functions in ``run_cpu_baseline_benchmarks.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_cpu_baseline_benchmarks as cpu_base
import run_v11_high_dimensional_interference as highdim
from benchmark_metrics import (
    NUMERICAL_FIT_R2_THRESHOLD,
    evaluate_expression,
    expression_complexity,
    regression_metrics,
    srbench_formula_recovery,
    strict_formula_recovery,
)


PASS_MSE_THRESHOLD = 100.0
EXACT_MSE_THRESHOLD = 1e-8
SKELETON_MSE_THRESHOLD = 1e-4

DEFAULT_CONFIGS = {
    "gplearn": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "gplearn_100s_1thread.yaml",
    "pyoperon": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "pyoperon_100s_1thread.yaml",
    "pysr": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "pysr_100s_1thread.yaml",
    "deap": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "deap_100s.yaml",
    "itea": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "itea_100s.yaml",
    "psrn": ROOT / "evaluation_suites" / "gpu_symbolic_regression_fourbench" / "configs" / "psrn_limited.yaml",
    "physo": ROOT / "evaluation_suites" / "gpu_symbolic_regression_fourbench" / "configs" / "physo_100s_fast.env",
    "dso": ROOT / "evaluation_suites" / "gpu_symbolic_regression_fourbench" / "configs" / "dso_limited.yaml",
    "ffx": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "ffx_fast.yaml",
    "bingo": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "bingo_limited.yaml",
    "rils_rols": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "rils_rols_limited.yaml",
}

FIT_FUNCTIONS = {
    "gplearn": cpu_base.fit_gplearn,
    "pyoperon": cpu_base.fit_pyoperon,
    "pysr": cpu_base.fit_pysr,
    "deap": cpu_base.fit_deap,
    "itea": cpu_base.fit_itea,
    "psrn": cpu_base.fit_psrn,
    "physo": None,
    "dso": cpu_base.fit_dso,
    "ffx": cpu_base.fit_ffx,
    "bingo": cpu_base.fit_bingo,
    "rils_rols": cpu_base.fit_rils_rols,
}


def parse_methods(text: str) -> list[str]:
    out = []
    for item in str(text or "").split(","):
        method = item.strip()
        if not method:
            continue
        if method not in FIT_FUNCTIONS:
            raise ValueError(f"unknown baseline method: {method}")
        out.append(method)
    if not out:
        raise ValueError("at least one method is required")
    return out


def json_safe(value):
    return highdim.json_safe(value)


def sanitize_name(text: str) -> str:
    return highdim.sanitize_name(text)


def method_config(method: str, override: str | None = None) -> Path:
    if override:
        return Path(override)
    value = os.environ.get(f"HIGHDIM_BASELINE_{method.upper()}_CONFIG")
    if value:
        return Path(value)
    return DEFAULT_CONFIGS[method]


def variables_used(expr: str | None, feature_names: list[str]) -> list[str]:
    if not expr:
        return []
    text = str(expr)
    used = set()
    if re.search(r"\bx0\b", text):
        for match in re.finditer(r"\bx(\d+)\b", text):
            idx = int(match.group(1))
            if 0 <= idx < len(feature_names):
                used.add(feature_names[idx])
    else:
        for name in feature_names:
            if re.search(rf"\b{re.escape(name)}\b", text):
                used.add(name)
    for match in re.finditer(r"\bX_?(\d+)\b", text):
        idx = int(match.group(1))
        if 0 <= idx < len(feature_names):
            used.add(feature_names[idx])
    return [name for name in feature_names if name in used]


def fit_physo_highdim(train_df, val_df, test_df, config_path, random_state=None):
    import run_physo_baseline as physo_runner

    class Args:
        pass

    args = Args()
    args.max_train_rows = int(os.environ.get("PHYSO_MAX_TRAIN_ROWS", "1000"))
    args.random_state = int(random_state if random_state is not None else 0)
    args.epochs = int(os.environ.get("PHYSO_EPOCHS", "8"))
    ops = os.environ.get("PHYSO_OP_NAMES", "add,sub,mul,div,sin,cos,exp,log")
    args.op_names = [x.strip() for x in ops.split(",") if x.strip()]
    args.stop_reward = float(os.environ.get("PHYSO_STOP_REWARD", "1.0"))
    args.max_evaluations = int(os.environ.get("PHYSO_MAX_EVALUATIONS", "500000"))
    args.stop_after_n_epochs = int(os.environ.get("PHYSO_STOP_AFTER_N_EPOCHS", str(args.epochs)))
    args.device = os.environ.get("PHYSO_DEVICE", "cuda")
    result = physo_runner.fit_physo(train_df, val_df, test_df, args)
    result["config_path"] = str(config_path)
    return result


FIT_FUNCTIONS["physo"] = fit_physo_highdim


def score_result(
    case: highdim.HighDimCase,
    result: dict,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> dict:
    best_expr = result.get("best_expr")
    active = variables_used(best_expr, case.feature_names)
    active_set = set(active)
    true_set = set(case.true_variables)
    proxy_set = set(case.proxy_variables)
    decoy_set = set(case.nonlinear_decoy_variables)

    true_variable_recall = len(active_set & true_set) / max(1, len(true_set))
    true_variable_precision = len(active_set & true_set) / max(1, len(active_set))
    false_count = len(active_set - true_set)
    false_variable_discovery_rate = false_count / max(1, len(active_set))
    irrelevant_variable_false_positive_rate = false_count / max(
        1, case.dimension - len(true_set)
    )
    exact_support_recovery = active_set == true_set
    proxy_misuse = bool(active_set & proxy_set)
    nonlinear_decoy_misuse = bool(active_set & decoy_set)
    irrelevant_misuse = bool(active_set - true_set - proxy_set - decoy_set)

    try:
        test_mse = float(result.get("best_test_mse"))
    except Exception:
        test_mse = float("inf")
    skeleton_recovery = bool(
        true_variable_recall >= 1.0
        and false_variable_discovery_rate <= 0.25
        and test_mse <= SKELETON_MSE_THRESHOLD
    )
    exact_recovery_proxy = bool(active_set == true_set and test_mse <= EXACT_MSE_THRESHOLD)
    strict_recovery = strict_formula_recovery(best_expr, case.true_expression, case.feature_names)
    srbench_recovery = srbench_formula_recovery(best_expr, case.true_expression, case.feature_names)
    metric_updates = {}
    if best_expr and train_df is not None and val_df is not None and test_df is not None:
        try:
            for split_name, frame in (
                ("train", train_df),
                ("val", val_df),
                ("test", test_df),
            ):
                predictions = evaluate_expression(best_expr, frame, case.feature_names)
                metrics = regression_metrics(frame["y"].to_numpy(dtype=float), predictions)
                for metric_name, metric_value in metrics.items():
                    metric_updates[f"{split_name}_{metric_name}"] = metric_value
        except Exception as exc:
            metric_updates["final_expr_metric_eval_error"] = repr(exc)
    test_r2 = metric_updates.get("test_r2", result.get("test_r2"))
    return {
        "active_variables": "|".join(active),
        "true_variables": "|".join(case.true_variables),
        "proxy_variables": "|".join(case.proxy_variables),
        "nonlinear_decoy_variables": "|".join(case.nonlinear_decoy_variables),
        "true_variable_recall": true_variable_recall,
        "true_variable_precision": true_variable_precision,
        "false_variable_discovery_rate": false_variable_discovery_rate,
        "irrelevant_variable_false_positive_rate": irrelevant_variable_false_positive_rate,
        "exact_support_recovery": exact_support_recovery,
        "wrong_variable_count": false_count,
        "proxy_misuse": proxy_misuse,
        "nonlinear_decoy_misuse": nonlinear_decoy_misuse,
        "irrelevant_misuse": irrelevant_misuse,
        "skeleton_recovery": skeleton_recovery,
        "exact_recovery": bool(strict_recovery),
        "strict_formula_recovery": bool(strict_recovery),
        "strict_formula_recovery_evaluable": strict_recovery is not None,
        "srbench_formula_recovery": bool(srbench_recovery),
        "srbench_formula_recovery_evaluable": srbench_recovery is not None,
        "exact_recovery_proxy": exact_recovery_proxy,
        "numerical_complete_fit": bool(
            test_r2 is not None and float(test_r2) > NUMERICAL_FIT_R2_THRESHOLD
        ),
        "passed": bool(test_mse <= PASS_MSE_THRESHOLD),
        "expr_complexity": expression_complexity(best_expr, case.feature_names).get("expr_complexity"),
        **metric_updates,
    }


def run_one(method: str, case: highdim.HighDimCase, args) -> dict:
    started = time.time()
    train_df, val_df, test_df = highdim.load_case_splits(
        case,
        args.random_state + args.repeat_seed * 1_000_000,
        args.n_train,
        args.n_val,
        args.n_test,
    )
    try:
        config_path = method_config(method, args.config)
        fit_fn = FIT_FUNCTIONS[method]
        result = fit_fn(
            train_df,
            val_df,
            test_df,
            str(config_path),
            random_state=args.random_state + case.case_index * 1000 + args.repeat_seed,
        )
        out = {
            **result,
            "method": method,
            "case_index": case.case_index,
            "case_name": case.case_name,
            "base_formula_id": case.base_formula_id,
            "source_family": case.source_family,
            "source_name": case.source_name,
            "dimension": case.dimension,
            "true_variable_count": case.true_variable_count,
            "interference_type": case.interference_type,
            "benchmark": case.benchmark,
            "structure_type": case.structure_type,
            "repeat_seed": int(args.repeat_seed),
            "runtime_sec": float(result.get("fit_runtime_sec") or (time.time() - started)),
            "timed_out": False,
            "true_expression_for_scoring": case.true_expression,
            "config_path": str(config_path),
        }
        out.update(score_result(case, out, train_df, val_df, test_df))
        return out
    except BaseException as exc:
        out = {
            "method": method,
            "case_index": case.case_index,
            "case_name": case.case_name,
            "base_formula_id": case.base_formula_id,
            "source_family": case.source_family,
            "source_name": case.source_name,
            "dimension": case.dimension,
            "true_variable_count": case.true_variable_count,
            "interference_type": case.interference_type,
            "benchmark": case.benchmark,
            "structure_type": case.structure_type,
            "repeat_seed": int(args.repeat_seed),
            "runtime_sec": float(time.time() - started),
            "timed_out": "time" in repr(exc).lower() or "timeout" in repr(exc).lower(),
            "valid_formula_found": False,
            "passed": False,
            "best_expr": None,
            "best_test_mse": None,
            "true_expression_for_scoring": case.true_expression,
            "true_variables": "|".join(case.true_variables),
            "active_variables": "",
            "true_variable_recall": 0.0,
            "true_variable_precision": 0.0,
            "false_variable_discovery_rate": 0.0,
            "wrong_variable_count": 0,
            "proxy_misuse": False,
            "nonlinear_decoy_misuse": False,
            "irrelevant_misuse": False,
            "skeleton_recovery": False,
            "exact_recovery": False,
            "error": repr(exc),
        }
        return out


def run_child(args) -> int:
    cases = highdim.make_cases(args.max_cases)
    case = cases[int(args.case_index) - 1]
    result = run_one(args.method, case, args)
    result_path = Path(args.single_result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def timeout_row(args, method: str, case: highdim.HighDimCase, runtime_sec: float, log_path: Path, parent_timeout_sec: float):
    return {
        "method": method,
        "case_index": int(case.case_index),
        "case_name": case.case_name,
        "base_formula_id": case.base_formula_id,
        "source_family": case.source_family,
        "source_name": case.source_name,
        "dimension": int(case.dimension),
        "true_variable_count": int(case.true_variable_count),
        "interference_type": case.interference_type,
        "benchmark": case.benchmark,
        "structure_type": case.structure_type,
        "repeat_seed": int(args.repeat_seed),
        "timed_out": True,
        "runtime_sec": float(runtime_sec),
        "valid_formula_found": False,
        "passed": False,
        "best_expr": None,
        "best_test_mse": None,
        "true_expression_for_scoring": case.true_expression,
        "true_variables": "|".join(case.true_variables),
        "active_variables": "",
        "true_variable_recall": 0.0,
        "true_variable_precision": 0.0,
        "false_variable_discovery_rate": 0.0,
        "wrong_variable_count": 0,
        "proxy_misuse": False,
        "nonlinear_decoy_misuse": False,
        "irrelevant_misuse": False,
        "skeleton_recovery": False,
        "exact_recovery": False,
        "case_log_path": str(log_path),
        "error": f"case exceeded outer timeout {parent_timeout_sec:.1f}s",
    }


def summarize(rows: list[dict], out_dir: Path):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    for column in [
        "timed_out",
        "passed",
        "exact_recovery",
        "strict_formula_recovery",
        "numerical_complete_fit",
        "exact_recovery_proxy",
        "skeleton_recovery",
        "proxy_misuse",
        "nonlinear_decoy_misuse",
        "irrelevant_misuse",
        "true_variable_recall",
        "true_variable_precision",
        "false_variable_discovery_rate",
        "irrelevant_variable_false_positive_rate",
        "exact_support_recovery",
        "wrong_variable_count",
        "true_variable_count",
        "dimension",
        "interference_type",
        "best_test_mse",
        "test_rmse",
        "test_nmse",
        "test_nrmse",
        "test_r2",
        "runtime_sec",
        "expr_complexity",
    ]:
        if column not in df:
            df[column] = np.nan
    df.to_csv(out_dir / "all_high_dimensional_interference_baseline_results.csv", index=False)
    summary = (
        df.groupby(["method", "dimension", "interference_type"], dropna=False)
        .agg(
            n=("case_index", "count"),
            timeout_rate=("timed_out", "mean"),
            pass_rate=("passed", "mean"),
            exact_recovery=("exact_recovery", "mean"),
            numerical_complete_fit=("numerical_complete_fit", "mean"),
            skeleton_recovery=("skeleton_recovery", "mean"),
            true_variable_recall=("true_variable_recall", "mean"),
            true_variable_precision=("true_variable_precision", "mean"),
            false_variable_discovery_rate=("false_variable_discovery_rate", "mean"),
            irrelevant_variable_false_positive_rate=("irrelevant_variable_false_positive_rate", "mean"),
            exact_support_recovery=("exact_support_recovery", "mean"),
            proxy_misuse_rate=("proxy_misuse", "mean"),
            nonlinear_decoy_misuse_rate=("nonlinear_decoy_misuse", "mean"),
            irrelevant_misuse_rate=("irrelevant_misuse", "mean"),
            median_test_mse=("best_test_mse", "median"),
            median_test_rmse=("test_rmse", "median"),
            median_test_nmse=("test_nmse", "median"),
            median_test_nrmse=("test_nrmse", "median"),
            median_complexity=("expr_complexity", "median"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary_high_dimensional_interference_baselines.csv", index=False)
    manifest = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": int(len(df)),
        "methods": sorted([str(x) for x in df["method"].dropna().unique()]),
        "result_csv": str(out_dir / "all_high_dimensional_interference_baseline_results.csv"),
        "summary_csv": str(out_dir / "summary_high_dimensional_interference_baselines.csv"),
    }
    (out_dir / "manifest_high_dimensional_interference_baselines.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_parent(args) -> int:
    methods = parse_methods(args.methods)
    cases = highdim.make_cases(args.max_cases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(case) for case in cases]).to_csv(out_dir / "selected_high_dimensional_interference_cases.csv", index=False)

    rows: list[dict] = []
    result_dir = out_dir / "case_results" / "baselines"
    log_dir = out_dir / "case_logs" / "baselines"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    parent_timeout_sec = float(args.parent_timeout_sec or 0.0)
    if parent_timeout_sec <= 0:
        parent_timeout_sec = float(args.case_timeout_sec) + float(args.timeout_grace_sec)

    for method in methods:
        for case in cases:
            safe = sanitize_name(case.case_name)
            result_json = result_dir / method / f"{case.case_index:03d}_{safe}_seed{args.repeat_seed}.json"
            log_path = log_dir / method / f"{case.case_index:03d}_{safe}_seed{args.repeat_seed}.log"
            result_json.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if args.resume and result_json.exists():
                rows.append(json.loads(result_json.read_text(encoding="utf-8")))
                continue
            child_python = str(args.child_python or sys.executable)
            cmd = [
                child_python,
                str(Path(__file__).resolve()),
                "--child",
                "--method",
                method,
                "--out-dir",
                str(out_dir),
                "--case-index",
                str(case.case_index),
                "--repeat-seed",
                str(args.repeat_seed),
                "--n-train",
                str(args.n_train),
                "--n-val",
                str(args.n_val),
                "--n-test",
                str(args.n_test),
                "--case-timeout-sec",
                str(args.case_timeout_sec),
                "--timeout-grace-sec",
                str(args.timeout_grace_sec),
                "--max-cases",
                str(args.max_cases or 0),
                "--random-state",
                str(args.random_state),
                "--single-result-json",
                str(result_json),
            ]
            if args.config:
                cmd.extend(["--config", args.config])
            started = time.time()
            print(f"[RUN {method} {case.case_index}/{len(cases)}] {case.case_name}", flush=True)
            child_env = os.environ.copy()
            child_env.setdefault("PYTHONUNBUFFERED", "1")
            with open(log_path, "w", encoding="utf-8") as log_fp:
                try:
                    subprocess.run(
                        cmd,
                        stdout=log_fp,
                        stderr=subprocess.STDOUT,
                        timeout=parent_timeout_sec,
                        check=False,
                        env=child_env,
                    )
                except subprocess.TimeoutExpired:
                    row = timeout_row(args, method, case, time.time() - started, log_path, parent_timeout_sec)
                    rows.append(row)
                    result_json.write_text(json.dumps(json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
                    summarize(rows, out_dir)
                    print(f"[TIMEOUT {method} {case.case_index}/{len(cases)}] sec={time.time() - started:.1f}", flush=True)
                    continue
            if result_json.exists():
                row = json.loads(result_json.read_text(encoding="utf-8"))
            else:
                row = {
                    **timeout_row(args, method, case, time.time() - started, log_path, parent_timeout_sec),
                    "timed_out": False,
                    "error": f"child exited without result; see {log_path}",
                }
            rows.append(row)
            summarize(rows, out_dir)
            print(
                f"[DONE {method} {case.case_index}/{len(cases)}] "
                f"recall={row.get('true_variable_recall')} fdr={row.get('false_variable_discovery_rate')} "
                f"mse={row.get('best_test_mse')} sec={row.get('runtime_sec')}",
                flush=True,
            )
    summarize(rows, out_dir)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="pyoperon,gplearn")
    parser.add_argument("--method", default="pyoperon")
    parser.add_argument("--config", default="")
    parser.add_argument("--child-python", default=os.environ.get("HIGHDIM_BASELINE_CHILD_PYTHON", ""))
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "highdim_interference_baselines"))
    parser.add_argument("--case-timeout-sec", type=float, default=300.0)
    parser.add_argument("--timeout-grace-sec", type=float, default=30.0)
    parser.add_argument("--parent-timeout-sec", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--repeat-seed", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--single-result-json", default="")
    args = parser.parse_args()
    if args.max_cases <= 0:
        args.max_cases = None
    return args


def main():
    args = parse_args()
    os.environ["CPU_SR_MSE_THRESHOLD"] = str(PASS_MSE_THRESHOLD)
    cpu_base.MSE_THRESHOLD = PASS_MSE_THRESHOLD
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
