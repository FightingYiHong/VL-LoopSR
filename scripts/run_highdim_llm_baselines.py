#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run direct LLM symbolic-regression baselines on high-dimensional interference cases."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_highdim_interference_baselines as highdim_base
import run_llm_direct_baseline as llm_base
import run_v11_high_dimensional_interference as highdim


def dataframe_table(df: pd.DataFrame, feature_cols: list[str]) -> str:
    cols = list(feature_cols) + ["y"]
    lines = [", ".join(cols)]
    for _, row in df[cols].iterrows():
        lines.append(", ".join(llm_base.compact_number(row[c]) for c in cols))
    return "\n".join(lines)


def make_prompt(case: highdim.HighDimCase, train_df: pd.DataFrame, args, feedback: str | None = None) -> str:
    if len(train_df) <= args.sample_rows:
        sampled = train_df.copy()
    else:
        sampled = train_df.sample(n=args.sample_rows, random_state=args.random_state + case.case_index).copy()
    sampled = sampled.reset_index(drop=True)
    feature_cols = list(case.feature_names)
    parts = [
        "You are a symbolic regression baseline.",
        "Infer one compact mathematical expression for target y from the observed data.",
        f"The dataset has {case.dimension} input variables named x1 through x{case.dimension}.",
        "Use only variables exactly as written in the table.",
        "Allowed operators/functions: +, -, *, /, **, sin, cos, tan, exp, log, sqrt, Abs.",
        "Avoid prose. Return exactly one line in the form: y = <expression>",
        f"Case: {case.case_name}",
        "Sampled training rows:",
        dataframe_table(sampled, feature_cols),
    ]
    if feedback:
        parts.extend(
            [
                "Previous candidate feedback:",
                feedback,
                "Revise the formula to reduce validation error while keeping it compact.",
            ]
        )
    return "\n".join(parts)


def run_one(case: highdim.HighDimCase, args) -> dict:
    started = time.time()
    train_df, val_df, test_df = highdim.load_case_splits(
        case,
        args.random_state + args.repeat_seed * 1_000_000,
        args.n_train,
        args.n_val,
        args.n_test,
    )
    client = llm_base.openai_client(args.api_base, args.api_key)
    model = llm_base.resolve_model(client, args.model)
    raw_responses: list[str] = []
    candidates: list[dict] = []
    best = None
    feedback = None
    attempts = 0
    try:
        deadline = started + max(1.0, float(args.case_timeout_sec))
        while attempts < args.max_attempts and time.time() < deadline:
            attempts += 1
            request_timeout = max(5.0, min(args.request_timeout_sec, deadline - time.time()))
            prompt = make_prompt(case, train_df, args, feedback=feedback)
            raw = llm_base.call_llm(
                client=client,
                model=model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=request_timeout,
            )
            raw_responses.append(raw)
            expr = llm_base.strip_candidate(raw)
            try:
                result = llm_base.evaluate_candidate(
                    args.method_name,
                    expr,
                    train_df,
                    val_df,
                    test_df,
                    case.feature_names,
                )
                result["fit_runtime_sec"] = float(time.time() - started)
                result["num_candidate_exprs"] = int(attempts)
                candidates.append(
                    {
                        "attempt": attempts,
                        "expr": expr,
                        "val_mse": result.get("best_val_mse"),
                        "test_mse": result.get("best_test_mse"),
                        "error": None,
                    }
                )
                if llm_base.better_result(result, best):
                    best = result
                feedback = (
                    f"Attempt {attempts}: expression `{expr}` produced "
                    f"train MSE={llm_base.compact_number(result.get('best_train_mse'))}, "
                    f"validation MSE={llm_base.compact_number(result.get('best_val_mse'))}. "
                    "Try a simpler or more accurate formula."
                )
                if result.get("perfect_fit_by_r2") or result.get("perfect_fit"):
                    break
            except Exception as exc:
                candidates.append({"attempt": attempts, "expr": expr, "val_mse": None, "test_mse": None, "error": repr(exc)})
                feedback = (
                    f"Attempt {attempts}: expression `{expr}` could not be evaluated: {repr(exc)}. "
                    "Return a valid expression using only the listed variables."
                )

        if best is None:
            out = {
                "method": args.method_name,
                "valid_formula_found": False,
                "passed": False,
                "best_expr": None,
                "best_test_mse": None,
                "error": "no valid LLM formula within budget",
            }
        else:
            out = dict(best)
        out.update(
            {
                "method": args.method_name,
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
                "timed_out": bool(time.time() >= deadline and best is None),
                "llm_model": str(model),
                "llm_api_base": str(args.api_base),
                "llm_attempts": int(attempts),
                "candidate_trace": json.dumps(candidates, ensure_ascii=False),
                "raw_response_tail": raw_responses[-1][-2000:] if raw_responses else None,
                "true_expression_for_scoring": case.true_expression,
            }
        )
        out.update(highdim_base.score_result(case, out, train_df, val_df, test_df))
        return out
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        out = {
            "method": args.method_name,
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
    result = run_one(case, args)
    result_path = Path(args.single_result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(highdim.json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def summarize(rows: list[dict], out_dir: Path) -> None:
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
        "true_variable_recall",
        "false_variable_discovery_rate",
        "proxy_misuse",
        "nonlinear_decoy_misuse",
        "irrelevant_misuse",
        "best_test_mse",
        "test_rmse",
        "test_nmse",
        "test_r2",
        "runtime_sec",
        "expr_complexity",
    ]:
        if column not in df:
            df[column] = None
    df.to_csv(out_dir / "all_high_dimensional_interference_llm_baseline_results.csv", index=False)
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
            false_variable_discovery_rate=("false_variable_discovery_rate", "mean"),
            proxy_misuse_rate=("proxy_misuse", "mean"),
            nonlinear_decoy_misuse_rate=("nonlinear_decoy_misuse", "mean"),
            irrelevant_misuse_rate=("irrelevant_misuse", "mean"),
            median_test_mse=("best_test_mse", "median"),
            median_test_rmse=("test_rmse", "median"),
            median_test_nmse=("test_nmse", "median"),
            median_complexity=("expr_complexity", "median"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary_high_dimensional_interference_llm_baselines.csv", index=False)
    manifest = {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_rows": int(len(df)),
        "methods": sorted([str(x) for x in df["method"].dropna().unique()]),
        "result_csv": str(out_dir / "all_high_dimensional_interference_llm_baseline_results.csv"),
        "summary_csv": str(out_dir / "summary_high_dimensional_interference_llm_baselines.csv"),
    }
    (out_dir / "manifest_high_dimensional_interference_llm_baselines.json").write_text(
        json.dumps(highdim.json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def timeout_row(args, case: highdim.HighDimCase, runtime_sec: float, log_path: Path, parent_timeout_sec: float) -> dict:
    return {
        "method": args.method_name,
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


def run_parent(args) -> int:
    cases = highdim.make_cases(args.max_cases)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(case) for case in cases]).to_csv(out_dir / "selected_high_dimensional_interference_cases.csv", index=False)
    (out_dir / "run_config.json").write_text(
        json.dumps(highdim.json_safe(vars(args)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result_dir = out_dir / "case_results" / args.method_name
    log_dir = out_dir / "case_logs" / args.method_name
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    parent_timeout_sec = float(args.parent_timeout_sec or 0.0)
    if parent_timeout_sec <= 0:
        parent_timeout_sec = float(args.case_timeout_sec) + float(args.timeout_grace_sec)
    rows: list[dict] = []

    for case in cases:
        safe = highdim.sanitize_name(case.case_name)
        result_json = result_dir / f"{case.case_index:03d}_{safe}_seed{args.repeat_seed}.json"
        log_path = log_dir / f"{case.case_index:03d}_{safe}_seed{args.repeat_seed}.log"
        if args.resume and result_json.exists():
            rows.append(json.loads(result_json.read_text(encoding="utf-8")))
            continue
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
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
            "--api-base",
            str(args.api_base),
            "--api-key",
            str(args.api_key),
            "--model",
            str(args.model),
            "--method-name",
            str(args.method_name),
            "--request-timeout-sec",
            str(args.request_timeout_sec),
            "--max-attempts",
            str(args.max_attempts),
            "--sample-rows",
            str(args.sample_rows),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--single-result-json",
            str(result_json),
        ]
        print(f"[RUN {args.method_name} {case.case_index}/{len(cases)}] {case.case_name}", flush=True)
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as log_fp:
            try:
                subprocess.run(cmd, stdout=log_fp, stderr=subprocess.STDOUT, timeout=parent_timeout_sec, check=False)
            except subprocess.TimeoutExpired:
                row = timeout_row(args, case, time.time() - started, log_path, parent_timeout_sec)
                rows.append(row)
                result_json.write_text(json.dumps(highdim.json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
                summarize(rows, out_dir)
                print(f"[TIMEOUT {args.method_name} {case.case_index}/{len(cases)}] sec={time.time() - started:.1f}", flush=True)
                continue
        if result_json.exists():
            row = json.loads(result_json.read_text(encoding="utf-8"))
        else:
            row = {
                **timeout_row(args, case, time.time() - started, log_path, parent_timeout_sec),
                "timed_out": False,
                "error": f"child exited without result; see {log_path}",
            }
        rows.append(row)
        summarize(rows, out_dir)
        print(
            f"[DONE {args.method_name} {case.case_index}/{len(cases)}] "
            f"recall={row.get('true_variable_recall')} fdr={row.get('false_variable_discovery_rate')} "
            f"mse={row.get('best_test_mse')} sec={row.get('runtime_sec')}",
            flush=True,
        )
    summarize(rows, out_dir)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "highdim_interference_llm_baselines"))
    parser.add_argument("--api-base", default=os.environ.get("LLMSR_BASELINE_API_BASE", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--api-key", default=os.environ.get("LLMSR_BASELINE_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("LLMSR_BASELINE_MODEL", ""))
    parser.add_argument("--method-name", default=os.environ.get("LLMSR_BASELINE_METHOD_NAME", "llm_sr_baseline"))
    parser.add_argument("--case-timeout-sec", type=float, default=300.0)
    parser.add_argument("--request-timeout-sec", type=float, default=80.0)
    parser.add_argument("--timeout-grace-sec", type=float, default=30.0)
    parser.add_argument("--parent-timeout-sec", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--sample-rows", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=256)
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
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
