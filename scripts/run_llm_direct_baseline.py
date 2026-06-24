#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run an LLM-only symbolic-regression baseline on the four recovery benchmarks."""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import run_cpu_baseline_benchmarks as base

RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S", time.localtime())


def compact_number(value):
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def sample_rows_for_prompt(train_df, max_rows, random_state):
    if len(train_df) <= max_rows:
        sampled = train_df.copy()
    else:
        sampled = train_df.sample(n=max_rows, random_state=random_state).copy()
    return sampled.reset_index(drop=True)


def dataframe_table(df, feature_cols):
    cols = list(feature_cols) + ["y"]
    lines = [", ".join(cols)]
    for _, row in df[cols].iterrows():
        lines.append(", ".join(compact_number(row[c]) for c in cols))
    return "\n".join(lines)


def make_prompt(benchmark, row, train_df, feature_cols, max_rows, random_state, feedback=None):
    sampled = sample_rows_for_prompt(train_df, max_rows=max_rows, random_state=random_state)
    variables = ", ".join(str(x) for x in feature_cols)
    case_name = base.row_case_name(benchmark, row)
    parts = [
        "You are a symbolic regression baseline.",
        "Infer one compact mathematical expression for target y from the observed data.",
        "Use only these variables exactly as written: " + variables,
        "Allowed operators/functions: +, -, *, /, **, sin, cos, tan, exp, log, sqrt, Abs.",
        "Avoid prose. Return exactly one line in the form: y = <expression>",
        f"Benchmark: {benchmark}",
        f"Case: {case_name}",
        "Sampled training rows:",
        dataframe_table(sampled, feature_cols),
    ]
    true_expr = row.get("true_expression")
    if true_expr:
        parts.append("Do not copy any hidden target formula; infer from data only.")
    if feedback:
        parts.extend([
            "Previous candidate feedback:",
            feedback,
            "Revise the formula to reduce validation error while keeping it compact.",
        ])
    return "\n".join(parts)


def openai_client(api_base, api_key):
    from openai import OpenAI

    return OpenAI(base_url=api_base, api_key=api_key)


def resolve_model(client, requested):
    if requested:
        return requested
    models = client.models.list()
    if not models.data:
        raise RuntimeError("vLLM returned no models")
    return models.data[0].id


def call_llm(client, model, prompt, temperature, max_tokens, request_timeout):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You produce valid symbolic-regression formulas and no extra explanation.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=request_timeout,
    )
    return response.choices[0].message.content or ""


def strip_candidate(text):
    text = str(text or "").strip()
    fenced = re.search(r"```(?:python|text|math)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(("formula:", "expression:", "expr:")):
            line = line.split(":", 1)[1].strip()
        if "=" in line:
            left, right = line.split("=", 1)
            if left.strip().lower() in {"y", "target", "f", "output"}:
                line = right.strip()
        line = line.strip().strip("`").strip()
        line = re.sub(r"\bmath\.", "", line)
        line = line.replace("^", "**")
        line = line.replace("ln(", "log(")
        if line:
            return line
    return text.replace("^", "**")


def evaluate_candidate(method_name, expr, train_df, val_df, test_df, feature_cols):
    pred_train = base.evaluate_expression(expr, train_df, feature_cols)
    pred_val = base.evaluate_expression(expr, val_df, feature_cols)
    pred_test = base.evaluate_expression(expr, test_df, feature_cols)
    _, y_train, _ = base.dataframe_to_xy(train_df)
    _, y_val, _ = base.dataframe_to_xy(val_df)
    _, y_test, _ = base.dataframe_to_xy(test_df)
    return base._metric_dict(
        method_name,
        expr,
        pred_train,
        pred_val,
        pred_test,
        y_train,
        y_val,
        y_test,
        feature_cols,
        0.0,
        num_candidates=None,
    )


def better_result(candidate, best):
    if best is None:
        return True
    cand_val = candidate.get("best_val_mse")
    best_val = best.get("best_val_mse")
    if cand_val is None:
        return False
    if best_val is None:
        return True
    return float(cand_val) < float(best_val)


def run_one_case(args, client, model, benchmark, mod, row, idx, total):
    start = time.time()
    raw_responses = []
    candidates = []
    try:
        train_df, val_df, test_df, meta = base.load_case(benchmark, mod, row)
        _, _, feature_cols = base.dataframe_to_xy(train_df)
        deadline = start + max(1.0, float(args.case_timeout_sec))
        best = None
        feedback = None
        attempts = 0
        while attempts < args.max_attempts and time.time() < deadline:
            attempts += 1
            request_timeout = max(5.0, min(args.request_timeout_sec, deadline - time.time()))
            prompt = make_prompt(
                benchmark=benchmark,
                row=row,
                train_df=train_df,
                feature_cols=feature_cols,
                max_rows=args.sample_rows,
                random_state=args.random_state + attempts,
                feedback=feedback,
            )
            raw = call_llm(
                client=client,
                model=model,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                request_timeout=request_timeout,
            )
            raw_responses.append(raw)
            expr = strip_candidate(raw)
            try:
                result = evaluate_candidate(args.method_name, expr, train_df, val_df, test_df, feature_cols)
                result["fit_runtime_sec"] = float(time.time() - start)
                result["num_candidate_exprs"] = int(attempts)
                candidates.append({
                    "attempt": attempts,
                    "expr": expr,
                    "val_mse": result.get("best_val_mse"),
                    "test_mse": result.get("best_test_mse"),
                    "error": None,
                })
                if better_result(result, best):
                    best = result
                feedback = (
                    f"Attempt {attempts}: expression `{expr}` produced "
                    f"train MSE={compact_number(result.get('best_train_mse'))}, "
                    f"validation MSE={compact_number(result.get('best_val_mse'))}. "
                    "Try a simpler or more accurate formula."
                )
                if result.get("perfect_fit_by_r2") or result.get("perfect_fit"):
                    break
            except Exception as exc:
                candidates.append({
                    "attempt": attempts,
                    "expr": expr,
                    "val_mse": None,
                    "test_mse": None,
                    "error": repr(exc),
                })
                feedback = (
                    f"Attempt {attempts}: expression `{expr}` could not be evaluated: {repr(exc)}. "
                    "Return a valid expression using only the listed variables."
                )

        if best is None:
            res = base.failure_result(
                benchmark,
                args.method_name,
                row,
                time.time() - start,
                "no valid LLM formula within budget",
            )
        else:
            res = best
        res.update(meta)
        res.update({
            "method": args.method_name,
            "n_features": int(train_df.shape[1] - 1),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "runtime_sec": float(time.time() - start),
            "case_timeout_sec": float(args.case_timeout_sec),
            "llm_model": str(model),
            "llm_api_base": str(args.api_base),
            "llm_attempts": int(attempts),
            "timeout_budget_exhausted": bool(time.time() >= deadline and not (res.get("perfect_fit") or res.get("perfect_fit_by_r2"))),
            "candidate_trace": json.dumps(candidates, ensure_ascii=False),
            "raw_response_tail": raw_responses[-1][-2000:] if raw_responses else None,
            "error": None if best is not None else res.get("error"),
        })
        return base.json_safe_record(res)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        res = base.failure_result(benchmark, args.method_name, row, time.time() - start, repr(exc))
        res.update({
            "completed_index": int(idx),
            "total_tasks": int(total),
            "case_timeout_sec": float(args.case_timeout_sec),
            "llm_model": str(model),
            "llm_api_base": str(args.api_base),
        })
        return base.json_safe_record(res)


def run_benchmark(args, client, model, benchmark):
    mod, df_tasks = base.collect_tasks(benchmark)
    if args.max_cases is not None and args.max_cases > 0:
        df_tasks = df_tasks.head(args.max_cases).copy()

    results_root = Path(args.results_root) / benchmark
    case_dir = results_root / "case_results"
    results_root.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    df_tasks.to_csv(results_root / f"{benchmark}_selected_tasks.csv", index=False, encoding="utf-8-sig")
    results_csv = results_root / f"all_{benchmark}_{args.method_name}_results.csv"

    total = len(df_tasks)
    all_results = []
    for idx, (_, row_series) in enumerate(df_tasks.iterrows(), start=1):
        row = row_series.to_dict()
        case_slug = base.sanitize_name(base.row_case_name(benchmark, row) or f"case_{idx}")
        result_json = case_dir / f"{idx:04d}_{case_slug}.json"
        if args.resume and result_json.exists():
            with open(result_json, "r", encoding="utf-8") as fp:
                res = json.load(fp)
            print(f"[{benchmark} {idx}/{total}] resume: {base.row_case_name(benchmark, row)}", flush=True)
        else:
            print(f"[{benchmark} {idx}/{total}] starting: {base.row_case_name(benchmark, row)}", flush=True)
            res = run_one_case(args, client, model, benchmark, mod, row, idx, total)
            res["completed_index"] = int(idx)
            res["total_tasks"] = int(total)
            with open(result_json, "w", encoding="utf-8") as fp:
                json.dump(base.json_safe_record(res), fp, ensure_ascii=False, allow_nan=False, indent=2)
        all_results.append(res)
        base.print_one_result(benchmark, idx, total, row, res)
        pd.DataFrame([base.json_safe_record(r) for r in all_results]).to_csv(results_csv, index=False)

    print(f"Saved {benchmark} results to: {results_csv}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run an LLM-only SR baseline on four benchmarks.")
    parser.add_argument("--benchmarks", default="sldbench,llmsrbench,srsd,srbench")
    parser.add_argument(
        "--results-root",
        default=str(base.PROJECT_ROOT / "runs" / f"llm_direct_refine_{RUN_TIMESTAMP}"),
    )
    parser.add_argument("--api-base", default=os.environ.get("LLMSR_BASELINE_API_BASE", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--api-key", default=os.environ.get("LLMSR_BASELINE_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("LLMSR_BASELINE_MODEL", ""))
    parser.add_argument("--method-name", default=os.environ.get("LLMSR_BASELINE_METHOD_NAME", "llm_sr_baseline"))
    parser.add_argument("--case-timeout-sec", type=float, default=900.0)
    parser.add_argument("--request-timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--sample-rows", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    client = openai_client(args.api_base, args.api_key)
    model = resolve_model(client, args.model)
    Path(args.results_root).mkdir(parents=True, exist_ok=True)
    with open(Path(args.results_root) / "run_config.json", "w", encoding="utf-8") as fp:
        json.dump(vars(args) | {"resolved_model": model}, fp, ensure_ascii=False, indent=2)
    benchmarks = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    for benchmark in benchmarks:
        run_benchmark(args, client, model, benchmark)


if __name__ == "__main__":
    main()
