#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the official deep-symbolic-mathematics/LLM-SR pipeline on four benchmarks.

This is an adapter only: the evolutionary loop, experience buffer, evaluator,
and profiler are imported from the official LLM-SR repository. The adapter
creates per-case LLM-SR specifications and calls an OpenAI-compatible vLLM
server because the official local server uses a custom /completions API.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OFFICIAL_ROOT = Path(
    os.environ.get("OFFICIAL_LLMSR_ROOT", str(PROJECT_ROOT / "data" / "external" / "methods" / "LLM-SR"))
)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(OFFICIAL_ROOT))

import run_cpu_baseline_benchmarks as base
from llmsr import config as official_config
from llmsr import evaluator as official_evaluator
from llmsr import pipeline as official_pipeline
from llmsr import sampler as official_sampler


class CaseBudgetExceeded(Exception):
    pass


class OfficialVLLMChatLLM(official_sampler.LLM):
    api_base = "http://127.0.0.1:8001/v1"
    api_key = "EMPTY"
    model = "llm-baseline-qwen2.5-32b"
    temperature = 0.8
    max_tokens = 512
    request_timeout_sec = 180.0

    def __init__(self, samples_per_prompt: int) -> None:
        super().__init__(samples_per_prompt)
        from openai import OpenAI

        self._client = OpenAI(base_url=self.api_base, api_key=self.api_key)

    def draw_samples(self, prompt: str, config: official_config.Config):
        system = (
            "You are running the official LLM-SR algorithm. Complete only the "
            "Python body of the equation function. Use numpy operations, the "
            "provided variables, and params for constants. Do not explain."
        )
        user_prompt = "\n".join([
            "Complete the function below. Return a Python function body or a full def equation block.",
            prompt,
        ])
        samples = []
        for _ in range(self._samples_per_prompt):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.request_timeout_sec,
            )
            text = response.choices[0].message.content or ""
            samples.append(official_sampler._extract_body(text, config))
        return samples


def sanitize_name(text: str) -> str:
    return base.sanitize_name(text)


def finite_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def make_case_spec(benchmark: str, row: dict, train_df: pd.DataFrame, max_params: int) -> str:
    _, _, feature_cols = base.dataframe_to_xy(train_df)
    args = ", ".join([f"x{i}: np.ndarray" for i in range(len(feature_cols))] + ["params: np.ndarray"])
    linear_terms = []
    for i in range(len(feature_cols)):
        param_idx = min(i + 1, max_params - 1)
        linear_terms.append(f"params[{param_idx}] * x{i}")
    initial_expr = " + ".join(["params[0]"] + linear_terms) if linear_terms else "params[0]"
    original_names = ", ".join(str(c) for c in feature_cols)
    case_name = base.row_case_name(benchmark, row)
    return f'''"""
Official LLM-SR symbolic regression task.
Benchmark: {benchmark}
Case: {case_name}
Original feature names mapped to x0..x{max(0, len(feature_cols) - 1)}: {original_names}
Find a compact mathematical function that predicts y from the provided variables.
"""

import numpy as np

MAX_NPARAMS = {int(max_params)}
params = [1.0] * MAX_NPARAMS


@evaluate.run
def evaluate(data: dict) -> float:
    """Evaluate the equation on training observations."""
    inputs, outputs = data["inputs"], data["outputs"]
{chr(10).join([f"    x{i} = inputs[:, {i}]" for i in range(len(feature_cols))])}

    from scipy.optimize import minimize

    def loss(params):
        try:
            y_pred = equation({", ".join([f"x{i}" for i in range(len(feature_cols))] + ["params"])})
            y_pred = np.asarray(y_pred, dtype=float)
            if y_pred.shape == ():
                y_pred = np.full_like(outputs, float(y_pred), dtype=float)
            y_pred = y_pred.reshape(-1)
            if len(y_pred) != len(outputs) or not np.all(np.isfinite(y_pred)):
                return 1e100
            value = np.mean((y_pred - outputs) ** 2)
            return float(value) if np.isfinite(value) else 1e100
        except Exception:
            return 1e100

    result = minimize(loss, [1.0] * MAX_NPARAMS, method="BFGS")
    score = -float(result.fun)
    if np.isnan(score) or np.isinf(score):
        return None
    return score


@equation.evolve
def equation({args}) -> np.ndarray:
    """Return predicted y for this symbolic regression task."""
    return {initial_expr}
'''


def load_best_sample(samples_dir: Path):
    best = None
    for path in sorted(samples_dir.glob("samples_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        score = finite_float(data.get("score"))
        if score is None:
            continue
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "function": data.get("function"),
                "sample_order": data.get("sample_order"),
                "path": str(path),
            }
    return best


def load_samples_in_generation_order(samples_dir: Path) -> list[dict]:
    """Return generated equation candidates in their documented sample order."""
    samples = []
    for fallback_order, path in enumerate(
        sorted(samples_dir.glob("samples_*.json")), start=1
    ):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        function = data.get("function")
        if not isinstance(function, str) or not function.strip():
            continue
        try:
            sample_order = int(data.get("sample_order"))
        except (TypeError, ValueError):
            sample_order = fallback_order
        samples.append(
            {
                "sample_order": sample_order,
                "function": function,
                "official_score": finite_float(data.get("score")),
                "path": str(path),
            }
        )
    return sorted(samples, key=lambda item: (item["sample_order"], item["path"]))


def evaluate_validation_first_hit(
    samples: list[dict],
    train_df,
    val_df,
    test_df,
    original_train_df,
    original_val_df,
    original_test_df,
    norm_info,
    *,
    max_params: int,
    timeout_sec: float,
    trace_path: Path,
    threshold: float = 0.999,
) -> dict:
    """Evaluate candidates in generation order using validation data only."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    observed = 0
    first_hit = None
    trace = []

    def _alarm_handler(_signum, _frame):
        raise TimeoutError("candidate validation fit timed out")

    for sample in samples:
        observed += 1
        function = sample["function"]
        seen.add(function)
        event = {
            "candidate_index": observed,
            "unique_candidate_index": len(seen),
            "native_evaluations": sample["sample_order"],
            "expression": extract_return_expression(function) or function,
            "official_score": sample["official_score"],
            "source_path": sample["path"],
            "status": "invalid",
            "val_r2": None,
        }
        previous_handler = signal.getsignal(signal.SIGALRM)
        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, max(1.0, float(timeout_sec)))
            predictions = fit_predict(
                function,
                train_df,
                val_df,
                test_df,
                original_train_df,
                original_val_df,
                original_test_df,
                norm_info,
                max_params=max_params,
            )
            val_r2 = base.safe_r2(predictions["y_val"], predictions["pred_val"])
            event["val_r2"] = finite_float(val_r2)
            event["status"] = "valid" if event["val_r2"] is not None else "invalid"
        except Exception as exc:
            event["error"] = repr(exc)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        trace.append(event)
        if event["val_r2"] is not None and event["val_r2"] > threshold:
            first_hit = event
            break

    with gzip.open(trace_path, "wt", encoding="utf-8") as handle:
        for event in trace:
            handle.write(json.dumps(base.json_safe_record(event), ensure_ascii=False))
            handle.write("\n")

    hit = first_hit or {}
    return {
        "validation_search_success": bool(first_hit),
        "evaluations_to_validation_success": hit.get("candidate_index"),
        "unique_evaluations_to_validation_success": hit.get(
            "unique_candidate_index"
        ),
        "native_evaluations_to_validation_success": hit.get("native_evaluations"),
        "first_validation_success_expression": hit.get("expression"),
        "first_validation_success_val_r2": hit.get("val_r2"),
        "validation_search_observed_evaluations": observed,
        "validation_search_unique_evaluations": len(seen),
        "validation_search_censored": not bool(first_hit),
        "validation_search_threshold": float(threshold),
        "validation_search_trace": str(trace_path),
        "validation_search_definition": (
            "first generated official LLM-SR equation with validation R2 > 0.999; "
            "test data are not used for first-hit selection"
        ),
    }


def count_ast_nodes(source: str):
    try:
        tree = ast.parse(source)
    except Exception:
        return len(str(source).replace(" ", "")), None, None
    return (
        sum(1 for _ in ast.walk(tree)),
        max((getattr(node, "col_offset", 0) for node in ast.walk(tree)), default=0),
        len(str(source).replace(" ", "")),
    )


def extract_return_expression(function_code: str) -> str | None:
    """Extract a compact expression from an official LLM-SR equation function."""
    try:
        tree = ast.parse(function_code)
    except Exception:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            try:
                return ast.unparse(node.value)
            except Exception:
                return None
    return None


def substitute_params(expr: str | None, params: np.ndarray) -> str | None:
    if not expr:
        return expr

    def _replace(match):
        idx = int(match.group(1))
        if idx >= len(params):
            return match.group(0)
        return repr(float(params[idx]))

    return re.sub(r"\bparams\s*\[\s*(\d+)\s*\]", _replace, str(expr))


def execute_equation(function_code: str):
    namespace = {"np": np, "numpy": np}
    exec(function_code, namespace)
    candidates = [v for k, v in namespace.items() if k.startswith("equation") and callable(v)]
    if not candidates:
        raise ValueError("no equation function found in official LLM-SR sample")
    return candidates[-1]


def normalize_splits(train_df, val_df, test_df):
    feature_cols = [c for c in train_df.columns if c != "y"]
    x_mu = train_df[feature_cols].to_numpy(dtype=float).mean(axis=0)
    x_scale = train_df[feature_cols].to_numpy(dtype=float).std(axis=0)
    x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
    y_mu = float(train_df["y"].to_numpy(dtype=float).mean())
    y_scale = float(train_df["y"].to_numpy(dtype=float).std())
    if y_scale <= 1e-12:
        y_scale = 1.0

    def _norm(df):
        out = df.copy()
        out[feature_cols] = (out[feature_cols].to_numpy(dtype=float) - x_mu) / x_scale
        out["y"] = (out["y"].to_numpy(dtype=float) - y_mu) / y_scale
        return out

    return _norm(train_df), _norm(val_df), _norm(test_df), {
        "x_mu": x_mu,
        "x_scale": x_scale,
        "y_mu": y_mu,
        "y_scale": y_scale,
    }


def fit_predict(function_code: str, train_df, val_df, test_df, original_train_df, original_val_df, original_test_df, norm_info, max_params: int):
    equation = execute_equation(function_code)
    X_train, y_train, feature_cols = base.dataframe_to_xy(train_df)
    X_val, y_val, _ = base.dataframe_to_xy(val_df)
    X_test, y_test, _ = base.dataframe_to_xy(test_df)

    def predict_with_params(X, params):
        arrays = [X[:, i] for i in range(X.shape[1])]
        pred = equation(*arrays, params)
        pred = np.asarray(pred, dtype=float)
        if pred.shape == ():
            pred = np.full(X.shape[0], float(pred), dtype=float)
        pred = pred.reshape(-1)
        if len(pred) != X.shape[0] or not np.all(np.isfinite(pred)):
            raise ValueError("invalid predictions")
        return pred

    def loss(params):
        try:
            pred = predict_with_params(X_train, params)
            value = np.mean((pred - y_train) ** 2)
            return float(value) if np.isfinite(value) else 1e100
        except Exception:
            return 1e100

    fit_start = time.time()
    opt = minimize(loss, np.ones(int(max_params), dtype=float), method="BFGS")
    fit_sec = time.time() - fit_start
    params = np.asarray(opt.x, dtype=float)
    pred_train = predict_with_params(X_train, params)
    pred_val = predict_with_params(X_val, params)
    pred_test = predict_with_params(X_test, params)
    y_scale = float(norm_info["y_scale"])
    y_mu = float(norm_info["y_mu"])
    return {
        "pred_train": pred_train * y_scale + y_mu,
        "pred_val": pred_val * y_scale + y_mu,
        "pred_test": pred_test * y_scale + y_mu,
        "y_train": original_train_df["y"].to_numpy(dtype=float),
        "y_val": original_val_df["y"].to_numpy(dtype=float),
        "y_test": original_test_df["y"].to_numpy(dtype=float),
        "feature_cols": feature_cols,
        "optimized_params": params,
        "fit_runtime_sec": fit_sec,
    }


def case_paths(result_root: Path, idx: int, case_name: str):
    slug = sanitize_name(case_name or f"case_{idx}")
    log_dir = result_root / "official_logs" / f"{idx:04d}_{slug}"
    spec_dir = result_root / "specs"
    case_dir = result_root / "case_results"
    spec_path = spec_dir / f"{idx:04d}_{slug}.py"
    result_json = case_dir / f"{idx:04d}_{slug}.json"
    return slug, log_dir, spec_dir, case_dir, spec_path, result_json


def finalize_case_result(
    args,
    benchmark: str,
    row: dict,
    idx: int,
    total: int,
    result_root: Path,
    runtime_sec: float,
    timed_out: bool,
    error: str | None = None,
):
    """Build a case JSON from official samples, even after a parent timeout."""
    case_name = base.row_case_name(benchmark, row)
    slug, log_dir, spec_dir, case_dir, spec_path, result_json = case_paths(
        result_root, idx, case_name
    )
    spec_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_df_raw, val_df_raw, test_df_raw, meta = base.load_case(benchmark, None, row)
    except Exception:
        # Some loaders need the imported benchmark module.
        mod, _ = base.collect_tasks(benchmark)
        train_df_raw, val_df_raw, test_df_raw, meta = base.load_case(benchmark, mod, row)

    train_df, val_df, test_df, norm_info = normalize_splits(train_df_raw, val_df_raw, test_df_raw)
    if not spec_path.exists():
        spec = make_case_spec(benchmark, row, train_df, max_params=args.max_params)
        spec_path.write_text(spec, encoding="utf-8")

    best = load_best_sample(log_dir / "samples")
    if best is None:
        message = error or "no valid official LLM-SR sample"
        res = base.failure_result(benchmark, "official_llm_sr", row, runtime_sec, message)
    else:
        try:
            res = metrics_from_best(
                best,
                train_df,
                val_df,
                test_df,
                train_df_raw,
                val_df_raw,
                test_df_raw,
                norm_info,
                max_params=args.max_params,
                runtime_sec=runtime_sec,
            )
        except Exception as exc:
            res = base.failure_result(benchmark, "official_llm_sr", row, runtime_sec, repr(exc))
            res["official_best_function"] = best.get("function")
            res["official_train_score"] = best.get("score")
            res["official_sample_order"] = best.get("sample_order")
            res["official_sample_path"] = best.get("path")

    ordered_samples = load_samples_in_generation_order(log_dir / "samples")
    search_metrics = evaluate_validation_first_hit(
        ordered_samples,
        train_df,
        val_df,
        test_df,
        train_df_raw,
        val_df_raw,
        test_df_raw,
        norm_info,
        max_params=args.max_params,
        timeout_sec=args.evaluate_timeout_sec,
        trace_path=(
            result_root
            / "candidate_traces"
            / f"{idx:04d}_{slug}.jsonl.gz"
        ),
    )
    res.update(search_metrics)
    res.update(meta)
    res.update({
        "method": "official_llm_sr",
        "n_features": int(train_df.shape[1] - 1),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "completed_index": int(idx),
        "total_tasks": int(total),
        "case_timeout_sec": float(args.case_timeout_sec),
        "timeout_budget_exhausted": bool(timed_out),
        "official_repo": "https://github.com/deep-symbolic-mathematics/LLM-SR",
        "official_root": str(OFFICIAL_ROOT),
        "official_commit": args.official_commit,
        "spec_path": str(spec_path),
        "official_log_dir": str(log_dir),
        "llm_model": args.model,
        "llm_api_base": args.api_base,
        "normalization": json.dumps({
            "x_mu": norm_info["x_mu"].tolist(),
            "x_scale": norm_info["x_scale"].tolist(),
            "y_mu": norm_info["y_mu"],
            "y_scale": norm_info["y_scale"],
        }),
    })
    result_json.write_text(json.dumps(base.json_safe_record(res), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return base.json_safe_record(res)


def metrics_from_best(best, train_df, val_df, test_df, original_train_df, original_val_df, original_test_df, norm_info, max_params: int, runtime_sec: float):
    fn = best["function"]
    expr = extract_return_expression(fn)
    preds = fit_predict(
        fn,
        train_df,
        val_df,
        test_df,
        original_train_df,
        original_val_df,
        original_test_df,
        norm_info,
        max_params=max_params,
    )
    test_mse = base.safe_mse(preds["y_test"], preds["pred_test"])
    test_r2 = base.safe_r2(preds["y_test"], preds["pred_test"])
    complexity, depth, strlen = count_ast_nodes(fn)
    expr = substitute_params(expr, preds["optimized_params"])
    return {
        "method": "official_llm_sr",
        "valid_formula_found": True,
        "num_candidate_exprs": int(best.get("sample_order") or 0),
        "best_expr": expr or fn,
        "best_expr_sympy": expr,
        "official_best_function": fn,
        "best_train_mse": base.safe_mse(preds["y_train"], preds["pred_train"]),
        "best_val_mse": base.safe_mse(preds["y_val"], preds["pred_val"]),
        "best_test_mse": test_mse,
        "train_mae": base.safe_mae(preds["y_train"], preds["pred_train"]),
        "val_mae": base.safe_mae(preds["y_val"], preds["pred_val"]),
        "test_mae": base.safe_mae(preds["y_test"], preds["pred_test"]),
        "train_r2": base.safe_r2(preds["y_train"], preds["pred_train"]),
        "val_r2": base.safe_r2(preds["y_val"], preds["pred_val"]),
        "test_r2": test_r2,
        "passed": bool(test_mse is not None and test_mse <= base.MSE_THRESHOLD),
        "perfect_fit": bool(test_mse is not None and test_mse <= base.PERFECT_FIT_TOL),
        "perfect_fit_by_r2": bool(test_r2 is not None and test_r2 >= base.PERFECT_FIT_R2_THRESHOLD),
        "expr_complexity": int(complexity) if complexity is not None else None,
        "expr_depth": int(depth) if depth is not None else None,
        "expr_string_length": int(strlen) if strlen is not None else None,
        "expr_sympy_ops": None,
        "feature_names": " | ".join(str(x) for x in preds["feature_cols"]),
        "optimized_params": json.dumps(preds["optimized_params"].tolist()),
        "official_train_score": best.get("score"),
        "official_sample_order": best.get("sample_order"),
        "official_sample_path": best.get("path"),
        "fit_runtime_sec": float(preds["fit_runtime_sec"]),
        "runtime_sec": float(runtime_sec),
        "error": None,
    }


def run_official_case(args, benchmark: str, mod, row: dict, idx: int, total: int, result_root: Path):
    start = time.time()
    case_name = base.row_case_name(benchmark, row)
    slug, log_dir, spec_dir, case_dir, spec_path, result_json = case_paths(result_root, idx, case_name)
    spec_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and result_json.exists():
        return json.loads(result_json.read_text(encoding="utf-8"))

    try:
        train_df_raw, val_df_raw, test_df_raw, meta = base.load_case(benchmark, mod, row)
        train_df, val_df, test_df, norm_info = normalize_splits(train_df_raw, val_df_raw, test_df_raw)
        spec = make_case_spec(benchmark, row, train_df, max_params=args.max_params)
        spec_path.write_text(spec, encoding="utf-8")
        X_train, y_train, _ = base.dataframe_to_xy(train_df)
        dataset = {"data": {"inputs": X_train, "outputs": y_train}}

        OfficialVLLMChatLLM.api_base = args.api_base
        OfficialVLLMChatLLM.api_key = args.api_key
        OfficialVLLMChatLLM.model = args.model
        OfficialVLLMChatLLM.temperature = args.temperature
        OfficialVLLMChatLLM.max_tokens = args.max_tokens
        OfficialVLLMChatLLM.request_timeout_sec = args.request_timeout_sec

        cfg = official_config.Config(
            num_samplers=1,
            num_evaluators=1,
            samples_per_prompt=args.samples_per_prompt,
            evaluate_timeout_seconds=args.evaluate_timeout_sec,
            use_api=False,
        )
        classes = official_config.ClassConfig(
            llm_class=OfficialVLLMChatLLM,
            sandbox_class=official_evaluator.LocalSandbox,
        )

        old_handler = signal.getsignal(signal.SIGALRM)

        def _handler(signum, frame):
            raise CaseBudgetExceeded(f"official LLM-SR case exceeded {args.case_timeout_sec:.1f}s")

        if not args.disable_inprocess_alarm:
            signal.signal(signal.SIGALRM, _handler)
            signal.setitimer(signal.ITIMER_REAL, max(1.0, float(args.case_timeout_sec)))
        timed_out = False
        try:
            official_pipeline.main(
                specification=spec,
                inputs=dataset,
                config=cfg,
                max_sample_nums=args.max_sample_nums,
                class_config=classes,
                log_dir=str(log_dir),
            )
        except CaseBudgetExceeded:
            timed_out = True
        finally:
            if not args.disable_inprocess_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        return finalize_case_result(
            args,
            benchmark,
            row,
            idx,
            total,
            result_root,
            runtime_sec=time.time() - start,
            timed_out=timed_out,
        )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        res = base.failure_result(benchmark, "official_llm_sr", row, time.time() - start, repr(exc))
        res.update({
            "completed_index": int(idx),
            "total_tasks": int(total),
            "case_timeout_sec": float(args.case_timeout_sec),
            "timeout_budget_exhausted": False,
            "official_repo": "https://github.com/deep-symbolic-mathematics/LLM-SR",
            "official_root": str(OFFICIAL_ROOT),
            "official_commit": args.official_commit,
            "spec_path": str(spec_path),
            "official_log_dir": str(log_dir),
        })
        result_json.write_text(json.dumps(base.json_safe_record(res), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        return base.json_safe_record(res)


def run_official_case_isolated(args, benchmark: str, row: dict, idx: int, total: int, result_root: Path):
    case_name = base.row_case_name(benchmark, row)
    _, _, _, _, _, result_json = case_paths(result_root, idx, case_name)
    if args.resume and result_json.exists():
        return json.loads(result_json.read_text(encoding="utf-8"))

    supervisor_log_dir = result_root / "supervisor_logs"
    supervisor_log_dir.mkdir(parents=True, exist_ok=True)
    supervisor_log = supervisor_log_dir / f"{idx:04d}_{sanitize_name(case_name or f'case_{idx}')}.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--benchmarks",
        benchmark,
        "--results-root",
        str(args.results_root),
        "--api-base",
        args.api_base,
        "--api-key",
        args.api_key,
        "--model",
        args.model,
        "--case-timeout-sec",
        str(args.case_timeout_sec),
        "--evaluate-timeout-sec",
        str(args.evaluate_timeout_sec),
        "--request-timeout-sec",
        str(args.request_timeout_sec),
        "--samples-per-prompt",
        str(args.samples_per_prompt),
        "--max-sample-nums",
        str(args.max_sample_nums),
        "--max-params",
        str(args.max_params),
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--official-commit",
        args.official_commit,
        "--single-case-index",
        str(idx),
        "--disable-inprocess-alarm",
    ]
    if args.resume:
        cmd.append("--resume")

    start = time.time()
    timed_out = False
    return_code = None
    with supervisor_log.open("w", encoding="utf-8") as fp:
        fp.write("COMMAND: " + " ".join(cmd) + "\n")
        fp.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=fp,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
        )
        try:
            return_code = proc.wait(timeout=max(1.0, float(args.case_timeout_sec)))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=float(args.kill_grace_sec))
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            return_code = 124
            fp.write(f"\n[PARENT-TIMEOUT] killed process group after {args.case_timeout_sec}s\n")
            fp.flush()

    if result_json.exists():
        res = json.loads(result_json.read_text(encoding="utf-8"))
        if timed_out and not res.get("timeout_budget_exhausted"):
            res["timeout_budget_exhausted"] = True
            res["parent_return_code"] = return_code
            res["supervisor_log"] = str(supervisor_log)
            result_json.write_text(json.dumps(base.json_safe_record(res), ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        return base.json_safe_record(res)

    return finalize_case_result(
        args,
        benchmark,
        row,
        idx,
        total,
        result_root,
        runtime_sec=time.time() - start,
        timed_out=timed_out,
        error=f"child_return_code={return_code}; see supervisor_log={supervisor_log}",
    )


def run_benchmark(args, benchmark: str):
    mod, tasks = base.collect_tasks(benchmark)
    if args.max_cases and args.max_cases > 0:
        tasks = tasks.head(args.max_cases).copy()
    result_root = Path(args.results_root) / benchmark
    result_root.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(result_root / f"{benchmark}_selected_tasks.csv", index=False, encoding="utf-8-sig")
    results_csv = result_root / f"all_{benchmark}_official_llm_sr_results.csv"
    rows = []
    total = len(tasks)
    for idx, (_, row_series) in enumerate(tasks.iterrows(), start=1):
        row = row_series.to_dict()
        print(f"[official-llm-sr {benchmark} {idx}/{total}] {base.row_case_name(benchmark, row)}", flush=True)
        if args.isolate_cases:
            res = run_official_case_isolated(args, benchmark, row, idx, total, result_root)
        else:
            res = run_official_case(args, benchmark, mod, row, idx, total, result_root)
        rows.append(res)
        base.print_one_result(benchmark, idx, total, row, res)
        pd.DataFrame([base.json_safe_record(r) for r in rows]).to_csv(results_csv, index=False)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", default="llmsrbench,srbench,srsd")
    parser.add_argument(
        "--results-root",
        default=str(PROJECT_ROOT / "runs" / "official_llm_sr_equal_v11_600s"),
    )
    parser.add_argument("--api-base", default=os.environ.get("OFFICIAL_LLMSR_API_BASE", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OFFICIAL_LLMSR_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("OFFICIAL_LLMSR_MODEL", "llm-baseline-qwen2.5-32b"))
    parser.add_argument("--case-timeout-sec", type=float, default=600.0)
    parser.add_argument("--evaluate-timeout-sec", type=int, default=30)
    parser.add_argument("--request-timeout-sec", type=float, default=180.0)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-sample-nums", type=int, default=100000)
    parser.add_argument("--max-params", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--official-commit", default=os.environ.get("OFFICIAL_LLMSR_COMMIT", "unknown"))
    parser.add_argument("--isolate-cases", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kill-grace-sec", type=float, default=20.0)
    parser.add_argument("--single-case-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--disable-inprocess-alarm", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.results_root).mkdir(parents=True, exist_ok=True)
    (Path(args.results_root) / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if args.single_case_index is not None:
        if len(benchmarks) != 1:
            raise ValueError("--single-case-index requires exactly one benchmark")
        benchmark = benchmarks[0]
        mod, tasks = base.collect_tasks(benchmark)
        if args.max_cases and args.max_cases > 0:
            tasks = tasks.head(args.max_cases).copy()
        if args.single_case_index < 1 or args.single_case_index > len(tasks):
            raise IndexError(f"single case index out of range: {args.single_case_index} / {len(tasks)}")
        row = tasks.iloc[args.single_case_index - 1].to_dict()
        result_root = Path(args.results_root) / benchmark
        result_root.mkdir(parents=True, exist_ok=True)
        run_official_case(args, benchmark, mod, row, args.single_case_index, len(tasks), result_root)
        return

    for benchmark in benchmarks:
        run_benchmark(args, benchmark)


if __name__ == "__main__":
    main()
