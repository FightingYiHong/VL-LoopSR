#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run main-experiment SR baselines on the controlled noise suite."""

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
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from run_cpu_baseline_benchmarks import METHOD_FITTERS, dataframe_to_xy  # noqa: E402
from run_physo_baseline import fit_physo  # noqa: E402
from benchmark_metrics import (  # noqa: E402
    NUMERICAL_FIT_R2_THRESHOLD,
    regression_metrics,
    strict_formula_recovery,
)
from run_v11_noise_robustness import (  # noqa: E402
    EXACT_CLEAN_MSE_THRESHOLD,
    PASS_MSE_THRESHOLD,
    SKELETON_CLEAN_MSE_THRESHOLD,
    expr_complexity,
    finite_mse,
    formula_fn,
    case_repeat_seed,
    load_fixed_split,
    load_manifest_cases,
    make_cases,
    make_split,
    parse_benchmark_filter,
    parse_noise_levels,
    residual_structure_score,
    sanitize_name,
)


DEFAULT_CONFIGS = {
    "gplearn": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "gplearn_100s_1thread.yaml",
    "pyoperon": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "pyoperon_100s_1thread.yaml",
    "pysr": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "pysr_100s_1thread.yaml",
    "deap": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "deap_100s.yaml",
    "itea": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "itea_100s.yaml",
    "aifeynman": ROOT / "evaluation_suites" / "cpu_symbolic_regression_fourbench" / "configs" / "aifeynman_100s.yaml",
    "psrn": ROOT / "evaluation_suites" / "gpu_symbolic_regression_fourbench" / "configs" / "psrn_100s_safe.yaml",
    "psrn_pse": ROOT / "evaluation_suites" / "gpu_symbolic_regression_fourbench" / "configs" / "psrn_100s_safe.yaml",
    "dso": ROOT / "evaluation_suites" / "gpu_symbolic_regression_fourbench" / "configs" / "dso_limited.yaml",
}
GPU_METHOD_ALIASES = {"psrn_pse": "psrn"}
EXTRA_METHODS = {"physo", "llm_direct", "official_llm_sr"}
SUPPORTED_METHODS = set(METHOD_FITTERS) | set(GPU_METHOD_ALIASES) | EXTRA_METHODS
DEFAULT_LLM_API_BASE = os.environ.get("NOISE_BASELINE_LLM_API_BASE", "http://127.0.0.1:8001/v1")
DEFAULT_LLM_MODEL = os.environ.get("NOISE_BASELINE_LLM_MODEL", "")
DEFAULT_OFFICIAL_LLMSR_ROOT = Path(
    os.environ.get("OFFICIAL_LLMSR_ROOT", str(ROOT / "data" / "external" / "methods" / "LLM-SR"))
)


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def parse_methods(text: str) -> list[str]:
    methods = [item.strip() for item in str(text or "").split(",") if item.strip()]
    if not methods:
        raise ValueError("at least one baseline method is required")
    unknown = [m for m in methods if m not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(f"unsupported methods: {unknown}; available={sorted(SUPPORTED_METHODS)}")
    return methods


def method_config(method: str, override_text: str | None = None) -> Path:
    overrides = {}
    for item in str(override_text or "").split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"config override must be method=path, got: {item}")
        name, path = item.split("=", 1)
        overrides[name.strip()] = Path(path.strip())
    return Path(overrides.get(method) or DEFAULT_CONFIGS.get(method) or "")


def manifest_config(method: str, override_text: str | None = None) -> str:
    if method in EXTRA_METHODS:
        return ""
    return str(method_config(method, override_text))


def compact_number(value) -> str:
    try:
        value = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6g}"


def sample_prompt_rows(train_df: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(train_df) <= max_rows:
        return train_df.copy().reset_index(drop=True)
    return train_df.sample(n=int(max_rows), random_state=int(random_state)).reset_index(drop=True)


def dataframe_table(df: pd.DataFrame, feature_names: list[str]) -> str:
    cols = list(feature_names) + ["y"]
    lines = [", ".join(cols)]
    for _, row in df[cols].iterrows():
        lines.append(", ".join(compact_number(row[c]) for c in cols))
    return "\n".join(lines)


def make_llm_direct_prompt(case, train_df: pd.DataFrame, max_rows: int, random_state: int, feedback: str | None = None) -> str:
    sampled = sample_prompt_rows(train_df, max_rows, random_state)
    parts = [
        "You are a symbolic regression baseline.",
        "Infer one compact mathematical expression for target y from noisy observations.",
        "Use only these variables exactly as written: " + ", ".join(case.feature_names),
        "Allowed operators/functions: +, -, *, /, **, sin, cos, tan, exp, log, sqrt, Abs.",
        "Avoid prose. Return exactly one line in the form: y = <expression>",
        f"Case: {case.case_name}",
        f"Noise level in train/validation target: {case.noise_level:g}",
        "Sampled training rows:",
        dataframe_table(sampled, case.feature_names),
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


def strip_llm_candidate(text: str) -> str:
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
        line = line.replace("^", "**").replace("ln(", "log(")
        if line:
            return line
    return text.replace("^", "**")


def metric_dict_from_expr(method: str, expr: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_names: list[str], fit_sec: float, num_candidates: int | None):
    import run_cpu_baseline_benchmarks as cpu_sr

    pred_train = evaluate_expression(expr, train_df, feature_names)
    pred_val = evaluate_expression(expr, val_df, feature_names)
    pred_test = evaluate_expression(expr, test_df, feature_names)
    return cpu_sr._metric_dict(
        method,
        expr,
        pred_train,
        pred_val,
        pred_test,
        train_df["y"].to_numpy(dtype=float),
        val_df["y"].to_numpy(dtype=float),
        test_df["y"].to_numpy(dtype=float),
        feature_names,
        fit_sec,
        num_candidates=num_candidates,
    )


def fit_llm_direct_noise(case, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, args, seed: int) -> dict:
    from openai import OpenAI

    client = OpenAI(base_url=args.api_base, api_key=args.api_key)
    model = args.model
    if not model:
        models = client.models.list()
        if not models.data:
            raise RuntimeError("vLLM returned no models")
        model = models.data[0].id

    started = time.time()
    deadline = started + max(1.0, float(args.case_budget_sec))
    best = None
    feedback = None
    traces = []
    attempts = 0
    while attempts < int(args.llm_max_attempts) and time.time() < deadline:
        attempts += 1
        prompt = make_llm_direct_prompt(case, train_df, int(args.llm_sample_rows), seed + attempts, feedback)
        request_timeout = max(5.0, min(float(args.request_timeout_sec), deadline - time.time()))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You produce valid symbolic-regression formulas and no extra explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=float(args.temperature),
            max_tokens=int(args.max_tokens),
            timeout=request_timeout,
        )
        raw = response.choices[0].message.content or ""
        expr = strip_llm_candidate(raw)
        try:
            result = metric_dict_from_expr(
                "llm_direct",
                expr,
                train_df,
                val_df,
                test_df,
                case.feature_names,
                time.time() - started,
                attempts,
            )
            traces.append({"attempt": attempts, "expr": expr, "val_mse": result.get("best_val_mse"), "error": None})
            if best is None or (
                result.get("best_val_mse") is not None
                and (best.get("best_val_mse") is None or float(result["best_val_mse"]) < float(best["best_val_mse"]))
            ):
                best = result
            feedback = (
                f"Attempt {attempts}: expression `{expr}` produced validation MSE="
                f"{compact_number(result.get('best_val_mse'))}. Try a simpler or more accurate formula."
            )
        except Exception as exc:
            traces.append({"attempt": attempts, "expr": expr, "val_mse": None, "error": repr(exc)})
            feedback = f"Attempt {attempts}: expression `{expr}` could not be evaluated: {repr(exc)}."
    if best is None:
        raise RuntimeError("llm_direct produced no valid formula")
    best.update(
        {
            "method": "llm_direct",
            "fit_runtime_sec": float(time.time() - started),
            "num_candidate_exprs": int(attempts),
            "llm_model": str(model),
            "llm_api_base": str(args.api_base),
            "llm_attempt_trace": json.dumps(json_safe(traces), ensure_ascii=False),
        }
    )
    return best


def fit_official_llmsr_noise(case, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, args) -> dict:
    os.environ["OFFICIAL_LLMSR_ROOT"] = str(args.llmsr_root)
    from evaluation_suites.official_llmsr_fourbench import run_official_llmsr_fourbench as llmsr

    started = time.time()
    row = {"case_name": case.case_name, "n_features": len(case.feature_names)}
    train_norm, val_norm, test_norm, norm_info = llmsr.normalize_splits(train_df, val_df, test_df)
    spec = llmsr.make_case_spec("noise_robustness", row, train_norm, max_params=int(args.max_params))
    X_train, y_train, _ = dataframe_to_xy(train_norm)
    dataset = {"data": {"inputs": X_train, "outputs": y_train}}

    out_dir = Path(args.out_dir)
    safe = sanitize_name(case.case_name)
    official_log_dir = out_dir / "official_logs" / "official_llm_sr" / f"{case.case_index:03d}_{safe}_seed{args.repeat_seed}"
    spec_dir = out_dir / "specs" / "official_llm_sr"
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / f"{case.case_index:03d}_{safe}_seed{args.repeat_seed}.py"
    spec_path.write_text(spec, encoding="utf-8")

    llmsr.OfficialVLLMChatLLM.api_base = args.api_base
    llmsr.OfficialVLLMChatLLM.api_key = args.api_key
    llmsr.OfficialVLLMChatLLM.model = args.model
    llmsr.OfficialVLLMChatLLM.temperature = float(args.temperature)
    llmsr.OfficialVLLMChatLLM.max_tokens = int(args.max_tokens)
    llmsr.OfficialVLLMChatLLM.request_timeout_sec = float(args.request_timeout_sec)

    cfg = llmsr.official_config.Config(
        num_samplers=1,
        num_evaluators=1,
        samples_per_prompt=int(args.samples_per_prompt),
        evaluate_timeout_seconds=int(args.evaluate_timeout_sec),
        use_api=False,
    )
    classes = llmsr.official_config.ClassConfig(
        llm_class=llmsr.OfficialVLLMChatLLM,
        sandbox_class=llmsr.official_evaluator.LocalSandbox,
    )

    import signal

    timed_out = False
    old_handler = signal.getsignal(signal.SIGALRM)

    def _handler(signum, frame):
        raise TimeoutError(f"official LLM-SR case exceeded {args.case_budget_sec:.1f}s")

    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, max(1.0, float(args.case_budget_sec)))
    try:
        try:
            llmsr.official_pipeline.main(
                specification=spec,
                inputs=dataset,
                config=cfg,
                max_sample_nums=int(args.max_sample_nums),
                class_config=classes,
                log_dir=str(official_log_dir),
            )
        except TimeoutError:
            timed_out = True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)

    best = llmsr.load_best_sample(official_log_dir / "samples")
    if best is None:
        raise RuntimeError("no valid official LLM-SR sample")
    result = llmsr.metrics_from_best(
        best,
        train_norm,
        val_norm,
        test_norm,
        train_df,
        val_df,
        test_df,
        norm_info,
        max_params=int(args.max_params),
        runtime_sec=time.time() - started,
    )
    result.update(
        {
            "method": "official_llm_sr",
            "fit_runtime_sec": float(time.time() - started),
            "timed_out": bool(timed_out),
            "official_log_dir": str(official_log_dir),
            "spec_path": str(spec_path),
            "llm_model": str(args.model),
            "llm_api_base": str(args.api_base),
        }
    )
    return result


def physo_args(args, seed: int):
    ops = [item.strip() for item in str(args.physo_op_names).split(",") if item.strip()]
    return SimpleNamespace(
        max_train_rows=int(args.physo_max_train_rows),
        random_state=int(seed),
        epochs=int(args.physo_epochs),
        stop_after_n_epochs=int(args.physo_stop_after_n_epochs),
        max_evaluations=int(args.physo_max_evaluations),
        stop_reward=float(args.physo_stop_reward),
        op_names=ops,
        device=str(args.physo_device),
    )


def fit_method(method: str, case, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, args, seed: int) -> dict:
    if method == "llm_direct":
        return fit_llm_direct_noise(case, train_df, val_df, test_df, args, seed)
    if method == "official_llm_sr":
        return fit_official_llmsr_noise(case, train_df, val_df, test_df, args)
    if method == "physo":
        result = fit_physo(train_df, val_df, test_df, physo_args(args, seed))
        result["method"] = "physo"
        return result
    actual = GPU_METHOD_ALIASES.get(method, method)
    config_path = method_config(method, args.configs)
    if not config_path.exists():
        raise FileNotFoundError(f"config not found for {method}: {config_path}")
    result = METHOD_FITTERS[actual](train_df, val_df, test_df, str(config_path), random_state=seed + case.case_index)
    result["method"] = method
    return result


def variables_used(expr: str | None, feature_names: list[str]) -> list[str]:
    if not expr:
        return []
    text = str(expr)
    used = []
    for idx, name in enumerate(feature_names):
        aliases = {str(name), f"x{idx}", f"X{idx}", f"ARG{idx}"}
        if any(re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases):
            used.append(str(name))
    return used


def family_set(expr: str | None) -> set[str]:
    text = str(expr or "").lower()
    fams = set()
    if any(fn in text for fn in ["sin", "cos", "tan", "tanh"]):
        fams.add("trigonometric")
    if "exp" in text:
        fams.add("exp")
    if "log" in text:
        fams.add("log")
    if "/" in text or "div" in text:
        fams.add("rational")
    if "*" in text or "mul" in text:
        fams.add("interaction")
    if "**" in text or "^" in text or "sqrt" in text:
        fams.add("power")
    if any(op in text for op in ["+", "-", "*", "add", "sub", "mul"]):
        fams.add("algebraic")
    if not fams:
        fams.add("algebraic")
    return fams


def expected_families(structure_type: str) -> set[str]:
    text = str(structure_type or "").lower()
    fams = set()
    if any(x in text for x in ["trig", "sin", "cos"]):
        fams.add("trigonometric")
    if "exp" in text:
        fams.add("exp")
    if "log" in text:
        fams.add("log")
    if any(x in text for x in ["rational", "division"]):
        fams.add("rational")
    if any(x in text for x in ["interaction", "composition", "mixed"]):
        fams.add("interaction")
    if "poly" in text:
        fams.update({"power", "interaction", "algebraic"})
    return fams or {"algebraic"}


def family_score(expr: str | None, structure_type: str) -> float:
    got = family_set(expr)
    want = expected_families(structure_type)
    if "algebraic" in want and got:
        return 1.0 if got & want else 0.5
    if not got and not want:
        return 1.0
    union = got | want
    return float(len(got & want) / max(1, len(union)))


def evaluate_expression(expr: str, df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    local_vars = {name: df[name].to_numpy(dtype=float) for name in feature_names}
    for idx, name in enumerate(feature_names):
        local_vars[f"x{idx}"] = df[name].to_numpy(dtype=float)
        local_vars[f"X{idx}"] = df[name].to_numpy(dtype=float)
    local_vars.update(
        {
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "tanh": np.tanh,
            "exp": lambda x: np.exp(np.clip(x, -30, 30)),
            "log": lambda x: np.log(np.abs(x) + 1.0e-12),
            "sqrt": lambda x: np.sqrt(np.abs(x)),
            "abs": np.abs,
            "Abs": np.abs,
            "max": np.maximum,
            "min": np.minimum,
            "maximum": np.maximum,
            "minimum": np.minimum,
            "id": lambda x: x,
            "pi": np.pi,
            "E": np.e,
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            "div": lambda x, y: np.where(np.abs(y) > 1.0e-12, x / y, 1.0),
        }
    )
    with np.errstate(all="ignore"):
        pred = eval(str(expr).replace("^", "**"), {"__builtins__": {}}, local_vars)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 0:
        pred = np.full(len(df), float(pred))
    return pred.reshape(-1)


def clean_eval_metrics(case, result: dict, clean: dict) -> dict:
    expr = result.get("best_expr") or result.get("best_expr_sympy")
    found_complexity = result.get("expr_complexity") or expr_complexity(expr)
    true_complexity = expr_complexity(case.true_expression)
    active = variables_used(expr, case.feature_names)
    active_set = set(active)
    true_set = set(case.true_variables)
    false_count = len(active_set - true_set)
    recall = len(active_set & true_set) / max(1, len(true_set))
    precision = len(active_set & true_set) / max(1, len(active_set))
    fdr = false_count / max(1, len(active_set))
    form_score = family_score(expr, case.structure_type)

    clean_metrics = {
        split: {name: None for name in ("mse", "rmse", "nmse", "mae", "r2")}
        for split in ("train", "val", "test")
    }
    residual_score = None
    residual_structured = None
    if expr:
        try:
            train_pred = evaluate_expression(expr, clean["train"], case.feature_names)
            val_pred = evaluate_expression(expr, clean["val"], case.feature_names)
            test_pred = evaluate_expression(expr, clean["test"], case.feature_names)
            clean_metrics["train"] = regression_metrics(clean["train"]["y"].to_numpy(), train_pred)
            clean_metrics["val"] = regression_metrics(clean["val"]["y"].to_numpy(), val_pred)
            clean_metrics["test"] = regression_metrics(clean["test"]["y"].to_numpy(), test_pred)
            clean_test_mse = clean_metrics["test"]["mse"]
            residual = clean["test"]["y"].to_numpy(dtype=float) - np.asarray(test_pred, dtype=float)
            residual_score = residual_structure_score(residual, clean["test"], case.feature_names)
            residual_structured = bool(
                residual_score is not None
                and residual_score >= 0.35
                and (clean_test_mse is None or clean_test_mse > SKELETON_CLEAN_MSE_THRESHOLD)
            )
        except Exception as exc:
            result["clean_eval_error"] = repr(exc)

    clean_train_mse = clean_metrics["train"]["mse"]
    clean_val_mse = clean_metrics["val"]["mse"]
    clean_test_mse = clean_metrics["test"]["mse"]
    clean_mse_for_pass = float(clean_test_mse) if clean_test_mse is not None else float("inf")
    skeleton = bool(
        clean_mse_for_pass <= SKELETON_CLEAN_MSE_THRESHOLD
        and recall >= 1.0
        and fdr <= 0.25
        and form_score >= 0.35
    )
    exact_proxy = bool(clean_mse_for_pass <= EXACT_CLEAN_MSE_THRESHOLD and active_set == true_set and form_score >= 0.75)
    strict_recovery = strict_formula_recovery(expr, case.true_expression, case.feature_names)
    clean_test_r2 = clean_metrics["test"]["r2"]
    complexity_bloat = None
    if true_complexity and found_complexity is not None:
        complexity_bloat = float(found_complexity) / max(1.0, float(true_complexity))

    return {
        "active_variables": "|".join(active),
        "true_variables": "|".join(case.true_variables),
        "true_variable_recall": recall,
        "true_variable_precision": precision,
        "false_variable_discovery_rate": fdr,
        "wrong_variable_count": false_count,
        "formula_form_score": form_score,
        "skeleton_recovery": skeleton,
        "exact_recovery": bool(strict_recovery),
        "strict_formula_recovery": bool(strict_recovery),
        "strict_formula_recovery_evaluable": strict_recovery is not None,
        "exact_recovery_proxy": exact_proxy,
        "numerical_complete_fit": bool(
            clean_test_r2 is not None and clean_test_r2 > NUMERICAL_FIT_R2_THRESHOLD
        ),
        "passed": bool(clean_mse_for_pass <= PASS_MSE_THRESHOLD),
        "pass_at_100": bool(clean_mse_for_pass <= 100.0),
        "pass_at_300": bool(clean_mse_for_pass <= 300.0),
        "clean_train_mse": clean_train_mse,
        "clean_val_mse": clean_val_mse,
        "clean_test_mse": clean_test_mse,
        "clean_train_rmse": clean_metrics["train"]["rmse"],
        "clean_val_rmse": clean_metrics["val"]["rmse"],
        "clean_test_rmse": clean_metrics["test"]["rmse"],
        "clean_train_nmse": clean_metrics["train"]["nmse"],
        "clean_val_nmse": clean_metrics["val"]["nmse"],
        "clean_test_nmse": clean_metrics["test"]["nmse"],
        "clean_train_r2": clean_metrics["train"]["r2"],
        "clean_val_r2": clean_metrics["val"]["r2"],
        "clean_test_r2": clean_test_r2,
        "mse_at_pass": clean_test_mse if clean_mse_for_pass <= PASS_MSE_THRESHOLD else None,
        "residual_structure_score": residual_score,
        "residual_structured": residual_structured,
        "expr_complexity": found_complexity,
        "true_expr_complexity": true_complexity,
        "complexity_bloat": complexity_bloat,
    }


def run_child(args) -> int:
    started = time.time()
    result_path = Path(args.single_result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cases = (
        load_manifest_cases(Path(args.dataset_manifest), args.max_cases, parse_benchmark_filter(args.benchmarks))
        if args.dataset_manifest
        else make_cases(parse_noise_levels(args.noise_levels), args.max_cases, parse_benchmark_filter(args.benchmarks))
    )
    case = cases[int(args.case_index) - 1]
    try:
        seed = args.random_state + args.repeat_seed * 1_000_000
        train_df, val_df, test_df, clean = load_fixed_split(case) if args.dataset_manifest else make_split(case, seed, args.n_train, args.n_val, args.n_test)
        result = fit_method(args.method, case, train_df, val_df, test_df, args, seed)
        config_path = method_config(args.method, args.configs) if args.method not in EXTRA_METHODS else ""
        out = {
            **result,
            "method": args.method,
            "case_index": int(case.case_index),
            "case_name": case.case_name,
            "base_case_name": case.base_case_name,
            "benchmark": case.benchmark,
            "structure_type": case.structure_type,
            "noise_level": float(case.noise_level),
            "noise_kind": "relative_gaussian_y_train_val",
            "repeat_seed": case_repeat_seed(case, args.repeat_seed),
            "budget_sec": float(args.case_budget_sec),
            "timed_out": False,
            "runtime_sec": float(result.get("fit_runtime_sec") or (time.time() - started)),
            "true_expression_for_scoring": case.true_expression,
            "feature_names": "|".join(case.feature_names),
            "config_path": str(config_path),
            "dataset_manifest": str(args.dataset_manifest or ""),
            "best_noisy_val_mse": result.get("best_val_mse"),
            "best_clean_test_mse_from_baseline": result.get("best_test_mse"),
        }
        out.update(clean_eval_metrics(case, out, clean))
    except BaseException as exc:
        out = {
            "method": args.method,
            "case_index": int(args.case_index),
            "case_name": case.case_name,
            "base_case_name": case.base_case_name,
            "benchmark": case.benchmark,
            "structure_type": case.structure_type,
            "noise_level": float(case.noise_level),
            "repeat_seed": case_repeat_seed(case, args.repeat_seed),
            "budget_sec": float(args.case_budget_sec),
            "timed_out": "time" in repr(exc).lower() or "timeout" in repr(exc).lower(),
            "runtime_sec": float(time.time() - started),
            "valid_formula_found": False,
            "passed": False,
            "pass_at_100": False,
            "pass_at_300": False,
            "best_expr": None,
            "best_val_mse": None,
            "best_test_mse": None,
            "clean_test_mse": None,
            "mse_at_pass": None,
            "true_expression_for_scoring": case.true_expression,
            "true_variables": "|".join(case.true_variables),
            "active_variables": "",
            "true_variable_recall": 0.0,
            "true_variable_precision": 0.0,
            "false_variable_discovery_rate": 0.0,
            "wrong_variable_count": 0,
            "formula_form_score": 0.0,
            "skeleton_recovery": False,
            "exact_recovery": False,
            "exact_recovery_proxy": False,
            "residual_structured": None,
            "error": repr(exc),
        }
    result_path.write_text(json.dumps(json_safe(out), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def timeout_row(args, case, repeat_seed: int, method: str, runtime_sec: float, log_path: Path, parent_timeout_sec: float):
    return {
        "method": method,
        "case_index": int(case.case_index),
        "case_name": case.case_name,
        "base_case_name": case.base_case_name,
        "benchmark": case.benchmark,
        "structure_type": case.structure_type,
        "noise_level": float(case.noise_level),
        "repeat_seed": case_repeat_seed(case, repeat_seed),
        "budget_sec": float(args.case_budget_sec),
        "timed_out": True,
        "runtime_sec": float(runtime_sec),
        "valid_formula_found": False,
        "passed": False,
        "pass_at_100": False,
        "pass_at_300": False,
        "best_expr": None,
        "best_val_mse": None,
        "best_test_mse": None,
        "clean_test_mse": None,
        "mse_at_pass": None,
        "true_expression_for_scoring": case.true_expression,
        "true_variables": "|".join(case.true_variables),
        "active_variables": "",
        "true_variable_recall": 0.0,
        "true_variable_precision": 0.0,
        "false_variable_discovery_rate": 0.0,
        "wrong_variable_count": 0,
        "formula_form_score": 0.0,
        "skeleton_recovery": False,
        "exact_recovery": False,
        "exact_recovery_proxy": False,
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
        "pass_at_100",
        "pass_at_300",
        "exact_recovery",
        "strict_formula_recovery",
        "numerical_complete_fit",
        "skeleton_recovery",
        "noise_level",
        "best_val_mse",
        "clean_test_mse",
        "clean_test_rmse",
        "clean_test_nmse",
        "clean_test_r2",
        "mse_at_pass",
        "runtime_sec",
        "expr_complexity",
        "complexity_bloat",
        "residual_structured",
    ]:
        if column not in df:
            df[column] = np.nan
    df.to_csv(out_dir / "all_noise_robustness_baseline_results.csv", index=False)
    summary = (
        df.groupby(["noise_level", "benchmark", "method"], dropna=False)
        .agg(
            n=("case_index", "count"),
            timeout_rate=("timed_out", "mean"),
            pass_at_100=("pass_at_100", "mean"),
            pass_at_300=("pass_at_300", "mean"),
            exact_recovery=("exact_recovery", "mean"),
            numerical_complete_fit=("numerical_complete_fit", "mean"),
            skeleton_recovery=("skeleton_recovery", "mean"),
            median_clean_test_mse=("clean_test_mse", "median"),
            median_clean_test_rmse=("clean_test_rmse", "median"),
            median_clean_test_nmse=("clean_test_nmse", "median"),
            median_noisy_val_mse=("best_val_mse", "median"),
            median_complexity=("expr_complexity", "median"),
            median_complexity_bloat=("complexity_bloat", "median"),
            residual_structured_rate=("residual_structured", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary_noise_robustness_baselines.csv", index=False)
    by_noise = (
        df.groupby(["noise_level", "method"], dropna=False)
        .agg(
            n=("case_index", "count"),
            timeout_rate=("timed_out", "mean"),
            pass_at_100=("pass_at_100", "mean"),
            pass_at_300=("pass_at_300", "mean"),
            exact_recovery=("exact_recovery", "mean"),
            numerical_complete_fit=("numerical_complete_fit", "mean"),
            skeleton_recovery=("skeleton_recovery", "mean"),
            median_clean_test_mse=("clean_test_mse", "median"),
            median_clean_test_rmse=("clean_test_rmse", "median"),
            median_clean_test_nmse=("clean_test_nmse", "median"),
            median_complexity=("expr_complexity", "median"),
            median_complexity_bloat=("complexity_bloat", "median"),
            residual_structured_rate=("residual_structured", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .reset_index()
    )
    by_noise.to_csv(out_dir / "summary_noise_robustness_baselines_by_noise.csv", index=False)
    plot_figures(df, out_dir)


def plot_figures(df: pd.DataFrame, out_dir: Path):
    if plt is None:
        return
    from tools.plot_style import COLOR_NEUTRAL_DARK, palette_for, save_nature_figure, set_nature_style

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_nature_style(plt)
    method_values = sorted(df["method"].dropna().unique())
    method_palette = palette_for(method_values)

    plt.figure(figsize=(8.4, 4.7))
    for method, sub in df.groupby("method", dropna=False):
        grouped = sub.groupby("noise_level")["skeleton_recovery"].mean().reset_index()
        plt.plot(grouped["noise_level"], grouped["skeleton_recovery"], marker="o", color=method_palette.get(method), label=str(method))
    plt.xscale("symlog", linthresh=0.001)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Relative target noise")
    plt.ylabel("Skeleton Recovery")
    plt.legend(title="method")
    plt.tight_layout()
    save_nature_figure(plt, fig_dir / "fig5a_baseline_noise_structural_stability.png", "noise_robustness_baselines", dpi=180)
    save_nature_figure(plt, fig_dir / "fig5a_baseline_noise_structural_stability.pdf", "noise_robustness_baselines")
    plt.close()

    plt.figure(figsize=(8.4, 4.7))
    for method, sub in df.groupby("method", dropna=False):
        grouped = sub.groupby("noise_level")["complexity_bloat"].median().reset_index()
        plt.plot(grouped["noise_level"], grouped["complexity_bloat"], marker="o", color=method_palette.get(method), label=str(method))
    plt.xscale("symlog", linthresh=0.001)
    plt.axhline(1.0, color=COLOR_NEUTRAL_DARK, linewidth=1.0, linestyle="--")
    plt.xlabel("Relative target noise")
    plt.ylabel("Expression complexity / true complexity")
    plt.legend(title="method")
    plt.tight_layout()
    save_nature_figure(plt, fig_dir / "fig5c_baseline_complexity_inflation.png", "noise_robustness_baselines", dpi=180)
    save_nature_figure(plt, fig_dir / "fig5c_baseline_complexity_inflation.pdf", "noise_robustness_baselines")
    plt.close()


def run_parent(args) -> int:
    out_dir = Path(args.out_dir)
    result_dir = out_dir / "case_results"
    log_dir = out_dir / "case_logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    parent_timeout_sec = float(args.parent_timeout_sec or 0.0)
    if parent_timeout_sec <= 0:
        parent_timeout_sec = float(args.case_budget_sec) + float(args.timeout_grace_sec)

    methods = parse_methods(args.methods)
    noise_levels = parse_noise_levels(args.noise_levels)
    cases = (
        load_manifest_cases(Path(args.dataset_manifest), args.max_cases, parse_benchmark_filter(args.benchmarks))
        if args.dataset_manifest
        else make_cases(noise_levels, args.max_cases, parse_benchmark_filter(args.benchmarks))
    )
    repeat_seeds = 1 if args.dataset_manifest else int(args.repeat_seeds)
    selected = []
    for case in cases:
        row = asdict(case)
        row["feature_names"] = "|".join(case.feature_names)
        row["true_variables"] = "|".join(case.true_variables)
        selected.append(row)
    pd.DataFrame(selected).to_csv(out_dir / "selected_noise_robustness_cases.csv", index=False)
    manifest = {
        "experiment": "noise_robustness_baselines",
        "methods": methods,
        "n_cases": len(cases),
        "repeat_seeds": int(repeat_seeds),
        "noise_levels": noise_levels,
        "dataset_manifest": str(args.dataset_manifest or ""),
        "benchmarks": args.benchmarks or "all",
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "case_budget_sec": float(args.case_budget_sec),
        "timeout_grace_sec": float(args.timeout_grace_sec),
        "parent_timeout_sec": float(parent_timeout_sec),
        "configs": {method: manifest_config(method, args.configs) for method in methods},
        "llm_api_base": args.api_base,
        "llm_model": args.model,
        "official_llmsr_root": str(args.llmsr_root),
        "noise_protocol": "train/val y noisy, test y clean",
    }
    (out_dir / "manifest_noise_robustness_baselines.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    total = len(cases) * int(repeat_seeds) * len(methods)
    run_idx = 0
    for method in methods:
        (result_dir / method).mkdir(parents=True, exist_ok=True)
        (log_dir / method).mkdir(parents=True, exist_ok=True)
        for repeat_seed in range(int(repeat_seeds)):
            for case in cases:
                run_idx += 1
                safe = sanitize_name(case.case_name)
                out_seed = case_repeat_seed(case, repeat_seed)
                result_json = result_dir / method / f"{method}_{case.case_index:03d}_{safe}_seed{out_seed}.json"
                log_path = log_dir / method / f"{method}_{case.case_index:03d}_{safe}_seed{out_seed}.log"
                if args.resume and result_json.exists():
                    cached = json.loads(result_json.read_text(encoding="utf-8"))
                    should_rerun = False
                    if args.rerun_timeouts and bool(cached.get("timed_out")):
                        should_rerun = True
                    if args.rerun_failures and not bool(cached.get("valid_formula_found")):
                        should_rerun = True
                    if not should_rerun:
                        rows.append(cached)
                        continue

                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--child",
                    "--method",
                    method,
                    "--methods",
                    args.methods,
                    "--configs",
                    args.configs or "",
                    "--out-dir",
                    str(out_dir),
                    "--case-index",
                    str(case.case_index),
                    "--repeat-seed",
                    str(out_seed),
                    "--repeat-seeds",
                    str(repeat_seeds),
                    "--noise-levels",
                    args.noise_levels,
                    "--n-train",
                    str(args.n_train),
                    "--n-val",
                    str(args.n_val),
                    "--n-test",
                    str(args.n_test),
                    "--case-budget-sec",
                    str(args.case_budget_sec),
                    "--timeout-grace-sec",
                    str(args.timeout_grace_sec),
                    "--max-cases",
                    str(args.max_cases or 0),
                    "--random-state",
                    str(args.random_state),
                    "--single-result-json",
                    str(result_json),
                    "--api-base",
                    args.api_base,
                    "--api-key",
                    args.api_key,
                    "--model",
                    args.model,
                    "--request-timeout-sec",
                    str(args.request_timeout_sec),
                    "--temperature",
                    str(args.temperature),
                    "--max-tokens",
                    str(args.max_tokens),
                    "--llm-max-attempts",
                    str(args.llm_max_attempts),
                    "--llm-sample-rows",
                    str(args.llm_sample_rows),
                    "--llmsr-root",
                    str(args.llmsr_root),
                    "--samples-per-prompt",
                    str(args.samples_per_prompt),
                    "--evaluate-timeout-sec",
                    str(args.evaluate_timeout_sec),
                    "--max-sample-nums",
                    str(args.max_sample_nums),
                    "--max-params",
                    str(args.max_params),
                    "--physo-device",
                    args.physo_device,
                    "--physo-epochs",
                    str(args.physo_epochs),
                    "--physo-stop-after-n-epochs",
                    str(args.physo_stop_after_n_epochs),
                    "--physo-max-evaluations",
                    str(args.physo_max_evaluations),
                    "--physo-max-train-rows",
                    str(args.physo_max_train_rows),
                    "--physo-op-names",
                    args.physo_op_names,
                    "--physo-stop-reward",
                    str(args.physo_stop_reward),
                ]
                if args.dataset_manifest:
                    cmd.extend(["--dataset-manifest", str(args.dataset_manifest)])
                if args.benchmarks:
                    cmd.extend(["--benchmarks", args.benchmarks])
                started = time.time()
                print(
                    f"[RUN {run_idx}/{total}] method={method} case={case.case_name} "
                    f"seed={out_seed} noise={case.noise_level:g}",
                    flush=True,
                )
                with open(log_path, "w", encoding="utf-8") as log_fp:
                    try:
                        subprocess.run(
                            cmd,
                            stdout=log_fp,
                            stderr=subprocess.STDOUT,
                            timeout=parent_timeout_sec,
                            check=False,
                            env=os.environ.copy(),
                        )
                    except subprocess.TimeoutExpired:
                        row = timeout_row(args, case, out_seed, method, time.time() - started, log_path, parent_timeout_sec)
                        rows.append(row)
                        result_json.write_text(json.dumps(json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
                        summarize(rows, out_dir)
                        print(f"[TIMEOUT {run_idx}/{total}] method={method} sec={time.time() - started:.1f}", flush=True)
                        continue
                if result_json.exists():
                    row = json.loads(result_json.read_text(encoding="utf-8"))
                else:
                    row = {
                        **timeout_row(args, case, out_seed, method, time.time() - started, log_path, parent_timeout_sec),
                        "timed_out": False,
                        "error": f"child exited without result; see {log_path}",
                    }
                rows.append(row)
                summarize(rows, out_dir)
                print(
                    f"[DONE {run_idx}/{total}] method={method} timeout={row.get('timed_out')} "
                    f"skeleton={row.get('skeleton_recovery')} clean_mse={row.get('clean_test_mse')} "
                    f"complexity={row.get('expr_complexity')} sec={row.get('runtime_sec')}",
                    flush=True,
                )
    summarize(rows, out_dir)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--methods", default="gplearn,pyoperon,pysr,deap")
    parser.add_argument("--method", default="gplearn")
    parser.add_argument("--configs", default="")
    parser.add_argument("--case-budget-sec", type=float, default=100.0)
    parser.add_argument("--timeout-grace-sec", type=float, default=30.0)
    parser.add_argument("--parent-timeout-sec", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--noise-levels", default="0,0.001,0.01")
    parser.add_argument("--dataset-manifest", default="", help="Optional fixed NoiseRobust-SR manifest.csv to read instead of generating splits.")
    parser.add_argument("--benchmarks", default=None)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--repeat-seeds", type=int, default=1)
    parser.add_argument("--repeat-seed", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-timeouts", action="store_true")
    parser.add_argument("--rerun-failures", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--single-result-json", default="")
    parser.add_argument("--api-base", default=DEFAULT_LLM_API_BASE)
    parser.add_argument("--api-key", default=os.environ.get("NOISE_BASELINE_LLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--request-timeout-sec", type=float, default=80.0)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--llm-max-attempts", type=int, default=4)
    parser.add_argument("--llm-sample-rows", type=int, default=40)
    parser.add_argument("--llmsr-root", type=Path, default=DEFAULT_OFFICIAL_LLMSR_ROOT)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--evaluate-timeout-sec", type=int, default=20)
    parser.add_argument("--max-sample-nums", type=int, default=100000)
    parser.add_argument("--max-params", type=int, default=10)
    parser.add_argument("--physo-device", default=os.environ.get("PHYSO_DEVICE", "cuda"))
    parser.add_argument("--physo-epochs", type=int, default=int(os.environ.get("PHYSO_EPOCHS", "3")))
    parser.add_argument("--physo-stop-after-n-epochs", type=int, default=int(os.environ.get("PHYSO_STOP_AFTER_N_EPOCHS", "3")))
    parser.add_argument("--physo-max-evaluations", type=int, default=int(os.environ.get("PHYSO_MAX_EVALUATIONS", "200000")))
    parser.add_argument("--physo-max-train-rows", type=int, default=int(os.environ.get("PHYSO_MAX_TRAIN_ROWS", "1000")))
    parser.add_argument("--physo-op-names", default=os.environ.get("PHYSO_OP_NAMES", "add,sub,mul,div,sin,cos,exp,log"))
    parser.add_argument("--physo-stop-reward", type=float, default=float(os.environ.get("PHYSO_STOP_REWARD", "0.999999")))
    args = parser.parse_args()
    if args.max_cases <= 0:
        args.max_cases = None
    if args.repeat_seeds <= 0:
        args.repeat_seeds = 1
    return args


def main():
    args = parse_args()
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
