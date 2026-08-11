#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the table-driven V11 ablation experiments.

Suites:

* high_dim: legacy distractor-variable stress test.
* component_ablation: full V11 versus disabled Observer, Critic, or Proposer.
* noise: legacy target-noise robustness with clean-test structural scoring.

The child process imports V11 only after method-specific environment variables
are set, so each ablation is a real V11 configuration rather than a post-hoc
label.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_metrics import (
    NUMERICAL_FIT_R2_THRESHOLD,
    evaluate_expression,
    expression_complexity as shared_expression_complexity,
    regression_metrics,
    srbench_formula_recovery,
    strict_formula_recovery,
)

DEFAULT_V11_PATH = ROOT / "scripts" / "vega_sr.py"
DEFAULT_DATASET_ROOT = ROOT / "data" / "generated" / "balanced_interference_96"

PASS_MSE_THRESHOLD = 100.0
EXACT_MSE_THRESHOLD = 1e-8
SKELETON_MSE_THRESHOLD = 1e-4


@dataclass(frozen=True)
class MethodConfig:
    name: str
    env: dict[str, str]
    description: str


@dataclass(frozen=True)
class FormulaCase:
    case_index: int
    case_name: str
    benchmark: str
    structure_type: str
    feature_names: list[str]
    true_variables: list[str]
    true_expression: str
    fn_name: str
    suite: str
    noise_level: float = 0.0


METHODS: dict[str, MethodConfig] = {
    "full": MethodConfig(
        name="full",
        env={
            "LLMSR_V11_ENABLE_VLM_OBSERVER": "1",
            "LLMSR_V11_ENABLE_CRITIC_LOOP": "1",
            "LLMSR_V11_ENABLE_STRUCTURAL_RESCUE": "1",
            "LLMSR_V11_ENABLE_STRUCTURE_EVALUATOR": "1",
        },
        description="full V11: Observer + structure evaluator + Critic loop + structural rescue",
    ),
    "w_o_observer": MethodConfig(
        name="w_o_observer",
        env={
            "LLMSR_V11_ENABLE_VLM_OBSERVER": "0",
            "LLMSR_V11_VLM_OBSERVER_GENERATE_IMAGES": "0",
            "LLMSR_V11_OBSERVER_INPUT_MODE": "numeric_only",
        },
        description="numeric-only arm: retain deterministic numerical analysis but remove function images",
    ),
    "w_o_observer_all": MethodConfig(
        name="w_o_observer_all",
        env={
            "LLMSR_V11_ENABLE_OBSERVER": "0",
            "LLMSR_V11_ENABLE_VLM_OBSERVER": "0",
            "LLMSR_V11_VLM_OBSERVER_GENERATE_IMAGES": "0",
            "LLMSR_FORCE_RECON_IN_OBSERVE": "0",
            "LLMSR_ENABLE_HIGH_DIM_RECONSTRUCTION": "0",
            "LLMSR_ENABLE_RESIDUAL_VARIABLE_DISCOVERY": "0",
            "LLMSR_SKIP_HIGH_DIM_MM": "1",
            "LLMSR_FORCE_MM_HIGH_DIM": "0",
            "LLMSR_ENABLE_DATA_DRIVEN_FEATURE_SEEDS": "0",
            "LLMSR_HIGH_DIM_DATA_DRIVEN_BRIDGE": "0",
            "LLMSR_ENABLE_HIGH_DIM_EXACT_DENOM_PRIORITY": "0",
            "LLMSR_ENABLE_HIGH_DIM_CLEAN_MECHANISTIC_RERANK": "0",
            "LLMSR_ENABLE_HIGH_DIM_SURROGATE_ESCAPE": "0",
            "LLMSR_HIGH_DIM_UNIVERSAL_COVERAGE_RESCUE": "0",
            "LLMSR_HIGH_DIM_RESCUE_ALL_FAMILIES": "0",
            "LLMSR_ENABLE_HIGH_DIM_STRUCTURAL_RERANK": "0",
            "LLMSR_ENABLE_EVIDENCE_PRESERVING_SELECTION": "0",
            "LLMSR_V11_ENABLE_EXTRAPOLATION_OLS_CANDIDATES": "0",
            "LLMSR_V11_ENABLE_SURFACE_ANALYTIC_CANDIDATES": "0",
            "LLMSR_V11_ENABLE_SURFACE_GRID_BASIS_CANDIDATES": "0",
            "LLMSR_V11_ENABLE_SURFACE_STABLE_BASIS_CANDIDATES": "0",
            "LLMSR_V11_ENABLE_CONSTRUCTED_EXTRAPOLATION_CANDIDATES": "0",
        },
        description="disable the full Observer stage; downstream agents receive an empty observation bundle",
    ),
    "w_o_proposer": MethodConfig(
        name="w_o_proposer",
        env={"LLMSR_V11_ENABLE_PROPOSER": "0"},
        description="disable the Proposer stage; no expression candidates are emitted",
    ),
    "w_o_critic": MethodConfig(
        name="w_o_critic",
        env={"LLMSR_V11_ENABLE_CRITIC_LOOP": "0"},
        description="single-shot ablation: skip Critic/Judge/Refiner feedback rounds after initial evaluation",
    ),
    "single_shot": MethodConfig(
        name="single_shot",
        env={"LLMSR_V11_ENABLE_CRITIC_LOOP": "0"},
        description="single-shot ablation alias: initial proposal plus evaluation only, with no iterative refine loop",
    ),
    "w_o_structural_rescue": MethodConfig(
        name="w_o_structural_rescue",
        env={"LLMSR_V11_ENABLE_STRUCTURAL_RESCUE": "0"},
        description="keep Critic loop but disable structural-rescue candidate injection",
    ),
    "w_o_structure_evaluator": MethodConfig(
        name="w_o_structure_evaluator",
        env={"LLMSR_V11_ENABLE_STRUCTURE_EVALUATOR": "0"},
        description="rank candidates by fit/complexity without Observer-derived structure score",
    ),
}


def load_experiment_config(config_path: str | None) -> dict:
    """Extend the static method table from a claim-validation config file."""
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"experiment config not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key, value in dict(cfg.get("shared_env", {}) or {}).items():
        os.environ[str(key)] = str(value)
    for name, body in dict(cfg.get("method_variants", {}) or {}).items():
        body = dict(body or {})
        METHODS[str(name)] = MethodConfig(
            name=str(name),
            env={str(k): str(v) for k, v in dict(body.get("env", {}) or {}).items()},
            description=str(body.get("description", name)),
        )
    return cfg


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


def sanitize_name(text: str) -> str:
    text = str(text).replace("/", "__").replace("\\", "__")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "case"


def formula_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    formulas = {
        "feynman_trig_interaction": lambda x: np.sin(x[:, 0]) + 0.4 * x[:, 1] ** 2 + 0.2 * x[:, 0] * x[:, 1],
        "srsd_exp_log_mix": lambda x: np.exp(0.15 * x[:, 0]) + np.log(np.abs(x[:, 1]) + 1.0),
        "nguyen_poly_trig": lambda x: x[:, 0] ** 3 + x[:, 0] ** 2 + x[:, 0] + np.sin(x[:, 1]),
        "real_like_rational": lambda x: (x[:, 0] + 0.5 * x[:, 1] ** 2) / (1.0 + x[:, 2] ** 2) + 0.25 * np.cos(x[:, 3]),
    }
    return formulas[name]


def formula_specs() -> list[tuple[str, str, str, list[str], str, str]]:
    return [
        (
            "feynman_trig_interaction",
            "Feynman",
            "trig_interaction",
            ["x1", "x2"],
            "sin(x1) + 0.4*x2**2 + 0.2*x1*x2",
            "feynman_trig_interaction",
        ),
        (
            "srsd_exp_log_mix",
            "SRSD",
            "exp_log",
            ["x1", "x2"],
            "exp(0.15*x1) + log(abs(x2) + 1.0)",
            "srsd_exp_log_mix",
        ),
        (
            "nguyen_poly_trig",
            "Nguyen",
            "polynomial_trig",
            ["x1", "x2"],
            "x1**3 + x1**2 + x1 + sin(x2)",
            "nguyen_poly_trig",
        ),
        (
            "real_like_rational",
            "real_like",
            "rational_trig",
            ["x1", "x2", "x3", "x4"],
            "(x1 + 0.5*x2**2)/(1 + x3**2) + 0.25*cos(x4)",
            "real_like_rational",
        ),
    ]


def make_formula_cases(suite: str, noise_levels: list[float]) -> list[FormulaCase]:
    cases: list[FormulaCase] = []
    idx = 1
    for name, benchmark, structure_type, variables, expr, fn_name in formula_specs():
        for noise_level in noise_levels:
            suffix = f"_noise_{noise_level:g}" if suite == "noise" else ""
            cases.append(
                FormulaCase(
                    case_index=idx,
                    case_name=f"{name}{suffix}",
                    benchmark=benchmark,
                    structure_type=structure_type,
                    feature_names=variables,
                    true_variables=variables,
                    true_expression=expr,
                    fn_name=fn_name,
                    suite=suite,
                    noise_level=float(noise_level),
                )
            )
            idx += 1
    return cases


def make_formula_split(case: FormulaCase, n: int, seed: int, noisy: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.5, 2.5, size=(n, len(case.feature_names)))
    y_clean = formula_fn(case.fn_name)(x)
    y = np.array(y_clean, copy=True)
    if noisy and case.noise_level > 0:
        scale = float(np.std(y_clean))
        if not math.isfinite(scale) or scale <= 0:
            scale = max(1.0, float(np.mean(np.abs(y_clean))))
        y = y_clean + rng.normal(0.0, case.noise_level * scale, size=n)
    df = pd.DataFrame(x, columns=case.feature_names)
    df["y"] = y
    df["y_clean"] = y_clean
    return df


def load_formula_splits(case: FormulaCase, seed: int, n_train: int, n_val: int, n_test: int):
    train = make_formula_split(case, n_train, seed + case.case_index * 10_000 + 1, noisy=True)
    val = make_formula_split(case, n_val, seed + case.case_index * 10_000 + 2, noisy=True)
    test = make_formula_split(case, n_test, seed + case.case_index * 10_000 + 3, noisy=False)
    clean_test = test[case.feature_names + ["y_clean"]].rename(columns={"y_clean": "y"})
    return train[case.feature_names + ["y"]], val[case.feature_names + ["y"]], clean_test


def import_v11_module(v11_path: Path, results_root: Path):
    if not v11_path.exists():
        raise FileNotFoundError(f"V11 file not found: {v11_path}")
    spec = importlib.util.spec_from_file_location("v11_ablation_runtime", str(v11_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["v11_ablation_runtime"] = module
    spec.loader.exec_module(module)

    module.RESULTS_ROOT = str(results_root)
    module.GLOBAL_SUMMARY_CSV = str(results_root / "all_results_detailed.csv")
    module.GLOBAL_SUMMARY_JSON = str(results_root / "global_summary.json")
    module.GLOBAL_SUMMARY_CSV_COMPACT = str(results_root / "global_summary.csv")
    module.TIMING_BREAKDOWN_CSV = str(results_root / "timing_breakdown.csv")
    module.TIMING_SUMMARY_JSON = str(results_root / "timing_summary.json")
    module.PER_CASE_JSON_DIR = str(results_root / "per_case_reports")
    module.SELECTED_TASKS_CSV = str(results_root / "selected_cases.csv")
    if os.environ.get("LLMSR_MAX_RUNTIME_PER_TASK_SEC"):
        module.MAX_RUNTIME_PER_TASK_SEC = float(os.environ["LLMSR_MAX_RUNTIME_PER_TASK_SEC"])

    base = getattr(module, "_base", None)
    if base is not None:
        for name in [
            "RESULTS_ROOT",
            "GLOBAL_SUMMARY_CSV",
            "GLOBAL_SUMMARY_JSON",
            "GLOBAL_SUMMARY_CSV_COMPACT",
            "TIMING_BREAKDOWN_CSV",
            "TIMING_SUMMARY_JSON",
            "PER_CASE_JSON_DIR",
            "SELECTED_TASKS_CSV",
            "MAX_RUNTIME_PER_TASK_SEC",
        ]:
            if hasattr(module, name):
                setattr(base, name, getattr(module, name))
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "per_case_reports").mkdir(parents=True, exist_ok=True)
    return module


def expr_complexity(expr) -> float | None:
    value = shared_expression_complexity(expr).get("expr_complexity")
    return float(value) if value is not None else None


def variables_used(v11, expr: str | None, feature_names: list[str]) -> list[str]:
    if not expr:
        return []
    try:
        sig = v11._base.extract_formula_form_signature(expr, feature_names)
        used = list(sig.get("variables_used", []) or [])
    except Exception:
        used = [name for name in feature_names if re.search(rf"\b{re.escape(name)}\b", str(expr))]
    used_set = set(used)
    return [name for name in feature_names if name in used_set]


def family_signature(v11, expr: str | None, feature_names: list[str]) -> set[str]:
    if not expr:
        return set()
    try:
        sig = v11._base.extract_formula_form_signature(expr, feature_names)
    except Exception:
        sig = {}
    families = set(str(x) for x in (sig.get("families", []) or sig.get("operator_families", []) or []) if str(x))
    text = str(expr).lower()
    token_map = {
        "trigonometric": ["sin", "cos", "tan"],
        "exponential": ["exp"],
        "logarithmic": ["log"],
        "rational": ["/"],
        "power": ["**"],
        "interaction": ["*"],
    }
    for family, tokens in token_map.items():
        if any(tok in text for tok in tokens):
            families.add(family)
    return families


def score_formula_result(
    v11,
    case: FormulaCase,
    result: dict,
    train_df: pd.DataFrame | None = None,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> dict:
    best_expr = result.get("best_expr")
    active = variables_used(v11, best_expr, case.feature_names)
    active_set = set(active)
    true_set = set(case.true_variables)
    test_mse = finite_float(result.get("best_test_mse"), default=float("inf"))
    candidate_families = family_signature(v11, best_expr, case.feature_names)
    true_families = family_signature(v11, case.true_expression, case.feature_names)
    family_overlap = len(candidate_families & true_families) / max(1, len(true_families))
    exact_recovery_proxy = bool(active_set == true_set and test_mse <= EXACT_MSE_THRESHOLD)
    skeleton_recovery = bool(active_set == true_set and family_overlap >= 0.6 and test_mse <= SKELETON_MSE_THRESHOLD)
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
        "true_variable_recall": len(active_set & true_set) / max(1, len(true_set)),
        "false_variable_discovery_rate": len(active_set - true_set) / max(1, len(active_set)),
        "family_overlap": family_overlap,
        "exact_recovery": bool(strict_recovery),
        "strict_formula_recovery": bool(strict_recovery),
        "strict_formula_recovery_evaluable": strict_recovery is not None,
        "srbench_formula_recovery": bool(srbench_recovery),
        "srbench_formula_recovery_evaluable": srbench_recovery is not None,
        "exact_recovery_proxy": exact_recovery_proxy,
        "skeleton_recovery": skeleton_recovery,
        "numerical_complete_fit": bool(
            test_r2 is not None and float(test_r2) > NUMERICAL_FIT_R2_THRESHOLD
        ),
        "passed": bool(test_mse <= PASS_MSE_THRESHOLD),
        "expr_complexity": result.get("expr_complexity") or expr_complexity(best_expr),
        "normalized_complexity": (result.get("expr_complexity") or expr_complexity(best_expr) or np.nan)
        / max(1.0, expr_complexity(case.true_expression) or 1.0),
        **metric_updates,
    }


def loads_json_maybe(value, default=None):
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def extract_initial_best_row(result: dict) -> dict:
    history = loads_json_maybe(result.get("evaluation_history"), default=[])
    for item in history if isinstance(history, list) else []:
        if not isinstance(item, dict):
            continue
        if str(item.get("stage", "")) != "initial_evaluator":
            continue
        topk = item.get("topk") or []
        if isinstance(topk, list) and topk:
            row = dict(topk[0] or {})
            expr = row.get("expr") or row.get("expression")
            return {
                "best_expr": expr,
                "best_val_mse": row.get("val_mse"),
                "best_test_mse": row.get("test_mse"),
                "expr_complexity": row.get("complexity"),
            }
    return {"best_expr": None, "best_val_mse": None, "best_test_mse": None, "expr_complexity": None}


def add_initial_metrics(v11, case, out: dict, scorer) -> None:
    initial = extract_initial_best_row(out)
    initial_candidate_count = out.get("num_candidate_exprs")
    if initial_candidate_count is None:
        raw_exprs = loads_json_maybe(out.get("raw_exprs"), default=[])
        initial_candidate_count = len(raw_exprs) if isinstance(raw_exprs, list) else None

    score_input = dict(initial)
    score = scorer(v11, case, score_input)
    out["initial_best_expr"] = initial.get("best_expr")
    out["initial_best_val_mse"] = initial.get("best_val_mse")
    out["initial_best_test_mse"] = initial.get("best_test_mse")
    out["initial_candidate_count"] = initial_candidate_count
    val = finite_float(initial.get("best_val_mse"))
    out["log10_initial_best_val_mse"] = math.log10(max(val, 1e-300)) if val is not None else None
    for key, value in score.items():
        if key in {"true_variables", "proxy_variables", "nonlinear_decoy_variables"}:
            out.setdefault(key, value)
            continue
        if key == "active_variables":
            out["initial_active_variables"] = value
        elif key == "expr_complexity":
            out["initial_expr_complexity"] = value
        else:
            out[f"initial_{key}"] = value


def finite_float(value, default=None):
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def has_dataset_root(args) -> bool:
    return bool(getattr(args, "dataset_root", "")) and Path(args.dataset_root).exists()


def split_pipe(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [part for part in text.split("|") if part]


def suite_manifest(args) -> pd.DataFrame:
    root = Path(args.dataset_root)
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    df = df[df["suite"].eq(args.suite)].copy()
    target_component = str(getattr(args, "target_component", "") or "").strip()
    if target_component:
        if "target_component" not in df.columns:
            raise ValueError(f"--target-component={target_component!r} requested, but manifest has no target_component column")
        df = df[df["target_component"].astype(str).eq(target_component)].copy()
    if df.empty:
        target_note = f", target_component={target_component!r}" if target_component else ""
        raise ValueError(f"no cases for suite={args.suite!r}{target_note} in {manifest_path}")
    df = df.sort_values("suite_case_index")
    if args.max_cases:
        df = df.head(int(args.max_cases))
    return df.reset_index(drop=True)


def load_dataset_case(args):
    try:
        from run_v11_high_dimensional_interference import HighDimCase
    except ModuleNotFoundError:
        from scripts.run_v11_high_dimensional_interference import HighDimCase

    root = Path(args.dataset_root)
    df = suite_manifest(args)
    row = df.iloc[int(args.case_index) - 1]
    feature_names = split_pipe(row.get("feature_names"))
    true_variables = split_pipe(row.get("true_variables"))
    train_df = pd.read_csv(root / str(row["train_path"]))
    val_df = pd.read_csv(root / str(row["val_path"]))
    test_df = pd.read_csv(root / str(row["test_path"]))

    proxy_variables = split_pipe(row.get("proxy_variables"))
    nonlinear_decoy_variables = split_pipe(row.get("nonlinear_decoy_variables"))
    uses_high_dim_scoring = bool(
        args.suite == "high_dim"
        or proxy_variables
        or nonlinear_decoy_variables
        or int(row.get("dimension", len(feature_names)) or len(feature_names)) > len(true_variables)
    )

    if uses_high_dim_scoring:
        case = HighDimCase(
            case_index=int(row["suite_case_index"]),
            case_name=str(row["case_name"]),
            base_formula_id=str(row.get("base_formula_id") or row["fn_name"]),
            source_family=str(row.get("source_family") or row.get("benchmark") or "ablation_synthetic"),
            source_name=str(row.get("source_name") or row.get("benchmark") or "ablation_synthetic"),
            dimension=int(row["dimension"]),
            true_variable_count=int(row["true_variable_count"]),
            interference_type=str(row["interference_type"]),
            benchmark=str(row["benchmark"]),
            structure_type=str(row["structure_type"]),
            feature_names=feature_names,
            true_variables=true_variables,
            proxy_variables=proxy_variables,
            nonlinear_decoy_variables=nonlinear_decoy_variables,
            true_expression=str(row["true_expression"]),
            family_hint=str(row["structure_type"]),
            fn_name=str(row["fn_name"]),
            input_low=[-2.0] * int(row["true_variable_count"]),
            input_high=[2.0] * int(row["true_variable_count"]),
        )
    else:
        case = FormulaCase(
            case_index=int(row["suite_case_index"]),
            case_name=str(row["case_name"]),
            benchmark=str(row["benchmark"]),
            structure_type=str(row["structure_type"]),
            feature_names=feature_names,
            true_variables=true_variables,
            true_expression=str(row["true_expression"]),
            fn_name=str(row["fn_name"]),
            suite=args.suite,
            noise_level=float(row.get("noise_level", 0.0) or 0.0),
        )

    row_meta = {
        "task_type": args.suite,
        "dataset_dir": str(root),
        "difficulty": str(row["benchmark"]),
        "base_name": str(row["case_name"]),
        "suite": args.suite,
        "benchmark": str(row["benchmark"]),
        "structure_type": str(row["structure_type"]),
        "noise_level": float(row.get("noise_level", 0.0) or 0.0),
        "true_expression": None,
        "dataset_case_id": str(row["case_id"]),
        "dataset_manifest": str(root / "manifest.csv"),
        "target_component": str(row.get("target_component") or ""),
        "failure_mode": str(row.get("failure_mode") or ""),
    }
    if uses_high_dim_scoring:
        row_meta.update(
            {
                "task_type": "high_dimensional_interference" if args.suite == "high_dim" else "component_stress_ablation",
                "interference_type": str(row["interference_type"]),
                "dimension": int(row["dimension"]),
                "true_variable_count": int(row["true_variable_count"]),
                "proxy_variables": "|".join(proxy_variables),
                "nonlinear_decoy_variables": "|".join(nonlinear_decoy_variables),
            }
        )
    return case, train_df, val_df, test_df, row_meta


def load_case(args):
    if has_dataset_root(args):
        return load_dataset_case(args)

    if args.suite == "high_dim":
        from run_v11_high_dimensional_interference import make_cases, load_case_splits

        cases = make_cases(args.max_cases)
        case = cases[int(args.case_index) - 1]
        train_df, val_df, test_df = load_case_splits(
            case,
            args.random_state + args.repeat_seed * 1_000_000,
            args.n_train,
            args.n_val,
            args.n_test,
        )
        row_meta = {
            "task_type": "high_dimensional_interference",
            "dataset_dir": "high_dimensional_interference",
            "difficulty": case.benchmark,
            "base_name": case.case_name,
            "suite": "high_dim",
            "structure_type": case.structure_type,
            "interference_type": case.interference_type,
            "dimension": case.dimension,
            "true_variable_count": case.true_variable_count,
            "true_expression": None,
        }
        return case, train_df, val_df, test_df, row_meta

    noise_levels = parse_noise_levels(args.noise_levels) if args.suite == "noise" else [0.0]
    cases = make_formula_cases(args.suite, noise_levels)
    if args.max_cases:
        cases = cases[: args.max_cases]
    case = cases[int(args.case_index) - 1]
    train_df, val_df, test_df = load_formula_splits(
        case,
        args.random_state + args.repeat_seed * 1_000_000,
        args.n_train,
        args.n_val,
        args.n_test,
    )
    row_meta = {
        "task_type": args.suite,
        "dataset_dir": args.suite,
        "difficulty": case.benchmark,
        "base_name": case.case_name,
        "suite": args.suite,
        "benchmark": case.benchmark,
        "structure_type": case.structure_type,
        "noise_level": case.noise_level,
        "true_expression": None,
    }
    return case, train_df, val_df, test_df, row_meta


def run_child(args) -> int:
    started = time.time()
    result_path = Path(args.single_result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    method = METHODS[args.method]
    os.environ.update(method.env)
    os.environ["LLMSR_MAX_RUNTIME_PER_TASK_SEC"] = str(args.case_budget_sec)
    try:
        case, train_df, val_df, test_df, row_meta = load_case(args)
        v11 = import_v11_module(Path(args.v11_path), Path(args.out_dir) / "v11_internal" / args.method)
        with TemporaryDirectory(prefix="v11_ablation_tmp_") as tmpdir_str:
            dataset = v11.build_dataset_from_explicit_splits(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                tmpdir=Path(tmpdir_str),
            )
            dataset.source_tag = row_meta["dataset_dir"]
            result = v11._run_core_pipeline(dataset=dataset, row_meta=row_meta)
        out = {
            **result,
            **row_meta,
            "method": args.method,
            "method_description": method.description,
            "case_index": int(args.case_index),
            "case_name": getattr(case, "case_name"),
            "repeat_seed": int(args.repeat_seed),
            "budget_sec": float(args.case_budget_sec),
            "timed_out": False,
            "runtime_sec": float(result.get("runtime_sec") or (time.time() - started)),
            "ablation_env": json.dumps(method.env, sort_keys=True),
        }
        if hasattr(case, "proxy_variables"):
            try:
                from run_v11_high_dimensional_interference import score_result
            except ModuleNotFoundError:
                from scripts.run_v11_high_dimensional_interference import score_result

            out["true_expression_for_scoring"] = getattr(case, "true_expression")
            out.update(score_result(v11, case, out, train_df, val_df, test_df))
            add_initial_metrics(v11, case, out, score_result)
        else:
            out["true_expression_for_scoring"] = getattr(case, "true_expression")
            out.update(score_formula_result(v11, case, out, train_df, val_df, test_df))
            add_initial_metrics(v11, case, out, score_formula_result)
        best_test = finite_float(out.get("best_test_mse"))
        out["log10_best_test_mse"] = math.log10(max(best_test, 1e-300)) if best_test is not None else None
    except BaseException as exc:
        out = {
            "method": args.method,
            "suite": args.suite,
            "case_index": int(args.case_index),
            "repeat_seed": int(args.repeat_seed),
            "budget_sec": float(args.case_budget_sec),
            "timed_out": "time" in repr(exc).lower() or "timeout" in repr(exc).lower(),
            "runtime_sec": float(time.time() - started),
            "valid_formula_found": False,
            "passed": False,
            "exact_recovery": False,
            "skeleton_recovery": False,
            "best_expr": None,
            "best_val_mse": None,
            "best_test_mse": None,
            "error": repr(exc),
        }
    result_path.write_text(json.dumps(json_safe(out), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def timeout_row(args, method: str, case_index: int, case_name: str, runtime_sec: float, log_path: Path, parent_timeout_sec: float):
    return {
        "method": method,
        "suite": args.suite,
        "case_index": int(case_index),
        "case_name": case_name,
        "repeat_seed": int(args.repeat_seed),
        "budget_sec": float(args.case_budget_sec),
        "timed_out": True,
        "runtime_sec": float(runtime_sec),
        "valid_formula_found": False,
        "passed": False,
        "exact_recovery": False,
        "skeleton_recovery": False,
        "best_expr": None,
        "best_val_mse": None,
        "best_test_mse": None,
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
        "srbench_formula_recovery",
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
        "test_nrmse",
        "test_r2",
        "expr_complexity",
        "normalized_complexity",
        "runtime_sec",
        "noise_level",
    ]:
        if column not in df:
            df[column] = np.nan
    df.to_csv(out_dir / "all_v11_ablation_results.csv", index=False)
    group_cols = ["suite", "method"]
    if df["noise_level"].notna().any():
        noise_summary = (
            df[df["suite"].eq("noise")]
            .groupby(["method", "noise_level"], dropna=False)
            .agg(
                n=("case_index", "count"),
                pass_rate=("passed", "mean"),
                exact_recovery=("exact_recovery", "mean"),
                strict_formula_recovery=("strict_formula_recovery", "mean"),
                srbench_formula_recovery=("srbench_formula_recovery", "mean"),
                numerical_complete_fit=("numerical_complete_fit", "mean"),
                skeleton_recovery=("skeleton_recovery", "mean"),
                median_test_mse=("best_test_mse", "median"),
                median_test_rmse=("test_rmse", "median"),
                median_test_nmse=("test_nmse", "median"),
                median_test_nrmse=("test_nrmse", "median"),
                median_complexity=("expr_complexity", "median"),
                median_normalized_complexity=("normalized_complexity", "median"),
                median_runtime_sec=("runtime_sec", "median"),
            )
            .reset_index()
        )
        noise_summary.to_csv(out_dir / "summary_noise_by_method_level.csv", index=False)
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("case_index", "count"),
            timeout_rate=("timed_out", "mean"),
            pass_rate=("passed", "mean"),
            exact_recovery=("exact_recovery", "mean"),
            strict_formula_recovery=("strict_formula_recovery", "mean"),
            srbench_formula_recovery=("srbench_formula_recovery", "mean"),
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
            median_test_nrmse=("test_nrmse", "median"),
            median_complexity=("expr_complexity", "median"),
            median_normalized_complexity=("normalized_complexity", "median"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary_v11_ablation_by_method.csv", index=False)
    if {"target_component", "failure_mode"}.issubset(df.columns) and df["target_component"].notna().any():
        component_summary = (
            df[df["suite"].eq("component_ablation")]
            .groupby(["target_component", "failure_mode", "method"], dropna=False)
            .agg(
                n=("case_index", "count"),
                pass_rate=("passed", "mean"),
                exact_recovery=("exact_recovery", "mean"),
                skeleton_recovery=("skeleton_recovery", "mean"),
                true_variable_recall=("true_variable_recall", "mean"),
                false_variable_discovery_rate=("false_variable_discovery_rate", "mean"),
                proxy_misuse_rate=("proxy_misuse", "mean"),
                nonlinear_decoy_misuse_rate=("nonlinear_decoy_misuse", "mean"),
                irrelevant_misuse_rate=("irrelevant_misuse", "mean"),
                median_test_mse=("best_test_mse", "median"),
                median_complexity=("expr_complexity", "median"),
                median_runtime_sec=("runtime_sec", "median"),
            )
            .reset_index()
        )
        component_summary.to_csv(out_dir / "summary_component_ablation_by_failure_mode.csv", index=False)
    write_design_doc(out_dir, df, summary)


def write_design_doc(out_dir: Path, rows: pd.DataFrame, summary: pd.DataFrame):
    table = summary.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    lines = [
        "# V11 Ablation Experiment Design",
        "",
        "Base code: `scripts/vl_loopsr.py`.",
        "",
        "Method switches:",
        "",
    ]
    for method in METHODS.values():
        lines.append(f"- `{method.name}`: {method.description}; env={json.dumps(method.env, sort_keys=True)}")
    lines.extend(
        [
            "",
            "Suites:",
            "",
            "- `high_dim`: legacy runner for distractor-variable stress tests; paper high-dimensional runs use `scripts/run_v11_high_dimensional_interference.py`.",
            "- `component_ablation`: matched balanced clean benchmark cases for full vs w/o Observer, w/o Critic, and w/o Proposer.",
            "- `noise`: legacy controlled formulas under target-scale noise levels; paper noise runs use `scripts/run_v11_noise_robustness.py`.",
            "",
            "Figure mapping:",
            "",
            "- Fig.7a: `summary_v11_ablation_by_method.csv` filtered to `suite == component_ablation`.",
            "- Fig.7b: derive failure types from `exact_recovery`, `skeleton_recovery`, variable-recall/FDR fields, and complexity.",
            "- Fig.7c: parse `proposal_history`, `evaluation_history`, and `meta_plan_history` columns for per-round trajectories.",
            "",
            "Current summary:",
            "",
            table.to_markdown(index=False),
            "",
        ]
    )
    (out_dir / "v11_ablation_experiment_design.md").write_text("\n".join(lines), encoding="utf-8")


def parse_noise_levels(text: str) -> list[float]:
    return [float(x) for x in re.split(r"[, ]+", str(text).strip()) if x]


def selected_case_names(args) -> list[str]:
    if has_dataset_root(args):
        return suite_manifest(args)["case_name"].astype(str).tolist()

    if args.suite == "high_dim":
        from run_v11_high_dimensional_interference import make_cases

        cases = make_cases(args.max_cases)
        return [case.case_name for case in cases]
    levels = parse_noise_levels(args.noise_levels) if args.suite == "noise" else [0.0]
    cases = make_formula_cases(args.suite, levels)
    if args.max_cases:
        cases = cases[: args.max_cases]
    return [case.case_name for case in cases]


def run_parent(args) -> int:
    out_dir = Path(args.out_dir)
    result_dir = out_dir / "case_results"
    log_dir = out_dir / "case_logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    parent_timeout_sec = float(args.parent_timeout_sec or 0.0)
    if parent_timeout_sec <= 0:
        parent_timeout_sec = float(args.case_budget_sec) + float(args.timeout_grace_sec)

    method_names = [m.strip() for m in re.split(r"[, ]+", args.methods) if m.strip()]
    unknown = [m for m in method_names if m not in METHODS]
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; available={sorted(METHODS)}")
    case_names = selected_case_names(args)
    if args.max_cases:
        case_names = case_names[: args.max_cases]
    manifest = {
        "suite": args.suite,
        "base_code": str(args.v11_path),
        "methods": {name: asdict(METHODS[name]) for name in method_names},
        "case_names": case_names,
        "noise_levels": parse_noise_levels(args.noise_levels) if args.suite == "noise" else [],
        "dataset_root": str(args.dataset_root) if has_dataset_root(args) else "",
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "case_budget_sec": float(args.case_budget_sec),
        "parent_timeout_sec": float(parent_timeout_sec),
    }
    (out_dir / "manifest_v11_ablation_experiments.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: list[dict] = []
    if args.interleave_methods:
        schedule = [
            (method, case_index, case_name)
            for case_index, case_name in enumerate(case_names, start=1)
            for method in method_names
        ]
    else:
        schedule = [
            (method, case_index, case_name)
            for method in method_names
            for case_index, case_name in enumerate(case_names, start=1)
        ]
    total = len(schedule)
    done = 0
    for method, case_index, case_name in schedule:
        done += 1
        safe = sanitize_name(f"{method}_{case_index:03d}_{case_name}_seed{args.repeat_seed}")
        result_json = result_dir / method / f"{safe}.json"
        log_path = log_dir / method / f"{safe}.log"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if args.resume and result_json.exists():
            cached = json.loads(result_json.read_text(encoding="utf-8"))
            should_rerun = bool(args.rerun_timeouts and cached.get("timed_out"))
            should_rerun = should_rerun or bool(args.rerun_failures and not cached.get("valid_formula_found"))
            if not should_rerun:
                rows.append(cached)
                continue

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--suite",
            args.suite,
            "--method",
            method,
            "--out-dir",
            str(out_dir),
            "--v11-path",
            str(args.v11_path),
            "--case-index",
            str(case_index),
            "--repeat-seed",
            str(args.repeat_seed),
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
            "--noise-levels",
            str(args.noise_levels),
            "--dataset-root",
            str(args.dataset_root),
            "--single-result-json",
            str(result_json),
        ]
        if getattr(args, "experiment_config", ""):
            cmd.extend(["--experiment-config", str(args.experiment_config)])
        if getattr(args, "target_component", ""):
            cmd.extend(["--target-component", str(args.target_component)])
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        child_env.setdefault("PYTHONHASHSEED", str(args.repeat_seed))
        child_env["LLMSR_REPEAT_SEED"] = str(args.repeat_seed)
        child_env["LLMSR_MAX_RUNTIME_PER_TASK_SEC"] = str(args.case_budget_sec)
        child_env.update(METHODS[method].env)
        started = time.time()
        print(f"[RUN {done}/{total}] suite={args.suite} method={method} case={case_name}", flush=True)
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
                row = timeout_row(args, method, case_index, case_name, time.time() - started, log_path, parent_timeout_sec)
                rows.append(row)
                result_json.write_text(json.dumps(json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
                summarize(rows, out_dir)
                print(f"[TIMEOUT {done}/{total}] sec={time.time() - started:.1f}", flush=True)
                continue
        if result_json.exists():
            row = json.loads(result_json.read_text(encoding="utf-8"))
        else:
            row = {
                **timeout_row(args, method, case_index, case_name, time.time() - started, log_path, parent_timeout_sec),
                "timed_out": False,
                "error": f"child exited without result; see {log_path}",
            }
        rows.append(row)
        summarize(rows, out_dir)
        print(
            f"[DONE {done}/{total}] timeout={row.get('timed_out')} "
            f"skeleton={row.get('skeleton_recovery')} mse={row.get('best_test_mse')} sec={row.get('runtime_sec')}",
            flush=True,
        )
    summarize(rows, out_dir)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["high_dim", "component_ablation", "noise"], default="component_ablation")
    parser.add_argument("--target-component", default="", help="Optional manifest target_component filter for component ablations.")
    parser.add_argument("--methods", default="full,w_o_observer_all,w_o_critic,w_o_proposer")
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "v11_ablation_experiments"))
    parser.add_argument("--v11-path", default=str(DEFAULT_V11_PATH))
    parser.add_argument("--case-budget-sec", type=float, default=600.0)
    parser.add_argument("--timeout-grace-sec", type=float, default=30.0)
    parser.add_argument("--parent-timeout-sec", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--repeat-seed", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--noise-levels", default="0 0.001 0.01")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--experiment-config", default="", help="Optional JSON config that contributes method variants and shared env.")
    parser.add_argument("--interleave-methods", action="store_true", help="Run all methods for a case before moving to the next case.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-timeouts", action="store_true")
    parser.add_argument("--rerun-failures", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--method", default="full")
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--single-result-json", default="")
    args = parser.parse_args()
    if args.max_cases <= 0:
        args.max_cases = None
    return args


def main():
    args = parse_args()
    load_experiment_config(args.experiment_config)
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
