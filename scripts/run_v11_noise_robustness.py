#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the VL-LoopSR pipeline on controlled target-noise robustness tasks.

This is the V11-backed version of the Fig.5 experiment in the design table:

* train/val targets receive relative Gaussian noise;
* the test target stays clean, so structural recovery measures whether V11
  recovers the underlying formula rather than the observation noise;
* each case records PASS@100, exact/skeleton recovery proxies, clean/noisy MSE,
  expression complexity, complexity bloat, and residual structure flags.

Ground-truth expressions are used only after the run for scoring.
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

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V11_PATH = ROOT / "scripts" / "test_fey_v11_complexity_exact.py"
PASS_MSE_THRESHOLD = 100.0
EXACT_CLEAN_MSE_THRESHOLD = 1e-8
SKELETON_CLEAN_MSE_THRESHOLD = 1e-3
POST_SKELETON_REFINE_ENABLED = os.environ.get("LLMSR_NOISE_POST_SKELETON_REFINE", "1").strip().lower() in {"1", "true", "yes", "y"}
POST_SKELETON_REFINE_MAX_CANDIDATES = int(os.environ.get("LLMSR_NOISE_POST_REFINE_MAX_CANDIDATES", "16"))
POST_SKELETON_REFINE_NEAR_TIE_REL = float(os.environ.get("LLMSR_NOISE_POST_REFINE_NEAR_TIE_REL", "0.05"))
POST_SKELETON_REFINE_DENOMINATOR = int(os.environ.get("LLMSR_NOISE_POST_REFINE_MAX_DENOMINATOR", "20"))


@dataclass(frozen=True)
class NoiseBaseCase:
    base_index: int
    benchmark: str
    case_name: str
    structure_type: str
    feature_names: list[str]
    true_variables: list[str]
    true_expression: str
    fn_name: str
    sample_low: float = -2.0
    sample_high: float = 2.0


@dataclass(frozen=True)
class NoiseCase:
    case_index: int
    benchmark: str
    case_name: str
    base_case_name: str
    structure_type: str
    feature_names: list[str]
    true_variables: list[str]
    true_expression: str
    fn_name: str
    noise_level: float
    sample_low: float
    sample_high: float
    fixed_repeat_seed: int = 0
    train_noisy_path: str = ""
    val_noisy_path: str = ""
    test_clean_path: str = ""
    train_clean_path: str = ""
    val_clean_path: str = ""
    train_noise_sigma: float = 0.0
    val_noise_sigma: float = 0.0


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


def formula_fn(name: str) -> Callable[[pd.DataFrame], np.ndarray]:
    formulas = {
        "nguyen_poly": lambda x: x["x1"].to_numpy() ** 3 + x["x1"].to_numpy() ** 2 + x["x1"].to_numpy(),
        "nguyen_rational": lambda x: x["x1"].to_numpy() / (1.0 + x["x2"].to_numpy() ** 2),
        "feynman_trig": lambda x: np.sin(x["x1"].to_numpy()) + 0.5 * np.cos(x["x2"].to_numpy()),
        "feynman_interaction": lambda x: x["x1"].to_numpy() * x["x2"].to_numpy() + 0.25 * x["x3"].to_numpy() ** 2,
        "srsd_exp_log": lambda x: np.exp(0.3 * x["x1"].to_numpy()) + np.log(np.abs(x["x2"].to_numpy()) + 2.0),
        "srsd_nested": lambda x: np.sin(x["x1"].to_numpy() * x["x2"].to_numpy()) / (1.0 + x["x3"].to_numpy() ** 2),
        "realistic_decay": lambda x: np.exp(-0.25 * np.abs(x["x1"].to_numpy())) * np.sin(x["x2"].to_numpy()) + 0.15 * x["x3"].to_numpy(),
        "realistic_sensor": lambda x: np.log(np.abs(x["x1"].to_numpy()) + 2.0) + 0.35 * x["x2"].to_numpy() ** 2 - 0.2 * np.cos(x["x3"].to_numpy()),
    }
    return formulas[name]


def base_cases() -> list[NoiseBaseCase]:
    return [
        NoiseBaseCase(
            1,
            "Nguyen",
            "nguyen_poly",
            "polynomial",
            ["x1"],
            ["x1"],
            "x1**3 + x1**2 + x1",
            "nguyen_poly",
        ),
        NoiseBaseCase(
            2,
            "Nguyen",
            "nguyen_rational",
            "rational",
            ["x1", "x2"],
            ["x1", "x2"],
            "x1/(1 + x2**2)",
            "nguyen_rational",
        ),
        NoiseBaseCase(
            3,
            "Feynman",
            "feynman_trig",
            "trigonometric",
            ["x1", "x2"],
            ["x1", "x2"],
            "sin(x1) + 0.5*cos(x2)",
            "feynman_trig",
        ),
        NoiseBaseCase(
            4,
            "Feynman",
            "feynman_interaction",
            "interaction",
            ["x1", "x2", "x3"],
            ["x1", "x2", "x3"],
            "x1*x2 + 0.25*x3**2",
            "feynman_interaction",
        ),
        NoiseBaseCase(
            5,
            "SRSD",
            "srsd_exp_log",
            "exp_log",
            ["x1", "x2"],
            ["x1", "x2"],
            "exp(0.3*x1) + log(abs(x2) + 2)",
            "srsd_exp_log",
        ),
        NoiseBaseCase(
            6,
            "SRSD",
            "srsd_nested",
            "function_composition",
            ["x1", "x2", "x3"],
            ["x1", "x2", "x3"],
            "sin(x1*x2)/(1 + x3**2)",
            "srsd_nested",
        ),
        NoiseBaseCase(
            7,
            "Realistic",
            "realistic_decay",
            "exp_trig",
            ["x1", "x2", "x3"],
            ["x1", "x2", "x3"],
            "exp(-0.25*abs(x1))*sin(x2) + 0.15*x3",
            "realistic_decay",
        ),
        NoiseBaseCase(
            8,
            "Realistic",
            "realistic_sensor",
            "mixed_smooth",
            ["x1", "x2", "x3"],
            ["x1", "x2", "x3"],
            "log(abs(x1) + 2) + 0.35*x2**2 - 0.2*cos(x3)",
            "realistic_sensor",
        ),
    ]


def parse_noise_levels(text: str) -> list[float]:
    levels = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        level = float(item)
        if level < 0:
            raise ValueError(f"noise level must be non-negative: {item}")
        levels.append(level)
    if not levels:
        raise ValueError("at least one noise level is required")
    return levels


def make_cases(noise_levels: list[float], max_cases: int | None = None, benchmarks: set[str] | None = None) -> list[NoiseCase]:
    cases: list[NoiseCase] = []
    idx = 1
    for base in base_cases():
        if benchmarks and base.benchmark.lower() not in benchmarks:
            continue
        for level in noise_levels:
            cases.append(
                NoiseCase(
                    case_index=idx,
                    benchmark=base.benchmark,
                    case_name=f"{base.case_name}_noise_{level:g}",
                    base_case_name=base.case_name,
                    structure_type=base.structure_type,
                    feature_names=base.feature_names,
                    true_variables=base.true_variables,
                    true_expression=base.true_expression,
                    fn_name=base.fn_name,
                    noise_level=float(level),
                    sample_low=base.sample_low,
                    sample_high=base.sample_high,
                )
            )
            idx += 1
    return cases[:max_cases] if max_cases else cases


def make_clean_features(case: NoiseCase, n: int, rng: np.random.Generator) -> pd.DataFrame:
    values = rng.uniform(case.sample_low, case.sample_high, size=(n, len(case.feature_names)))
    return pd.DataFrame(values, columns=case.feature_names)


def add_relative_noise(y: np.ndarray, level: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    y = np.asarray(y, dtype=float)
    sigma = float(np.std(y)) * float(level)
    if sigma <= 0:
        return y.copy(), 0.0
    return y + rng.normal(0.0, sigma, size=len(y)), sigma


def make_split(case: NoiseCase, seed: int, n_train: int, n_val: int, n_test: int):
    train_rng = np.random.default_rng(seed + case.case_index * 10_000 + 1)
    val_rng = np.random.default_rng(seed + case.case_index * 10_000 + 2)
    test_rng = np.random.default_rng(seed + case.case_index * 10_000 + 3)
    noise_rng = np.random.default_rng(seed + case.case_index * 10_000 + 4)

    fn = formula_fn(case.fn_name)
    train_x = make_clean_features(case, n_train, train_rng)
    val_x = make_clean_features(case, n_val, val_rng)
    test_x = make_clean_features(case, n_test, test_rng)

    train_clean = fn(train_x)
    val_clean = fn(val_x)
    test_clean = fn(test_x)
    train_noisy, train_sigma = add_relative_noise(train_clean, case.noise_level, noise_rng)
    val_noisy, val_sigma = add_relative_noise(val_clean, case.noise_level, noise_rng)

    train_df = train_x.copy()
    train_df["y"] = train_noisy
    val_df = val_x.copy()
    val_df["y"] = val_noisy
    test_df = test_x.copy()
    test_df["y"] = test_clean

    clean = {
        "train": train_x.assign(y=train_clean),
        "val": val_x.assign(y=val_clean),
        "test": test_x.assign(y=test_clean),
        "train_sigma": train_sigma,
        "val_sigma": val_sigma,
    }
    return train_df, val_df, test_df, clean


def load_manifest_cases(manifest_path: Path, max_cases: int | None = None, benchmarks: set[str] | None = None) -> list[NoiseCase]:
    df = pd.read_csv(manifest_path)
    cases: list[NoiseCase] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        benchmark = str(row.get("benchmark") or "")
        if benchmarks and benchmark.lower() not in benchmarks:
            continue
        variables = [x for x in str(row.get("variables") or row.get("true_variables") or "").split("|") if x]
        true_variables = [x for x in str(row.get("true_variables") or row.get("variables") or "").split("|") if x]
        cases.append(
            NoiseCase(
                case_index=len(cases) + 1,
                benchmark=benchmark,
                case_name=str(row.get("case_name") or row.get("case_id") or f"case_{idx + 1}"),
                base_case_name=str(row.get("base_case_name") or row.get("formula_id") or ""),
                structure_type=str(row.get("structure_type") or ""),
                feature_names=variables,
                true_variables=true_variables or variables,
                true_expression=str(row.get("expression") or row.get("true_expression") or ""),
                fn_name=str(row.get("formula_id") or ""),
                noise_level=float(row.get("noise_level") or 0.0),
                sample_low=float(row.get("sample_low") or -2.0),
                sample_high=float(row.get("sample_high") or 2.0),
                fixed_repeat_seed=int(row.get("repeat_seed") or 0),
                train_noisy_path=str(row.get("train_noisy_path") or ""),
                val_noisy_path=str(row.get("val_noisy_path") or ""),
                test_clean_path=str(row.get("test_clean_path") or ""),
                train_clean_path=str(row.get("train_clean_path") or ""),
                val_clean_path=str(row.get("val_clean_path") or ""),
                train_noise_sigma=float(row.get("train_noise_sigma") or 0.0),
                val_noise_sigma=float(row.get("val_noise_sigma") or 0.0),
            )
        )
        if max_cases and len(cases) >= max_cases:
            break
    return cases


def load_fixed_split(case: NoiseCase):
    train_df = pd.read_csv(case.train_noisy_path)
    val_df = pd.read_csv(case.val_noisy_path)
    test_df = pd.read_csv(case.test_clean_path)
    train_clean = pd.read_csv(case.train_clean_path) if case.train_clean_path else train_df.copy()
    val_clean = pd.read_csv(case.val_clean_path) if case.val_clean_path else val_df.copy()
    clean = {
        "train": train_clean,
        "val": val_clean,
        "test": test_df.copy(),
        "train_sigma": float(case.train_noise_sigma),
        "val_sigma": float(case.val_noise_sigma),
    }
    return train_df, val_df, test_df, clean


def case_repeat_seed(case: NoiseCase, fallback: int) -> int:
    return int(case.fixed_repeat_seed if (case.train_noisy_path or case.test_clean_path) else fallback)


def apply_v11_budget_env(env: dict[str, str], budget_sec: float) -> dict[str, str]:
    mode = os.environ.get("LLMSR_NOISE_RUNTIME_MODE", "fast").strip().lower()
    real_llm_mode = mode in {"real_llm", "llm", "full", "enhanced"}
    if real_llm_mode:
        profile_sec = float(os.environ.get("LLMSR_NOISE_V11_PROFILE_SEC", str(float(budget_sec))))
        env["LLMSR_MAX_RUNTIME_PER_TASK_SEC"] = str(float(budget_sec))
        env["LLMSR_V11_BUDGET_AWARE_MODE"] = "enabled"
        env["LLMSR_V11_DISABLE_RUNTIME_FAST_PATH"] = "1"
        env["LLMSR_V11_ALLOW_FULL_CALLS_UNDER_180"] = os.environ.get("LLMSR_V11_ALLOW_FULL_CALLS_UNDER_180", "1")
        env["LLMSR_V11_FULL_BUDGET"] = os.environ.get("LLMSR_V11_FULL_BUDGET", "1")
        env["LLMSR_V11_FULL_BUDGET_TEXT_CALLS"] = os.environ.get("LLMSR_V11_FULL_BUDGET_TEXT_CALLS", "4")
        env["LLMSR_V11_FULL_BUDGET_MM_CALLS"] = os.environ.get("LLMSR_V11_FULL_BUDGET_MM_CALLS", "0")
        env["LLMSR_V11_FULL_BUDGET_PROPOSAL_K"] = os.environ.get("LLMSR_V11_FULL_BUDGET_PROPOSAL_K", "16")
        env["LLMSR_V11_FULL_BUDGET_REFINED_K"] = os.environ.get("LLMSR_V11_FULL_BUDGET_REFINED_K", "8")
        env["LLMSR_V11_FULL_BUDGET_REFINE_ROUNDS"] = os.environ.get("LLMSR_V11_FULL_BUDGET_REFINE_ROUNDS", "0")
        env["LLMSR_V11_LOW_DIM_FULL_BUDGET"] = os.environ.get("LLMSR_V11_LOW_DIM_FULL_BUDGET", "1")
        env["LLMSR_V11_LOW_DIM_FULL_TEXT_CALLS"] = os.environ.get("LLMSR_V11_LOW_DIM_FULL_TEXT_CALLS", "4")
        env["LLMSR_V11_LOW_DIM_FULL_MM_CALLS"] = os.environ.get("LLMSR_V11_LOW_DIM_FULL_MM_CALLS", "0")
        env["LLMSR_V11_LOW_DIM_FULL_PROPOSAL_K"] = os.environ.get("LLMSR_V11_LOW_DIM_FULL_PROPOSAL_K", "14")
        env["LLMSR_V11_LOW_DIM_FULL_REFINED_K"] = os.environ.get("LLMSR_V11_LOW_DIM_FULL_REFINED_K", "7")
        env["LLMSR_V11_LOW_DIM_FULL_REFINE_ROUNDS"] = os.environ.get("LLMSR_V11_LOW_DIM_FULL_REFINE_ROUNDS", "0")
        env["LLMSR_MIN_LLM_CALL_TIMEOUT_SEC"] = os.environ.get("LLMSR_MIN_LLM_CALL_TIMEOUT_SEC", "8")
        env["LLMSR_PROPOSAL_TIMEOUT"] = os.environ.get("LLMSR_PROPOSAL_TIMEOUT", "24")
        env["LLMSR_OBSERVER_TIMEOUT"] = os.environ.get("LLMSR_OBSERVER_TIMEOUT", "12")
        env["LLMSR_CRITIC_TIMEOUT"] = os.environ.get("LLMSR_CRITIC_TIMEOUT", "12")
        env["LLMSR_JUDGE_TIMEOUT"] = os.environ.get("LLMSR_JUDGE_TIMEOUT", "8")
        env["LLMSR_REFINER_TIMEOUT"] = os.environ.get("LLMSR_REFINER_TIMEOUT", "16")
        env["LLMSR_V11_ENABLE_CRITIC_LOOP"] = os.environ.get("LLMSR_V11_ENABLE_CRITIC_LOOP", "1")
        env["LLMSR_V11_ENABLE_STRUCTURAL_RESCUE"] = os.environ.get("LLMSR_V11_ENABLE_STRUCTURAL_RESCUE", "1")
        env["LLMSR_V11_RESTORE_TEXT_PROPOSER"] = os.environ.get("LLMSR_V11_RESTORE_TEXT_PROPOSER", "1")
        env["LLMSR_V11_ENABLE_VLM_OBSERVER"] = os.environ.get("LLMSR_V11_ENABLE_VLM_OBSERVER", "0")
        env["LLMSR_V11_VLM_OBSERVER_GENERATE_IMAGES"] = os.environ.get("LLMSR_V11_VLM_OBSERVER_GENERATE_IMAGES", "0")
        env["LLMSR_V11_IMAGE_LOOP_MAX_ROUNDS"] = os.environ.get("LLMSR_V11_IMAGE_LOOP_MAX_ROUNDS", "0")
        env["LLMSR_NOISE_RUNTIME_MODE_ACTIVE"] = mode
    else:
        env["LLMSR_MAX_RUNTIME_PER_TASK_SEC"] = str(float(budget_sec))
    if not real_llm_mode and float(budget_sec) <= 120.0:
        env["LLMSR_V11_BUDGET_AWARE_MODE"] = "enabled"
        env["LLMSR_MIN_LLM_CALL_TIMEOUT_SEC"] = "3"
        env["LLMSR_PROPOSAL_TIMEOUT"] = "12"
        env["LLMSR_OBSERVER_TIMEOUT"] = "6"
        env["LLMSR_CRITIC_TIMEOUT"] = "6"
        env["LLMSR_JUDGE_TIMEOUT"] = "5"
        env["LLMSR_REFINER_TIMEOUT"] = "8"
        env["LLMSR_V11_ENABLE_VLM_OBSERVER"] = "0"
        env["LLMSR_V11_VLM_OBSERVER_GENERATE_IMAGES"] = "0"
        env["LLMSR_V11_ENABLE_CRITIC_LOOP"] = "0"
        env["LLMSR_V11_ENABLE_STRUCTURAL_RESCUE"] = "0"
        env["LLMSR_V11_IMAGE_LOOP_MAX_ROUNDS"] = "0"
        env["LLMSR_V11_IMAGE_LOOP_MAX_AUGMENTED_CANDIDATES"] = "24"
        env["LLMSR_V11_SURFACE_ANALYTIC_MAX_CANDIDATES"] = "8"
        env["LLMSR_V11_SURFACE_GRID_MAX_CANDIDATES"] = "3"
        env["LLMSR_V11_SURFACE_STABLE_MAX_CANDIDATES"] = "4"
        env["LLMSR_V11_CONSTRUCTED_EXTRAPOLATION_MAX_CANDIDATES"] = "6"
        env["LLMSR_V11_EXTRAPOLATION_OLS_MAX_CANDIDATES"] = "3"
    return env


def import_v11_module(v11_path: Path, results_root: Path):
    if not v11_path.exists():
        raise FileNotFoundError(f"V11 file not found: {v11_path}")
    spec = importlib.util.spec_from_file_location("v11_noise_runtime", str(v11_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["v11_noise_runtime"] = module
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
    if not isinstance(expr, str) or not expr.strip():
        return None
    try:
        import sympy as sp

        parsed = sp.sympify(expr)
        return float(sp.count_ops(parsed, visual=False) + len(parsed.free_symbols))
    except Exception:
        tokens = [tok for tok in re.split(r"[^A-Za-z0-9_.]+", expr) if tok]
        return float(len(tokens)) if tokens else None


def finite_mse(y_true, y_pred) -> float | None:
    try:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return None
        return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))
    except Exception:
        return None


def _sympy_locals(feature_names: list[str]) -> dict:
    import sympy as sp

    local = {name: sp.Symbol(name) for name in feature_names}
    local.update(
        {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "tanh": sp.tanh,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "Abs": sp.Abs,
            "pi": sp.pi,
            "E": sp.E,
        }
    )
    return local


def _parse_sympy_expr(expr: str | None, feature_names: list[str]):
    if not expr:
        return None
    try:
        import sympy as sp

        return sp.sympify(str(expr).replace("^", "**"), locals=_sympy_locals(feature_names))
    except Exception:
        return None


def _snap_float_to_rational(value: float):
    import sympy as sp
    from fractions import Fraction

    if not math.isfinite(value):
        return sp.Float(value)
    if abs(value) <= 1e-10:
        return sp.Integer(0)
    rounded_int = round(value)
    if abs(value - rounded_int) <= 1e-10:
        return sp.Integer(int(rounded_int))
    frac = Fraction(float(value)).limit_denominator(max(1, int(POST_SKELETON_REFINE_DENOMINATOR)))
    snapped = float(frac)
    if abs(value - snapped) <= max(1e-4, 0.03 * max(1.0, abs(value))):
        return sp.Rational(frac.numerator, frac.denominator)
    return sp.Float(float(value), 12)


def _snap_expression_constants(parsed):
    if parsed is None:
        return None
    try:
        import sympy as sp

        replacements = {}
        for atom in parsed.atoms(sp.Float):
            replacements[atom] = _snap_float_to_rational(float(atom))
        for atom in parsed.atoms(sp.Rational):
            if atom.q > max(1, int(POST_SKELETON_REFINE_DENOMINATOR)):
                replacements[atom] = _snap_float_to_rational(float(atom))
        return sp.simplify(parsed.xreplace(replacements))
    except Exception:
        return parsed


def _expr_to_string(parsed) -> str | None:
    if parsed is None:
        return None
    try:
        import sympy as sp

        return sp.sstr(parsed)
    except Exception:
        return str(parsed)


def _evaluate_expr_mse(v11, expr: str, df: pd.DataFrame) -> float | None:
    try:
        pred = v11._base.evaluate_expression_on_df(expr, df)
        return finite_mse(df["y"].to_numpy(dtype=float), pred)
    except Exception:
        return None


def _valid_candidate_metrics(v11, expr: str, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict | None:
    train_mse = _evaluate_expr_mse(v11, expr, train_df)
    val_mse = _evaluate_expr_mse(v11, expr, val_df)
    test_mse = _evaluate_expr_mse(v11, expr, test_df)
    if train_mse is None or val_mse is None or test_mse is None:
        return None
    if not all(math.isfinite(float(x)) for x in [train_mse, val_mse, test_mse]):
        return None
    return {
        "expr": expr,
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "test_mse": float(test_mse),
        "complexity": expr_complexity(expr),
    }


def _fit_affine_expr(v11, expr: str, train_df: pd.DataFrame, feature_names: list[str]) -> str | None:
    parsed = _parse_sympy_expr(expr, feature_names)
    if parsed is None:
        return None
    try:
        pred = np.asarray(v11._base.evaluate_expression_on_df(expr, train_df), dtype=float).reshape(-1)
        y = train_df["y"].to_numpy(dtype=float)
        mask = np.isfinite(pred) & np.isfinite(y)
        if int(mask.sum()) < 3 or float(np.std(pred[mask])) <= 1e-12:
            return None
        X = np.column_stack([pred[mask], np.ones(int(mask.sum()))])
        coef, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        import sympy as sp

        fitted = sp.Float(float(coef[0]), 12) * parsed + sp.Float(float(coef[1]), 12)
        return _expr_to_string(_snap_expression_constants(fitted))
    except Exception:
        return None


def _fit_additive_term_expr(v11, expr: str, train_df: pd.DataFrame, feature_names: list[str]) -> str | None:
    parsed = _parse_sympy_expr(expr, feature_names)
    if parsed is None:
        return None
    try:
        import sympy as sp

        expanded = sp.expand(parsed)
        raw_terms = expanded.as_ordered_terms() if isinstance(expanded, sp.Add) else [expanded]
        basis_exprs = []
        for term in raw_terms:
            _coeff, basis = term.as_coeff_Mul()
            if basis == 1:
                continue
            if basis not in basis_exprs:
                basis_exprs.append(basis)
        if not basis_exprs:
            return None

        cols = []
        y = train_df["y"].to_numpy(dtype=float)
        mask = np.isfinite(y)
        for basis in basis_exprs:
            basis_str = sp.sstr(basis)
            values = np.asarray(v11._base.evaluate_expression_on_df(basis_str, train_df), dtype=float).reshape(-1)
            cols.append(values)
            mask &= np.isfinite(values)
        if int(mask.sum()) <= len(basis_exprs) + 1:
            return None
        X = np.column_stack([col[mask] for col in cols] + [np.ones(int(mask.sum()))])
        coef, *_ = np.linalg.lstsq(X, y[mask], rcond=None)
        fitted = sp.Integer(0)
        for value, basis in zip(coef[:-1], basis_exprs):
            if abs(float(value)) > 1e-10:
                fitted += sp.Float(float(value), 12) * basis
        if abs(float(coef[-1])) > 1e-10:
            fitted += sp.Float(float(coef[-1]), 12)
        return _expr_to_string(_snap_expression_constants(fitted))
    except Exception:
        return None


def post_skeleton_refine_result(
    v11,
    result: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """Refit constants of the discovered structure using train/val only.

    This post-step is deliberately blind to the clean scoring labels except for
    recording the selected expression's final test MSE. Selection uses noisy
    validation MSE with a small near-tie preference for simpler snapped forms.
    """

    if not POST_SKELETON_REFINE_ENABLED:
        result["post_skeleton_refine_enabled"] = False
        return result
    expr = result.get("best_expr")
    if not isinstance(expr, str) or not expr.strip():
        result["post_skeleton_refine_enabled"] = True
        result["post_skeleton_refine_applied"] = False
        result["post_skeleton_refine_reason"] = "missing_best_expr"
        return result

    candidates: list[tuple[str, str]] = [(expr, "original")]
    parsed = _parse_sympy_expr(expr, feature_names)
    snapped = _expr_to_string(_snap_expression_constants(parsed))
    if snapped:
        candidates.append((snapped, "snap_constants"))
    affine = _fit_affine_expr(v11, expr, train_df, feature_names)
    if affine:
        candidates.append((affine, "affine_refit_snap"))
    additive = _fit_additive_term_expr(v11, expr, train_df, feature_names)
    if additive:
        candidates.append((additive, "additive_term_refit_snap"))

    seen = set()
    scored = []
    for cand_expr, source in candidates[: max(1, int(POST_SKELETON_REFINE_MAX_CANDIDATES))]:
        key = str(cand_expr).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        metrics = _valid_candidate_metrics(v11, key, train_df, val_df, test_df)
        if metrics is not None:
            metrics["source"] = source
            scored.append(metrics)
    if not scored:
        result["post_skeleton_refine_enabled"] = True
        result["post_skeleton_refine_applied"] = False
        result["post_skeleton_refine_reason"] = "no_valid_refined_candidate"
        return result

    original = scored[0]
    current_val = result.get("best_val_mse")
    try:
        current_val = float(current_val)
    except Exception:
        current_val = float(original["val_mse"])
    if not math.isfinite(current_val):
        current_val = float(original["val_mse"])
    near_tie = max(1e-10, abs(current_val) * float(POST_SKELETON_REFINE_NEAR_TIE_REL))

    def candidate_key(item):
        outside = max(0.0, float(item["val_mse"]) - (current_val + near_tie))
        complexity = item.get("complexity")
        complexity_value = float(complexity) if complexity is not None and math.isfinite(float(complexity)) else 1e9
        source_bonus = 0 if item.get("source") != "original" else 1
        return (outside > 0.0, outside, complexity_value, source_bonus, float(item["val_mse"]))

    best = min(scored, key=candidate_key)
    applied = str(best["expr"]) != str(expr)
    result["post_skeleton_refine_enabled"] = True
    result["post_skeleton_refine_applied"] = bool(applied)
    result["post_skeleton_refine_source"] = best.get("source")
    result["post_skeleton_refine_original_expr"] = expr
    result["post_skeleton_refine_original_val_mse"] = original.get("val_mse")
    result["post_skeleton_refine_original_test_mse"] = original.get("test_mse")
    result["post_skeleton_refine_trace"] = json.dumps(json_safe(scored[:8]), ensure_ascii=False)
    if applied:
        result["best_expr"] = best["expr"]
        result["best_expr_source"] = str(result.get("best_expr_source") or "v11") + "+post_skeleton_refine"
        result["best_train_mse"] = best["train_mse"]
        result["best_val_mse"] = best["val_mse"]
        result["best_test_mse"] = best["test_mse"]
        result["expr_complexity"] = best.get("complexity")
    return result


def residual_structure_score(residual: np.ndarray, df: pd.DataFrame, feature_names: list[str]) -> float | None:
    try:
        residual = np.asarray(residual, dtype=float)
        if np.std(residual) <= 1e-12:
            return 0.0
        best = 0.0
        probes: list[np.ndarray] = []
        for name in feature_names:
            x = df[name].to_numpy(dtype=float)
            probes.extend([x, x ** 2, np.sin(x), np.cos(x)])
        for i, name_i in enumerate(feature_names):
            for name_j in feature_names[i + 1 :]:
                probes.append(df[name_i].to_numpy(dtype=float) * df[name_j].to_numpy(dtype=float))
        for probe in probes:
            if np.std(probe) <= 1e-12:
                continue
            corr = np.corrcoef(residual, probe)[0, 1]
            if math.isfinite(corr):
                best = max(best, abs(float(corr)))
        return best
    except Exception:
        return None


def variables_used(v11, expr: str | None, feature_names: list[str]) -> list[str]:
    if not expr:
        return []
    try:
        sig = v11._base.extract_formula_form_signature(expr, feature_names)
        used = list(sig.get("variables_used", []) or [])
    except Exception:
        used = [name for name in feature_names if re.search(rf"\b{re.escape(name)}\b", str(expr))]
    feature_set = set(feature_names)
    return [name for name in feature_names if name in set(used) and name in feature_set]


def formula_form_score(v11, expr: str | None, true_expr: str, feature_names: list[str]) -> dict:
    if not expr:
        return {"score": 0.0}
    try:
        return v11._base.score_formula_form_match(expr, true_expr, feature_names)
    except Exception:
        return {"score": 0.0}


def clean_eval_metrics(v11, case: NoiseCase, result: dict, clean: dict) -> dict:
    expr = result.get("best_expr")
    true_complexity = expr_complexity(case.true_expression)
    found_complexity = result.get("expr_complexity") or expr_complexity(expr)
    active = variables_used(v11, expr, case.feature_names)
    active_set = set(active)
    true_set = set(case.true_variables)
    false_count = len(active_set - true_set)
    recall = len(active_set & true_set) / max(1, len(true_set))
    precision = len(active_set & true_set) / max(1, len(active_set))
    fdr = false_count / max(1, len(active_set))
    form = formula_form_score(v11, expr, case.true_expression, case.feature_names)
    form_score = float(form.get("score") or 0.0)

    clean_train_mse = clean_val_mse = clean_test_mse = None
    residual_score = None
    residual_structured = None
    if expr:
        try:
            train_pred = v11._base.evaluate_expression_on_df(expr, clean["train"])
            val_pred = v11._base.evaluate_expression_on_df(expr, clean["val"])
            test_pred = v11._base.evaluate_expression_on_df(expr, clean["test"])
            clean_train_mse = finite_mse(clean["train"]["y"].to_numpy(), train_pred)
            clean_val_mse = finite_mse(clean["val"]["y"].to_numpy(), val_pred)
            clean_test_mse = finite_mse(clean["test"]["y"].to_numpy(), test_pred)
            residual = clean["test"]["y"].to_numpy(dtype=float) - np.asarray(test_pred, dtype=float)
            residual_score = residual_structure_score(residual, clean["test"], case.feature_names)
            residual_structured = bool(
                residual_score is not None
                and residual_score >= 0.35
                and (clean_test_mse is None or clean_test_mse > SKELETON_CLEAN_MSE_THRESHOLD)
            )
        except Exception:
            pass

    clean_mse_for_pass = float(clean_test_mse) if clean_test_mse is not None else float("inf")
    skeleton = bool(
        clean_mse_for_pass <= SKELETON_CLEAN_MSE_THRESHOLD
        and recall >= 1.0
        and fdr <= 0.25
        and form_score >= 0.65
    )
    exact = bool(
        clean_mse_for_pass <= EXACT_CLEAN_MSE_THRESHOLD
        and active_set == true_set
        and form_score >= 0.90
    )
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
        "exact_recovery": exact,
        "exact_recovery_proxy": exact,
        "passed": bool(clean_mse_for_pass <= PASS_MSE_THRESHOLD),
        "pass_at_100": bool(clean_mse_for_pass <= 100.0),
        "pass_at_300": bool(clean_mse_for_pass <= 300.0),
        "clean_train_mse": clean_train_mse,
        "clean_val_mse": clean_val_mse,
        "clean_test_mse": clean_test_mse,
        "mse_at_pass": clean_test_mse if clean_mse_for_pass <= PASS_MSE_THRESHOLD else None,
        "residual_structure_score": residual_score,
        "residual_structured": residual_structured,
        "expr_complexity": found_complexity,
        "true_expr_complexity": true_complexity,
        "complexity_bloat": complexity_bloat,
    }


def row_meta_for_case(case: NoiseCase, train_sigma: float, val_sigma: float) -> dict:
    return {
        "task_type": "noise_robustness",
        "dataset_dir": "noise_robustness",
        "suite": "noise_robustness",
        "difficulty": case.benchmark,
        "benchmark": case.benchmark,
        "base_name": case.case_name,
        "case_name": case.case_name,
        "base_case_name": case.base_case_name,
        "structure_type": case.structure_type,
        "n_features": len(case.feature_names),
        "noise_level": float(case.noise_level),
        "noise_kind": "relative_gaussian_y_train_val",
        "train_noise_sigma": float(train_sigma),
        "val_noise_sigma": float(val_sigma),
        "true_expression": None,
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
        v11 = import_v11_module(Path(args.v11_path), Path(args.out_dir))
        seed = args.random_state + args.repeat_seed * 1_000_000
        train_df, val_df, test_df, clean = load_fixed_split(case) if args.dataset_manifest else make_split(case, seed, args.n_train, args.n_val, args.n_test)
        row_meta = row_meta_for_case(case, clean["train_sigma"], clean["val_sigma"])
        with TemporaryDirectory(prefix="v11_noise_tmp_") as tmpdir_str:
            dataset = v11.build_dataset_from_explicit_splits(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                tmpdir=Path(tmpdir_str),
            )
            dataset.source_tag = "noise_robustness"
            result = v11._run_core_pipeline(dataset=dataset, row_meta=row_meta)
        result = post_skeleton_refine_result(
            v11=v11,
            result=result,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_names=case.feature_names,
        )

        out = {
            **result,
            **row_meta,
            "method": "vl_loopsr",
            "case_index": int(case.case_index),
            "repeat_seed": case_repeat_seed(case, args.repeat_seed),
            "budget_sec": float(args.case_budget_sec),
            "timed_out": False,
            "runtime_sec": float(result.get("runtime_sec") or (time.time() - started)),
            "true_expression_for_scoring": case.true_expression,
            "feature_names": "|".join(case.feature_names),
            "dataset_manifest": str(args.dataset_manifest or ""),
            "best_noisy_val_mse": result.get("best_val_mse"),
            "best_clean_test_mse_from_v11": result.get("best_test_mse"),
        }
        out.update(clean_eval_metrics(v11, case, out, clean))
    except BaseException as exc:
        case_meta = asdict(case)
        out = {
            "method": "vl_loopsr",
            "case_index": int(args.case_index),
            "case_name": case_meta.get("case_name"),
            "base_case_name": case_meta.get("base_case_name"),
            "benchmark": case_meta.get("benchmark"),
            "structure_type": case_meta.get("structure_type"),
            "noise_level": case_meta.get("noise_level"),
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
            "true_expression_for_scoring": case_meta.get("true_expression"),
            "true_variables": "|".join(case_meta.get("true_variables") or []),
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


def timeout_row(args, case: NoiseCase, repeat_seed: int, runtime_sec: float, log_path: Path, parent_timeout_sec: float):
    return {
        "method": "vl_loopsr",
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
        "skeleton_recovery",
        "noise_level",
        "best_val_mse",
        "clean_test_mse",
        "mse_at_pass",
        "runtime_sec",
        "expr_complexity",
        "complexity_bloat",
        "residual_structured",
        "residual_structure_score",
    ]:
        if column not in df:
            df[column] = np.nan
    df.to_csv(out_dir / "all_noise_robustness_vl_loopsr_results.csv", index=False)

    summary = (
        df.groupby(["noise_level", "benchmark", "method"], dropna=False)
        .agg(
            n=("case_index", "count"),
            timeout_rate=("timed_out", "mean"),
            pass_at_100=("pass_at_100", "mean"),
            pass_at_300=("pass_at_300", "mean"),
            exact_recovery=("exact_recovery", "mean"),
            skeleton_recovery=("skeleton_recovery", "mean"),
            median_clean_test_mse=("clean_test_mse", "median"),
            median_noisy_val_mse=("best_val_mse", "median"),
            median_complexity=("expr_complexity", "median"),
            median_complexity_bloat=("complexity_bloat", "median"),
            residual_structured_rate=("residual_structured", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary_noise_robustness_vl_loopsr.csv", index=False)

    by_method = (
        df.groupby(["noise_level", "method"], dropna=False)
        .agg(
            n=("case_index", "count"),
            timeout_rate=("timed_out", "mean"),
            pass_at_100=("pass_at_100", "mean"),
            pass_at_300=("pass_at_300", "mean"),
            exact_recovery=("exact_recovery", "mean"),
            skeleton_recovery=("skeleton_recovery", "mean"),
            median_clean_test_mse=("clean_test_mse", "median"),
            median_complexity=("expr_complexity", "median"),
            median_complexity_bloat=("complexity_bloat", "median"),
            residual_structured_rate=("residual_structured", "mean"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .reset_index()
    )
    by_method.to_csv(out_dir / "summary_noise_robustness_by_noise_vl_loopsr.csv", index=False)
    plot_figures(df, by_method, out_dir)
    write_design_doc(out_dir, by_method)


def plot_figures(df: pd.DataFrame, by_noise: pd.DataFrame, out_dir: Path):
    if plt is None:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    if sns is not None:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use("ggplot")

    plt.figure(figsize=(8.2, 4.6))
    if sns is not None:
        sns.lineplot(data=df, x="noise_level", y="skeleton_recovery", hue="method", marker="o", errorbar=("ci", 95))
    else:
        for method, sub in df.groupby("method", dropna=False):
            grouped = sub.groupby("noise_level")["skeleton_recovery"].mean().reset_index()
            plt.plot(grouped["noise_level"], grouped["skeleton_recovery"], marker="o", label=str(method))
        plt.legend(title="method")
    plt.xscale("symlog", linthresh=0.001)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Relative target noise")
    plt.ylabel("Skeleton Recovery")
    plt.title("Fig.5a Noise-Structure Stability")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5a_noise_structural_stability.png", dpi=180)
    plt.close()

    pareto = df.copy()
    pareto["clean_test_mse_plot"] = pd.to_numeric(pareto["clean_test_mse"], errors="coerce").clip(lower=1e-14)
    plt.figure(figsize=(8.2, 4.8))
    if sns is not None:
        sns.scatterplot(
            data=pareto,
            x="clean_test_mse_plot",
            y="expr_complexity",
            hue="noise_level",
            style="benchmark",
            s=78,
            alpha=0.86,
        )
    else:
        for noise_level, sub in pareto.groupby("noise_level", dropna=False):
            plt.scatter(
                sub["clean_test_mse_plot"],
                sub["expr_complexity"],
                s=72,
                alpha=0.82,
                label=f"noise={noise_level:g}" if pd.notna(noise_level) else "noise=nan",
            )
        plt.legend()
    plt.xscale("log")
    plt.xlabel("Clean test MSE")
    plt.ylabel("Final expression complexity")
    plt.title("Fig.5b MSE-Complexity Pareto Under Noise")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5b_noise_mse_complexity_pareto.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.2, 4.6))
    if sns is not None:
        sns.lineplot(data=df, x="noise_level", y="complexity_bloat", hue="method", marker="o", errorbar=("ci", 95))
    else:
        for method, sub in df.groupby("method", dropna=False):
            grouped = sub.groupby("noise_level")["complexity_bloat"].median().reset_index()
            plt.plot(grouped["noise_level"], grouped["complexity_bloat"], marker="o", label=str(method))
        plt.legend(title="method")
    plt.xscale("symlog", linthresh=0.001)
    plt.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    plt.xlabel("Relative target noise")
    plt.ylabel("Expression complexity / true complexity")
    plt.title("Fig.5c Complexity Inflation Under Noise")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5c_noise_complexity_inflation.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.2, 4.6))
    if sns is not None:
        sns.lineplot(data=by_noise, x="noise_level", y="residual_structured_rate", hue="method", marker="o")
    else:
        for method, sub in by_noise.groupby("method", dropna=False):
            plt.plot(sub["noise_level"], sub["residual_structured_rate"], marker="o", label=str(method))
        plt.legend(title="method")
    plt.xscale("symlog", linthresh=0.001)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Relative target noise")
    plt.ylabel("Residual structured rate")
    plt.title("Residual Structure Remaining After Recovery")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig5d_residual_structure_rate.png", dpi=180)
    plt.close()


def write_design_doc(out_dir: Path, by_noise: pd.DataFrame):
    doc = out_dir / "noise_robustness_design.md"
    table = by_noise.copy()
    for col in [
        "timeout_rate",
        "pass_at_100",
        "pass_at_300",
        "exact_recovery",
        "skeleton_recovery",
        "median_clean_test_mse",
        "median_complexity",
        "median_complexity_bloat",
        "residual_structured_rate",
        "median_runtime_sec",
    ]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    lines = [
        "# V11 Noise Robustness Experiment",
        "",
        "Base method: `scripts/test_fey_v11_complexity_exact.py` (V11).",
        "",
        "Design:",
        "",
        "- Benchmarks: Nguyen, Feynman, SRSD, and a small realistic smooth-signal group.",
        "- Noise levels: relative Gaussian target noise on train/val only; clean test target is held out for recovery scoring.",
        "- Default levels: `0,0.001,0.01`.",
        "- V11 receives no true expression or true-variable metadata; those are used only after each run for scoring.",
        "",
        "Metrics:",
        "",
        "- `pass_at_100`, `exact_recovery`, `skeleton_recovery`, `clean_test_mse`, `mse_at_pass`.",
        "- `expr_complexity`, `complexity_bloat`, `residual_structure_score`, `residual_structured`.",
        "",
        "Figure mapping:",
        "",
        "- `figures/fig5a_noise_structural_stability.png`: noise vs Skeleton Recovery with bootstrap CI.",
        "- `figures/fig5b_noise_mse_complexity_pareto.png`: clean MSE vs final complexity.",
        "- `figures/fig5c_noise_complexity_inflation.png`: complexity bloat as noise increases.",
        "- `figures/fig5d_residual_structure_rate.png`: whether residuals still contain learnable structure.",
        "",
        "Current summary:",
        "",
        table.to_markdown(index=False),
        "",
    ]
    doc.write_text("\n".join(lines), encoding="utf-8")


def parse_benchmark_filter(text: str | None) -> set[str] | None:
    if not text:
        return None
    items = {item.strip().lower() for item in text.split(",") if item.strip()}
    return items or None


def run_parent(args) -> int:
    out_dir = Path(args.out_dir)
    result_dir = out_dir / "case_results" / "vl_loopsr"
    log_dir = out_dir / "case_logs" / "vl_loopsr"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    parent_timeout_sec = float(args.parent_timeout_sec or 0.0)
    if parent_timeout_sec <= 0:
        parent_timeout_sec = float(args.case_budget_sec) + float(args.timeout_grace_sec)

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
        "method": "vl_loopsr",
        "experiment": "noise_robustness",
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
        "v11_path": str(args.v11_path),
        "noise_runtime_mode": os.environ.get("LLMSR_NOISE_RUNTIME_MODE", "fast"),
        "noise_v11_profile_sec": os.environ.get("LLMSR_NOISE_V11_PROFILE_SEC", ""),
        "noise_protocol": "train/val y noisy, test y clean",
    }
    (out_dir / "manifest_noise_robustness_vl_loopsr.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    total = len(cases) * int(repeat_seeds)
    run_idx = 0
    for repeat_seed in range(int(repeat_seeds)):
        for case in cases:
            run_idx += 1
            safe = sanitize_name(case.case_name)
            out_seed = case_repeat_seed(case, repeat_seed)
            result_json = result_dir / f"{case.case_index:03d}_{safe}_seed{out_seed}.json"
            log_path = log_dir / f"{case.case_index:03d}_{safe}_seed{out_seed}.log"
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
                "--out-dir",
                str(out_dir),
                "--v11-path",
                str(args.v11_path),
                "--case-index",
                str(case.case_index),
                "--repeat-seed",
                str(repeat_seed),
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
            ]
            if args.dataset_manifest:
                cmd.extend(["--dataset-manifest", str(args.dataset_manifest)])
            if args.benchmarks:
                cmd.extend(["--benchmarks", args.benchmarks])
            started = time.time()
            print(
                f"[RUN {run_idx}/{total}] {case.case_name} seed={out_seed} "
                f"noise={case.noise_level:g}",
                flush=True,
            )
            child_env = os.environ.copy()
            child_env.setdefault("PYTHONUNBUFFERED", "1")
            child_env = apply_v11_budget_env(child_env, args.case_budget_sec)
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
                    row = timeout_row(args, case, out_seed, time.time() - started, log_path, parent_timeout_sec)
                    rows.append(row)
                    result_json.write_text(json.dumps(json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
                    summarize(rows, out_dir)
                    print(f"[TIMEOUT {run_idx}/{total}] sec={time.time() - started:.1f}", flush=True)
                    continue
            if result_json.exists():
                row = json.loads(result_json.read_text(encoding="utf-8"))
            else:
                row = {
                    **timeout_row(args, case, out_seed, time.time() - started, log_path, parent_timeout_sec),
                    "timed_out": False,
                    "error": f"child exited without result; see {log_path}",
                }
            rows.append(row)
            summarize(rows, out_dir)
            print(
                f"[DONE {run_idx}/{total}] timeout={row.get('timed_out')} "
                f"skeleton={row.get('skeleton_recovery')} clean_mse={row.get('clean_test_mse')} "
                f"complexity={row.get('expr_complexity')} sec={row.get('runtime_sec')}",
                flush=True,
            )
    summarize(rows, out_dir)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "v11_noise_robustness"))
    parser.add_argument("--v11-path", default=str(DEFAULT_V11_PATH))
    parser.add_argument("--case-budget-sec", type=float, default=300.0)
    parser.add_argument("--timeout-grace-sec", type=float, default=30.0)
    parser.add_argument("--parent-timeout-sec", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--noise-levels", default="0,0.001,0.01")
    parser.add_argument("--dataset-manifest", default="", help="Optional fixed NoiseRobust-SR manifest.csv to read instead of generating splits.")
    parser.add_argument("--benchmarks", default=None, help="Optional comma-separated filter: Nguyen,Feynman,SRSD,Realistic.")
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
    args = parser.parse_args()
    if args.max_cases <= 0:
        args.max_cases = None
    if args.repeat_seeds <= 0:
        args.repeat_seeds = 1
    return args


def main():
    args = parse_args()
    apply_v11_budget_env(os.environ, args.case_budget_sec)
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
