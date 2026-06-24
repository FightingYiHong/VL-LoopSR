#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the VL-LoopSR pipeline on high-dimensional distractor-variable tasks.

The experiment matches the paper-table design:

* dimensions d = 200/500/1000;
* each formula uses only 2-5 true variables;
* distractors are independent irrelevant variables, correlated proxies, or
  nonlinear decoys;
* metrics track true-variable recall, false discovery, proxy misuse, exact and
  skeleton recovery, MSE, complexity, and runtime.

Ground-truth formulas and active-variable lists are kept out of the metadata
passed to VL-LoopSR. They are used only by this runner for post-hoc scoring.
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
sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_metrics import evaluate_expression, safe_mse
DEFAULT_V11_PATH = ROOT / "scripts" / "test_fey_v11_complexity_exact.py"
PASS_MSE_THRESHOLD = 100.0
EXACT_MSE_THRESHOLD = 1e-8
SKELETON_MSE_THRESHOLD = 1e-4


@dataclass(frozen=True)
class BaseFormula:
    formula_id: str
    source_family: str
    source_name: str
    true_variable_count: int
    structure_type: str
    expression_template: str
    fn: Callable[[np.ndarray], np.ndarray]
    input_low: tuple[float, ...]
    input_high: tuple[float, ...]


@dataclass(frozen=True)
class HighDimCase:
    case_index: int
    case_name: str
    base_formula_id: str
    source_family: str
    source_name: str
    dimension: int
    true_variable_count: int
    interference_type: str
    benchmark: str
    structure_type: str
    feature_names: list[str]
    true_variables: list[str]
    proxy_variables: list[str]
    nonlinear_decoy_variables: list[str]
    true_expression: str
    family_hint: str
    fn_name: str
    input_low: list[float]
    input_high: list[float]


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


def base_formula_catalog() -> list[BaseFormula]:
    """Benchmark-derived formulas used as low-dimensional cores.

    The formulas are not raw PMLB/SRBench tables. They are controlled symbolic
    cores adapted from common benchmark families, then embedded in higher
    dimensions with known distractor variables.
    """

    return [
        BaseFormula(
            "feynman_trig_interaction_k2",
            "Feynman",
            "Feynman-style trigonometric interaction",
            2,
            "trig_interaction",
            "1.25*{x1} + sin({x2}) + 0.35*{x1}*{x2}",
            lambda z: 1.25 * z[:, 0] + np.sin(z[:, 1]) + 0.35 * z[:, 0] * z[:, 1],
            (-2.0, -2.0),
            (2.0, 2.0),
        ),
        BaseFormula(
            "nguyen_rational_k2",
            "Nguyen",
            "Nguyen-style rational",
            2,
            "rational",
            "{x1}/(1 + {x2}**2) + 0.5*{x2}",
            lambda z: z[:, 0] / (1.0 + z[:, 1] ** 2) + 0.5 * z[:, 1],
            (-2.0, -2.0),
            (2.0, 2.0),
        ),
        BaseFormula(
            "livermore_exp_trig_k2",
            "Livermore",
            "Livermore-style exp-trig mix",
            2,
            "exp_trig",
            "exp(0.3*{x1}) - cos({x2})",
            lambda z: np.exp(0.3 * z[:, 0]) - np.cos(z[:, 1]),
            (-2.0, -2.0),
            (2.0, 2.0),
        ),
        BaseFormula(
            "srsd_log_sqrt_k2",
            "SRSD",
            "SRSD-style log-sqrt relation",
            2,
            "log_sqrt",
            "log(1 + {x1}**2) + sqrt(1 + {x2}**2)",
            lambda z: np.log1p(z[:, 0] ** 2) + np.sqrt(1.0 + z[:, 1] ** 2),
            (-2.0, -2.0),
            (2.0, 2.0),
        ),
        BaseFormula(
            "nguyen_poly_product_k2",
            "Nguyen",
            "Nguyen-style polynomial product",
            2,
            "polynomial_interaction",
            "{x1}**2 - 0.75*{x1}*{x2} + 0.5*{x2}**3",
            lambda z: z[:, 0] ** 2 - 0.75 * z[:, 0] * z[:, 1] + 0.5 * z[:, 1] ** 3,
            (-1.5, -1.5),
            (1.5, 1.5),
        ),
        BaseFormula(
            "feynman_decay_product_k2",
            "Feynman",
            "Feynman-style decayed interaction",
            2,
            "exp_interaction",
            "exp(-0.4*{x1}**2)*({x2} + 0.2)",
            lambda z: np.exp(-0.4 * z[:, 0] ** 2) * (z[:, 1] + 0.2),
            (-2.0, -2.0),
            (2.0, 2.0),
        ),
        BaseFormula(
            "feynman_sparse_interaction_k3",
            "Feynman",
            "Feynman-style sparse interaction",
            3,
            "sparse_interaction",
            "sin({x1}) + {x2}*{x3} + 0.25*{x3}**2",
            lambda z: np.sin(z[:, 0]) + z[:, 1] * z[:, 2] + 0.25 * z[:, 2] ** 2,
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "friedman_core_k3",
            "Friedman/PMLB-FRI",
            "Friedman-1 core",
            3,
            "friedman_interaction",
            "10*sin(pi*{x1}*{x2}) + 20*({x3} - 0.5)**2",
            lambda z: 10.0 * np.sin(np.pi * z[:, 0] * z[:, 1]) + 20.0 * (z[:, 2] - 0.5) ** 2,
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ),
        BaseFormula(
            "srsd_exp_log_k3",
            "SRSD",
            "SRSD-style exp-log mix",
            3,
            "exp_log",
            "exp(0.2*{x1}) + log(1 + {x2}**2) + {x3}",
            lambda z: np.exp(0.2 * z[:, 0]) + np.log1p(z[:, 1] ** 2) + z[:, 2],
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "nguyen_trig_poly_k3",
            "Nguyen",
            "Nguyen-style trig-polynomial mix",
            3,
            "trig_polynomial",
            "{x1}**2 + {x2}*sin({x3}) + 0.5*{x1}*{x2}",
            lambda z: z[:, 0] ** 2 + z[:, 1] * np.sin(z[:, 2]) + 0.5 * z[:, 0] * z[:, 1],
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "srsd_ratio_log_k3",
            "SRSD",
            "SRSD-style ratio-log mix",
            3,
            "rational_log",
            "({x1} + log(1 + {x2}**2))/(1 + 0.5*{x3}**2)",
            lambda z: (z[:, 0] + np.log1p(z[:, 1] ** 2)) / (1.0 + 0.5 * z[:, 2] ** 2),
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "feynman_wave_coupling_k3",
            "Feynman",
            "Feynman-style wave coupling",
            3,
            "trig_interaction",
            "sin({x1} + {x2}) + 0.4*{x2}*{x3}",
            lambda z: np.sin(z[:, 0] + z[:, 1]) + 0.4 * z[:, 1] * z[:, 2],
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "feynman_rational_trig_k4",
            "Feynman",
            "Feynman-style rational-trig mix",
            4,
            "rational_trig",
            "({x1} + 0.5*{x2}**2)/(1 + {x3}**2) + 0.4*cos({x4})",
            lambda z: (z[:, 0] + 0.5 * z[:, 1] ** 2) / (1.0 + z[:, 2] ** 2) + 0.4 * np.cos(z[:, 3]),
            (-2.0, -2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "friedman2_safe_k4",
            "Friedman/PMLB-FRI",
            "Friedman-2 safe-range motif",
            4,
            "sqrt_rational",
            "sqrt({x1}**2 + ({x2}*{x3} - 1/({x2}*{x4}))**2)",
            lambda z: np.sqrt(z[:, 0] ** 2 + (z[:, 1] * z[:, 2] - 1.0 / (z[:, 1] * z[:, 3])) ** 2),
            (0.5, 0.5, 0.5, 0.5),
            (2.0, 2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "srsd_energy_like_k4",
            "SRSD",
            "SRSD-style energy relation",
            4,
            "polynomial_trig",
            "{x1}*{x2}**2 + 0.1*sin({x3}) - 0.3*{x4}",
            lambda z: z[:, 0] * z[:, 1] ** 2 + 0.1 * np.sin(z[:, 2]) - 0.3 * z[:, 3],
            (-2.0, -2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "livermore_rational_interaction_k4",
            "Livermore",
            "Livermore-style rational interaction",
            4,
            "rational_interaction",
            "{x1}*{x2}/(1 + {x3}**2) + sin({x4})",
            lambda z: z[:, 0] * z[:, 1] / (1.0 + z[:, 2] ** 2) + np.sin(z[:, 3]),
            (-2.0, -2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "nguyen_nested_sqrt_k4",
            "Nguyen",
            "Nguyen-style nested square-root mix",
            4,
            "sqrt_interaction",
            "sqrt(1 + {x1}**2) + {x2}*sin({x3}) - 0.25*{x4}",
            lambda z: np.sqrt(1.0 + z[:, 0] ** 2) + z[:, 1] * np.sin(z[:, 2]) - 0.25 * z[:, 3],
            (-2.0, -2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "real_like_logistic_k4",
            "LLM-SRBench/real-like",
            "Real-like saturating interaction",
            4,
            "saturating_interaction",
            "{x1}/(1 + exp(-{x2})) + 0.3*{x3}**2 - cos({x4})",
            lambda z: z[:, 0] / (1.0 + np.exp(-z[:, 1])) + 0.3 * z[:, 2] ** 2 - np.cos(z[:, 3]),
            (-2.0, -2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "friedman1_full_k5",
            "Friedman/PMLB-FRI",
            "Friedman-1 full",
            5,
            "friedman_full",
            "10*sin(pi*{x1}*{x2}) + 20*({x3} - 0.5)**2 + 10*{x4} + 5*{x5}",
            lambda z: 10.0 * np.sin(np.pi * z[:, 0] * z[:, 1]) + 20.0 * (z[:, 2] - 0.5) ** 2 + 10.0 * z[:, 3] + 5.0 * z[:, 4],
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 1.0, 1.0),
        ),
        BaseFormula(
            "feynman_gravity_like_k5",
            "Feynman",
            "Feynman-style inverse-square relation",
            5,
            "inverse_square_interaction",
            "{x1}*{x2}/({x3}**2 + 0.5) + 0.2*{x4} - 0.1*{x5}",
            lambda z: z[:, 0] * z[:, 1] / (z[:, 2] ** 2 + 0.5) + 0.2 * z[:, 3] - 0.1 * z[:, 4],
            (-2.0, -2.0, -2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "srsd_exp_interaction_k5",
            "SRSD",
            "SRSD-style exp interaction",
            5,
            "exp_interaction",
            "exp(0.25*{x1}) + {x2}*{x3} - 0.3*{x4} + sin({x5})",
            lambda z: np.exp(0.25 * z[:, 0]) + z[:, 1] * z[:, 2] - 0.3 * z[:, 3] + np.sin(z[:, 4]),
            (-2.0, -2.0, -2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "real_like_saturating_k5",
            "LLM-SRBench/real-like",
            "Real-like saturating response",
            5,
            "saturating_log_interaction",
            "{x1}/(1 + sqrt({x2}**2)) + log(1 + {x3}**2) + tanh({x4}*{x5})",
            lambda z: z[:, 0] / (1.0 + np.sqrt(z[:, 1] ** 2)) + np.log1p(z[:, 2] ** 2) + np.tanh(z[:, 3] * z[:, 4]),
            (-2.0, -2.0, -2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "nguyen_mixed_rational_k5",
            "Nguyen",
            "Nguyen-style mixed rational interaction",
            5,
            "rational_interaction",
            "({x1}*{x2} + {x3})/(1 + {x4}**2) + 0.2*{x5}",
            lambda z: (z[:, 0] * z[:, 1] + z[:, 2]) / (1.0 + z[:, 3] ** 2) + 0.2 * z[:, 4],
            (-2.0, -2.0, -2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0, 2.0, 2.0),
        ),
        BaseFormula(
            "real_like_multiscale_k5",
            "LLM-SRBench/real-like",
            "Real-like multiscale smooth response",
            5,
            "exp_log_trig_interaction",
            "exp(0.15*{x1})*cos({x2}) + log(1 + {x3}**2) - {x4}/(1 + {x5}**2)",
            lambda z: np.exp(0.15 * z[:, 0]) * np.cos(z[:, 1]) + np.log1p(z[:, 2] ** 2) - z[:, 3] / (z[:, 4] ** 2 + 1.0),
            (-2.0, -2.0, -2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0, 2.0, 2.0),
        ),
    ]


FORMULA_CATALOG = base_formula_catalog()
FORMULA_BY_ID = {formula.formula_id: formula for formula in FORMULA_CATALOG}


def formula_fn(name: str) -> Callable[[np.ndarray], np.ndarray]:
    return FORMULA_BY_ID[name].fn


def render_expression(formula: BaseFormula, names: list[str]) -> str:
    mapping = {f"x{i + 1}": name for i, name in enumerate(names)}
    return formula.expression_template.format(**mapping)


def make_cases(max_cases: int | None = None) -> list[HighDimCase]:
    cases: list[HighDimCase] = []
    idx = 1
    for dimension in [200, 500, 1000]:
        feature_names = [f"x{i}" for i in range(1, dimension + 1)]
        for formula in FORMULA_CATALOG:
            true_count = formula.true_variable_count
            true_variables = feature_names[:true_count]
            expression = render_expression(formula, true_variables)
            for interference_type in ["independent_irrelevant", "correlated_proxy", "nonlinear_decoy"]:
                proxy_variables: list[str] = []
                nonlinear_decoy_variables: list[str] = []
                decoy_count = min(true_count * 2, 12, dimension - true_count)
                if interference_type == "correlated_proxy":
                    proxy_variables = feature_names[true_count : true_count + decoy_count]
                elif interference_type == "nonlinear_decoy":
                    nonlinear_decoy_variables = feature_names[true_count : true_count + decoy_count]
                cases.append(
                    HighDimCase(
                        case_index=idx,
                        case_name=f"hd_{formula.formula_id}_d{dimension}_{interference_type}",
                        base_formula_id=formula.formula_id,
                        source_family=formula.source_family,
                        source_name=formula.source_name,
                        dimension=dimension,
                        true_variable_count=true_count,
                        interference_type=interference_type,
                        benchmark=f"d={dimension}",
                        structure_type=formula.structure_type,
                        feature_names=feature_names,
                        true_variables=true_variables,
                        proxy_variables=proxy_variables,
                        nonlinear_decoy_variables=nonlinear_decoy_variables,
                        true_expression=expression,
                        family_hint=formula.structure_type,
                        fn_name=formula.formula_id,
                        input_low=list(formula.input_low),
                        input_high=list(formula.input_high),
                    )
                )
                idx += 1
    return cases[:max_cases] if max_cases else cases


def make_split(case: HighDimCase, n: int, rng: np.random.Generator) -> pd.DataFrame:
    true_count = case.true_variable_count
    low = np.asarray(case.input_low, dtype=float)
    high = np.asarray(case.input_high, dtype=float)
    z = rng.uniform(low, high, size=(n, true_count))
    x = rng.normal(0.0, 1.0, size=(n, case.dimension))
    x[:, :true_count] = z

    start = true_count
    decoy_count = min(true_count * 2, 12, case.dimension - start)
    if case.interference_type == "correlated_proxy":
        for local_idx in range(decoy_count):
            base = z[:, local_idx % true_count]
            scale = [1.0, 0.75, 1.25, -1.0][local_idx % 4]
            x[:, start + local_idx] = scale * base + 0.04 * rng.normal(size=n)
    elif case.interference_type == "nonlinear_decoy":
        transforms = [
            lambda v: np.sin(v),
            lambda v: np.cos(v),
            lambda v: v ** 2,
            lambda v: np.exp(np.clip(0.25 * v, -3, 3)),
            lambda v: np.tanh(v),
        ]
        for local_idx in range(decoy_count):
            base = z[:, local_idx % true_count]
            x[:, start + local_idx] = transforms[local_idx % len(transforms)](base) + 0.06 * rng.normal(size=n)

    y = formula_fn(case.fn_name)(z)
    df = pd.DataFrame(x, columns=case.feature_names)
    df["y"] = y
    return df


def load_case_splits(case: HighDimCase, seed: int, n_train: int, n_val: int, n_test: int):
    train_rng = np.random.default_rng(seed + case.case_index * 10_000 + 1)
    val_rng = np.random.default_rng(seed + case.case_index * 10_000 + 2)
    test_rng = np.random.default_rng(seed + case.case_index * 10_000 + 3)
    return (
        make_split(case, n_train, train_rng),
        make_split(case, n_val, val_rng),
        make_split(case, n_test, test_rng),
    )


def _abs_corr(a, b) -> float:
    try:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.size < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            return 0.0
        out = float(np.corrcoef(a, b)[0, 1])
        return abs(out) if math.isfinite(out) else 0.0
    except Exception:
        return 0.0


def _rank_array(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(dtype=float)


def select_highdim_prefilter_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_names: list[str],
    top_m: int,
    random_state: int,
) -> tuple[list[str], dict]:
    """Select a compact no-leakage feature subset before VL-LoopSR.

    The selector only uses train/validation X-y statistics. It deliberately
    combines univariate, nonlinear, and pair-interaction signals because some
    benchmark-derived cores, e.g. Friedman-style formulas, have weak individual
    marginal correlations but strong interaction evidence.
    """

    if top_m <= 0 or len(feature_names) <= top_m:
        return list(feature_names), {
            "enabled": False,
            "reason": "disabled_or_not_high_dimensional",
            "n_features_before": int(len(feature_names)),
            "n_features_after": int(len(feature_names)),
        }

    fit_df = pd.concat([train_df, val_df], ignore_index=True)
    y = fit_df["y"].to_numpy(dtype=float)
    scores = {name: 0.0 for name in feature_names}
    components = {name: {} for name in feature_names}

    y_rank = _rank_array(y)
    for name in feature_names:
        x = fit_df[name].to_numpy(dtype=float)
        pearson = _abs_corr(x, y)
        spearman = _abs_corr(_rank_array(x), y_rank)
        scores[name] += max(pearson, spearman)
        components[name]["corr"] = max(pearson, spearman)

    X = fit_df[feature_names].to_numpy(dtype=float)
    try:
        from sklearn.feature_selection import mutual_info_regression

        mi = mutual_info_regression(X, y, random_state=random_state, n_neighbors=5)
        if np.isfinite(mi).any() and float(np.max(mi)) > 0:
            mi = mi / (float(np.max(mi)) + 1e-12)
            for name, value in zip(feature_names, mi):
                value = float(value) if math.isfinite(float(value)) else 0.0
                scores[name] += 0.75 * value
                components[name]["mi"] = value
    except Exception as exc:
        for name in feature_names:
            components[name]["mi_error"] = repr(exc)

    try:
        from sklearn.ensemble import ExtraTreesRegressor

        model = ExtraTreesRegressor(
            n_estimators=64,
            max_depth=6,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=1,
        )
        model.fit(X, y)
        imp = np.asarray(model.feature_importances_, dtype=float)
        if np.isfinite(imp).any() and float(np.max(imp)) > 0:
            imp = imp / (float(np.max(imp)) + 1e-12)
            for name, value in zip(feature_names, imp):
                value = float(value) if math.isfinite(float(value)) else 0.0
                scores[name] += 0.75 * value
                components[name]["tree"] = value
    except Exception as exc:
        for name in feature_names:
            components[name]["tree_error"] = repr(exc)

    univariate_ranked = sorted(feature_names, key=lambda name: scores.get(name, 0.0), reverse=True)
    pair_pool = univariate_ranked[: min(32, len(univariate_ranked))]
    pair_bonus = {name: 0.0 for name in feature_names}
    pair_records = []
    for i, a_name in enumerate(pair_pool):
        a = fit_df[a_name].to_numpy(dtype=float)
        for b_name in pair_pool[i + 1 :]:
            b = fit_df[b_name].to_numpy(dtype=float)
            pair_score = max(
                _abs_corr(a * b, y),
                _abs_corr(np.sin(a * b), y),
                _abs_corr(a / (1.0 + b * b), y),
            )
            if pair_score <= 0:
                continue
            pair_records.append((pair_score, a_name, b_name))
    for pair_score, a_name, b_name in sorted(pair_records, reverse=True)[: max(8, top_m)]:
        pair_bonus[a_name] = max(pair_bonus[a_name], float(pair_score))
        pair_bonus[b_name] = max(pair_bonus[b_name], float(pair_score))
    for name, value in pair_bonus.items():
        scores[name] += 0.5 * value
        components[name]["pair"] = float(value)

    selected_by_score = sorted(feature_names, key=lambda name: scores.get(name, 0.0), reverse=True)[:top_m]
    selected_set = set(selected_by_score)
    selected = [name for name in feature_names if name in selected_set]
    trace = {
        "enabled": True,
        "n_features_before": int(len(feature_names)),
        "n_features_after": int(len(selected)),
        "top_m": int(top_m),
        "selected_features": selected,
        "top_scores": [
            {
                "feature": name,
                "score": float(scores.get(name, 0.0)),
                "components": components.get(name, {}),
            }
            for name in selected_by_score[: min(20, len(selected_by_score))]
        ],
        "top_pair_records": [
            {"score": float(score), "a": a, "b": b}
            for score, a, b in sorted(pair_records, reverse=True)[:20]
        ],
    }
    return selected, trace


def apply_highdim_prefilter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    top_m: int,
    random_state: int,
):
    selected, trace = select_highdim_prefilter_features(train_df, val_df, feature_names, top_m, random_state)
    if not trace.get("enabled"):
        return train_df, val_df, test_df, selected, trace
    cols = selected + ["y"]
    return train_df[cols].copy(), val_df[cols].copy(), test_df[cols].copy(), selected, trace


def import_v11_module(v11_path: Path, results_root: Path):
    if not v11_path.exists():
        raise FileNotFoundError(f"V11 file not found: {v11_path}")
    spec = importlib.util.spec_from_file_location("v11_highdim_runtime", str(v11_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["v11_highdim_runtime"] = module
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


def recompute_final_expression_mse(result: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    expr = result.get("best_expr")
    feature_names = [c for c in train_df.columns if c != "y"]
    out = {
        "v11_reported_train_mse": result.get("best_train_mse"),
        "v11_reported_val_mse": result.get("best_val_mse"),
        "v11_reported_test_mse": result.get("best_test_mse"),
    }
    try:
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            preds = evaluate_expression(expr, df, feature_names)
            out[f"best_{split_name}_mse"] = safe_mse(np.asarray(df["y"], dtype=float), preds)
        out["final_expr_metric_eval_error"] = None
    except Exception as exc:
        out["final_expr_metric_eval_error"] = repr(exc)
    return out


def score_result(v11, case: HighDimCase, result: dict, train_df: pd.DataFrame | None = None, val_df: pd.DataFrame | None = None, test_df: pd.DataFrame | None = None) -> dict:
    best_expr = result.get("best_expr")
    active = variables_used(v11, best_expr, case.feature_names)
    active_set = set(active)
    true_set = set(case.true_variables)
    proxy_set = set(case.proxy_variables)
    decoy_set = set(case.nonlinear_decoy_variables)

    true_variable_recall = len(active_set & true_set) / max(1, len(true_set))
    true_variable_precision = len(active_set & true_set) / max(1, len(active_set))
    false_count = len(active_set - true_set)
    false_variable_discovery_rate = false_count / max(1, len(active_set))
    proxy_misuse = bool(active_set & proxy_set)
    nonlinear_decoy_misuse = bool(active_set & decoy_set)
    irrelevant_misuse = bool(active_set - true_set - proxy_set - decoy_set)

    metric_updates = {}
    if train_df is not None and val_df is not None and test_df is not None:
        metric_updates = recompute_final_expression_mse(result, train_df, val_df, test_df)
        result.update({k: v for k, v in metric_updates.items() if k.startswith("best_")})

    test_mse = result.get("best_test_mse")
    try:
        test_mse_float = float(test_mse)
    except Exception:
        test_mse_float = float("inf")
    skeleton_recovery = bool(
        true_variable_recall >= 1.0
        and false_variable_discovery_rate <= 0.25
        and test_mse_float <= SKELETON_MSE_THRESHOLD
    )
    exact_recovery = bool(
        active_set == true_set
        and test_mse_float <= EXACT_MSE_THRESHOLD
    )

    return {
        "active_variables": "|".join(active),
        "true_variables": "|".join(case.true_variables),
        "proxy_variables": "|".join(case.proxy_variables),
        "nonlinear_decoy_variables": "|".join(case.nonlinear_decoy_variables),
        "true_variable_recall": true_variable_recall,
        "true_variable_precision": true_variable_precision,
        "false_variable_discovery_rate": false_variable_discovery_rate,
        "wrong_variable_count": false_count,
        "proxy_misuse": proxy_misuse,
        "nonlinear_decoy_misuse": nonlinear_decoy_misuse,
        "irrelevant_misuse": irrelevant_misuse,
        "skeleton_recovery": skeleton_recovery,
        "exact_recovery": exact_recovery,
        "passed": bool(test_mse_float <= PASS_MSE_THRESHOLD),
        "expr_complexity": result.get("expr_complexity") or expr_complexity(best_expr),
        **metric_updates,
    }


def run_child(args) -> int:
    started = time.time()
    result_path = Path(args.single_result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cases = make_cases(args.max_cases)
    case = cases[int(args.case_index) - 1]
    try:
        v11 = import_v11_module(Path(args.v11_path), Path(args.out_dir))
        train_df, val_df, test_df = load_case_splits(case, args.random_state + args.repeat_seed * 1_000_000, args.n_train, args.n_val, args.n_test)
        prefilter_top = int(args.prefilter_top_features or 0)
        train_df, val_df, test_df, selected_features, prefilter_trace = apply_highdim_prefilter(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_names=case.feature_names,
            top_m=prefilter_top,
            random_state=args.random_state + case.case_index * 1000 + args.repeat_seed,
        )
        row_meta = {
            "task_type": "high_dimensional_interference",
            "dataset_dir": "high_dimensional_interference",
            "difficulty": case.benchmark,
            "base_name": case.case_name,
            "suite": "high_dimensional_interference",
            "structure_type": case.structure_type,
            "interference_type": case.interference_type,
            "dimension": case.dimension,
            "true_variable_count": case.true_variable_count,
            "base_formula_id": case.base_formula_id,
            "source_family": case.source_family,
            "source_name": case.source_name,
            "original_dimension": case.dimension,
            "prefilter_top_features": prefilter_top,
            "prefilter_selected_feature_count": len(selected_features),
            "true_expression": None,
        }
        with TemporaryDirectory(prefix="v11_highdim_tmp_") as tmpdir_str:
            dataset = v11.build_dataset_from_explicit_splits(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                tmpdir=Path(tmpdir_str),
            )
            dataset.source_tag = "high_dimensional_interference"
            result = v11._run_core_pipeline(dataset=dataset, row_meta=row_meta)
        out = {
            **result,
            **row_meta,
            "method": "vl_loopsr",
            "case_index": case.case_index,
            "case_name": case.case_name,
            "repeat_seed": int(args.repeat_seed),
            "budget_sec": float(args.case_budget_sec),
            "timed_out": False,
            "runtime_sec": float(result.get("runtime_sec") or (time.time() - started)),
            "true_expression_for_scoring": case.true_expression,
            "family_hint": case.family_hint,
            "prefilter_enabled": bool(prefilter_trace.get("enabled", False)),
            "prefilter_selected_features": "|".join(selected_features),
            "prefilter_trace": json.dumps(json_safe(prefilter_trace), ensure_ascii=False),
        }
        out.update(score_result(v11, case, out, train_df=train_df, val_df=val_df, test_df=test_df))
    except BaseException as exc:
        case_meta = asdict(case)
        out = {
            "method": "vl_loopsr",
            "case_index": int(args.case_index),
            "case_name": case_meta.get("case_name"),
            "base_formula_id": case_meta.get("base_formula_id"),
            "source_family": case_meta.get("source_family"),
            "source_name": case_meta.get("source_name"),
            "dimension": case_meta.get("dimension"),
            "original_dimension": case_meta.get("dimension"),
            "true_variable_count": case_meta.get("true_variable_count"),
            "interference_type": case_meta.get("interference_type"),
            "repeat_seed": int(args.repeat_seed),
            "budget_sec": float(args.case_budget_sec),
            "timed_out": "time" in repr(exc).lower() or "timeout" in repr(exc).lower(),
            "runtime_sec": float(time.time() - started),
            "valid_formula_found": False,
            "passed": False,
            "best_expr": None,
            "best_val_mse": None,
            "best_test_mse": None,
            "true_expression_for_scoring": case_meta.get("true_expression"),
            "true_variables": "|".join(case_meta.get("true_variables") or []),
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
    result_path.write_text(json.dumps(json_safe(out), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def timeout_row(args, case: HighDimCase, runtime_sec: float, log_path: Path, parent_timeout_sec: float):
    return {
        "method": "vl_loopsr",
        "case_index": int(case.case_index),
        "case_name": case.case_name,
        "base_formula_id": case.base_formula_id,
        "source_family": case.source_family,
        "source_name": case.source_name,
        "dimension": int(case.dimension),
        "true_variable_count": int(case.true_variable_count),
        "interference_type": case.interference_type,
        "repeat_seed": int(args.repeat_seed),
        "budget_sec": float(args.case_budget_sec),
        "timed_out": True,
        "runtime_sec": float(runtime_sec),
        "valid_formula_found": False,
        "passed": False,
        "best_expr": None,
        "best_val_mse": None,
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
        "skeleton_recovery",
        "proxy_misuse",
        "nonlinear_decoy_misuse",
        "irrelevant_misuse",
        "true_variable_recall",
        "true_variable_precision",
        "false_variable_discovery_rate",
        "wrong_variable_count",
        "true_variable_count",
        "dimension",
        "interference_type",
        "best_test_mse",
        "runtime_sec",
        "expr_complexity",
    ]:
        if column not in df:
            df[column] = np.nan
    df.to_csv(out_dir / "all_high_dimensional_interference_vl_loopsr_results.csv", index=False)
    summary = (
        df.groupby(["dimension", "interference_type", "method"], dropna=False)
        .agg(
            n=("case_index", "count"),
            timeout_rate=("timed_out", "mean"),
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
    summary.to_csv(out_dir / "summary_high_dimensional_interference_vl_loopsr.csv", index=False)
    plot_figures(df, summary, out_dir)
    write_design_doc(out_dir, summary)


def plot_figures(df: pd.DataFrame, summary: pd.DataFrame, out_dir: Path):
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import seaborn as sns
    except Exception:
        sns = None
    if sns is not None:
        sns.set_theme(style="whitegrid")
    else:
        plt.style.use("ggplot")

    heat = summary.copy()
    heat["interference_label"] = heat["interference_type"].map(
        {
            "independent_irrelevant": "independent",
            "correlated_proxy": "proxy",
            "nonlinear_decoy": "nonlinear",
        }
    ).fillna(heat["interference_type"])
    pivot_recall = heat.pivot_table(index="dimension", columns="interference_label", values="true_variable_recall", aggfunc="mean")
    pivot_error = heat.pivot_table(index="dimension", columns="interference_label", values="false_variable_discovery_rate", aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    if sns is not None:
        sns.heatmap(pivot_recall, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, ax=axes[0])
    else:
        im0 = axes[0].imshow(pivot_recall.fillna(0.0).to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
        axes[0].set_xticks(range(len(pivot_recall.columns)), pivot_recall.columns, rotation=25, ha="right")
        axes[0].set_yticks(range(len(pivot_recall.index)), pivot_recall.index)
        for i in range(len(pivot_recall.index)):
            for j in range(len(pivot_recall.columns)):
                value = pivot_recall.iloc[i, j]
                axes[0].text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title("true variable recall")
    if sns is not None:
        sns.heatmap(pivot_error, annot=True, fmt=".2f", cmap="OrRd", vmin=0, vmax=1, ax=axes[1])
    else:
        im1 = axes[1].imshow(pivot_error.fillna(0.0).to_numpy(), cmap="OrRd", vmin=0, vmax=1, aspect="auto")
        axes[1].set_xticks(range(len(pivot_error.columns)), pivot_error.columns, rotation=25, ha="right")
        axes[1].set_yticks(range(len(pivot_error.index)), pivot_error.index)
        for i in range(len(pivot_error.index)):
            for j in range(len(pivot_error.columns)):
                value = pivot_error.iloc[i, j]
                axes[1].text(j, i, "" if pd.isna(value) else f"{value:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("wrong/proxy variable rate")
    fig.suptitle("Fig.3a High-Dim Interference Heatmap")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3a_high_dim_interference_heatmap.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(8.5, 4.8))
    if sns is not None:
        sns.scatterplot(
            data=df,
            x="true_variable_recall",
            y="false_variable_discovery_rate",
            hue="interference_type",
            style="dimension",
            size="true_variable_count",
            sizes=(55, 140),
            alpha=0.85,
        )
    else:
        markers = {200: "o", 500: "s", 1000: "^"}
        colors = {
            "independent_irrelevant": "#2b6cb0",
            "correlated_proxy": "#c05621",
            "nonlinear_decoy": "#2f855a",
        }
        for _, row in df.iterrows():
            size = 45 + 18 * float(row.get("true_variable_count", 3) or 3)
            plt.scatter(
                row.get("true_variable_recall"),
                row.get("false_variable_discovery_rate"),
                s=size,
                marker=markers.get(int(row.get("dimension", 50) or 50), "o"),
                color=colors.get(row.get("interference_type"), "#444444"),
                alpha=0.82,
                label=str(row.get("interference_type")),
            )
        handles, labels = plt.gca().get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        plt.legend(dedup.values(), dedup.keys(), fontsize=8, frameon=True)
    plt.xlim(-0.04, 1.04)
    plt.ylim(-0.04, 1.04)
    plt.title("Fig.3b Variable Selection: Recall vs False Discovery")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3b_variable_selection_scatter.png", dpi=180)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
    if sns is not None:
        sns.lineplot(data=df, x="dimension", y="skeleton_recovery", hue="interference_type", marker="o", errorbar=None, ax=ax1)
    else:
        grouped = df.groupby(["interference_type", "dimension"], dropna=False)["skeleton_recovery"].mean().reset_index()
        for interference_type, sub in grouped.groupby("interference_type"):
            sub = sub.sort_values("dimension")
            ax1.plot(sub["dimension"], sub["skeleton_recovery"], marker="o", label=str(interference_type))
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_ylabel("Skeleton Recovery")
    ax2 = ax1.twinx()
    runtime = df.groupby("dimension", dropna=False)["runtime_sec"].median().reset_index()
    ax2.plot(runtime["dimension"], runtime["runtime_sec"], color="#333333", linestyle="--", marker="s", label="median runtime")
    ax2.set_ylabel("Median Runtime (s)")
    ax1.set_title("Fig.3c Dimension Scaling")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(fig_dir / "fig3c_dimension_scaling_runtime.png", dpi=180)
    plt.close(fig)


def write_design_doc(out_dir: Path, summary: pd.DataFrame):
    doc = out_dir / "high_dimensional_interference_design.md"
    table = summary.copy()
    for col in [
        "timeout_rate",
        "pass_rate",
        "exact_recovery",
        "skeleton_recovery",
        "true_variable_recall",
        "false_variable_discovery_rate",
        "proxy_misuse_rate",
        "median_test_mse",
        "median_runtime_sec",
    ]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    lines = [
        "# High-Dimensional Interference Experiment",
        "",
        "Base method: `scripts/test_fey_v11_complexity_exact.py` (V11).",
        "",
        "Design:",
        "",
        "- Dimensions: `d=200,500,1000`.",
        "- True variables per formula: `k=2,3,4,5`; all remaining dimensions are distractors.",
        "- Interference types: independent irrelevant variables, correlated proxy variables, nonlinear decoys.",
        "- V11 receives only the generated train/val/test data plus generic case metadata; true variables/formulas are used only for scoring.",
        "",
        "Metrics:",
        "",
        "- `true_variable_recall`, `false_variable_discovery_rate`, `proxy_misuse_rate`.",
        "- `exact_recovery`, `skeleton_recovery`, `best_test_mse`, `expr_complexity`, `runtime_sec`.",
        "",
        "Figure mapping:",
        "",
        "- `figures/fig3a_high_dim_interference_heatmap.png`: recall and false-discovery heatmaps.",
        "- `figures/fig3b_variable_selection_scatter.png`: recall vs false discovery; lower-right is best.",
        "- `figures/fig3c_dimension_scaling_runtime.png`: skeleton recovery as dimension grows, with runtime on the right axis.",
        "",
        "Current summary:",
        "",
        table.to_markdown(index=False),
        "",
    ]
    doc.write_text("\n".join(lines), encoding="utf-8")


def run_parent(args) -> int:
    out_dir = Path(args.out_dir)
    result_dir = out_dir / "case_results" / "vl_loopsr"
    log_dir = out_dir / "case_logs" / "vl_loopsr"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    parent_timeout_sec = float(args.parent_timeout_sec or 0.0)
    if parent_timeout_sec <= 0:
        parent_timeout_sec = float(args.case_budget_sec) + float(args.timeout_grace_sec)

    cases = make_cases(args.max_cases)
    selected = []
    for case in cases:
        row = asdict(case)
        row["true_variables"] = "|".join(case.true_variables)
        row["proxy_variables"] = "|".join(case.proxy_variables)
        row["nonlinear_decoy_variables"] = "|".join(case.nonlinear_decoy_variables)
        selected.append(row)
    pd.DataFrame(selected).to_csv(out_dir / "selected_high_dimensional_interference_cases.csv", index=False)
    manifest = {
        "method": "vl_loopsr",
        "experiment": "high_dimensional_interference",
        "n_cases": len(cases),
        "dimensions": [200, 500, 1000],
        "true_variable_counts": [2, 3, 4, 5],
        "interference_types": ["independent_irrelevant", "correlated_proxy", "nonlinear_decoy"],
        "n_train": int(args.n_train),
        "n_val": int(args.n_val),
        "n_test": int(args.n_test),
        "case_budget_sec": float(args.case_budget_sec),
        "timeout_grace_sec": float(args.timeout_grace_sec),
        "parent_timeout_sec": float(parent_timeout_sec),
        "prefilter_top_features": int(args.prefilter_top_features or 0),
        "v11_path": str(args.v11_path),
    }
    (out_dir / "manifest_high_dimensional_interference_vl_loopsr.json").write_text(
        json.dumps(json_safe(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = []
    for case in cases:
        safe = sanitize_name(case.case_name)
        result_json = result_dir / f"{case.case_index:03d}_{safe}_seed{args.repeat_seed}.json"
        log_path = log_dir / f"{case.case_index:03d}_{safe}_seed{args.repeat_seed}.log"
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
            "--prefilter-top-features",
            str(args.prefilter_top_features or 0),
            "--max-cases",
            str(args.max_cases or 0),
            "--random-state",
            str(args.random_state),
            "--single-result-json",
            str(result_json),
        ]
        started = time.time()
        print(f"[RUN {case.case_index}/{len(cases)}] {case.case_name}", flush=True)
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        child_env["LLMSR_MAX_RUNTIME_PER_TASK_SEC"] = str(args.case_budget_sec)
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
                row = timeout_row(args, case, time.time() - started, log_path, parent_timeout_sec)
                rows.append(row)
                result_json.write_text(json.dumps(json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
                summarize(rows, out_dir)
                print(f"[TIMEOUT {case.case_index}/{len(cases)}] sec={time.time() - started:.1f}", flush=True)
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
            f"[DONE {case.case_index}/{len(cases)}] timeout={row.get('timed_out')} "
            f"recall={row.get('true_variable_recall')} fdr={row.get('false_variable_discovery_rate')} "
            f"mse={row.get('best_test_mse')} sec={row.get('runtime_sec')}",
            flush=True,
        )
    summarize(rows, out_dir)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(ROOT / "reports" / "v11_high_dimensional_interference"))
    parser.add_argument("--v11-path", default=str(DEFAULT_V11_PATH))
    parser.add_argument("--case-budget-sec", type=float, default=900.0)
    parser.add_argument("--timeout-grace-sec", type=float, default=30.0)
    parser.add_argument("--parent-timeout-sec", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-val", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--repeat-seed", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--prefilter-top-features", type=int, default=int(os.environ.get("HIGHDIM_V11_PREFILTER_TOP_FEATURES", "0")))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-timeouts", action="store_true")
    parser.add_argument("--rerun-failures", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--single-result-json", default="")
    args = parser.parse_args()
    if args.max_cases <= 0:
        args.max_cases = None
    return args


def main():
    args = parse_args()
    os.environ["LLMSR_MAX_RUNTIME_PER_TASK_SEC"] = str(args.case_budget_sec)
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
