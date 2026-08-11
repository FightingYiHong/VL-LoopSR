#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared benchmark metrics, recovery checks, and Pareto utilities."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Optional

import numpy as np
import sympy


_SYMPY_LOCALS = {
    "Abs": sympy.Abs,
    "abs": sympy.Abs,
    "sqrt": sympy.sqrt,
    "log": sympy.log,
    "exp": sympy.exp,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "id": lambda x: x,
    "pi": sympy.pi,
    "E": sympy.E,
    "e": sympy.E,
}

NUMERICAL_FIT_R2_THRESHOLD = 0.999
EARLY_STOP_TRAIN_R2_THRESHOLD = 0.99999


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def safe_mse(y_true, y_pred) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return None
    return float(np.mean((y_true - y_pred) ** 2))


def safe_rmse(y_true, y_pred) -> Optional[float]:
    mse = safe_mse(y_true, y_pred)
    return None if mse is None else float(math.sqrt(mse))


def safe_nmse(y_true, y_pred, tol: float = 1e-12) -> Optional[float]:
    """Return MSE normalized by the population variance of the target."""
    y_true = np.asarray(y_true, dtype=float)
    mse = safe_mse(y_true, y_pred)
    if mse is None:
        return None
    variance = float(np.var(y_true))
    if variance <= tol:
        return None
    return float(mse / variance)


def safe_nrmse(y_true, y_pred, tol: float = 1e-12) -> Optional[float]:
    """Return RMSE normalized by target standard deviation."""
    nmse = safe_nmse(y_true, y_pred, tol=tol)
    return None if nmse is None else float(math.sqrt(max(0.0, nmse)))


def safe_mae(y_true, y_pred) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return None
    return float(np.mean(np.abs(y_true - y_pred)))


def safe_r2(y_true, y_pred, tol: float = 1e-12) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= tol:
        return None
    return float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true, y_pred) -> Dict[str, Optional[float]]:
    """Compute the common numerical-fit metrics from one prediction vector."""
    return {
        "mse": safe_mse(y_true, y_pred),
        "rmse": safe_rmse(y_true, y_pred),
        "nmse": safe_nmse(y_true, y_pred),
        "nrmse": safe_nrmse(y_true, y_pred),
        "mae": safe_mae(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
    }


def _expression_symbols(expr: str, feature_names: Iterable[str]):
    locals_map = dict(_SYMPY_LOCALS)
    expr = str(expr).replace("^", "**")
    feature_names = [str(x) for x in feature_names]
    for name in feature_names:
        if name:
            locals_map[name] = sympy.Symbol(name)
    for idx in range(len(feature_names)):
        locals_map[f"x{idx}"] = sympy.Symbol(f"x{idx}")
        locals_map[f"x{idx + 1}"] = sympy.Symbol(f"x{idx + 1}")
    parsed = sympy.sympify(str(expr), locals=locals_map)
    return parsed, locals_map


def expression_complexity(expr: Any, feature_names: Iterable[str] = ()) -> Dict[str, Any]:
    expr_text = "" if expr is None else str(expr).strip()
    if not expr_text:
        return {
            "expr_complexity": None,
            "expr_depth": None,
            "expr_string_length": 0,
            "expr_sympy_ops": None,
        }
    try:
        parsed, _ = _expression_symbols(expr_text, feature_names)
        return {
            "expr_complexity": int(sum(1 for _ in sympy.preorder_traversal(parsed))),
            "expr_depth": int(_sympy_depth(parsed)),
            "expr_string_length": int(len(expr_text.replace(" ", ""))),
            "expr_sympy_ops": int(sympy.count_ops(parsed, visual=False)),
        }
    except Exception:
        return {
            "expr_complexity": int(len(expr_text.replace(" ", ""))),
            "expr_depth": None,
            "expr_string_length": int(len(expr_text.replace(" ", ""))),
            "expr_sympy_ops": None,
        }


def strict_formula_recovery(
    predicted_expr: Any,
    target_expr: Any,
    feature_names: Iterable[str] = (),
) -> Optional[bool]:
    """Check exact symbolic equivalence after SymPy simplification.

    ``None`` means that one of the expressions could not be parsed, so callers
    can distinguish "not recovered" from "not evaluable".
    """
    if predicted_expr is None or target_expr is None:
        return None
    if not str(predicted_expr).strip() or not str(target_expr).strip():
        return None
    try:
        predicted, _ = _expression_symbols(str(predicted_expr), feature_names)
        target, _ = _expression_symbols(str(target_expr), feature_names)
        predicted = _canonicalize_feature_symbols(predicted, feature_names)
        target = _canonicalize_feature_symbols(target, feature_names)
        difference = sympy.cancel(sympy.together(predicted - target))
        return bool(sympy.simplify(difference) == 0)
    except Exception:
        return None


def srbench_formula_recovery(
    predicted_expr: Any,
    target_expr: Any,
    feature_names: Iterable[str] = (),
) -> Optional[bool]:
    """Apply the SRBench symbolic-recovery convention.

    A prediction is recovered when it is strictly equivalent to the target,
    differs from it only by an additive constant, or has a non-zero constant
    ratio to it. ``None`` means that symbolic comparison was not evaluable.
    """
    if predicted_expr is None or target_expr is None:
        return None
    if not str(predicted_expr).strip() or not str(target_expr).strip():
        return None
    try:
        predicted, _ = _expression_symbols(str(predicted_expr), feature_names)
        target, _ = _expression_symbols(str(target_expr), feature_names)
        predicted = _canonicalize_feature_symbols(predicted, feature_names)
        target = _canonicalize_feature_symbols(target, feature_names)

        difference = sympy.simplify(sympy.cancel(sympy.together(predicted - target)))
        if difference == 0 or not difference.free_symbols:
            return True

        ratio = sympy.simplify(sympy.cancel(sympy.together(predicted / target)))
        if not ratio.free_symbols and ratio != 0 and ratio.is_finite is not False:
            return True
        return False
    except Exception:
        return None


def _canonicalize_feature_symbols(expr, feature_names: Iterable[str]):
    """Map zero- or one-indexed x aliases to the same canonical variables."""
    feature_names = [str(name) for name in feature_names]
    symbol_names = {str(symbol) for symbol in getattr(expr, "free_symbols", set())}
    zero_indexed = "x0" in symbol_names
    replacements = {}
    for idx, feature_name in enumerate(feature_names):
        canonical = sympy.Symbol(f"__feature_{idx}")
        if zero_indexed:
            replacements[sympy.Symbol(f"x{idx}")] = canonical
            if not re.fullmatch(r"x\d+", feature_name):
                replacements[sympy.Symbol(feature_name)] = canonical
        else:
            replacements[sympy.Symbol(feature_name)] = canonical
            replacements.setdefault(sympy.Symbol(f"x{idx + 1}"), canonical)
    return expr.xreplace(replacements)


def pareto_front_indices(errors, complexities) -> list[int]:
    """Return non-dominated indices for minimization of error and complexity."""
    error_values = [_finite_float(value) for value in errors]
    complexity_values = [_finite_float(value) for value in complexities]
    front = []
    for idx, (error, complexity) in enumerate(zip(error_values, complexity_values)):
        if error is None or complexity is None:
            continue
        dominated = False
        for other_idx, (other_error, other_complexity) in enumerate(
            zip(error_values, complexity_values)
        ):
            if other_idx == idx or other_error is None or other_complexity is None:
                continue
            weakly_better = other_error <= error and other_complexity <= complexity
            strictly_better = other_error < error or other_complexity < complexity
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(idx)
    return front


def pareto_knee_index(errors, complexities) -> Optional[int]:
    """Choose a knee from the error-complexity Pareto front.

    The knee is the point furthest from the chord joining the simplest and
    lowest-error Pareto solutions after log-error and min-max normalization.
    """
    front = pareto_front_indices(errors, complexities)
    if not front:
        return None
    if len(front) == 1:
        return front[0]

    points = []
    for idx in front:
        error = max(float(errors[idx]), 1e-300)
        points.append((idx, float(complexities[idx]), math.log10(error)))
    points.sort(key=lambda item: (item[1], item[2]))

    complexity_array = np.asarray([item[1] for item in points], dtype=float)
    error_array = np.asarray([item[2] for item in points], dtype=float)
    complexity_span = float(np.ptp(complexity_array))
    error_span = float(np.ptp(error_array))
    x = (
        (complexity_array - float(np.min(complexity_array))) / complexity_span
        if complexity_span > 0
        else np.zeros_like(complexity_array)
    )
    y = (
        (error_array - float(np.min(error_array))) / error_span
        if error_span > 0
        else np.zeros_like(error_array)
    )

    start = np.asarray([x[0], y[0]], dtype=float)
    end = np.asarray([x[-1], y[-1]], dtype=float)
    chord = end - start
    chord_norm = float(np.linalg.norm(chord))
    if chord_norm <= 1e-12:
        return min(points, key=lambda item: (item[1], item[2]))[0]

    distances = []
    for point_x, point_y in zip(x, y):
        offset = np.asarray([point_x, point_y], dtype=float) - start
        distance = abs(float(chord[0] * offset[1] - chord[1] * offset[0])) / chord_norm
        distances.append(distance)
    max_distance = max(distances)
    candidates = [
        points[idx]
        for idx, distance in enumerate(distances)
        if math.isclose(distance, max_distance, rel_tol=1e-12, abs_tol=1e-12)
    ]
    return min(candidates, key=lambda item: (item[1], item[2]))[0]


def _sympy_depth(expr) -> int:
    args = getattr(expr, "args", ())
    if not args:
        return 0
    return 1 + max(_sympy_depth(arg) for arg in args)


def _feature_value_map(df, feature_names, zero_indexed_x: bool = False):
    columns = [c for c in df.columns if c != "y"]
    values = {str(col): np.asarray(df[col], dtype=float) for col in columns}
    for idx, col in enumerate(columns):
        arr = np.asarray(df[col], dtype=float)
        if zero_indexed_x:
            values[f"x{idx}"] = arr
        else:
            values.setdefault(f"x{idx}", arr)
            values.setdefault(f"x{idx + 1}", arr)
    for idx, name in enumerate(feature_names or []):
        if idx < len(columns):
            values.setdefault(str(name), np.asarray(df[columns[idx]], dtype=float))
    return values


def evaluate_expression(expr: Any, df, feature_names: Iterable[str]):
    expr_text = "" if expr is None else str(expr).strip()
    if not expr_text:
        return None
    parsed, _ = _expression_symbols(expr_text, feature_names)
    symbols = sorted(parsed.free_symbols, key=lambda s: str(s))
    zero_indexed_x = any(str(s) == "x0" for s in symbols)
    value_map = _feature_value_map(df, feature_names, zero_indexed_x=zero_indexed_x)
    if not symbols:
        return np.full(len(df), float(parsed), dtype=float)
    missing = [str(s) for s in symbols if str(s) not in value_map]
    if missing:
        raise ValueError(f"expression uses unknown symbols: {missing}")
    func = sympy.lambdify(symbols, parsed, modules=["numpy"])
    preds = func(*[value_map[str(s)] for s in symbols])
    preds = np.asarray(preds, dtype=float)
    if preds.shape == ():
        preds = np.full(len(df), float(preds), dtype=float)
    preds = preds.reshape(-1)
    if len(preds) != len(df):
        raise ValueError(f"prediction length mismatch: {len(preds)} vs {len(df)}")
    if not np.isfinite(preds).all():
        raise ValueError("non-finite predictions")
    return preds


def enrich_result_metrics(
    result: Dict[str, Any],
    train_df,
    val_df,
    test_df,
    feature_names: Iterable[str],
    perfect_fit_r2_threshold: float = NUMERICAL_FIT_R2_THRESHOLD,
    early_stop_train_r2_threshold: float = EARLY_STOP_TRAIN_R2_THRESHOLD,
    ground_truth_expression: Any = None,
) -> Dict[str, Any]:
    expr = result.get("best_expr")
    result.update(expression_complexity(expr, feature_names))
    try:
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            preds = evaluate_expression(expr, df, feature_names)
            y_true = np.asarray(df["y"], dtype=float)
            metrics = regression_metrics(y_true, preds)
            if result.get(f"best_{split_name}_mse") is None:
                result[f"best_{split_name}_mse"] = metrics["mse"]
            for metric_name, metric_value in metrics.items():
                result[f"{split_name}_{metric_name}"] = metric_value
        test_r2 = _finite_float(result.get("test_r2"))
        train_r2 = _finite_float(result.get("train_r2"))
        result["numerical_complete_fit"] = bool(
            test_r2 is not None and test_r2 > float(perfect_fit_r2_threshold)
        )
        result["perfect_fit_by_r2"] = bool(
            result["numerical_complete_fit"]
        )
        result["early_stop_train_fit"] = bool(
            train_r2 is not None and train_r2 > float(early_stop_train_r2_threshold)
        )
        result.setdefault("metric_eval_error", None)
    except Exception as exc:
        result.setdefault("train_r2", None)
        result.setdefault("val_r2", None)
        result.setdefault("test_r2", None)
        result.setdefault("perfect_fit_by_r2", False)
        result.setdefault("numerical_complete_fit", False)
        result.setdefault("early_stop_train_fit", False)
        result["metric_eval_error"] = repr(exc)

    target_expr = ground_truth_expression
    if target_expr is None:
        for key in (
            "true_expression",
            "true_expression_for_scoring",
            "ground_truth_expression",
            "target_expression",
        ):
            if result.get(key) is not None:
                target_expr = result.get(key)
                break
    strict_recovery = strict_formula_recovery(expr, target_expr, feature_names)
    srbench_recovery = srbench_formula_recovery(expr, target_expr, feature_names)
    result["strict_formula_recovery"] = bool(strict_recovery) if strict_recovery is not None else False
    result["strict_formula_recovery_evaluable"] = strict_recovery is not None
    result["srbench_formula_recovery"] = bool(srbench_recovery) if srbench_recovery is not None else False
    result["srbench_formula_recovery_evaluable"] = srbench_recovery is not None
    return result
