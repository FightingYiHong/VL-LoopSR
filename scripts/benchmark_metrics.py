#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared benchmark result metrics for wrapper scripts."""

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
        return 1.0 if ss_res <= tol else 0.0
    return float(1.0 - ss_res / ss_tot)


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
    perfect_fit_r2_threshold: float = 0.999,
) -> Dict[str, Any]:
    expr = result.get("best_expr")
    result.update(expression_complexity(expr, feature_names))
    try:
        for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            preds = evaluate_expression(expr, df, feature_names)
            y_true = np.asarray(df["y"], dtype=float)
            mse = safe_mse(y_true, preds)
            mae = safe_mae(y_true, preds)
            r2 = safe_r2(y_true, preds)
            if result.get(f"best_{split_name}_mse") is None:
                result[f"best_{split_name}_mse"] = mse
            if result.get(f"{split_name}_mae") is None:
                result[f"{split_name}_mae"] = mae
            result[f"{split_name}_r2"] = r2
        test_r2 = _finite_float(result.get("test_r2"))
        result["perfect_fit_by_r2"] = bool(
            test_r2 is not None and test_r2 >= float(perfect_fit_r2_threshold)
        )
        result.setdefault("metric_eval_error", None)
    except Exception as exc:
        result.setdefault("train_r2", None)
        result.setdefault("val_r2", None)
        result.setdefault("test_r2", None)
        result.setdefault("perfect_fit_by_r2", False)
        result["metric_eval_error"] = repr(exc)
    return result
