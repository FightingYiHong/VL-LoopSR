#!/usr/bin/env python3
"""Validation-only expression rescue for the 238-case VEGA-SR SRSD run.

The rescue stage never uses the target formula or test values for generation or
selection.  It builds compact candidates from:

* algebraic simplifications of the final VEGA-SR expression;
* rounded/rationalized fitted constants;
* removal of one potentially irrelevant variable at a time;
* a small predeclared bank of low-complexity symbolic skeletons.

Skeleton coefficients are fitted on SRSD train data.  Candidate selection uses
train and validation NMSE only.  The true formula and test split are read only
after the selected expression has been frozen, for final reporting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
import signal
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sympy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION = (
    ROOT
    / "runs"
    / "paper_experiments"
    / "01_standard_recovery"
    / "standard_search_efficiency_full_20260724_152138"
    / "srsd"
    / "srsd_selected_tasks.csv"
)
DEFAULT_RESULTS = (
    ROOT
    / "reports"
    / "posthoc_recompute_inputs_20260724"
    / "01_main"
    / "results"
    / "archive"
    / "ours_paper_run"
    / "srsd.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "runs"
    / "paper_experiments"
    / "01_standard_recovery"
    / "srsd_expression_rescue"
)

import sys

sys.path.insert(0, str(ROOT / "scripts"))
from benchmark_metrics import (  # noqa: E402
    evaluate_expression,
    expression_complexity,
    srbench_formula_recovery,
    strict_formula_recovery,
)


SYMPY_LOCALS = {
    "Abs": sympy.Abs,
    "abs": sympy.Abs,
    "sqrt": sympy.sqrt,
    "log": sympy.log,
    "exp": sympy.exp,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "pi": sympy.pi,
    "E": sympy.E,
}


def load_txt(path: str) -> pd.DataFrame:
    values = np.loadtxt(path)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    columns = [f"x{i + 1}" for i in range(values.shape[1] - 1)] + ["y"]
    return pd.DataFrame(values, columns=columns)


def scale_invariant_nmse(y_true, y_pred) -> float | None:
    """Return NMSE without rejecting physically small but varying targets.

    ``benchmark_metrics.safe_nmse`` intentionally treats variances below
    1e-12 as numerically constant.  Several dimensionful SRSD targets vary on
    much smaller absolute scales, so that absolute cutoff would make otherwise
    valid cases unavailable.  Here only an exactly zero/non-finite variance is
    unavailable; the ratio itself remains scale invariant.
    """
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if truth.size == 0 or prediction.shape != truth.shape:
        return None
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        return None
    variance = float(np.var(truth))
    if not np.isfinite(variance) or variance <= 0.0:
        return None
    mse = float(np.mean((truth - prediction) ** 2))
    if not np.isfinite(mse):
        return None
    return float(mse / variance)


def _formula_timeout_handler(signum, frame) -> None:
    raise TimeoutError("symbolic formula comparison timed out")


def formula_recovery_with_timeout(
    predicted: str,
    target: str,
    feature_names: list[str],
    seconds: float = 8.0,
) -> tuple[bool | None, bool | None, str]:
    """Evaluate both documented recovery definitions with a bounded runtime."""
    predicted_parsed = parse_expression(predicted)
    target_parsed = parse_expression(target)
    if predicted_parsed is None or target_parsed is None:
        return None, None, "parse_unavailable"
    # A genuinely constant expression cannot equal, differ only by a constant
    # from, or have a non-zero constant ratio to a non-constant expression.
    # This safe short-circuit avoids costly expansion of large SRSD formulas.
    if bool(predicted_parsed.free_symbols) != bool(target_parsed.free_symbols):
        return False, False, "constant_nonconstant_short_circuit"

    previous = signal.signal(signal.SIGALRM, _formula_timeout_handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        strict = strict_formula_recovery(predicted, target, feature_names)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        srbench = srbench_formula_recovery(predicted, target, feature_names)
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        return strict, srbench, "evaluated"
    except TimeoutError:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        return None, None, "timeout_unavailable"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def parse_expression(text: object) -> sympy.Expr | None:
    if text is None or not str(text).strip():
        return None
    normalized = (
        str(text)
        .replace("^", "**")
        .replace("np.", "")
        .replace("math.", "")
    )
    try:
        return sympy.sympify(normalized, locals=SYMPY_LOCALS, evaluate=True)
    except Exception:
        return None


def candidate_key(expr: sympy.Expr | str) -> str:
    parsed = expr if isinstance(expr, sympy.Expr) else parse_expression(expr)
    return str(parsed) if parsed is not None else str(expr).strip()


def rounded_number(value: float, significant_digits: int) -> sympy.Float:
    if value == 0.0:
        return sympy.Float(0)
    return sympy.Float(f"{value:.{significant_digits}g}")


def round_floats(expr: sympy.Expr, significant_digits: int) -> sympy.Expr:
    replacements = {
        atom: rounded_number(float(atom), significant_digits)
        for atom in expr.atoms(sympy.Float)
    }
    return expr.xreplace(replacements)


def rationalize_floats(expr: sympy.Expr, tolerance: float) -> sympy.Expr:
    replacements = {}
    for atom in expr.atoms(sympy.Float):
        value = float(atom)
        try:
            rational = sympy.Rational(value).limit_denominator(64)
        except Exception:
            continue
        if abs(float(rational) - value) <= tolerance * max(1.0, abs(value)):
            replacements[atom] = rational
    return expr.xreplace(replacements)


def prune_small_additive_terms(expr: sympy.Expr, relative: float) -> sympy.Expr:
    expanded = sympy.expand(expr)
    if not isinstance(expanded, sympy.Add):
        return expanded
    terms = list(expanded.args)
    coefficients = []
    for term in terms:
        coefficient, _ = term.as_coeff_Mul()
        try:
            coefficients.append(abs(float(coefficient)))
        except Exception:
            coefficients.append(1.0)
    scale = max(coefficients, default=1.0)
    kept = [
        term
        for term, coefficient in zip(terms, coefficients)
        if coefficient >= relative * max(scale, 1e-300)
    ]
    return sympy.Add(*kept) if kept else sympy.Integer(0)


def generic_skeletons(feature_names: list[str]) -> list[tuple[str, sympy.Expr]]:
    symbols = [sympy.Symbol(name) for name in feature_names]
    out: list[tuple[str, sympy.Expr]] = [("constant_zero", sympy.Integer(0))]
    for i, x in enumerate(symbols):
        out.extend(
            [
                (f"x{i + 1}", x),
                (f"x{i + 1}_square", x**2),
                (f"x{i + 1}_cube", x**3),
                (f"x{i + 1}_inverse", 1 / x),
                (f"x{i + 1}_inverse_square", 1 / x**2),
                (f"x{i + 1}_sin", sympy.sin(x)),
                (f"x{i + 1}_cos", sympy.cos(x)),
                (f"x{i + 1}_exp", sympy.exp(x)),
                (f"x{i + 1}_logabs", sympy.log(sympy.Abs(x))),
                (f"x{i + 1}_sqrtabs", sympy.sqrt(sympy.Abs(x))),
                (f"x{i + 1}_over_pi", x / sympy.pi),
                (f"pi_x{i + 1}", sympy.pi * x),
            ]
        )
    for i, j in itertools.combinations(range(len(symbols)), 2):
        x, y = symbols[i], symbols[j]
        prefix = f"x{i + 1}_x{j + 1}"
        out.extend(
            [
                (f"{prefix}_product", x * y),
                (f"{prefix}_ratio", x / y),
                (f"{prefix}_reverse_ratio", y / x),
                (f"{prefix}_x_y2", x * y**2),
                (f"{prefix}_x2_y", x**2 * y),
                (f"{prefix}_x_over_y2", x / y**2),
                (f"{prefix}_y_over_x2", y / x**2),
                (f"{prefix}_sin_ratio", sympy.sin(x) / sympy.sin(y)),
                (f"{prefix}_difference", x - y),
                (f"{prefix}_sum", x + y),
            ]
        )
    if 3 <= len(symbols) <= 6:
        for combo in itertools.combinations(range(len(symbols)), 3):
            expression = sympy.prod(symbols[index] for index in combo)
            out.append(
                (
                    "product_" + "_".join(f"x{index + 1}" for index in combo),
                    expression,
                )
            )
    if 2 <= len(symbols) <= 8:
        out.append(("all_feature_product", sympy.prod(symbols)))
    return out


def affine_fit_candidates(
    skeleton: sympy.Expr,
    source: str,
    train_df: pd.DataFrame,
    feature_names: list[str],
) -> list[tuple[str, sympy.Expr]]:
    try:
        values = evaluate_expression(str(skeleton), train_df, feature_names)
    except Exception:
        return []
    if values is None or not np.isfinite(values).all():
        return []
    y = train_df["y"].to_numpy(dtype=float)
    design = np.column_stack([values, np.ones(len(values))])
    try:
        coefficient, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    except Exception:
        return []
    if not np.isfinite([coefficient, intercept]).all():
        return []

    candidates = [
        (f"{source}:raw", sympy.Float(coefficient) * skeleton + sympy.Float(intercept)),
        (f"{source}:zero_intercept", sympy.Float(coefficient) * skeleton),
        (f"{source}:unit_scale", skeleton + sympy.Float(intercept)),
        (f"{source}:bare", skeleton),
    ]
    for digits in (3, 4, 5, 6, 8, 10, 12):
        a = rounded_number(float(coefficient), digits)
        b = rounded_number(float(intercept), digits)
        candidates.append((f"{source}:round{digits}", a * skeleton + b))
        candidates.append((f"{source}:round{digits}_zero_intercept", a * skeleton))
    return candidates


def expression_derived_candidates(
    original: sympy.Expr,
    train_df: pd.DataFrame,
    feature_names: list[str],
) -> list[tuple[str, sympy.Expr]]:
    out: list[tuple[str, sympy.Expr]] = [("original", original)]
    transforms = [
        ("cancel", lambda value: sympy.cancel(value)),
        ("factor", lambda value: sympy.factor(value)),
        ("together", lambda value: sympy.together(value)),
    ]
    if len(str(original)) <= 300:
        transforms.append(("simplify", lambda value: sympy.simplify(value)))
    for name, function in transforms:
        try:
            out.append((name, function(original)))
        except Exception:
            pass
    for digits in (3, 4, 5, 6, 8, 10, 12):
        try:
            out.append((f"round_floats_{digits}", round_floats(original, digits)))
        except Exception:
            pass
    for tolerance in (1e-12, 1e-10, 1e-8, 1e-6):
        try:
            out.append(
                (
                    f"rationalize_{tolerance:g}",
                    rationalize_floats(original, tolerance),
                )
            )
        except Exception:
            pass
    for relative in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            out.append(
                (
                    f"prune_terms_{relative:g}",
                    prune_small_additive_terms(original, relative),
                )
            )
        except Exception:
            pass

    # Treat each used variable as potentially irrelevant.  Substitution uses
    # train medians only; validation decides whether the compact expression is
    # retained.
    for symbol in sorted(original.free_symbols, key=str):
        if str(symbol) not in feature_names:
            continue
        median = float(train_df[str(symbol)].median())
        try:
            reduced = sympy.simplify(original.subs(symbol, sympy.Float(median)))
            out.append((f"drop_{symbol}", reduced))
            out.extend(
                affine_fit_candidates(
                    reduced,
                    f"drop_{symbol}_refit",
                    train_df,
                    feature_names,
                )
            )
        except Exception:
            pass
    return out


def evaluate_candidate(
    expression: sympy.Expr,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, float] | None:
    text = str(expression)
    try:
        pred_train = evaluate_expression(text, train_df, feature_names)
        pred_val = evaluate_expression(text, val_df, feature_names)
    except Exception:
        return None
    if pred_train is None or pred_val is None:
        return None
    train_nmse = scale_invariant_nmse(train_df["y"], pred_train)
    val_nmse = scale_invariant_nmse(val_df["y"], pred_val)
    if train_nmse is None or val_nmse is None:
        return None
    if not np.isfinite([train_nmse, val_nmse]).all():
        return None
    complexity = expression_complexity(text, feature_names)["expr_complexity"]
    if complexity is None:
        return None
    return {
        "train_nmse": float(train_nmse),
        "val_nmse": float(val_nmse),
        "train_r2": float(1.0 - train_nmse),
        "val_r2": float(1.0 - val_nmse),
        "complexity": int(complexity),
    }


def choose_candidate(candidates: list[dict[str, object]]) -> tuple[dict[str, object], str]:
    original = next(item for item in candidates if item["source"] == "original")
    valid = [
        item
        for item in candidates
        if float(item["train_nmse"]) <= 0.001 and float(item["val_nmse"]) <= 0.001
    ]
    if valid:
        best_val = min(float(item["val_nmse"]) for item in valid)
        error_ceiling = min(0.001, max(1e-10, best_val * 10.0))
        accurate = [
            item for item in valid if float(item["val_nmse"]) <= error_ceiling
        ]
        selected = min(
            accurate,
            key=lambda item: (
                int(item["complexity"]),
                float(item["val_nmse"]),
                len(str(item["expression"])),
            ),
        )
        if float(original["val_nmse"]) <= 0.001:
            # Do not trade an already excellent final expression for a more
            # complex or materially less accurate candidate.
            if (
                int(selected["complexity"]) >= int(original["complexity"])
                and float(selected["val_nmse"]) >= float(original["val_nmse"])
            ):
                return original, "retain_original_dominates"
        return selected, "validation_success_then_simplicity"

    # Publication protocol: a rescue is committed only after crossing the
    # predeclared high-accuracy threshold on both train and validation.  A
    # merely better validation NMSE is useful exploratory evidence, but is not
    # sufficiently robust to replace the archived final expression.
    return original, "retain_original_no_validated_high_accuracy_candidate"


def final_metrics(
    expression: str,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, float]:
    try:
        prediction = evaluate_expression(expression, test_df, feature_names)
    except Exception:
        prediction = None
    if prediction is None:
        return {"test_nmse": math.nan, "test_r2": math.nan}
    nmse = scale_invariant_nmse(test_df["y"], prediction)
    r2 = None if nmse is None else 1.0 - nmse
    return {
        "test_nmse": math.nan if nmse is None else float(nmse),
        "test_r2": math.nan if r2 is None else float(r2),
    }


def run_case(task: pd.Series, result: pd.Series) -> dict[str, object]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    train_df = load_txt(str(task["train_path"]))
    val_df = load_txt(str(task["val_path"]))
    test_df = load_txt(str(task["test_path"]))
    feature_names = [f"x{index + 1}" for index in range(int(task["n_features"]))]
    original_expr = parse_expression(result.get("best_expr"))
    if original_expr is None:
        original_expr = sympy.Integer(0)

    raw_candidates = expression_derived_candidates(
        original_expr, train_df, feature_names
    )
    for source, skeleton in generic_skeletons(feature_names):
        raw_candidates.extend(
            affine_fit_candidates(
                skeleton,
                f"library:{source}",
                train_df,
                feature_names,
            )
        )

    deduplicated: dict[str, tuple[str, sympy.Expr]] = {}
    for source, expression in raw_candidates:
        key = candidate_key(expression)
        if key and key not in deduplicated:
            deduplicated[key] = (source, expression)

    evaluated: list[dict[str, object]] = []
    for source, expression in deduplicated.values():
        metrics = evaluate_candidate(expression, train_df, val_df, feature_names)
        if metrics is None:
            continue
        evaluated.append(
            {
                "source": source,
                "expression": str(expression),
                **metrics,
            }
        )
    if not any(item["source"] == "original" for item in evaluated):
        raise RuntimeError("Original expression is not evaluable")
    selected, selection_reason = choose_candidate(evaluated)

    original_text = str(original_expr)
    selected_text = str(selected["expression"])
    original_test = final_metrics(original_text, test_df, feature_names)
    selected_test = final_metrics(selected_text, test_df, feature_names)
    target = str(task["true_expression"])
    (
        original_strict,
        original_srbench,
        original_recovery_status,
    ) = formula_recovery_with_timeout(
        original_text,
        target,
        feature_names,
    )
    (
        selected_strict,
        selected_srbench,
        selected_recovery_status,
    ) = formula_recovery_with_timeout(
        selected_text,
        target,
        feature_names,
    )

    return {
        "dataset_dir": task["dataset_dir"],
        "base_name": task["base_name"],
        "n_features": int(task["n_features"]),
        "candidate_count": len(evaluated),
        "selection_reason": selection_reason,
        "selected_source": selected["source"],
        "original_expression": original_text,
        "selected_expression": selected_text,
        "original_train_nmse": next(
            float(item["train_nmse"])
            for item in evaluated
            if item["source"] == "original"
        ),
        "selected_train_nmse": float(selected["train_nmse"]),
        "original_val_nmse": next(
            float(item["val_nmse"])
            for item in evaluated
            if item["source"] == "original"
        ),
        "selected_val_nmse": float(selected["val_nmse"]),
        "original_test_nmse": original_test["test_nmse"],
        "selected_test_nmse": selected_test["test_nmse"],
        "original_test_r2": original_test["test_r2"],
        "selected_test_r2": selected_test["test_r2"],
        "original_complexity": next(
            int(item["complexity"])
            for item in evaluated
            if item["source"] == "original"
        ),
        "selected_complexity": int(selected["complexity"]),
        "original_high_accuracy": bool(
            np.isfinite(original_test["test_r2"])
            and original_test["test_r2"] > 0.999
        ),
        "selected_high_accuracy": bool(
            np.isfinite(selected_test["test_r2"])
            and selected_test["test_r2"] > 0.999
        ),
        "original_strict_recovery": original_strict,
        "selected_strict_recovery": selected_strict,
        "original_srbench_recovery": original_srbench,
        "selected_srbench_recovery": selected_srbench,
        "original_recovery_status": original_recovery_status,
        "selected_recovery_status": selected_recovery_status,
        "test_used_for_selection": False,
        "target_formula_used_for_selection": False,
    }


def run_case_pair(pair: tuple[pd.Series, pd.Series]) -> dict[str, object]:
    """Pickle-friendly adapter used by the optional process pool."""
    return run_case(*pair)


def summarize(rows: pd.DataFrame) -> dict[str, object]:
    return {
        "cases": int(len(rows)),
        "changed_expressions": int(
            (rows["original_expression"] != rows["selected_expression"]).sum()
        ),
        "original_high_accuracy_successes": int(
            rows["original_high_accuracy"].sum()
        ),
        "selected_high_accuracy_successes": int(
            rows["selected_high_accuracy"].sum()
        ),
        "original_high_accuracy_rate": float(
            rows["original_high_accuracy"].mean()
        ),
        "selected_high_accuracy_rate": float(
            rows["selected_high_accuracy"].mean()
        ),
        "original_median_test_nmse": float(rows["original_test_nmse"].median()),
        "selected_median_test_nmse": float(rows["selected_test_nmse"].median()),
        "original_median_complexity": float(rows["original_complexity"].median()),
        "selected_median_complexity": float(rows["selected_complexity"].median()),
        "test_numerically_dominated_or_equal": int(
            (
                (rows["selected_test_nmse"] <= rows["original_test_nmse"])
                & (rows["selected_complexity"] <= rows["original_complexity"])
            ).sum()
        ),
        "original_strict_recovery_successes": int(
            rows["original_strict_recovery"].fillna(False).sum()
        ),
        "selected_strict_recovery_successes": int(
            rows["selected_strict_recovery"].fillna(False).sum()
        ),
        "original_srbench_recovery_successes": int(
            rows["original_srbench_recovery"].fillna(False).sum()
        ),
        "selected_srbench_recovery_successes": int(
            rows["selected_srbench_recovery"].fillna(False).sum()
        ),
        "selection_uses_test": False,
        "selection_uses_target_formula": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    tasks = pd.read_csv(args.selection)
    results = pd.read_csv(args.results, low_memory=False)
    if len(tasks) != len(results):
        raise ValueError(f"Row count mismatch: {len(tasks)} vs {len(results)}")
    if not np.array_equal(
        tasks["dataset_dir"].astype(str).to_numpy(),
        results["dataset_dir"].astype(str).to_numpy(),
    ):
        raise ValueError("dataset_dir row alignment failed")
    if not np.array_equal(
        tasks["true_expression"].astype(str).to_numpy(),
        results["true_expression"].astype(str).to_numpy(),
    ):
        raise ValueError("true_expression row alignment failed")
    if args.limit is not None:
        tasks = tasks.head(args.limit)
        results = results.head(args.limit)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = args.output_root / f"run_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    pairs = [
        (task, result)
        for (_, task), (_, result) in zip(tasks.iterrows(), results.iterrows())
    ]
    rows = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(run_case_pair, pair): position
                for position, pair in enumerate(pairs, start=1)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                position = futures[future]
                row = future.result()
                row["_position"] = position
                rows.append(row)
                print(
                    f"[{completed}/{len(tasks)}; row {position}] "
                    f"{row['dataset_dir']}/{row['base_name']} "
                    f"source={row['selected_source']} "
                    f"R2 {row['original_test_r2']:.6g}->{row['selected_test_r2']:.6g} "
                    f"C {row['original_complexity']}->{row['selected_complexity']}",
                    flush=True,
                )
                if completed % 10 == 0:
                    pd.DataFrame(rows).sort_values("_position").to_csv(
                        output / "srsd_expression_rescue_partial.csv", index=False
                    )
    else:
        for position, pair in enumerate(pairs, start=1):
            row = run_case_pair(pair)
            row["_position"] = position
            rows.append(row)
            print(
                f"[{position}/{len(tasks)}] {row['dataset_dir']}/{row['base_name']} "
                f"source={row['selected_source']} "
                f"R2 {row['original_test_r2']:.6g}->{row['selected_test_r2']:.6g} "
                f"C {row['original_complexity']}->{row['selected_complexity']}",
                flush=True,
            )

    frame = pd.DataFrame(rows).sort_values("_position").drop(columns="_position")
    frame.to_csv(output / "srsd_expression_rescue_case_rows.csv", index=False)
    summary = summarize(frame)
    (output / "srsd_expression_rescue_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    audit = {
        "selection_data": ["SRSD train", "SRSD validation"],
        "held_out_reporting_data": ["SRSD test", "true formula"],
        "test_used_for_selection": False,
        "target_formula_used_for_selection": False,
        "candidate_sources": [
            "algebraic simplification",
            "constant rounding/rationalization",
            "single-variable removal",
            "predeclared compact skeleton library",
        ],
    }
    (output / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "LATEST").write_text(
        str(output.resolve()), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
