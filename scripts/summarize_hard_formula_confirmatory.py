#!/usr/bin/env python3
"""Summarize equal-budget, repeated hard-formula confirmation runs."""

from __future__ import annotations

import argparse
import ast
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHOD_DIRS = {
    "Ours": "ours",
    "LLM-SR": "llmsr",
    "DSO": "dso",
    "gplearn": "gplearn",
    "PSE": "psrn",
    "PySR": "pysr",
    "ICSR": "icsr",
}
METHOD_ORDER = list(METHOD_DIRS)
CASE_ORDER = [
    "II.11.3_2_0",
    "feynman-ii.2.42",
    "feynman-i.12.11",
    "feynman-i.39.11",
    "feynman-ii.38.14",
    "feynman-ii.13.17",
]
CASE_LABELS = {
    "II.11.3_2_0": "Feynman II.11.3 (transformed)",
    "feynman-ii.2.42": "Feynman II.2.42",
    "feynman-i.12.11": "Feynman I.12.11",
    "feynman-i.39.11": "Feynman I.39.11",
    "feynman-ii.38.14": "Feynman II.38.14",
    "feynman-ii.13.17": "Feynman II.13.17",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.999)
    return parser.parse_args()


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def repeat_seed(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def normalize_case_name(value: object) -> str | None:
    text = str(value or "").strip()
    if text in CASE_LABELS:
        return text
    lowered = text.lower()
    for case_name in CASE_LABELS:
        if lowered == case_name.lower():
            return case_name
    return None


def feature_names(result: dict) -> list[str]:
    value = result.get("feature_names", [])
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            value = [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []


def _tolerance_formula_recovery_impl(predicted: object, target: object, names: list[str]) -> bool:
    try:
        import sympy
        from benchmark_metrics import _canonicalize_feature_symbols, _expression_symbols

        predicted_expr, _ = _expression_symbols(str(predicted), names)
        target_expr, _ = _expression_symbols(str(target), names)
        replacements = {}
        for atom in predicted_expr.atoms(sympy.Float):
            value = float(atom)
            if abs(value) < 1e-8:
                replacements[atom] = sympy.Integer(0)
            elif abs(value - 1.0) < 1e-8:
                replacements[atom] = sympy.Integer(1)
            elif abs(value + 1.0) < 1e-8:
                replacements[atom] = sympy.Integer(-1)
        predicted_expr = predicted_expr.xreplace(replacements)
        predicted_expr = _canonicalize_feature_symbols(predicted_expr, names)
        target_expr = _canonicalize_feature_symbols(target_expr, names)
        difference = sympy.cancel(sympy.together(predicted_expr - target_expr))
        return bool(sympy.simplify(difference) == 0)
    except Exception:
        return False


def _formula_recovery_worker(queue, predicted: object, target: object, names: list[str]) -> None:
    queue.put(_tolerance_formula_recovery_impl(predicted, target, names))


def tolerance_formula_recovery(predicted: object, target: object, names: list[str]) -> bool:
    """Symbolic equivalence after coefficient chop, with a hard process timeout."""
    if predicted is None or target is None:
        return False
    context = mp.get_context("fork")
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_formula_recovery_worker,
        args=(queue, predicted, target, names),
        daemon=True,
    )
    process.start()
    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
        return False
    try:
        return bool(queue.get_nowait())
    except Exception:
        return False


def row_from_mapping(method: str, seed: int, benchmark: str, result: dict, source: Path) -> dict | None:
    case_name = normalize_case_name(result.get("case_name", result.get("base_name")))
    if case_name is None:
        return None
    val_r2 = finite(
        result.get(
            "first_validation_success_r2",
            result.get("first_validation_success_val_r2", result.get("val_r2")),
        )
    )
    validation_hit = as_bool(result.get("validation_search_success"))
    if not validation_hit and np.isfinite(val_r2):
        validation_hit = bool(val_r2 > 0.999)
    test_r2 = finite(result.get("test_r2"))
    if not np.isfinite(test_r2):
        test_mse = finite(result.get("best_test_mse"))
        test_path = result.get("test_path")
        if np.isfinite(test_mse) and test_path:
            try:
                target = np.loadtxt(test_path, dtype=float)[:, -1]
                target_variance = float(np.mean((target - np.mean(target)) ** 2))
                if target_variance > 0:
                    test_r2 = 1.0 - test_mse / target_variance
            except (OSError, TypeError, ValueError, IndexError):
                pass
    expression = result.get(
        "best_expr",
        result.get("expression", result.get("first_validation_success_expression")),
    )
    target_expression = result.get(
        "true_expression",
        result.get("true_expression_for_scoring", result.get("ground_truth_expression")),
    )
    names = feature_names(result)
    recovery = result.get("srbench_formula_recovery")
    if recovery is None:
        recovery = False
    tolerant_recovery = tolerance_formula_recovery(expression, target_expression, names)
    return {
        "method": method,
        "repeat_seed": seed,
        "benchmark": benchmark,
        "case_name": case_name,
        "case_label": CASE_LABELS[case_name],
        "validation_hit": validation_hit,
        "validation_r2": val_r2,
        "test_r2": test_r2,
        "evaluations_to_hit": finite(
            result.get(
                "evaluations_to_validation_success",
                result.get("native_evaluations_to_validation_success"),
            )
        ),
        "runtime_sec": finite(result.get("runtime_sec")),
        "expression": expression,
        "target_expression": target_expression,
        "expr_complexity": finite(result.get("expr_complexity")),
        "strict_formula_recovery": as_bool(result.get("strict_formula_recovery")),
        "srbench_formula_recovery": as_bool(recovery),
        "tolerance_formula_recovery": tolerant_recovery,
        "error": result.get("error"),
        "source_file": str(source),
    }


def read_ours(run_root: Path) -> list[dict]:
    rows = []
    for path in sorted((run_root / "ours").glob("seed_*/*/all_results_detailed.csv")):
        seed = repeat_seed(path)
        if seed is None:
            continue
        benchmark = path.parent.name
        frame = pd.read_csv(path)
        for result in frame.to_dict(orient="records"):
            row = row_from_mapping("Ours", seed, benchmark, result, path)
            if row is not None:
                rows.append(row)

    # Older SRSD runs could write a valid per-case report and then deadlock
    # while returning the large result to the parent process. Recover those
    # reports when the aggregate CSV contains only the outer timeout record.
    for path in sorted((run_root / "ours").glob("seed_*/*/per_case_reports/*.json")):
        seed = repeat_seed(path)
        if seed is None:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        result = {
            **dict(report.get("task_meta") or {}),
            **dict(report.get("result_core") or {}),
        }
        row = row_from_mapping("Ours", seed, path.parent.parent.name, result, path)
        if row is not None:
            rows.append(row)

    def result_quality(row: dict) -> tuple[bool, bool, bool, bool, bool]:
        return (
            bool(row["validation_hit"]),
            bool(np.isfinite(row["test_r2"])),
            bool(row["expression"]),
            not bool(row["error"]),
            bool(np.isfinite(row["expr_complexity"])),
        )

    unique = {}
    for row in rows:
        key = (row["method"], row["repeat_seed"], row["case_name"])
        old = unique.get(key)
        if old is None or result_quality(row) > result_quality(old):
            unique[key] = row
    return list(unique.values())


def read_json_method(run_root: Path, method: str, method_dir: str) -> list[dict]:
    rows = []
    for path in sorted((run_root / method_dir).glob("seed_*/**/*.json")):
        if path.name in {"manifest.json", "run_config.json", "summary.json"}:
            continue
        seed = repeat_seed(path)
        if seed is None:
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict):
            continue
        benchmark = "srsd" if "srsd" in path.parts else "llmsrbench"
        row = row_from_mapping(method, seed, benchmark, result, path)
        if row is not None:
            rows.append(row)
    unique = {}
    for row in rows:
        key = (row["method"], row["repeat_seed"], row["case_name"])
        old = unique.get(key)
        if old is None or (not old["validation_hit"] and row["validation_hit"]):
            unique[key] = row
    return list(unique.values())


def summarize(case_rows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = case_rows.copy()
    frame["test_hit"] = pd.to_numeric(frame["test_r2"], errors="coerce").gt(threshold)
    frame["confirmed_hit"] = frame["validation_hit"].astype(bool) & frame["test_hit"]
    frame["confirmed_formula_recovery"] = (
        frame["confirmed_hit"] & frame["tolerance_formula_recovery"].astype(bool)
    )
    frame["valid_run"] = ~(
        frame["expression"].isna()
        & frame["error"].astype(str).str.contains("worker hard timeout", case=False, na=False)
    )
    frame["validation_hit_score"] = frame["validation_hit"].astype(float).where(frame["valid_run"])
    frame["test_hit_score"] = frame["test_hit"].astype(float).where(frame["valid_run"])
    frame["confirmed_hit_score"] = frame["confirmed_hit"].astype(float).where(frame["valid_run"])
    frame["formula_recovery_score"] = (
        frame["confirmed_formula_recovery"].astype(float).where(frame["valid_run"])
    )
    summary = (
        frame.groupby(["method", "case_name", "case_label"], sort=False)
        .agg(
            attempted_repeats=("repeat_seed", "nunique"),
            valid_repeats=("valid_run", "sum"),
            validation_hit_rate=("validation_hit_score", "mean"),
            test_hit_rate=("test_hit_score", "mean"),
            confirmed_hit_rate=("confirmed_hit_score", "mean"),
            formula_recovery_rate=("formula_recovery_score", "mean"),
            median_test_r2=("test_r2", "median"),
            median_expr_complexity=("expr_complexity", "median"),
            median_evaluations_to_hit=("evaluations_to_hit", "median"),
            median_runtime_sec=("runtime_sec", "median"),
        )
        .reset_index()
    )
    return summary


def save_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    from tools.plot_style import NATURE_COLORS, save_nature_figure, set_nature_style

    set_nature_style()
    observed = set(summary["case_name"].astype(str))
    case_order = [case_name for case_name in CASE_ORDER if case_name in observed]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    images = []
    for ax, metric, title in zip(
        axes,
        ["confirmed_hit_rate", "formula_recovery_rate"],
        [
            "a  Numerical confirmation ($R^2>0.999$)",
            "b  Formula recovery after 1e-8 coefficient chop",
        ],
    ):
        pivot = (
            summary.pivot(index="method", columns="case_name", values=metric)
            .reindex(index=METHOD_ORDER, columns=case_order)
        )
        image = ax.imshow(pivot.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        images.append(image)
        for row_index, method in enumerate(pivot.index):
            for col_index, case_name in enumerate(pivot.columns):
                value = pivot.loc[method, case_name]
                label = "NA" if not np.isfinite(value) else f"{100 * value:.0f}%"
                ax.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color="white" if np.isfinite(value) and value >= 0.55 else NATURE_COLORS["dark"],
                )
        ax.set_xticks(
            range(len(case_order)),
            [CASE_LABELS[name] for name in case_order],
            rotation=16,
            ha="right",
        )
        ax.set_yticks(
            range(len(METHOD_ORDER)),
            ["VEGA-SR" if name == "Ours" else name for name in METHOD_ORDER],
        )
        ax.set_title(title, loc="left")
        ax.tick_params(length=0)
    colorbar = fig.colorbar(images[-1], ax=axes, fraction=0.025, pad=0.03)
    colorbar.set_label("Confirmed runs")
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.set_ticklabels(["0%", "50%", "100%"])
    fig.subplots_adjust(left=0.13, right=0.91, bottom=0.24, top=0.90, wspace=0.12)
    save_nature_figure(
        fig,
        output_dir / "fig_hard_formula_confirmatory.png",
        section="standard_recovery",
        dpi=400,
        bbox_inches="tight",
    )
    save_nature_figure(
        fig,
        output_dir / "fig_hard_formula_confirmatory.pdf",
        section="standard_recovery",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_ours(args.run_root)
    for method, method_dir in METHOD_DIRS.items():
        if method == "Ours":
            continue
        rows.extend(read_json_method(args.run_root, method, method_dir))
    if not rows:
        raise RuntimeError(f"No completed case results found under {args.run_root}")
    case_rows = pd.DataFrame(rows)
    case_rows["test_hit"] = pd.to_numeric(case_rows["test_r2"], errors="coerce").gt(args.threshold)
    case_rows["confirmed_hit"] = case_rows["validation_hit"].astype(bool) & case_rows["test_hit"]
    case_rows["confirmed_formula_recovery"] = (
        case_rows["confirmed_hit"] & case_rows["tolerance_formula_recovery"].astype(bool)
    )
    case_rows["valid_run"] = ~(
        case_rows["expression"].isna()
        & case_rows["error"].astype(str).str.contains("worker hard timeout", case=False, na=False)
    )
    summary = summarize(case_rows, args.threshold)
    case_rows.to_csv(args.output_dir / "hard_formula_confirmatory_case_rows.csv", index=False)
    summary.to_csv(args.output_dir / "hard_formula_confirmatory_summary.csv", index=False)
    save_figure(summary, args.output_dir)
    print(summary.to_string(index=False))
    print(f"[INFO] wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
