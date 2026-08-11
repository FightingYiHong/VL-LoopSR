#!/usr/bin/env python3
"""Post-hoc metric alignment and paper-style plots for SR result tables."""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_metrics import (  # noqa: E402
    expression_complexity,
    pareto_front_indices,
    srbench_formula_recovery,
    strict_formula_recovery,
)
from tools.plot_style import (  # noqa: E402
    COLOR_NEUTRAL_DARK,
    NATURE_COLORS,
    palette_for,
    save_nature_figure,
    set_nature_style,
)


EXPR_COLUMNS = ("best_expr", "expression", "predicted_expression", "formula")
TARGET_COLUMNS = (
    "true_expression_for_scoring",
    "true_expression",
    "ground_truth_expression",
    "target_expression",
)
MSE_COLUMNS = ("test_mse", "clean_test_mse", "best_test_mse", "ood_mse")


def first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in frame.columns), None)


def finite_float(value) -> float | None:
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def feature_names(predicted, target) -> list[str]:
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", f"{predicted} {target}"))
    functions = {
        "Abs", "abs", "sqrt", "log", "exp", "sin", "cos", "tan",
        "asin", "acos", "atan", "sinh", "cosh", "tanh", "pi", "E", "e",
    }
    variables = [name for name in names if name not in functions]

    def key(name: str):
        match = re.fullmatch(r"x(\d+)", name)
        return (0, int(match.group(1))) if match else (1, name)

    return sorted(variables, key=key)


def symbolic_call_with_timeout(function, *args, timeout_sec: float = 0.25):
    if not hasattr(signal, "setitimer"):
        return function(*args)

    def raise_timeout(_signum, _frame):
        raise TimeoutError("symbolic comparison timed out")

    previous_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_sec)
    try:
        return function(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def align_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    expr_column = first_column(out, EXPR_COLUMNS)
    target_column = first_column(out, TARGET_COLUMNS)
    if expr_column is None:
        raise ValueError(f"no expression column found; expected one of {EXPR_COLUMNS}")

    strict_values = []
    strict_evaluable = []
    srbench_values = []
    srbench_evaluable = []
    complexities = []
    depths = []
    sympy_ops = []
    total_rows = len(out)
    for position, (_, row) in enumerate(out.iterrows(), start=1):
        predicted = row.get(expr_column)
        target = row.get(target_column) if target_column else None
        names = feature_names(predicted, target)
        complexity = expression_complexity(predicted, names)
        complexities.append(complexity["expr_complexity"])
        depths.append(complexity["expr_depth"])
        sympy_ops.append(complexity["expr_sympy_ops"])
        strict = symbolic_call_with_timeout(
            strict_formula_recovery, predicted, target, names
        )
        srbench = (
            True
            if strict is True
            else symbolic_call_with_timeout(
                srbench_formula_recovery, predicted, target, names
            )
        )
        strict_values.append(bool(strict) if strict is not None else False)
        strict_evaluable.append(strict is not None)
        srbench_values.append(bool(srbench) if srbench is not None else False)
        srbench_evaluable.append(srbench is not None)
        if position % 100 == 0 or position == total_rows:
            print(f"[posthoc] symbolic metrics {position}/{total_rows}", flush=True)

    out["posthoc_expr_complexity"] = complexities
    out["posthoc_expr_depth"] = depths
    out["posthoc_expr_sympy_ops"] = sympy_ops
    out["strict_formula_recovery"] = strict_values
    out["strict_formula_recovery_evaluable"] = strict_evaluable
    out["srbench_formula_recovery"] = srbench_values
    out["srbench_formula_recovery_evaluable"] = srbench_evaluable

    if "test_r2" in out:
        r2 = pd.to_numeric(out["test_r2"], errors="coerce")
        out["test_r2_evaluable"] = np.isfinite(r2)
        nmse = 1.0 - r2
        out["test_nmse"] = nmse.where(np.isfinite(nmse))
        out["test_nrmse"] = np.sqrt(out["test_nmse"].clip(lower=0.0))
        out["numerical_complete_fit"] = r2 > 0.999
        out["negative_test_r2"] = r2 < 0.0
    else:
        out["test_r2_evaluable"] = False
        out["test_nmse"] = np.nan
        out["test_nrmse"] = np.nan
        out["numerical_complete_fit"] = False
        out["negative_test_r2"] = False

    mse_column = first_column(out, MSE_COLUMNS)
    out["posthoc_test_mse"] = (
        pd.to_numeric(out[mse_column], errors="coerce")
        if mse_column else np.nan
    )
    if "timed_out" in out:
        out["posthoc_timeout"] = out["timed_out"].fillna(False).astype(bool)
    elif "timeout" in out:
        out["posthoc_timeout"] = out["timeout"].fillna(False).astype(bool)
    else:
        out["posthoc_timeout"] = False
    return out


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    if "method" not in frame:
        frame = frame.assign(method="unknown")
    group_columns = ["method"]
    if "noise_level" in frame:
        group_columns.append("noise_level")
    runtime_column = "runtime_sec" if "runtime_sec" in frame else None
    rows = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        r2_evaluable = group["test_r2_evaluable"].fillna(False).astype(bool)
        row.update(
            {
                "n": len(group),
                "timeout_rate": group["posthoc_timeout"].mean(),
                "test_r2_evaluable_n": int(r2_evaluable.sum()),
                "test_r2_coverage": float(r2_evaluable.mean()),
                "numerical_complete_fit_rate": (
                    group.loc[r2_evaluable, "numerical_complete_fit"].mean()
                    if r2_evaluable.any() else np.nan
                ),
                "strict_recovery_rate": group["strict_formula_recovery"].mean(),
                "srbench_recovery_rate": group["srbench_formula_recovery"].mean(),
                "median_test_mse": group["posthoc_test_mse"].median(),
                "median_test_nmse": group["test_nmse"].median(),
                "median_test_nrmse": group["test_nrmse"].median(),
                "median_tree_nodes": pd.to_numeric(
                    group["posthoc_expr_complexity"], errors="coerce"
                ).median(),
                "median_runtime_sec": (
                    pd.to_numeric(group[runtime_column], errors="coerce").median()
                    if runtime_column else np.nan
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def parse_candidate_pareto(frame: pd.DataFrame) -> pd.DataFrame:
    if "evaluation_history" not in frame:
        return pd.DataFrame()
    records = []
    for row_index, value in frame["evaluation_history"].items():
        try:
            history = json.loads(value) if isinstance(value, str) else value
        except Exception:
            continue
        if not isinstance(history, list):
            continue
        for stage in history:
            if not isinstance(stage, dict):
                continue
            for candidate in stage.get("pareto_candidates") or []:
                if not isinstance(candidate, dict):
                    continue
                record = dict(candidate)
                record["source_row"] = row_index
                record["stage"] = stage.get("stage")
                records.append(record)
    candidates = pd.DataFrame(records)
    if candidates.empty:
        return candidates
    candidates["val_nmse"] = pd.to_numeric(candidates.get("val_nmse"), errors="coerce")
    candidates["val_mse"] = pd.to_numeric(candidates.get("val_mse"), errors="coerce")
    candidates["complexity"] = pd.to_numeric(candidates.get("complexity"), errors="coerce")
    objective = candidates["val_nmse"].where(
        np.isfinite(candidates["val_nmse"]), candidates["val_mse"]
    )
    candidates["pareto_error_objective"] = objective
    candidates["on_validation_pareto_front"] = False
    for _, indices in candidates.groupby(["source_row", "stage"], dropna=False).groups.items():
        part = candidates.loc[indices]
        front = pareto_front_indices(
            part["pareto_error_objective"].tolist(),
            part["complexity"].tolist(),
        )
        candidates.loc[part.index[front], "on_validation_pareto_front"] = True
    return candidates


def save_figure(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    save_nature_figure(
        fig,
        path.with_suffix(".pdf"),
        section="metric_alignment",
        bbox_inches="tight",
    )


def plot_summary(frame: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    set_nature_style(plt)
    methods = [str(value) for value in frame["method"].dropna().unique()]
    colors = palette_for(methods)
    overall = (
        frame.groupby("method", dropna=False)
        .agg(
            strict=("strict_formula_recovery", "mean"),
            srbench=("srbench_formula_recovery", "mean"),
            median_mse=("posthoc_test_mse", "median"),
            median_complexity=("posthoc_expr_complexity", "median"),
        )
        .reset_index()
    )
    numerical = (
        frame[frame["test_r2_evaluable"]]
        .groupby("method", dropna=False)["numerical_complete_fit"]
        .mean()
    )
    overall["numerical"] = overall["method"].map(numerical)
    overall["method"] = overall["method"].astype(str)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    metric_specs = [
        ("strict", "Strict recovery"),
        ("srbench", "SRBench recovery"),
        ("numerical", r"Test $R^2 > 0.999$"),
    ]
    y = np.arange(len(overall))
    offsets = np.linspace(-0.18, 0.18, len(metric_specs))
    for offset, (column, label) in zip(offsets, metric_specs):
        axes[0].scatter(
            overall[column],
            y + offset,
            s=22,
            label=label,
            color=NATURE_COLORS[
                {"strict": "blue", "srbench": "orange", "numerical": "green"}[column]
            ],
            edgecolor="white",
            linewidth=0.35,
        )
    axes[0].set_yticks(y, overall["method"])
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_xlabel("Rate")
    axes[0].set_title("Aligned recovery criteria")
    axes[0].legend(loc="lower right")

    valid_mse = overall[np.isfinite(pd.to_numeric(overall["median_mse"], errors="coerce"))]
    axes[1].scatter(
        valid_mse["median_complexity"],
        valid_mse["median_mse"],
        c=[colors.get(method, COLOR_NEUTRAL_DARK) for method in valid_mse["method"]],
        s=30,
        edgecolor="white",
        linewidth=0.4,
    )
    for _, row in valid_mse.iterrows():
        axes[1].annotate(
            row["method"],
            (row["median_complexity"], row["median_mse"]),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=6.3,
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Median expression tree nodes")
    axes[1].set_ylabel("Median test MSE")
    axes[1].set_title("Accuracy–complexity profile")

    if "noise_level" in frame:
        noise = (
            frame.groupby(["method", "noise_level"], dropna=False)[
                "strict_formula_recovery"
            ]
            .mean()
            .reset_index()
        )
        for method, group in noise.groupby("method"):
            group = group.sort_values("noise_level")
            axes[2].plot(
                group["noise_level"],
                group["strict_formula_recovery"],
                marker="o",
                ms=3,
                lw=1.1,
                color=colors.get(str(method), COLOR_NEUTRAL_DARK),
                label=str(method),
            )
        axes[2].set_xscale("symlog", linthresh=1e-4)
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].set_xlabel("Relative target-noise level")
        axes[2].set_ylabel("Strict recovery rate")
        axes[2].set_title("Noise robustness")
    else:
        axes[2].axis("off")
    save_figure(fig, out_dir / "posthoc_metric_alignment")
    plt.close(fig)


def plot_candidate_pareto(candidates: pd.DataFrame, out_dir: Path) -> None:
    if candidates.empty:
        return
    import matplotlib.pyplot as plt

    set_nature_style(plt)
    key = (
        candidates.groupby(["source_row", "stage"], dropna=False)
        .size()
        .sort_values(ascending=False)
        .index[0]
    )
    source_row, stage = key
    part = candidates[
        (candidates["source_row"] == source_row)
        & (candidates["stage"].fillna("") == ("" if pd.isna(stage) else stage))
    ].copy()
    part = part.dropna(subset=["complexity", "pareto_error_objective"])
    if part.empty:
        return
    front = part[part["on_validation_pareto_front"]].sort_values("complexity")
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    ax.scatter(
        part["complexity"],
        part["pareto_error_objective"],
        s=20,
        color=NATURE_COLORS["midgray"],
        alpha=0.65,
        label="Candidate",
    )
    ax.plot(
        front["complexity"],
        front["pareto_error_objective"],
        marker="o",
        ms=4,
        lw=1.3,
        color=NATURE_COLORS["blue"],
        label="Pareto front",
    )
    positive = part["pareto_error_objective"] > 0
    if positive.all():
        ax.set_yscale("log")
    ax.set_xlabel("Expression tree nodes")
    ax.set_ylabel("Validation NMSE")
    ax.set_title("Candidate-level Pareto analysis")
    ax.legend()
    save_figure(fig, out_dir / "candidate_validation_pareto")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = align_rows(pd.read_csv(args.input_csv))
    summary = summarize(frame)
    candidates = parse_candidate_pareto(frame)
    frame.to_csv(out_dir / "aligned_result_rows.csv", index=False)
    summary.to_csv(out_dir / "aligned_summary.csv", index=False)
    if not candidates.empty:
        candidates.to_csv(out_dir / "candidate_pareto_rows.csv", index=False)
    plot_summary(frame, out_dir)
    plot_candidate_pareto(candidates, out_dir)
    manifest = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "row_count": len(frame),
        "summary_row_count": len(summary),
        "candidate_count": len(candidates),
        "metric_definitions": {
            "strict_recovery": "simplify(predicted - target) == 0",
            "srbench_recovery": "strict, additive-constant, or nonzero constant-ratio equivalence",
            "pareto_objectives": ["validation_nmse", "expression_tree_nodes"],
            "numerical_complete_fit": "test R2 > 0.999",
        },
    }
    (out_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
