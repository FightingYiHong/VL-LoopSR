#!/usr/bin/env python3
"""Plot a unified post-hoc evaluation of standard SR benchmark results.

Primary quantities are computed only where their inputs are evaluable:
test R² and NMSE require a finite test MSE and positive target variance;
formula recovery requires parseable final and target expressions; and the
empirical Pareto front uses only final selected expressions with finite NMSE
and manuscript-defined operational complexity.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import signal
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_standard_recovery_r2 import case_key  # noqa: E402


TITLE = "Unified Evaluation Separates Numerical Accuracy, Symbolic Recovery and Expression Simplicity"
OUTPUT_STEM = "fig_unified_evaluation"
DEFAULT_OUTPUT_DIR = ROOT / "figs" / "nature_subjournal" / "unified_evaluation"
PAPER_ROWS = (
    ROOT
    / "reports"
    / "posthoc_recompute_inputs_20260724"
    / "01_main"
    / "paper_reference"
    / "fig1_main_case_rows.csv"
)
VARIANCE_CACHE = (
    ROOT
    / "runs"
    / "paper_experiments"
    / "01_standard_recovery"
    / "standard_search_efficiency_full_20260724_152138"
    / "other_methods_r2"
    / "standard_recovery_test_target_variances.csv"
)
SEARCH_EFFICIENCY_CASES = (
    ROOT
    / "runs"
    / "paper_experiments"
    / "01_standard_recovery"
    / "baseline_search_efficiency_full_affinity_20260726_174347"
    / "search_efficiency"
    / "cross_method_search_efficiency_cases.csv"
)
MAIN_RESULTS = (
    ROOT
    / "reports"
    / "posthoc_recompute_inputs_20260724"
    / "01_main"
    / "results"
)
SRSD_RESCUE_FINAL = (
    ROOT
    / "runs"
    / "paper_experiments"
    / "01_standard_recovery"
    / "srsd_expression_rescue"
    / "run_20260725_225847"
    / "srsd_expression_rescue_conservative_case_rows.csv"
)
MAIN_RESCUE_FINAL = (
    ROOT
    / "runs"
    / "paper_experiments"
    / "01_standard_recovery"
    / "main_candidate_union_rescue"
    / "run_20260726_180500"
    / "all_candidate_union_rescue.csv"
)
PARETO_NA_FILL_ROOT = (
    ROOT
    / "runs"
    / "paper_experiments"
    / "01_standard_recovery"
    / "pareto_na_fill_100s"
)
ICSR_NA_FIX_MERGED_ROOT = (
    ROOT
    / "runs"
    / "paper_experiments"
    / "01_standard_recovery"
    / "pareto_na_fix_180s_full_merged"
    / "icsr"
)

TEXT = "#27313A"
AUXILIARY = "#E8EDF1"
GRID = "#D9DEE3"
WHITE = "#FFFFFF"
VEGA_BLUE = "#214F73"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "restrained_blue",
    ["#F2F5F7", "#DCE5EB", "#B9CBD7", "#789CB4", VEGA_BLUE],
)

METHOD_ORDER = ["VEGA-SR", "LLM-SR", "DSO", "gplearn", "PSE", "PySR", "ICSR"]
METHOD_COLORS = {
    "VEGA-SR": VEGA_BLUE,
    "LLM-SR": "#A8754F",
    "DSO": "#71806A",
    "gplearn": "#756C8F",
    "PSE": "#8A706A",
    "PySR": "#6F8FAF",
    "ICSR": "#7D8790",
}
METHOD_LABEL_MAP = {"Ours": "VEGA-SR"}
# The main unified recovery figure is restricted to suites that expose
# equation-recovery-compatible outputs.  SLDBench contains empirical scaling
# law prediction tasks and no ground-truth formulas in the public interface;
# it remains in source data and is reported separately with task-level R².
BENCHMARK_ORDER = ["llmsrbench", "srbench", "srsd"]
BENCHMARK_LABELS = {
    "llmsrbench": "LLMSRBench",
    "srbench": "SRBench",
    "sldbench": "SLDBench",
    "srsd": "SRSD",
}

BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260725
SYMBOLIC_TIMEOUT_SEC = 1.0


RESULT_SPECS = [
    ("VEGA-SR", "llmsrbench", "archive/ours_paper_run/llmsrbench.csv"),
    ("VEGA-SR", "sldbench", "archive/ours_paper_run/sldbench.csv"),
    ("VEGA-SR", "srbench", "archive/ours_paper_run/srbench.csv"),
    ("VEGA-SR", "srsd", "archive/ours_paper_run/srsd.csv"),
    ("LLM-SR", "llmsrbench", "paper_100s_supplement/LLM-SR/llmsrbench.csv"),
    ("LLM-SR", "sldbench", "paper_100s_supplement/LLM-SR/sldbench.csv"),
    ("LLM-SR", "srbench", "paper_100s_supplement/LLM-SR/srbench.csv"),
    ("LLM-SR", "srsd", "paper_100s_supplement/LLM-SR/srsd.csv"),
    ("DSO", "llmsrbench", "archive/baselines/DSO/llmsrbench.csv"),
    ("DSO", "sldbench", "archive/baselines/DSO/sldbench.csv"),
    ("DSO", "srbench", "archive/baselines/DSO/srbench.csv"),
    ("DSO", "srsd", "archive/baselines/DSO/srsd.csv"),
    ("gplearn", "llmsrbench", "archive/baselines/gplearn/llmsrbench.csv"),
    ("gplearn", "sldbench", "archive/baselines/gplearn/sldbench.csv"),
    ("gplearn", "srbench", "archive/baselines/gplearn/srbench.csv"),
    ("gplearn", "srsd", "archive/baselines/gplearn/srsd.csv"),
    ("PSE", "llmsrbench", "archive/baselines/PSE/llmsrbench.csv"),
    ("PSE", "sldbench", "archive/baselines/PSE/sldbench.csv"),
    ("PSE", "srbench", "paper_100s_supplement/PSE/srbench.csv"),
    ("PySR", "srsd", "paper_100s_supplement/PySR/srsd.csv"),
    ("ICSR", "llmsrbench", "archive/baselines/ICSR/llmsrbench.csv"),
    ("ICSR", "sldbench", "archive/baselines/ICSR/sldbench.csv"),
    (
        "PySR",
        "llmsrbench",
        PARETO_NA_FILL_ROOT
        / "pysr"
        / "llmsrbench"
        / "all_llmsrbench_pysr_results.csv",
    ),
    (
        "PySR",
        "srbench",
        PARETO_NA_FILL_ROOT
        / "pysr"
        / "srbench"
        / "all_srbench_pysr_results.csv",
    ),
    (
        "PSE",
        "srsd",
        PARETO_NA_FILL_ROOT
        / "psrn"
        / "srsd"
        / "all_srsd_psrn_results.csv",
    ),
]

OPTIONAL_RESULT_SPECS = [
    (
        "ICSR",
        "srbench",
        ICSR_NA_FIX_MERGED_ROOT
        / "srbench"
        / "all_srbench_official_icsr_results.csv",
    ),
    (
        "ICSR",
        "srsd",
        ICSR_NA_FIX_MERGED_ROOT
        / "srsd"
        / "all_srsd_official_icsr_results.csv",
    ),
]

NUMERIC_NA_FILL_SPECS = RESULT_SPECS[-3:] + OPTIONAL_RESULT_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return parser.parse_args()


def set_style() -> None:
    rc = {
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.edgecolor": WHITE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8.3,
        "axes.labelsize": 7.5,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 6.9,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 6.7,
        "legend.frameon": False,
        "text.color": TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "hatch.color": "#7D8790",
        "hatch.linewidth": 0.45,
    }
    sns.set_theme(context="paper", style="white", rc=rc)
    mpl.rcParams.update(rc)


def finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values))


def nonempty_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def first_text(row: pd.Series, names: Iterable[str]) -> str | None:
    for name in names:
        if name in row.index:
            text = nonempty_text(row.get(name))
            if text is not None:
                return text
    return None


def result_group_column(frame: pd.DataFrame, benchmark: str) -> pd.Series:
    """Return the case-group field that reproduces the manuscript case key."""
    if benchmark == "srsd" and "difficulty" in frame:
        return frame["difficulty"]
    if "dataset_group" in frame:
        return frame["dataset_group"]
    if "dataset_dir" in frame:
        return frame["dataset_dir"]
    return pd.Series([""] * len(frame), index=frame.index)


def parse_feature_names(row: pd.Series) -> list[str]:
    text = first_text(
        row,
        (
            "feature_names_original",
            "original_feature_names",
            "feature_names",
            "true_variables",
        ),
    )
    if text is None:
        # Some archived SRSD extracts retain only ``n_features`` while their
        # targets use zero-indexed x0, x1, ... and predictions use one-indexed
        # x1, x2, ....  Supplying the canonical one-indexed feature list lets
        # ``canonicalize_symbols`` align those documented aliases instead of
        # silently scoring equivalent formulas as different.
        try:
            n_features = int(row.get("n_features"))
        except (TypeError, ValueError):
            n_features = 0
        return [f"x{index + 1}" for index in range(max(0, n_features))]
    if text.startswith("["):
        try:
            values = ast.literal_eval(text)
            if isinstance(values, (list, tuple)):
                return [str(value).strip() for value in values if str(value).strip()]
        except Exception:
            pass
    parts = re.split(r"\s*\|\s*|\s*,\s*", text)
    return [part.strip() for part in parts if part.strip()]


def extract_return_expression(text: str) -> tuple[str, bool]:
    """Extract the final symbolic body; bool indicates unresolved parameters."""
    if "def equation" not in text and "return " not in text:
        normalized = text
    else:
        matches = re.findall(r"^\s*return\s+(.+?)\s*$", text, flags=re.MULTILINE)
        if not matches:
            return text, True
        normalized = matches[-1]
    unresolved = bool(re.search(r"\bparams\s*\[", normalized))
    normalized = re.sub(r"\bparams\s*\[\s*(\d+)\s*\]", r"p_\1", normalized)
    normalized = re.sub(r"\bnp\.", "", normalized)
    normalized = re.sub(r"\bmath\.", "", normalized)
    normalized = re.sub(r"\bX(\d+)\b", r"x\1", normalized)
    return normalized.replace("^", "**"), unresolved


SYMPY_LOCALS = {
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
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "pow": lambda a, b: a**b,
    "neg": lambda a: -a,
    "inv": lambda a: 1 / a,
    "square": lambda a: a**2,
    "cube": lambda a: a**3,
    "pi": sympy.pi,
    "E": sympy.E,
    "e": sympy.E,
}


class SymbolicTimeout(BaseException):
    pass


def _timeout_handler(_signum, _frame) -> None:
    raise SymbolicTimeout("symbolic operation timed out")


def with_symbolic_timeout(function, *args):
    if not hasattr(signal, "setitimer"):
        return function(*args)
    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, SYMBOLIC_TIMEOUT_SEC)
    try:
        return function(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


@lru_cache(maxsize=20_000)
def parse_and_simplify(expression: str) -> sympy.Expr:
    # ``sympify(..., evaluate=True)`` performs deterministic canonical
    # simplification (constant folding, flattened Add/Mul trees and like-term
    # reduction) without invoking the much more expensive heuristic simplify().
    return sympy.sympify(expression, locals=SYMPY_LOCALS, evaluate=True)


def simplify_final_expression(value: object) -> tuple[str | None, float, str, bool]:
    text = nonempty_text(value)
    if text is None:
        return None, float("nan"), "unavailable expression", False
    normalized, unresolved = extract_return_expression(text)
    try:
        simplified = parse_and_simplify(normalized)
        nodes = float(sum(1 for _ in sympy.preorder_traversal(simplified)))
        return str(simplified), nodes, "evaluable", unresolved
    except SymbolicTimeout:
        return None, float("nan"), "simplification timeout", unresolved
    except Exception:
        return None, float("nan"), "parse failure", unresolved


def operational_complexity(expression: object) -> float:
    """Return the manuscript-defined count_ops + free-symbol complexity."""
    text = nonempty_text(expression)
    if text is None:
        return float("nan")
    try:
        parsed = parse_and_simplify(text)
        return float(sympy.count_ops(parsed, visual=False) + len(parsed.free_symbols))
    except BaseException:
        return float("nan")


def preprocess_target(target: str, feature_names: list[str]) -> str:
    text = target.replace("^", "**")
    for name in feature_names:
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            continue
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\([^()]*(?:\([^()]*\)[^()]*)*\)",
            name,
            text,
        )
    return text


def canonicalize_symbols(expr: sympy.Expr, feature_names: list[str]) -> sympy.Expr:
    symbol_names = {str(symbol) for symbol in expr.free_symbols}
    zero_indexed = "x0" in symbol_names
    replacements: dict[sympy.Symbol, sympy.Symbol] = {}
    for index, feature_name in enumerate(feature_names):
        canonical = sympy.Symbol(f"__feature_{index}")
        alias = f"x{index}" if zero_indexed else f"x{index + 1}"
        replacements[sympy.Symbol(alias)] = canonical
        if not re.fullmatch(r"x\d+", feature_name):
            replacements[sympy.Symbol(feature_name)] = canonical
    return expr.xreplace(replacements)


def recovery_pair(
    predicted: str | None,
    target: str | None,
    feature_names: list[str],
    unresolved_parameters: bool,
) -> tuple[float, float, str]:
    if predicted is None or target is None:
        return float("nan"), float("nan"), "missing or unparsable expression"
    if unresolved_parameters:
        return float("nan"), float("nan"), "unresolved fitted parameters"
    try:
        target_expr = parse_and_simplify(preprocess_target(target, feature_names))
        predicted_expr = parse_and_simplify(predicted)
        predicted_expr = canonicalize_symbols(predicted_expr, feature_names)
        target_expr = canonicalize_symbols(target_expr, feature_names)

        def compare() -> tuple[float, float]:
            difference = sympy.cancel(sympy.together(predicted_expr - target_expr))
            strict = float(difference == 0)
            if strict == 1.0 or not difference.free_symbols:
                return strict, 1.0
            ratio = sympy.cancel(sympy.together(predicted_expr / target_expr))
            srbench = float(
                not ratio.free_symbols
                and ratio != 0
                and ratio.is_finite is not False
            )
            return strict, srbench

        strict, srbench = with_symbolic_timeout(compare)
        return strict, srbench, "evaluable"
    except SymbolicTimeout:
        return float("nan"), float("nan"), "symbolic comparison timeout"
    except Exception:
        return float("nan"), float("nan"), "symbolic comparison failure"


def load_numeric_cases() -> pd.DataFrame:
    paper = pd.read_csv(PAPER_ROWS).copy()
    paper["method"] = paper["method_label"].replace(METHOD_LABEL_MAP)
    paper = paper[paper["method"].isin(METHOD_ORDER)].copy()
    paper["benchmark"] = paper["benchmark"].astype(str).str.lower()
    paper["case_key"] = [
        case_key(benchmark, case_name, dataset_group)
        for benchmark, case_name, dataset_group in zip(
            paper["benchmark"],
            paper["case_name"],
            paper["dataset_group"],
        )
    ]
    variances = pd.read_csv(VARIANCE_CACHE)[
        ["case_key", "target_variance", "n_test", "variance_source"]
    ].drop_duplicates("case_key")
    paper = paper.merge(variances, how="left", on="case_key", validate="many_to_one")
    paper["test_mse"] = finite_numeric(paper["test_mse"])
    # Preserve the immutable archive value used to audit row-order-only result
    # extracts before any validated post-hoc result is overlaid.
    paper["archived_test_mse_for_alignment"] = paper["test_mse"]
    paper["target_variance"] = finite_numeric(paper["target_variance"])
    valid = (
        paper["test_mse"].notna()
        & paper["target_variance"].notna()
        & paper["target_variance"].gt(0)
    )
    paper["test_r2"] = np.nan
    paper.loc[valid, "test_r2"] = (
        1.0 - paper.loc[valid, "test_mse"] / paper.loc[valid, "target_variance"]
    )
    paper["test_nmse"] = np.nan
    paper.loc[valid, "test_nmse"] = (
        paper.loc[valid, "test_mse"] / paper.loc[valid, "target_variance"]
    )
    paper["test_rmse"] = np.sqrt(paper["test_mse"].where(paper["test_mse"].ge(0)))
    paper["r2_gt_0_999"] = np.where(
        paper["test_r2"].notna(),
        paper["test_r2"].gt(0.999).astype(float),
        np.nan,
    )
    paper["numerical_status"] = "evaluable"
    paper.loc[paper["test_mse"].isna(), "numerical_status"] = "missing test MSE"
    paper.loc[
        paper["test_mse"].notna() & paper["target_variance"].isna(),
        "numerical_status",
    ] = "missing target variance"
    paper.loc[
        paper["test_mse"].notna()
        & paper["target_variance"].notna()
        & paper["target_variance"].le(0),
        "numerical_status",
    ] = "nonpositive target variance"
    paper["source_file"] = str(PAPER_ROWS)
    if SRSD_RESCUE_FINAL.exists():
        rescue = pd.read_csv(SRSD_RESCUE_FINAL)
        rescue["case_key"] = [
            case_key("srsd", base_name, dataset_dir)
            for base_name, dataset_dir in zip(
                rescue["base_name"], rescue["dataset_dir"]
            )
        ]
        target_mask = paper["method"].eq("VEGA-SR") & paper["benchmark"].eq("srsd")
        target_keys = set(paper.loc[target_mask, "case_key"])
        rescue_keys = set(rescue["case_key"])
        if len(rescue) != 238 or target_keys != rescue_keys:
            raise ValueError("Conservative SRSD rescue case-key audit failed")
        rescue_by_key = rescue.set_index("case_key")
        keys = paper.loc[target_mask, "case_key"]
        final_nmse = keys.map(rescue_by_key["final_test_nmse"]).to_numpy(dtype=float)
        final_r2 = keys.map(rescue_by_key["final_test_r2"]).to_numpy(dtype=float)
        variance = paper.loc[target_mask, "target_variance"].to_numpy(dtype=float)
        paper.loc[target_mask, "test_nmse"] = final_nmse
        paper.loc[target_mask, "test_r2"] = final_r2
        paper.loc[target_mask, "test_mse"] = final_nmse * variance
        paper.loc[target_mask, "test_rmse"] = np.sqrt(final_nmse * variance)
        paper.loc[target_mask, "r2_gt_0_999"] = (final_r2 > 0.999).astype(float)
        paper.loc[target_mask, "numerical_status"] = "evaluable"
        paper.loc[target_mask, "source_file"] = str(SRSD_RESCUE_FINAL)
    if MAIN_RESCUE_FINAL.exists():
        rescue = pd.read_csv(MAIN_RESCUE_FINAL)
        rescue = rescue[rescue["error"].fillna("").eq("")].copy()
        rescue["case_key"] = [
            case_key(benchmark, case_name, group_name)
            for benchmark, case_name, group_name in zip(
                rescue["benchmark"],
                rescue["case_name"],
                rescue["group_name"],
            )
        ]
        rescue_by_key = rescue.set_index(["benchmark", "case_key"])
        for benchmark in ("llmsrbench", "srbench"):
            target_mask = (
                paper["method"].eq("VEGA-SR")
                & paper["benchmark"].eq(benchmark)
            )
            available_mask = target_mask & paper["case_key"].isin(
                rescue.loc[
                    rescue["benchmark"].eq(benchmark), "case_key"
                ]
            )
            keys = pd.MultiIndex.from_arrays(
                [
                    paper.loc[available_mask, "benchmark"],
                    paper.loc[available_mask, "case_key"],
                ]
            )
            final_nmse = rescue_by_key.loc[keys, "selected_test_nmse"].to_numpy(
                dtype=float
            )
            final_r2 = rescue_by_key.loc[keys, "selected_test_r2"].to_numpy(
                dtype=float
            )
            variance = paper.loc[
                available_mask, "target_variance"
            ].to_numpy(dtype=float)
            paper.loc[available_mask, "test_nmse"] = final_nmse
            paper.loc[available_mask, "test_r2"] = final_r2
            paper.loc[available_mask, "test_mse"] = final_nmse * variance
            paper.loc[available_mask, "test_rmse"] = np.sqrt(
                final_nmse * variance
            )
            paper.loc[available_mask, "r2_gt_0_999"] = (
                final_r2 > 0.999
            ).astype(float)
            paper.loc[available_mask, "numerical_status"] = "evaluable"
            paper.loc[available_mask, "source_file"] = str(MAIN_RESCUE_FINAL)

    # Replace unavailable archived rows with completed, same-budget reruns.
    # Values remain missing when a rerun itself has no finite final prediction.
    variance_by_key = variances.set_index("case_key")
    for method, benchmark, result_path in NUMERIC_NA_FILL_SPECS:
        path = Path(result_path)
        if not path.exists():
            continue
        rerun = pd.read_csv(path, low_memory=False).copy()
        if "case_name" not in rerun:
            raise ValueError(f"Missing explicit case identifiers in {path}")
        groups = result_group_column(rerun, benchmark)
        rerun["case_key"] = [
            case_key(benchmark, case_name, group)
            for case_name, group in zip(rerun["case_name"], groups)
        ]
        if rerun["case_key"].duplicated().any():
            raise ValueError(f"Duplicate case keys in {path}")
        mse_column = next(
            (
                column
                for column in ("best_test_mse", "test_mse", "clean_test_mse")
                if column in rerun
            ),
            None,
        )
        if mse_column is None:
            raise ValueError(f"Missing test MSE in {path}")
        overlay = pd.DataFrame(index=rerun.index)
        overlay["method"] = method
        overlay["benchmark"] = benchmark
        overlay["case_key"] = rerun["case_key"]
        overlay["case_name"] = rerun["case_name"]
        overlay["dataset_group"] = groups
        overlay["test_mse"] = finite_numeric(rerun[mse_column])
        overlay["archived_test_mse_for_alignment"] = overlay["test_mse"]
        overlay["target_variance"] = overlay["case_key"].map(
            variance_by_key["target_variance"]
        )
        overlay["n_test"] = overlay["case_key"].map(variance_by_key["n_test"])
        overlay["variance_source"] = overlay["case_key"].map(
            variance_by_key["variance_source"]
        )
        valid_overlay = (
            overlay["test_mse"].notna()
            & overlay["target_variance"].notna()
            & overlay["target_variance"].gt(0)
        )
        overlay["test_nmse"] = np.nan
        overlay.loc[valid_overlay, "test_nmse"] = (
            overlay.loc[valid_overlay, "test_mse"]
            / overlay.loc[valid_overlay, "target_variance"]
        )
        overlay["test_r2"] = np.nan
        overlay.loc[valid_overlay, "test_r2"] = (
            1.0 - overlay.loc[valid_overlay, "test_nmse"]
        )
        overlay["test_rmse"] = np.sqrt(
            overlay["test_mse"].where(overlay["test_mse"].ge(0))
        )
        overlay["r2_gt_0_999"] = np.where(
            overlay["test_r2"].notna(),
            overlay["test_r2"].gt(0.999).astype(float),
            np.nan,
        )
        overlay["numerical_status"] = np.where(
            valid_overlay, "evaluable", "rerun unavailable"
        )
        runtime_column = next(
            (
                column
                for column in ("fit_runtime_sec", "runtime_sec")
                if column in rerun
            ),
            None,
        )
        overlay["runtime_sec"] = (
            finite_numeric(rerun[runtime_column])
            if runtime_column is not None
            else np.nan
        )
        overlay["source_file"] = str(path)
        retained = ~(
            paper["method"].eq(method)
            & paper["benchmark"].eq(benchmark)
        )
        paper = pd.concat([paper.loc[retained], overlay], ignore_index=True, sort=False)
    return paper


def expression_column(frame: pd.DataFrame) -> str | None:
    for name in ("best_expr_sympy", "best_expr", "expression", "predicted_expression"):
        if name in frame.columns and frame[name].notna().any():
            if name == "best_expr_sympy" and not frame[name].astype(str).str.strip().replace("nan", "").ne("").any():
                continue
            return name
    return None


def target_column(frame: pd.DataFrame) -> str | None:
    for name in (
        "true_expression_for_scoring",
        "true_expression",
        "ground_truth_expression",
        "target_expression",
    ):
        if name in frame.columns and frame[name].notna().any():
            return name
    return None


def load_final_expressions(numeric: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for method, benchmark, relative_path in RESULT_SPECS + OPTIONAL_RESULT_SPECS:
        path = MAIN_RESULTS / relative_path
        if not path.exists():
            continue
        frame = pd.read_csv(path).copy()
        expr_col = expression_column(frame)
        if expr_col is None:
            continue
        target_col = target_column(frame)
        frame["method"] = method
        frame["benchmark"] = benchmark
        frame["case_alignment"] = "explicit case identifier"
        if "case_name" in frame:
            group = result_group_column(frame, benchmark)
            frame["case_key"] = [
                case_key(benchmark, case_name, dataset_group)
                for case_name, dataset_group in zip(frame["case_name"], group)
            ]
        else:
            # The archived VEGA-SR SRSD extract omitted case names. Its row order
            # is usable only after an exact audit against the paper case table.
            reference = numeric[
                numeric["method"].eq(method) & numeric["benchmark"].eq(benchmark)
            ]
            raw_mse_column = next(
                (
                    column
                    for column in ("best_test_mse", "test_mse", "clean_test_mse")
                    if column in frame
                ),
                None,
            )
            if raw_mse_column is None or len(reference) != len(frame):
                raise ValueError(f"Cannot audit row-order alignment for {path}")
            raw_mse = pd.to_numeric(frame[raw_mse_column], errors="coerce").to_numpy(dtype=float)
            alignment_column = (
                "archived_test_mse_for_alignment"
                if "archived_test_mse_for_alignment" in reference
                else "test_mse"
            )
            reference_mse = pd.to_numeric(
                reference[alignment_column], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.allclose(
                raw_mse,
                reference_mse,
                rtol=1e-12,
                atol=1e-15,
                equal_nan=True,
            ):
                raise ValueError(f"Row-order MSE audit failed for {path}")
            frame["case_key"] = reference["case_key"].to_numpy()
            frame["case_alignment"] = "audited row order; exact test-MSE sequence match"
        frame["final_expression_raw"] = frame[expr_col]
        frame["target_expression_raw"] = frame[target_col] if target_col else np.nan
        frame["feature_names_list"] = frame.apply(parse_feature_names, axis=1)
        simplified = frame["final_expression_raw"].map(simplify_final_expression)
        frame["simplified_expression"] = [item[0] for item in simplified]
        frame["simplified_node_count"] = [item[1] for item in simplified]
        frame["operational_complexity"] = frame["simplified_expression"].map(
            operational_complexity
        )
        frame["complexity_status"] = [item[2] for item in simplified]
        frame["unresolved_parameters"] = [item[3] for item in simplified]

        recovery = [
            recovery_pair(
                predicted,
                nonempty_text(target),
                features,
                unresolved,
            )
            for predicted, target, features, unresolved in zip(
                frame["simplified_expression"],
                frame["target_expression_raw"],
                frame["feature_names_list"],
                frame["unresolved_parameters"],
            )
        ]
        frame["strict_formula_recovery"] = [item[0] for item in recovery]
        frame["srbench_formula_recovery"] = [item[1] for item in recovery]
        frame["formula_status"] = [item[2] for item in recovery]
        frame["expression_source_file"] = str(path)
        rows.append(
            frame[
                [
                    "method",
                    "benchmark",
                    "case_key",
                    "final_expression_raw",
                    "simplified_expression",
                    "simplified_node_count",
                    "operational_complexity",
                    "complexity_status",
                    "target_expression_raw",
                    "feature_names_list",
                    "strict_formula_recovery",
                    "srbench_formula_recovery",
                    "formula_status",
                    "unresolved_parameters",
                    "case_alignment",
                    "expression_source_file",
                ]
            ]
        )
    expressions = pd.concat(rows, ignore_index=True)
    expressions = expressions.drop_duplicates(["method", "case_key"], keep="last")
    numeric_metrics = numeric[
        ["method", "benchmark", "case_key", "test_mse", "test_rmse", "test_r2", "test_nmse"]
    ]
    expressions = expressions.merge(
        numeric_metrics,
        how="left",
        on=["method", "benchmark", "case_key"],
        validate="one_to_one",
    )
    if SRSD_RESCUE_FINAL.exists():
        rescue = pd.read_csv(SRSD_RESCUE_FINAL)
        rescue["case_key"] = [
            case_key("srsd", base_name, dataset_dir)
            for base_name, dataset_dir in zip(
                rescue["base_name"], rescue["dataset_dir"]
            )
        ]
        target_mask = (
            expressions["method"].eq("VEGA-SR")
            & expressions["benchmark"].eq("srsd")
        )
        target_keys = set(expressions.loc[target_mask, "case_key"])
        rescue_keys = set(rescue["case_key"])
        if len(rescue) != 238 or target_keys != rescue_keys:
            raise ValueError("Conservative SRSD expression-rescue audit failed")
        rescue_by_key = rescue.set_index("case_key")
        keys = expressions.loc[target_mask, "case_key"]
        final_expressions = keys.map(rescue_by_key["final_expression"]).tolist()
        simplified = [simplify_final_expression(value) for value in final_expressions]
        expressions.loc[target_mask, "final_expression_raw"] = final_expressions
        expressions.loc[target_mask, "simplified_expression"] = [
            item[0] for item in simplified
        ]
        expressions.loc[target_mask, "simplified_node_count"] = [
            item[1] for item in simplified
        ]
        expressions.loc[target_mask, "operational_complexity"] = [
            operational_complexity(item[0]) for item in simplified
        ]
        expressions.loc[target_mask, "complexity_status"] = [
            item[2] for item in simplified
        ]
        expressions.loc[target_mask, "unresolved_parameters"] = [
            item[3] for item in simplified
        ]
        expressions.loc[target_mask, "strict_formula_recovery"] = (
            keys.map(rescue_by_key["final_strict_recovery"])
            .astype("boolean")
            .astype(float)
            .to_numpy()
        )
        expressions.loc[target_mask, "srbench_formula_recovery"] = (
            keys.map(rescue_by_key["final_srbench_recovery"])
            .astype("boolean")
            .astype(float)
            .to_numpy()
        )
        recovery_status = keys.map(rescue_by_key["final_recovery_status"]).astype(str)
        expressions.loc[target_mask, "formula_status"] = np.where(
            recovery_status.isin(
                ["evaluated", "constant_nonconstant_short_circuit"]
            ),
            "evaluable",
            recovery_status,
        )
        expressions.loc[target_mask, "case_alignment"] = (
            "explicit case key; conservative validation-only rescue"
        )
        expressions.loc[target_mask, "expression_source_file"] = str(
            SRSD_RESCUE_FINAL
        )
    if MAIN_RESCUE_FINAL.exists():
        rescue = pd.read_csv(MAIN_RESCUE_FINAL)
        rescue = rescue[rescue["error"].fillna("").eq("")].copy()
        rescue["case_key"] = [
            case_key(benchmark, case_name, group_name)
            for benchmark, case_name, group_name in zip(
                rescue["benchmark"],
                rescue["case_name"],
                rescue["group_name"],
            )
        ]
        rescue_by_key = rescue.set_index(["benchmark", "case_key"])
        for benchmark in ("llmsrbench", "srbench"):
            target_mask = (
                expressions["method"].eq("VEGA-SR")
                & expressions["benchmark"].eq(benchmark)
                & expressions["case_key"].isin(
                    rescue.loc[
                        rescue["benchmark"].eq(benchmark), "case_key"
                    ]
                )
            )
            keys = pd.MultiIndex.from_arrays(
                [
                    expressions.loc[target_mask, "benchmark"],
                    expressions.loc[target_mask, "case_key"],
                ]
            )
            final_expressions = rescue_by_key.loc[
                keys, "selected_expression"
            ].tolist()
            simplified = [
                simplify_final_expression(value) for value in final_expressions
            ]
            expressions.loc[target_mask, "final_expression_raw"] = (
                final_expressions
            )
            expressions.loc[target_mask, "simplified_expression"] = [
                item[0] for item in simplified
            ]
            expressions.loc[target_mask, "simplified_node_count"] = [
                item[1] for item in simplified
            ]
            expressions.loc[target_mask, "operational_complexity"] = [
                operational_complexity(item[0]) for item in simplified
            ]
            expressions.loc[target_mask, "complexity_status"] = [
                item[2] for item in simplified
            ]
            expressions.loc[target_mask, "unresolved_parameters"] = [
                item[3] for item in simplified
            ]
            recovery = [
                recovery_pair(
                    predicted,
                    nonempty_text(target),
                    features,
                    unresolved,
                )
                for predicted, target, features, unresolved in zip(
                    expressions.loc[target_mask, "simplified_expression"],
                    expressions.loc[target_mask, "target_expression_raw"],
                    expressions.loc[target_mask, "feature_names_list"],
                    expressions.loc[target_mask, "unresolved_parameters"],
                )
            ]
            expressions.loc[target_mask, "strict_formula_recovery"] = [
                item[0] for item in recovery
            ]
            expressions.loc[target_mask, "srbench_formula_recovery"] = [
                item[1] for item in recovery
            ]
            expressions.loc[target_mask, "formula_status"] = [
                item[2] for item in recovery
            ]
            expressions.loc[target_mask, "case_alignment"] = (
                "explicit case key; validation-only candidate-union rescue"
            )
            expressions.loc[target_mask, "expression_source_file"] = str(
                MAIN_RESCUE_FINAL
            )
    return expressions


def bootstrap_rate(
    values: Iterable[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    estimates = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        int(array.size),
    )


def summarize_panel_a(numeric: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expected = numeric.groupby("benchmark")["case_key"].nunique().to_dict()
    for method in METHOD_ORDER:
        mdf = numeric[numeric["method"].eq(method)]
        for benchmark in BENCHMARK_ORDER:
            values = mdf.loc[mdf["benchmark"].eq(benchmark), "r2_gt_0_999"]
            n_expected = int(expected.get(benchmark, 0))
            n_evaluable = int(values.notna().sum())
            n_success = int(values.eq(1).sum())
            high_accuracy_yield = (
                n_success / n_expected if n_expected else float("nan")
            )
            rows.append(
                {
                    "panel": "a",
                    "record_type": "summary",
                    "benchmark": benchmark,
                    "method": method,
                    "metric": "coverage_adjusted_test_r2_gt_0.999_yield",
                    "value": high_accuracy_yield,
                    "n": n_evaluable,
                    "n_expected": n_expected,
                    "n_success": n_success,
                    "notes": (
                        "Numerator is the number of evaluable outputs with test "
                        "R² > 0.999; denominator is every scheduled benchmark case. "
                        "Unavailable outputs remain explicitly counted in n versus "
                        "n_expected and are not assigned a numerical R² value."
                    ),
                }
            )
        main_mdf = mdf[mdf["benchmark"].isin(BENCHMARK_ORDER)]
        n_expected_total = int(
            sum(expected.get(benchmark, 0) for benchmark in BENCHMARK_ORDER)
        )
        n_evaluable_total = int(main_mdf["r2_gt_0_999"].notna().sum())
        n_success_total = int(main_mdf["r2_gt_0_999"].eq(1).sum())
        rows.append(
            {
                "panel": "a",
                "record_type": "summary",
                "benchmark": "all_cases",
                "method": method,
                "metric": "coverage_adjusted_test_r2_gt_0.999_yield",
                "value": n_success_total / n_expected_total,
                "n": n_evaluable_total,
                "n_expected": n_expected_total,
                "n_success": n_success_total,
                "notes": (
                    "Pooled coverage-adjusted high-accuracy yield across the "
                    "895 scheduled equation-recovery cases. SLDBench empirical "
                    "scaling-law tasks are reported separately."
                ),
            }
        )
    return pd.DataFrame(rows)


def load_search_efficiency_cases() -> pd.DataFrame:
    """Load first-hit candidate counts without treating censored runs as hits."""
    if not SEARCH_EFFICIENCY_CASES.exists():
        raise FileNotFoundError(
            f"Search-efficiency case table not found: {SEARCH_EFFICIENCY_CASES}"
        )
    cases = pd.read_csv(SEARCH_EFFICIENCY_CASES)
    cases["method"] = cases["method"].replace(METHOD_LABEL_MAP)
    cases["success"] = (
        cases["success"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    )
    cases["event_count"] = finite_numeric(cases["event_count"])
    cases["observed_count"] = finite_numeric(cases["observed_count"])
    cases = cases[
        cases["method"].isin(METHOD_ORDER)
        & cases["benchmark"].isin(BENCHMARK_ORDER)
    ].copy()
    cases["qualifying_hit"] = (
        cases["success"] & cases["event_count"].notna() & cases["event_count"].gt(0)
    )
    return cases


def summarize_panel_b(
    search_efficiency: pd.DataFrame,
    *,
    resamples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(METHOD_ORDER):
        method_cases = search_efficiency[search_efficiency["method"].eq(method)]
        values = finite_numeric(
            method_cases.loc[method_cases["qualifying_hit"], "event_count"]
        ).dropna()
        values = values[values.gt(0)]
        mean, ci_low, ci_high, n_hits = bootstrap_rate(
            values,
            resamples=resamples,
            seed=BOOTSTRAP_SEED + 50 + method_index,
        )
        rows.append(
            {
                "panel": "b",
                "record_type": "summary",
                "method": method,
                "metric": "mean_candidate_expressions_to_first_validation_hit",
                "value": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "q1": float(values.quantile(0.25)) if len(values) else np.nan,
                "q3": float(values.quantile(0.75)) if len(values) else np.nan,
                "n": n_hits,
                "n_completed": int(len(method_cases)),
                "n_expected": int(sum(
                    numeric_total
                    for benchmark, numeric_total in {
                        "llmsrbench": 240,
                        "srbench": 417,
                        "srsd": 238,
                    }.items()
                    if benchmark in BENCHMARK_ORDER
                )),
                "notes": (
                    "Mean and task-level bootstrap 95% CI among successful cases "
                    "only. Count is the "
                    "number of candidate expressions evaluated before the first "
                    "validation R² > 0.999 hit. Censored runs are retained in the "
                    "source table but are not assigned a first-hit count."
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_panel_c(
    expressions: pd.DataFrame,
    *,
    resamples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = expressions.dropna(
        subset=["test_r2", "test_nmse", "operational_complexity"]
    )
    valid = valid[
        valid["benchmark"].isin(BENCHMARK_ORDER)
        & valid["test_r2"].gt(0.999)
        & valid["test_nmse"].ge(0)
        & valid["operational_complexity"].ge(0)
    ]
    for method_index, method in enumerate(METHOD_ORDER):
        part = valid[valid["method"].eq(method)]
        x = part["test_nmse"].to_numpy(dtype=float)
        y = part["operational_complexity"].to_numpy(dtype=float)
        x_mean, x_low, x_high, x_n = bootstrap_rate(
            x,
            resamples=resamples,
            seed=BOOTSTRAP_SEED + 200 + method_index,
        )
        y_mean, y_low, y_high, y_n = bootstrap_rate(
            y,
            resamples=resamples,
            seed=BOOTSTRAP_SEED + 300 + method_index,
        )
        rows.append(
            {
                "panel": "c",
                "record_type": "summary",
                "method": method,
                "metric": "conditional_mean_nmse_and_nodes_among_high_accuracy_hits",
                "value": x_mean,
                "ci_low": x_low,
                "ci_high": x_high,
                "complexity_value": y_mean,
                "complexity_ci_low": y_low,
                "complexity_ci_high": y_high,
                "n": int(min(x_n, y_n)),
                "notes": (
                    "Arithmetic means and task-level bootstrap 95% CIs among final "
                    "expressions satisfying test R² > 0.999. Missing, invalid, and "
                    "sub-threshold outputs are excluded from this conditional "
                    "quality summary but remain penalized in panel a's scheduled-"
                    "case success-rate denominator."
                ),
            }
        )
    return pd.DataFrame(rows)


def mark_final_output_pareto(expressions: pd.DataFrame) -> pd.DataFrame:
    # Keep assignment one-to-one even if upstream ingestion later preserves
    # source-table indices rather than concatenating with ``ignore_index``.
    out = expressions.copy().reset_index(drop=True)
    out["pareto_evaluable"] = False
    out["on_final_output_pareto_front"] = np.nan
    valid = (
        out["test_nmse"].notna()
        & out["operational_complexity"].notna()
        & out["test_nmse"].ge(0)
        & out["operational_complexity"].ge(0)
    )
    for _, indices in out[valid].groupby(["benchmark", "case_key"]).groups.items():
        indices = list(indices)
        if len(indices) < 2:
            continue
        values = out.loc[indices, ["test_nmse", "operational_complexity"]].to_numpy(dtype=float)
        included = np.ones(len(indices), dtype=bool)
        for index, (error, complexity) in enumerate(values):
            weak = (values[:, 0] <= error) & (values[:, 1] <= complexity)
            strict = (values[:, 0] < error) | (values[:, 1] < complexity)
            weak[index] = False
            strict[index] = False
            if np.any(weak & strict):
                included[index] = False
        out.loc[indices, "pareto_evaluable"] = True
        out.loc[indices, "on_final_output_pareto_front"] = included.astype(float)
    return out


def summarize_panel_d(
    expressions: pd.DataFrame,
    numeric: pd.DataFrame,
    *,
    resamples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    case_universe = (
        numeric.loc[
            numeric["benchmark"].isin(BENCHMARK_ORDER),
            ["benchmark", "case_key"],
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    benchmark_groups = [*BENCHMARK_ORDER, "all_cases"]
    for benchmark in benchmark_groups:
        universe = (
            case_universe
            if benchmark == "all_cases"
            else case_universe[case_universe["benchmark"].eq(benchmark)]
        )
        universe_keys = set(zip(universe["benchmark"], universe["case_key"]))
        for method_index, method in enumerate(METHOD_ORDER):
            part = expressions[
                expressions["method"].eq(method)
                & expressions["benchmark"].isin(BENCHMARK_ORDER)
                & expressions["on_final_output_pareto_front"].eq(1)
            ]
            if benchmark != "all_cases":
                part = part[part["benchmark"].eq(benchmark)]
            success_keys = set(zip(part["benchmark"], part["case_key"])) & universe_keys
            values = np.fromiter(
                (
                    float((bench, key) in success_keys)
                    for bench, key in zip(universe["benchmark"], universe["case_key"])
                ),
                dtype=float,
                count=len(universe),
            )
            mean, ci_low, ci_high, n_expected = bootstrap_rate(
                values,
                resamples=resamples,
                seed=BOOTSTRAP_SEED + 100 + method_index,
            )
            rows.append(
                {
                    "panel": "d",
                    "record_type": "summary",
                    "benchmark": benchmark,
                    "method": method,
                    "metric": "total_final_output_pareto_yield",
                    "value": mean,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": int(expressions[
                        expressions["method"].eq(method)
                        & expressions["benchmark"].isin(
                            BENCHMARK_ORDER if benchmark == "all_cases" else [benchmark]
                        )
                        & expressions["pareto_evaluable"]
                    ].shape[0]),
                    "n_expected": n_expected,
                    "n_success": int(len(success_keys)),
                    "notes": (
                        "Numerator is the number of scheduled cases for which the "
                        "method's final selected expression is on the empirical "
                        "lower-NMSE/lower-complexity front. Denominator is every "
                        "scheduled case, including cases without an evaluable final "
                        "expression. The 95% CI bootstraps benchmark cases."
                    ),
                }
            )
    return pd.DataFrame(rows)


def panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )


def draw_heatmap(
    ax: mpl.axes.Axes,
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    column_labels: list[str],
    annotate_n: np.ndarray | None = None,
    annotate_denominator: np.ndarray | None = None,
    annotate_values: bool = True,
    annotation_decimals: int = 0,
    colorbar_label: str,
) -> mpl.cm.ScalarMappable:
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                color = HEATMAP_CMAP(norm(value))
                hatch = None
            else:
                color = WHITE
                hatch = "////"
            ax.add_patch(
                Rectangle(
                    (column - 0.49, row - 0.49),
                    0.98,
                    0.98,
                    facecolor=color,
                    edgecolor=WHITE if np.isfinite(value) else GRID,
                    linewidth=0.55,
                    hatch=hatch,
                )
            )
            if np.isfinite(value) and annotate_values:
                text_color = WHITE if value >= 0.64 else TEXT
                label = f"{100 * value:.{annotation_decimals}f}%"
                if annotate_n is not None:
                    n_value = int(annotate_n[row, column])
                    if annotate_denominator is not None:
                        label += f"\n$e$={n_value}"
                    elif n_value > 0:
                        label += f"\n$n$={n_value}"
                ax.text(
                    column,
                    row,
                    label,
                    ha="center",
                    va="center",
                    fontsize=6.0 if annotate_n is not None else 6.4,
                    color=text_color,
                    linespacing=0.9,
                )
            elif not np.isfinite(value):
                ax.text(column, row, "NA", ha="center", va="center", fontsize=6.0, color="#7D8790")
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    horizontal_labels = [
        label.replace("-SR", "-\nSR")
        if label in {"VEGA-SR", "LLM-SR"}
        else label
        for label in column_labels
    ]
    ax.set_xticks(
        np.arange(len(column_labels)),
        horizontal_labels,
        rotation=0,
        ha="center",
    )
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=HEATMAP_CMAP)
    mappable.set_array(matrix)
    colorbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.035, pad=0.025)
    colorbar.ax.set_title(colorbar_label, fontsize=6.2, pad=3.0, color=TEXT)
    colorbar.ax.tick_params(labelsize=6.2, width=0.6, length=2.5)
    colorbar.set_ticks([0, 0.5, 1.0], labels=["0", "50", "100"])
    return mappable


def plot_panel_a(ax: mpl.axes.Axes, summary: pd.DataFrame) -> None:
    rows = BENCHMARK_ORDER + ["all_cases"]
    matrix = np.full((len(rows), len(METHOD_ORDER)), np.nan)
    evaluable = np.zeros_like(matrix)
    expected = np.zeros_like(matrix)
    for row_index, benchmark in enumerate(rows):
        for column_index, method in enumerate(METHOD_ORDER):
            match = summary[
                summary["benchmark"].eq(benchmark) & summary["method"].eq(method)
            ]
            if len(match):
                matrix[row_index, column_index] = match["value"].iloc[0]
                evaluable[row_index, column_index] = match["n"].iloc[0]
                expected[row_index, column_index] = match["n_expected"].iloc[0]
    draw_heatmap(
        ax,
        matrix,
        row_labels=[
            (
                f"{BENCHMARK_LABELS.get(row, 'All cases')}\n"
                f"$N$={int(expected[row_index, 0])}"
            )
            for row_index, row in enumerate(rows)
        ],
        column_labels=METHOD_ORDER,
        annotate_n=evaluable,
        annotate_denominator=expected,
        annotate_values=False,
        annotation_decimals=1,
        colorbar_label="Yield\n(%)",
    )
    ax.axhline(len(BENCHMARK_ORDER) - 0.5, color=TEXT, lw=0.8)
    ax.set_title("High-accuracy yield ($R^2>0.999$)", pad=5.0)
    ax.text(
        0.0,
        -0.25,
        "Denominator: all scheduled recovery cases; SLDBench is evaluated separately",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=TEXT,
    )
    panel_label(ax, "a")


def plot_panel_b(ax: mpl.axes.Axes, summary: pd.DataFrame) -> None:
    y = np.arange(len(METHOD_ORDER))
    for yi, method in enumerate(METHOD_ORDER):
        row = summary[summary["method"].eq(method)].iloc[0]
        if int(row["n"]) == 0:
            status = (
                "not audited"
                if int(row["n_completed"]) == 0
                else "no qualifying hits"
            )
            ax.text(
                0.015,
                yi,
                status,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=6.1,
                color="#7D8790",
            )
            continue
        color = METHOD_COLORS[method]
        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [yi, yi],
            color=color,
            lw=1.6,
            solid_capstyle="round",
        )
        ax.scatter(
            row["value"],
            yi,
            s=46 if method == "VEGA-SR" else 31,
            color=color,
            edgecolor=WHITE,
            linewidth=0.55,
            zorder=3,
        )
        ax.text(
            0.985,
            yi,
            f"$n_{{hit}}$={int(row['n'])}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.1,
            bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.6, "alpha": 0.9},
            clip_on=False,
        )
    positive_low = summary.loc[summary["ci_low"].gt(0), "ci_low"]
    if len(positive_low) and summary["ci_high"].max() / positive_low.min() > 10:
        ax.set_xscale("log")
        ax.set_xlim(
            left=max(1.0, float(positive_low.min()) / 2.0),
            right=4.0 * float(summary["ci_high"].max()),
        )
    ax.set_yticks(y, METHOD_ORDER)
    ax.set_ylim(len(METHOD_ORDER) - 0.5, -0.5)
    ax.set_xlabel("Candidate expressions to first hit")
    ax.set_title("Search efficiency", pad=5.0)
    ax.text(
        0.0,
        -0.20,
        r"Mean and 95% bootstrap CI among hits; validation $R^2>0.999$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=TEXT,
    )
    panel_label(ax, "b")


def plot_panel_c(ax: mpl.axes.Axes, summary: pd.DataFrame) -> None:
    valid = summary.dropna(
        subset=[
            "value",
            "ci_low",
            "ci_high",
            "complexity_value",
            "complexity_ci_low",
            "complexity_ci_high",
        ]
    )
    for _, row in valid.iterrows():
        method = row["method"]
        color = METHOD_COLORS[method]
        x = float(row["value"])
        y = float(row["complexity_value"])
        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [y, y],
            color=color,
            lw=1.45,
            solid_capstyle="round",
        )
        ax.plot(
            [x, x],
            [row["complexity_ci_low"], row["complexity_ci_high"]],
            color=color,
            lw=1.45,
            solid_capstyle="round",
        )
        size = 66 if method == "VEGA-SR" else 42
        ax.scatter(
            x,
            y,
            s=size,
            color=color,
            edgecolor=WHITE,
            linewidth=0.6,
            zorder=3,
        )
        if method == "VEGA-SR":
            ax.annotate(
                method,
                (x, y),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=6.6,
                fontweight="bold",
                color=TEXT,
            )
    positive = valid.loc[valid["ci_low"].gt(0), "ci_low"]
    if len(positive) and valid["ci_high"].max() / positive.min() > 10:
        if (valid["ci_low"] > 0).all():
            ax.set_xscale("log")
        else:
            ax.set_xscale("symlog", linthresh=max(positive.min() / 10, 1e-16))
    ax.set_xlabel("Mean test NMSE among hits")
    ax.set_ylabel(r"Mean complexity $C$ among hits")
    ax.set_title("Accuracy–complexity trade-off", pad=5.0)
    ax.text(
        0.0,
        -0.20,
        r"Mean and 95% bootstrap CI; final outputs with test $R^2>0.999$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=TEXT,
    )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            color="none",
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=WHITE,
            markersize=5,
            label=method,
        )
        for method in METHOD_ORDER
        if method in set(valid["method"])
    ]
    ax.legend(
        handles=handles,
        loc="best",
        ncol=2,
        handletextpad=0.25,
        columnspacing=0.7,
        borderaxespad=0.3,
    )
    panel_label(ax, "c")


def plot_panel_d(ax: mpl.axes.Axes, summary: pd.DataFrame) -> None:
    total = (
        summary[summary["benchmark"].eq("all_cases")]
        .set_index("method")
        .reindex(METHOD_ORDER)
        .reset_index()
    )
    y = np.arange(len(total))
    for yi, row in total.iterrows():
        method = str(row["method"])
        color = METHOD_COLORS[method]
        value = 100.0 * float(row["value"])
        low = 100.0 * float(row["ci_low"])
        high = 100.0 * float(row["ci_high"])
        ax.plot(
            [low, high],
            [yi, yi],
            color=color,
            lw=1.55,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter(
            value,
            yi,
            s=54 if method == "VEGA-SR" else 34,
            color=color,
            edgecolor=WHITE,
            linewidth=0.6,
            zorder=3,
        )
    ax.set_yticks(y, METHOD_ORDER)
    ax.set_ylim(len(METHOD_ORDER) - 0.5, -0.5)
    ax.set_xlim(left=0)
    ax.set_xlabel("Total Pareto yield (%)")
    ax.set_title("Final-output Pareto performance", pad=5.0)
    ax.text(
        0.0,
        -0.20,
        r"95% bootstrap CI; all 895 cases in denominator",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color=TEXT,
    )
    panel_label(ax, "d")


def case_source_rows(
    numeric: pd.DataFrame,
    expressions: pd.DataFrame,
    search_efficiency: pd.DataFrame,
) -> pd.DataFrame:
    numeric_rows = numeric[
        [
            "method",
            "benchmark",
            "case_key",
            "test_mse",
            "test_rmse",
            "runtime_sec",
            "test_r2",
            "test_nmse",
            "r2_gt_0_999",
            "numerical_status",
            "target_variance",
            "n_test",
            "variance_source",
            "source_file",
        ]
    ].copy()
    numeric_rows["panel"] = "a"
    numeric_rows["record_type"] = "case_value"
    numeric_rows["metric"] = "unified_test_metrics"
    numeric_rows["value"] = numeric_rows["r2_gt_0_999"]
    numeric_rows["notes"] = (
        "MSE and RMSE are supplementary. Panel a uses the scheduled-case "
        "denominator; this row preserves evaluability and the R² threshold indicator."
    )

    formula_rows = expressions[
        [
            "method",
            "benchmark",
            "case_key",
            "strict_formula_recovery",
            "srbench_formula_recovery",
            "formula_status",
            "target_expression_raw",
            "simplified_expression",
            "expression_source_file",
        ]
    ].copy()
    formula_rows["srbench_minus_strict_paired_difference"] = (
        formula_rows["srbench_formula_recovery"]
        - formula_rows["strict_formula_recovery"]
    )
    formula_rows["panel"] = "supplementary"
    formula_rows["record_type"] = "supplementary_case_value"
    formula_rows["metric"] = "formula_recovery"
    formula_rows["notes"] = (
        "Formula-recovery results retained as supplementary source data; "
        "they are not displayed in the revised main figure."
    )

    efficiency_rows = search_efficiency[
        [
            "method",
            "benchmark",
            "case_name",
            "success",
            "qualifying_hit",
            "event_count",
            "observed_count",
            "source_json",
        ]
    ].copy()
    efficiency_rows = efficiency_rows.rename(columns={"case_name": "case_key"})
    efficiency_rows["panel"] = "b"
    efficiency_rows["record_type"] = "case_value"
    efficiency_rows["metric"] = "candidate_expressions_to_first_validation_hit"
    efficiency_rows["value"] = efficiency_rows["event_count"].where(
        efficiency_rows["qualifying_hit"]
    )
    efficiency_rows["notes"] = (
        "First validation R² > 0.999 hit. Censored or unsuccessful runs retain "
        "their observed count and do not receive an event count."
    )

    complexity_rows = expressions[
        [
            "method",
            "benchmark",
            "case_key",
            "test_mse",
            "test_rmse",
            "test_r2",
            "test_nmse",
            "simplified_node_count",
            "operational_complexity",
            "complexity_status",
            "simplified_expression",
            "expression_source_file",
        ]
    ].copy()
    complexity_rows["qualifying_accuracy_hit"] = (
        pd.to_numeric(complexity_rows["test_r2"], errors="coerce").gt(0.999)
        & pd.to_numeric(complexity_rows["test_nmse"], errors="coerce").notna()
        & pd.to_numeric(
            complexity_rows["operational_complexity"], errors="coerce"
        ).notna()
    )
    complexity_rows["panel"] = "c"
    complexity_rows["record_type"] = "case_value"
    complexity_rows["metric"] = "conditional_mean_source_nmse_and_operational_complexity"
    complexity_rows["value"] = complexity_rows["test_nmse"]
    complexity_rows["notes"] = (
        "Final selected expression only. The qualifying flag identifies outputs "
        "included in the conditional mean: test R² > 0.999 with finite NMSE and "
        "operational complexity."
    )

    pareto_rows = expressions[
        [
            "method",
            "benchmark",
            "case_key",
            "test_nmse",
            "simplified_node_count",
            "operational_complexity",
            "pareto_evaluable",
            "on_final_output_pareto_front",
            "expression_source_file",
        ]
    ].copy()
    pareto_rows["panel"] = "d"
    pareto_rows["record_type"] = "case_value"
    pareto_rows["metric"] = "final_output_pareto_inclusion"
    pareto_rows["value"] = pareto_rows["on_final_output_pareto_front"]
    pareto_rows["notes"] = (
        "Dominance evaluated across available final method outputs for the same "
        "case using lower NMSE and lower manuscript-defined operational complexity."
    )
    return pd.concat(
        [numeric_rows, efficiency_rows, complexity_rows, pareto_rows, formula_rows],
        ignore_index=True,
        sort=False,
    )


def save_source_data(
    output_path: Path,
    *,
    numeric: pd.DataFrame,
    expressions: pd.DataFrame,
    search_efficiency: pd.DataFrame,
    summaries: list[pd.DataFrame],
) -> None:
    source = pd.concat(
        [case_source_rows(numeric, expressions, search_efficiency), *summaries],
        ignore_index=True,
        sort=False,
    )
    preferred = [
        "panel",
        "record_type",
        "benchmark",
        "method",
        "case_key",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "q1",
        "q3",
        "complexity_value",
        "complexity_ci_low",
        "complexity_ci_high",
        "n",
        "n_completed",
        "n_expected",
        "n_success",
        "test_r2",
        "r2_gt_0_999",
        "test_nmse",
        "test_mse",
        "test_rmse",
        "runtime_sec",
        "success",
        "qualifying_hit",
        "event_count",
        "observed_count",
        "simplified_node_count",
        "operational_complexity",
        "qualifying_accuracy_hit",
        "strict_formula_recovery",
        "srbench_formula_recovery",
        "srbench_minus_strict_paired_difference",
        "on_final_output_pareto_front",
        "pareto_evaluable",
        "numerical_status",
        "formula_status",
        "complexity_status",
        "target_variance",
        "n_test",
        "variance_source",
        "simplified_expression",
        "target_expression_raw",
        "source_file",
        "source_json",
        "expression_source_file",
        "notes",
    ]
    columns = [column for column in preferred if column in source.columns]
    columns += [column for column in source.columns if column not in preferred]
    source[columns].to_csv(output_path, index=False)


def build_figure(
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    panel_c: pd.DataFrame,
    panel_d: pd.DataFrame,
) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(7.2, 6.45))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.105,
        right=0.945,
        top=0.875,
        bottom=0.105,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.42,
        hspace=0.58,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    plot_panel_a(ax_a, panel_a)
    plot_panel_b(ax_b, panel_b)
    plot_panel_c(ax_c, panel_c)
    plot_panel_d(ax_d, panel_d)
    fig.suptitle(
        "Unified Evaluation Separates Numerical Accuracy, Symbolic Recovery\n"
        "and Expression Simplicity",
        x=0.5,
        y=0.985,
        ha="center",
        va="top",
        fontsize=8.9,
        fontweight="bold",
        color=TEXT,
        linespacing=1.12,
    )
    return fig


def save_figure(
    fig: mpl.figure.Figure,
    output_dir: Path,
    output_stem: str = OUTPUT_STEM,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {"facecolor": WHITE, "edgecolor": WHITE, "bbox_inches": None}
    svg_path = output_dir / f"{output_stem}.svg"
    pdf_path = output_dir / f"{output_stem}.pdf"
    tiff_path = output_dir / f"{output_stem}.tiff"
    png_path = output_dir / f"{output_stem}.png"
    fig.savefig(svg_path, format="svg", **common)
    fig.savefig(pdf_path, format="pdf", **common)
    fig.savefig(
        tiff_path,
        format="tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **common,
    )
    fig.savefig(png_path, format="png", dpi=300, **common)
    for path, dpi, kwargs in [
        (tiff_path, (600, 600), {"compression": "tiff_lzw"}),
        (png_path, (300, 300), {}),
    ]:
        with Image.open(path) as image:
            rgb = Image.new("RGB", image.size, WHITE)
            if image.mode == "RGBA":
                rgb.paste(image, mask=image.getchannel("A"))
            else:
                rgb.paste(image.convert("RGB"))
            rgb.save(path, dpi=dpi, **kwargs)


def main() -> None:
    args = parse_args()
    set_style()
    numeric = load_numeric_cases()
    expressions = load_final_expressions(numeric)
    search_efficiency = load_search_efficiency_cases()
    panel_a = summarize_panel_a(numeric)
    panel_b = summarize_panel_b(
        search_efficiency,
        resamples=args.bootstrap_resamples,
    )
    panel_c = summarize_panel_c(
        expressions,
        resamples=args.bootstrap_resamples,
    )
    expressions = mark_final_output_pareto(expressions)
    panel_d = summarize_panel_d(
        expressions,
        numeric,
        resamples=args.bootstrap_resamples,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_source_data(
        args.output_dir / f"{args.output_stem}_source_data.csv",
        numeric=numeric,
        expressions=expressions,
        search_efficiency=search_efficiency,
        summaries=[panel_a, panel_b, panel_c, panel_d],
    )
    figure = build_figure(panel_a, panel_b, panel_c, panel_d)
    save_figure(figure, args.output_dir, args.output_stem)
    plt.close(figure)


if __name__ == "__main__":
    main()
