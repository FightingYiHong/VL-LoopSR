#!/usr/bin/env python3
"""Export a frozen, tidy non-SLDBench search-efficiency plotting snapshot."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


BENCHMARKS = ("llmsrbench", "srbench", "srsd")
TARGETS = {"llmsrbench": 240, "srbench": 417, "srsd": 238}
METHODS = ("VEGA-SR", "LLM-SR", "DSO", "gplearn", "PSE", "PySR", "ICSR")
METHOD_DIRS = {
    "LLM-SR": "llmsr",
    "DSO": "dso",
    "gplearn": "gplearn",
    "PSE": "psrn",
    "PySR": "pysr",
    "ICSR": "icsr",
}
METHOD_COLORS = {
    "VEGA-SR": "#214F73",
    "LLM-SR": "#6F8FAF",
    "DSO": "#4F7F88",
    "gplearn": "#A8754F",
    "PSE": "#71806A",
    "PySR": "#756C8F",
    "ICSR": "#7D8790",
}
OURS_FILES = {
    "llmsrbench": "all_results_detailed.csv",
    "srbench": "all_srbench_results_v10.csv",
    "srsd": "all_srds_results_v10.csv",
}
THRESHOLD = 0.999


def finite(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def first_present(mapping, names, default=None):
    for name in names:
        value = mapping.get(name)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        if str(value).strip().lower() in {"", "nan", "none", "null"}:
            continue
        return value
    return default


def case_index(path: Path, result: dict, fallback: int) -> int:
    value = finite(result.get("case_index"))
    if value is not None:
        return int(value)
    match = re.match(r"^(\d+)_", path.stem)
    return int(match.group(1)) if match else fallback


def classify_error(error) -> str:
    text = str(error or "").lower()
    if not text:
        return ""
    if any(token in text for token in ("proxyerror", "connecttimeout", "connection refused", "file not found")):
        return "data_access_failure"
    if any(token in text for token in ("non-finite", "nan", "overflow", "numerical")):
        return "numerical_failure"
    return "execution_failure"


def is_search_timeout_error(error) -> bool:
    text = str(error or "").lower()
    return any(
        token in text
        for token in (
            "outer timeout",
            "worker hard timeout",
            "search timeout",
            "search phase timeout",
            "time budget exceeded",
            "time_budget_exceeded",
        )
    )


def outcome_status(success: bool, timed_out: bool, error, valid_formula) -> str:
    if success:
        return "high_accuracy_success"
    if timed_out:
        return "search_timeout"
    if error:
        return classify_error(error)
    if as_bool(valid_formula):
        return "valid_expression_below_threshold"
    return "no_high_accuracy_recovery"


def common_row(
    *,
    method: str,
    benchmark: str,
    index: int,
    case_name: str,
    dataset_dir,
    source_path,
    success,
    event_count,
    observed_count,
    valid_formula,
    test_r2,
    test_nmse,
    node_count,
    runtime_sec,
    timed_out,
    budget_sec,
    error,
) -> dict:
    success_bool = bool(success)
    timeout_bool = bool(timed_out)
    return {
        "method_order": METHODS.index(method) + 1,
        "method": method,
        "method_color": METHOD_COLORS[method],
        "benchmark": benchmark,
        "case_index": int(index),
        "case_name": str(case_name),
        "dataset_dir": "" if dataset_dir is None else str(dataset_dir),
        "availability_status": "available",
        "outcome_status": outcome_status(success_bool, timeout_bool, error, valid_formula),
        "validation_r2_threshold": THRESHOLD,
        "high_accuracy_success": success_bool,
        "first_hit_expression_count": finite(event_count),
        "observed_expression_count": finite(observed_count),
        "valid_formula_found": as_bool(valid_formula),
        "test_r2": finite(test_r2),
        "test_nmse": finite(test_nmse),
        "simplified_node_count": finite(node_count),
        "runtime_sec": finite(runtime_sec),
        "timed_out": timeout_bool,
        "configured_budget_sec": finite(budget_sec),
        "error": "" if error is None else str(error),
        "error_category": classify_error(error),
        "source_path": str(source_path),
    }


def load_vega_rows(root: Path) -> list[dict]:
    rows = []
    for benchmark in BENCHMARKS:
        path = root / benchmark / OURS_FILES[benchmark]
        frame = pd.read_csv(path)
        if len(frame) != TARGETS[benchmark]:
            raise ValueError(f"{path} has {len(frame)} rows; expected {TARGETS[benchmark]}")
        for zero_index, series in frame.iterrows():
            result = series.to_dict()
            event_count = finite(result.get("evaluations_to_validation_success"))
            success = as_bool(result.get("validation_search_success")) and event_count is not None
            error = first_present(result, ("error", "metric_eval_error"), "")
            timed_out = as_bool(
                first_present(result, ("time_budget_hit", "timed_out", "timeout"), False)
            ) or is_search_timeout_error(error)
            rows.append(common_row(
                method="VEGA-SR",
                benchmark=benchmark,
                index=zero_index + 1,
                case_name=first_present(
                    result,
                    ("case_name", "base_name", "llmsrbench_case_name"),
                    f"{benchmark}_{zero_index + 1:04d}",
                ),
                dataset_dir=first_present(result, ("dataset_dir", "difficulty"), benchmark),
                source_path=path,
                success=success,
                event_count=event_count,
                observed_count=first_present(
                    result,
                    ("validation_search_observed_evaluations", "total_candidate_evaluations"),
                ),
                valid_formula=result.get("valid_formula_found", False),
                test_r2=result.get("test_r2"),
                test_nmse=result.get("test_nmse"),
                node_count=first_present(
                    result,
                    ("simplified_node_count", "expr_complexity", "expr_sympy_ops"),
                ),
                runtime_sec=result.get("runtime_sec"),
                timed_out=timed_out,
                budget_sec=first_present(
                    result,
                    ("time_budget_sec", "configured_wall_budget_sec", "case_timeout_sec"),
                ),
                error=error,
            ))
    return rows


def load_baseline_rows(root: Path) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    rows = []
    canonical = {}
    for method in METHODS[1:]:
        method_dir = METHOD_DIRS[method]
        for benchmark in BENCHMARKS:
            paths = sorted((root / method_dir / benchmark / "case_results").glob("*.json"))
            # Freeze the file list at function entry so a live ICSR worker cannot
            # change the snapshot halfway through this export.
            for fallback, path in enumerate(paths, start=1):
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    raise ValueError(f"Cannot read {path}: {exc}") from exc
                index = case_index(path, result, fallback)
                event_count = finite(result.get("evaluations_to_validation_success"))
                success = as_bool(result.get("validation_search_success")) and event_count is not None
                error = first_present(
                    result,
                    ("error", "failure_reason", "termination_reason"),
                    "",
                )
                timed_out = as_bool(
                    first_present(result, ("timed_out", "timeout"), False)
                ) or is_search_timeout_error(error)
                row = common_row(
                    method=method,
                    benchmark=benchmark,
                    index=index,
                    case_name=first_present(
                        result,
                        ("case_name", "base_name"),
                        path.stem,
                    ),
                    dataset_dir=first_present(
                        result,
                        ("dataset_dir", "difficulty"),
                        benchmark,
                    ),
                    source_path=path,
                    success=success,
                    event_count=event_count,
                    observed_count=result.get("validation_search_observed_evaluations"),
                    valid_formula=first_present(
                        result,
                        ("valid_formula_found", "valid_expression_found"),
                        bool(first_present(result, ("best_expr", "expression"), "")),
                    ),
                    test_r2=result.get("test_r2"),
                    test_nmse=result.get("test_nmse"),
                    node_count=first_present(
                        result,
                        ("simplified_node_count", "expr_complexity", "expr_sympy_ops"),
                    ),
                    runtime_sec=result.get("runtime_sec"),
                    timed_out=timed_out,
                    budget_sec=first_present(
                        result,
                        ("configured_wall_budget_sec", "case_timeout_sec"),
                    ),
                    error=error,
                )
                rows.append(row)
                if method == "PSE":
                    canonical[(benchmark, index)] = row
    return rows, canonical


def add_unavailable_rows(rows: list[dict], canonical: dict[tuple[str, int], dict]) -> None:
    present = {
        (row["method"], row["benchmark"], row["case_index"])
        for row in rows
    }
    for method in METHODS:
        for benchmark in BENCHMARKS:
            for index in range(1, TARGETS[benchmark] + 1):
                key = (method, benchmark, index)
                if key in present:
                    continue
                template = canonical.get((benchmark, index), {})
                rows.append({
                    "method_order": METHODS.index(method) + 1,
                    "method": method,
                    "method_color": METHOD_COLORS[method],
                    "benchmark": benchmark,
                    "case_index": index,
                    "case_name": template.get("case_name", f"{benchmark}_{index:04d}"),
                    "dataset_dir": template.get("dataset_dir", benchmark),
                    "availability_status": "unavailable_incomplete_run",
                    "outcome_status": "unavailable_incomplete_run",
                    "validation_r2_threshold": THRESHOLD,
                    "high_accuracy_success": pd.NA,
                    "first_hit_expression_count": np.nan,
                    "observed_expression_count": np.nan,
                    "valid_formula_found": pd.NA,
                    "test_r2": np.nan,
                    "test_nmse": np.nan,
                    "simplified_node_count": np.nan,
                    "runtime_sec": np.nan,
                    "timed_out": pd.NA,
                    "configured_budget_sec": np.nan,
                    "error": "Result unavailable at frozen snapshot time; not treated as a numerical zero.",
                    "error_category": "unavailable_incomplete_run",
                    "source_path": "",
                })


def bootstrap_rate_ci(success: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(success) == 0:
        return math.nan, math.nan
    draws = rng.choice(success, size=(20_000, len(success)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarize(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260729)
    scopes = [*BENCHMARKS, "overall"]
    for method in METHODS:
        method_rows = cases[cases["method"] == method]
        for scope in scopes:
            frame = method_rows if scope == "overall" else method_rows[method_rows["benchmark"] == scope]
            denominator = sum(TARGETS.values()) if scope == "overall" else TARGETS[scope]
            available = frame["availability_status"].eq("available")
            success = frame["high_accuracy_success"].eq(True)
            hit_counts = pd.to_numeric(
                frame.loc[success, "first_hit_expression_count"],
                errors="coerce",
            ).dropna()
            complete = int(available.sum()) == denominator
            if complete:
                low, high = bootstrap_rate_ci(success.astype(float).to_numpy(), rng)
            else:
                low, high = math.nan, math.nan
            rows.append({
                "method_order": METHODS.index(method) + 1,
                "method": method,
                "method_color": METHOD_COLORS[method],
                "benchmark_scope": scope,
                "denominator_total_cases": denominator,
                "available_result_records": int(available.sum()),
                "unavailable_result_records": int((~available).sum()),
                "snapshot_complete": complete,
                "high_accuracy_successes": int(success.sum()),
                "high_accuracy_success_rate": float(success.sum() / denominator),
                "high_accuracy_success_rate_percent": float(100 * success.sum() / denominator),
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "bootstrap_95ci_low_percent": 100 * low if math.isfinite(low) else math.nan,
                "bootstrap_95ci_high_percent": 100 * high if math.isfinite(high) else math.nan,
                "successful_cases_for_expression_count": int(len(hit_counts)),
                "mean_expressions_to_success_successes_only": float(hit_counts.mean()) if len(hit_counts) else math.nan,
                "median_expressions_to_success_successes_only": float(hit_counts.median()) if len(hit_counts) else math.nan,
                "q25_expressions_to_success_successes_only": float(hit_counts.quantile(0.25)) if len(hit_counts) else math.nan,
                "q75_expressions_to_success_successes_only": float(hit_counts.quantile(0.75)) if len(hit_counts) else math.nan,
                "timeouts_available_records": int(
                    frame.loc[available, "timed_out"].eq(True).sum()
                ),
                "rate_denominator_note": "All benchmark cases in scope, including unavailable ICSR cases.",
                "expression_count_note": "Mean/median/IQR use successful cases only.",
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vega-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    snapshot_time = datetime.now().astimezone().isoformat(timespec="seconds")
    baseline_rows, canonical = load_baseline_rows(args.baseline_root)
    rows = load_vega_rows(args.vega_root) + baseline_rows
    add_unavailable_rows(rows, canonical)
    cases = pd.DataFrame(rows).sort_values(
        ["method_order", "benchmark", "case_index"],
        kind="stable",
    ).reset_index(drop=True)
    summary = summarize(cases)

    expected_rows = len(METHODS) * sum(TARGETS.values())
    if len(cases) != expected_rows:
        raise ValueError(f"Expected {expected_rows} tidy rows; got {len(cases)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_path = args.output_dir / "search_efficiency_plot_cases_non_sldbench.csv"
    summary_path = args.output_dir / "search_efficiency_plot_summary_non_sldbench.csv"
    manifest_path = args.output_dir / "search_efficiency_plot_snapshot_manifest.json"
    cases.to_csv(case_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest = {
        "snapshot_time": snapshot_time,
        "scope": list(BENCHMARKS),
        "excluded_benchmarks": ["sldbench"],
        "validation_r2_success_rule": "validation R^2 > 0.999",
        "denominator_total_cases_per_method": sum(TARGETS.values()),
        "benchmark_targets": TARGETS,
        "method_order": list(METHODS),
        "method_colors": METHOD_COLORS,
        "icsr_policy": (
            "Use the frozen partial ICSR result. Missing cases remain explicitly unavailable "
            "in the tidy case table and remain in the total-case denominator."
        ),
        "budget_warning": (
            "This operational snapshot contains rows produced before and after the transition "
            "from a 600-second to a 100-second search budget. Do not describe it as a "
            "strictly equal-budget comparison without stratifying by configured_budget_sec."
        ),
        "vega_root": str(args.vega_root.resolve()),
        "baseline_root": str(args.baseline_root.resolve()),
        "case_csv": str(case_path.resolve()),
        "summary_csv": str(summary_path.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary[summary["benchmark_scope"] == "overall"].to_string(index=False))
    print(f"\nWrote {case_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
