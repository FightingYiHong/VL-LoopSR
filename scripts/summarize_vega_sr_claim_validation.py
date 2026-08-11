#!/usr/bin/env python3
"""Summarize VEGA-SR claim-validation runs from the table-driven runner."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_json_maybe(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def log10_metric(value: Any) -> float | None:
    val = finite_float(value)
    return math.log10(max(val, 1e-300)) if val is not None else None


def normalize_expr_pattern(expr: Any) -> str:
    text = str(expr or "").strip().lower()
    text = re.sub(r"\b\d+(?:\.\d*)?(?:e[+-]?\d+)?\b", "c", text)
    text = re.sub(r"\s+", "", text)
    return text


def proposal_exprs(row: dict) -> list[str]:
    history = load_json_maybe(row.get("proposal_history"), [])
    exprs: list[str] = []
    for item in history if isinstance(history, list) else []:
        if not isinstance(item, dict):
            continue
        values = item.get("exprs")
        if values is None:
            values = []
        if isinstance(values, list):
            exprs.extend(str(x) for x in values if str(x or "").strip())
    if not exprs:
        raw_exprs = load_json_maybe(row.get("raw_exprs"), [])
        if isinstance(raw_exprs, list):
            exprs.extend(str(x) for x in raw_exprs if str(x or "").strip())
    return exprs


def add_efficiency_metrics(row: dict) -> dict:
    out = dict(row)
    exprs = proposal_exprs(out)
    patterns = [normalize_expr_pattern(x) for x in exprs if normalize_expr_pattern(x)]
    unique_patterns = set(patterns)
    out["total_proposed_candidate_count"] = len(exprs) if exprs else finite_float(out.get("num_candidate_exprs"))
    out["unique_candidate_pattern_count"] = len(unique_patterns) if patterns else None
    out["revisited_candidate_fraction"] = (
        max(0, len(patterns) - len(unique_patterns)) / max(1, len(patterns))
        if patterns else None
    )

    evaluation_history = load_json_maybe(out.get("evaluation_history"), [])
    evaluated = 0
    for item in evaluation_history if isinstance(evaluation_history, list) else []:
        if not isinstance(item, dict):
            continue
        count = finite_float(item.get("evaluated_candidate_count"))
        if count is not None:
            evaluated += int(count)
    out["total_evaluated_candidate_count"] = evaluated if evaluated else out.get("total_proposed_candidate_count")

    refine_history = load_json_maybe(out.get("refine_history"), [])
    if not isinstance(refine_history, list):
        refine_history = []
    out["completed_refinement_rounds"] = len(refine_history)
    out["validation_improvement_fraction"] = (
        sum(1 for x in refine_history if isinstance(x, dict) and bool(x.get("improved"))) / max(1, len(refine_history))
        if refine_history else 0.0
    )
    out["feedback_improvement_fraction"] = out["validation_improvement_fraction"]
    out.setdefault("log10_best_test_mse", log10_metric(out.get("best_test_mse")))
    out.setdefault("log10_initial_best_val_mse", log10_metric(out.get("initial_best_val_mse")))
    return out


def infer_case_name_from_path(row: dict, path: Path) -> str:
    existing = str(row.get("case_name", "") or "").strip()
    if existing:
        return existing
    method = str(row.get("method", "") or "").strip()
    case_index = row.get("case_index")
    repeat_seed = row.get("repeat_seed")
    stem = path.stem
    try:
        prefix = f"{method}_{int(case_index):03d}_"
        suffix = f"_seed{int(repeat_seed)}"
        if stem.startswith(prefix) and stem.endswith(suffix):
            return stem[len(prefix):-len(suffix)]
    except Exception:
        pass
    return stem


def infer_study(path: Path, output_root: Path) -> str:
    try:
        rel = path.relative_to(output_root)
    except ValueError:
        return ""
    parts = rel.parts
    return parts[0] if parts else ""


def collect_rows(output_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(output_root.glob("**/case_results/**/*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        row["result_json"] = str(path)
        row["study"] = infer_study(path, output_root)
        row["case_name"] = infer_case_name_from_path(row, path)
        rows.append(add_efficiency_metrics(row))
    return pd.DataFrame(rows)


def metric_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    return float(vals.mean()) if vals.notna().any() else float("nan")


def summarize_by_method(df: pd.DataFrame, studies: list[dict], efficiency_metrics: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for study in studies:
        name = study["name"]
        methods = list(study.get("methods", []) or [])
        metrics = list(dict.fromkeys(list(study.get("primary_metrics", []) or []) + list(efficiency_metrics)))
        sdf = df[df["study"].eq(name)].copy()
        for method in methods:
            mdf = sdf[sdf["method"].eq(method)].copy()
            row = {
                "study": name,
                "method": method,
                "runs": int(len(mdf)),
                "tasks": int(mdf["case_name"].nunique()) if "case_name" in mdf else 0,
                "timeout_rate": metric_mean(mdf.get("timed_out", pd.Series(dtype=float))) if not mdf.empty else float("nan"),
            }
            for metric in metrics:
                if metric in mdf:
                    row[f"mean_{metric}"] = metric_mean(mdf[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, resamples: int, confidence: float, seed: int = 20260526) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(int(resamples), dtype=float)
    for idx in range(int(resamples)):
        sample = rng.choice(values, size=values.size, replace=True)
        boot[idx] = np.mean(sample)
    alpha = 1.0 - float(confidence)
    return float(np.mean(values)), float(np.quantile(boot, alpha / 2.0)), float(np.quantile(boot, 1.0 - alpha / 2.0))


def planned_contrast_table(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows: list[dict] = []
    studies = list(config.get("studies", []) or [])
    method_to_study = {
        method: study
        for study in studies
        for method in list(study.get("methods", []) or [])
    }
    analysis = dict(config.get("analysis", {}) or {})
    resamples = int(analysis.get("bootstrap_resamples", 10000))
    confidence = float(analysis.get("confidence_level", 0.95))
    efficiency = list(analysis.get("efficiency_metrics", []) or [])
    directions = dict(analysis.get("metric_direction", {}) or {})

    for a, b in list(analysis.get("planned_contrasts", []) or []):
        study = method_to_study.get(a) or method_to_study.get(b)
        if not study:
            continue
        study_name = study["name"]
        metrics = list(dict.fromkeys(list(study.get("primary_metrics", []) or []) + efficiency))
        sdf = df[df["study"].eq(study_name) & df["method"].isin([a, b])].copy()
        for metric in metrics:
            if metric not in sdf:
                continue
            task_means = (
                sdf[["case_name", "method", metric]]
                .assign(value=pd.to_numeric(sdf[metric], errors="coerce"))
                .groupby(["case_name", "method"], dropna=False)["value"]
                .mean()
                .reset_index()
            )
            wide = task_means.pivot(index="case_name", columns="method", values="value")
            if a not in wide or b not in wide:
                continue
            paired = wide[[a, b]].dropna()
            if paired.empty:
                continue
            diff = (paired[a] - paired[b]).to_numpy(dtype=float)
            mean, lo, hi = bootstrap_ci(diff, resamples=resamples, confidence=confidence)
            rows.append({
                "study": study_name,
                "method_a": a,
                "method_b": b,
                "metric": metric,
                "direction": directions.get(metric, ""),
                "paired_tasks": int(len(paired)),
                "mean_diff_a_minus_b": mean,
                "ci_low": lo,
                "ci_high": hi,
            })
    return pd.DataFrame(rows)


def summarize(config_path: Path, output_root: Path) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    df = collect_rows(output_root)
    raw_path = output_root / "claim_validation_all_runs.csv"
    df.to_csv(raw_path, index=False)

    by_method = summarize_by_method(
        df,
        studies=list(config.get("studies", []) or []),
        efficiency_metrics=list((config.get("analysis", {}) or {}).get("efficiency_metrics", []) or []),
    )
    by_method.to_csv(output_root / "claim_validation_summary_by_method.csv", index=False)

    contrasts = planned_contrast_table(df, config)
    contrasts.to_csv(output_root / "claim_validation_planned_contrasts.csv", index=False)

    summary = {
        "config": str(config_path),
        "output_root": str(output_root),
        "run_count": int(len(df)),
        "study_counts": df.groupby("study").size().to_dict() if not df.empty and "study" in df else {},
        "files": {
            "all_runs": str(raw_path),
            "summary_by_method": str(output_root / "claim_validation_summary_by_method.csv"),
            "planned_contrasts": str(output_root / "claim_validation_planned_contrasts.csv"),
        },
    }
    (output_root / "claim_validation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vega_sr_claim_validation_v1.json")
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summarize(Path(args.config).resolve(), Path(args.output_root).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
