#!/usr/bin/env python3
"""Summarize first-hit search efficiency on the 972-task main benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.plot_style import BENCHMARK_COLORS, NATURE_COLORS, save_nature_figure, set_nature_style


BENCHMARKS = ("llmsrbench", "sldbench", "srbench", "srsd")
EXPECTED_TOTALS = {
    "llmsrbench": 240,
    "sldbench": 77,
    "srbench": 417,
    "srsd": 238,
}
RESULT_FILENAMES = (
    "all_results_detailed.csv",
    "all_sldbench_results_v10.csv",
    "all_llmsrbench_results_v10.csv",
    "all_srds_results_v10.csv",
    "all_srbench_results_v10.csv",
    "global_summary.csv",
)
SUCCESS_AT_K = (1, 5, 10, 20, 50, 100, 200)


def as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def finite_float(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def first_value(row: pd.Series, names, default=None):
    for name in names:
        if name not in row:
            continue
        value = row[name]
        if pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null"}:
            continue
        return value
    return default


def find_result_csv(root: Path, benchmark: str) -> Path | None:
    bench_dir = root / benchmark
    for name in RESULT_FILENAMES:
        path = bench_dir / name
        if path.exists() and path.stat().st_size:
            return path
    for path in sorted(bench_dir.glob("all_*results*.csv")):
        if path.stat().st_size:
            return path
    return None


def parse_history(value) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def derive_search_fields(row: pd.Series, threshold: float) -> dict:
    history = parse_history(row.get("candidate_evaluation_history"))
    first_hit = next(
        (
            item
            for item in history
            if finite_float(item.get("val_r2")) is not None
            and float(item["val_r2"]) > threshold
        ),
        None,
    )

    total = finite_float(first_value(
        row,
        ("total_candidate_evaluations", "expression_evaluations"),
        len(history),
    ))
    total = int(total or 0)
    total_unique = finite_float(first_value(
        row,
        ("total_unique_candidate_evaluations",),
        max((int(item.get("unique_evaluations_seen") or 0) for item in history), default=0),
    ))
    total_unique = int(total_unique or 0)

    reported_success = first_value(row, ("validation_search_success",), None)
    success = as_bool(reported_success) if reported_success is not None else first_hit is not None
    evaluations_to_hit = finite_float(first_value(
        row,
        ("evaluations_to_validation_success",),
        first_hit.get("evaluation_index") if first_hit else None,
    ))
    unique_to_hit = finite_float(first_value(
        row,
        ("unique_evaluations_to_validation_success",),
        first_hit.get("unique_evaluations_seen") if first_hit else None,
    ))
    if evaluations_to_hit is None:
        success = False
    observed = int(evaluations_to_hit) if success else total

    return {
        "validation_search_success": bool(success),
        "validation_search_censored": not bool(success),
        "evaluations_to_validation_success": int(evaluations_to_hit) if success else np.nan,
        "unique_evaluations_to_validation_success": int(unique_to_hit) if success and unique_to_hit is not None else np.nan,
        "validation_search_observed_evaluations": int(observed),
        "total_candidate_evaluations": int(total),
        "total_unique_candidate_evaluations": int(total_unique),
        "first_validation_success_stage": first_value(
            row,
            ("first_validation_success_stage",),
            first_hit.get("stage") if first_hit else "",
        ),
        "first_validation_success_expression": first_value(
            row,
            ("first_validation_success_expression",),
            (first_hit.get("fitted_expression") or first_hit.get("expression")) if first_hit else "",
        ),
        "first_validation_success_val_r2": finite_float(first_value(
            row,
            ("first_validation_success_val_r2",),
            first_hit.get("val_r2") if first_hit else None,
        )),
    }


def normalize_results(path: Path, benchmark: str, threshold: float) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows = []
    for idx, row in raw.iterrows():
        case_name = first_value(
            row,
            ("case_name", "base_name", "llmsrbench_case_name", "dataset_dir"),
            f"{benchmark}_{idx:04d}",
        )
        fields = derive_search_fields(row, threshold)
        rows.append({
            "benchmark": benchmark,
            "case_name": str(case_name),
            "source_csv": str(path),
            "runtime_sec": finite_float(first_value(row, ("runtime_sec", "elapsed_sec"))),
            "evaluations_per_sec": finite_float(first_value(row, ("evaluations_per_sec",))),
            "time_budget_hit": as_bool(first_value(row, ("time_budget_hit", "timed_out", "timeout"), False)),
            **fields,
        })
    return pd.DataFrame(rows)


def summarize_group(frame: pd.DataFrame, benchmark: str, threshold: float) -> dict:
    n = len(frame)
    hit = frame["validation_search_success"].astype(bool)
    hit_counts = pd.to_numeric(
        frame.loc[hit, "evaluations_to_validation_success"],
        errors="coerce",
    ).dropna()
    row = {
        "benchmark": benchmark,
        "n": int(n),
        "expected_n": int(sum(EXPECTED_TOTALS.values()) if benchmark == "overall" else EXPECTED_TOTALS[benchmark]),
        "validation_r2_threshold": float(threshold),
        "success_n": int(hit.sum()),
        "success_rate": float(hit.mean()) if n else math.nan,
        "censored_n": int((~hit).sum()),
        "median_evaluations_to_success_successes_only": float(hit_counts.median()) if len(hit_counts) else math.nan,
        "q25_evaluations_to_success_successes_only": float(hit_counts.quantile(0.25)) if len(hit_counts) else math.nan,
        "q75_evaluations_to_success_successes_only": float(hit_counts.quantile(0.75)) if len(hit_counts) else math.nan,
        "mean_total_candidate_evaluations": float(frame["total_candidate_evaluations"].mean()) if n else math.nan,
        "median_total_candidate_evaluations": float(frame["total_candidate_evaluations"].median()) if n else math.nan,
        "median_unique_candidate_evaluations": float(frame["total_unique_candidate_evaluations"].median()) if n else math.nan,
        "median_runtime_sec": float(pd.to_numeric(frame["runtime_sec"], errors="coerce").median()) if n else math.nan,
        "median_evaluations_per_sec": float(pd.to_numeric(frame["evaluations_per_sec"], errors="coerce").median()) if n else math.nan,
        "timeout_n": int(frame["time_budget_hit"].astype(bool).sum()),
    }
    for k in SUCCESS_AT_K:
        row[f"success_at_{k}_n"] = int((hit & frame["evaluations_to_validation_success"].le(k)).sum())
        row[f"success_at_{k}_rate"] = (
            float((hit & frame["evaluations_to_validation_success"].le(k)).sum() / n)
            if n else math.nan
        )
    return row


def build_success_curve(frame: pd.DataFrame) -> pd.DataFrame:
    max_count = int(max(1, frame["validation_search_observed_evaluations"].max()))
    points = sorted(set([1, *SUCCESS_AT_K, *range(10, max_count + 1, 10), max_count]))
    rows = []
    groups = [(name, sub) for name, sub in frame.groupby("benchmark", sort=False)]
    groups.append(("overall", frame))
    for benchmark, sub in groups:
        n = len(sub)
        hit = sub["validation_search_success"].astype(bool)
        for count in points:
            rows.append({
                "benchmark": benchmark,
                "evaluation_count": int(count),
                "success_n": int((hit & sub["evaluations_to_validation_success"].le(count)).sum()),
                "success_fraction_all_tasks": (
                    float((hit & sub["evaluations_to_validation_success"].le(count)).sum() / n)
                    if n else math.nan
                ),
                "n": int(n),
            })
    return pd.DataFrame(rows)


def plot_results(case_rows: pd.DataFrame, curve: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt

    set_nature_style(plt=plt)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1.0]})

    for benchmark in (*BENCHMARKS, "overall"):
        sub = curve[curve["benchmark"] == benchmark].sort_values("evaluation_count")
        if sub.empty:
            continue
        color = NATURE_COLORS["dark"] if benchmark == "overall" else BENCHMARK_COLORS[benchmark]
        axes[0].step(
            sub["evaluation_count"],
            100.0 * sub["success_fraction_all_tasks"],
            where="post",
            color=color,
            linewidth=1.8 if benchmark == "overall" else 1.2,
            label="Overall" if benchmark == "overall" else benchmark.upper(),
        )
    axes[0].set_xlabel("Candidate evaluations")
    axes[0].set_ylabel("Tasks reaching validation $R^2 > 0.999$ (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", alpha=0.35)

    labels = []
    data = []
    colors = []
    for benchmark in BENCHMARKS:
        sub = case_rows[
            (case_rows["benchmark"] == benchmark)
            & case_rows["validation_search_success"].astype(bool)
        ]
        if sub.empty:
            continue
        labels.append(benchmark.upper())
        data.append(sub["evaluations_to_validation_success"].dropna().to_numpy(dtype=float))
        colors.append(BENCHMARK_COLORS[benchmark])
    if data:
        box = axes[1].boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True, widths=0.58)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor(color)
        for median in box["medians"]:
            median.set_color("white")
            median.set_linewidth(1.4)
        rng = np.random.default_rng(42)
        for idx, (values, color) in enumerate(zip(data, colors), start=1):
            jitter = rng.normal(0.0, 0.035, size=len(values))
            axes[1].scatter(
                np.full(len(values), idx, dtype=float) + jitter,
                values,
                s=12,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
                alpha=0.8,
            )
    else:
        axes[1].text(0.5, 0.5, "No threshold hits", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_xticks([])
    axes[1].set_ylabel("Evaluations to first hit (successful tasks)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.35)

    for ax, panel in zip(axes, ("a", "b")):
        ax.text(-0.13, 1.04, panel, transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.tight_layout()

    png = output_dir / "fig_standard_search_efficiency_validation_r2.png"
    pdf = output_dir / "fig_standard_search_efficiency_validation_r2.pdf"
    save_nature_figure(fig, png, section="standard_recovery", dpi=300, bbox_inches="tight")
    save_nature_figure(fig, pdf, section="standard_recovery", bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.999)
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_root / "search_efficiency"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    missing = []
    for benchmark in BENCHMARKS:
        path = find_result_csv(args.results_root, benchmark)
        if path is None:
            missing.append(benchmark)
            continue
        frames.append(normalize_results(path, benchmark, args.threshold))
    if not frames:
        raise FileNotFoundError(f"No benchmark result CSVs found below {args.results_root}")

    case_rows = pd.concat(frames, ignore_index=True)
    summary_rows = [
        summarize_group(sub, benchmark, args.threshold)
        for benchmark, sub in case_rows.groupby("benchmark", sort=False)
    ]
    summary_rows.append(summarize_group(case_rows, "overall", args.threshold))
    summary = pd.DataFrame(summary_rows)
    curve = build_success_curve(case_rows)

    case_rows.to_csv(output_dir / "standard_search_efficiency_case_rows.csv", index=False)
    summary.to_csv(output_dir / "standard_search_efficiency_summary.csv", index=False)
    curve.to_csv(output_dir / "standard_search_efficiency_curve.csv", index=False)
    plot_results(case_rows, curve, output_dir)

    print(summary.to_string(index=False))
    print(f"[INFO] case rows: {len(case_rows)} / expected {sum(EXPECTED_TOTALS.values())}")
    if missing:
        print(f"[WARN] missing benchmark results: {','.join(missing)}")
    print(f"[INFO] outputs: {output_dir}")


if __name__ == "__main__":
    main()
