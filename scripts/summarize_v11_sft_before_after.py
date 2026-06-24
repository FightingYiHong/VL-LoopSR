#!/usr/bin/env python3
"""Summarize and plot before/after-SFT V11 experiment results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BENCHMARKS = ("sldbench", "llmsrbench", "srsd", "srbench")
EXPECTED_TOTALS = {
    "sldbench": 77,
    "llmsrbench": 240,
    "srsd": 238,
    "srbench": 417,
}
RESULT_CANDIDATES = (
    "global_summary.csv",
    "all_results_detailed.csv",
    "all_sldbench_results_v10.csv",
    "all_llmsrbench_results_v10.csv",
    "all_srds_results_v10.csv",
    "all_srbench_results_v10.csv",
)


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "t"}


def finite_numeric(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def first_existing(row: pd.Series, names: Iterable[str], default=""):
    for name in names:
        if name in row and pd.notna(row[name]):
            value = row[name]
            if str(value).strip().lower() not in {"", "nan", "none", "null"}:
                return value
    return default


def find_result_csv(root: Path, benchmark: str) -> Path | None:
    bench_dir = root / benchmark
    for name in RESULT_CANDIDATES:
        path = bench_dir / name
        if path.exists() and path.stat().st_size > 0:
            return path
    matches = sorted(bench_dir.glob("all_*results*.csv"))
    for path in matches:
        if path.stat().st_size > 0:
            return path
    return None


def make_case_id(df: pd.DataFrame) -> pd.Series:
    parts = []
    for name in ("case_name", "base_name", "llmsrbench_case_name", "dataset_dir", "difficulty", "task_type"):
        if name in df.columns:
            parts.append(df[name].fillna("").astype(str))
    if not parts:
        return pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + "::" + part
    row_fallback = pd.Series([f"row_{i}" for i in range(len(df))], index=df.index)
    return combined.where(combined.str.replace("::", "", regex=False).str.len() > 0, row_fallback)


def normalize_result_frame(path: Path, benchmark: str, method_label: str, pass_threshold: float) -> pd.DataFrame:
    raw = pd.read_csv(path)
    out = pd.DataFrame(index=raw.index)
    out["method"] = method_label
    out["benchmark"] = benchmark
    out["source_csv"] = str(path)
    out["case_id"] = make_case_id(raw)
    out["case_name"] = raw.apply(
        lambda r: first_existing(r, ("case_name", "base_name", "llmsrbench_case_name", "dataset_dir"), ""),
        axis=1,
    )
    out["best_expr"] = raw.apply(lambda r: first_existing(r, ("best_expr", "expression", "expr", "formula"), ""), axis=1)
    out["valid_formula_found"] = raw.apply(
        lambda r: as_bool(first_existing(r, ("valid_formula_found", "valid_formula", "generated_formula"), False)),
        axis=1,
    )
    out["best_test_mse"] = finite_numeric(raw.get("best_test_mse", raw.get("test_mse", pd.Series(np.nan, index=raw.index))))
    out["test_r2"] = finite_numeric(raw.get("test_r2", raw.get("r2", pd.Series(np.nan, index=raw.index))))
    out["runtime_sec"] = finite_numeric(raw.get("runtime_sec", raw.get("elapsed_sec", pd.Series(np.nan, index=raw.index))))
    out["expr_complexity"] = finite_numeric(raw.get("expr_complexity", raw.get("best_complexity", pd.Series(np.nan, index=raw.index))))
    out["num_candidate_exprs"] = finite_numeric(raw.get("num_candidate_exprs", raw.get("candidate_count", pd.Series(np.nan, index=raw.index))))
    out["time_budget_hit"] = raw.apply(lambda r: as_bool(first_existing(r, ("time_budget_hit", "timed_out", "timeout"), False)), axis=1)
    out["error"] = raw.get("error", pd.Series("", index=raw.index)).fillna("").astype(str)

    reported_pass = raw.apply(lambda r: first_existing(r, ("passed", "pass", "is_pass"), None), axis=1)
    has_reported = reported_pass.notna()
    out["passed_reported"] = reported_pass.map(as_bool)
    out["pass_at_threshold"] = out["best_test_mse"].le(pass_threshold).fillna(False)
    out.loc[has_reported, "pass_at_threshold"] = out.loc[has_reported, "passed_reported"]
    out["pass_at_100"] = out["best_test_mse"].le(100.0).fillna(False)
    out["pass_at_300"] = out["best_test_mse"].le(300.0).fillna(False)

    if "perfect_fit" in raw.columns:
        out["perfect_fit"] = raw["perfect_fit"].map(as_bool)
    else:
        out["perfect_fit"] = out["best_test_mse"].le(1.0e-12).fillna(False)
    if "perfect_fit_by_r2" in raw.columns:
        out["perfect_fit_by_r2"] = raw["perfect_fit_by_r2"].map(as_bool)
    else:
        out["perfect_fit_by_r2"] = out["test_r2"].ge(0.999999).fillna(False)
    if "skeleton_recovery" in raw.columns:
        out["skeleton_recovery"] = raw["skeleton_recovery"].map(as_bool)
    else:
        out["skeleton_recovery"] = out["perfect_fit_by_r2"] | out["pass_at_threshold"]
    return out


def load_run(root: Path, method_label: str, pass_threshold: float) -> pd.DataFrame:
    frames = []
    missing = []
    for benchmark in BENCHMARKS:
        path = find_result_csv(root, benchmark)
        if path is None:
            missing.append(benchmark)
            continue
        frames.append(normalize_result_frame(path, benchmark, method_label, pass_threshold))
    if not frames:
        raise FileNotFoundError(f"no result CSV files found under {root}")
    out = pd.concat(frames, ignore_index=True)
    out.attrs["missing_benchmarks"] = missing
    return out


def mean_or_nan(series: pd.Series) -> float:
    series = finite_numeric(series).dropna()
    return float(series.mean()) if len(series) else math.nan


def median_or_nan(series: pd.Series) -> float:
    series = finite_numeric(series).dropna()
    return float(series.median()) if len(series) else math.nan


def summarize_group(df: pd.DataFrame, label: str, pass_threshold: float) -> dict:
    n = int(len(df))
    pass_mask = df["pass_at_threshold"].astype(bool)
    pass100 = df["pass_at_100"].astype(bool)
    pass300 = df["pass_at_300"].astype(bool)
    valid = df["valid_formula_found"].astype(bool)
    finite_mse = df["best_test_mse"].notna()
    return {
        "method": label,
        "benchmark": str(df["benchmark"].iloc[0]) if n else "",
        "n": n,
        "expected_n": EXPECTED_TOTALS.get(str(df["benchmark"].iloc[0]), n) if n else n,
        "valid_n": int(valid.sum()),
        "finite_mse_n": int(finite_mse.sum()),
        "pass_threshold": float(pass_threshold),
        "pass_n": int(pass_mask.sum()),
        "pass_rate": float(pass_mask.mean()) if n else math.nan,
        "pass100_n": int(pass100.sum()),
        "pass100_rate": float(pass100.mean()) if n else math.nan,
        "pass300_n": int(pass300.sum()),
        "pass300_rate": float(pass300.mean()) if n else math.nan,
        "perfect_fit_n": int(df["perfect_fit"].astype(bool).sum()),
        "perfect_fit_rate": float(df["perfect_fit"].astype(bool).mean()) if n else math.nan,
        "r2_perfect_n": int(df["perfect_fit_by_r2"].astype(bool).sum()),
        "r2_perfect_rate": float(df["perfect_fit_by_r2"].astype(bool).mean()) if n else math.nan,
        "skeleton_n": int(df["skeleton_recovery"].astype(bool).sum()),
        "skeleton_rate": float(df["skeleton_recovery"].astype(bool).mean()) if n else math.nan,
        "mean_mse_pass": mean_or_nan(df.loc[pass_mask, "best_test_mse"]),
        "median_mse_pass": median_or_nan(df.loc[pass_mask, "best_test_mse"]),
        "median_mse_finite": median_or_nan(df.loc[finite_mse, "best_test_mse"]),
        "mean_r2_finite": mean_or_nan(df["test_r2"]),
        "median_r2_finite": median_or_nan(df["test_r2"]),
        "mean_runtime_sec": mean_or_nan(df["runtime_sec"]),
        "median_runtime_sec": median_or_nan(df["runtime_sec"]),
        "mean_complexity": mean_or_nan(df.loc[valid, "expr_complexity"]),
        "median_complexity": median_or_nan(df.loc[valid, "expr_complexity"]),
        "time_budget_hit_n": int(df["time_budget_hit"].astype(bool).sum()),
        "error_n": int(df["error"].fillna("").astype(str).str.strip().ne("").sum()),
    }


def build_summary(df: pd.DataFrame, pass_threshold: float) -> pd.DataFrame:
    rows = []
    for (method, benchmark), sub in df.groupby(["method", "benchmark"], sort=False):
        rows.append(summarize_group(sub, str(method), pass_threshold))
    for method, sub in df.groupby("method", sort=False):
        overall = summarize_group(sub.assign(benchmark="overall"), str(method), pass_threshold)
        overall["benchmark"] = "overall"
        overall["expected_n"] = sum(EXPECTED_TOTALS.values())
        rows.append(overall)
    return pd.DataFrame(rows)


def build_pairwise(before: pd.DataFrame, after: pd.DataFrame, before_label: str, after_label: str, mse_tol: float) -> pd.DataFrame:
    left = before.add_prefix("before_")
    right = after.add_prefix("after_")
    merged = left.merge(
        right,
        left_on=["before_benchmark", "before_case_id"],
        right_on=["after_benchmark", "after_case_id"],
        how="inner",
    )
    merged["benchmark"] = merged["before_benchmark"]
    merged["case_id"] = merged["before_case_id"]
    merged["before_label"] = before_label
    merged["after_label"] = after_label
    merged["mse_delta"] = merged["after_best_test_mse"] - merged["before_best_test_mse"]
    merged["log_mse_before"] = np.log10(np.clip(merged["before_best_test_mse"].astype(float), 1.0e-12, 1.0e300))
    merged["log_mse_after"] = np.log10(np.clip(merged["after_best_test_mse"].astype(float), 1.0e-12, 1.0e300))
    merged["log_mse_delta"] = merged["log_mse_after"] - merged["log_mse_before"]
    before_mse = merged["before_best_test_mse"].astype(float)
    after_mse = merged["after_best_test_mse"].astype(float)
    finite_pair = np.isfinite(before_mse) & np.isfinite(after_mse)
    merged["mse_win"] = "tie"
    merged.loc[finite_pair & (after_mse < before_mse * (1.0 - mse_tol)), "mse_win"] = "after"
    merged.loc[finite_pair & (before_mse < after_mse * (1.0 - mse_tol)), "mse_win"] = "before"
    merged.loc[~finite_pair & np.isfinite(after_mse), "mse_win"] = "after"
    merged.loc[~finite_pair & np.isfinite(before_mse), "mse_win"] = "before"
    merged["pass_change"] = "same"
    merged.loc[(~merged["before_pass_at_threshold"].astype(bool)) & merged["after_pass_at_threshold"].astype(bool), "pass_change"] = "gain"
    merged.loc[merged["before_pass_at_threshold"].astype(bool) & (~merged["after_pass_at_threshold"].astype(bool)), "pass_change"] = "loss"
    merged["exact_change"] = "same"
    merged.loc[(~merged["before_perfect_fit"].astype(bool)) & merged["after_perfect_fit"].astype(bool), "exact_change"] = "gain"
    merged.loc[merged["before_perfect_fit"].astype(bool) & (~merged["after_perfect_fit"].astype(bool)), "exact_change"] = "loss"
    merged["complexity_delta"] = merged["after_expr_complexity"].astype(float) - merged["before_expr_complexity"].astype(float)
    return merged


def summarize_pairwise(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for benchmark, sub in list(pairwise.groupby("benchmark", sort=False)) + [("overall", pairwise)]:
        n = int(len(sub))
        rows.append(
            {
                "benchmark": benchmark,
                "paired_n": n,
                "after_mse_win": int((sub["mse_win"] == "after").sum()),
                "before_mse_win": int((sub["mse_win"] == "before").sum()),
                "mse_tie": int((sub["mse_win"] == "tie").sum()),
                "pass_gain": int((sub["pass_change"] == "gain").sum()),
                "pass_loss": int((sub["pass_change"] == "loss").sum()),
                "pass_same": int((sub["pass_change"] == "same").sum()),
                "exact_gain": int((sub["exact_change"] == "gain").sum()),
                "exact_loss": int((sub["exact_change"] == "loss").sum()),
                "mean_log_mse_delta": mean_or_nan(sub["log_mse_delta"]),
                "median_log_mse_delta": median_or_nan(sub["log_mse_delta"]),
                "mean_complexity_delta": mean_or_nan(sub["complexity_delta"]),
                "median_complexity_delta": median_or_nan(sub["complexity_delta"]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, pair_summary: pd.DataFrame, out_path: Path, before_label: str, after_label: str) -> None:
    def pct(x):
        if pd.isna(x):
            return ""
        return f"{100.0 * float(x):.1f}%"

    overall = summary[summary["benchmark"] == "overall"].copy()
    lines = [
        "# V11 SFT Before/After Comparison",
        "",
        f"- before: `{before_label}`",
        f"- after: `{after_label}`",
        "",
        "## Overall",
        "",
        "| method | PASS@threshold | PASS@100 | PASS@300 | Exact | R2-perfect | median MSE@PASS | mean runtime | median complexity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in overall.iterrows():
        lines.append(
            "| {method} | {pass_n}/{n} ({pass_rate}) | {p100_n}/{n} ({p100_rate}) | {p300_n}/{n} ({p300_rate}) | {exact} | {r2p} | {mse:.4g} | {runtime:.1f}s | {comp:.1f} |".format(
                method=row["method"],
                pass_n=int(row["pass_n"]),
                n=int(row["n"]),
                pass_rate=pct(row["pass_rate"]),
                p100_n=int(row["pass100_n"]),
                p100_rate=pct(row["pass100_rate"]),
                p300_n=int(row["pass300_n"]),
                p300_rate=pct(row["pass300_rate"]),
                exact=int(row["perfect_fit_n"]),
                r2p=int(row["r2_perfect_n"]),
                mse=float(row["median_mse_pass"]) if pd.notna(row["median_mse_pass"]) else math.nan,
                runtime=float(row["mean_runtime_sec"]) if pd.notna(row["mean_runtime_sec"]) else math.nan,
                comp=float(row["median_complexity"]) if pd.notna(row["median_complexity"]) else math.nan,
            )
        )
    lines.extend(["", "## Paired Win/Tie/Loss", ""])
    lines.extend(
        [
            "| benchmark | after MSE wins | before MSE wins | ties | pass gains | pass losses | exact gains | exact losses | median log10 MSE delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in pair_summary.iterrows():
        lines.append(
            "| {benchmark} | {aw} | {bw} | {tie} | {pg} | {pl} | {eg} | {el} | {delta:.3f} |".format(
                benchmark=row["benchmark"],
                aw=int(row["after_mse_win"]),
                bw=int(row["before_mse_win"]),
                tie=int(row["mse_tie"]),
                pg=int(row["pass_gain"]),
                pl=int(row["pass_loss"]),
                eg=int(row["exact_gain"]),
                el=int(row["exact_loss"]),
                delta=float(row["median_log_mse_delta"]) if pd.notna(row["median_log_mse_delta"]) else math.nan,
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(summary: pd.DataFrame, pairwise: pd.DataFrame, out_dir: Path, before_label: str, after_label: str) -> None:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_benchmarks = [b for b in BENCHMARKS if b in set(summary["benchmark"])]
    x = np.arange(len(plot_benchmarks))
    width = 0.36
    before = summary[(summary["method"] == before_label) & summary["benchmark"].isin(plot_benchmarks)].set_index("benchmark").reindex(plot_benchmarks)
    after = summary[(summary["method"] == after_label) & summary["benchmark"].isin(plot_benchmarks)].set_index("benchmark").reindex(plot_benchmarks)

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - width / 2, before["pass100_rate"] * 100.0, width, label=before_label, color="#8c8c8c")
    ax.bar(x + width / 2, after["pass100_rate"] * 100.0, width, label=after_label, color="#1f5aa6")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_benchmarks, rotation=20, ha="right")
    ax.set_ylabel("PASS@100 (%)")
    ax.set_title("End-to-end V11 recovery before/after SFT")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "pass100_before_after.png", dpi=300)
    fig.savefig(out_dir / "pass100_before_after.pdf")
    plt.close(fig)

    finite = pairwise[np.isfinite(pairwise["before_best_test_mse"].astype(float)) & np.isfinite(pairwise["after_best_test_mse"].astype(float))].copy()
    if not finite.empty:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        colors = finite["benchmark"].map({"sldbench": "#4c78a8", "llmsrbench": "#f58518", "srsd": "#54a24b", "srbench": "#b279a2"}).fillna("#777777")
        ax.scatter(
            np.clip(finite["before_best_test_mse"].astype(float), 1.0e-12, 1.0e8),
            np.clip(finite["after_best_test_mse"].astype(float), 1.0e-12, 1.0e8),
            s=13,
            alpha=0.55,
            color=colors,
            linewidth=0,
        )
        lo, hi = 1.0e-12, 1.0e8
        ax.plot([lo, hi], [lo, hi], color="#222222", lw=0.9, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{before_label} test MSE")
        ax.set_ylabel(f"{after_label} test MSE")
        ax.set_title("Case-level MSE shift")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.22)
        fig.tight_layout()
        fig.savefig(out_dir / "case_mse_scatter.png", dpi=300)
        fig.savefig(out_dir / "case_mse_scatter.pdf")
        plt.close(fig)

    pair_summary = summarize_pairwise(pairwise)
    ps = pair_summary[pair_summary["benchmark"].isin(plot_benchmarks)].set_index("benchmark").reindex(plot_benchmarks)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bottom = np.zeros(len(plot_benchmarks))
    for col, color, label in [
        ("after_mse_win", "#1f5aa6", "after wins"),
        ("mse_tie", "#d9d9d9", "tie"),
        ("before_mse_win", "#9e3d22", "before wins"),
    ]:
        vals = ps[col].fillna(0).to_numpy()
        ax.bar(x, vals, bottom=bottom, color=color, label=label)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(plot_benchmarks, rotation=20, ha="right")
    ax.set_ylabel("paired cases")
    ax.set_title("MSE win/tie/loss by benchmark")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "win_tie_loss.png", dpi=300)
    fig.savefig(out_dir / "win_tie_loss.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-root", required=True, type=Path)
    parser.add_argument("--after-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--before-label", default="before_sft")
    parser.add_argument("--after-label", default="after_sft")
    parser.add_argument("--pass-threshold", type=float, default=100.0)
    parser.add_argument("--mse-win-rel-tol", type=float, default=0.01)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    before = load_run(args.before_root, args.before_label, args.pass_threshold)
    after = load_run(args.after_root, args.after_label, args.pass_threshold)
    all_rows = pd.concat([before, after], ignore_index=True)
    summary = build_summary(all_rows, args.pass_threshold)
    pairwise = build_pairwise(before, after, args.before_label, args.after_label, args.mse_win_rel_tol)
    pair_summary = summarize_pairwise(pairwise)

    all_rows.to_csv(args.out_dir / "case_metrics_long.csv", index=False)
    summary.to_csv(args.out_dir / "summary_by_benchmark.csv", index=False)
    pairwise.to_csv(args.out_dir / "paired_case_comparison.csv", index=False)
    pair_summary.to_csv(args.out_dir / "paired_summary_by_benchmark.csv", index=False)
    write_markdown(summary, pair_summary, args.out_dir / "comparison_report.md", args.before_label, args.after_label)
    if not args.no_plots:
        make_plots(summary, pairwise, args.out_dir / "figures", args.before_label, args.after_label)

    manifest = {
        "before_root": str(args.before_root),
        "after_root": str(args.after_root),
        "before_label": args.before_label,
        "after_label": args.after_label,
        "pass_threshold": args.pass_threshold,
        "mse_win_rel_tol": args.mse_win_rel_tol,
        "outputs": {
            "case_metrics_long": str(args.out_dir / "case_metrics_long.csv"),
            "summary_by_benchmark": str(args.out_dir / "summary_by_benchmark.csv"),
            "paired_case_comparison": str(args.out_dir / "paired_case_comparison.csv"),
            "paired_summary_by_benchmark": str(args.out_dir / "paired_summary_by_benchmark.csv"),
            "comparison_report": str(args.out_dir / "comparison_report.md"),
        },
        "missing_benchmarks": {
            args.before_label: before.attrs.get("missing_benchmarks", []),
            args.after_label: after.attrs.get("missing_benchmarks", []),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"summary={args.out_dir / 'summary_by_benchmark.csv'}")
    print(f"paired={args.out_dir / 'paired_summary_by_benchmark.csv'}")
    print(f"report={args.out_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
