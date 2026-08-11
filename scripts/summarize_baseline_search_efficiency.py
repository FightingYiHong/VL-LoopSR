#!/usr/bin/env python3
"""Build live cross-method search-efficiency tables and figures."""

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

from tools.plot_style import METHOD_COLORS, save_nature_figure, set_nature_style


METHODS = ("Ours", "LLM-SR", "DSO", "gplearn", "PSE", "PySR", "ICSR")
METHOD_DIRS = {
    "LLM-SR": "llmsr",
    "DSO": "dso",
    "gplearn": "gplearn",
    "PSE": "psrn",
    "PySR": "pysr",
    "ICSR": "icsr",
}
BENCHMARKS = ("llmsrbench", "sldbench", "srbench", "srsd")
EXPECTED_PER_METHOD = 972


def finite(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_baseline_rows(root: Path) -> pd.DataFrame:
    rows = []
    for method, method_dir in METHOD_DIRS.items():
        for benchmark in BENCHMARKS:
            for path in sorted((root / method_dir / benchmark / "case_results").glob("*.json")):
                try:
                    result = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                hit_count = finite(result.get("evaluations_to_validation_success"))
                observed = finite(result.get("validation_search_observed_evaluations"))
                success = bool(result.get("validation_search_success")) and hit_count is not None
                rows.append({
                    "method": method,
                    "benchmark": benchmark,
                    "case_name": result.get("case_name") or result.get("base_name") or path.stem,
                    "success": success,
                    "event_count": int(hit_count) if success else np.nan,
                    "observed_count": int(max(0, observed or 0)),
                    "unique_observed_count": finite(result.get("validation_search_unique_evaluations")),
                    "native_event_count": finite(result.get("native_evaluations_to_validation_success")),
                    "first_hit_elapsed_sec": finite(result.get("first_validation_success_elapsed_sec")),
                    "configured_wall_budget_sec": finite(
                        result.get("configured_wall_budget_sec", result.get("case_timeout_sec"))
                    ),
                    "runtime_sec": finite(result.get("runtime_sec")),
                    "timed_out": bool(result.get("timed_out")) or "timeout" in str(result.get("error", "")).lower(),
                    "source_json": str(path),
                })
    return pd.DataFrame(rows)


def load_ours_rows(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    raw = pd.read_csv(path)
    hit = raw["validation_search_success"].astype(str).str.lower().isin({"true", "1", "yes"})
    event = pd.to_numeric(raw["evaluations_to_validation_success"], errors="coerce")
    observed = pd.to_numeric(raw["validation_search_observed_evaluations"], errors="coerce").fillna(0)
    return pd.DataFrame({
        "method": "Ours",
        "benchmark": raw["benchmark"],
        "case_name": raw["case_name"],
        "success": hit & event.notna(),
        "event_count": event,
        "observed_count": observed.astype(int),
        "unique_observed_count": pd.to_numeric(
            raw.get("total_unique_candidate_evaluations"), errors="coerce"
        ),
        "native_event_count": event,
        "first_hit_elapsed_sec": np.nan,
        "configured_wall_budget_sec": pd.to_numeric(raw.get("time_budget_sec"), errors="coerce"),
        "runtime_sec": pd.to_numeric(raw.get("runtime_sec"), errors="coerce"),
        "timed_out": raw.get("time_budget_hit", False),
        "source_json": raw.get("source_csv", ""),
    })


def kaplan_meier_success(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["time"] = np.where(work["success"], work["event_count"], work["observed_count"])
    work["time"] = pd.to_numeric(work["time"], errors="coerce").fillna(0).clip(lower=0)
    work = work[work["time"] > 0].copy()
    if work.empty:
        return pd.DataFrame()
    survival = 1.0
    rows = [{"evaluation_count": 0, "success_probability": 0.0, "at_risk": len(work)}]
    for count in sorted(work["time"].unique()):
        at_risk = int((work["time"] >= count).sum())
        events = int(((work["time"] == count) & work["success"]).sum())
        if at_risk and events:
            survival *= 1.0 - events / at_risk
        rows.append({
            "evaluation_count": int(count),
            "success_probability": float(1.0 - survival),
            "at_risk": at_risk,
        })
    return pd.DataFrame(rows)


def summarize(cases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        frame = cases[cases["method"] == method]
        if frame.empty:
            continue
        hits = pd.to_numeric(frame.loc[frame["success"], "event_count"], errors="coerce").dropna()
        rows.append({
            "method": method,
            "completed": int(len(frame)),
            "expected": EXPECTED_PER_METHOD,
            "successes": int(frame["success"].sum()),
            "success_rate": float(frame["success"].mean()),
            "censored": int((~frame["success"]).sum()),
            "median_evaluations_to_hit_successes_only": float(hits.median()) if len(hits) else np.nan,
            "q25_evaluations_to_hit_successes_only": float(hits.quantile(0.25)) if len(hits) else np.nan,
            "q75_evaluations_to_hit_successes_only": float(hits.quantile(0.75)) if len(hits) else np.nan,
            "median_observed_evaluations": float(frame["observed_count"].median()),
            "median_runtime_sec": float(pd.to_numeric(frame["runtime_sec"], errors="coerce").median()),
            "timeouts": int(frame["timed_out"].astype(bool).sum()),
        })
    return pd.DataFrame(rows)


def plot(cases: pd.DataFrame, summary: pd.DataFrame, output_dir: Path):
    import matplotlib.pyplot as plt

    set_nature_style(plt=plt)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1.0]})
    curve_rows = []
    for method in METHODS:
        frame = cases[cases["method"] == method]
        curve = kaplan_meier_success(frame)
        if curve.empty:
            continue
        curve["method"] = method
        curve_rows.append(curve)
        color = METHOD_COLORS.get(method.lower(), "#7E8B96")
        axes[0].step(
            curve["evaluation_count"].clip(lower=1),
            100 * curve["success_probability"],
            where="post",
            label=f"{method} (n={len(frame)})",
            color=color,
            linewidth=1.7 if method == "Ours" else 1.25,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Validation candidate evaluations")
    axes[0].set_ylabel("Estimated probability of reaching $R^2 > 0.999$ (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.35)
    axes[0].legend(loc="best", ncol=2)

    shown = summary[summary["method"].isin(METHODS)].copy()
    shown["order"] = shown["method"].map({name: idx for idx, name in enumerate(METHODS)})
    shown = shown.sort_values("order")
    y = np.arange(len(shown))
    colors = [METHOD_COLORS.get(str(name).lower(), "#7E8B96") for name in shown["method"]]
    axes[1].barh(y, 100 * shown["success_rate"], color=colors, alpha=0.82, height=0.64)
    axes[1].set_yticks(y, shown["method"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Observed validation success rate (%)")
    axes[1].grid(axis="x", alpha=0.35)
    for yi, (_, row) in zip(y, shown.iterrows()):
        axes[1].text(
            min(99, 100 * row["success_rate"] + 1.5),
            yi,
            f'{int(row["successes"])}/{int(row["completed"])}',
            va="center",
            fontsize=6.8,
        )
    for ax, panel in zip(axes, ("a", "b")):
        ax.text(-0.13, 1.04, panel, transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.tight_layout()
    png = output_dir / "fig_cross_method_search_efficiency_live.png"
    pdf = output_dir / "fig_cross_method_search_efficiency_live.pdf"
    save_nature_figure(fig, png, section="standard_recovery", dpi=300, bbox_inches="tight")
    save_nature_figure(fig, pdf, section="standard_recovery", bbox_inches="tight")
    plt.close(fig)
    curves = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    curves.to_csv(output_dir / "cross_method_search_efficiency_km.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--ours-case-rows", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_baseline_rows(args.baseline_root), load_ours_rows(args.ours_case_rows)]
    cases = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    cases.to_csv(args.output_dir / "cross_method_search_efficiency_cases.csv", index=False)
    summary = summarize(cases)
    summary.to_csv(args.output_dir / "cross_method_search_efficiency_summary.csv", index=False)
    if not cases.empty:
        plot(cases, summary, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
