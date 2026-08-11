#!/usr/bin/env python3
"""Create a case-wise, publication-ready standard-recovery figure.

The figure is derived from the manuscript's normalized standard-recovery rows.
Repeated rows that resolve to the same benchmark case are collapsed before any
summary is computed, so benchmark cases—not repeated runs—are the independent
statistical units.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "reports"
    / "existing_metric_results_20260724"
    / "00_paper_source"
    / "fig1_main_case_rows.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "figs" / "nature_subjournal" / "standard_recovery_casewise"
OUTPUT_STEM = "fig_standard_recovery_casewise"

TEXT = "#27313A"
AUXILIARY = "#E8EDF1"
GRID = "#D9DEE3"
WHITE = "#FFFFFF"

METHOD_ORDER = ["VEGA-SR", "LLM-SR", "DSO", "gplearn", "PSE", "PySR", "ICSR"]
SOURCE_METHOD_LABELS = {"Ours": "VEGA-SR"}
METHOD_COLORS = {
    "VEGA-SR": "#214F73",
    "LLM-SR": "#A8754F",
    "DSO": "#71806A",
    "gplearn": "#756C8F",
    "PSE": "#8A706A",
    "PySR": "#6F8FAF",
    "ICSR": "#7D8790",
}
METHOD_MARKERS = {method: "o" for method in METHOD_ORDER}

BENCHMARK_ORDER = ["llmsrbench", "srbench", "sldbench", "srsd"]
BENCHMARK_LABELS = {
    "llmsrbench": "LLMSRBench",
    "srbench": "SRBench",
    "sldbench": "SLDBench",
    "srsd": "SRSD",
}

BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 20260725


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
        "axes.titleweight": "normal",
        "axes.labelsize": 7.5,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.fontsize": 6.8,
        "legend.frameon": False,
        "text.color": TEXT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
    sns.set_theme(context="paper", style="white", rc=rc)
    mpl.rcParams.update(rc)


def canonical_case_name(value: object) -> str:
    """Resolve path-like and pipe-delimited names to one benchmark case key."""
    text = str(value).strip().replace("\\", "/")
    if "|" in text:
        text = text.split("|")[-1].strip()
    text = text.rsplit("/", maxsplit=1)[-1]
    text = re.sub(r"\.(pkl|csv|json)$", "", text, flags=re.IGNORECASE)
    return text.lower()


def finite_any(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(numeric).any())


def finite_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    return float(np.median(numeric)) if numeric.size else float("nan")


def prepare_case_table(source_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(source_path)
    required = {
        "method_label",
        "benchmark",
        "case_name",
        "passed",
        "test_mse",
        "runtime_sec",
        "expr_complexity",
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    raw = raw.copy()
    raw["method"] = raw["method_label"].replace(SOURCE_METHOD_LABELS)
    raw = raw[raw["method"].isin(METHOD_ORDER)].copy()
    raw["benchmark"] = raw["benchmark"].astype(str).str.lower()
    raw = raw[raw["benchmark"].isin(BENCHMARK_ORDER)].copy()
    raw["canonical_case_name"] = raw["case_name"].map(canonical_case_name)
    raw["case_id"] = raw["benchmark"] + "::" + raw["canonical_case_name"]

    passed_text = raw["passed"].astype(str).str.strip().str.lower()
    passed_map = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    raw["passed_numeric"] = passed_text.map(passed_map)
    if raw["passed_numeric"].isna().any():
        bad = raw.loc[raw["passed_numeric"].isna(), "passed"].drop_duplicates().tolist()
        raise ValueError(f"Unrecognized recovery labels: {bad}")

    case = (
        raw.groupby(["benchmark", "case_id", "canonical_case_name", "method"], sort=False)
        .agg(
            recovery_score=("passed_numeric", "mean"),
            repeated_rows=("passed_numeric", "size"),
            runtime_sec=("runtime_sec", finite_median),
            expr_complexity=("expr_complexity", finite_median),
            finite_numeric_result=("test_mse", finite_any),
        )
        .reset_index()
    )
    case["status"] = np.where(case["finite_numeric_result"], "finite result", "numerical failure")
    case["source_file"] = str(source_path)
    return case


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def panel_label(ax: mpl.axes.Axes, label: str, *, x: float = -0.16, y: float = 1.12) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )


def recovery_summary(
    case: pd.DataFrame,
    *,
    resamples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seed_offset = 0
    for benchmark in BENCHMARK_ORDER:
        bdf = case[case["benchmark"].eq(benchmark)]
        expected = int(bdf["case_id"].nunique())
        for method in METHOD_ORDER:
            values = bdf.loc[bdf["method"].eq(method), "recovery_score"].to_numpy(dtype=float)
            mean, low, high = bootstrap_mean_ci(
                values,
                resamples=resamples,
                seed=BOOTSTRAP_SEED + seed_offset,
            )
            seed_offset += 1
            rows.append(
                {
                    "panel": "a",
                    "record_type": "summary",
                    "benchmark": benchmark,
                    "method": method,
                    "measure": "recovery_rate_percent",
                    "value": 100.0 * mean,
                    "ci_low": 100.0 * low,
                    "ci_high": 100.0 * high,
                    "n_cases": int(np.isfinite(values).sum()),
                    "n_expected": expected,
                    "notes": "Mean case-level recovery score; percentile bootstrap 95% CI over cases.",
                }
            )
    return pd.DataFrame(rows)


def paired_recovery_summary(
    case: pd.DataFrame,
    *,
    resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    recovery = case.pivot(index="case_id", columns="method", values="recovery_score")
    benchmark_by_case = case.drop_duplicates("case_id").set_index("case_id")["benchmark"]

    for index, baseline in enumerate(METHOD_ORDER[1:]):
        paired = recovery[["VEGA-SR", baseline]].dropna()
        differences = 100.0 * (paired["VEGA-SR"] - paired[baseline])
        mean, low, high = bootstrap_mean_ci(
            differences.to_numpy(),
            resamples=resamples,
            seed=BOOTSTRAP_SEED + 100 + index,
        )
        for case_id, value in differences.items():
            case_rows.append(
                {
                    "panel": "b",
                    "record_type": "case_value",
                    "benchmark": benchmark_by_case.loc[case_id],
                    "method": baseline,
                    "reference_method": "VEGA-SR",
                    "case_id": case_id,
                    "measure": "paired_recovery_difference_pp",
                    "value": float(value),
                    "notes": "VEGA-SR minus baseline case-level recovery score, in percentage points.",
                }
            )
        summary_rows.append(
            {
                "panel": "b",
                "record_type": "summary",
                "method": baseline,
                "reference_method": "VEGA-SR",
                "measure": "paired_recovery_difference_pp",
                "value": mean,
                "ci_low": low,
                "ci_high": high,
                "n_cases": int(len(paired)),
                "notes": "Mean paired difference; percentile bootstrap 95% CI over matched cases.",
            }
        )
    return pd.DataFrame(case_rows), pd.DataFrame(summary_rows)


def runtime_summary(case: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        values = case.loc[case["method"].eq(method), "runtime_sec"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "panel": "c",
                "record_type": "summary",
                "method": method,
                "measure": "runtime_sec",
                "value": float(np.median(values)),
                "q1": float(np.quantile(values, 0.25)),
                "q3": float(np.quantile(values, 0.75)),
                "n_cases": int(values.size),
                "notes": "Median and interquartile range across benchmark cases.",
            }
        )
    return pd.DataFrame(rows)


def status_table(case: pd.DataFrame) -> pd.DataFrame:
    expected_by_benchmark = {
        benchmark: set(case.loc[case["benchmark"].eq(benchmark), "case_id"])
        for benchmark in BENCHMARK_ORDER
    }
    all_expected = set().union(*expected_by_benchmark.values())
    rows: list[dict[str, object]] = []

    for method in METHOD_ORDER:
        method_rows = case[case["method"].eq(method)].set_index("case_id")
        statuses: dict[str, int] = {"finite result": 0, "numerical failure": 0, "unavailable": 0}
        for case_id in all_expected:
            if case_id not in method_rows.index:
                statuses["unavailable"] += 1
            elif bool(method_rows.loc[case_id, "finite_numeric_result"]):
                statuses["finite result"] += 1
            else:
                statuses["numerical failure"] += 1
        for status, count in statuses.items():
            rows.append(
                {
                    "panel": "d",
                    "record_type": "summary",
                    "method": method,
                    "status": status,
                    "measure": "case_status_percent",
                    "value": 100.0 * count / len(all_expected),
                    "count": count,
                    "n_expected": len(all_expected),
                    "notes": (
                        "Unavailable means no row for the case; numerical failure means a row "
                        "exists but no finite test MSE is recorded."
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_panel_a(
    axes: list[mpl.axes.Axes],
    summary: pd.DataFrame,
) -> None:
    y = np.arange(len(METHOD_ORDER))
    for index, (ax, benchmark) in enumerate(zip(axes, BENCHMARK_ORDER)):
        sdf = summary[summary["benchmark"].eq(benchmark)].set_index("method").loc[METHOD_ORDER]
        expected = int(sdf["n_expected"].iloc[0])
        for yi, method in enumerate(METHOD_ORDER):
            row = sdf.loc[method]
            value = float(row["value"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            color = METHOD_COLORS[method]
            ax.plot([low, high], [yi, yi], color=color, lw=1.5, solid_capstyle="round", zorder=2)
            ax.scatter(
                value,
                yi,
                s=18,
                marker=METHOD_MARKERS[method],
                facecolor=color,
                edgecolor=WHITE,
                linewidth=0.45,
                zorder=3,
            )
            n_cases = int(row["n_cases"])
            if n_cases < expected:
                x_text = min(value + 4.0, 98.0)
                ha = "left" if value <= 94 else "right"
                ax.text(
                    x_text,
                    yi,
                    f"n={n_cases}",
                    ha=ha,
                    va="center",
                    fontsize=6.1,
                    color=TEXT,
                )
        ax.set_title(
            f"{BENCHMARK_LABELS[benchmark]}\n$n$={expected}",
            pad=3.0,
            fontsize=6.9,
        )
        ax.set_xlim(-2, 104)
        ax.set_xticks([0, 50, 100])
        ax.set_ylim(len(METHOD_ORDER) - 0.5, -0.5)
        ax.set_yticks(y)
        if index == 0:
            ax.set_yticklabels(METHOD_ORDER)
            panel_label(ax, "a", x=-0.48, y=1.08)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)
            ax.spines["left"].set_visible(False)


def plot_panel_b(ax: mpl.axes.Axes, summary: pd.DataFrame) -> None:
    baselines = METHOD_ORDER[1:]
    sdf = summary.set_index("method").loc[baselines]
    y = np.arange(len(baselines))
    for yi, method in enumerate(baselines):
        row = sdf.loc[method]
        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [yi, yi],
            color=METHOD_COLORS[method],
            lw=1.7,
            solid_capstyle="round",
        )
        ax.scatter(
            row["value"],
            yi,
            s=21,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor=WHITE,
            linewidth=0.45,
            zorder=3,
        )
        ax.text(
            0.985,
            yi,
            f"n={int(row['n_cases'])}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.2,
            color=TEXT,
            bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.8, "alpha": 0.9},
        )
    ax.axvline(0, color=GRID, lw=0.9, zorder=0)
    ax.set_yticks(y, baselines)
    ax.set_ylim(len(baselines) - 0.5, -0.5)
    all_ci = np.r_[sdf["ci_low"].to_numpy(dtype=float), sdf["ci_high"].to_numpy(dtype=float)]
    left = min(-7.0, np.floor(np.nanmin(all_ci) / 5.0) * 5.0 - 2.0)
    right = max(12.0, np.ceil(np.nanmax(all_ci) / 5.0) * 5.0 + 18.0)
    ax.set_xlim(left, right)
    ax.set_xlabel("Paired recovery difference (pp)")
    ax.set_title("VEGA-SR relative to each baseline", pad=5.0)
    ax.text(
        0.02,
        -0.18,
        "baseline favoured",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=TEXT,
    )
    ax.text(
        0.98,
        -0.18,
        "VEGA-SR favoured",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        color=TEXT,
    )
    panel_label(ax, "b", x=-0.32, y=1.08)


def plot_panel_c(
    ax: mpl.axes.Axes,
    case: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    y_positions = np.arange(len(METHOD_ORDER))
    for yi, method in enumerate(METHOD_ORDER):
        values = case.loc[case["method"].eq(method), "runtime_sec"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        jitter = rng.uniform(-0.19, 0.19, size=values.size)
        ax.scatter(
            values,
            yi + jitter,
            s=3.6,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            alpha=0.13,
            linewidths=0,
            rasterized=False,
            zorder=1,
        )

    box_values = [
        case.loc[case["method"].eq(method), "runtime_sec"].dropna().to_numpy(dtype=float)
        for method in METHOD_ORDER
    ]
    boxes = ax.boxplot(
        box_values,
        positions=y_positions,
        vert=False,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
        boxprops={"facecolor": AUXILIARY, "edgecolor": TEXT, "linewidth": 0.8},
        whiskerprops={"color": TEXT, "linewidth": 0.75},
        capprops={"color": TEXT, "linewidth": 0.75},
        medianprops={"color": TEXT, "linewidth": 1.5},
        zorder=2,
    )
    for box, median, method in zip(boxes["boxes"], boxes["medians"], METHOD_ORDER):
        box.set_edgecolor(METHOD_COLORS[method])
        median.set_color(METHOD_COLORS[method])

    sdf = summary.set_index("method").loc[METHOD_ORDER]
    for yi, method in enumerate(METHOD_ORDER):
        ax.text(
            0.985,
            yi,
            f"n={int(sdf.loc[method, 'n_cases'])}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=6.1,
            color=TEXT,
            bbox={"facecolor": WHITE, "edgecolor": "none", "pad": 0.8, "alpha": 0.86},
        )
    ax.set_xscale("log")
    ax.set_yticks(y_positions, METHOD_ORDER)
    ax.set_ylim(len(METHOD_ORDER) - 0.5, -0.5)
    ax.set_xlabel("Observed runtime per case (s)")
    ax.set_title("Runtime by case (median and IQR)", pad=5.0)
    panel_label(ax, "c", x=-0.20, y=1.08)


def plot_panel_d(ax: mpl.axes.Axes, status: pd.DataFrame) -> None:
    y = np.arange(len(METHOD_ORDER))
    left = np.zeros(len(METHOD_ORDER), dtype=float)
    status_order = ["finite result", "numerical failure", "unavailable"]
    for result_status in status_order:
        values = np.array(
            [
                status.loc[
                    status["method"].eq(method) & status["status"].eq(result_status),
                    "value",
                ].iloc[0]
                for method in METHOD_ORDER
            ],
            dtype=float,
        )
        if result_status == "finite result":
            for yi, method in enumerate(METHOD_ORDER):
                ax.barh(
                    yi,
                    values[yi],
                    left=left[yi],
                    height=0.54,
                    color=METHOD_COLORS[method],
                    edgecolor=WHITE,
                    linewidth=0.45,
                )
        elif result_status == "numerical failure":
            ax.barh(
                y,
                values,
                left=left,
                height=0.54,
                color=AUXILIARY,
                edgecolor=TEXT,
                linewidth=0.45,
                hatch="////",
            )
        else:
            ax.barh(
                y,
                values,
                left=left,
                height=0.54,
                color=WHITE,
                edgecolor=TEXT,
                linewidth=0.55,
                hatch="..",
            )
        left += values

    for yi, method in enumerate(METHOD_ORDER):
        mdf = status[status["method"].eq(method)].set_index("status")
        failure_count = int(mdf.loc["numerical failure", "count"])
        unavailable_count = int(mdf.loc["unavailable", "count"])
        finite_pct = float(mdf.loc["finite result", "value"])
        failure_pct = float(mdf.loc["numerical failure", "value"])
        if failure_count:
            ax.text(
                finite_pct + failure_pct / 2.0,
                yi,
                str(failure_count),
                ha="center",
                va="center",
                fontsize=5.9,
                color=TEXT,
            )
        if unavailable_count:
            ax.text(
                finite_pct + failure_pct + float(mdf.loc["unavailable", "value"]) / 2.0,
                yi,
                str(unavailable_count),
                ha="center",
                va="center",
                fontsize=5.9,
                color=TEXT,
            )

    expected = int(status["n_expected"].iloc[0])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Share of benchmark cases (%)")
    ax.set_yticks(y, METHOD_ORDER)
    ax.set_ylim(len(METHOD_ORDER) - 0.5, -0.5)
    ax.set_title(f"Result status ($n$={expected} cases)", pad=5.0)
    handles = [
        Patch(facecolor="#6F8FAF", edgecolor=WHITE, label="finite result"),
        Patch(facecolor=AUXILIARY, edgecolor=TEXT, hatch="////", label="numerical failure"),
        Patch(facecolor=WHITE, edgecolor=TEXT, hatch="..", label="unavailable"),
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        fontsize=6.2,
        handlelength=1.1,
        columnspacing=0.8,
        handletextpad=0.35,
        borderaxespad=0,
    )
    panel_label(ax, "d", x=-0.32, y=1.08)


def case_source_rows(case: pd.DataFrame) -> pd.DataFrame:
    common = case[
        [
            "benchmark",
            "method",
            "case_id",
            "canonical_case_name",
            "repeated_rows",
            "status",
            "source_file",
        ]
    ].copy()

    recovery = common.copy()
    recovery["panel"] = "a"
    recovery["record_type"] = "case_value"
    recovery["measure"] = "recovery_score"
    recovery["value"] = case["recovery_score"]
    recovery["notes"] = "Mean pass indicator across repeated rows for this benchmark case."

    runtime = common.copy()
    runtime["panel"] = "c"
    runtime["record_type"] = "case_value"
    runtime["measure"] = "runtime_sec"
    runtime["value"] = case["runtime_sec"]
    runtime["notes"] = "Median runtime across repeated rows for this benchmark case."
    return pd.concat([recovery, runtime], ignore_index=True)


def save_source_data(
    output_path: Path,
    *,
    case: pd.DataFrame,
    recovery: pd.DataFrame,
    paired_cases: pd.DataFrame,
    paired_summary: pd.DataFrame,
    runtime: pd.DataFrame,
    status: pd.DataFrame,
) -> None:
    frames = [
        case_source_rows(case),
        recovery,
        paired_cases,
        paired_summary,
        runtime,
        status,
    ]
    source = pd.concat(frames, ignore_index=True, sort=False)
    preferred = [
        "panel",
        "record_type",
        "benchmark",
        "method",
        "reference_method",
        "case_id",
        "canonical_case_name",
        "status",
        "measure",
        "value",
        "ci_low",
        "ci_high",
        "q1",
        "q3",
        "count",
        "n_cases",
        "n_expected",
        "repeated_rows",
        "source_file",
        "notes",
    ]
    ordered = preferred + [column for column in source.columns if column not in preferred]
    source[ordered].to_csv(output_path, index=False)


def build_figure(
    case: pd.DataFrame,
    recovery: pd.DataFrame,
    paired: pd.DataFrame,
    runtime: pd.DataFrame,
    status: pd.DataFrame,
) -> mpl.figure.Figure:
    fig = plt.figure(figsize=(7.2, 6.4))
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.47, 1.0],
        height_ratios=[1.0, 1.05],
        left=0.11,
        right=0.985,
        top=0.865,
        bottom=0.15,
        wspace=0.36,
        hspace=0.50,
    )
    panel_a_grid = outer[0, 0].subgridspec(1, 4, wspace=0.16)
    axes_a = [fig.add_subplot(panel_a_grid[0, index]) for index in range(4)]
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])

    plot_panel_a(axes_a, recovery)
    plot_panel_b(ax_b, paired)
    plot_panel_c(ax_c, case, runtime)
    plot_panel_d(ax_d, status)

    first_a = axes_a[0].get_position()
    last_a = axes_a[-1].get_position()
    fig.text(
        (first_a.x0 + last_a.x1) / 2.0,
        first_a.y0 - 0.052,
        "Recovery (%)",
        ha="center",
        va="top",
        fontsize=7.5,
        color=TEXT,
    )
    fig.suptitle(
        "Standard formula recovery across benchmark cases",
        x=0.11,
        y=0.975,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=TEXT,
    )
    return fig


def save_figure(fig: mpl.figure.Figure, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "facecolor": WHITE,
        "edgecolor": WHITE,
        "bbox_inches": None,
    }
    fig.savefig(output_dir / f"{OUTPUT_STEM}.svg", format="svg", **common)
    fig.savefig(output_dir / f"{OUTPUT_STEM}.pdf", format="pdf", **common)
    tiff_path = output_dir / f"{OUTPUT_STEM}.tiff"
    png_path = output_dir / f"{OUTPUT_STEM}.png"
    fig.savefig(
        tiff_path,
        format="tiff",
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
        **common,
    )
    fig.savefig(png_path, format="png", dpi=300, **common)

    # Matplotlib emits RGBA rasters; flatten onto the specified white canvas so
    # the submission TIFF and preview PNG are conventional RGB files.
    for path, dpi, save_kwargs in [
        (tiff_path, (600, 600), {"compression": "tiff_lzw"}),
        (png_path, (300, 300), {}),
    ]:
        with Image.open(path) as image:
            rgb = Image.new("RGB", image.size, WHITE)
            if image.mode == "RGBA":
                rgb.paste(image, mask=image.getchannel("A"))
            else:
                rgb.paste(image.convert("RGB"))
            rgb.save(path, dpi=dpi, **save_kwargs)


def main() -> None:
    args = parse_args()
    set_style()
    case = prepare_case_table(args.input)
    recovery = recovery_summary(case, resamples=args.bootstrap_resamples)
    paired_cases, paired = paired_recovery_summary(case, resamples=args.bootstrap_resamples)
    runtime = runtime_summary(case)
    status = status_table(case)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_source_data(
        args.output_dir / f"{OUTPUT_STEM}_source_data.csv",
        case=case,
        recovery=recovery,
        paired_cases=paired_cases,
        paired_summary=paired,
        runtime=runtime,
        status=status,
    )
    fig = build_figure(case, recovery, paired, runtime, status)
    save_figure(fig, args.output_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
