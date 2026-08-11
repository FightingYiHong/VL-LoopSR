#!/usr/bin/env python3
"""Compare standard-recovery methods across test-R2 success thresholds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_standard_recovery_r2 import case_key, derive_r2
from tools.plot_style import METHOD_COLORS, NATURE_COLORS, save_nature_figure, set_nature_style


METHOD_ORDER = ["Ours", "DSO", "gplearn", "PSE", "PySR", "ICSR"]
METHOD_COLOR = {
    "Ours": METHOD_COLORS["ours"],
    "DSO": METHOD_COLORS["dso"],
    "gplearn": METHOD_COLORS["gplearn"],
    "PSE": METHOD_COLORS["pse"],
    "PySR": METHOD_COLORS["pysr"],
    "ICSR": METHOD_COLORS["icsr"],
}
EXPECTED_N = 972


def load_ours(paper_rows_path: Path, variance_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(paper_rows_path)
    frame = frame[frame["method"] == "ours_v11_pass300"].copy()
    frame["method"] = "Ours"
    frame["benchmark"] = frame["benchmark"].astype(str).str.lower()
    frame["case_key"] = [
        case_key(benchmark, case_name, dataset_group)
        for benchmark, case_name, dataset_group in zip(
            frame["benchmark"], frame["case_name"], frame["dataset_group"]
        )
    ]
    frame["test_mse"] = pd.to_numeric(frame["test_mse"], errors="coerce")
    return derive_r2(frame, pd.read_csv(variance_path))


def threshold_sweep(case_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 1001),
        np.array([0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]),
    ]))
    sweep_rows = []
    summary_rows = []
    for method in METHOD_ORDER:
        method_rows = case_rows.loc[case_rows["method"] == method]
        values = pd.to_numeric(method_rows["test_r2"], errors="coerce").to_numpy(dtype=float)
        complexity = pd.to_numeric(
            method_rows["expr_complexity"], errors="coerce"
        ).to_numpy(dtype=float)
        finite = np.isfinite(values)
        clipped = np.where(finite, np.clip(values, 0.0, 1.0), 0.0)
        for threshold in thresholds:
            success_n = int(np.sum(finite & (values > threshold)))
            sweep_rows.append({
                "method": method,
                "threshold": float(threshold),
                "success_n": success_n,
                "success_rate_expected": success_n / EXPECTED_N,
            })
        summary_rows.append({
            "method": method,
            "expected_n": EXPECTED_N,
            "finite_r2_n": int(finite.sum()),
            "r2_gt_0_8_n": int(np.sum(finite & (values > 0.8))),
            "r2_gt_0_8_rate_expected": float(np.sum(finite & (values > 0.8)) / EXPECTED_N),
            "r2_gt_0_99_n": int(np.sum(finite & (values > 0.99))),
            "r2_gt_0_99_rate_expected": float(
                np.sum(finite & (values > 0.99)) / EXPECTED_N
            ),
            "r2_gt_0_999_n": int(np.sum(finite & (values > 0.999))),
            "r2_gt_0_999_rate_expected": float(
                np.sum(finite & (values > 0.999)) / EXPECTED_N
            ),
            "median_complexity_r2_gt_0_99": float(np.nanmedian(np.where(
                finite & (values > 0.99) & np.isfinite(complexity),
                complexity,
                np.nan,
            ))),
            "median_complexity_r2_gt_0_999": float(np.nanmedian(np.where(
                finite & (values > 0.999) & np.isfinite(complexity),
                complexity,
                np.nan,
            ))),
            "success_curve_auc_0_1": float(np.sum(clipped) / EXPECTED_N),
        })
    return pd.DataFrame(sweep_rows), pd.DataFrame(summary_rows)


def plot_thresholds(sweep: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    set_nature_style(plt=plt)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.0),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )

    for method in METHOD_ORDER:
        part = sweep[sweep["method"] == method]
        axes[0].plot(
            part["threshold"],
            100.0 * part["success_rate_expected"],
            color=METHOD_COLOR[method],
            linewidth=2.0 if method == "Ours" else 1.1,
            label=method,
            zorder=5 if method == "Ours" else 2,
        )
    axes[0].axvline(0.8, color=NATURE_COLORS["midgray"], linestyle="--", linewidth=0.8)
    axes[0].axvline(0.999, color=NATURE_COLORS["midgray"], linestyle=":", linewidth=0.8)
    axes[0].text(0.8, 1.5, "0.8", ha="right", va="bottom", fontsize=6.5)
    axes[0].text(0.999, 1.5, "0.999", ha="right", va="bottom", fontsize=6.5)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 80)
    axes[0].set_xlabel("Test $R^2$ success threshold")
    axes[0].set_ylabel("Share of 972 tasks above threshold (%)")
    axes[0].set_title("Threshold-sensitivity profile")
    axes[0].grid(axis="y", alpha=0.35)
    axes[0].legend(ncol=2, loc="upper right")

    ordered = summary.set_index("method").loc[METHOD_ORDER]
    x = np.arange(len(METHOD_ORDER))
    axes[1].bar(
        x,
        ordered["success_curve_auc_0_1"],
        color=[METHOD_COLOR[method] for method in METHOD_ORDER],
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_xticks(x, METHOD_ORDER, rotation=25, ha="right")
    axes[1].set_ylim(0, 0.6)
    axes[1].set_ylabel("Success-curve AUC (threshold 0 to 1)")
    axes[1].set_title("Threshold-integrated performance")
    axes[1].grid(axis="y", alpha=0.35)

    for ax, panel in zip(axes, ("a", "b")):
        ax.text(-0.13, 1.05, panel, transform=ax.transAxes, fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_nature_figure(
        fig,
        output_dir / "fig_standard_recovery_r2_threshold_sweep.png",
        section="standard_recovery",
        dpi=300,
        bbox_inches="tight",
    )
    save_nature_figure(
        fig,
        output_dir / "fig_standard_recovery_r2_threshold_sweep.pdf",
        section="standard_recovery",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--paper-case-rows",
        type=Path,
        default=ROOT / "dxevolve-nature-latex" / "supplement_submission_assets_3"
        / "source_data" / "fig1" / "fig1_case_rows.csv",
    )
    parser.add_argument("--baseline-case-rows", type=Path, required=True)
    parser.add_argument("--variance-cache", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ours = load_ours(args.paper_case_rows, args.variance_cache)
    baselines = pd.read_csv(args.baseline_case_rows)
    case_rows = pd.concat([ours, baselines], ignore_index=True, sort=False)
    sweep, summary = threshold_sweep(case_rows)
    sweep.to_csv(output_dir / "standard_recovery_r2_threshold_sweep.csv", index=False)
    summary.to_csv(output_dir / "standard_recovery_r2_threshold_summary.csv", index=False)
    plot_thresholds(sweep, summary, output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
