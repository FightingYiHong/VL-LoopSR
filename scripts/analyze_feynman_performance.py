#!/usr/bin/env python3
"""Audit VEGA-SR performance on every Feynman-derived benchmark slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STANDARD_ROOT = (
    ROOT
    / "runs/paper_experiments/01_standard_recovery"
    / "standard_search_efficiency_full_20260724_152138"
)
BASELINE_ROOT = (
    ROOT
    / "runs/paper_experiments/01_standard_recovery"
    / "baseline_search_efficiency_full_affinity_20260726_174347"
)
ARCHIVE_ROOT = (
    ROOT
    / "reports/posthoc_recompute_inputs_20260724/01_main/results/archive/ours_paper_run"
)
RESCUE_ROWS = (
    ROOT
    / "runs/paper_experiments/01_standard_recovery/srsd_expression_rescue"
    / "run_20260725_225847/srsd_expression_rescue_case_rows.csv"
)
OTHER_R2_ROWS = (
    STANDARD_ROOT / "other_methods_r2/standard_recovery_other_methods_r2_case_rows.csv"
)
OTHER_R2_SUMMARY = (
    STANDARD_ROOT / "other_methods_r2/standard_recovery_other_methods_r2_summary.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "runs/paper_experiments/01_standard_recovery"
    / "feynman_performance_analysis_20260802"
)

METHOD_DIRS = {
    "LLM-SR": "llmsr",
    "DSO": "dso",
    "gplearn": "gplearn",
    "PSE": "psrn",
    "PySR": "pysr",
    "ICSR": "icsr",
}
SUBSETS = {
    "SRBench-Feynman": ("srbench", "feynman", 119),
    "SRSD-original": ("srsd", "original", 120),
    "SRSD-dummy": ("srsd", "dummy", 118),
    "LLMSR-Feynman-transform": ("llmsrbench", "lsr_transform", 111),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.999)
    return parser.parse_args()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def bool_sum(values: pd.Series) -> int:
    return int(values.map(as_bool).sum())


def summarize_r2(
    *,
    evidence: str,
    subset: str,
    method: str,
    values: pd.Series,
    expected_n: int,
    threshold: float,
) -> dict[str, object]:
    scores = pd.to_numeric(values, errors="coerce")
    hits = scores.gt(threshold)
    return {
        "evidence": evidence,
        "subset": subset,
        "method": method,
        "expected_n": expected_n,
        "finite_r2_n": int(scores.notna().sum()),
        "r2_gt_threshold_n": int(hits.sum()),
        "r2_gt_threshold_rate": float(hits.sum() / expected_n),
        "threshold": threshold,
        "available_for_comparison": bool(scores.notna().any()),
    }


def paper_aligned_rows(threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    ours_srbench = pd.read_csv(ARCHIVE_ROOT / "srbench.csv", low_memory=False)
    rows.append(
        summarize_r2(
            evidence="paper_archive_reported_test_r2",
            subset="SRBench-Feynman",
            method="Ours",
            values=ours_srbench.loc[
                ours_srbench["srbench_group_name"].eq("feynman"), "test_r2"
            ],
            expected_n=119,
            threshold=threshold,
        )
    )

    ours_llmsr = pd.read_csv(ARCHIVE_ROOT / "llmsrbench.csv", low_memory=False)
    rows.append(
        summarize_r2(
            evidence="paper_archive_reported_test_r2",
            subset="LLMSR-Feynman-transform",
            method="Ours",
            values=ours_llmsr.loc[
                ours_llmsr["llmsrbench_split_name"].eq("lsr_transform"), "test_r2"
            ],
            expected_n=111,
            threshold=threshold,
        )
    )

    other = pd.read_csv(OTHER_R2_ROWS, low_memory=False)
    group_specs = {
        "SRBench-Feynman": ("srbench", ["feynman"], 119),
        "LLMSR-Feynman-transform": ("llmsrbench", ["lsr_transform"], 111),
    }
    for subset, (benchmark, groups, expected_n) in group_specs.items():
        for method in ["DSO", "gplearn", "PSE", "PySR", "ICSR"]:
            selected = other[
                other["benchmark"].eq(benchmark)
                & other["method"].eq(method)
                & other["dataset_group"].isin(groups)
            ]
            if selected.empty:
                continue
            rows.append(
                summarize_r2(
                    evidence="paper_baseline_posthoc_r2",
                    subset=subset,
                    method=method,
                    values=selected["test_r2"],
                    expected_n=expected_n,
                    threshold=threshold,
                )
            )

    rescue = pd.read_csv(RESCUE_ROWS, low_memory=False)
    original_scores = 1.0 - pd.to_numeric(rescue["original_test_nmse"], errors="coerce")
    rescued_scores = 1.0 - pd.to_numeric(rescue["selected_test_nmse"], errors="coerce")
    rows.extend(
        [
            summarize_r2(
                evidence="canonical_srsd_expression_recompute",
                subset="SRSD-all",
                method="Ours (raw)",
                values=original_scores,
                expected_n=238,
                threshold=threshold,
            ),
            summarize_r2(
                evidence="canonical_srsd_validation_only_rescue",
                subset="SRSD-all",
                method="Ours + rescue",
                values=rescued_scores,
                expected_n=238,
                threshold=threshold,
            ),
        ]
    )

    summary = pd.read_csv(OTHER_R2_SUMMARY)
    for method in ["DSO", "gplearn", "PSE", "PySR", "ICSR"]:
        selected = summary[
            summary["benchmark"].eq("srsd") & summary["method"].eq(method)
        ]
        if selected.empty:
            continue
        result = selected.iloc[0]
        rows.append(
            {
                "evidence": "paper_baseline_posthoc_r2",
                "subset": "SRSD-all",
                "method": method,
                "expected_n": 238,
                "finite_r2_n": int(result["finite_r2_n"]),
                "r2_gt_threshold_n": int(result["r2_gt_0_999_n"]),
                "r2_gt_threshold_rate": float(result["r2_gt_0_999_rate_expected"]),
                "threshold": threshold,
                "available_for_comparison": bool(result["finite_r2_n"] > 0),
            }
        )
    return pd.DataFrame(rows)


def subset_mask(frame: pd.DataFrame, benchmark: str, subset_key: str) -> pd.Series:
    if benchmark == "srbench":
        return frame["difficulty"].eq("feynman")
    if benchmark == "srsd" and subset_key == "original":
        return frame["difficulty"].isin(["easy_set", "medium_set", "hard_set"])
    if benchmark == "srsd" and subset_key == "dummy":
        return frame["difficulty"].astype(str).str.endswith("_dummy")
    return frame["difficulty"].eq("lsr_transform")


def current_fixed_budget_rows(threshold: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frames = {
        benchmark: pd.read_csv(STANDARD_ROOT / benchmark / "all_results_detailed.csv")
        for benchmark in {spec[0] for spec in SUBSETS.values()}
    }
    for subset, (benchmark, subset_key, expected_n) in SUBSETS.items():
        frame = frames[benchmark]
        mask = subset_mask(frame, benchmark, subset_key)
        indices = np.flatnonzero(mask.to_numpy()) + 1
        selected = frame.loc[mask]
        test_r2 = pd.to_numeric(selected["test_r2"], errors="coerce")
        validation_hit = selected["validation_search_success"].map(as_bool)
        rows.append(
            {
                "subset": subset,
                "method": "Ours",
                "expected_n": expected_n,
                "available_result_n": expected_n,
                "finite_r2_n": int(test_r2.notna().sum()),
                "error_n": int(selected["error"].notna().sum()),
                "test_r2_hit_n": int(test_r2.gt(threshold).sum()),
                "confirmed_validation_and_test_hit_n": int(
                    (validation_hit & test_r2.gt(threshold)).sum()
                ),
            }
        )
        for method, method_dir in METHOD_DIRS.items():
            results = []
            for index in indices:
                matches = sorted(
                    (BASELINE_ROOT / method_dir / benchmark / "case_results").glob(
                        f"{index:04d}_*.json"
                    )
                )
                if not matches:
                    results.append({})
                    continue
                try:
                    results.append(json.loads(matches[0].read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    results.append({})
            scores = pd.Series([finite(result.get("test_r2")) for result in results])
            hits = pd.Series(
                [as_bool(result.get("validation_search_success")) for result in results]
            )
            rows.append(
                {
                    "subset": subset,
                    "method": method,
                    "expected_n": expected_n,
                    "available_result_n": int(sum(bool(result) for result in results)),
                    "finite_r2_n": int(scores.notna().sum()),
                    "error_n": int(
                        sum(bool(result.get("error")) for result in results if result)
                    ),
                    "test_r2_hit_n": int(scores.gt(threshold).sum()),
                    "confirmed_validation_and_test_hit_n": int(
                        (hits & scores.gt(threshold)).sum()
                    ),
                }
            )
    output = pd.DataFrame(rows)
    output["confirmed_hit_rate"] = (
        output["confirmed_validation_and_test_hit_n"] / output["expected_n"]
    )
    return output


def formula_recovery_rows() -> pd.DataFrame:
    frame = pd.read_csv(RESCUE_ROWS, low_memory=False)
    rows = []
    for subset, mask in {
        "SRSD-original": frame["dataset_dir"].isin(
            ["easy_set", "medium_set", "hard_set"]
        ),
        "SRSD-dummy": frame["dataset_dir"].astype(str).str.endswith("_dummy"),
        "SRSD-all": pd.Series(True, index=frame.index),
    }.items():
        selected = frame.loc[mask]
        for stage, prefix in [("raw", "original"), ("rescue", "selected")]:
            rows.append(
                {
                    "subset": subset,
                    "stage": stage,
                    "expected_n": len(selected),
                    "r2_gt_0_999_n": int(
                        pd.to_numeric(
                            selected[f"{prefix}_test_r2"], errors="coerce"
                        ).gt(0.999).sum()
                    ),
                    "strict_formula_recovery_n": int(
                        bool_sum(selected[f"{prefix}_strict_recovery"])
                    ),
                    "srbench_formula_recovery_n": int(
                        bool_sum(selected[f"{prefix}_srbench_recovery"])
                    ),
                }
            )
    return pd.DataFrame(rows)


def save_figure(frame: pd.DataFrame, output_dir: Path) -> None:
    from tools.plot_style import METHOD_COLORS, NATURE_COLORS, save_nature_figure, set_nature_style

    set_nature_style()
    panels = [
        ("SRBench-Feynman", "a  SRBench Feynman (119)"),
        ("SRSD-all", "b  SRSD Feynman (238)"),
        ("LLMSR-Feynman-transform", "c  Feynman transforms (111)"),
    ]
    preferred_order = [
        "Ours",
        "Ours (raw)",
        "Ours + rescue",
        "PSE",
        "PySR",
        "gplearn",
        "DSO",
        "ICSR",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, (subset, title) in zip(axes, panels):
        selected = frame[
            frame["subset"].eq(subset) & frame["available_for_comparison"]
        ].copy()
        selected["order"] = selected["method"].map(
            {name: index for index, name in enumerate(preferred_order)}
        )
        selected = selected.sort_values("order")
        colors = []
        for method in selected["method"]:
            if method == "Ours + rescue":
                colors.append(NATURE_COLORS["blue"])
            elif method.startswith("Ours"):
                colors.append(NATURE_COLORS["sky"])
            else:
                colors.append(METHOD_COLORS.get(method.lower(), NATURE_COLORS["midgray"]))
        values = 100.0 * selected["r2_gt_threshold_rate"].to_numpy(dtype=float)
        bars = ax.bar(np.arange(len(selected)), values, color=colors, width=0.72)
        ax.set_xticks(np.arange(len(selected)), selected["method"], rotation=42, ha="right")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 60)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=6.5)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Test $R^2>0.999$ (%)")
    fig.tight_layout(w_pad=1.2)
    save_nature_figure(
        fig,
        output_dir / "fig_feynman_performance_audit.png",
        section="standard_recovery",
        dpi=400,
        bbox_inches="tight",
    )
    save_nature_figure(
        fig,
        output_dir / "fig_feynman_performance_audit.pdf",
        section="standard_recovery",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paper = paper_aligned_rows(args.threshold)
    current = current_fixed_budget_rows(args.threshold)
    recovery = formula_recovery_rows()
    paper.to_csv(args.output_dir / "feynman_paper_aligned_r2.csv", index=False)
    current.to_csv(args.output_dir / "feynman_current_fixed_budget.csv", index=False)
    recovery.to_csv(args.output_dir / "feynman_ours_formula_recovery.csv", index=False)
    manifest = {
        "threshold": args.threshold,
        "rate_denominator": "all expected cases in each Feynman slice",
        "paper_aligned_note": (
            "SRBench and LLMSR rows use archived paper test R2; SRSD Ours uses the "
            "canonical expression recompute because the archived SRSD test_r2 field disagrees "
            "with reevaluation. Missing baseline coverage is marked unavailable."
        ),
        "current_fixed_budget_note": (
            "Validation and held-out test must both exceed the threshold. Fixed-budget errors "
            "and timeouts remain failures in the all-case denominator."
        ),
        "sources": [
            str(ARCHIVE_ROOT),
            str(RESCUE_ROWS),
            str(OTHER_R2_ROWS),
            str(OTHER_R2_SUMMARY),
            str(STANDARD_ROOT),
            str(BASELINE_ROOT),
        ],
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    save_figure(paper, args.output_dir)
    print(paper.to_string(index=False))
    print("\nCurrent fixed-budget audit:")
    print(current.to_string(index=False))
    print("\nOurs formula recovery:")
    print(recovery.to_string(index=False))
    print(f"[INFO] wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
