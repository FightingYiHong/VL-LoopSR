#!/usr/bin/env python3
"""Screen difficult Feynman-derived formulas for a confirmatory case study."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OURS_ROOT = (
    ROOT
    / "runs/paper_experiments/01_standard_recovery"
    / "standard_search_efficiency_full_20260724_152138"
)
DEFAULT_BASELINE_ROOT = (
    ROOT
    / "runs/paper_experiments/01_standard_recovery"
    / "baseline_search_efficiency_full_affinity_20260726_174347"
)
DEFAULT_OUTPUT_DIR = DEFAULT_BASELINE_ROOT / "hard_formula_case_study"

METHOD_DIRS = {
    "LLM-SR": "llmsr",
    "DSO": "dso",
    "gplearn": "gplearn",
    "PSE": "psrn",
    "PySR": "pysr",
    "ICSR": "icsr",
}
METHOD_ORDER = ["Ours", *METHOD_DIRS]
BENCHMARKS = ["llmsrbench", "srsd"]
TASK_INDEX_PATTERN = re.compile(r"^(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-root", type=Path, default=DEFAULT_OURS_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--min-features", type=int, default=5)
    parser.add_argument("--min-true-complexity", type=int, default=10)
    return parser.parse_args()


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def expression_complexity(expr: object) -> float:
    if pd.isna(expr):
        return float("nan")
    try:
        from benchmark_metrics import expression_complexity as score_expression

        return float(score_expression(str(expr))["expr_complexity"])
    except Exception:
        return float("nan")


def read_ours(ours_root: Path) -> pd.DataFrame:
    frames = []
    for benchmark in BENCHMARKS:
        path = ours_root / benchmark / "all_results_detailed.csv"
        frame = pd.read_csv(path)
        frame.insert(0, "task_index", np.arange(1, len(frame) + 1, dtype=int))
        frame.insert(0, "benchmark", benchmark)
        if "case_name" not in frame:
            frame["case_name"] = frame.get("base_name")
        frame["case_name"] = frame["case_name"].fillna(frame.get("base_name"))
        frames.append(frame)
    ours = pd.concat(frames, ignore_index=True, sort=False)
    ours["true_expr_complexity"] = ours["true_expression"].map(expression_complexity)
    return ours


def find_case_json(case_dir: Path, task_index: int) -> Path | None:
    matches = sorted(case_dir.glob(f"{task_index:04d}_*.json"))
    return matches[0] if matches else None


def load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def configured_budget(result: dict) -> float:
    for key in ("configured_wall_budget_sec", "case_timeout_sec", "time_budget_sec"):
        value = finite(result.get(key))
        if np.isfinite(value):
            return value
    return float("nan")


def screening_candidates(ours: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    is_feynman = ours["benchmark"].eq("srsd") | ours["difficulty"].eq("lsr_transform")
    validation_hit = ours["validation_search_success"].map(as_bool)
    test_hit = pd.to_numeric(ours["test_r2"], errors="coerce").gt(args.threshold)
    enough_features = pd.to_numeric(ours["n_features"], errors="coerce").ge(args.min_features)
    enough_complexity = ours["true_expr_complexity"].ge(args.min_true_complexity)
    selected = ours[is_feynman & validation_hit & test_hit & enough_features & enough_complexity].copy()
    selected["case_id"] = (
        selected["benchmark"] + "::" + selected["task_index"].map(lambda value: f"{value:04d}")
    )
    selected["formula_label"] = (
        "Feynman "
        + selected["case_name"]
        .astype(str)
        .str.replace(r"^feynman-", "", regex=True)
        .str.replace("_", " ")
        .str.upper()
    )
    return selected.sort_values(["true_expr_complexity", "case_id"], ascending=[False, True])


def build_method_rows(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for _, case in candidates.iterrows():
        ours_success = as_bool(case["validation_search_success"])
        rows.append(
            {
                "case_id": case["case_id"],
                "formula_label": case["formula_label"],
                "benchmark": case["benchmark"],
                "task_index": int(case["task_index"]),
                "case_name": case["case_name"],
                "method": "Ours",
                "screening_success": ours_success,
                "evaluations_to_hit": finite(case.get("evaluations_to_validation_success")),
                "validation_r2": finite(case.get("first_validation_success_val_r2")),
                "test_r2": finite(case.get("test_r2")),
                "runtime_sec": finite(case.get("runtime_sec")),
                "configured_budget_sec": finite(case.get("time_budget_sec")),
                "expression": case.get("best_expr"),
                "error": case.get("error"),
                "source_file": str(
                    args.ours_root / str(case["benchmark"]) / "all_results_detailed.csv"
                ),
            }
        )

        for method, method_dir in METHOD_DIRS.items():
            source = find_case_json(
                args.baseline_root
                / method_dir
                / str(case["benchmark"])
                / "case_results",
                int(case["task_index"]),
            )
            result = load_json(source)
            rows.append(
                {
                    "case_id": case["case_id"],
                    "formula_label": case["formula_label"],
                    "benchmark": case["benchmark"],
                    "task_index": int(case["task_index"]),
                    "case_name": case["case_name"],
                    "method": method,
                    "screening_success": as_bool(result.get("validation_search_success")),
                    "evaluations_to_hit": finite(
                        result.get("evaluations_to_validation_success")
                    ),
                    "validation_r2": finite(
                        result.get("first_validation_success_r2", result.get("val_r2"))
                    ),
                    "test_r2": finite(result.get("test_r2")),
                    "runtime_sec": finite(result.get("runtime_sec")),
                    "configured_budget_sec": configured_budget(result),
                    "expression": result.get("best_expr", result.get("expression")),
                    "error": result.get("error"),
                    "source_file": str(source) if source else None,
                }
            )
    return pd.DataFrame(rows)


def save_figure(candidates: pd.DataFrame, method_rows: pd.DataFrame, output_dir: Path) -> None:
    from tools.plot_style import NATURE_COLORS, save_nature_figure, set_nature_style

    set_nature_style()
    case_order = candidates["case_id"].tolist()
    labels = candidates.set_index("case_id").loc[case_order, "formula_label"].tolist()
    heat = (
        method_rows.pivot(index="method", columns="case_id", values="screening_success")
        .reindex(index=METHOD_ORDER, columns=case_order)
        .astype(float)
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1.35, 1.0]})
    axes[0].imshow(heat.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for row_index, method in enumerate(heat.index):
        for col_index, case_id in enumerate(heat.columns):
            value = heat.loc[method, case_id]
            text = "Hit" if value == 1 else ("No hit" if value == 0 else "NA")
            axes[0].text(
                col_index,
                row_index,
                text,
                ha="center",
                va="center",
                color="white" if value == 1 else NATURE_COLORS["dark"],
                fontsize=7,
            )
    axes[0].set_xticks(range(len(labels)), labels, rotation=18, ha="right")
    axes[0].set_yticks(
        range(len(METHOD_ORDER)),
        ["VEGA-SR" if method == "Ours" else method for method in METHOD_ORDER],
    )
    axes[0].set_title("a  Main-run screening (validation $R^2>0.999$)", loc="left")
    axes[0].tick_params(length=0)

    x = np.arange(len(candidates))
    complexity = candidates["true_expr_complexity"].to_numpy(dtype=float)
    evaluations = pd.to_numeric(
        candidates["evaluations_to_validation_success"], errors="coerce"
    ).to_numpy(dtype=float)
    bars = axes[1].bar(
        x - 0.18,
        complexity,
        width=0.36,
        color=NATURE_COLORS["blue"],
        label="True-expression complexity",
    )
    axes[1].bar(
        x + 0.18,
        evaluations,
        width=0.36,
        color=NATURE_COLORS["vermillion"],
        label="Candidates to first hit",
    )
    axes[1].set_xticks(x, labels, rotation=18, ha="right")
    axes[1].set_ylabel("Count")
    axes[1].set_title("b  Structural difficulty and search cost", loc="left")
    axes[1].legend(loc="upper right")
    axes[1].bar_label(bars, fmt="%.0f", padding=2, fontsize=6.5)
    axes[1].set_ylim(0, max(np.nanmax(complexity), np.nanmax(evaluations)) * 1.28)
    fig.tight_layout(w_pad=2.0)
    save_nature_figure(
        fig,
        output_dir / "fig_hard_formula_screening.png",
        section="standard_recovery",
        dpi=400,
        bbox_inches="tight",
    )
    save_nature_figure(
        fig,
        output_dir / "fig_hard_formula_screening.pdf",
        section="standard_recovery",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ours = read_ours(args.ours_root)
    candidates = screening_candidates(ours, args)
    method_rows = build_method_rows(candidates, args)

    baseline_rows = method_rows[method_rows["method"].ne("Ours")]
    coverage = baseline_rows.groupby("case_id")["source_file"].apply(lambda values: values.notna().sum())
    baseline_hits = baseline_rows.groupby("case_id")["screening_success"].sum()
    candidates["baseline_coverage_n"] = candidates["case_id"].map(coverage).fillna(0).astype(int)
    candidates["baseline_hit_n"] = candidates["case_id"].map(baseline_hits).fillna(0).astype(int)
    candidates["all_screened_baselines_no_hit"] = candidates["baseline_hit_n"].eq(0)

    candidate_columns = [
        "case_id",
        "formula_label",
        "benchmark",
        "task_index",
        "case_name",
        "difficulty",
        "n_features",
        "true_expr_complexity",
        "true_expression",
        "best_expr",
        "first_validation_success_expression",
        "first_validation_success_val_r2",
        "test_r2",
        "test_nrmse",
        "strict_formula_recovery",
        "srbench_formula_recovery",
        "evaluations_to_validation_success",
        "runtime_sec",
        "baseline_coverage_n",
        "baseline_hit_n",
        "all_screened_baselines_no_hit",
    ]
    candidates[candidate_columns].to_csv(
        args.output_dir / "hard_formula_screening_candidates.csv", index=False
    )
    method_rows.to_csv(args.output_dir / "hard_formula_screening_method_rows.csv", index=False)

    manifest = {
        "analysis_type": "retrospective_screen_for_confirmatory_case_study",
        "source_suite": "standard_recovery_972",
        "validation_success_rule": f"validation R2 strictly greater than {args.threshold}",
        "test_success_rule": f"test R2 strictly greater than {args.threshold}",
        "eligibility_rules": [
            "Feynman source: SRSD or LLMSRBench lsr_transform",
            f"number of observed input variables >= {args.min_features}",
            f"ground-truth expression complexity >= {args.min_true_complexity}",
            "Ours met both validation and held-out test R2 rules in the 972-case main run",
        ],
        "confirmatory_requirement": (
            "Rerun every selected formula and method with a fresh multi-seed, equal 600-second "
            "budget before making a comparative claim."
        ),
        "selected_case_ids": candidates["case_id"].tolist(),
        "selected_case_count": int(len(candidates)),
        "ours_root": str(args.ours_root.resolve()),
        "baseline_root": str(args.baseline_root.resolve()),
    }
    (args.output_dir / "selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_figure(candidates, method_rows, args.output_dir)
    print(candidates[candidate_columns].to_string(index=False))
    print(f"[INFO] wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
