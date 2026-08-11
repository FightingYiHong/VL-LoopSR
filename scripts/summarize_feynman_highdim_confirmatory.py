#!/usr/bin/env python3
"""Summarize the frozen 34-case high-dimensional Feynman confirmation run."""

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

METHOD_DIRS = {
    "Ours": "ours",
    "LLM-SR": "llmsr",
    "DSO": "dso",
    "gplearn": "gplearn",
    "PSE": "psrn",
    "PySR": "pysr",
    "ICSR": "icsr",
}
METHOD_ORDER = list(METHOD_DIRS)
EXPECTED_CASE_COUNT = 34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.999)
    return parser.parse_args()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def repeat_seed(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def normalize_case_name(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "|" in text:
        text = text.split("|")[-1].strip()
    return text if text.lower().startswith("feynman") else None


def result_case_name(result: dict) -> str | None:
    for key in (
        "case_name",
        "base_name",
        "dataset_name",
        "srbench_dataset_name",
    ):
        name = normalize_case_name(result.get(key))
        if name is not None:
            return name
    return None


def expected_cases(run_root: Path) -> list[str]:
    candidates: set[str] = set()
    for path in run_root.glob("**/srbench_selected_tasks.csv"):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for column in ("dataset_name", "base_name", "case_name"):
            if column not in frame:
                continue
            candidates.update(
                name
                for name in frame[column].map(normalize_case_name).dropna().astype(str)
            )
    if len(candidates) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_CASE_COUNT} frozen cases, found {len(candidates)}"
        )
    return sorted(candidates)


def row_from_mapping(
    method: str,
    seed: int,
    result: dict,
    source: Path,
) -> dict | None:
    case_name = result_case_name(result)
    if case_name is None:
        return None
    validation_r2 = finite(
        result.get(
            "first_validation_success_r2",
            result.get("first_validation_success_val_r2", result.get("val_r2")),
        )
    )
    validation_hit = as_bool(result.get("validation_search_success"))
    expression = result.get(
        "best_expr",
        result.get("expression", result.get("first_validation_success_expression")),
    )
    return {
        "method": method,
        "repeat_seed": seed,
        "case_name": case_name,
        "result_available": True,
        "same_run_confirmation_available": True,
        "validation_hit": validation_hit,
        "validation_r2": validation_r2,
        "test_r2": finite(result.get("test_r2", result.get("official_r2_test"))),
        "evaluations_to_hit": finite(
            result.get(
                "evaluations_to_validation_success",
                result.get("native_evaluations_to_validation_success"),
            )
        ),
        "runtime_sec": finite(result.get("runtime_sec")),
        "expr_complexity": finite(result.get("expr_complexity")),
        "expression": expression,
        "strict_formula_recovery": as_bool(result.get("strict_formula_recovery")),
        "srbench_formula_recovery": as_bool(result.get("srbench_formula_recovery")),
        "error": result.get("error"),
        "r2_source_kind": "current_same_run_result",
        "source_file": str(source),
    }


def read_ours(run_root: Path) -> list[dict]:
    rows = []
    for path in sorted((run_root / "ours").glob("seed_*/srbench/all_results_detailed.csv")):
        seed = repeat_seed(path)
        if seed is None:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for result in frame.to_dict(orient="records"):
            row = row_from_mapping("Ours", seed, result, path)
            if row is not None:
                rows.append(row)
    return rows


def read_json_method(run_root: Path, method: str, method_dir: str) -> list[dict]:
    rows = []
    for path in sorted((run_root / method_dir).glob("seed_*/**/*.json")):
        if path.name in {"manifest.json", "run_config.json", "summary.json"}:
            continue
        seed = repeat_seed(path)
        if seed is None:
            continue
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(result, dict):
            continue
        row = row_from_mapping(method, seed, result, path)
        if row is not None:
            rows.append(row)
    unique = {}
    for row in rows:
        key = (row["method"], row["repeat_seed"], row["case_name"])
        old = unique.get(key)
        if old is None or (not old["validation_hit"] and row["validation_hit"]):
            unique[key] = row
    return list(unique.values())


def read_reuse_inventory(run_root: Path) -> list[dict]:
    path = run_root / "reuse/reuse_inventory.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, low_memory=False)
    rows = []
    for result in frame.to_dict(orient="records"):
        if not as_bool(result.get("r2_available")):
            continue
        rows.append(
            {
                "method": result["method"],
                "repeat_seed": 611,
                "case_name": result["case_name"],
                "result_available": True,
                "same_run_confirmation_available": False,
                "validation_hit": as_bool(result.get("validation_search_success")),
                "validation_r2": float("nan"),
                "test_r2": finite(result.get("reusable_test_r2")),
                "evaluations_to_hit": finite(
                    result.get("evaluations_to_validation_success")
                ),
                "runtime_sec": finite(result.get("search_runtime_sec")),
                "expr_complexity": finite(result.get("expr_complexity")),
                "expression": result.get("expression"),
                "strict_formula_recovery": False,
                "srbench_formula_recovery": False,
                "error": None,
                "r2_source_kind": result.get("r2_source_kind"),
                "source_file": result.get("r2_source"),
            }
        )
    return rows


def complete_grid(rows: list[dict], cases: list[str], run_root: Path) -> pd.DataFrame:
    available = pd.DataFrame(rows)
    seeds = sorted(
        {
            seed
            for method_dir in METHOD_DIRS.values()
            for path in (run_root / method_dir).glob("seed_*")
            if (seed := repeat_seed(path)) is not None
        }
    )
    if not seeds:
        raise RuntimeError("No repeat seed directories found")
    index = pd.MultiIndex.from_product(
        [METHOD_ORDER, seeds, cases], names=["method", "repeat_seed", "case_name"]
    )
    if available.empty:
        frame = index.to_frame(index=False)
    else:
        available = available.drop_duplicates(
            ["method", "repeat_seed", "case_name"], keep="last"
        ).set_index(["method", "repeat_seed", "case_name"])
        frame = available.reindex(index).reset_index()
    frame["result_available"] = frame["result_available"].fillna(False).astype(bool)
    frame["same_run_confirmation_available"] = (
        frame["same_run_confirmation_available"].fillna(False).astype(bool)
    )
    frame["validation_hit"] = frame["validation_hit"].fillna(False).astype(bool)
    frame["strict_formula_recovery"] = (
        frame["strict_formula_recovery"].fillna(False).astype(bool)
    )
    frame["srbench_formula_recovery"] = (
        frame["srbench_formula_recovery"].fillna(False).astype(bool)
    )
    frame["error"] = frame["error"].where(frame["result_available"], "missing result")
    return frame


def bootstrap_case_ci(case_rates: pd.Series, seed: int = 20260802) -> tuple[float, float]:
    values = case_rates.to_numpy(dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def summarize(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = frame.copy()
    data["test_hit"] = pd.to_numeric(data["test_r2"], errors="coerce").gt(threshold)
    data["confirmed_hit"] = (
        data["same_run_confirmation_available"]
        & data["validation_hit"]
        & data["test_hit"]
    )
    data["confirmed_formula_recovery"] = (
        data["confirmed_hit"] & data["srbench_formula_recovery"]
    )
    per_seed = (
        data.groupby(["method", "repeat_seed"], sort=False)
        .agg(
            expected_cases=("case_name", "size"),
            available_results=("result_available", "sum"),
            finite_test_r2=("test_r2", lambda values: pd.to_numeric(values, errors="coerce").notna().sum()),
            test_r2_hits=("test_hit", "sum"),
            confirmed_hits=("confirmed_hit", "sum"),
            formula_recoveries=("confirmed_formula_recovery", "sum"),
        )
        .reset_index()
    )
    per_seed["test_r2_hit_rate"] = per_seed["test_r2_hits"] / per_seed["expected_cases"]
    per_seed["confirmed_hit_rate"] = per_seed["confirmed_hits"] / per_seed["expected_cases"]

    rows = []
    for method in METHOD_ORDER:
        selected = data[data["method"].eq(method)]
        case_rates = selected.groupby("case_name")["test_hit"].mean()
        ci_low, ci_high = bootstrap_case_ci(case_rates)
        hit_rows = selected[selected["test_hit"]]
        search_hit_rows = selected[
            selected["validation_hit"]
            & pd.to_numeric(selected["evaluations_to_hit"], errors="coerce").notna()
        ]
        rows.append(
            {
                "method": method,
                "expected_case_seed_rows": len(selected),
                "available_result_rows": int(selected["result_available"].sum()),
                "finite_test_r2_rows": int(
                    pd.to_numeric(selected["test_r2"], errors="coerce").notna().sum()
                ),
                "error_or_missing_rows": int(selected["error"].notna().sum()),
                "test_r2_hits": int(selected["test_hit"].sum()),
                "test_r2_hit_rate": float(selected["test_hit"].mean()),
                "confirmed_hits": int(selected["confirmed_hit"].sum()),
                "confirmed_hit_rate": float(selected["confirmed_hit"].mean()),
                "case_bootstrap_95ci_low": ci_low,
                "case_bootstrap_95ci_high": ci_high,
                "formula_recoveries": int(selected["confirmed_formula_recovery"].sum()),
                "formula_recovery_rate": float(
                    selected["confirmed_formula_recovery"].mean()
                ),
                "median_evaluations_to_hit": finite(
                    pd.to_numeric(
                        search_hit_rows["evaluations_to_hit"], errors="coerce"
                    ).median()
                ),
                "median_complexity_successes": finite(
                    pd.to_numeric(hit_rows["expr_complexity"], errors="coerce").median()
                ),
                "median_runtime_sec": finite(
                    pd.to_numeric(selected["runtime_sec"], errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame(rows), per_seed


def save_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    from tools.plot_style import NATURE_COLORS, palette_for, save_nature_figure, set_nature_style

    set_nature_style()
    selected = summary.set_index("method").reindex(METHOD_ORDER).reset_index()
    colors = palette_for(METHOD_ORDER)
    x = np.arange(len(selected))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    values = 100 * selected["test_r2_hit_rate"].to_numpy(dtype=float)
    lower = 100 * (
        selected["test_r2_hit_rate"] - selected["case_bootstrap_95ci_low"]
    ).clip(lower=0).to_numpy(dtype=float)
    upper = 100 * (
        selected["case_bootstrap_95ci_high"] - selected["test_r2_hit_rate"]
    ).clip(lower=0).to_numpy(dtype=float)
    bars = axes[0].bar(
        x,
        values,
        color=[colors[name] for name in METHOD_ORDER],
        yerr=np.vstack([lower, upper]),
        capsize=2,
        width=0.72,
    )
    axes[0].bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=6.5)
    axes[0].set_ylabel("Test $R^2>0.999$ (%)")
    axes[0].set_title("a  High-dimensional Feynman accuracy", loc="left")

    coverage = 100 * (
        selected["finite_test_r2_rows"] / selected["expected_case_seed_rows"]
    ).to_numpy(dtype=float)
    bars = axes[1].bar(
        x,
        coverage,
        color=[colors[name] for name in METHOD_ORDER],
        width=0.72,
    )
    axes[1].bar_label(bars, labels=[f"{value:.0f}" for value in coverage], padding=3, fontsize=6.5)
    axes[1].set_ylabel("Finite test $R^2$ coverage (%)")
    axes[1].set_title("b  Completed evaluation coverage", loc="left")

    for ax in axes:
        ax.set_xticks(x, ["VEGA-SR" if name == "Ours" else name for name in METHOD_ORDER], rotation=36, ha="right")
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 108)
    fig.tight_layout(w_pad=1.8)
    save_nature_figure(
        fig,
        output_dir / "fig_feynman_highdim_confirmatory.png",
        section="standard_recovery",
        dpi=400,
        bbox_inches="tight",
    )
    save_nature_figure(
        fig,
        output_dir / "fig_feynman_highdim_confirmatory.pdf",
        section="standard_recovery",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = expected_cases(args.run_root)
    rows = read_reuse_inventory(args.run_root)
    rows.extend(read_ours(args.run_root))
    for method, method_dir in METHOD_DIRS.items():
        if method != "Ours":
            rows.extend(read_json_method(args.run_root, method, method_dir))
    frame = complete_grid(rows, cases, args.run_root)
    summary, per_seed = summarize(frame, args.threshold)
    frame.to_csv(args.output_dir / "feynman_highdim_case_seed_rows.csv", index=False)
    summary.to_csv(args.output_dir / "feynman_highdim_method_summary.csv", index=False)
    per_seed.to_csv(args.output_dir / "feynman_highdim_seed_summary.csv", index=False)
    pd.DataFrame({"case_name": cases}).to_csv(
        args.output_dir / "frozen_feynman_highdim_cases.csv", index=False
    )
    save_figure(summary, args.output_dir)
    print(summary.to_string(index=False))
    print(f"[INFO] wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
