#!/usr/bin/env python3
"""Build a per-case reuse inventory for the frozen high-dimensional Feynman run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
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
OURS_ARCHIVE = (
    ROOT
    / "reports/posthoc_recompute_inputs_20260724/01_main/results/archive"
    / "ours_paper_run/srbench.csv"
)
PAPER_BASELINES = (
    STANDARD_ROOT
    / "other_methods_r2/standard_recovery_other_methods_r2_case_rows.csv"
)
METHOD_DIRS = {
    "LLM-SR": "llmsr",
    "DSO": "dso",
    "gplearn": "gplearn",
    "PSE": "psrn",
    "PySR": "pysr",
    "ICSR": "icsr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def finite(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_case_name(value: object) -> str:
    text = str(value or "").strip()
    if " | " in text:
        text = text.split(" | ")[-1].strip()
    return text


def selected_cases(run_root: Path) -> list[str]:
    paths = sorted(run_root.glob("**/srbench_selected_tasks.csv"))
    if not paths:
        raise FileNotFoundError(f"No SRBench selection found under {run_root}")
    frame = pd.read_csv(paths[0])
    cases = frame["dataset_name"].map(normalize_case_name).tolist()
    if len(cases) != 34 or len(set(cases)) != 34:
        raise RuntimeError(f"Expected 34 frozen cases, found {len(set(cases))}")
    return cases


def load_json_results(directory: Path, cases: set[str]) -> dict[str, tuple[dict, Path]]:
    rows: dict[str, tuple[dict, Path]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        case_name = normalize_case_name(
            result.get("case_name", result.get("base_name", result.get("dataset_name")))
        )
        if case_name in cases:
            rows[case_name] = (result, path)
    return rows


def paper_r2_maps(cases: set[str]) -> dict[str, dict[str, tuple[float, str]]]:
    maps: dict[str, dict[str, tuple[float, str]]] = {
        method: {} for method in ["Ours", *METHOD_DIRS]
    }
    ours = pd.read_csv(OURS_ARCHIVE, low_memory=False)
    for row in ours.to_dict(orient="records"):
        case_name = normalize_case_name(row.get("srbench_dataset_name"))
        score = finite(row.get("test_r2"))
        if case_name in cases and np.isfinite(score):
            maps["Ours"][case_name] = (score, str(OURS_ARCHIVE))

    baselines = pd.read_csv(PAPER_BASELINES, low_memory=False)
    selected = baselines[
        baselines["benchmark"].eq("srbench")
        & baselines["method"].isin(["DSO", "gplearn", "PSE", "PySR", "ICSR"])
    ]
    for row in selected.to_dict(orient="records"):
        method = str(row["method"])
        case_name = normalize_case_name(row.get("case_name"))
        score = finite(row.get("test_r2"))
        if case_name in cases and np.isfinite(score):
            maps[method][case_name] = (score, str(PAPER_BASELINES))
    return maps


def current_results(run_root: Path, method_dir: str, cases: set[str]) -> dict[str, tuple[dict, Path]]:
    directory = run_root / method_dir / "seed_611" / "srbench" / "case_results"
    return load_json_results(directory, cases)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = selected_cases(args.run_root)
    case_set = set(cases)
    paper_maps = paper_r2_maps(case_set)

    ours_fixed = pd.read_csv(
        STANDARD_ROOT / "srbench/all_results_detailed.csv", low_memory=False
    )
    ours_fixed = {
        normalize_case_name(row.get("srbench_dataset_name")): row
        for row in ours_fixed.to_dict(orient="records")
        if normalize_case_name(row.get("srbench_dataset_name")) in case_set
    }
    baseline_fixed = {
        method: load_json_results(
            BASELINE_ROOT / method_dir / "srbench/case_results", case_set
        )
        for method, method_dir in METHOD_DIRS.items()
    }
    current = {
        method: current_results(args.run_root, method_dir, case_set)
        for method, method_dir in METHOD_DIRS.items()
    }

    rows = []
    for case_index, case_name in enumerate(cases, start=1):
        for method in ["Ours", *METHOD_DIRS]:
            fixed_result: dict = {}
            fixed_source = None
            if method == "Ours":
                fixed_result = ours_fixed.get(case_name, {})
                if fixed_result:
                    fixed_source = str(
                        STANDARD_ROOT / "srbench/all_results_detailed.csv"
                    )
            else:
                fixed_pair = baseline_fixed[method].get(case_name)
                if fixed_pair:
                    fixed_result, fixed_path = fixed_pair
                    fixed_source = str(fixed_path)

            current_result: dict = {}
            current_source = None
            if method != "Ours":
                current_pair = current[method].get(case_name)
                if current_pair:
                    current_result, current_path = current_pair
                    current_source = str(current_path)

            current_r2 = finite(
                current_result.get(
                    "test_r2", current_result.get("official_r2_test")
                )
            )
            fixed_r2 = finite(
                fixed_result.get("test_r2", fixed_result.get("official_r2_test"))
            )
            paper_pair = paper_maps[method].get(case_name)
            paper_r2 = paper_pair[0] if paper_pair else float("nan")
            paper_source = paper_pair[1] if paper_pair else None

            if np.isfinite(current_r2):
                reusable_r2 = current_r2
                r2_source = current_source
                r2_source_kind = "current_600s_result"
            elif np.isfinite(paper_r2):
                reusable_r2 = paper_r2
                r2_source = paper_source
                r2_source_kind = "paper_aligned_main_result"
            elif np.isfinite(fixed_r2):
                reusable_r2 = fixed_r2
                r2_source = fixed_source
                r2_source_kind = "existing_fixed_budget_result"
            else:
                reusable_r2 = float("nan")
                r2_source = None
                r2_source_kind = "missing"

            observed = finite(
                fixed_result.get(
                    "validation_search_observed_evaluations",
                    fixed_result.get("validation_search_native_evaluations"),
                )
            )
            evaluations_to_hit = finite(
                fixed_result.get(
                    "evaluations_to_validation_success",
                    fixed_result.get("native_evaluations_to_validation_success"),
                )
            )
            search_available = bool(
                np.isfinite(observed)
                or fixed_result.get("validation_search_success") is not None
            )
            expression = fixed_result.get(
                "best_expr",
                fixed_result.get(
                    "expression",
                    fixed_result.get("validation_search_best_expression"),
                ),
            )
            rows.append(
                {
                    "case_index": case_index,
                    "case_name": case_name,
                    "method": method,
                    "reusable_test_r2": reusable_r2,
                    "r2_available": bool(np.isfinite(reusable_r2)),
                    "r2_source_kind": r2_source_kind,
                    "r2_source": r2_source,
                    "search_record_available": search_available,
                    "validation_search_success": as_bool(
                        fixed_result.get("validation_search_success")
                    ),
                    "validation_search_observed_evaluations": observed,
                    "evaluations_to_validation_success": evaluations_to_hit,
                    "search_runtime_sec": finite(fixed_result.get("runtime_sec")),
                    "expr_complexity": finite(fixed_result.get("expr_complexity")),
                    "expression": expression,
                    "search_source": fixed_source if search_available else None,
                    "needs_run": not bool(np.isfinite(reusable_r2)),
                }
            )

    inventory = pd.DataFrame(rows)
    inventory.to_csv(args.output_dir / "reuse_inventory.csv", index=False)
    for method, filename in [("PySR", "missing_pysr_cases.txt"), ("ICSR", "missing_icsr_cases.txt")]:
        missing = inventory[inventory["method"].eq(method) & inventory["needs_run"]]
        (args.output_dir / filename).write_text(
            "\n".join(missing["case_name"].astype(str)) + ("\n" if len(missing) else ""),
            encoding="utf-8",
        )

    summary = (
        inventory.groupby("method", sort=False)
        .agg(
            expected_cases=("case_name", "size"),
            reusable_r2_cases=("r2_available", "sum"),
            search_record_cases=("search_record_available", "sum"),
            missing_cases=("needs_run", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(args.output_dir / "reuse_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[INFO] wrote reuse inventory to {args.output_dir}")


if __name__ == "__main__":
    main()
