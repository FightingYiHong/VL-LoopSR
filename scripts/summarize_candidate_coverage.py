#!/usr/bin/env python3
"""Summarize Fig. 6 post-hoc held-out candidate coverage.

The candidate table must contain ``method``, ``task_id``, ``evaluation_index``
and ``test_r2``. The final-expression table must contain ``method``, ``task_id``
and ``test_r2``. Test scores are consumed only by this post-hoc script; they
must never be passed back to the search or validation selector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_METHODS = ("VEGA-SR", "PSE", "PySR", "DSO", "gplearn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--finals-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--expected-tasks", type=int, default=895)
    parser.add_argument("--max-budget", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.999)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def validate_finals(
    finals: pd.DataFrame, methods: list[str], expected_tasks: int
) -> dict[str, set[str]]:
    duplicated = finals.duplicated(["method", "task_id"], keep=False)
    if duplicated.any():
        row = finals.loc[duplicated, ["method", "task_id"]].iloc[0].to_dict()
        raise ValueError(f"duplicate final-expression row: {row}")

    task_sets: dict[str, set[str]] = {}
    for method in methods:
        task_ids = set(finals.loc[finals["method"] == method, "task_id"].astype(str))
        if len(task_ids) != expected_tasks:
            raise ValueError(
                f"{method}: expected {expected_tasks} final tasks, found {len(task_ids)}"
            )
        task_sets[method] = task_ids
    reference = task_sets[methods[0]]
    for method, task_ids in task_sets.items():
        if task_ids != reference:
            raise ValueError(f"{method}: final task identifiers do not match {methods[0]}")
    return task_sets


def main() -> None:
    args = parse_args()
    if args.expected_tasks <= 0 or args.max_budget <= 0:
        raise ValueError("expected-tasks and max-budget must be positive")

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be a non-empty unique comma-separated list")

    candidates = pd.read_csv(args.candidates_csv)
    finals = pd.read_csv(args.finals_csv)
    require_columns(
        candidates,
        {"method", "task_id", "evaluation_index", "test_r2"},
        "candidate table",
    )
    require_columns(finals, {"method", "task_id", "test_r2"}, "final table")

    candidates = candidates[candidates["method"].isin(methods)].copy()
    finals = finals[finals["method"].isin(methods)].copy()
    candidates["task_id"] = candidates["task_id"].astype(str)
    finals["task_id"] = finals["task_id"].astype(str)
    candidates["evaluation_index"] = pd.to_numeric(
        candidates["evaluation_index"], errors="raise"
    ).astype(int)
    if (candidates["evaluation_index"] < 1).any():
        raise ValueError("candidate evaluation_index values must start at one")
    duplicate_candidates = candidates.duplicated(
        ["method", "task_id", "evaluation_index"], keep=False
    )
    if duplicate_candidates.any():
        row = candidates.loc[
            duplicate_candidates, ["method", "task_id", "evaluation_index"]
        ].iloc[0].to_dict()
        raise ValueError(f"duplicate evaluated-candidate position: {row}")

    task_sets = validate_finals(finals, methods, args.expected_tasks)
    for method in methods:
        candidate_tasks = set(
            candidates.loc[candidates["method"] == method, "task_id"].astype(str)
        )
        unexpected = candidate_tasks.difference(task_sets[method])
        if unexpected:
            raise ValueError(f"{method}: candidate table contains unknown task {min(unexpected)}")

    candidates["test_r2"] = pd.to_numeric(candidates["test_r2"], errors="coerce")
    finals["test_r2"] = pd.to_numeric(finals["test_r2"], errors="coerce")
    hits = candidates[candidates["test_r2"] > args.threshold]
    first_hits = (
        hits.groupby(["method", "task_id"], as_index=False)["evaluation_index"]
        .min()
        .rename(columns={"evaluation_index": "first_hit_position"})
    )

    coverage_rows: list[dict] = []
    for method in methods:
        positions = first_hits.loc[
            first_hits["method"] == method, "first_hit_position"
        ].to_numpy(dtype=int)
        for budget in range(1, args.max_budget + 1):
            solved = int(np.count_nonzero(positions <= budget))
            coverage_rows.append(
                {
                    "method": method,
                    "candidate_budget": budget,
                    "tasks_solved": solved,
                    "task_denominator": args.expected_tasks,
                    "coverage_rate": solved / args.expected_tasks,
                    "success_definition": f"held-out test R2 > {args.threshold:g}",
                }
            )

    coverage = pd.DataFrame(coverage_rows)
    final_rows = []
    for method in methods:
        method_finals = finals[finals["method"] == method]
        within_budget = int(
            coverage.loc[
                (coverage["method"] == method)
                & (coverage["candidate_budget"] == args.max_budget),
                "tasks_solved",
            ].iloc[0]
        )
        final_rows.append(
            {
                "method": method,
                f"within_{args.max_budget}_candidates": within_budget,
                "final_selected_expression": int(
                    (method_finals["test_r2"] > args.threshold).sum()
                ),
                "task_denominator": args.expected_tasks,
            }
        )
    final_summary = pd.DataFrame(final_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.output_dir / "candidate_coverage_curve.csv", index=False)
    first_hits.to_csv(args.output_dir / "candidate_first_hits.csv", index=False)
    final_summary.to_csv(args.output_dir / "candidate_coverage_summary.csv", index=False)
    manifest = {
        "candidate_source": str(args.candidates_csv.resolve()),
        "final_expression_source": str(args.finals_csv.resolve()),
        "methods": methods,
        "expected_tasks_per_method": args.expected_tasks,
        "maximum_candidate_budget": args.max_budget,
        "success_definition": f"held-out test R2 > {args.threshold:g}",
        "test_scores_used_during_search": False,
    }
    (args.output_dir / "candidate_coverage_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
