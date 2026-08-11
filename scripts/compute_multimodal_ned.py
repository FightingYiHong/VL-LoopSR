#!/usr/bin/env python3
"""Compute SRSD-style normalized equation-tree edit distance (NED).

The implementation follows Matsubara et al. (2024) and the public
``omron-sinicx/srsd-benchmark`` implementation:

    NED = min(TED(T_pred, T_true), |T_true|) / |T_true|

Constants share the label ``Const``; variable identities and SymPy operator
labels are retained.  Insertions, deletions, and node relabellings have unit
cost, matching the metric definition and the dependency set of the public code.

Besides final-output NED, this script reports the minimum NED among fitted,
successfully evaluated candidates.  The latter is explicitly labelled an
oracle candidate-pool diagnostic and must not be presented as final performance.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sympy
from sympy.utilities.misc import func_name


@dataclass
class Tree:
    label: str
    children: list["Tree"]


def label_distance(left: str, right: str) -> int:
    """Unit-cost insert, delete, or relabel operation."""
    return int(left != right)


def sympy_to_tree(expression: sympy.Expr) -> Tree:
    if expression.is_number:
        label = "Const"
    elif isinstance(expression, sympy.Symbol):
        label = str(expression)
    else:
        label = func_name(expression)
    return Tree(label=label, children=[sympy_to_tree(child) for child in expression.args])


def count_nodes(tree: Tree) -> int:
    return 1 + sum(count_nodes(child) for child in tree.children)


def _postorder_annotations(root: Tree) -> tuple[list[Tree | None], list[int], list[int]]:
    """Return one-indexed postorder nodes, leftmost descendants, and keyroots."""
    nodes: list[Tree | None] = [None]
    leftmost: list[int] = [0]

    def visit(node: Tree) -> int:
        child_leftmost: list[int] = []
        for child in node.children:
            child_leftmost.append(visit(child))
        node_index = len(nodes)
        nodes.append(node)
        leftmost.append(child_leftmost[0] if child_leftmost else node_index)
        return leftmost[node_index]

    visit(root)
    last_for_leftmost: dict[int, int] = {}
    for index in range(1, len(nodes)):
        last_for_leftmost[leftmost[index]] = index
    return nodes, leftmost, sorted(last_for_leftmost.values())


def tree_edit_distance(left: Tree, right: Tree) -> int:
    """Exact ordered tree-edit distance with zss-compatible label costs."""
    left_nodes, left_lmd, left_keyroots = _postorder_annotations(left)
    right_nodes, right_lmd, right_keyroots = _postorder_annotations(right)
    tree_distance = np.zeros((len(left_nodes), len(right_nodes)), dtype=np.int64)

    for left_root in left_keyroots:
        for right_root in right_keyroots:
            left_start = left_lmd[left_root]
            right_start = right_lmd[right_root]
            forest_distance = np.zeros(
                (left_root - left_start + 2, right_root - right_start + 2),
                dtype=np.int64,
            )

            for row in range(1, forest_distance.shape[0]):
                left_index = left_start + row - 1
                label = left_nodes[left_index].label  # type: ignore[union-attr]
                forest_distance[row, 0] = forest_distance[row - 1, 0] + label_distance(label, "")
            for column in range(1, forest_distance.shape[1]):
                right_index = right_start + column - 1
                label = right_nodes[right_index].label  # type: ignore[union-attr]
                forest_distance[0, column] = forest_distance[0, column - 1] + label_distance("", label)

            for row in range(1, forest_distance.shape[0]):
                left_index = left_start + row - 1
                left_label = left_nodes[left_index].label  # type: ignore[union-attr]
                for column in range(1, forest_distance.shape[1]):
                    right_index = right_start + column - 1
                    right_label = right_nodes[right_index].label  # type: ignore[union-attr]
                    delete_cost = forest_distance[row - 1, column] + label_distance(left_label, "")
                    insert_cost = forest_distance[row, column - 1] + label_distance("", right_label)

                    if left_lmd[left_index] == left_start and right_lmd[right_index] == right_start:
                        update_cost = (
                            forest_distance[row - 1, column - 1]
                            + label_distance(left_label, right_label)
                        )
                        best = min(delete_cost, insert_cost, update_cost)
                        forest_distance[row, column] = best
                        tree_distance[left_index, right_index] = best
                    else:
                        preceding_left_forest = left_lmd[left_index] - left_start
                        preceding_right_forest = right_lmd[right_index] - right_start
                        subtree_cost = (
                            forest_distance[preceding_left_forest, preceding_right_forest]
                            + tree_distance[left_index, right_index]
                        )
                        forest_distance[row, column] = min(
                            delete_cost, insert_cost, subtree_cost
                        )

    return int(tree_distance[-1, -1])


@lru_cache(maxsize=200_000)
def _parse_and_normalize_text(text: str) -> sympy.Expr:
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        raise ValueError("missing expression")
    text = text.replace("^", "**")
    locals_map = {
        "abs": sympy.Abs,
        "Abs": sympy.Abs,
        "ln": sympy.log,
        "log": sympy.log,
        "sqrt": sympy.sqrt,
        "sign": sympy.sign,
        "sgn": sympy.sign,
        "asin": sympy.asin,
        "acos": sympy.acos,
        "atan": sympy.atan,
        "arcsin": sympy.asin,
        "arccos": sympy.acos,
        "arctan": sympy.atan,
        "exp": sympy.exp,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "sinh": sympy.sinh,
        "cosh": sympy.cosh,
        "tanh": sympy.tanh,
        "pi": sympy.pi,
        "E": sympy.E,
    }
    parsed = sympy.sympify(text, locals=locals_map)
    # Match the normalization sequence in srsd-benchmark/eq_comparator.py.
    parsed = sympy.sympify(str(parsed))
    parsed = (
        parsed.subs(sympy.pi, sympy.pi.evalf())
        .evalf()
        .factor()
        .simplify()
        .subs(1.0, 1)
    )
    return sympy.sympify(str(parsed))


def parse_and_normalize(expression: Any) -> sympy.Expr:
    if expression is None or (isinstance(expression, float) and math.isnan(expression)):
        raise ValueError("missing expression")
    return _parse_and_normalize_text(str(expression).strip())


def compute_ned(predicted: Any, truth: Any) -> dict[str, Any]:
    try:
        truth_expr = parse_and_normalize(truth)
        truth_tree = sympy_to_tree(truth_expr)
    except Exception as exc:
        return {
            "ned_status": "invalid_ground_truth",
            "ned_error": f"{type(exc).__name__}: {exc}",
            "raw_tree_edit_distance": np.nan,
            "true_tree_nodes": np.nan,
            "predicted_tree_nodes": np.nan,
            "ned": np.nan,
            "structural_similarity": np.nan,
        }
    try:
        predicted_expr = parse_and_normalize(predicted)
        predicted_tree = sympy_to_tree(predicted_expr)
    except Exception as exc:
        return {
            "ned_status": "unavailable_prediction",
            "ned_error": f"{type(exc).__name__}: {exc}",
            "raw_tree_edit_distance": np.nan,
            "true_tree_nodes": count_nodes(truth_tree),
            "predicted_tree_nodes": np.nan,
            "ned": np.nan,
            "structural_similarity": np.nan,
        }
    raw_distance = tree_edit_distance(predicted_tree, truth_tree)
    truth_nodes = count_nodes(truth_tree)
    ned = min(raw_distance, truth_nodes) / truth_nodes
    return {
        "ned_status": "evaluable",
        "ned_error": "",
        "raw_tree_edit_distance": raw_distance,
        "true_tree_nodes": truth_nodes,
        "predicted_tree_nodes": count_nodes(predicted_tree),
        "ned": ned,
        "structural_similarity": 1.0 - ned,
    }


def load_json(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        return json.loads(str(value))
    except Exception:
        return []


def candidate_records(row: pd.Series, truth: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    history = load_json(row.get("candidate_evaluation_history"))
    if not isinstance(history, list):
        return records
    for item in history:
        if not isinstance(item, dict):
            continue
        predicted = item.get("fitted_expression")
        if not item.get("success") or predicted is None:
            continue
        metrics = compute_ned(predicted, truth)
        records.append(
            {
                "method": row.get("method"),
                "case_index": row.get("case_index"),
                "case_name": row.get("case_name"),
                "repeat_seed": row.get("repeat_seed"),
                "evaluation_index": item.get("evaluation_index"),
                "unique_evaluations_seen": item.get("unique_evaluations_seen"),
                "is_new_unique_candidate": item.get("is_new_unique_candidate"),
                "stage": item.get("stage"),
                "candidate_expression": item.get("expression"),
                "fitted_expression": predicted,
                "candidate_success": item.get("success"),
                "candidate_val_mse": item.get("val_mse"),
                "candidate_val_r2": item.get("val_r2"),
                "true_expression": truth,
                **metrics,
            }
        )
    return records


def percentile_interval(values: Iterable[float], reps: int, seed: int) -> tuple[float, float]:
    clean = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not len(clean):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(clean, size=(reps, len(clean)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_methods(
    casewise: pd.DataFrame, reps: int, seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in casewise.groupby("method", sort=False):
        final_values = pd.to_numeric(group["final_ned"], errors="coerce")
        final_total = pd.to_numeric(group["final_ned_total_denominator"], errors="coerce")
        candidate_values = pd.to_numeric(group["oracle_candidate_pool_min_ned"], errors="coerce")
        initial_values = pd.to_numeric(group["oracle_initial_pool_min_ned"], errors="coerce")
        final_low, final_high = percentile_interval(final_values, reps, seed)
        total_low, total_high = percentile_interval(final_total, reps, seed)
        candidate_low, candidate_high = percentile_interval(candidate_values, reps, seed)
        rows.append(
            {
                "method": method,
                "n_completed_cases": len(group),
                "n_final_evaluable": int(final_values.notna().sum()),
                "mean_final_ned_evaluable": final_values.mean(),
                "median_final_ned_evaluable": final_values.median(),
                "mean_final_structural_similarity_evaluable": 1.0 - final_values.mean(),
                "mean_final_ned_total_denominator": final_total.mean(),
                "mean_final_ned_evaluable_ci95_low": final_low,
                "mean_final_ned_evaluable_ci95_high": final_high,
                "mean_final_ned_total_ci95_low": total_low,
                "mean_final_ned_total_ci95_high": total_high,
                "mean_oracle_candidate_pool_min_ned": candidate_values.mean(),
                "mean_oracle_candidate_pool_similarity": 1.0 - candidate_values.mean(),
                "mean_oracle_candidate_pool_min_ned_ci95_low": candidate_low,
                "mean_oracle_candidate_pool_min_ned_ci95_high": candidate_high,
                "mean_oracle_initial_pool_min_ned": initial_values.mean(),
            }
        )
    return pd.DataFrame(rows)


def paired_differences(
    casewise: pd.DataFrame, reference_method: str, reps: int, seed: int
) -> pd.DataFrame:
    metrics = [
        ("final_ned_total_denominator", "lower_is_better"),
        ("final_ned", "lower_is_better"),
        ("oracle_candidate_pool_min_ned", "lower_is_better"),
        ("oracle_initial_pool_min_ned", "lower_is_better"),
    ]
    rows: list[dict[str, Any]] = []
    methods = [method for method in casewise["method"].dropna().unique() if method != reference_method]
    for comparator in methods:
        for metric, direction in metrics:
            subset = casewise[
                casewise["method"].isin([reference_method, comparator])
            ].pivot_table(
                index=["case_index", "repeat_seed"],
                columns="method",
                values=metric,
                aggfunc="first",
            )
            if reference_method not in subset or comparator not in subset:
                continue
            paired = subset[[reference_method, comparator]].dropna()
            differences = paired[reference_method] - paired[comparator]
            low, high = percentile_interval(differences, reps, seed)
            rows.append(
                {
                    "reference_method": reference_method,
                    "comparator_method": comparator,
                    "metric": metric,
                    "direction": direction,
                    "difference_definition": "reference_minus_comparator",
                    "n_paired_cases": len(paired),
                    "reference_mean_on_paired_cases": paired[reference_method].mean(),
                    "comparator_mean_on_paired_cases": paired[comparator].mean(),
                    "mean_paired_difference": differences.mean(),
                    "paired_difference_ci95_low": low,
                    "paired_difference_ci95_high": high,
                    "reference_win_rate": float((differences < 0).mean()) if len(paired) else np.nan,
                    "tie_rate": float(np.isclose(differences, 0).mean()) if len(paired) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def audit_candidate_sequence_identity(
    candidates: pd.DataFrame, reference_method: str
) -> pd.DataFrame:
    """Check whether paired methods actually evaluated different candidates."""
    if candidates.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    methods = [
        method
        for method in candidates["method"].dropna().unique()
        if method != reference_method
    ]
    for comparator in methods:
        reference_cases = set(
            candidates.loc[candidates["method"] == reference_method, "case_index"]
        )
        comparator_cases = set(
            candidates.loc[candidates["method"] == comparator, "case_index"]
        )
        for case_index in sorted(reference_cases & comparator_cases):
            left = candidates[
                (candidates["method"] == reference_method)
                & (candidates["case_index"] == case_index)
            ].sort_values("evaluation_index")
            right = candidates[
                (candidates["method"] == comparator)
                & (candidates["case_index"] == case_index)
            ].sort_values("evaluation_index")
            common = min(len(left), len(right))
            left_expr = left["fitted_expression"].astype(str).iloc[:common].to_numpy()
            right_expr = right["fitted_expression"].astype(str).iloc[:common].to_numpy()
            left_ned = pd.to_numeric(left["ned"], errors="coerce").iloc[:common].to_numpy()
            right_ned = pd.to_numeric(right["ned"], errors="coerce").iloc[:common].to_numpy()
            rows.append(
                {
                    "case_index": case_index,
                    "reference_method": reference_method,
                    "comparator_method": comparator,
                    "reference_candidate_count": len(left),
                    "comparator_candidate_count": len(right),
                    "common_prefix_candidate_count": common,
                    "common_prefix_exact_expression_match_rate": (
                        float(np.mean(left_expr == right_expr)) if common else np.nan
                    ),
                    "common_prefix_ned_match_rate": (
                        float(np.mean(np.isclose(left_ned, right_ned, equal_nan=True)))
                        if common
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-method", default="qwen3_vl_multimodal")
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260730)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = pd.read_csv(args.results_csv)
    if args.manifest_csv and args.manifest_csv.exists():
        manifest = pd.read_csv(args.manifest_csv)
        manifest_columns = [
            column
            for column in ["global_case_index", "case_name", "structure_type", "true_expression"]
            if column in manifest
        ]
        manifest = manifest[manifest_columns].rename(
            columns={
                "global_case_index": "case_index",
                "true_expression": "manifest_true_expression",
                "case_name": "manifest_case_name",
                "structure_type": "manifest_structure_type",
            }
        )
        results = results.merge(manifest, on="case_index", how="left")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for _, row in results.iterrows():
        truth = row.get("true_expression_for_scoring")
        if truth is None or (isinstance(truth, float) and math.isnan(truth)):
            truth = row.get("manifest_true_expression", row.get("true_expression"))
        text_proposal_stats = load_json(row.get("text_proposal_stats"))
        final_metrics = compute_ned(row.get("best_expr"), truth)
        candidates = candidate_records(row, truth)
        candidate_rows.extend(candidates)
        candidate_frame = pd.DataFrame(candidates)
        if candidate_frame.empty:
            candidate_min = initial_min = np.nan
            candidate_count = initial_count = 0
        else:
            candidate_ned = pd.to_numeric(candidate_frame["ned"], errors="coerce")
            candidate_min = candidate_ned.min()
            candidate_count = int(candidate_ned.notna().sum())
            initial_mask = candidate_frame["stage"].astype(str).str.startswith("initial")
            initial_ned = candidate_ned[initial_mask]
            initial_min = initial_ned.min()
            initial_count = int(initial_ned.notna().sum())
        final_ned = final_metrics["ned"]
        # The published SRSD benchmark assigns maximal NED=1 to a missing or
        # invalid prediction.  We retain the raw unavailable status separately.
        final_total = final_ned if np.isfinite(final_ned) else 1.0
        case_rows.append(
            {
                "method": row.get("method"),
                "method_description": row.get("method_description"),
                "case_index": row.get("case_index"),
                "case_name": row.get("case_name", row.get("manifest_case_name")),
                "repeat_seed": row.get("repeat_seed"),
                "structure_type": row.get("structure_type", row.get("manifest_structure_type")),
                "true_expression": truth,
                "final_expression": row.get("best_expr"),
                "strict_formula_recovery": row.get("strict_formula_recovery"),
                "skeleton_recovery": row.get("skeleton_recovery"),
                "exact_support_recovery": row.get("exact_support_recovery"),
                "mm_requested": row.get("mm_requested"),
                "mm_assets_succeeded": row.get("mm_assets_succeeded"),
                "mm_candidate_count": row.get("mm_candidate_count"),
                "mm_used_in_evaluation": row.get("mm_used_in_evaluation"),
                "num_plots": row.get("num_plots"),
                "vlm_observer_used": row.get("vlm_observer_used"),
                "text_proposer_skipped": (
                    text_proposal_stats.get("skipped")
                    if isinstance(text_proposal_stats, dict)
                    else np.nan
                ),
                "text_proposer_skip_reason": (
                    text_proposal_stats.get("skip_reason")
                    if isinstance(text_proposal_stats, dict)
                    else ""
                ),
                "final_ned": final_ned,
                "final_structural_similarity": final_metrics["structural_similarity"],
                "final_ned_status": final_metrics["ned_status"],
                "final_ned_error": final_metrics["ned_error"],
                "final_raw_tree_edit_distance": final_metrics["raw_tree_edit_distance"],
                "true_tree_nodes": final_metrics["true_tree_nodes"],
                "final_predicted_tree_nodes": final_metrics["predicted_tree_nodes"],
                "final_ned_total_denominator": final_total,
                "candidate_pool_evaluable_count": candidate_count,
                "oracle_candidate_pool_min_ned": candidate_min,
                "oracle_candidate_pool_max_similarity": (
                    1.0 - candidate_min if np.isfinite(candidate_min) else np.nan
                ),
                "initial_pool_evaluable_count": initial_count,
                "oracle_initial_pool_min_ned": initial_min,
                "oracle_initial_pool_max_similarity": (
                    1.0 - initial_min if np.isfinite(initial_min) else np.nan
                ),
            }
        )

    casewise = pd.DataFrame(case_rows)
    candidates = pd.DataFrame(candidate_rows)
    summary = summarize_methods(casewise, args.bootstrap_reps, args.bootstrap_seed)
    paired = paired_differences(
        casewise, args.reference_method, args.bootstrap_reps, args.bootstrap_seed
    )
    sequence_audit = audit_candidate_sequence_identity(candidates, args.reference_method)

    casewise.to_csv(args.out_dir / "ned_casewise.csv", index=False)
    candidates.to_csv(args.out_dir / "ned_candidate_trajectory.csv", index=False)
    summary.to_csv(args.out_dir / "ned_summary_by_method.csv", index=False)
    paired.to_csv(args.out_dir / "ned_paired_differences.csv", index=False)
    sequence_audit.to_csv(args.out_dir / "candidate_sequence_identity_audit.csv", index=False)

    notes = f"""# Normalized equation-tree edit distance analysis

- Definition: `NED = min(TED(predicted, truth), |truth|) / |truth|`; lower is better.
- Tree representation and normalization follow the public SRSD benchmark code.
- Constants are all labelled `Const`; variable names and operator identities are retained.
- Final-output NED is the confirmatory outcome.
- `oracle_candidate_pool_min_ned` and `oracle_initial_pool_min_ned` use the
  ground truth to inspect whether a structurally close candidate was generated.
  They are mechanism diagnostics, not deployable selection performance.
- Missing or invalid final predictions remain explicitly marked in
  `final_ned_status`; `final_ned_total_denominator` assigns them NED=1 exactly
  as the published benchmark implementation does.
- Bootstrap intervals resample benchmark cases ({args.bootstrap_reps:,} draws,
  seed {args.bootstrap_seed}). Current outputs are snapshots of completed rows.
- `candidate_sequence_identity_audit.csv` tests whether paired methods actually
  evaluated different fitted expressions before interpreting any NED difference.
- Source results: `{args.results_csv.resolve()}`
"""
    (args.out_dir / "README.md").write_text(notes, encoding="utf-8")

    print(f"Wrote {len(casewise)} case rows and {len(candidates)} candidate rows to {args.out_dir}")
    print(summary.to_string(index=False))
    if not paired.empty:
        print("\nPaired differences (reference minus comparator; negative favours reference):")
        print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
