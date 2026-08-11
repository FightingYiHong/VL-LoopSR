#!/usr/bin/env python3
"""Freeze the conservative, publication-facing SRSD rescue result.

The input contains the archived VEGA-SR expression and the best exploratory
validation-only candidate for each case.  The final expression changes only
when that candidate achieves NMSE <= 0.001 on both train and validation.
Test outcomes and target-formula comparisons are copied only after this gate
has been applied; they never determine which expression is selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    ROOT
    / "runs"
    / "paper_experiments"
    / "01_standard_recovery"
    / "srsd_expression_rescue"
    / "run_20260725_225847"
)
THRESHOLD = 0.001


def bootstrap_interval(
    original: np.ndarray,
    final: np.ndarray,
    statistic,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, float]:
    n_cases = len(original)
    values = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = rng.integers(0, n_cases, size=n_cases)
        values[index] = statistic(final[sample]) - statistic(original[sample])
    estimate = statistic(final) - statistic(original)
    return {
        "estimate": float(estimate),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def as_boolean(series: pd.Series) -> pd.Series:
    """Normalize bool/object recovery columns without silent downcasting."""
    return series.astype("boolean").fillna(False).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    source = pd.read_csv(args.run / "srsd_expression_rescue_case_rows.csv")
    use_rescue = (
        (source["selected_train_nmse"] <= THRESHOLD)
        & (source["selected_val_nmse"] <= THRESHOLD)
    )
    final = source.copy()
    final["final_uses_rescue"] = use_rescue
    final["final_selection_reason"] = np.where(
        use_rescue,
        "train_and_validation_nmse_at_most_0.001",
        "retain_archived_expression",
    )

    suffixes = [
        "expression",
        "train_nmse",
        "val_nmse",
        "test_nmse",
        "test_r2",
        "complexity",
        "high_accuracy",
        "strict_recovery",
        "srbench_recovery",
        "recovery_status",
    ]
    for suffix in suffixes:
        final[f"final_{suffix}"] = np.where(
            use_rescue,
            source[f"selected_{suffix}"],
            source[f"original_{suffix}"],
        )

    n_cases = len(final)
    unavailable_statuses = {"parse_unavailable", "timeout_unavailable"}
    unavailable_formula = final["final_recovery_status"].isin(unavailable_statuses)
    summary = {
        "cases": int(n_cases),
        "denominator_definition": "all SRSD benchmark cases",
        "validation_gate_passed_cases": int(use_rescue.sum()),
        "changed_expressions": int(
            (final["final_expression"] != final["original_expression"]).sum()
        ),
        "original_high_accuracy_successes": int(
            final["original_high_accuracy"].astype(bool).sum()
        ),
        "final_high_accuracy_successes": int(
            final["final_high_accuracy"].astype(bool).sum()
        ),
        "original_high_accuracy_rate": float(
            final["original_high_accuracy"].astype(bool).mean()
        ),
        "final_high_accuracy_rate": float(
            final["final_high_accuracy"].astype(bool).mean()
        ),
        "original_strict_recovery_successes": int(
            as_boolean(final["original_strict_recovery"]).sum()
        ),
        "final_strict_recovery_successes": int(
            as_boolean(final["final_strict_recovery"]).sum()
        ),
        "original_srbench_recovery_successes": int(
            as_boolean(final["original_srbench_recovery"]).sum()
        ),
        "final_srbench_recovery_successes": int(
            as_boolean(final["final_srbench_recovery"]).sum()
        ),
        "original_median_test_nmse": float(final["original_test_nmse"].median()),
        "final_median_test_nmse": float(final["final_test_nmse"].median()),
        "original_median_complexity": float(final["original_complexity"].median()),
        "final_median_complexity": float(final["final_complexity"].median()),
        "original_mean_complexity": float(final["original_complexity"].mean()),
        "final_mean_complexity": float(final["final_complexity"].mean()),
        "high_accuracy_gains": int(
            (
                ~final["original_high_accuracy"].astype(bool)
                & final["final_high_accuracy"].astype(bool)
            ).sum()
        ),
        "high_accuracy_losses": int(
            (
                final["original_high_accuracy"].astype(bool)
                & ~final["final_high_accuracy"].astype(bool)
            ).sum()
        ),
        "strict_recovery_gains": int(
            (
                ~as_boolean(final["original_strict_recovery"])
                & as_boolean(final["final_strict_recovery"])
            ).sum()
        ),
        "strict_recovery_losses": int(
            (
                as_boolean(final["original_strict_recovery"])
                & ~as_boolean(final["final_strict_recovery"])
            ).sum()
        ),
        "srbench_recovery_gains": int(
            (
                ~as_boolean(final["original_srbench_recovery"])
                & as_boolean(final["final_srbench_recovery"])
            ).sum()
        ),
        "srbench_recovery_losses": int(
            (
                as_boolean(final["original_srbench_recovery"])
                & ~as_boolean(final["final_srbench_recovery"])
            ).sum()
        ),
        "unavailable_formula_comparisons": int(unavailable_formula.sum()),
        "selection_uses_test": False,
        "selection_uses_target_formula": False,
    }

    rng = np.random.default_rng(args.seed)
    original_accuracy = final["original_high_accuracy"].astype(float).to_numpy()
    final_accuracy = final["final_high_accuracy"].astype(float).to_numpy()
    original_strict = (
        as_boolean(final["original_strict_recovery"]).astype(float).to_numpy()
    )
    final_strict = (
        as_boolean(final["final_strict_recovery"]).astype(float).to_numpy()
    )
    original_srbench = (
        as_boolean(final["original_srbench_recovery"]).astype(float).to_numpy()
    )
    final_srbench = (
        as_boolean(final["final_srbench_recovery"]).astype(float).to_numpy()
    )
    bootstrap = {
        "resamples": int(args.bootstrap_resamples),
        "seed": int(args.seed),
        "independent_unit": "SRSD benchmark case",
        "high_accuracy_rate_difference": bootstrap_interval(
            original_accuracy,
            final_accuracy,
            np.mean,
            rng,
            args.bootstrap_resamples,
        ),
        "strict_recovery_rate_difference": bootstrap_interval(
            original_strict,
            final_strict,
            np.mean,
            rng,
            args.bootstrap_resamples,
        ),
        "srbench_recovery_rate_difference": bootstrap_interval(
            original_srbench,
            final_srbench,
            np.mean,
            rng,
            args.bootstrap_resamples,
        ),
        "mean_node_count_difference": bootstrap_interval(
            final["original_complexity"].to_numpy(dtype=float),
            final["final_complexity"].to_numpy(dtype=float),
            np.mean,
            rng,
            args.bootstrap_resamples,
        ),
        "median_test_nmse_difference": bootstrap_interval(
            final["original_test_nmse"].to_numpy(dtype=float),
            final["final_test_nmse"].to_numpy(dtype=float),
            np.median,
            rng,
            args.bootstrap_resamples,
        ),
    }

    final.to_csv(
        args.run / "srsd_expression_rescue_conservative_case_rows.csv",
        index=False,
    )
    (args.run / "srsd_expression_rescue_conservative_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.run / "srsd_expression_rescue_conservative_bootstrap.json").write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(bootstrap, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
