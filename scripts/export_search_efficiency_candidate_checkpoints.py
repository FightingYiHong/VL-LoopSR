#!/usr/bin/env python3
"""Export cumulative high-accuracy recovery at every candidate checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    args = parser.parse_args()

    cases = pd.read_csv(args.case_csv)
    required = {
        "method_order",
        "method",
        "method_color",
        "availability_status",
        "first_hit_expression_count",
    }
    missing = required.difference(cases.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    rng = np.random.default_rng(20260730)
    for method_order, method_frame in cases.groupby("method_order", sort=True):
        method = str(method_frame["method"].iloc[0])
        color = str(method_frame["method_color"].iloc[0])
        denominator = len(method_frame)
        available = int(method_frame["availability_status"].eq("available").sum())
        unavailable = denominator - available
        complete = unavailable == 0
        first_hit = pd.to_numeric(
            method_frame["first_hit_expression_count"],
            errors="coerce",
        )
        for k in range(1, args.max_candidates + 1):
            success = first_hit.le(k).to_numpy(dtype=float)
            count = int(success.sum())
            if complete:
                low, high = bootstrap_ci(success, rng)
            else:
                low, high = math.nan, math.nan
            rows.append({
                "method_order": int(method_order),
                "method": method,
                "method_color": color,
                "candidate_checkpoint_k": k,
                "checkpoint_definition": (
                    "within first K evaluated expressions, validation R^2 > 0.999"
                ),
                "denominator_total_cases": denominator,
                "available_result_records": available,
                "unavailable_result_records": unavailable,
                "snapshot_complete": complete,
                "cumulative_high_accuracy_successes": count,
                "cumulative_recovery_rate": count / denominator,
                "cumulative_recovery_rate_percent": 100 * count / denominator,
                "bootstrap_95ci_low": low,
                "bootstrap_95ci_high": high,
                "bootstrap_95ci_low_percent": 100 * low if math.isfinite(low) else math.nan,
                "bootstrap_95ci_high_percent": 100 * high if math.isfinite(high) else math.nan,
                "is_multiple_of_3": k % 3 == 0,
                "is_multiple_of_4": k % 4 == 0,
                "is_multiple_of_5": k % 5 == 0,
                "rate_denominator_note": (
                    "All 895 non-SLDBench cases; unavailable ICSR cases remain in the denominator."
                ),
            })

    output = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(output.to_string(index=False))
    print(f"\nWrote {args.output_csv}")


if __name__ == "__main__":
    main()
