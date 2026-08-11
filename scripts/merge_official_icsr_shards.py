#!/usr/bin/env python3
"""Audit and merge non-overlapping official-ICSR execution shards."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


EXPECTED_CASES = {"srbench": 417, "srsd": 238}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmarks", default="srbench,srsd")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write an explicitly partial merge instead of requiring full coverage.",
    )
    return parser.parse_args()


def merge_benchmark(
    shards_root: Path,
    output_root: Path,
    benchmark: str,
    *,
    allow_incomplete: bool,
) -> dict[str, object]:
    sources = sorted(shards_root.glob(f"*/icsr/{benchmark}/case_results/*.json"))
    rows: dict[int, tuple[dict, Path]] = {}
    for source in sources:
        result = json.loads(source.read_text(encoding="utf-8"))
        index = int(result["case_index"])
        if result.get("benchmark") != benchmark:
            raise ValueError(f"Benchmark mismatch in {source}")
        if index in rows:
            previous, previous_path = rows[index]
            if result != previous:
                raise ValueError(
                    f"Conflicting duplicate case {benchmark}:{index}: "
                    f"{previous_path} and {source}"
                )
            continue
        rows[index] = (result, source)

    expected = EXPECTED_CASES[benchmark]
    expected_indices = set(range(1, expected + 1))
    observed_indices = set(rows)
    missing = sorted(expected_indices - observed_indices)
    extra = sorted(observed_indices - expected_indices)
    if extra:
        raise ValueError(f"Out-of-range {benchmark} case indices: {extra}")
    if missing and not allow_incomplete:
        raise ValueError(
            f"Incomplete {benchmark} shards: {len(missing)} missing cases; "
            f"first missing indices are {missing[:20]}"
        )

    benchmark_root = output_root / benchmark
    case_root = benchmark_root / "case_results"
    case_root.mkdir(parents=True, exist_ok=True)
    merged_rows: list[dict] = []
    for index in sorted(rows):
        result, source = rows[index]
        destination = case_root / source.name
        shutil.copy2(source, destination)
        merged_rows.append(result)

    frame = pd.DataFrame(merged_rows)
    csv_path = benchmark_root / f"all_{benchmark}_official_icsr_results.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    expression = (
        frame["expression"].fillna("").astype(str).str.strip()
        if "expression" in frame
        else pd.Series(dtype=str)
    )
    test_mse = (
        pd.to_numeric(frame["test_mse"], errors="coerce")
        if "test_mse" in frame
        else pd.Series(dtype=float)
    )
    summary = {
        "benchmark": benchmark,
        "expected_cases": expected,
        "observed_cases": len(frame),
        "missing_case_indices": missing,
        "complete": not missing,
        "n_with_final_expression": int(expression.ne("").sum()),
        "n_with_finite_test_mse": int(test_mse.notna().sum()),
        "source_shards_root": str(shards_root),
        "result_csv": str(csv_path),
    }
    (benchmark_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for benchmark in [
        value.strip() for value in args.benchmarks.split(",") if value.strip()
    ]:
        if benchmark not in EXPECTED_CASES:
            raise ValueError(f"Unsupported benchmark: {benchmark}")
        summaries[benchmark] = merge_benchmark(
            args.shards_root,
            args.output_root,
            benchmark,
            allow_incomplete=args.allow_incomplete,
        )
    (args.output_root / "merge_manifest.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
