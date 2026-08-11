#!/usr/bin/env python3
"""Backfill missing per-case JSON records from CPU-baseline aggregate CSVs."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def json_value(value):
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    created = 0
    existing = 0
    unusable = 0
    for method in args.methods:
        for csv_path in sorted((args.run_root / method).glob("*/all_*_results.csv")):
            try:
                frame = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                continue
            for raw in frame.to_dict(orient="records"):
                log_path = raw.get("case_log_path")
                if not isinstance(log_path, str) or "/case_logs/" not in log_path:
                    unusable += 1
                    continue
                result_path = Path(
                    log_path.replace("/case_logs/", "/case_results/")
                ).with_suffix(".json")
                if result_path.exists():
                    existing += 1
                    continue
                record = {key: json_value(value) for key, value in raw.items()}
                if args.write:
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = result_path.with_suffix(result_path.suffix + ".tmp")
                    with open(tmp_path, "w", encoding="utf-8") as fp:
                        json.dump(
                            record,
                            fp,
                            ensure_ascii=False,
                            allow_nan=False,
                            indent=2,
                        )
                    os.replace(tmp_path, result_path)
                created += 1
    action = "created" if args.write else "would_create"
    print(f"{action}={created} existing={existing} unusable={unusable}")


if __name__ == "__main__":
    main()
