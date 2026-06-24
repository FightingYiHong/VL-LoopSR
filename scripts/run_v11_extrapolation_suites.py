#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run V11 on controlled and public extrapolation splits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_strict_extrapolation_cpu_baselines import json_safe, make_split
from run_surfacebench_strict_cpu_baselines import list_surface_cases, load_surface_split


DEFAULT_V11_PATH = ROOT / "scripts" / "test_fey_v11_complexity_exact.py"
DEFAULT_SURFACEBENCH_PATH = ROOT / "data" / "surfacebench_public" / "dataset.h5"


def sanitize_name(text: str) -> str:
    text = str(text).replace("/", "__").replace("\\", "__")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "case"


def expr_complexity(expr) -> float | None:
    if not isinstance(expr, str) or not expr.strip():
        return None
    try:
        import sympy as sp

        parsed = sp.sympify(expr)
        return float(sp.count_ops(parsed, visual=False) + len(parsed.free_symbols))
    except Exception:
        tokens = [tok for tok in re.split(r"[^A-Za-z0-9_.]+", expr) if tok]
        return float(len(tokens)) if tokens else None


def import_v11_module(v11_path: Path, results_root: Path):
    if not v11_path.exists():
        raise FileNotFoundError(f"V11 file not found: {v11_path}")

    spec = importlib.util.spec_from_file_location("v11_extrapolation_runtime", str(v11_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["v11_extrapolation_runtime"] = module
    spec.loader.exec_module(module)

    module.RESULTS_ROOT = str(results_root)
    module.GLOBAL_SUMMARY_CSV = str(results_root / "all_results_detailed.csv")
    module.GLOBAL_SUMMARY_JSON = str(results_root / "global_summary.json")
    module.GLOBAL_SUMMARY_CSV_COMPACT = str(results_root / "global_summary.csv")
    module.TIMING_BREAKDOWN_CSV = str(results_root / "timing_breakdown.csv")
    module.TIMING_SUMMARY_JSON = str(results_root / "timing_summary.json")
    module.PER_CASE_JSON_DIR = str(results_root / "per_case_reports")
    module.SELECTED_TASKS_CSV = str(results_root / "selected_cases.csv")
    if os.environ.get("LLMSR_MAX_RUNTIME_PER_TASK_SEC"):
        module.MAX_RUNTIME_PER_TASK_SEC = float(os.environ["LLMSR_MAX_RUNTIME_PER_TASK_SEC"])

    base = getattr(module, "_base", None)
    if base is not None:
        base.RESULTS_ROOT = module.RESULTS_ROOT
        base.GLOBAL_SUMMARY_CSV = module.GLOBAL_SUMMARY_CSV
        base.GLOBAL_SUMMARY_JSON = module.GLOBAL_SUMMARY_JSON
        base.GLOBAL_SUMMARY_CSV_COMPACT = module.GLOBAL_SUMMARY_CSV_COMPACT
        base.TIMING_BREAKDOWN_CSV = module.TIMING_BREAKDOWN_CSV
        base.TIMING_SUMMARY_JSON = module.TIMING_SUMMARY_JSON
        base.PER_CASE_JSON_DIR = module.PER_CASE_JSON_DIR
        base.SELECTED_TASKS_CSV = module.SELECTED_TASKS_CSV
        if os.environ.get("LLMSR_MAX_RUNTIME_PER_TASK_SEC"):
            base.MAX_RUNTIME_PER_TASK_SEC = float(os.environ["LLMSR_MAX_RUNTIME_PER_TASK_SEC"])

    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "per_case_reports").mkdir(parents=True, exist_ok=True)
    return module


def controlled_cases(max_cases: int | None):
    from run_strict_extrapolation_cpu_baselines import make_extrapolation_cases

    cases = make_extrapolation_cases()
    if max_cases:
        cases = cases[:max_cases]
    out = []
    for idx, case in enumerate(cases, start=1):
        out.append(
            {
                "suite": "constructed_ood",
                "case_index": idx,
                "case_name": case.case_name,
                "benchmark": case.benchmark,
                "structure_type": case.structure_type,
                "train_range": case.train_range_label,
                "test_range": case.test_range_label,
            }
        )
    return out


def surface_cases(path: Path, max_cases: int | None):
    cases = list_surface_cases(path, max_cases=max_cases)
    out = []
    for idx, case in enumerate(cases, start=1):
        row = dict(case)
        row.update(
            {
                "suite": "surfacebench_public_ood",
                "case_index": idx,
                "benchmark": "SurfaceBench",
                "structure_type": "",
                "train_range": "SurfaceBench train_data",
                "test_range": "SurfaceBench ood_test",
            }
        )
        out.append(row)
    return out


def load_case_split(args):
    if args.suite == "constructed":
        _, train_df, val_df, test_df, meta = make_split(
            args.case_index - 1,
            args.repeat_seed,
            args.n_train,
            args.n_val,
            args.n_test,
        )
        row_meta = {
            "task_type": "constructed_extrapolation",
            "dataset_dir": "constructed_extrapolation",
            "difficulty": meta["benchmark"],
            "base_name": meta["case_name"],
            "true_expression": None,
            **meta,
        }
        return train_df, val_df, test_df, row_meta

    cases = list_surface_cases(Path(args.surfacebench_path), max_cases=args.max_cases)
    case = cases[args.case_index - 1]
    train_df, val_df, test_df, meta = load_surface_split(
        Path(args.surfacebench_path),
        case,
        args.n_train,
        args.n_val,
        args.n_test,
        args.random_state + args.case_index * 100 + args.repeat_seed,
    )
    row_meta = {
        "task_type": "surfacebench_public_ood",
        "dataset_dir": "surfacebench_public_ood",
        "difficulty": meta["category"],
        "base_name": meta["case_name"],
        "true_expression": None,
        **meta,
    }
    return train_df, val_df, test_df, row_meta


def run_child(args) -> int:
    started = time.time()
    result_path = Path(args.single_result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        v11 = import_v11_module(Path(args.v11_path), Path(args.out_dir))
        train_df, val_df, test_df, row_meta = load_case_split(args)
        with TemporaryDirectory(prefix="v11_extrap_tmp_") as tmpdir_str:
            dataset = v11.build_dataset_from_explicit_splits(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                tmpdir=Path(tmpdir_str),
            )
            dataset.source_tag = row_meta["dataset_dir"]
            out = v11._run_core_pipeline(dataset=dataset, row_meta=row_meta)
        out.update(row_meta)
        out.update(
            {
                "method": "vl_loopsr",
                "suite": args.suite,
                "case_index": int(args.case_index),
                "repeat_seed": int(args.repeat_seed),
                "budget_sec": float(args.case_budget_sec),
                "timed_out": False,
                "runtime_sec": float(out.get("runtime_sec") or (time.time() - started)),
                "train_mse": out.get("best_train_mse"),
                "id_mse": out.get("best_val_mse"),
                "ood_mse": out.get("best_test_mse"),
                "extrapolation_mse": out.get("best_test_mse"),
                "extrapolation_r2": out.get("test_r2"),
            }
        )
        if out.get("expr_complexity") is None:
            out["expr_complexity"] = expr_complexity(out.get("best_expr"))
    except BaseException as exc:
        out = {
            "method": "vl_loopsr",
            "suite": args.suite,
            "case_index": int(args.case_index),
            "repeat_seed": int(args.repeat_seed),
            "budget_sec": float(args.case_budget_sec),
            "timed_out": "time" in repr(exc).lower() or "timeout" in repr(exc).lower(),
            "runtime_sec": float(time.time() - started),
            "valid_formula_found": False,
            "passed": False,
            "best_expr": None,
            "best_val_mse": None,
            "best_test_mse": None,
            "ood_mse": None,
            "extrapolation_mse": None,
            "extrapolation_r2": None,
            "error": repr(exc),
        }
    result_path.write_text(json.dumps(json_safe(out), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def timeout_row(args, case: dict, runtime_sec: float, log_path: Path, parent_timeout_sec: float | None = None):
    outer_timeout = float(parent_timeout_sec) if parent_timeout_sec else float(args.case_budget_sec) + float(args.timeout_grace_sec)
    return {
        "method": "vl_loopsr",
        "suite": args.suite,
        "case_index": int(case["case_index"]),
        "case_name": case.get("case_name"),
        "benchmark": case.get("benchmark"),
        "structure_type": case.get("structure_type"),
        "repeat_seed": int(args.repeat_seed),
        "budget_sec": float(args.case_budget_sec),
        "timed_out": True,
        "runtime_sec": float(runtime_sec),
        "valid_formula_found": False,
        "passed": False,
        "best_expr": None,
        "best_val_mse": None,
        "best_test_mse": None,
        "ood_mse": None,
        "extrapolation_mse": None,
        "extrapolation_r2": None,
        "case_log_path": str(log_path),
        "error": f"case exceeded outer timeout {outer_timeout:.1f}s",
    }


def summarize(rows: list[dict], out_dir: Path, suite: str):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    for column in [
        "timed_out",
        "passed",
        "best_val_mse",
        "best_test_mse",
        "ood_mse",
        "extrapolation_mse",
        "extrapolation_r2",
        "runtime_sec",
        "expr_complexity",
    ]:
        if column not in df:
            df[column] = np.nan
    df.to_csv(out_dir / f"all_{suite}_vl_loopsr_strict_results.csv", index=False)
    group_cols = ["benchmark", "method"] if suite == "constructed" else ["difficulty", "method"]
    for col in group_cols:
        if col not in df:
            df[col] = ""
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("case_index", "count"),
            timeout_rate=("timed_out", "mean"),
            pass_rate=("passed", "mean"),
            median_id_mse=("best_val_mse", "median"),
            median_ood_mse=("best_test_mse", "median"),
            median_ood_r2=("extrapolation_r2", "median"),
            median_runtime=("runtime_sec", "median"),
            median_complexity=("expr_complexity", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / f"summary_{suite}_vl_loopsr_strict.csv", index=False)


def run_parent(args) -> int:
    out_dir = Path(args.out_dir)
    result_dir = out_dir / "case_results" / "vl_loopsr"
    log_dir = out_dir / "case_logs" / "vl_loopsr"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    parent_timeout_sec = float(args.parent_timeout_sec or 0.0)
    if parent_timeout_sec <= 0:
        parent_timeout_sec = float(args.case_budget_sec) + float(args.timeout_grace_sec)

    cases = controlled_cases(args.max_cases) if args.suite == "constructed" else surface_cases(Path(args.surfacebench_path), args.max_cases)
    pd.DataFrame(cases).to_csv(out_dir / "selected_cases.csv", index=False)
    manifest = {
        "method": "vl_loopsr",
        "suite": args.suite,
        "n_cases": len(cases),
        "case_budget_sec": float(args.case_budget_sec),
        "timeout_grace_sec": float(args.timeout_grace_sec),
        "parent_timeout_sec": float(parent_timeout_sec),
        "v11_path": str(args.v11_path),
        "surfacebench_path": str(args.surfacebench_path) if args.suite == "surfacebench" else None,
    }
    (out_dir / "manifest_vl_loopsr.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for case in cases:
        idx = int(case["case_index"])
        safe = sanitize_name(case.get("case_name", f"case_{idx:03d}"))
        result_json = result_dir / f"{idx:03d}_{safe}_seed{args.repeat_seed}.json"
        log_path = log_dir / f"{idx:03d}_{safe}_seed{args.repeat_seed}.log"
        if args.resume and result_json.exists():
            cached = json.loads(result_json.read_text(encoding="utf-8"))
            should_rerun = False
            if args.rerun_timeouts and bool(cached.get("timed_out")):
                should_rerun = True
            if args.rerun_failures and not bool(cached.get("valid_formula_found")):
                should_rerun = True
            if not should_rerun:
                rows.append(cached)
                continue

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--suite",
            args.suite,
            "--out-dir",
            str(out_dir),
            "--v11-path",
            str(args.v11_path),
            "--case-index",
            str(idx),
            "--repeat-seed",
            str(args.repeat_seed),
            "--n-train",
            str(args.n_train),
            "--n-val",
            str(args.n_val),
            "--n-test",
            str(args.n_test),
            "--case-budget-sec",
            str(args.case_budget_sec),
            "--timeout-grace-sec",
            str(args.timeout_grace_sec),
            "--max-cases",
            str(args.max_cases or 0),
            "--random-state",
            str(args.random_state),
            "--surfacebench-path",
            str(args.surfacebench_path),
            "--single-result-json",
            str(result_json),
        ]
        started = time.time()
        print(f"[RUN {idx}/{len(cases)}] vl_loopsr {case.get('case_name')}", flush=True)
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUNBUFFERED", "1")
        with open(log_path, "w", encoding="utf-8") as log_fp:
            try:
                subprocess.run(
                    cmd,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    timeout=parent_timeout_sec,
                    check=False,
                    env=child_env,
                )
            except subprocess.TimeoutExpired:
                rows.append(timeout_row(args, case, time.time() - started, log_path, parent_timeout_sec))
                result_json.write_text(json.dumps(json_safe(rows[-1]), ensure_ascii=False, indent=2), encoding="utf-8")
                summarize(rows, out_dir, args.suite)
                print(f"[TIMEOUT {idx}/{len(cases)}] sec={time.time() - started:.1f}", flush=True)
                continue
        if result_json.exists():
            row = json.loads(result_json.read_text(encoding="utf-8"))
        else:
            row = {
                **timeout_row(args, case, time.time() - started, log_path, parent_timeout_sec),
                "timed_out": False,
                "error": f"child exited without result; see {log_path}",
            }
        rows.append(row)
        summarize(rows, out_dir, args.suite)
        print(
            f"[DONE {idx}/{len(cases)}] timeout={row.get('timed_out')} "
            f"mse={row.get('best_test_mse')} sec={row.get('runtime_sec')}",
            flush=True,
        )

    summarize(rows, out_dir, args.suite)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["constructed", "surfacebench"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--v11-path", default=str(DEFAULT_V11_PATH))
    parser.add_argument("--surfacebench-path", default=str(DEFAULT_SURFACEBENCH_PATH))
    parser.add_argument("--case-budget-sec", type=float, default=100.0)
    parser.add_argument("--timeout-grace-sec", type=float, default=15.0)
    parser.add_argument("--parent-timeout-sec", type=float, default=0.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-val", type=int, default=128)
    parser.add_argument("--n-test", type=int, default=512)
    parser.add_argument("--repeat-seed", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-timeouts", action="store_true")
    parser.add_argument("--rerun-failures", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--single-result-json", default="")
    args = parser.parse_args()
    if args.max_cases <= 0:
        args.max_cases = None
    return args


def main():
    args = parse_args()
    os.environ["LLMSR_MAX_RUNTIME_PER_TASK_SEC"] = str(args.case_budget_sec)
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
