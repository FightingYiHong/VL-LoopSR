#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a GPU PhySO symbolic-regression baseline on the benchmark wrappers."""

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import run_cpu_baseline_benchmarks as base


def _sample_xy(X, y, max_rows, random_state):
    if not max_rows or max_rows <= 0 or len(X) <= max_rows:
        return X, y, False
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=int(max_rows), replace=False)
    idx.sort()
    return X[idx], y[idx], True


def _as_numpy_prediction(pred):
    try:
        import torch

        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
    except Exception:
        pass
    pred = np.asarray(pred, dtype=float).reshape(-1)
    if not np.isfinite(pred).all():
        raise ValueError("non-finite predictions")
    return pred


def _expression_strings(expr):
    best_expr = str(expr)
    best_sympy = best_expr
    try:
        best_expr = str(expr.get_infix_str())
    except Exception:
        pass
    try:
        best_sympy = str(expr.get_infix_sympy(do_simplify=True, evaluate_consts=True))
    except Exception:
        best_sympy = best_expr
    return best_expr, best_sympy


def _count_candidates(logs):
    try:
        complexities, expressions, rewards, rmse = logs.get_pareto_front()
        return int(len(expressions))
    except Exception:
        return 1


def fit_physo(train_df, val_df, test_df, args):
    import torch
    import physo
    import physo.learn.monitoring as monitoring

    X_train, y_train, feature_cols = base.dataframe_to_xy(train_df)
    X_val, y_val, _ = base.dataframe_to_xy(val_df)
    X_test, y_test, _ = base.dataframe_to_xy(test_df)
    fit_X, fit_y, train_subsampled = _sample_xy(
        X_train,
        y_train,
        args.max_train_rows,
        args.random_state,
    )

    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    X_names = [f"x{i}" for i in range(len(feature_cols))]
    units = [[0, 0, 0] for _ in X_names]
    run_logger = lambda: monitoring.RunLogger(save_path="physo.log", do_save=False)
    run_visualiser = lambda: monitoring.RunVisualiser(
        epoch_refresh_rate=max(1, args.epochs),
        save_path="physo_curves.png",
        do_show=False,
        do_prints=False,
        do_save=False,
    )

    start = time.time()
    expr, logs = physo.SR(
        fit_X.T,
        fit_y,
        X_names=X_names,
        X_units=units,
        y_name="y",
        y_units=[0, 0, 0],
        fixed_consts=[1.0],
        fixed_consts_units=[[0, 0, 0]],
        free_consts_names=["c0", "c1"],
        free_consts_units=[[0, 0, 0], [0, 0, 0]],
        op_names=args.op_names,
        use_protected_ops=True,
        stop_reward=args.stop_reward,
        max_n_evaluations=args.max_evaluations,
        stop_after_n_epochs=args.stop_after_n_epochs,
        epochs=args.epochs,
        run_config=physo.config.config0.config0,
        get_run_logger=run_logger,
        get_run_visualiser=run_visualiser,
        parallel_mode=False,
        device=args.device,
    )
    fit_sec = time.time() - start

    pred_train = _as_numpy_prediction(expr.execute(torch.tensor(X_train.T, dtype=torch.float32, device=args.device)))
    pred_val = _as_numpy_prediction(expr.execute(torch.tensor(X_val.T, dtype=torch.float32, device=args.device)))
    pred_test = _as_numpy_prediction(expr.execute(torch.tensor(X_test.T, dtype=torch.float32, device=args.device)))
    best_expr, best_sympy = _expression_strings(expr)

    metrics = base._metric_dict(
        "physo",
        best_sympy,
        pred_train,
        pred_val,
        pred_test,
        y_train,
        y_val,
        y_test,
        feature_cols,
        fit_sec,
        num_candidates=_count_candidates(logs),
    )
    metrics["best_expr"] = best_expr
    metrics["best_expr_sympy"] = best_sympy
    metrics.update(
        {
            "fit_n_train": int(len(fit_X)),
            "fit_train_subsampled": bool(train_subsampled),
            "physo_epochs": int(args.epochs),
            "physo_max_train_rows": int(args.max_train_rows),
            "physo_device": args.device,
            "physo_op_names": " | ".join(args.op_names),
        }
    )
    return metrics


def run_one(args, mod, row):
    start = time.time()
    try:
        train_df, val_df, test_df, meta = base.load_case(args.benchmark, mod, row)
        result = fit_physo(train_df, val_df, test_df, args)
        result.update(meta)
        result.update(
            {
                "n_features": int(train_df.shape[1] - 1),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "runtime_sec": float(time.time() - start),
                "error": None,
            }
        )
        return base.json_safe_record(result)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        return base.json_safe_record(base.failure_result(args.benchmark, "physo", row, time.time() - start, repr(exc)))


def terminate_process_group(pgid, grace_sec=10.0):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_isolated_case(args, idx, total, row, results_root):
    case_name = base.row_case_name(args.benchmark, row)
    slug = base.sanitize_name(case_name or f"case_{idx}")
    log_dir = Path(results_root) / "case_logs"
    result_dir = Path(results_root) / "case_results"
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    case_log = log_dir / f"{idx:04d}_{slug}.log"
    result_json = result_dir / f"{idx:04d}_{slug}.json"
    if result_json.exists():
        result_json.unlink()

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--benchmark",
        args.benchmark,
        "--results-root",
        str(results_root),
        "--random-state",
        str(args.random_state),
        "--epochs",
        str(args.epochs),
        "--max-train-rows",
        str(args.max_train_rows),
        "--case-timeout-sec",
        str(args.case_timeout_sec),
        "--device",
        args.device,
        "--single-case-index",
        str(idx),
        "--single-result-json",
        str(result_json),
    ]
    if args.max_cases:
        cmd.extend(["--max-cases", str(args.max_cases)])

    print(f"   isolated_case_log:   {case_log}", flush=True)
    start = time.time()
    with open(case_log, "w", encoding="utf-8", errors="replace") as fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=os.environ.copy(),
            stdout=fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            returncode = proc.wait(timeout=args.case_timeout_sec)
        except subprocess.TimeoutExpired:
            terminate_process_group(proc.pid)
            return base.json_safe_record(
                base.failure_result(
                    args.benchmark,
                    "physo",
                    row,
                    time.time() - start,
                    f"isolated child exceeded timeout {args.case_timeout_sec:.1f}s; log_tail={base.read_log_tail(case_log)}",
                )
            )

    if result_json.exists():
        with open(result_json, "r", encoding="utf-8") as fp:
            res = json.load(fp)
        if returncode != 0 and not res.get("error"):
            res["error"] = f"isolated child returned non-zero status {returncode}"
    else:
        res = base.failure_result(
            args.benchmark,
            "physo",
            row,
            time.time() - start,
            f"isolated child failed without result json; returncode={returncode}; log_tail={base.read_log_tail(case_log)}",
        )
    res["completed_index"] = int(idx)
    res["total_tasks"] = int(total)
    res["case_log_path"] = str(case_log)
    return base.json_safe_record(res)


def write_results_csv(results, path):
    if not results:
        return
    keys = []
    for row in results:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=["sldbench", "llmsrbench", "srbench", "srsd"])
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=1000)
    parser.add_argument("--case-timeout-sec", type=float, default=1200.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--op-names", nargs="+", default=["add", "sub", "mul", "div", "sin", "cos", "exp", "log"])
    parser.add_argument("--stop-reward", type=float, default=1.0)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--stop-after-n-epochs", type=int, default=8)
    parser.add_argument("--single-case-index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-result-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    mod, df_tasks = base.collect_tasks(args.benchmark)
    if args.max_cases and args.max_cases > 0:
        df_tasks = df_tasks.head(args.max_cases).reset_index(drop=True)
    total = len(df_tasks)
    if total == 0:
        raise RuntimeError(f"no tasks collected for {args.benchmark}")

    if args.single_case_index is not None:
        idx = int(args.single_case_index)
        row = df_tasks.iloc[idx - 1].to_dict()
        print(f"[{idx}/{total}] Starting PhySO: {base.row_case_name(args.benchmark, row)}", flush=True)
        res = run_one(args, mod, row)
        res["completed_index"] = int(idx)
        res["total_tasks"] = int(total)
        base.print_one_result(args.benchmark, idx, total, row, res)
        if args.single_result_json:
            with open(args.single_result_json, "w", encoding="utf-8") as fp:
                json.dump(res, fp, ensure_ascii=False, allow_nan=False, indent=2)
        return

    results = []
    csv_path = results_root / f"all_{args.benchmark}_physo_results.csv"
    for idx, (_, row_obj) in enumerate(df_tasks.iterrows(), start=1):
        row = row_obj.to_dict()
        print(f"[{idx}/{total}] Starting isolated PhySO: {base.row_case_name(args.benchmark, row)}", flush=True)
        res = run_isolated_case(args, idx, total, row, results_root)
        results.append(res)
        base.print_one_result(args.benchmark, idx, total, row, res)
        write_results_csv(results, csv_path)
        print(f"   results_csv:         {csv_path}", flush=True)

    summary_path = results_root / f"summary_{args.benchmark}_physo.json"
    with open(summary_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "benchmark": args.benchmark,
                "method": "physo",
                "total": total,
                "completed": len(results),
                "valid": sum(bool(r.get("valid_formula_found")) for r in results),
                "passed": sum(bool(r.get("passed")) for r in results),
                "perfect_fit_by_r2": sum(bool(r.get("perfect_fit_by_r2")) for r in results),
            },
            fp,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
    print(f"summary_json: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
