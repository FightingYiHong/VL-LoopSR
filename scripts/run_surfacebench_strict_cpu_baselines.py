#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run strict-budget CPU SR baselines on public SurfaceBench OOD splits."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_cpu_baseline_benchmarks as cpu_sr
from run_strict_extrapolation_cpu_baselines import DEFAULT_CONFIGS, json_safe, prepare_budget_config


PREFERRED_CATEGORIES = [
    "Complex_Composite_Surfaces",
    "Hybrid_Multi-Modal_Symbolic_Surfaces",
    "Symbolic-Numeric_Composite_Surfaces",
    "Nonlinear_Dynamical_System_Surfaces",
    "Quantum_Inspired_Surfaces",
    "Algebraic_Manifolds_of_Higher_Degree",
    "Non-Canonical_3D_Geometric_Surfaces",
    "Piecewise_Regime_Surfaces",
    "Procedural_Fractal_Surfaces",
    "Surrogate_Distilled_Symbolic_Approximations",
]


def dataframe_from_surface_array(arr: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"expected surface array shape (n, >=3), got {arr.shape}")
    return pd.DataFrame({"x0": arr[:, 0], "x1": arr[:, 1], "y": arr[:, 2]})


def dataset_key(group, candidates):
    for key in candidates:
        if key in group and hasattr(group[key], "shape"):
            return key
    return None


def list_surface_cases(path: Path, max_cases: int | None = None, categories: list[str] | None = None) -> list[dict]:
    categories = categories or PREFERRED_CATEGORIES
    out: list[dict] = []
    with h5py.File(path, "r") as h5:
        for category in categories:
            if category not in h5:
                continue
            for instance in sorted(h5[category].keys()):
                group = h5[category][instance]
                train_key = dataset_key(group, ["train_data"])
                id_key = dataset_key(group, ["test_data"])
                ood_key = dataset_key(group, ["ood_test", "ood_test_data", "ood_test_eval"])
                if not train_key or not id_key or not ood_key:
                    continue
                if h5[f"{category}/{instance}/{train_key}"].shape[1] != 3:
                    continue
                if h5[f"{category}/{instance}/{ood_key}"].shape[1] != 3:
                    continue
                out.append(
                    {
                        "category": category,
                        "instance": instance,
                        "train_key": train_key,
                        "id_key": id_key,
                        "ood_key": ood_key,
                        "case_name": f"{category}/{instance}",
                    }
                )
                if max_cases and len(out) >= max_cases:
                    return out
    return out


def load_surface_split(path: Path, case: dict, n_train: int, n_val: int, n_test: int, seed: int):
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as h5:
        group = h5[f"{case['category']}/{case['instance']}"]
        train = np.asarray(group[case["train_key"]][:], dtype=float)
        val = np.asarray(group[case["id_key"]][:], dtype=float)
        test = np.asarray(group[case["ood_key"]][:], dtype=float)

    def subsample(arr, n):
        if n <= 0 or len(arr) <= n:
            return arr
        idx = rng.choice(len(arr), size=n, replace=False)
        return arr[np.sort(idx)]

    train_df = dataframe_from_surface_array(subsample(train, n_train))
    val_df = dataframe_from_surface_array(subsample(val, n_val))
    test_df = dataframe_from_surface_array(subsample(test, n_test))
    meta = {
        "experiment": "public_surfacebench_ood",
        "benchmark": "SurfaceBench",
        "case_name": case["case_name"],
        "category": case["category"],
        "instance": case["instance"],
        "structure_type": category_to_structure(case["category"]),
        "n_features": 2,
        "train_range": "SurfaceBench train_data",
        "test_range": "SurfaceBench ood_test",
        "distance_to_train_range": 1.0,
        "n_train_source": int(len(train)),
        "n_val_source": int(len(val)),
        "n_test_source": int(len(test)),
    }
    return train_df, val_df, test_df, meta


def category_to_structure(category: str) -> str:
    text = category.lower()
    if "composite" in text or "morphological" in text or "coordinate" in text:
        return "function_composition"
    if "trig" in text or "oscillatory" in text:
        return "trigonometric"
    if "algebraic" in text or "polynomial" in text:
        return "polynomial"
    if "dynamical" in text or "quantum" in text or "nonlinear" in text:
        return "variable_interaction"
    if "piecewise" in text:
        return "piecewise"
    return "surface"


def run_child(args) -> int:
    cases = list_surface_cases(Path(args.surfacebench_path), max_cases=args.max_cases, categories=args.categories_list)
    case = cases[args.case_index - 1]
    source_config = args.config or str(DEFAULT_CONFIGS[args.method])
    config = prepare_budget_config(args.method, source_config, args.case_budget_sec, args.single_result_json)
    started = time.time()
    try:
        train_df, val_df, test_df, meta = load_surface_split(
            Path(args.surfacebench_path),
            case,
            args.n_train,
            args.n_val,
            args.n_test,
            args.random_state + args.case_index * 100 + args.repeat_seed,
        )
        result = cpu_sr.METHOD_FITTERS[args.method](
            train_df,
            val_df,
            test_df,
            config_path=str(config),
            random_state=args.random_state + args.case_index * 100 + args.repeat_seed,
        )
        result.update(meta)
        result.update(
            {
                "method": args.method,
                "case_index": int(args.case_index),
                "repeat_seed": int(args.repeat_seed),
                "budget_sec": float(args.case_budget_sec),
                "timed_out": False,
                "runtime_sec": float(time.time() - started),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
                "n_test": int(len(test_df)),
                "train_mse": result.get("best_train_mse"),
                "id_mse": result.get("best_val_mse"),
                "ood_mse": result.get("best_test_mse"),
                "extrapolation_mse": result.get("best_test_mse"),
                "extrapolation_r2": result.get("test_r2"),
                "config_path": str(config),
                "source_config_path": str(source_config),
                "surfacebench_path": str(args.surfacebench_path),
                "error": result.get("error"),
            }
        )
    except BaseException as exc:
        timed_out = isinstance(exc, TimeoutError) or "timeout" in repr(exc).lower() or "exceeded" in repr(exc).lower()
        result = {
            "experiment": "public_surfacebench_ood",
            "benchmark": "SurfaceBench",
            "case_name": case["case_name"],
            "category": case["category"],
            "instance": case["instance"],
            "method": args.method,
            "case_index": int(args.case_index),
            "repeat_seed": int(args.repeat_seed),
            "budget_sec": float(args.case_budget_sec),
            "timed_out": bool(timed_out),
            "runtime_sec": float(time.time() - started),
            "valid_formula_found": False,
            "passed": False,
            "best_expr": "",
            "best_test_mse": None,
            "test_r2": None,
            "train_mse": None,
            "id_mse": None,
            "ood_mse": None,
            "extrapolation_mse": None,
            "extrapolation_r2": None,
            "config_path": str(config),
            "source_config_path": str(source_config),
            "surfacebench_path": str(args.surfacebench_path),
            "error": repr(exc),
        }
    Path(args.single_result_json).write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def terminate_group(pid: int):
    try:
        os.killpg(pid, signal.SIGTERM)
    except Exception:
        return
    time.sleep(2)
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        pass


def timeout_result(args, case: dict, case_idx: int, repeat_seed: int, runtime_sec: float, log_path: Path):
    return {
        "experiment": "public_surfacebench_ood",
        "benchmark": "SurfaceBench",
        "case_name": case["case_name"],
        "category": case["category"],
        "instance": case["instance"],
        "method": args.method,
        "case_index": int(case_idx),
        "repeat_seed": int(repeat_seed),
        "budget_sec": float(args.case_budget_sec),
        "timed_out": True,
        "runtime_sec": float(runtime_sec),
        "valid_formula_found": False,
        "passed": False,
        "best_expr": "",
        "best_test_mse": None,
        "test_r2": None,
        "train_mse": None,
        "id_mse": None,
        "ood_mse": None,
        "extrapolation_mse": None,
        "extrapolation_r2": None,
        "case_log_path": str(log_path),
        "surfacebench_path": str(args.surfacebench_path),
        "error": f"case exceeded strict budget of {args.case_budget_sec}s",
    }


def summarize(rows: list[dict], out_dir: Path, method: str):
    df = pd.DataFrame(rows)
    for column in ["timed_out", "passed", "train_mse", "id_mse", "ood_mse", "extrapolation_mse", "extrapolation_r2", "runtime_sec", "expr_complexity"]:
        if column not in df:
            df[column] = np.nan
    df.to_csv(out_dir / f"all_surfacebench_{method}_strict100_results.csv", index=False)
    summary = (
        df.groupby(["category", "method"], dropna=False)
        .agg(
            n=("case_name", "count"),
            timeout_rate=("timed_out", "mean"),
            pass_rate=("passed", "mean"),
            median_train_mse=("train_mse", "median"),
            median_id_mse=("id_mse", "median"),
            median_ood_mse=("ood_mse", "median"),
            median_ood_r2=("extrapolation_r2", "median"),
            median_runtime=("runtime_sec", "median"),
            median_complexity=("expr_complexity", "median"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / f"summary_surfacebench_{method}_strict100.csv", index=False)


def run_parent(args) -> int:
    path = Path(args.surfacebench_path)
    cases = list_surface_cases(path, max_cases=args.max_cases, categories=args.categories_list)
    if not cases:
        raise RuntimeError(f"no usable SurfaceBench cases found in {path}")
    out_dir = Path(args.out_dir)
    log_dir = out_dir / "case_logs" / args.method
    result_dir = out_dir / "case_results" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cases).to_csv(out_dir / "surfacebench_selected_cases.csv", index=False)
    manifest = {
        "method": args.method,
        "dataset": "pandoradox/symbolic-regression-surfaces",
        "surfacebench_path": str(path),
        "n_cases": len(cases),
        "case_budget_sec": args.case_budget_sec,
        "repeat_seeds": args.repeat_seeds,
        "n_train": args.n_train,
        "n_val": args.n_val,
        "n_test": args.n_test,
        "categories": args.categories_list,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / f"manifest_surfacebench_{args.method}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict] = []
    total = len(cases) * args.repeat_seeds
    done = 0
    for case_idx, case in enumerate(cases, start=1):
        for repeat_seed in range(args.repeat_seeds):
            done += 1
            slug = f"{case_idx:03d}_{case['category']}__{case['instance']}_seed{repeat_seed}".replace("/", "__")
            result_json = result_dir / f"{slug}.json"
            log_path = log_dir / f"{slug}.log"
            if args.resume and result_json.exists():
                row = json.loads(result_json.read_text(encoding="utf-8"))
                rows.append(row)
                summarize(rows, out_dir, args.method)
                print(f"[SKIP {done}/{total}] {case['case_name']} seed={repeat_seed}", flush=True)
                continue
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--method",
                args.method,
                "--surfacebench-path",
                str(path),
                "--out-dir",
                str(out_dir),
                "--case-index",
                str(case_idx),
                "--repeat-seed",
                str(repeat_seed),
                "--max-cases",
                str(args.max_cases or 0),
                "--categories",
                args.categories,
                "--n-train",
                str(args.n_train),
                "--n-val",
                str(args.n_val),
                "--n-test",
                str(args.n_test),
                "--case-budget-sec",
                str(args.case_budget_sec),
                "--random-state",
                str(args.random_state),
                "--single-result-json",
                str(result_json),
            ]
            if args.config:
                cmd.extend(["--config", str(args.config)])
            started = time.time()
            print(f"[RUN {done}/{total}] {args.method} {case['case_name']} seed={repeat_seed}", flush=True)
            with open(log_path, "w", encoding="utf-8", errors="replace") as fp:
                proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fp, stderr=subprocess.STDOUT, text=True, start_new_session=True)
                try:
                    parent_timeout = float(args.case_budget_sec) + float(args.parent_timeout_grace_sec)
                    if args.method == "pysr":
                        parent_timeout += float(args.pysr_extra_grace_sec)
                    rc = proc.wait(timeout=parent_timeout)
                except subprocess.TimeoutExpired:
                    terminate_group(proc.pid)
                    row = timeout_result(args, case, case_idx, repeat_seed, time.time() - started, log_path)
                    result_json.write_text(json.dumps(json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    if result_json.exists():
                        row = json.loads(result_json.read_text(encoding="utf-8"))
                    else:
                        row = timeout_result(args, case, case_idx, repeat_seed, time.time() - started, log_path)
                        row["timed_out"] = False
                        row["error"] = f"child exited rc={rc} without result json"
                    row["return_code"] = int(rc)
                    row["case_log_path"] = str(log_path)
                    result_json.write_text(json.dumps(json_safe(row), ensure_ascii=False, indent=2), encoding="utf-8")
            rows.append(row)
            summarize(rows, out_dir, args.method)
            print(
                f"[DONE {done}/{total}] timeout={row.get('timed_out')} ood_mse={row.get('ood_mse')} "
                f"ood_r2={row.get('extrapolation_r2')} sec={row.get('runtime_sec'):.1f}",
                flush=True,
            )
    return 0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=sorted(DEFAULT_CONFIGS))
    parser.add_argument("--surfacebench-path", default=str(ROOT / "data" / "surfacebench_public" / "dataset.h5"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--case-budget-sec", type=float, default=100.0)
    parser.add_argument("--parent-timeout-grace-sec", type=float, default=30.0)
    parser.add_argument("--pysr-extra-grace-sec", type=float, default=45.0)
    parser.add_argument("--repeat-seeds", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--categories", default=",".join(PREFERRED_CATEGORIES[:6]))
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--repeat-seed", type=int, default=0)
    parser.add_argument("--single-result-json", default=None)
    args = parser.parse_args()
    args.categories_list = [item.strip() for item in args.categories.split(",") if item.strip()]
    return args


def main():
    args = parse_args()
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
