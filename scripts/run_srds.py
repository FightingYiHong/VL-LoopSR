#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SRSD with StructPlan-SR v10/test_fey.

Core idea:
  SRSD is used only as the task/data provider.
  The actual solver is delegated to test_fey.py (or another compatible v10
  file) via:
    build_dataset_from_explicit_splits(...)
    _run_core_pipeline(...)

This wrapper keeps SRSD-specific dataset enumeration and filtering outside the
solver so we can reuse exactly the same core pipeline as other benchmark
wrappers in this repo.
"""

import os
import re
import sys
import time
import pickle
import hashlib
import random
import warnings
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =========================================================
# Paths
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_metrics import enrich_result_metrics

V10_PATH = Path(os.environ.get(
    "SRSD_V10_PATH",
    os.environ.get("LLMSR_V10_PATH", str(SCRIPT_DIR / "test_fey.py")),
)).resolve()


# =========================================================
# SRSD config
# =========================================================
SRSD_ROOT = Path(os.environ.get(
    "SRSD_ROOT",
    str(PROJECT_ROOT / "srsd-benchmark" / "resource" / "datasets" / "srsd"),
)).resolve()

default_dataset_dirs = ",".join([
    "easy_set",
    "medium_set",
    "hard_set",
    "easy_set_dummy",
    "medium_set_dummy",
    "hard_set_dummy",
])
SRSD_DATASET_DIRS = [x.strip() for x in os.environ.get("SRSD_DATASET_DIRS", default_dataset_dirs).split(",") if x.strip()]

RUN_TIMESTAMP = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{int((time.time() % 1.0) * 1000):03d}"
RESULTS_ROOT = os.environ.get(
    "SRSD_V10_RESULTS_ROOT",
    str(PROJECT_ROOT / "runs" / f"srsd_results_v10_{RUN_TIMESTAMP}"),
)
GLOBAL_SUMMARY_CSV = os.path.join(RESULTS_ROOT, "all_srds_results_v10.csv")
GLOBAL_SELECTED_TASKS_CSV = os.path.join(RESULTS_ROOT, "srsd_selected_tasks.csv")

MIN_TRAIN_SAMPLES = int(os.environ.get("SRSD_MIN_TRAIN_SAMPLES", "50"))
MIN_VAL_SAMPLES = int(os.environ.get("SRSD_MIN_VAL_SAMPLES", "20"))
MIN_TEST_SAMPLES = int(os.environ.get("SRSD_MIN_TEST_SAMPLES", "20"))

max_train_env = os.environ.get("SRSD_MAX_TRAIN_SAMPLES", "none")
MAX_TRAIN_SAMPLES = None if str(max_train_env).lower() in {"none", "", "0"} else int(max_train_env)
max_val_env = os.environ.get("SRSD_MAX_VAL_SAMPLES", "none")
MAX_VAL_SAMPLES = None if str(max_val_env).lower() in {"none", "", "0"} else int(max_val_env)
max_test_env = os.environ.get("SRSD_MAX_TEST_SAMPLES", "none")
MAX_TEST_SAMPLES = None if str(max_test_env).lower() in {"none", "", "0"} else int(max_test_env)

MIN_FEATURES = int(os.environ.get("SRSD_MIN_FEATURES", "1"))
max_features_env = os.environ.get("SRSD_MAX_FEATURES", "8")
MAX_FEATURES = None if str(max_features_env).lower() in {"none", "", "0"} else int(max_features_env)

NAME_KEYWORDS = [x.strip().lower() for x in os.environ.get("SRSD_NAME_KEYWORDS", "").split(",") if x.strip()]
ALLOW_BASENAMES = [x.strip() for x in os.environ.get("SRSD_ALLOW_BASENAMES", "").split(",") if x.strip()]

max_files_env = os.environ.get("SRSD_MAX_FILES_PER_DATASET", "none")
MAX_FILES_PER_DATASET = None if str(max_files_env).lower() in {"none", "", "0"} else int(max_files_env)
RANDOM_SAMPLE_K = os.environ.get("SRSD_RANDOM_SAMPLE_K", "")
RANDOM_SAMPLE_K = None if RANDOM_SAMPLE_K == "" else int(RANDOM_SAMPLE_K)
RANDOM_SEED = int(os.environ.get("SRSD_RANDOM_SEED", "42"))
DRY_RUN_SELECTION_ONLY = os.environ.get("SRSD_DRY_RUN", "0").lower() in {"1", "true", "yes", "y"}

MSE_THRESHOLD = float(os.environ.get("SRSD_MSE_THRESHOLD", "100.0"))
PERFECT_FIT_TOL = float(os.environ.get("SRSD_PERFECT_FIT_TOL", "1e-10"))


def import_v10_module():
    if not V10_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find v10 file: {V10_PATH}\n"
            "Set SRSD_V10_PATH or LLMSR_V10_PATH to the actual v10 python file."
        )

    spec = importlib.util.spec_from_file_location("srsd_v10_runtime", str(V10_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["srsd_v10_runtime"] = module
    spec.loader.exec_module(module)

    module.RESULTS_ROOT = RESULTS_ROOT
    module.GLOBAL_SUMMARY_CSV = os.path.join(RESULTS_ROOT, "all_results_detailed.csv")
    module.GLOBAL_SUMMARY_JSON = os.path.join(RESULTS_ROOT, "global_summary.json")
    module.GLOBAL_SUMMARY_CSV_COMPACT = os.path.join(RESULTS_ROOT, "global_summary.csv")
    module.TIMING_BREAKDOWN_CSV = os.path.join(RESULTS_ROOT, "timing_breakdown.csv")
    module.TIMING_SUMMARY_JSON = os.path.join(RESULTS_ROOT, "timing_summary.json")
    module.PER_CASE_JSON_DIR = os.path.join(RESULTS_ROOT, "per_case_reports")
    module.SELECTED_TASKS_CSV = GLOBAL_SELECTED_TASKS_CSV

    if os.environ.get("LLMSR_MAX_RUNTIME_PER_TASK_SEC"):
        module.MAX_RUNTIME_PER_TASK_SEC = float(os.environ["LLMSR_MAX_RUNTIME_PER_TASK_SEC"])

    module.MSE_THRESHOLD = MSE_THRESHOLD
    module.PERFECT_FIT_TOL = PERFECT_FIT_TOL

    os.makedirs(module.RESULTS_ROOT, exist_ok=True)
    os.makedirs(module.PER_CASE_JSON_DIR, exist_ok=True)
    return module


def sanitize_name(text: str) -> str:
    text = str(text).replace("/", "__").replace("\\", "__")
    text = re.sub(r"[^a-zA-Z0-9_.\-]+", "_", text)
    suffix = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{text}_{suffix}"


def load_txt_dataset(txt_path: str) -> pd.DataFrame:
    arr = np.loadtxt(txt_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    arr = np.asarray(arr, dtype=float)
    n_cols = arr.shape[1]
    if n_cols < 2:
        raise ValueError(f"invalid txt file: {txt_path}, shape={arr.shape}")

    columns = [f"x{i+1}" for i in range(n_cols - 1)] + ["y"]
    return pd.DataFrame(arr, columns=columns)


def load_true_expression(true_eq_path: str):
    if not true_eq_path:
        return None
    try:
        with open(true_eq_path, "rb") as f:
            obj = pickle.load(f)
        return str(obj)
    except Exception:
        return None


def file_passes_filters(base_name, n_train, n_val, n_test, n_features):
    if n_train < MIN_TRAIN_SAMPLES or n_val < MIN_VAL_SAMPLES or n_test < MIN_TEST_SAMPLES:
        return False
    if MAX_TRAIN_SAMPLES is not None and n_train > MAX_TRAIN_SAMPLES:
        return False
    if MAX_VAL_SAMPLES is not None and n_val > MAX_VAL_SAMPLES:
        return False
    if MAX_TEST_SAMPLES is not None and n_test > MAX_TEST_SAMPLES:
        return False

    if n_features < MIN_FEATURES:
        return False
    if MAX_FEATURES is not None and n_features > MAX_FEATURES:
        return False

    if ALLOW_BASENAMES and base_name not in ALLOW_BASENAMES:
        return False

    if NAME_KEYWORDS:
        low = base_name.lower()
        if not any(k in low for k in NAME_KEYWORDS):
            return False

    return True


def collect_raw_tasks_for_dataset(dataset_dir_name: str):
    root = SRSD_ROOT / dataset_dir_name
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    true_eq_dir = root / "true_eq"

    if not train_dir.exists():
        raise FileNotFoundError(f"missing train dir: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"missing val dir: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"missing test dir: {test_dir}")

    train_files = {p.stem: p for p in train_dir.glob("*.txt")}
    val_files = {p.stem: p for p in val_dir.glob("*.txt")}
    test_files = {p.stem: p for p in test_dir.glob("*.txt")}
    common = sorted(set(train_files) & set(val_files) & set(test_files))

    rows = []
    for base_name in common:
        train_df = load_txt_dataset(str(train_files[base_name]))
        val_df = load_txt_dataset(str(val_files[base_name]))
        test_df = load_txt_dataset(str(test_files[base_name]))

        n_features = int(train_df.shape[1] - 1)
        eq_path = true_eq_dir / f"{base_name}.pkl"
        true_eq_path = str(eq_path) if eq_path.exists() else None

        rows.append({
            "dataset_dir": dataset_dir_name,
            "difficulty": dataset_dir_name,
            "base_name": base_name,
            "train_path": str(train_files[base_name]),
            "val_path": str(val_files[base_name]),
            "test_path": str(test_files[base_name]),
            "true_eq_path": true_eq_path,
            "true_expression": load_true_expression(true_eq_path),
            "n_features": n_features,
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    print("\n[DEBUG] all raw files before filtering:")
    print(df[["base_name", "n_features", "n_train", "n_val", "n_test"]].to_string(index=False))

    keep = df.apply(
        lambda r: file_passes_filters(
            r["base_name"], r["n_train"], r["n_val"], r["n_test"], r["n_features"]
        ),
        axis=1,
    )
    df = df[keep].copy()

    print("\n[DEBUG] raw files after filtering:")
    if len(df) == 0:
        print("(none)")
    else:
        print(df[["base_name", "n_features", "n_train", "n_val", "n_test"]].to_string(index=False))

    df = df.sort_values(by=["n_features", "base_name"]).reset_index(drop=True)

    if RANDOM_SAMPLE_K is not None and RANDOM_SAMPLE_K > 0 and len(df) > RANDOM_SAMPLE_K:
        rng = random.Random(RANDOM_SEED)
        idxs = list(range(len(df)))
        rng.shuffle(idxs)
        idxs = sorted(idxs[:RANDOM_SAMPLE_K])
        df = df.iloc[idxs].reset_index(drop=True)

    if MAX_FILES_PER_DATASET is not None:
        df = df.head(MAX_FILES_PER_DATASET).copy()

    return df.reset_index(drop=True)


def make_row_meta(row, dataset):
    return {
        "task_type": "srsd",
        "dataset_dir": str(row.get("dataset_dir", "")),
        "difficulty": str(row.get("difficulty", row.get("dataset_dir", ""))),
        "base_name": str(row.get("base_name", "")),
        "train_path": str(row.get("train_path", "")),
        "val_path": str(row.get("val_path", "")),
        "test_path": str(row.get("test_path", "")),
        "true_eq_path": str(row.get("true_eq_path", "")),
        "true_expression": row.get("true_expression"),
        "n_features": len(dataset.feature_names),
        "n_train": len(dataset.train_df),
        "n_val": len(dataset.val_df),
        "n_test": len(dataset.test_df),
    }


def run_one_raw_task_v10(v10, row, tmpdir: Path):
    start = time.time()
    result_prefix = {
        "dataset_dir": row.get("dataset_dir"),
        "difficulty": row.get("difficulty"),
        "base_name": row.get("base_name"),
        "train_path": row.get("train_path"),
        "val_path": row.get("val_path"),
        "test_path": row.get("test_path"),
        "true_eq_path": row.get("true_eq_path"),
        "true_expression": row.get("true_expression"),
    }

    try:
        train_df = load_txt_dataset(row["train_path"])
        val_df = load_txt_dataset(row["val_path"])
        test_df = load_txt_dataset(row["test_path"])

        dataset = v10.build_dataset_from_explicit_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            tmpdir=tmpdir,
        )
        dataset.source_tag = "srsd"

        row_meta = make_row_meta(row, dataset)
        out = v10._run_core_pipeline(dataset=dataset, row_meta=row_meta)
        out.update(result_prefix)
        out["runtime_sec"] = out.get("runtime_sec") or (time.time() - start)
        enrich_result_metrics(
            out,
            train_df=dataset.train_df,
            val_df=dataset.val_df,
            test_df=dataset.test_df,
            feature_names=dataset.feature_names,
        )
        return out

    except Exception as e:
        return {
            **result_prefix,
            "eval_profile": getattr(v10, "EVAL_PROFILE", None),
            "method_mode": getattr(v10, "METHOD_MODE", None),
            "no_leakage_mode": getattr(v10, "NO_LEAKAGE_MODE", None),
            "task_type": "srsd",
            "n_features": row.get("n_features"),
            "n_train": row.get("n_train"),
            "n_val": row.get("n_val"),
            "n_test": row.get("n_test"),
            "valid_formula_found": False,
            "num_candidate_exprs": 0,
            "best_expr": None,
            "best_val_mse": None,
            "best_test_mse": None,
            "train_r2": None,
            "val_r2": None,
            "test_r2": None,
            "expr_complexity": None,
            "expr_depth": None,
            "expr_string_length": None,
            "expr_sympy_ops": None,
            "perfect_fit_by_r2": False,
            "metric_eval_error": None,
            "passed": False,
            "perfect_fit": False,
            "runtime_sec": time.time() - start,
            "error": repr(e),
        }


def print_one_result(idx, total, row, res):
    print(f"[{idx}/{total}] Processing: {row['base_name']}")
    print(f"   dataset_dir:         {row['dataset_dir']}")
    print(f"   train/val/test:      ({res.get('n_train')}, {res.get('n_val')}, {res.get('n_test')})")
    print(f"   n_features:          {res.get('n_features')}")
    print(f"   valid_formula_found: {res.get('valid_formula_found')}")
    print(f"   num_candidate_exprs: {res.get('num_candidate_exprs')}")
    print(f"   best_expr:           {res.get('best_expr')}")
    print(f"   best_val_mse:        {res.get('best_val_mse')}")
    print(f"   best_test_mse:       {res.get('best_test_mse')}")
    print(f"   train/val/test R2:   ({res.get('train_r2')}, {res.get('val_r2')}, {res.get('test_r2')})")
    print(f"   expr_complexity:     {res.get('expr_complexity')}")
    print(f"   expr_depth:          {res.get('expr_depth')}")
    print(f"   expr_sympy_ops:      {res.get('expr_sympy_ops')}")
    print(f"   passed:              {res.get('passed')}")
    print(f"   perfect_fit:         {res.get('perfect_fit')}")
    print(f"   perfect_fit_by_r2:   {res.get('perfect_fit_by_r2')}")
    print(f"   metric_eval_error:   {res.get('metric_eval_error')}")
    print(f"   runtime_sec:         {res.get('runtime_sec')}")
    print(f"   error:               {res.get('error')}")
    print("-" * 72)


def print_local_summary(title, results):
    total = len(results)
    valid_results = [r for r in results if r.get("valid_formula_found")]
    calculated_test_mses = [r.get("best_test_mse") for r in results if r.get("best_test_mse") is not None]
    calculated_test_r2s = [r.get("test_r2") for r in results if r.get("test_r2") is not None]
    complexities = [r.get("expr_complexity") for r in results if r.get("expr_complexity") is not None]
    passed_results = [r for r in results if r.get("passed")]
    passed_test_mses = [r.get("best_test_mse") for r in passed_results if r.get("best_test_mse") is not None]
    perfect_fits = [r for r in results if r.get("perfect_fit")]

    valid_count = len(valid_results)
    passed_count = len(passed_results)
    high_error_count = len(calculated_test_mses) - passed_count

    print("\n" + "=" * 72)
    print(f"SUMMARY: {title}")
    print("=" * 72)
    print(f"Total Files:            {total}")
    print(f"Valid Formulas Found:   {valid_count} ({(valid_count / total * 100 if total else 0):.1f}%)")
    print(f"Calculated Test MSEs:   {len(calculated_test_mses)}")
    print(f"Calculated Test R2s:    {len(calculated_test_r2s)}")
    print("-" * 52)
    print(f"PASSED (test MSE <= {MSE_THRESHOLD}): {passed_count} ({(passed_count / total * 100 if total else 0):.1f}%)")

    if passed_test_mses:
        print(f"   Mean Test MSE:       {np.mean(passed_test_mses):.6f}")
        print(f"   Median Test MSE:     {np.median(passed_test_mses):.6f}")
        print(f"   Perfect Fits:        {len(perfect_fits)}")
    else:
        print("   No files passed the threshold.")
    if calculated_test_r2s:
        print(f"   Mean Test R2:        {np.mean(calculated_test_r2s):.6f}")
        print(f"   Median Test R2:      {np.median(calculated_test_r2s):.6f}")
    if complexities:
        print(f"   Mean Expr Complexity:{np.mean(complexities):.2f}")
        print(f"   Median Expr Complexity: {np.median(complexities):.2f}")

    print("-" * 52)
    print("FAILED Details:")
    print(f"   High Error (> {MSE_THRESHOLD}):     {high_error_count}")
    print(f"   Extraction/Valid Fail:              {total - valid_count}")
    print("=" * 72)


def main():
    overall_start = time.time()
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    dataset_tasks_map = {}
    all_selected = []

    for dataset_dir_name in SRSD_DATASET_DIRS:
        print("\n" + "#" * 100)
        print(f"SCAN DATASET DIR: {dataset_dir_name}")
        print("#" * 100)

        df_tasks = collect_raw_tasks_for_dataset(dataset_dir_name)
        dataset_tasks_map[dataset_dir_name] = df_tasks

        selected_csv = os.path.join(RESULTS_ROOT, f"{sanitize_name(dataset_dir_name)}_selected_files.csv")
        df_tasks.to_csv(selected_csv, index=False, encoding="utf-8-sig")
        print(f"Selection saved to: {selected_csv}")

        if len(df_tasks) > 0:
            all_selected.append(df_tasks)

    if all_selected:
        pd.concat(all_selected, axis=0, ignore_index=True).to_csv(GLOBAL_SELECTED_TASKS_CSV, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(GLOBAL_SELECTED_TASKS_CSV, index=False, encoding="utf-8-sig")
    print(f"Global selection saved to: {GLOBAL_SELECTED_TASKS_CSV}")

    total_selected = sum(len(df) for df in dataset_tasks_map.values())
    if total_selected == 0:
        print("没有筛选到可运行的 SRSD 任务。")
        return

    if DRY_RUN_SELECTION_ONLY:
        print("SRSD_DRY_RUN=1，仅完成任务筛选。")
        return

    v10 = import_v10_module()
    print(f"[INFO] Using v10 file: {V10_PATH}")
    print(f"[INFO] Results root:   {RESULTS_ROOT}")
    print(f"[INFO] v10 profile:    {getattr(v10, 'EVAL_PROFILE', None)}")
    print(f"[INFO] v10 mode:       {getattr(v10, 'METHOD_MODE', None)}")
    print(f"[INFO] Selected {total_selected} SRSD cases. Starting execution...", flush=True)

    all_results_global = []
    global_idx = 0

    with TemporaryDirectory(prefix="srsd_v10_tmp_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)

        for dataset_dir_name in SRSD_DATASET_DIRS:
            df_tasks = dataset_tasks_map[dataset_dir_name]
            if len(df_tasks) == 0:
                print(f"没有筛选到可运行的文件: {dataset_dir_name}")
                continue

            print("\n" + "#" * 100)
            print(f"RUN DATASET DIR: {dataset_dir_name}")
            print("#" * 100)

            dataset_results = []
            dataset_csv = os.path.join(RESULTS_ROOT, f"{sanitize_name(dataset_dir_name)}_results.csv")

            for _, row in df_tasks.iterrows():
                global_idx += 1
                row_dict = row.to_dict()
                print(f"[{global_idx}/{total_selected}] Starting: {row_dict['base_name']}", flush=True)
                res = run_one_raw_task_v10(v10, row_dict, tmpdir=tmpdir)
                dataset_results.append(res)
                all_results_global.append(res)
                print_one_result(global_idx, total_selected, row_dict, res)

                pd.DataFrame(dataset_results).to_csv(dataset_csv, index=False)
                pd.DataFrame(all_results_global).to_csv(GLOBAL_SUMMARY_CSV, index=False)

            print(f"Saved dataset results to: {dataset_csv}")
            print_local_summary(dataset_dir_name, dataset_results)

    try:
        v10.save_all_outputs(all_results_global, overall_start)
        v10.print_summary(all_results_global, overall_start)
    except Exception as e:
        print(f"[WARN] v10 summary saving failed: {repr(e)}")
        pd.DataFrame(all_results_global).to_csv(GLOBAL_SUMMARY_CSV, index=False)
        print(f"Saved global results to: {GLOBAL_SUMMARY_CSV}")
        print_local_summary("ALL SRSD FILES", all_results_global)

    print(f"Total Runtime: {time.time() - overall_start:.2f} seconds")


if __name__ == "__main__":
    main()
