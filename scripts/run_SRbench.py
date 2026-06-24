#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SRBench with StructPlan-SR v10/test_fey.

Core idea:
  SRBench is used only as the dataset/task provider.
  The actual solver is delegated to test_fey.py (or another compatible v10
  file) via:
    build_dataset_from_explicit_splits(...)
    _run_core_pipeline(...)

Supported SRBench sources:
  - SRBench metadata CSV:
      <srbench-root>/docs/csv/datasets_info.csv
  - Local cached PMLB TSVs:
      <repo>/.cache/pmlb_cache/<dataset>/<dataset>.tsv.gz
  - Local CSV mirror:
      <repo>/data/pmlb_regression_csv/<dataset>/<dataset>.csv
  - Remote fallback through pmlb.fetch_data(...)

This wrapper keeps the benchmark logic outside the solver so we can reuse the
same core pipeline as SLDBench / LLM-SRBench.
"""

import os
import re
import sys
import time
import json
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
PMLB_SOURCE_ROOT = PROJECT_ROOT / "pmlb"
if PMLB_SOURCE_ROOT.exists():
    sys.path.insert(0, str(PMLB_SOURCE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_metrics import enrich_result_metrics

V10_PATH = Path(os.environ.get(
    "SRBENCH_V10_PATH",
    os.environ.get("LLMSR_V10_PATH", str(SCRIPT_DIR / "test_fey.py")),
)).resolve()


# =========================================================
# SRBench config
# =========================================================
SRBENCH_ROOT = os.environ.get("SRBENCH_ROOT", str(PROJECT_ROOT / "srbench"))
SRBENCH_DATASETS_INFO_CSV = os.environ.get(
    "SRBENCH_DATASETS_INFO_CSV",
    os.path.join(SRBENCH_ROOT, "docs", "csv", "datasets_info.csv"),
)
SRBENCH_PMLB_CACHE_DIR = os.environ.get("SRBENCH_PMLB_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "pmlb_cache"))
SRBENCH_LOCAL_CSV_ROOT = os.environ.get("SRBENCH_LOCAL_CSV_ROOT", str(PROJECT_ROOT / "data" / "pmlb_regression_csv"))

ALLOW_GROUPS = [x.strip() for x in os.environ.get("SRBENCH_ALLOW_GROUPS", "").split(",") if x.strip()]
ALLOW_CASE_NAMES = [x.strip() for x in os.environ.get("SRBENCH_ALLOW_CASE_NAMES", "").split(",") if x.strip()]
NAME_KEYWORDS = [x.strip().lower() for x in os.environ.get("SRBENCH_NAME_KEYWORDS", "").split(",") if x.strip()]

MIN_FEATURES = int(os.environ.get("SRBENCH_MIN_FEATURES", "1"))
max_features_env = os.environ.get("SRBENCH_MAX_FEATURES", "none")
MAX_FEATURES = None if str(max_features_env).lower() in {"none", "", "0"} else int(max_features_env)

MIN_SAMPLES = int(os.environ.get("SRBENCH_MIN_SAMPLES", "1"))
max_samples_env = os.environ.get("SRBENCH_MAX_SAMPLES", "none")
MAX_SAMPLES = None if str(max_samples_env).lower() in {"none", "", "0"} else int(max_samples_env)

MAX_FILES = os.environ.get("SRBENCH_MAX_FILES", "none")
MAX_FILES = None if str(MAX_FILES).lower() in {"none", "", "0"} else int(MAX_FILES)

RANDOM_SAMPLE_K = os.environ.get("SRBENCH_RANDOM_SAMPLE_K", "")
RANDOM_SAMPLE_K = None if RANDOM_SAMPLE_K == "" else int(RANDOM_SAMPLE_K)
RANDOM_SEED = int(os.environ.get("SRBENCH_RANDOM_SEED", "42"))
DRY_RUN_SELECTION_ONLY = os.environ.get("SRBENCH_DRY_RUN", "0").lower() in {"1", "true", "yes", "y"}

TRAIN_TEST_SPLIT_SEED = int(os.environ.get("SRBENCH_SPLIT_SEED", "42"))
SPLIT_SHUFFLE = os.environ.get("SRBENCH_SPLIT_SHUFFLE", "1").lower() in {"1", "true", "yes", "y"}
TEST_RATIO = float(os.environ.get("SRBENCH_TEST_RATIO", "0.25"))
VAL_RATIO_WITHIN_TRAINVAL = float(os.environ.get("SRBENCH_VAL_RATIO_WITHIN_TRAINVAL", "0.2"))
SMALL_DATASET_VAL_MAX_ROWS = int(os.environ.get("SRBENCH_SMALL_DATASET_VAL_MAX_ROWS", "4"))
SMALL_DATASET_TOTAL_ROWS = int(os.environ.get("SRBENCH_SMALL_DATASET_TOTAL_ROWS", "80"))
SMALL_DATASET_VAL_RATIO_WITHIN_TRAINVAL = float(os.environ.get("SRBENCH_SMALL_DATASET_VAL_RATIO_WITHIN_TRAINVAL", "0.1"))

MIN_TRAIN_SAMPLES = int(os.environ.get("SRBENCH_MIN_TRAIN_SAMPLES", "1"))
MIN_VAL_SAMPLES = int(os.environ.get("SRBENCH_MIN_VAL_SAMPLES", "1"))
MIN_TEST_SAMPLES = int(os.environ.get("SRBENCH_MIN_TEST_SAMPLES", "1"))

RUN_TIMESTAMP = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{int((time.time() % 1.0) * 1000):03d}"
RESULTS_ROOT = os.environ.get(
    "SRBENCH_V10_RESULTS_ROOT",
    str(PROJECT_ROOT / "runs" / f"srbench_results_v10_{RUN_TIMESTAMP}"),
)
SELECTED_TASKS_CSV = os.path.join(RESULTS_ROOT, "srbench_selected_tasks.csv")
GLOBAL_SUMMARY_CSV = os.path.join(RESULTS_ROOT, "all_srbench_results_v10.csv")
CHECKPOINT_EVERY = int(os.environ.get("SRBENCH_CHECKPOINT_EVERY", "25"))
CHECKPOINT_DIR = os.path.join(RESULTS_ROOT, "checkpoints")
RUN_PROGRESS_JSON = os.path.join(RESULTS_ROOT, "run_progress.json")

MSE_THRESHOLD = float(os.environ.get("SRBENCH_MSE_THRESHOLD", "100.0"))
PERFECT_FIT_TOL = float(os.environ.get("SRBENCH_PERFECT_FIT_TOL", "1e-10"))


def import_v10_module():
    if not V10_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find v10 file: {V10_PATH}\n"
            "Set SRBENCH_V10_PATH or LLMSR_V10_PATH to the actual v10 python file."
        )

    spec = importlib.util.spec_from_file_location("srbench_v10_runtime", str(V10_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["srbench_v10_runtime"] = module
    spec.loader.exec_module(module)

    module.RESULTS_ROOT = RESULTS_ROOT
    module.GLOBAL_SUMMARY_CSV = os.path.join(RESULTS_ROOT, "all_results_detailed.csv")
    module.GLOBAL_SUMMARY_JSON = os.path.join(RESULTS_ROOT, "global_summary.json")
    module.GLOBAL_SUMMARY_CSV_COMPACT = os.path.join(RESULTS_ROOT, "global_summary.csv")
    module.TIMING_BREAKDOWN_CSV = os.path.join(RESULTS_ROOT, "timing_breakdown.csv")
    module.TIMING_SUMMARY_JSON = os.path.join(RESULTS_ROOT, "timing_summary.json")
    module.PER_CASE_JSON_DIR = os.path.join(RESULTS_ROOT, "per_case_reports")
    module.SELECTED_TASKS_CSV = SELECTED_TASKS_CSV

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


def now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def sanitize_variable_name(name: str, used_names=None) -> str:
    used_names = used_names if used_names is not None else set()
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip())
    safe = safe.strip("_")
    if not safe:
        safe = "x"
    if safe[0].isdigit():
        safe = f"v_{safe}"
    if safe == "y":
        safe = "target_var"

    base = safe
    suffix = 2
    while safe in used_names:
        safe = f"{base}_{suffix}"
        suffix += 1
    used_names.add(safe)
    return safe


def normalize_group_name(group_name: str) -> str:
    return str(group_name).strip().lower()


def task_passes_filters(group_name, case_name, n_features, n_samples):
    if n_features < MIN_FEATURES:
        return False
    if MAX_FEATURES is not None and n_features > MAX_FEATURES:
        return False
    if n_samples < MIN_SAMPLES:
        return False
    if MAX_SAMPLES is not None and n_samples > MAX_SAMPLES:
        return False
    if ALLOW_GROUPS and normalize_group_name(group_name) not in {normalize_group_name(x) for x in ALLOW_GROUPS}:
        return False
    if ALLOW_CASE_NAMES and case_name not in ALLOW_CASE_NAMES:
        return False
    if NAME_KEYWORDS:
        low = case_name.lower()
        if not any(k in low for k in NAME_KEYWORDS):
            return False
    return True


def collect_srbench_tasks():
    if not os.path.exists(SRBENCH_DATASETS_INFO_CSV):
        raise FileNotFoundError(f"SRBench metadata CSV not found: {SRBENCH_DATASETS_INFO_CSV}")

    df = pd.read_csv(SRBENCH_DATASETS_INFO_CSV)
    if len(df) == 0:
        return df

    rename_map = {}
    if "name" in df.columns:
        rename_map["name"] = "dataset_name"
    if "Group" in df.columns:
        rename_map["Group"] = "group_name"
    if "nsamples" in df.columns:
        rename_map["nsamples"] = "n_samples_meta"
    if "nfeatures" in df.columns:
        rename_map["nfeatures"] = "n_features_meta"
    df = df.rename(columns=rename_map)

    required = {"dataset_name", "group_name", "n_samples_meta", "n_features_meta"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required SRBench metadata columns: {sorted(missing)}")

    keep = df.apply(
        lambda r: task_passes_filters(
            r["group_name"],
            r["dataset_name"],
            int(r["n_features_meta"]),
            int(r["n_samples_meta"]),
        ),
        axis=1,
    )
    df = df[keep].copy()

    if len(df) == 0:
        return df

    out = pd.DataFrame({
        "dataset_name": df["dataset_name"].astype(str),
        "group_name": df["group_name"].astype(str),
        "n_samples_meta": df["n_samples_meta"].astype(int),
        "n_features_meta": df["n_features_meta"].astype(int),
    })
    out = out.sort_values(by=["group_name", "n_features_meta", "n_samples_meta", "dataset_name"]).reset_index(drop=True)

    if RANDOM_SAMPLE_K is not None and RANDOM_SAMPLE_K > 0 and len(out) > RANDOM_SAMPLE_K:
        rng = random.Random(RANDOM_SEED)
        idxs = list(range(len(out)))
        rng.shuffle(idxs)
        idxs = sorted(idxs[:RANDOM_SAMPLE_K])
        out = out.iloc[idxs].reset_index(drop=True)

    if MAX_FILES is not None:
        out = out.head(MAX_FILES).copy()

    return out.reset_index(drop=True)


def resolve_local_dataset_path(dataset_name: str):
    cache_path = Path(SRBENCH_PMLB_CACHE_DIR) / dataset_name / f"{dataset_name}.tsv.gz"
    if cache_path.exists():
        return str(cache_path), "pmlb_cache_tsv"

    csv_path = Path(SRBENCH_LOCAL_CSV_ROOT) / dataset_name / f"{dataset_name}.csv"
    if csv_path.exists():
        return str(csv_path), "local_csv"

    return None, "remote_fetch"


def load_raw_dataset_frame(dataset_name: str):
    data_path, source_kind = resolve_local_dataset_path(dataset_name)
    if source_kind == "pmlb_cache_tsv":
        df = pd.read_csv(data_path, sep="\t", compression="gzip")
        return df, source_kind, data_path

    if source_kind == "local_csv":
        df = pd.read_csv(data_path)
        return df, source_kind, data_path

    from pmlb import fetch_data

    df = fetch_data(dataset_name, return_X_y=False, local_cache_dir=SRBENCH_PMLB_CACHE_DIR, dropna=True)
    fetched_path = str(Path(SRBENCH_PMLB_CACHE_DIR) / dataset_name / f"{dataset_name}.tsv.gz")
    return df, source_kind, fetched_path


def detect_target_column(columns):
    for candidate in ["target", "y"]:
        if candidate in columns:
            return candidate
    raise KeyError(f"Cannot detect target column. Existing columns: {list(columns)}")


def make_numeric_xy_dataframe(df_raw: pd.DataFrame):
    df_raw = df_raw.copy()
    df_raw.columns = [str(c).strip().replace(".", "_") for c in df_raw.columns]

    target_col = detect_target_column(df_raw.columns)
    feature_cols = [c for c in df_raw.columns if c != target_col]
    if not feature_cols:
        raise ValueError("No feature columns found after removing target column.")

    encoded_columns = []
    numeric_features = {}
    for col in feature_cols:
        series = df_raw[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_features[col] = pd.to_numeric(series, errors="coerce").astype(float)
        else:
            codes, _ = pd.factorize(series, sort=True)
            codes = codes.astype(float)
            codes[codes < 0] = np.nan
            numeric_features[col] = pd.Series(codes, index=df_raw.index, dtype=float)
            encoded_columns.append(col)

    X_num = pd.DataFrame(numeric_features, index=df_raw.index)
    y_num = pd.to_numeric(df_raw[target_col], errors="coerce").astype(float)

    keep_mask = np.isfinite(y_num.to_numpy(dtype=float))
    keep_mask &= np.isfinite(X_num.to_numpy(dtype=float)).all(axis=1)

    X_num = X_num.loc[keep_mask].reset_index(drop=True)
    y_num = y_num.loc[keep_mask].reset_index(drop=True)

    if len(X_num) < 3:
        raise ValueError(f"Not enough valid numeric rows after cleaning: n={len(X_num)}")

    n_features = X_num.shape[1]
    used = set()
    sanitized_feature_names = [
        sanitize_variable_name(col, used_names=used)
        for col in feature_cols
    ]
    out = pd.DataFrame({
        sanitized_feature_names[i]: X_num.iloc[:, i].to_numpy(dtype=float)
        for i in range(n_features)
    })
    out["y"] = y_num.to_numpy(dtype=float)

    extra_info = {
        "original_target_name": target_col,
        "original_feature_names": [str(x) for x in feature_cols],
        "sanitized_feature_names": [str(x) for x in sanitized_feature_names],
        "encoded_feature_names": encoded_columns,
        "dropped_rows": int(len(df_raw) - len(out)),
        "n_features_inferred": int(n_features),
        "n_rows_clean": int(len(out)),
    }
    return out, extra_info


def choose_val_count(n_train_val, total_rows, default_ratio):
    n_val = max(1, int(round(n_train_val * default_ratio)))

    if total_rows <= SMALL_DATASET_TOTAL_ROWS:
        small_ratio_count = max(1, int(round(n_train_val * SMALL_DATASET_VAL_RATIO_WITHIN_TRAINVAL)))
        n_val = min(n_val, small_ratio_count, SMALL_DATASET_VAL_MAX_ROWS)

    n_val = min(n_val, n_train_val - 1)
    return n_val


def split_train_val_test(df: pd.DataFrame, test_ratio=0.25, val_ratio_within_trainval=0.2, seed=42, shuffle=True):
    if len(df) < 3:
        raise ValueError(f"Need at least 3 rows to create train/val/test split, got {len(df)}")

    idx = np.arange(len(df))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_test = max(1, int(round(len(df) * test_ratio)))
    n_test = min(n_test, len(df) - 2)

    test_idx = idx[:n_test]
    train_val_idx = idx[n_test:]

    n_val = choose_val_count(
        n_train_val=len(train_val_idx),
        total_rows=len(df),
        default_ratio=val_ratio_within_trainval,
    )

    val_idx = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def make_row_meta(row, dataset, extra_info, source_kind, source_path):
    return {
        "task_type": "srbench",
        "dataset_dir": "srbench",
        "difficulty": str(row.get("group_name", "srbench")),
        "base_name": str(row.get("dataset_name", "srbench_case")),
        "group_name": str(row.get("group_name", "")),
        "dataset_name": str(row.get("dataset_name", "")),
        "original_feature_names": " | ".join(extra_info.get("original_feature_names", [])),
        "sanitized_feature_names": " | ".join(extra_info.get("sanitized_feature_names", [])),
        "encoded_feature_names": " | ".join(extra_info.get("encoded_feature_names", [])),
        "original_target_name": str(extra_info.get("original_target_name", "")),
        "dropped_rows": int(extra_info.get("dropped_rows", 0)),
        "n_samples_meta": row.get("n_samples_meta"),
        "n_features_meta": row.get("n_features_meta"),
        "data_source_kind": source_kind,
        "data_source_path": source_path,
        "n_features": len(dataset.feature_names),
        "n_train": len(dataset.train_df),
        "n_val": len(dataset.val_df),
        "n_test": len(dataset.test_df),
    }


def run_one_srbench_task_v10(v10, row, tmpdir: Path):
    start = time.time()
    result_prefix = {
        "srbench_group_name": row.get("group_name"),
        "srbench_dataset_name": row.get("dataset_name"),
        "case_name": row.get("dataset_name"),
    }

    try:
        raw_df, source_kind, source_path = load_raw_dataset_frame(str(row["dataset_name"]))
        xy_df, extra_info = make_numeric_xy_dataframe(raw_df)
        train_df, val_df, test_df = split_train_val_test(
            xy_df,
            test_ratio=TEST_RATIO,
            val_ratio_within_trainval=VAL_RATIO_WITHIN_TRAINVAL,
            seed=TRAIN_TEST_SPLIT_SEED,
            shuffle=SPLIT_SHUFFLE,
        )

        if len(train_df) < MIN_TRAIN_SAMPLES:
            raise ValueError(f"train split too small: {len(train_df)} < {MIN_TRAIN_SAMPLES}")
        if len(val_df) < MIN_VAL_SAMPLES:
            raise ValueError(f"val split too small: {len(val_df)} < {MIN_VAL_SAMPLES}")
        if len(test_df) < MIN_TEST_SAMPLES:
            raise ValueError(f"test split too small: {len(test_df)} < {MIN_TEST_SAMPLES}")

        dataset = v10.build_dataset_from_explicit_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            tmpdir=tmpdir,
        )
        dataset.source_tag = "srbench"

        row_meta = make_row_meta(row, dataset, extra_info, source_kind, source_path)
        out = v10._run_core_pipeline(dataset=dataset, row_meta=row_meta)
        out.update(result_prefix)
        out["data_source_kind"] = source_kind
        out["data_source_path"] = source_path
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
            "task_type": "srbench",
            "dataset_dir": "srbench",
            "difficulty": row.get("group_name"),
            "base_name": row.get("dataset_name"),
            "n_features": None,
            "n_train": None,
            "n_val": None,
            "n_test": None,
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
    print(f"[{idx}/{total}] Processing: {row['dataset_name']}")
    print(f"   group_name:          {row['group_name']}")
    print(f"   data_source_kind:    {res.get('data_source_kind')}")
    print(f"   train/val/test:      ({res.get('n_train')}, {res.get('n_val')}, {res.get('n_test')})")
    print(f"   inferred_features:   {res.get('n_features')}")
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


def update_result_runtime_metadata(res, idx, total, overall_start):
    elapsed_sec = time.time() - overall_start
    res["completed_index"] = int(idx)
    res["total_tasks"] = int(total)
    res["elapsed_since_run_start_sec"] = float(elapsed_sec)
    res["finished_at"] = now_string()
    return res


def build_progress_payload(all_results, total, overall_start):
    elapsed_sec = float(time.time() - overall_start)
    completed = len(all_results)
    runtimes = [float(r.get("runtime_sec")) for r in all_results if r.get("runtime_sec") is not None]
    test_r2s = [float(r.get("test_r2")) for r in all_results if r.get("test_r2") is not None]
    complexities = [float(r.get("expr_complexity")) for r in all_results if r.get("expr_complexity") is not None]
    passed = sum(1 for r in all_results if r.get("passed"))
    valid = sum(1 for r in all_results if r.get("valid_formula_found"))

    return {
        "updated_at": now_string(),
        "results_root": RESULTS_ROOT,
        "completed_tasks": int(completed),
        "total_tasks": int(total),
        "remaining_tasks": int(max(0, total - completed)),
        "progress_pct": float((completed / total) * 100.0) if total else 0.0,
        "elapsed_sec": elapsed_sec,
        "elapsed_min": elapsed_sec / 60.0,
        "mean_runtime_sec": float(np.mean(runtimes)) if runtimes else None,
        "median_runtime_sec": float(np.median(runtimes)) if runtimes else None,
        "mean_test_r2": float(np.mean(test_r2s)) if test_r2s else None,
        "median_test_r2": float(np.median(test_r2s)) if test_r2s else None,
        "mean_expr_complexity": float(np.mean(complexities)) if complexities else None,
        "median_expr_complexity": float(np.median(complexities)) if complexities else None,
        "valid_formula_count": int(valid),
        "passed_count": int(passed),
        "checkpoint_every": int(CHECKPOINT_EVERY),
    }


def save_progress_snapshot(all_results, group_results_map, total, overall_start, checkpoint_label=None):
    pd.DataFrame(all_results).to_csv(GLOBAL_SUMMARY_CSV, index=False)

    progress_payload = build_progress_payload(all_results, total, overall_start)
    with open(RUN_PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump(progress_payload, f, indent=2, ensure_ascii=False)

    if checkpoint_label is None:
        return

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_csv = os.path.join(CHECKPOINT_DIR, f"srbench_results_{checkpoint_label}.csv")
    checkpoint_json = os.path.join(CHECKPOINT_DIR, f"run_progress_{checkpoint_label}.json")
    pd.DataFrame(all_results).to_csv(checkpoint_csv, index=False)
    with open(checkpoint_json, "w", encoding="utf-8") as f:
        json.dump(progress_payload, f, indent=2, ensure_ascii=False)

    for group_name, group_results in group_results_map.items():
        group_checkpoint_csv = os.path.join(
            CHECKPOINT_DIR,
            f"{sanitize_name(group_name)}_{checkpoint_label}.csv",
        )
        pd.DataFrame(group_results).to_csv(group_checkpoint_csv, index=False)

    print(
        f"[INFO] Checkpoint saved: {checkpoint_label} "
        f"({progress_payload['completed_tasks']}/{progress_payload['total_tasks']}, "
        f"elapsed={progress_payload['elapsed_min']:.2f} min)",
        flush=True,
    )


def main():
    overall_start = time.time()
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    df_tasks = collect_srbench_tasks()
    df_tasks.to_csv(SELECTED_TASKS_CSV, index=False, encoding="utf-8-sig")
    print(f"Selection saved to: {SELECTED_TASKS_CSV}")

    if len(df_tasks) == 0:
        print("没有筛选到可运行的 SRBench 任务。")
        return

    if DRY_RUN_SELECTION_ONLY:
        print("SRBENCH_DRY_RUN=1，仅完成任务筛选。")
        return

    v10 = import_v10_module()
    print(f"[INFO] Using v10 file: {V10_PATH}")
    print(f"[INFO] Results root:   {RESULTS_ROOT}")
    print(f"[INFO] v10 profile:    {getattr(v10, 'EVAL_PROFILE', None)}")
    print(f"[INFO] v10 mode:       {getattr(v10, 'METHOD_MODE', None)}")
    print(f"[INFO] Selected {len(df_tasks)} SRBench datasets. Starting execution...", flush=True)

    all_results = []
    group_results_map = {}

    with TemporaryDirectory(prefix="srbench_v10_tmp_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        total = len(df_tasks)
        for idx, (_, row) in enumerate(df_tasks.iterrows(), start=1):
            row_dict = row.to_dict()
            print(f"[{idx}/{total}] Starting: {row_dict['dataset_name']}", flush=True)
            res = run_one_srbench_task_v10(v10, row_dict, tmpdir=tmpdir)
            update_result_runtime_metadata(res, idx, total, overall_start)
            all_results.append(res)
            group_results_map.setdefault(row_dict["group_name"], []).append(res)
            print_one_result(idx, total, row_dict, res)
            save_progress_snapshot(all_results, group_results_map, total, overall_start)

            if CHECKPOINT_EVERY > 0 and (idx % CHECKPOINT_EVERY == 0):
                checkpoint_label = f"checkpoint_{idx:04d}"
                save_progress_snapshot(
                    all_results,
                    group_results_map,
                    total,
                    overall_start,
                    checkpoint_label=checkpoint_label,
                )

    save_progress_snapshot(
        all_results,
        group_results_map,
        len(df_tasks),
        overall_start,
        checkpoint_label="final",
    )

    for group_name, group_results in group_results_map.items():
        group_csv = os.path.join(RESULTS_ROOT, f"{sanitize_name(group_name)}_results.csv")
        pd.DataFrame(group_results).to_csv(group_csv, index=False)
        print(f"Saved group results to: {group_csv}")

    try:
        v10.save_all_outputs(all_results, overall_start)
        v10.print_summary(all_results, overall_start)
    except Exception as e:
        print(f"[WARN] v10 summary saving failed: {repr(e)}")
        pd.DataFrame(all_results).to_csv(GLOBAL_SUMMARY_CSV, index=False)
        print(f"Saved global results to: {GLOBAL_SUMMARY_CSV}")
        print_local_summary("ALL SRBENCH DATASETS", all_results)

    print(f"Total Runtime: {time.time() - overall_start:.2f} seconds")


if __name__ == "__main__":
    main()
