#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run SLDBench with StructPlan-SR v10.

Core idea:
  SLDBench is used only as the data/task provider.
  The solver is v10._run_core_pipeline(), i.e. the full v10 method:
    observer/visual evidence -> LLM structural planner -> plan-conditioned expansion
    -> high/low-dimensional rescue -> evaluator/refine/final selection.

This script intentionally does NOT import scripts/run_sldbench.py, to avoid
self-import/circular-import conflicts when this file is renamed or copied.
"""

import os
import re
import sys
import time
import json
import hashlib
import random
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

# =========================================================
# Repository-local defaults
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Must be set before importing datasets.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))

import numpy as np
import pandas as pd
import datasets

# =========================================================
# Paths
# =========================================================
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_metrics import enrich_result_metrics

# v10 file path. You can override it if your filename is different:
#   export LLMSR_V10_PATH=/path/to/this/repo/scripts/xxx.py
def resolve_v10_path() -> Path:
    env_path = os.environ.get("LLMSR_V10_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()

    preferred_candidates = [
        SCRIPT_DIR / "structplan_sr_highdim_llm_companion_v10.py",
        SCRIPT_DIR / "test_fey.py",
    ]
    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate.resolve()

    # Final fallback: auto-detect a compatible runtime file in scripts/.
    for candidate in sorted(SCRIPT_DIR.glob("*.py")):
        if candidate.name == Path(__file__).name:
            continue
        try:
            text = candidate.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "def build_dataset_from_explicit_splits" in text and "def _run_core_pipeline" in text:
            return candidate.resolve()

    return preferred_candidates[0].resolve()


V10_PATH = resolve_v10_path()

# =========================================================
# SLDBench config
# =========================================================
SLDBENCH_HUB_REPO = os.environ.get("SLDBENCH_HUB_REPO", "pkuHaowei/sldbench")

TASK_SCHEMA_MAP = {
    "data_constrained_scaling_law": {
        "feature_names": ["unique_tokens", "params", "tokens"],
        "target_name": "loss",
    },
    "domain_mixture_scaling_law": {
        "feature_names": [f"proportion_domain_{i+1}" for i in range(5)],
        "target_name": [f"loss_domain_{i+1}" for i in range(5)],
    },
    "lr_bsz_scaling_law": {
        "feature_names": ["lr", "bsz", "data_size", "non_embedding_param_size"],
        "target_name": "lm_loss",
    },
    "moe_scaling_law": {
        "feature_names": ["num_experts", "dense_parameter_count"],
        "target_name": "loss_validation",
    },
    "sft_scaling_law": {
        "feature_names": ["sft_data_size"],
        "target_name": "sft_loss",
    },
    "vocab_scaling_law": {
        "feature_names": ["non_vocab_parameters", "vocab_size", "num_characters"],
        "target_name": "unigram_normalized_loss",
    },
    "parallel_scaling_law": {
        "feature_names": ["num_params", "parallel_size"],
        "target_name": "loss",
    },
    "easy_question_scaling_law": {
        "feature_names": ["log_flops"],
        "target_name": "brier_score",
    },
}

ALL_TASKS = list(TASK_SCHEMA_MAP.keys())

# Optional filters via env.
# Examples:
#   export SLDBENCH_ALLOW_TASKS=sft_scaling_law,moe_scaling_law
#   export SLDBENCH_MAX_FILES_PER_TASK=1
ALLOW_TASKS = [x.strip() for x in os.environ.get("SLDBENCH_ALLOW_TASKS", "").split(",") if x.strip()]
ALLOW_GROUPS = [x.strip() for x in os.environ.get("SLDBENCH_ALLOW_GROUPS", "").split(",") if x.strip()]
ALLOW_CASE_NAMES = [x.strip() for x in os.environ.get("SLDBENCH_ALLOW_CASE_NAMES", "").split(",") if x.strip()]
NAME_KEYWORDS = [x.strip().lower() for x in os.environ.get("SLDBENCH_NAME_KEYWORDS", "").split(",") if x.strip()]
MAX_FILES_PER_TASK = os.environ.get("SLDBENCH_MAX_FILES_PER_TASK", "none")
MAX_FILES_PER_TASK = None if str(MAX_FILES_PER_TASK).lower() in {"none", "", "0"} else int(MAX_FILES_PER_TASK)
RANDOM_SAMPLE_K = os.environ.get("SLDBENCH_RANDOM_SAMPLE_K", "")
RANDOM_SAMPLE_K = None if RANDOM_SAMPLE_K == "" else int(RANDOM_SAMPLE_K)
RANDOM_SEED = int(os.environ.get("SLDBENCH_RANDOM_SEED", "42"))
DRY_RUN_SELECTION_ONLY = os.environ.get("SLDBENCH_DRY_RUN", "0").lower() in {"1", "true", "yes", "y"}

MIN_FEATURES = int(os.environ.get("SLDBENCH_MIN_FEATURES", "1"))
MAX_FEATURES = int(os.environ.get("SLDBENCH_MAX_FEATURES", "20"))
MIN_TRAIN_SAMPLES = int(os.environ.get("SLDBENCH_MIN_TRAIN_SAMPLES", "1"))
MIN_VAL_SAMPLES = int(os.environ.get("SLDBENCH_MIN_VAL_SAMPLES", "1"))
MIN_TEST_SAMPLES = int(os.environ.get("SLDBENCH_MIN_TEST_SAMPLES", "1"))
VAL_RATIO = float(os.environ.get("SLDBENCH_VAL_RATIO", "0.2"))
VAL_SPLIT_SEED = int(os.environ.get("SLDBENCH_VAL_SPLIT_SEED", "42"))
SHUFFLE_BEFORE_SPLIT = os.environ.get("SLDBENCH_SHUFFLE", "1").lower() in {"1", "true", "yes", "y"}

RUN_TIMESTAMP = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{int((time.time() % 1.0) * 1000):03d}"
RESULTS_ROOT = os.environ.get(
    "SLDBENCH_V10_RESULTS_ROOT",
    str(PROJECT_ROOT / "runs" / f"sldbench_results_v10_{RUN_TIMESTAMP}"),
)
SELECTED_TASKS_CSV = os.path.join(RESULTS_ROOT, "sldbench_selected_tasks.csv")
GLOBAL_SUMMARY_CSV = os.path.join(RESULTS_ROOT, "all_sldbench_results_v10.csv")

MSE_THRESHOLD = float(os.environ.get("SLDBENCH_MSE_THRESHOLD", "100.0"))
PERFECT_FIT_TOL = float(os.environ.get("SLDBENCH_PERFECT_FIT_TOL", "1e-10"))


def import_v10_module():
    if not V10_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find v10 file: {V10_PATH}\n"
            "Set LLMSR_V10_PATH to the actual v10 python file. "
            "By default this script will try "
            "`structplan_sr_highdim_llm_companion_v10.py` and then `test_fey.py`."
        )
    spec = importlib.util.spec_from_file_location("structplan_v10_runtime", str(V10_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["structplan_v10_runtime"] = module
    spec.loader.exec_module(module)

    # Redirect v10 output paths to SLDBench-specific directory.
    module.RESULTS_ROOT = RESULTS_ROOT
    module.GLOBAL_SUMMARY_CSV = os.path.join(RESULTS_ROOT, "all_results_detailed.csv")
    module.GLOBAL_SUMMARY_JSON = os.path.join(RESULTS_ROOT, "global_summary.json")
    module.GLOBAL_SUMMARY_CSV_COMPACT = os.path.join(RESULTS_ROOT, "global_summary.csv")
    module.TIMING_BREAKDOWN_CSV = os.path.join(RESULTS_ROOT, "timing_breakdown.csv")
    module.TIMING_SUMMARY_JSON = os.path.join(RESULTS_ROOT, "timing_summary.json")
    module.PER_CASE_JSON_DIR = os.path.join(RESULTS_ROOT, "per_case_reports")
    module.SELECTED_TASKS_CSV = SELECTED_TASKS_CSV

    # Optional runtime override without editing v10 source.
    if os.environ.get("LLMSR_MAX_RUNTIME_PER_TASK_SEC"):
        module.MAX_RUNTIME_PER_TASK_SEC = float(os.environ["LLMSR_MAX_RUNTIME_PER_TASK_SEC"])

    # Keep reporting threshold consistent with this benchmark wrapper.
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


def normalize_target_names(target_name):
    if isinstance(target_name, list):
        return [str(x) for x in target_name]
    return [str(target_name)]


def split_train_val(X, y, val_ratio=0.2, seed=42, shuffle=True):
    n = len(X)
    if n < 2:
        raise ValueError(f"Not enough samples to split train/val: n={n}")
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def sanitize_variable_names(names):
    used = set()
    safe_names = []
    for idx, raw_name in enumerate(names or [], start=1):
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw_name).strip())
        safe = safe.strip("_")
        if not safe:
            safe = f"x{idx}"
        if safe[0].isdigit():
            safe = f"v_{safe}"
        if safe == "y":
            safe = "target_var"
        base = safe
        suffix = 2
        while safe in used:
            safe = f"{base}_{suffix}"
            suffix += 1
        used.add(safe)
        safe_names.append(safe)
    return safe_names


def make_xy_dataframe(X: np.ndarray, y: np.ndarray, keep_original_feature_names=None):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape={y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X/y length mismatch: {len(X)} vs {len(y)}")

    n_features = X.shape[1]
    if keep_original_feature_names:
        safe_feature_names = sanitize_variable_names(keep_original_feature_names[:n_features])
        if len(safe_feature_names) != n_features:
            raise ValueError(
                f"feature name count mismatch after sanitization: {len(safe_feature_names)} vs {n_features}"
            )
    else:
        safe_feature_names = [f"x{i+1}" for i in range(n_features)]
    cols = safe_feature_names + ["y"]
    arr = np.concatenate([X, y.reshape(-1, 1)], axis=1)
    df = pd.DataFrame(arr, columns=cols)
    if keep_original_feature_names:
        df.attrs["feature_map"] = {
            safe_feature_names[i]: keep_original_feature_names[i]
            for i in range(min(n_features, len(keep_original_feature_names)))
        }
        df.attrs["original_feature_names"] = [str(x) for x in keep_original_feature_names[:n_features]]
    return df


def load_sldbench_split(task_name: str, split: str):
    if task_name not in TASK_SCHEMA_MAP:
        raise ValueError(f"Unknown task_name: {task_name}")
    schema = TASK_SCHEMA_MAP[task_name]
    feature_names = [str(x) for x in schema["feature_names"]]
    target_names = normalize_target_names(schema["target_name"])

    try:
        ds = datasets.load_dataset(SLDBENCH_HUB_REPO, name=task_name, split=split)
    except Exception:
        cache_root = Path(os.environ.get("HF_DATASETS_CACHE", "")).expanduser()
        if not str(cache_root) or not cache_root.exists():
            cache_root = Path(os.environ.get("HF_HOME", "")).expanduser() / "datasets"
        arrow_matches = sorted(
            cache_root.glob(
                f"pkuHaowei___sldbench/{task_name}/0.0.0/*/sldbench-{split}.arrow"
            )
        )
        if not arrow_matches:
            raise
        ds = datasets.Dataset.from_file(str(arrow_matches[-1]))
    if "group" not in ds.column_names:
        raise KeyError(f"'group' column not found in task={task_name}, split={split}")

    out = {}
    for group_key in sorted(set(ds["group"])):
        group_ds = ds.filter(lambda example, g=group_key: example["group"] == g)
        X = np.stack([np.asarray(group_ds[fname], dtype=float) for fname in feature_names], axis=1)
        y = np.stack([np.asarray(group_ds[tname], dtype=float) for tname in target_names], axis=1)
        if y.shape[1] == 1:
            y = y.squeeze(axis=1)
        out[group_key] = {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "target_names": target_names,
        }
    return out


def task_filter_reason(task_name, group_key, case_name, n_features, n_train_full, n_test):
    if n_features < MIN_FEATURES:
        return f"n_features {n_features} < MIN_FEATURES {MIN_FEATURES}"
    if MAX_FEATURES is not None and n_features > MAX_FEATURES:
        return f"n_features {n_features} > MAX_FEATURES {MAX_FEATURES}"
    if n_train_full < MIN_TRAIN_SAMPLES:
        return f"n_train_full {n_train_full} < MIN_TRAIN_SAMPLES {MIN_TRAIN_SAMPLES}"
    if n_test < MIN_TEST_SAMPLES:
        return f"n_test {n_test} < MIN_TEST_SAMPLES {MIN_TEST_SAMPLES}"
    if ALLOW_TASKS and task_name not in ALLOW_TASKS:
        return f"task_name {task_name} not in ALLOW_TASKS"
    if ALLOW_GROUPS and str(group_key) not in ALLOW_GROUPS:
        return f"group_key {group_key} not in ALLOW_GROUPS"
    if ALLOW_CASE_NAMES and case_name not in ALLOW_CASE_NAMES:
        return f"case_name {case_name} not in ALLOW_CASE_NAMES"
    if NAME_KEYWORDS:
        low = f"{task_name} {group_key} {case_name}".lower()
        if not any(k in low for k in NAME_KEYWORDS):
            return f"no NAME_KEYWORDS matched: {NAME_KEYWORDS}"
    return None


def collect_sldbench_tasks():
    rows = []
    filtered = []
    task_names = [t for t in ALL_TASKS if not ALLOW_TASKS or t in ALLOW_TASKS]

    for task_name in task_names:
        print(f"\n[INFO] Loading SLDBench metadata: {task_name}")
        train_groups = load_sldbench_split(task_name, split="train")
        test_groups = load_sldbench_split(task_name, split="test")
        common_groups = sorted(set(train_groups.keys()) & set(test_groups.keys()))

        for group_key in common_groups:
            train_obj = train_groups[group_key]
            test_obj = test_groups[group_key]
            X_train_full = train_obj["X"]
            y_train_full = train_obj["y"]
            X_test = test_obj["X"]
            feature_names = train_obj["feature_names"]
            target_names = train_obj["target_names"]
            n_features = int(X_train_full.shape[1])
            n_train_full = int(len(X_train_full))
            n_test = int(len(X_test))

            if np.asarray(y_train_full).ndim == 1:
                target_items = [(0, target_names[0])]
            else:
                target_items = list(enumerate(target_names))

            for target_idx, target_name in target_items:
                case_name = f"{task_name}__group_{group_key}__{target_name}"
                reason = task_filter_reason(task_name, group_key, case_name, n_features, n_train_full, n_test)
                row = {
                    "task_name": task_name,
                    "group_key": group_key,
                    "target_idx": int(target_idx),
                    "target_name": str(target_name),
                    "case_name": case_name,
                    "base_name": case_name,
                    "feature_names": feature_names,
                    "n_features": n_features,
                    "n_train_full": n_train_full,
                    "n_test": n_test,
                    "filter_reason": reason,
                }
                if reason is None:
                    rows.append(row)
                else:
                    filtered.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("[WARN] No SLDBench cases selected.")
        if filtered:
            print(pd.DataFrame(filtered)[["case_name", "filter_reason"]].head(20).to_string(index=False))
        return df

    # Limit per task.
    if MAX_FILES_PER_TASK is not None:
        parts = []
        for _, part in df.groupby("task_name", sort=False):
            parts.append(part.head(MAX_FILES_PER_TASK))
        df = pd.concat(parts, axis=0).reset_index(drop=True)

    # Optional global random sample.
    if RANDOM_SAMPLE_K is not None and len(df) > RANDOM_SAMPLE_K:
        df = df.sample(n=RANDOM_SAMPLE_K, random_state=RANDOM_SEED).reset_index(drop=True)

    return df.reset_index(drop=True)


def load_sldbench_case(row):
    task_name = str(row["task_name"])
    group_key = row["group_key"]
    target_idx = int(row["target_idx"])

    train_groups = load_sldbench_split(task_name, split="train")
    test_groups = load_sldbench_split(task_name, split="test")
    if group_key not in train_groups:
        raise KeyError(f"group_key={group_key} not found in train split of {task_name}")
    if group_key not in test_groups:
        raise KeyError(f"group_key={group_key} not found in test split of {task_name}")

    train_obj = train_groups[group_key]
    test_obj = test_groups[group_key]
    X_train_full = np.asarray(train_obj["X"], dtype=float)
    y_train_full = np.asarray(train_obj["y"], dtype=float)
    X_test = np.asarray(test_obj["X"], dtype=float)
    y_test_full = np.asarray(test_obj["y"], dtype=float)
    feature_names = train_obj["feature_names"]

    y_train = y_train_full[:, target_idx] if y_train_full.ndim == 2 else y_train_full
    y_test = y_test_full[:, target_idx] if y_test_full.ndim == 2 else y_test_full

    X_train, y_train, X_val, y_val = split_train_val(
        X_train_full,
        y_train,
        val_ratio=VAL_RATIO,
        seed=VAL_SPLIT_SEED,
        shuffle=SHUFFLE_BEFORE_SPLIT,
    )

    if len(X_train) < MIN_TRAIN_SAMPLES:
        raise ValueError(f"train split too small after val split: {len(X_train)}")
    if len(X_val) < MIN_VAL_SAMPLES:
        raise ValueError(f"val split too small: {len(X_val)}")
    if len(X_test) < MIN_TEST_SAMPLES:
        raise ValueError(f"test split too small: {len(X_test)}")

    train_df = make_xy_dataframe(X_train, y_train, keep_original_feature_names=feature_names)
    val_df = make_xy_dataframe(X_val, y_val, keep_original_feature_names=feature_names)
    test_df = make_xy_dataframe(X_test, y_test, keep_original_feature_names=feature_names)
    return train_df, val_df, test_df


def make_row_meta(row, dataset):
    # v10 uses row_meta for logging, routing, no-leakage audit, and per-case naming.
    return {
        "task_type": "sldbench",
        "dataset_dir": "sldbench",
        "difficulty": str(row.get("task_name", "sldbench")),
        "base_name": str(row.get("case_name", row.get("base_name", "sldbench_case"))),
        "true_expression": None,  # SLDBench has no ground-truth formula exposed here.
        "task_name": str(row.get("task_name", "")),
        "group_key": str(row.get("group_key", "")),
        "target_idx": int(row.get("target_idx", 0)),
        "target_name": str(row.get("target_name", "")),
        "original_feature_names": " | ".join([str(x) for x in row.get("feature_names", [])]),
        "n_features": len(dataset.feature_names),
        "n_train": len(dataset.train_df),
        "n_val": len(dataset.val_df),
        "n_test": len(dataset.test_df),
    }


def run_one_sldbench_task_v10(v10, row, tmpdir: Path):
    start = time.time()
    result_prefix = {
        "sldbench_task_name": row.get("task_name"),
        "sldbench_group_key": row.get("group_key"),
        "sldbench_target_idx": row.get("target_idx"),
        "sldbench_target_name": row.get("target_name"),
        "case_name": row.get("case_name"),
        "original_feature_names": " | ".join([str(x) for x in row.get("feature_names", [])]),
    }
    try:
        train_df, val_df, test_df = load_sldbench_case(row)
        dataset = v10.build_dataset_from_explicit_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            tmpdir=tmpdir,
        )
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
            "task_type": "sldbench",
            "dataset_dir": "sldbench",
            "difficulty": row.get("task_name"),
            "base_name": row.get("case_name"),
            "true_expression": None,
            "n_features": row.get("n_features"),
            "n_train": None,
            "n_val": None,
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
    print(f"[{idx}/{total}] Processing: {row['case_name']}")
    print(f"   task_name:           {row['task_name']}")
    print(f"   group_key:           {row['group_key']}")
    print(f"   target_name:         {row['target_name']}")
    print(f"   original_features:   {row['feature_names']}")
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


def main():
    overall_start = time.time()
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    v10 = import_v10_module()
    print(f"[INFO] Using v10 file: {V10_PATH}")
    print(f"[INFO] Results root:   {RESULTS_ROOT}")
    print(f"[INFO] v10 profile:    {getattr(v10, 'EVAL_PROFILE', None)}")
    print(f"[INFO] v10 mode:       {getattr(v10, 'METHOD_MODE', None)}")

    df_tasks = collect_sldbench_tasks()
    df_tasks.to_csv(SELECTED_TASKS_CSV, index=False, encoding="utf-8-sig")
    print(f"Selection saved to: {SELECTED_TASKS_CSV}")

    if len(df_tasks) == 0:
        print("No SLDBench tasks selected.")
        return
    if DRY_RUN_SELECTION_ONLY:
        print("SLDBENCH_DRY_RUN=1, exiting after selection.")
        return

    print(f"[INFO] Selected {len(df_tasks)} SLDBench cases. Starting execution...", flush=True)

    all_results = []
    task_results_map = {}

    with TemporaryDirectory(prefix="sldbench_v10_tmp_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        total = len(df_tasks)
        for idx, (_, row) in enumerate(df_tasks.iterrows(), start=1):
            row_dict = row.to_dict()
            print(f"[{idx}/{total}] Starting: {row_dict['case_name']}", flush=True)
            res = run_one_sldbench_task_v10(v10, row_dict, tmpdir=tmpdir)
            all_results.append(res)
            task_results_map.setdefault(row_dict["task_name"], []).append(res)
            print_one_result(idx, total, row_dict, res)

            # Save incrementally for crash safety.
            pd.DataFrame(all_results).to_csv(GLOBAL_SUMMARY_CSV, index=False)

    for task_name, task_results in task_results_map.items():
        task_csv = os.path.join(RESULTS_ROOT, f"{sanitize_name(task_name)}_results.csv")
        pd.DataFrame(task_results).to_csv(task_csv, index=False)
        print(f"Saved task results to: {task_csv}")

    # Reuse v10's summary/output functions so timing/per-case summaries match v10 style.
    try:
        v10.save_all_outputs(all_results, overall_start)
        v10.print_summary(all_results, overall_start)
    except Exception as e:
        print(f"[WARN] v10 summary saving failed: {repr(e)}")
        pd.DataFrame(all_results).to_csv(GLOBAL_SUMMARY_CSV, index=False)
        print(f"Saved global results to: {GLOBAL_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
