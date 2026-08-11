#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run LLM-SRBench with StructPlan-SR v10/test_fey.

Core idea:
  LLM-SRBench is used only as the task/data provider.
  The actual solver is delegated to v10._run_core_pipeline(), so this script
  works as a benchmark wrapper in the same style as scripts/run_sldbench.py.

Data convention:
  - data/*.parquet       : metadata (name / symbols / expression / ...)
  - lsr_bench_data.hdf5  : numeric arrays for train / test / ood_test

Some LLM-SRBench groups, notably lsr_transform, ship only train/test.
For those cases this wrapper deterministically splits train into train/val
and keeps the HDF5 test split as the held-out test set.

Important convention:
  - Each HDF5 array is interpreted as:
      first n-1 columns -> inputs
      last 1 column     -> target y
"""

import os
import re
import sys
import time
import ast
import hashlib
import random
import warnings
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd
import yaml

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
    "LLMSRBENCH_V10_PATH",
    os.environ.get("LLMSR_V10_PATH", str(SCRIPT_DIR / "test_fey.py")),
)).resolve()


# =========================================================
# LLM-SRBench config
# =========================================================
LLMSRBENCH_ROOT = os.environ.get("LLMSRBENCH_ROOT", str(PROJECT_ROOT / "data" / "llmsrbench"))
LLMSRBENCH_HDF5 = os.environ.get("LLMSRBENCH_HDF5", os.path.join(LLMSRBENCH_ROOT, "lsr_bench_data.hdf5"))
LLMSRBENCH_CASES_ROOT = os.environ.get("LLMSRBENCH_CASES_ROOT", "")

PARQUET_FILES = {
    "lsr_transform": os.path.join(
        LLMSRBENCH_ROOT, "data", "lsr_transform-00000-of-00001.parquet"
    ),
    "lsr_synth_bio_pop_growth": os.path.join(
        LLMSRBENCH_ROOT, "data", "lsr_synth_bio_pop_growth-00000-of-00001.parquet"
    ),
    "lsr_synth_chem_react": os.path.join(
        LLMSRBENCH_ROOT, "data", "lsr_synth_chem_react-00000-of-00001.parquet"
    ),
    "lsr_synth_matsci": os.path.join(
        LLMSRBENCH_ROOT, "data", "lsr_synth_matsci-00000-of-00001.parquet"
    ),
    "lsr_synth_phys_osc": os.path.join(
        LLMSRBENCH_ROOT, "data", "lsr_synth_phys_osc-00000-of-00001.parquet"
    ),
}

MIN_FEATURES = int(os.environ.get("LLMSRBENCH_MIN_FEATURES", "1"))
max_features_env = os.environ.get("LLMSRBENCH_MAX_FEATURES", "8")
MAX_FEATURES = None if str(max_features_env).lower() in {"none", "", "0"} else int(max_features_env)

ALLOW_CASE_NAMES = [x.strip() for x in os.environ.get("LLMSRBENCH_ALLOW_CASE_NAMES", "").split(",") if x.strip()]
NAME_KEYWORDS = [x.strip().lower() for x in os.environ.get("LLMSRBENCH_NAME_KEYWORDS", "").split(",") if x.strip()]

default_allow_splits = ""
ALLOW_SPLITS = [x.strip() for x in os.environ.get("LLMSRBENCH_ALLOW_SPLITS", default_allow_splits).split(",") if x.strip()]

MAX_FILES = os.environ.get("LLMSRBENCH_MAX_FILES", "none")
MAX_FILES = None if str(MAX_FILES).lower() in {"none", "", "0"} else int(MAX_FILES)

RANDOM_SAMPLE_K = os.environ.get("LLMSRBENCH_RANDOM_SAMPLE_K", "")
RANDOM_SAMPLE_K = None if RANDOM_SAMPLE_K == "" else int(RANDOM_SAMPLE_K)
RANDOM_SEED = int(os.environ.get("LLMSRBENCH_RANDOM_SEED", "42"))
DRY_RUN_SELECTION_ONLY = os.environ.get("LLMSRBENCH_DRY_RUN", "0").lower() in {"1", "true", "yes", "y"}

RUN_TIMESTAMP = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime())}_{int((time.time() % 1.0) * 1000):03d}"
RESULTS_ROOT = os.environ.get(
    "LLMSRBENCH_V10_RESULTS_ROOT",
    str(PROJECT_ROOT / "runs" / f"llmsrbench_results_v10_{RUN_TIMESTAMP}"),
)
SELECTED_TASKS_CSV = os.path.join(RESULTS_ROOT, "llmsrbench_selected_files.csv")
GLOBAL_SUMMARY_CSV = os.path.join(RESULTS_ROOT, "all_llmsrbench_results_v10.csv")

MSE_THRESHOLD = float(os.environ.get("LLMSRBENCH_MSE_THRESHOLD", "100.0"))
PERFECT_FIT_TOL = float(os.environ.get("LLMSRBENCH_PERFECT_FIT_TOL", "1e-10"))
FALLBACK_TO_TEST_WHEN_NO_OOD = os.environ.get(
    "LLMSRBENCH_FALLBACK_TO_TEST_WHEN_NO_OOD", "1"
).strip().lower() in {"1", "true", "yes", "y"}
FALLBACK_VAL_FRACTION = float(os.environ.get("LLMSRBENCH_FALLBACK_VAL_FRACTION", "0.2"))
FALLBACK_MIN_VAL_ROWS = int(os.environ.get("LLMSRBENCH_FALLBACK_MIN_VAL_ROWS", "8"))


def import_v10_module():
    if not V10_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find v10 file: {V10_PATH}\n"
            "Set LLMSRBENCH_V10_PATH or LLMSR_V10_PATH to the actual v10 python file."
        )

    spec = importlib.util.spec_from_file_location("llmsrbench_v10_runtime", str(V10_PATH))
    module = importlib.util.module_from_spec(spec)
    sys.modules["llmsrbench_v10_runtime"] = module
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


def normalize_symbols(symbols):
    if symbols is None:
        return []
    if isinstance(symbols, np.ndarray):
        return [str(x) for x in symbols.tolist()]
    if isinstance(symbols, (list, tuple)):
        return [str(x) for x in symbols]
    return [str(symbols)]


def normalize_symbol_properties(values):
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return [str(x) for x in values.tolist()]
    if isinstance(values, (list, tuple)):
        return [str(x) for x in values]
    return [str(values)]


def count_effective_features(symbols, symbol_properties):
    symbol_properties = normalize_symbol_properties(symbol_properties)
    if symbol_properties and len(symbol_properties) == len(symbols):
        non_output = [x for x in symbol_properties if str(x).upper() != "O"]
        if len(non_output) >= 1:
            return len(non_output)
    return len(symbols)


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


def map_split_to_hdf5_group(split_name: str) -> str:
    mapping = {
        "lsr_synth_bio_pop_growth": "lsr_synth/bio_pop_growth",
        "lsr_synth_chem_react": "lsr_synth/chem_react",
        "lsr_synth_matsci": "lsr_synth/matsci",
        "lsr_synth_phys_osc": "lsr_synth/phys_osc",
        "lsr_transform": "lsr_transform",
    }
    if split_name not in mapping:
        raise ValueError(f"unknown split_name: {split_name}")
    return mapping[split_name]


DIRECTORY_FAMILY_TO_SPLIT = {
    "bio_pop_growth": "lsr_synth_bio_pop_growth",
    "chem_react": "lsr_synth_chem_react",
    "matsci": "lsr_synth_matsci",
    "phys_osc": "lsr_synth_phys_osc",
    "lsrtransform": "lsr_transform",
}


def formula_expression_from_file(path: Path) -> str:
    if not path.exists():
        return ""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for statement in ast.walk(node):
            if isinstance(statement, ast.Return) and statement.value is not None:
                if hasattr(ast, "unparse"):
                    return ast.unparse(statement.value)
                import astor

                return astor.to_source(statement.value).strip()
    return ""


def collect_llmsrbench_directory_tasks(root: Path) -> pd.DataFrame:
    rows = []
    for metadata_path in sorted(root.glob("*/*/metadata.yaml")):
        case_dir = metadata_path.parent
        family_name = case_dir.parent.name
        split_name = DIRECTORY_FAMILY_TO_SPLIT.get(family_name)
        if split_name is None:
            continue
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        dataset_meta = dict(metadata.get("dataset") or {})
        features = list(dataset_meta.get("features") or [])
        target = dict(dataset_meta.get("target") or {})
        feature_names = [str(item.get("name")) for item in features if item.get("name")]
        target_name = str(target.get("name") or "y")
        case_name = case_dir.name
        rows.append({
            "split_name": split_name,
            "hdf5_group_prefix": map_split_to_hdf5_group(split_name),
            "case_name": case_name,
            "symbols": feature_names + [target_name],
            "symbol_descs": [
                str(item.get("description") or "")
                for item in features
            ] + [str(target.get("description") or "")],
            "symbol_properties": ["I"] * len(feature_names) + ["O"],
            "true_expression": formula_expression_from_file(case_dir / "formula.py"),
            "n_features_meta": len(feature_names),
            "case_dir": str(case_dir.resolve()),
        })
    return pd.DataFrame(rows)


def task_passes_filters(split_name, case_name, n_features_meta):
    if n_features_meta < MIN_FEATURES:
        return False
    if MAX_FEATURES is not None and n_features_meta > MAX_FEATURES:
        return False
    if ALLOW_CASE_NAMES and case_name not in ALLOW_CASE_NAMES:
        return False
    if ALLOW_SPLITS and split_name not in ALLOW_SPLITS:
        return False
    if NAME_KEYWORDS:
        low = case_name.lower()
        if not any(k in low for k in NAME_KEYWORDS):
            return False
    return True


def collect_llmsrbench_tasks():
    if LLMSRBENCH_CASES_ROOT:
        directory_root = Path(LLMSRBENCH_CASES_ROOT).expanduser().resolve()
        if not directory_root.exists():
            raise FileNotFoundError(f"LLMSRBENCH_CASES_ROOT not found: {directory_root}")
        df = collect_llmsrbench_directory_tasks(directory_root)
        if len(df) == 0:
            raise FileNotFoundError(f"No LLMSRBench directory tasks found under: {directory_root}")
        keep = df.apply(
            lambda r: task_passes_filters(
                r["split_name"], r["case_name"], r["n_features_meta"]
            ),
            axis=1,
        )
        df = df[keep].copy()
        df = df.sort_values(by=["split_name", "n_features_meta", "case_name"]).reset_index(drop=True)
        if RANDOM_SAMPLE_K is not None and RANDOM_SAMPLE_K > 0 and len(df) > RANDOM_SAMPLE_K:
            rng = random.Random(RANDOM_SEED)
            idxs = list(range(len(df)))
            rng.shuffle(idxs)
            df = df.iloc[sorted(idxs[:RANDOM_SAMPLE_K])].reset_index(drop=True)
        if MAX_FILES is not None:
            df = df.head(MAX_FILES).copy()
        return df.reset_index(drop=True)

    rows = []

    for split_name, parquet_path in PARQUET_FILES.items():
        if not os.path.exists(parquet_path):
            print(f"[WARN] parquet not found: {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)
        if len(df) == 0:
            continue

        for _, r in df.iterrows():
            case_name = str(r["name"])
            symbols = normalize_symbols(r.get("symbols", []))
            symbol_descs = normalize_symbols(r.get("symbol_descs", []))
            symbol_properties = normalize_symbol_properties(r.get("symbol_properties", []))
            true_expression = str(r.get("expression", "")).strip()
            effective_n_features = count_effective_features(symbols, symbol_properties)

            rows.append({
                "split_name": split_name,
                "hdf5_group_prefix": map_split_to_hdf5_group(split_name),
                "case_name": case_name,
                "symbols": symbols,
                "symbol_descs": symbol_descs,
                "symbol_properties": symbol_properties,
                "true_expression": true_expression,
                "n_features_meta": effective_n_features,
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    print("\n[DEBUG] all raw tasks before filtering:")
    print(df[["split_name", "case_name", "n_features_meta"]].to_string(index=False))

    keep = df.apply(
        lambda r: task_passes_filters(
            r["split_name"], r["case_name"], r["n_features_meta"]
        ),
        axis=1,
    )
    df = df[keep].copy()

    print("\n[DEBUG] raw tasks after filtering:")
    if len(df) == 0:
        print("(none)")
    else:
        print(df[["split_name", "case_name", "n_features_meta"]].to_string(index=False))

    df = df.sort_values(by=["split_name", "n_features_meta", "case_name"]).reset_index(drop=True)

    if RANDOM_SAMPLE_K is not None and RANDOM_SAMPLE_K > 0 and len(df) > RANDOM_SAMPLE_K:
        rng = random.Random(RANDOM_SEED)
        idxs = list(range(len(df)))
        rng.shuffle(idxs)
        idxs = sorted(idxs[:RANDOM_SAMPLE_K])
        df = df.iloc[idxs].reset_index(drop=True)

    if MAX_FILES is not None:
        df = df.head(MAX_FILES).copy()

    return df.reset_index(drop=True)


def infer_io_layout(symbols, symbol_properties, n_cols):
    symbols = list(symbols or [])
    symbol_properties = list(symbol_properties or [])

    if symbol_properties and len(symbol_properties) == n_cols:
        output_indices = [i for i, x in enumerate(symbol_properties) if str(x).upper() == "O"]
        if len(output_indices) == 1:
            output_idx = output_indices[0]
            input_indices = [i for i in range(n_cols) if i != output_idx]
            return input_indices, output_idx, "symbol_properties"

    input_indices = list(range(max(0, n_cols - 1)))
    output_idx = n_cols - 1
    return input_indices, output_idx, "fallback_last_column"


def build_feature_names(symbols, input_indices):
    symbols = list(symbols or [])
    used = set()
    out = []
    for order, col_idx in enumerate(input_indices, start=1):
        raw_name = symbols[col_idx] if col_idx < len(symbols) else f"x{order}"
        out.append(sanitize_variable_name(raw_name, used_names=used))
    return out


def make_xy_dataframe(arr: np.ndarray, symbols=None, symbol_properties=None) -> pd.DataFrame:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"array must be 2D, got shape={arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(f"invalid array shape={arr.shape}, need at least 2 columns")

    input_indices, output_idx, layout_source = infer_io_layout(symbols, symbol_properties, arr.shape[1])
    feature_names = build_feature_names(symbols, input_indices)
    X = arr[:, input_indices]
    y = arr[:, output_idx]

    df = pd.DataFrame(X, columns=feature_names)
    df["y"] = y
    df.attrs["layout_source"] = layout_source
    df.attrs["input_indices"] = list(input_indices)
    df.attrs["output_idx"] = int(output_idx)
    df.attrs["feature_names_original"] = [
        str(symbols[i]) if i < len(symbols) else f"x{j+1}"
        for j, i in enumerate(input_indices)
    ]
    if symbols and output_idx < len(symbols):
        df.attrs["target_name_original"] = str(symbols[output_idx])
    return df


def deterministic_train_val_split(arr: np.ndarray, case_name: str):
    arr = np.asarray(arr)
    n_rows = int(arr.shape[0])
    if n_rows < 2:
        return arr, arr.copy()

    val_rows = max(int(FALLBACK_MIN_VAL_ROWS), int(round(n_rows * FALLBACK_VAL_FRACTION)))
    val_rows = min(max(1, val_rows), n_rows - 1)

    seed = int(hashlib.md5(str(case_name).encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    val_idx = np.sort(perm[:val_rows])
    train_idx = np.sort(perm[val_rows:])
    return arr[train_idx], arr[val_idx]


def load_llmsrbench_case_from_hdf5(row):
    case_name = str(row["case_name"])
    split_name = str(row["split_name"])
    group_prefix = str(row["hdf5_group_prefix"])
    symbols = list(row.get("symbols", []))
    symbol_properties = list(row.get("symbol_properties", []))

    train_key = f"{group_prefix}/{case_name}/train"
    test_key = f"{group_prefix}/{case_name}/test"
    ood_test_key = f"{group_prefix}/{case_name}/ood_test"
    val_source = "test"
    test_source = "ood_test"
    fallback_note = None

    with h5py.File(LLMSRBENCH_HDF5, "r") as f:
        if train_key not in f:
            raise KeyError(f"missing dataset in hdf5: {train_key}")
        if test_key not in f:
            raise KeyError(f"missing dataset in hdf5: {test_key}")

        train_arr = f[train_key][:]
        test_arr = f[test_key][:]
        if ood_test_key in f:
            val_arr = test_arr
            final_test_arr = f[ood_test_key][:]
        elif FALLBACK_TO_TEST_WHEN_NO_OOD:
            train_arr, val_arr = deterministic_train_val_split(train_arr, case_name)
            final_test_arr = test_arr
            val_source = "train_split"
            test_source = "test"
            fallback_note = f"missing {ood_test_key}; split train for val and used test as held-out test"
        else:
            raise KeyError(f"missing dataset in hdf5: {ood_test_key}")

    train_df = make_xy_dataframe(train_arr, symbols=symbols, symbol_properties=symbol_properties)
    val_df = make_xy_dataframe(val_arr, symbols=symbols, symbol_properties=symbol_properties)
    test_df = make_xy_dataframe(final_test_arr, symbols=symbols, symbol_properties=symbol_properties)

    extra_info = {
        "split_name": split_name,
        "case_name": case_name,
        "symbols": symbols,
        "symbol_descs": list(row.get("symbol_descs", [])),
        "symbol_properties": list(row.get("symbol_properties", [])),
        "true_expression": row.get("true_expression"),
        "train_shape": tuple(train_arr.shape),
        "val_shape": tuple(val_arr.shape),
        "test_shape": tuple(final_test_arr.shape),
        "val_source": val_source,
        "test_source": test_source,
        "split_fallback_note": fallback_note,
        "n_features_inferred": int(train_df.shape[1] - 1),
        "hdf5_group_prefix": group_prefix,
        "layout_source": train_df.attrs.get("layout_source"),
        "input_indices": train_df.attrs.get("input_indices"),
        "output_idx": train_df.attrs.get("output_idx"),
        "feature_names_original": train_df.attrs.get("feature_names_original"),
        "target_name_original": train_df.attrs.get("target_name_original"),
    }
    return train_df, val_df, test_df, extra_info


def load_llmsrbench_case_from_directory(row):
    case_dir = Path(str(row["case_dir"]))
    symbols = list(row.get("symbols", []))
    target_name = symbols[-1] if symbols else "y"
    feature_names = symbols[:-1]

    def load_split(filename: str) -> pd.DataFrame:
        path = case_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"missing LLMSRBench split: {path}")
        frame = pd.read_csv(path)
        required = set(feature_names + [target_name])
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{path} missing columns: {sorted(missing)}")
        out = frame[feature_names + [target_name]].copy()
        out = out.rename(columns={target_name: "y"})
        return out

    train_df = load_split("train.csv")
    val_df = load_split("valid.csv")
    test_df = load_split("id_test.csv")
    extra_info = {
        "split_name": row.get("split_name"),
        "case_name": row.get("case_name"),
        "symbols": symbols,
        "symbol_descs": list(row.get("symbol_descs", [])),
        "symbol_properties": list(row.get("symbol_properties", [])),
        "true_expression": row.get("true_expression"),
        "train_shape": tuple(train_df.shape),
        "val_shape": tuple(val_df.shape),
        "test_shape": tuple(test_df.shape),
        "val_source": "official_valid",
        "test_source": "official_id_test",
        "split_fallback_note": None,
        "n_features_inferred": len(feature_names),
        "hdf5_group_prefix": row.get("hdf5_group_prefix"),
        "layout_source": "directory_csv",
        "input_indices": list(range(len(feature_names))),
        "output_idx": len(feature_names),
        "feature_names_original": feature_names,
        "target_name_original": target_name,
    }
    return train_df, val_df, test_df, extra_info


def load_llmsrbench_case(row):
    if str(row.get("case_dir", "") or "").strip():
        return load_llmsrbench_case_from_directory(row)
    return load_llmsrbench_case_from_hdf5(row)


def make_row_meta(row, dataset, extra_info):
    return {
        "task_type": "llmsrbench",
        "dataset_dir": "llmsrbench",
        "difficulty": str(row.get("split_name", "llmsrbench")),
        "base_name": str(row.get("case_name", "llmsrbench_case")),
        "true_expression": row.get("true_expression"),
        "split_name": str(row.get("split_name", "")),
        "case_name": str(row.get("case_name", "")),
        "hdf5_group_prefix": str(row.get("hdf5_group_prefix", "")),
        "original_symbols": " | ".join([str(x) for x in row.get("symbols", [])]),
        "symbol_descs": " | ".join([str(x) for x in row.get("symbol_descs", [])]),
        "symbol_properties": " | ".join([str(x) for x in row.get("symbol_properties", [])]),
        "feature_names_original": " | ".join([str(x) for x in (extra_info.get("feature_names_original") or [])]),
        "target_name_original": str(extra_info.get("target_name_original", "")),
        "layout_source": str(extra_info.get("layout_source", "")),
        "input_indices": str(extra_info.get("input_indices")),
        "output_idx": str(extra_info.get("output_idx")),
        "train_shape": str(extra_info.get("train_shape")),
        "val_shape": str(extra_info.get("val_shape")),
        "test_shape": str(extra_info.get("test_shape")),
        "val_source": str(extra_info.get("val_source", "")),
        "test_source": str(extra_info.get("test_source", "")),
        "split_fallback_note": str(extra_info.get("split_fallback_note") or ""),
        "n_features": len(dataset.feature_names),
        "n_train": len(dataset.train_df),
        "n_val": len(dataset.val_df),
        "n_test": len(dataset.test_df),
    }


def run_one_llmsrbench_task_v10(v10, row, tmpdir: Path):
    start = time.time()
    result_prefix = {
        "llmsrbench_split_name": row.get("split_name"),
        "llmsrbench_case_name": row.get("case_name"),
        "case_name": row.get("case_name"),
        "true_expression": row.get("true_expression"),
        "original_symbols": " | ".join([str(x) for x in row.get("symbols", [])]),
    }
    try:
        train_df, val_df, test_df, extra_info = load_llmsrbench_case(row)
        dataset = v10.build_dataset_from_explicit_splits(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            tmpdir=tmpdir,
        )
        dataset.source_tag = "llmsrbench"
        row_meta = make_row_meta(row, dataset, extra_info)
        out = v10._run_core_pipeline(dataset=dataset, row_meta=row_meta)
        out.update(result_prefix)
        out["feature_names_original"] = " | ".join([str(x) for x in (extra_info.get("feature_names_original") or [])])
        out["target_name_original"] = extra_info.get("target_name_original")
        out["layout_source"] = extra_info.get("layout_source")
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
            "task_type": "llmsrbench",
            "dataset_dir": "llmsrbench",
            "difficulty": row.get("split_name"),
            "base_name": row.get("case_name"),
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
    print(f"[{idx}/{total}] Processing: {row['case_name']}")
    print(f"   split_name:          {row['split_name']}")
    print(f"   true_expression:     {row['true_expression']}")
    print(f"   original_symbols:    {row['symbols']}")
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


def main():
    overall_start = time.time()
    os.makedirs(RESULTS_ROOT, exist_ok=True)

    if not LLMSRBENCH_CASES_ROOT and not os.path.exists(LLMSRBENCH_HDF5):
        raise FileNotFoundError(f"hdf5 not found: {LLMSRBENCH_HDF5}")

    df_tasks = collect_llmsrbench_tasks()
    df_tasks.to_csv(SELECTED_TASKS_CSV, index=False, encoding="utf-8-sig")
    print(f"Selection saved to: {SELECTED_TASKS_CSV}")

    if len(df_tasks) == 0:
        print("没有筛选到可运行的 llm-srbench 任务。")
        return
    if DRY_RUN_SELECTION_ONLY:
        print("LLMSRBENCH_DRY_RUN=1，仅完成任务筛选。")
        return

    v10 = import_v10_module()
    print(f"[INFO] Using v10 file: {V10_PATH}")
    print(f"[INFO] Results root:   {RESULTS_ROOT}")
    print(f"[INFO] v10 profile:    {getattr(v10, 'EVAL_PROFILE', None)}")
    print(f"[INFO] v10 mode:       {getattr(v10, 'METHOD_MODE', None)}")

    print(f"[INFO] Selected {len(df_tasks)} LLM-SRBench cases. Starting execution...", flush=True)

    all_results = []
    split_results_map = {}

    with TemporaryDirectory(prefix="llmsrbench_v10_tmp_") as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        total = len(df_tasks)
        for idx, (_, row) in enumerate(df_tasks.iterrows(), start=1):
            row_dict = row.to_dict()
            print(f"[{idx}/{total}] Starting: {row_dict['case_name']}", flush=True)
            res = run_one_llmsrbench_task_v10(v10, row_dict, tmpdir=tmpdir)
            all_results.append(res)
            split_results_map.setdefault(row_dict["split_name"], []).append(res)
            print_one_result(idx, total, row_dict, res)

            pd.DataFrame(all_results).to_csv(GLOBAL_SUMMARY_CSV, index=False)

    for split_name, split_results in split_results_map.items():
        split_csv = os.path.join(RESULTS_ROOT, f"{sanitize_name(split_name)}_results.csv")
        pd.DataFrame(split_results).to_csv(split_csv, index=False)
        print(f"Saved split results to: {split_csv}")

    try:
        v10.save_all_outputs(all_results, overall_start)
        v10.print_summary(all_results, overall_start)
    except Exception as e:
        print(f"[WARN] v10 summary saving failed: {repr(e)}")
        pd.DataFrame(all_results).to_csv(GLOBAL_SUMMARY_CSV, index=False)
        print(f"Saved global results to: {GLOBAL_SUMMARY_CSV}")
        print_local_summary("ALL LLM-SRBENCH FILES", all_results)


if __name__ == "__main__":
    main()
