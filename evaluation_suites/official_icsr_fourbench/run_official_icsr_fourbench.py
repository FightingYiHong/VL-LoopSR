#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the official ICSR code on the local four-benchmark suite.

This is a thin adapter around merlerm/In-Context-Symbolic-Regression. It
creates official Hydra function configs plus train/test point files, runs the
official ``main.py`` entry point, and converts its ``results.json`` into the
same per-case metric shape used by our other four-benchmark baselines.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
OFFICIAL_ROOT = Path(
    os.environ.get(
        "OFFICIAL_ICSR_ROOT",
        str(PROJECT_ROOT / "data" / "external" / "methods" / "In-Context-Symbolic-Regression"),
    )
)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import run_cpu_baseline_benchmarks as base


BENCHMARKS = ("llmsrbench", "srsd", "srbench")
MODEL_NAME = os.environ.get("OFFICIAL_ICSR_MODEL_NAME", "llm-baseline-qwen2.5-32b")
OPENAI_BASE_URL = os.environ.get("OFFICIAL_ICSR_OPENAI_BASE_URL", "http://127.0.0.1:8001/v1")
OPENAI_API_KEY = os.environ.get("OFFICIAL_ICSR_OPENAI_API_KEY", "EMPTY")
LLMSRBENCH_ROOT = Path(os.environ.get("LLMSRBENCH_ROOT", str(PROJECT_ROOT / "data" / "external" / "llmsrbench")))
LLMSRBENCH_HDF5 = Path(os.environ.get("LLMSRBENCH_HDF5", str(LLMSRBENCH_ROOT / "lsr_bench_data.hdf5")))

# The shared benchmark loader reads these variables when it imports
# scripts/run_llmsrbench.py. Set remote-safe defaults before collect_tasks().
os.environ.setdefault("LLMSRBENCH_ROOT", str(LLMSRBENCH_ROOT))
os.environ.setdefault("LLMSRBENCH_HDF5", str(LLMSRBENCH_HDF5))


def safe_float(value):
    try:
        value = float(value)
    except Exception:
        return None
    return value if math.isfinite(value) else None


def safe_mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred) or not np.all(np.isfinite(y_pred)):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.mean((y_true - y_pred) ** 2))
    return value if math.isfinite(value) else None


def safe_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred) or not np.all(np.isfinite(y_pred)):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        value = float(np.mean(np.abs(y_true - y_pred)))
    return value if math.isfinite(value) else None


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_pred) or not np.all(np.isfinite(y_pred)):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if not math.isfinite(ss_res) or not math.isfinite(ss_tot):
        return None
    if ss_tot <= 0:
        return 1.0 if ss_res <= 1e-10 else 0.0
    value = float(1.0 - ss_res / ss_tot)
    return value if math.isfinite(value) else None


def sanitize(text: str) -> str:
    return base.sanitize_name(str(text))[:140]


def dataframe_to_points(df: pd.DataFrame) -> np.ndarray:
    X, y, _ = base.dataframe_to_xy(df)
    return np.concatenate([X, y.reshape(-1, 1)], axis=1).astype(float)


def ensure_official_compatibility(
    official_root: Path,
    *,
    max_new_tokens: int = 2048,
    top_p: float = 0.90,
    top_k: int = 60,
    temperature: float = 1.0,
) -> None:
    """Add only vLLM/OpenAI-compatible plumbing to the official checkout."""
    compatibility_lock = open(
        official_root / ".icsr_compatibility.lock",
        "a+",
        encoding="utf-8",
    )
    fcntl.flock(compatibility_lock.fileno(), fcntl.LOCK_EX)
    model_file = official_root / "models" / "openai_model.py"
    utils_file = official_root / "utils.py"
    main_file = official_root / "main.py"
    optimizer_file = official_root / "optimizer.py"
    current_functions_file = official_root / "current_functions.py"
    models_init_file = official_root / "models" / "__init__.py"
    if not model_file.exists() or not utils_file.exists():
        raise FileNotFoundError(f"official ICSR checkout is incomplete: {official_root}")

    text = model_file.read_text(encoding="utf-8")
    if "OPENAI_BASE_URL" not in text:
        text = text.replace(
            "self.client = openai.Client(\n"
            "            api_key=self.api_key,\n"
            "            organization=self.organization_id\n"
            "        )",
            "client_kwargs = {\"api_key\": self.api_key, \"organization\": self.organization_id}\n"
            "        base_url = os.environ.get(\"OPENAI_BASE_URL\")\n"
            "        if base_url:\n"
            "            client_kwargs[\"base_url\"] = base_url\n"
            "        self.client = openai.Client(**client_kwargs)",
        )
        model_file.write_text(text, encoding="utf-8")

    text = utils_file.read_text(encoding="utf-8")
    if "ICSR_FORCE_OPENAI" not in text:
        text = text.replace(
            "elif 'gpt' in model_name:\n"
            "        model = OpenAIModel(model_name, device, dtype, cache_dir, **model_args)",
            "elif 'gpt' in model_name or os.environ.get('ICSR_FORCE_OPENAI') == '1' or os.environ.get('OPENAI_BASE_URL'):\n"
            "        model = OpenAIModel(model_name, device, dtype, cache_dir, **model_args)",
        )
        utils_file.write_text(text, encoding="utf-8")

    text = models_init_file.read_text(encoding="utf-8")
    if "ICSR compatibility: model backends are imported lazily" not in text:
        eager_model_imports = (
            "from .hf_model import HuggingFaceModel\n"
            "from .llava_model_hf import LLaVaModelHF\n"
            "from .openai_model import OpenAIModel"
        )
        if eager_model_imports not in text:
            raise RuntimeError(
                "Could not locate eager model imports; refusing to apply an "
                "unaudited lazy-backend compatibility patch."
            )
        text = text.replace(
            eager_model_imports,
            "# ICSR compatibility: model backends are imported lazily by utils.py.\n"
            "# This prevents unused Hugging Face and vision registries from loading\n"
            "# in the OpenAI-compatible benchmark path.",
        )
        models_init_file.write_text(text, encoding="utf-8")

    text = utils_file.read_text(encoding="utf-8")
    if "ICSR compatibility: import only the selected model backend" not in text:
        eager_utils_import = (
            "from models import LLaVaModelHF, HuggingFaceModel, OpenAIModel"
        )
        if eager_utils_import not in text:
            raise RuntimeError(
                "Could not locate utility model imports; refusing to apply an "
                "unaudited lazy-backend compatibility patch."
            )
        text = text.replace(
            eager_utils_import,
            "# ICSR compatibility: import only the selected model backend.\n"
            "from models.openai_model import OpenAIModel",
        )
        text = text.replace(
            "    if 'llava' in model_name:\n"
            "        model = LLaVaModelHF(model_name, device, dtype, cache_dir, **model_args)",
            "    if 'llava' in model_name:\n"
            "        from models.llava_model_hf import LLaVaModelHF\n"
            "        model = LLaVaModelHF(model_name, device, dtype, cache_dir, **model_args)",
        )
        text = text.replace(
            "    else:\n"
            "        model = HuggingFaceModel(model_name, device, dtype, cache_dir, **model_args)",
            "    else:\n"
            "        from models.hf_model import HuggingFaceModel\n"
            "        model = HuggingFaceModel(model_name, device, dtype, cache_dir, **model_args)",
        )
        utils_file.write_text(text, encoding="utf-8")

    text = main_file.read_text(encoding="utf-8")
    if "ICSR compatibility: lightweight local set_seed" not in text:
        transformers_seed_import = "from transformers import set_seed\n"
        if transformers_seed_import not in text:
            raise RuntimeError(
                "Could not locate the official transformers set_seed import; "
                "refusing to apply an unaudited startup patch."
            )
        text = text.replace(
            transformers_seed_import,
            "# ICSR compatibility: lightweight local set_seed avoids importing the full\n"
            "# transformers model registry in each isolated benchmark process. This\n"
            "# preserves the RNG semantics used here without changing the SR algorithm.\n"
            "import random\n\n\n"
            "def set_seed(seed):\n"
            "    random.seed(seed)\n"
            "    np.random.seed(seed)\n"
            "    torch.manual_seed(seed)\n"
            "    if torch.cuda.is_available():\n"
            "        torch.cuda.manual_seed_all(seed)\n",
        )
        main_file.write_text(text, encoding="utf-8")

    text = main_file.read_text(encoding="utf-8")
    if '"candidate_functions": [' not in text:
        text = text.replace(
            '            "best_found_at": 0,\n'
            '            "sympy_equivalent": False,',
            '            "best_found_at": 0,\n'
            '            # ICSR compatibility: preserve optimized candidates for validation search-efficiency.\n'
            '            "candidate_functions": [\n'
            '                {"expression": str(value), "stage": "seed", "iteration": 0}\n'
            '                for value in self.current_functions.functions.values()\n'
            '            ],\n'
            '            "sympy_equivalent": False,',
        )
        text = text.replace(
            '                    score = self.scorer.score(opt_function)\n'
            '                    self.logger.info(f"New function: {str(opt_function)}. Score: {score}.")',
            '                    score = self.scorer.score(opt_function)\n'
            '                    # ICSR compatibility: candidate order is needed for post-hoc validation R2.\n'
            '                    self.results["candidate_functions"].append({\n'
            '                        "expression": str(opt_function),\n'
            '                        "stage": "iteration",\n'
            '                        "iteration": len(self.results["tries_per_iteration"]),\n'
            '                        "training_score": float(score),\n'
            '                    })\n'
            '                    self.logger.info(f"New function: {str(opt_function)}. Score: {score}.")',
        )
        main_file.write_text(text, encoding="utf-8")

    text = main_file.read_text(encoding="utf-8")
    legacy_ready_marker = (
        "    # ICSR compatibility: exact boundary between process setup and search.\n"
        "    print(\"ICSR_BENCHMARK_READY\", flush=True)\n"
    )
    if legacy_ready_marker in text:
        text = text.replace(legacy_ready_marker, "")
    if "ICSR_SEARCH_READY" not in text:
        text = text.replace(
            "        # Seed functions\n",
            "        # ICSR compatibility: include generated seeds, optimization, and\n"
            "        # iterative refinement in the benchmark search budget.\n"
            "        print(\"ICSR_SEARCH_READY\", flush=True)\n\n"
            "        # Seed functions\n",
        )
    main_file.write_text(text, encoding="utf-8")

    text = main_file.read_text(encoding="utf-8")
    if "ICSR compatibility: stream candidate history for timeout recovery" not in text:
        text = text.replace(
            '        if "test_function" in self.cfg.experiment.function:\n',
            '        # ICSR compatibility: stream candidate history for timeout recovery.\n'
            '        with open(self.output_path + "candidate_functions.jsonl", "w") as candidate_fp:\n'
            '            for candidate in self.results["candidate_functions"]:\n'
            '                candidate_fp.write(json.dumps(candidate) + "\\n")\n'
            '        if "test_function" in self.cfg.experiment.function:\n',
        )
        text = text.replace(
            '                        "training_score": float(score),\n'
            '                    })\n'
            '                    self.logger.info(f"New function: {str(opt_function)}. Score: {score}.")',
            '                        "training_score": float(score),\n'
            '                    })\n'
            '                    with open(self.output_path + "candidate_functions.jsonl", "a") as candidate_fp:\n'
            '                        candidate_fp.write(json.dumps(self.results["candidate_functions"][-1]) + "\\n")\n'
            '                    self.logger.info(f"New function: {str(opt_function)}. Score: {score}.")',
        )
        main_file.write_text(text, encoding="utf-8")

    text = optimizer_file.read_text(encoding="utf-8")
    if "ICSR compatibility: mloggers masks require logger classes" not in text:
        incompatible_warning_handler = (
            'warnings.showwarning = lambda *args, **kwargs: '
            'self.logger.warning(str(args[0]), mask=["file"])'
        )
        if incompatible_warning_handler not in text:
            raise RuntimeError(
                "Could not locate the official optimizer warning handler; "
                "refusing to apply an unaudited mloggers compatibility patch."
            )
        text = text.replace(
            incompatible_warning_handler,
            "# ICSR compatibility: mloggers masks require logger classes rather than\n"
            "        # string names. The upstream string mask raises TypeError whenever\n"
            "        # scipy emits a warning and incorrectly discards an otherwise fitted\n"
            "        # candidate. With one optimizer thread, normal logging is safe.\n"
            "        warnings.showwarning = lambda *args, **kwargs: self.logger.warning(str(args[0]))",
        )
        optimizer_file.write_text(text, encoding="utf-8")

    text = optimizer_file.read_text(encoding="utf-8")
    if "ICSR compatibility: retain finite fitted parameters" not in text:
        covariance_rejection = (
            "        if pcov is None or np.isinf(pcov).any() or np.isnan(pcov).any():\n"
            '            raise ValueError("Optimization failed: covariance matrix is invalid")\n'
        )
        if covariance_rejection not in text:
            raise RuntimeError(
                "Could not locate the official covariance check; "
                "refusing to apply an unaudited numerical compatibility patch."
            )
        text = text.replace(
            covariance_rejection,
            "        # ICSR compatibility: retain finite fitted parameters even when\n"
            "        # scipy cannot estimate their covariance. Covariance is not used\n"
            "        # downstream; discarding a finite predictive expression creates\n"
            "        # a false numerical failure.\n"
            "        if popt is None or not np.all(np.isfinite(popt)):\n"
            '            raise ValueError("Optimization failed: fitted parameters are invalid")\n'
            "        if pcov is None or np.isinf(pcov).any() or np.isnan(pcov).any():\n"
            '            self.logger.warning("ICSR compatibility: covariance unavailable; retaining finite fitted parameters.")\n',
        )
        optimizer_file.write_text(text, encoding="utf-8")

    text = main_file.read_text(encoding="utf-8")
    if "ICSR compatibility: initialize candidate stream before seed optimization" not in text:
        text = text.replace(
            "        self.current_functions = CurrentFunctions(self.seed_functions, self.scorer, self.optimizer, self.prompt_size, self.logger, self.num_variables)\n",
            "        # ICSR compatibility: initialize candidate stream before seed optimization\n"
            "        # so a time-limited run retains every successfully fitted candidate.\n"
            '        candidate_stream_path = self.output_path + "candidate_functions.jsonl"\n'
            '        open(candidate_stream_path, "w").close()\n'
            '        os.environ["ICSR_CANDIDATE_STREAM_PATH"] = candidate_stream_path\n'
            "        self.current_functions = CurrentFunctions(self.seed_functions, self.scorer, self.optimizer, self.prompt_size, self.logger, self.num_variables)\n",
        )
        main_file.write_text(text, encoding="utf-8")

    text = current_functions_file.read_text(encoding="utf-8")
    if "ICSR compatibility: persist each fitted seed immediately" not in text:
        text = text.replace("import re\n", "import re\nimport json\nimport os\n")
        text = text.replace(
            "                self.functions[coeff_function] = optimized_function\n"
            '                self.logger.info(f"Optimized seed function: {str(coeff_function)}.")',
            "                self.functions[coeff_function] = optimized_function\n"
            "                # ICSR compatibility: persist each fitted seed immediately.\n"
            '                stream_path = os.environ.get("ICSR_CANDIDATE_STREAM_PATH")\n'
            "                if stream_path:\n"
            '                    with open(stream_path, "a") as candidate_fp:\n'
            "                        candidate_fp.write(json.dumps({\n"
            '                            "expression": str(optimized_function),\n'
            '                            "stage": "seed",\n'
            '                            "iteration": 0,\n'
            '                        }) + "\\n")\n'
            '                self.logger.info(f"Optimized seed function: {str(coeff_function)}.")',
        )
        current_functions_file.write_text(text, encoding="utf-8")

    text = main_file.read_text(encoding="utf-8")
    if "ICSR compatibility: initialize visual_model before dimensionality check" not in text:
        text = text.replace(
            "self.num_variables = cfg.experiment.function.num_variables\n"
            "        if self.num_variables > 2 and self.visual_model:",
            "self.num_variables = cfg.experiment.function.num_variables\n"
            "        # ICSR compatibility: initialize visual_model before dimensionality check.\n"
            "        self.visual_model = cfg.model.visual\n"
            "        if self.num_variables > 2 and self.visual_model:",
        )
        main_file.write_text(text, encoding="utf-8")
    text = main_file.read_text(encoding="utf-8")
    if "ICSR compatibility: skip final plot for high-dimensional cases" not in text:
        text = text.replace(
            "fig, ax = self.plotter.plot_results(best_function, self.test_function)\n"
            "        fig.savefig(self.output_path + \"final.png\")",
            "if self.num_variables <= 2:\n"
            "            fig, ax = self.plotter.plot_results(best_function, self.test_function)\n"
            "            fig.savefig(self.output_path + \"final.png\")\n"
            "        else:\n"
            "            self.logger.warning(\"ICSR compatibility: skip final plot for high-dimensional cases.\")",
        )
        main_file.write_text(text, encoding="utf-8")

    # Recent scikit-learn versions return a Python float from
    # mean_squared_error. The official scorers assumed a NumPy scalar and
    # called ``.astype`` on it, which aborts an otherwise valid run.
    for scorer_relative in (
        "scorers/basic_scorer.py",
        "scorers/minmax_scorer.py",
        "scorers/complexity_scorer.py",
    ):
        scorer_file = official_root / scorer_relative
        scorer_text = scorer_file.read_text(encoding="utf-8")
        if "fit.astype(np.float64)" in scorer_text:
            scorer_text = scorer_text.replace(
                "fit.astype(np.float64)",
                "np.float64(fit)",
            )
            scorer_file.write_text(scorer_text, encoding="utf-8")

    model_cfg = official_root / "conf" / "model" / "vllm-qwen.yaml"
    model_cfg.write_text(
        f"""defaults:
  - base_prompt: basic_text
  - _self_

name: {MODEL_NAME}
tokenizer_pad: "\\\\[PAD\\\\]"
tokenizer_padding_side: left
visual: false
cache_dir: ''

seed_function_prompt: seed_functions/generate_seed.txt

max_new_tokens: {int(max_new_tokens)}
top_p: {float(top_p)}
top_k: {int(top_k)}
num_beams: 1

temperature: {float(temperature)}
temperature_schedule: false
temperature_schedule_gamma: 0.995
""",
        encoding="utf-8",
    )
    fcntl.flock(compatibility_lock.fileno(), fcntl.LOCK_UN)
    compatibility_lock.close()


def write_case_config(official_root: Path, benchmark: str, case_id: str, train_df, val_df, test_df, iterations: int) -> Path:
    data_dir = official_root / "data" / "fourbench" / benchmark / case_id
    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "train_points.npy", dataframe_to_points(train_df))
    np.save(data_dir / "test_points.npy", dataframe_to_points(pd.concat([val_df, test_df], axis=0, ignore_index=True)))

    n_vars = len([c for c in train_df.columns if c != "y"])
    cfg_dir = official_root / "conf" / "experiment" / "function" / "fourbench" / benchmark
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{case_id}.yaml"
    cfg_path.write_text(
        f"""name: "{case_id}"
group: "fourbench/{benchmark}"

train_points:
  generate_points: false
  data_folder: "data/fourbench/{benchmark}/{case_id}"
  random_points: false
  min_points: -1
  max_points: 1
  num_points: {len(train_df)}
  xs_noise_std: 0
  ys_noise_std: 0
  add_extremes: false

test_points:
  min_points: -1
  max_points: 1
  num_points: {len(val_df) + len(test_df)}

tolerance: 1.1
num_variables: {n_vars}
iterations: {int(iterations)}
""",
        encoding="utf-8",
    )
    return cfg_path


def adapted_seed_functions(n_vars: int) -> list[str]:
    if n_vars <= 1:
        return [
            "x",
            "x^2",
            "x^3",
            "sin(x)",
            "exp(x)",
            "log(abs(x)+1)",
        ]
    vars_ = [f"x{i}" for i in range(1, n_vars + 1)]
    sum_all = " + ".join(vars_)
    lin = " + ".join(f"{i + 1}*{name}" for i, name in enumerate(vars_))
    quad = " + ".join(f"{name}^2" for name in vars_[: min(3, n_vars)])
    pair = f"{vars_[0]}*{vars_[1]}" if n_vars >= 2 else vars_[0]
    return [
        vars_[0],
        sum_all,
        lin,
        quad,
        pair,
        f"sin({vars_[0]}) + cos({vars_[1]})" if n_vars >= 2 else f"sin({vars_[0]})",
    ]


def write_seed_config(official_root: Path, case_id: str, n_vars: int) -> str:
    cfg_name = f"fourbench_adapted_{sanitize(case_id)}"
    cfg_path = official_root / "conf" / "experiment" / "seed_functions" / f"{cfg_name}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "functions: " + json.dumps(adapted_seed_functions(n_vars), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return cfg_name


def latest_run_dir(official_root: Path, benchmark: str, case_id: str, min_mtime: float | None = None) -> Path | None:
    root = official_root / "runs" / "fourbench" / benchmark / case_id
    if not root.exists():
        return None
    candidates = [p for p in root.glob("*/*") if p.is_dir()]
    if min_mtime is not None:
        candidates = [p for p in candidates if p.stat().st_mtime >= min_mtime]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def evaluate_expression(expr_text: str, train_df, val_df, test_df):
    import sympy as sp

    X_train, y_train, feature_cols = base.dataframe_to_xy(train_df)
    X_val, y_val, _ = base.dataframe_to_xy(val_df)
    X_test, y_test, _ = base.dataframe_to_xy(test_df)
    n_vars = X_train.shape[1]
    symbols = sp.symbols(" ".join([f"x{i+1}" for i in range(n_vars)]))
    if n_vars == 1:
        symbols = (symbols,)
    locals_map = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }
    for sym in symbols:
        locals_map[str(sym)] = sym
    expr = sp.sympify(str(expr_text).replace("^", "**"), locals=locals_map)
    fn = sp.lambdify(symbols, expr, modules=["numpy"])

    def pred(X):
        y = fn(*[X[:, i] for i in range(n_vars)])
        y = np.asarray(y, dtype=float)
        if y.shape == ():
            y = np.full(X.shape[0], float(y), dtype=float)
        return y.reshape(-1)

    pred_train = pred(X_train)
    pred_val = pred(X_val)
    pred_test = pred(X_test)
    return {
        "train_mse": safe_mse(y_train, pred_train),
        "val_mse": safe_mse(y_val, pred_val),
        "test_mse": safe_mse(y_test, pred_test),
        "train_mae": safe_mae(y_train, pred_train),
        "val_mae": safe_mae(y_val, pred_val),
        "test_mae": safe_mae(y_test, pred_test),
        "train_r2": safe_r2(y_train, pred_train),
        "val_r2": safe_r2(y_val, pred_val),
        "test_r2": safe_r2(y_test, pred_test),
        "expr_complexity": int(sp.count_ops(expr, visual=False)),
        "expr_string_length": len(str(expr).replace(" ", "")),
    }


def evaluate_validation_expression(expr_text: str, val_df):
    import sympy as sp

    X_val, y_val, _ = base.dataframe_to_xy(val_df)
    n_vars = X_val.shape[1]
    symbols = sp.symbols(" ".join([f"x{i+1}" for i in range(n_vars)]))
    if n_vars == 1:
        symbols = (symbols,)
    locals_map = {
        **{str(sym): sym for sym in symbols},
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
        "abs": sp.Abs,
        "pi": sp.pi,
        "E": sp.E,
    }
    expr = sp.sympify(str(expr_text).replace("^", "**"), locals=locals_map)
    fn = sp.lambdify(symbols, expr, modules=["numpy"])
    pred = np.asarray(fn(*[X_val[:, i] for i in range(n_vars)]), dtype=float)
    if pred.shape == ():
        pred = np.full(X_val.shape[0], float(pred), dtype=float)
    return safe_r2(y_val, pred.reshape(-1))


def evaluate_candidate_history(candidates, val_df, trace_path: Path, threshold: float = 0.999):
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    observed = 0
    first_hit = None
    best = None
    started = time.time()
    with gzip.open(trace_path, "wt", encoding="utf-8") as fp:
        for candidate in candidates or []:
            expr = str(candidate.get("expression", ""))
            if not expr:
                continue
            observed += 1
            key = re.sub(r"\s+", "", expr)
            is_unique = key not in seen
            if is_unique:
                seen.add(key)
            try:
                val_r2 = evaluate_validation_expression(expr, val_df)
            except Exception:
                val_r2 = None
            event = {
                "method": "official_icsr",
                "candidate_index": observed,
                "unique_candidate_index": len(seen),
                "is_unique": is_unique,
                "native_evaluations": observed,
                "stage": candidate.get("stage", "iteration"),
                "iteration": candidate.get("iteration"),
                "expression": expr,
                "validation_r2": val_r2,
                "elapsed_sec": time.time() - started,
                "validation_success": bool(val_r2 is not None and val_r2 > threshold),
            }
            fp.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            if val_r2 is not None and (best is None or val_r2 > best["validation_r2"]):
                best = dict(event)
            if event["validation_success"] and first_hit is None:
                first_hit = dict(event)
    hit = first_hit or {}
    return {
        "validation_search_threshold": float(threshold),
        "validation_search_rule": "strictly_greater",
        "validation_search_success": bool(first_hit),
        "evaluations_to_validation_success": hit.get("candidate_index"),
        "unique_evaluations_to_validation_success": hit.get("unique_candidate_index"),
        "native_evaluations_to_validation_success": hit.get("native_evaluations"),
        "first_validation_success_stage": hit.get("stage"),
        "first_validation_success_expression": hit.get("expression"),
        "first_validation_success_r2": hit.get("validation_r2"),
        "validation_search_observed_evaluations": observed,
        "validation_search_unique_evaluations": len(seen),
        "validation_search_native_evaluations": observed,
        "validation_search_best_r2": (best or {}).get("validation_r2"),
        "validation_search_best_expression": (best or {}).get("expression"),
        "validation_search_trace": str(trace_path),
        "validation_search_count_semantics": (
            "all valid optimized ICSR candidates in generation order, including seeds"
        ),
    }


def parse_official_result(run_dir: Path | None):
    if run_dir is None:
        return {}, None
    result_path = run_dir / "results.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8")), result_path
    checkpoints = sorted(run_dir.glob("results_checkpoint_*.json"))
    if checkpoints:
        path = checkpoints[-1]
        return json.loads(path.read_text(encoding="utf-8")), path
    return {}, None


def parse_streamed_candidates(run_dir: Path | None):
    if run_dir is None:
        return []
    path = run_dir / "candidate_functions.jsonl"
    if not path.exists():
        return []
    candidates = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            candidates.append(json.loads(line))
        except Exception:
            continue
    return candidates


def run_case(args, benchmark: str, row_index: int, row: dict, total: int, mod, result_root: Path):
    case_name = base.row_case_name(benchmark, row)
    case_id = f"{row_index:04d}_{sanitize(case_name)}"
    case_json = result_root / benchmark / "case_results" / f"{case_id}.json"
    case_json.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and case_json.exists():
        print(f"[SKIP] {benchmark} {row_index}/{total} {case_name}", flush=True)
        return json.loads(case_json.read_text(encoding="utf-8"))

    try:
        train_df, val_df, test_df, meta = base.load_case(benchmark, mod, row)
    except Exception as exc:
        result = {
            "method": args.method_name,
            "official_repo": "https://github.com/merlerm/In-Context-Symbolic-Regression",
            "benchmark": benchmark,
            "case_index": int(row_index),
            "case_name": case_name,
            "case_id": case_id,
            "return_code": None,
            "timed_out": False,
            "timeout_phase": None,
            "case_timeout_sec": int(args.case_timeout_sec),
            "startup_timeout_sec": int(args.startup_timeout_sec),
            "startup_runtime_sec": 0.0,
            "search_runtime_sec": 0.0,
            "runtime_sec": 0.0,
            "official_run_dir": None,
            "official_result_path": None,
            "log_path": None,
            "expression": "",
            "official_iterations": None,
            "official_best_found_at": None,
            "official_test_score": None,
            "official_r2_train": None,
            "official_r2_test": None,
            "evaluation_error": f"load_case_failed: {repr(exc)}",
        }
        case_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[LOAD-FAIL] {benchmark} {row_index}/{total} {case_name}: {repr(exc)}",
            flush=True,
        )
        return result
    n_vars = len([c for c in train_df.columns if c != "y"])
    write_case_config(OFFICIAL_ROOT, benchmark, case_id, train_df, val_df, test_df, args.iterations)
    seed_cfg_name = None
    if args.seed_mode == "fixed":
        seed_cfg_name = write_seed_config(OFFICIAL_ROOT, case_id, n_vars)

    log_dir = result_root / benchmark / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{case_id}.log"
    env = os.environ.copy()
    env.update({
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_BASE_URL": OPENAI_BASE_URL,
        "ICSR_FORCE_OPENAI": "1",
        # The model endpoint is local. Inherited SOCKS/HTTP proxy variables can
        # make httpx require optional proxy packages before search even starts.
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "http_proxy": "",
        "https_proxy": "",
        "all_proxy": "",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "HYDRA_FULL_ERROR": "1",
        "PYTHONUNBUFFERED": "1",
    })
    cmd = [
        sys.executable,
        "main.py",
        "root=" + str(OFFICIAL_ROOT),
        "model=vllm-qwen",
        f"experiment/function=fourbench/{benchmark}/{case_id}",
        "plotter.save_video=false",
        "plotter.save_frames=false",
        "logger.loggers=[file]",
        "device=cpu",
        f"experiment.function.iterations={int(args.iterations)}",
        f"checkpoints={args.checkpoints}",
        f"max_retries={int(args.max_retries)}",
        f"max_points_in_prompt={int(args.max_points_in_prompt)}",
        f"model.base_prompt.prompt_size={int(args.prompt_size)}",
        f"experiment.optimizer.timeout={int(args.optimizer_timeout_sec)}",
        f"experiment.optimizer.optimizer_threads={int(args.optimizer_threads)}",
    ]
    if seed_cfg_name:
        cmd.extend([
            "experiment.generate_seed_functions=false",
            f"experiment/seed_functions={seed_cfg_name}",
        ])
    started = time.time()
    started_monotonic = time.monotonic()
    ready_monotonic = None
    timed_out = False
    timeout_phase = None
    return_code = None
    with open(log_path, "w", encoding="utf-8") as log_fp:
        log_fp.write("COMMAND: " + " ".join(cmd) + "\n")
        log_fp.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(OFFICIAL_ROOT),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
        while proc.poll() is None:
            now = time.monotonic()
            if ready_monotonic is None:
                log_fp.flush()
                try:
                    ready = "ICSR_SEARCH_READY" in log_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )
                except OSError:
                    ready = False
                if ready:
                    ready_monotonic = now
                elif now - started_monotonic >= args.startup_timeout_sec:
                    timed_out = True
                    timeout_phase = "startup"
                    proc.kill()
                    proc.wait()
                    return_code = 124
                    break
            elif now - ready_monotonic >= args.case_timeout_sec:
                timed_out = True
                timeout_phase = "search"
                proc.kill()
                proc.wait()
                return_code = 124
                break
            time.sleep(0.25)
        if return_code is None:
            return_code = proc.returncode

    finished_monotonic = time.monotonic()
    elapsed = finished_monotonic - started_monotonic
    if ready_monotonic is None:
        startup_elapsed = elapsed
        search_elapsed = 0.0
    else:
        startup_elapsed = ready_monotonic - started_monotonic
        search_elapsed = finished_monotonic - ready_monotonic
    run_dir = latest_run_dir(OFFICIAL_ROOT, benchmark, case_id, min_mtime=started - 1.0)
    official, official_result_path = parse_official_result(run_dir)
    if not official.get("candidate_functions"):
        official["candidate_functions"] = parse_streamed_candidates(run_dir)
    trace_path = result_root / benchmark / "candidate_traces" / f"{case_id}.jsonl.gz"
    search_metrics = evaluate_candidate_history(
        official.get("candidate_functions", []),
        val_df,
        trace_path,
        threshold=0.999,
    )
    expression_candidates = (
        (
            "first_validation_success",
            search_metrics.get("first_validation_success_expression"),
        ),
        ("official_best_function", official.get("best_function")),
        ("official_best_expr", official.get("best_expr")),
        (
            "validation_best_available",
            search_metrics.get("validation_search_best_expression"),
        ),
    )
    expression_selection_source, expr = next(
        ((source, value) for source, value in expression_candidates if value),
        ("unavailable", ""),
    )
    metrics = {}
    if expr:
        try:
            metrics = evaluate_expression(expr, train_df, val_df, test_df)
        except Exception as exc:
            metrics = {"evaluation_error": str(exc)}

    result = {
        "method": args.method_name,
        "official_repo": "https://github.com/merlerm/In-Context-Symbolic-Regression",
        "benchmark": benchmark,
        "case_index": int(row_index),
        "case_name": case_name,
        "case_id": case_id,
        "return_code": return_code,
        "timed_out": timed_out,
        "timeout_phase": timeout_phase,
        "case_timeout_sec": int(args.case_timeout_sec),
        "startup_timeout_sec": int(args.startup_timeout_sec),
        "startup_runtime_sec": startup_elapsed,
        "search_runtime_sec": search_elapsed,
        "runtime_sec": elapsed,
        "official_run_dir": str(run_dir) if run_dir else None,
        "official_result_path": str(official_result_path) if official_result_path else None,
        "log_path": str(log_path),
        "expression": expr,
        "expression_selection_source": expression_selection_source,
        "official_iterations": official.get("iterations"),
        "official_best_found_at": official.get("best_found_at"),
        "official_test_score": safe_float(official.get("test_score")),
        "official_r2_train": safe_float(official.get("r2_train")),
        "official_r2_test": safe_float(official.get("r2_test")),
        **search_metrics,
        **meta,
        **metrics,
    }
    case_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[DONE] {benchmark} {row_index}/{total} {case_name} rc={return_code} "
        f"timeout={timed_out} phase={timeout_phase} test_r2={result.get('test_r2')} "
        f"startup={startup_elapsed:.1f}s search={search_elapsed:.1f}s total={elapsed:.1f}s",
        flush=True,
    )
    return result


def summarize(result_root: Path, benchmark: str, rows: list[dict]) -> None:
    bench_root = result_root / benchmark
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(bench_root / f"all_{benchmark}_official_icsr_results.csv", index=False, encoding="utf-8-sig")
    test_r2 = pd.to_numeric(df["test_r2"], errors="coerce") if "test_r2" in df else pd.Series(dtype=float)
    test_mse = pd.to_numeric(df["test_mse"], errors="coerce") if "test_mse" in df else pd.Series(dtype=float)
    summary = {
        "benchmark": benchmark,
        "method": rows[0].get("method", "official_icsr") if rows else "official_icsr",
        "n_cases": int(len(df)),
        "n_completed": int((df["timed_out"] == False).sum()) if "timed_out" in df else 0,
        "n_timed_out": int((df["timed_out"] == True).sum()) if "timed_out" in df else 0,
        "mean_test_r2": safe_float(test_r2.mean()),
        "median_test_mse": safe_float(test_mse.median()),
        "result_csv": str(bench_root / f"all_{benchmark}_official_icsr_results.csv"),
    }
    (bench_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", default="llmsrbench,srbench,srsd")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--case-timeout-sec", type=int, default=600)
    parser.add_argument("--startup-timeout-sec", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--optimizer-timeout-sec", type=int, default=10)
    parser.add_argument("--optimizer-threads", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--checkpoints", default="[0,1,2,3,4,5,10,20,30,40,50,75,100,150,200,300,400,500]")
    parser.add_argument("--method-name", default="official_icsr")
    parser.add_argument("--seed-mode", choices=["generate", "fixed"], default="generate")
    parser.add_argument("--prompt-size", type=int, default=5)
    parser.add_argument("--max-points-in-prompt", type=int, default=40)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--case-start",
        type=int,
        default=1,
        help="One-based first case index, for non-overlapping execution shards.",
    )
    parser.add_argument(
        "--case-end",
        type=int,
        default=None,
        help="One-based inclusive final case index, for execution shards.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    ensure_official_compatibility(
        OFFICIAL_ROOT,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
    )
    result_root = Path(args.results_root)
    result_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": args.method_name,
        "official_repo": "https://github.com/merlerm/In-Context-Symbolic-Regression",
        "official_root": str(OFFICIAL_ROOT),
        "model": MODEL_NAME,
        "openai_base_url": OPENAI_BASE_URL,
        "llmsrbench_root": str(LLMSRBENCH_ROOT),
        "llmsrbench_hdf5": str(LLMSRBENCH_HDF5),
        "case_timeout_sec": args.case_timeout_sec,
        "startup_timeout_sec": args.startup_timeout_sec,
        "case_timeout_semantics": (
            "search budget begins immediately before generated seed functions at "
            "the exact ICSR_SEARCH_READY marker, so generated seeds, optimization, "
            "and iterative refinement are all included; pure setup is separate"
        ),
        "iterations": args.iterations,
        "optimizer_timeout_sec": args.optimizer_timeout_sec,
        "optimizer_threads": args.optimizer_threads,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "checkpoints": args.checkpoints,
        "seed_mode": args.seed_mode,
        "prompt_size": args.prompt_size,
        "max_points_in_prompt": args.max_points_in_prompt,
        "max_retries": args.max_retries,
        "case_start": args.case_start,
        "case_end": args.case_end,
        "compatibility_patches": [
            (
                "RNG-equivalent local set_seed replaces the full transformers "
                "registry import in each isolated CPU process"
            ),
            (
                "official scorer casts sklearn scalar MSE with np.float64 "
                "instead of calling ndarray-only astype"
            ),
            (
                "httpx warning logging uses the installed mloggers API, so "
                "scipy warnings do not discard fitted candidates"
            ),
            (
                "finite fitted parameters are retained when scipy cannot "
                "estimate an unused covariance matrix"
            ),
            (
                "each successfully fitted seed and iterative candidate is "
                "streamed before the time-limited process can terminate"
            ),
            (
                "non-finite aggregate metrics are classified as numerical "
                "failures instead of being serialized as invalid JSON values"
            ),
        ],
        "adapter": str(Path(__file__).resolve()),
    }
    (result_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for benchmark in [b.strip() for b in args.benchmarks.split(",") if b.strip()]:
        if benchmark not in BENCHMARKS:
            raise ValueError(f"unknown benchmark: {benchmark}")
        mod, df_tasks = base.collect_tasks(benchmark)
        total_all = len(df_tasks)
        case_start = max(1, int(args.case_start))
        case_end = (
            min(total_all, int(args.case_end))
            if args.case_end is not None
            else total_all
        )
        if case_start > case_end:
            raise ValueError(
                f"empty case range {case_start}:{case_end} for {benchmark} "
                f"with {total_all} cases"
            )
        df_tasks = df_tasks.iloc[case_start - 1 : case_end].copy()
        if args.max_cases is not None:
            df_tasks = df_tasks.head(args.max_cases).copy()
        bench_root = result_root / benchmark
        bench_root.mkdir(parents=True, exist_ok=True)
        df_tasks.to_csv(bench_root / f"{benchmark}_selected_tasks.csv", index=False, encoding="utf-8-sig")
        rows = []
        total = total_all
        for pos, (_, row) in enumerate(df_tasks.iterrows(), start=case_start):
            rows.append(run_case(args, benchmark, pos, row.to_dict(), total, mod, result_root))
            summarize(result_root, benchmark, rows)


if __name__ == "__main__":
    main()
